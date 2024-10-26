use numpy::{ndarray::{Array1, Axis}, IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::arch::aarch64::*;
use std::path::PathBuf;

/// Extracts the sign, exponent, and mantissa from a 32-bit float.
fn extract_float_parts(f: f32) -> (i32, i32, i32) {
    let bits = f.to_bits();
    let sign = ((bits >> 31) as i32) & 1;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = (bits & 0x7FFFFF) as i32 | (1 << 23);
    (sign, exponent, mantissa)
}

/// Linear-complexity multiplication using integer addition.
#[pyfunction]
fn l_mul_scalar(a: f32, b: f32) -> PyResult<f32> {
    let (sign_a, exp_a, man_a) = extract_float_parts(a);
    let (sign_b, exp_b, man_b) = extract_float_parts(b);

    let result_sign = sign_a ^ sign_b;
    let mut man_result = man_a + man_b - (1 << 23);
    let mut exp_result = exp_a + exp_b;
    
    if man_result & (1 << 24) != 0 {
        man_result >>= 1;
        exp_result += 1;
    }

    let result_bits = ((result_sign as u32) << 31)
        | (((exp_result + 127) as u32) << 23)
        | ((man_result as u32) & 0x7FFFFF);

    Ok(f32::from_bits(result_bits))
}

/// SIMD-based linear-addition multiplication for two arrays.
#[pyfunction]
fn simd_l_mul_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<f32>,
    b: PyReadonlyArrayDyn<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let array_a = a.as_array();
    let array_b = b.as_array();
    let mut result = Array1::<f32>::zeros(array_a.len());

    for (i, (chunk_a, chunk_b)) in array_a
        .axis_chunks_iter(Axis(0), 4)
        .zip(array_b.axis_chunks_iter(Axis(0), 4))
        .enumerate()
    {
        unsafe {
            let va = vld1q_f32(chunk_a.as_ptr());
            let vb = vld1q_f32(chunk_b.as_ptr());

            let sign_mask = vdupq_n_u32(0x80000000);
            let signs_a = vandq_u32(vreinterpretq_u32_f32(va), sign_mask);
            let signs_b = vandq_u32(vreinterpretq_u32_f32(vb), sign_mask);
            let result_signs = veorq_u32(signs_a, signs_b);

            let va_unsigned = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(va), vdupq_n_u32(!0x80000000))
            );
            let vb_unsigned = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(vb), vdupq_n_u32(!0x80000000))
            );

            let ma = vreinterpretq_s32_f32(va_unsigned);
            let mb = vreinterpretq_s32_f32(vb_unsigned);

            let mut man_result = vaddq_s32(ma, mb);
            man_result = vsubq_s32(man_result, vdupq_n_s32(0x3F800000));

            let final_result = vorrq_u32(
                vreinterpretq_u32_s32(man_result),
                result_signs
            );

            vst1q_f32(
                result.as_slice_mut().unwrap().as_mut_ptr().add(i * 4),
                vreinterpretq_f32_u32(final_result)
            );
        }
    }

    let remainder = array_a.len() % 4;
    if remainder != 0 {
        for i in (array_a.len() - remainder)..array_a.len() {
            result[i] = l_mul_scalar(array_a[i], array_b[i])?;
        }
    }

    Ok(result.into_pyarray(py))
}

// Metal implementation
use core_graphics::geometry::CGSize;
use metal::*;
use objc::rc::autoreleasepool;
use cocoa::foundation::NSUInteger;
use std::mem;

pub struct MetalState {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl MetalState {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();
        
        // Get the path to the metallib relative to the current executable
        let exe_path = std::env::current_exe().expect("Failed to get executable path");
        let exe_dir = exe_path.parent().expect("Failed to get executable directory");
        let metallib_path = exe_dir.join("lmul.metallib");
        
        if !metallib_path.exists() {
            // Try one directory up (for development)
            let metallib_path = exe_dir.parent().unwrap().join("lmul/lmul.metallib");
            if !metallib_path.exists() {
                panic!("Could not find lmul.metallib");
            }
        }
        
        let metallib_url = metal::URL::new_with_string(&format!(
            "file://{}",
            metallib_path.to_str().unwrap()
        ));
        
        let library = device
            .new_library_with_file(metallib_url)
            .expect("Failed to load metallib");
            
        let kernel = library
            .get_function("l_mul_kernel", None)
            .expect("Failed to find kernel function");
            
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline state");
            
        Self {
            device,
            command_queue,
            pipeline_state,
        }
    }
    
    pub fn l_mul_arrays(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        autoreleasepool(|| {
            let buffer_a = self.device.new_buffer_with_data(
                a.as_ptr() as *const _,
                (a.len() * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeManaged,
            );
            
            let buffer_b = self.device.new_buffer_with_data(
                b.as_ptr() as *const _,
                (b.len() * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeManaged,
            );
            
            let buffer_result = self.device.new_buffer(
                (a.len() * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeManaged,
            );
            
            let command_buffer = self.command_queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            
            compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
            compute_encoder.set_buffer(0, Some(&buffer_a), 0);
            compute_encoder.set_buffer(1, Some(&buffer_b), 0);
            compute_encoder.set_buffer(2, Some(&buffer_result), 0);
            
            let thread_execution_width = self.pipeline_state.thread_execution_width();
            let max_threads_per_group = self.pipeline_state.max_total_threads_per_threadgroup();
            let thread_group_size = thread_execution_width.min(max_threads_per_group);
            
            let grid_size = (a.len() as u64 + thread_group_size as u64 - 1) / thread_group_size as u64;
            
            let thread_group_count = MTLSize {
                width: grid_size as NSUInteger,
                height: 1,
                depth: 1,
            };
            
            let thread_group_size = MTLSize {
                width: thread_group_size as NSUInteger,
                height: 1,
                depth: 1,
            };
            
            compute_encoder.dispatch_threads(thread_group_count, thread_group_size);
            compute_encoder.end_encoding();
            
            let blit_encoder = command_buffer.new_blit_command_encoder();
            blit_encoder.synchronize_resource(&buffer_result);
            blit_encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            let result_ptr = buffer_result.contents() as *const f32;
            let mut result = vec![0.0f32; a.len()];
            unsafe {
                std::ptr::copy_nonoverlapping(result_ptr, result.as_mut_ptr(), a.len());
            }
            
            result
        })
    }
}

static METAL_STATE: once_cell::sync::Lazy<MetalState> = once_cell::sync::Lazy::new(|| {
    MetalState::new()
});

#[pyfunction]
fn metal_l_mul_arrays<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<f32>,
    b: PyReadonlyArrayDyn<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let array_a = a.as_array();
    let array_b = b.as_array();
    
    let a_slice = array_a.as_slice().unwrap();
    let b_slice = array_b.as_slice().unwrap();
    
    let result = METAL_STATE.l_mul_arrays(a_slice, b_slice);
    
    let result_array = Array1::from_vec(result);
    Ok(result_array.into_pyarray(py))
}

#[pymodule]
fn lmul(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(l_mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(simd_l_mul_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(metal_l_mul_arrays, m)?)?;
    Ok(())
}