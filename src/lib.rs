use numpy::{ndarray::{Array1, Axis}, IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::arch::aarch64::*;

/// Extracts the sign, exponent, and mantissa from a 32-bit float.
/// Mantissa is normalized to include the implicit leading 1.
fn extract_float_parts(f: f32) -> (i32, i32, i32) {
    let bits = f.to_bits(); // Get the raw bits of the float
    let sign = ((bits >> 31) as i32) & 1;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127; // Unbiased exponent
    let mantissa = (bits & 0x7FFFFF) as i32 | (1 << 23); // Include implicit 1
    (sign, exponent, mantissa)
}

/// Linear-complexity multiplication using integer addition.
#[pyfunction]
fn l_mul_scalar(a: f32, b: f32) -> PyResult<f32> {
    let (sign_a, exp_a, man_a) = extract_float_parts(a);
    let (sign_b, exp_b, man_b) = extract_float_parts(b);

    // XOR the signs for the result's sign
    let result_sign = sign_a ^ sign_b;

    // Approximate mantissa multiplication with integer addition
    let mut man_result = man_a + man_b - (1 << 23);

    // Handle overflow in mantissa addition
    let mut exp_result = exp_a + exp_b;
    if man_result & (1 << 24) != 0 {
        man_result >>= 1;
        exp_result += 1;
    }

    // Assemble the final result
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

    // Process elements in chunks of 4 using SIMD
    for (i, (chunk_a, chunk_b)) in array_a
        .axis_chunks_iter(Axis(0), 4)
        .zip(array_b.axis_chunks_iter(Axis(0), 4))
        .enumerate()
    {
        unsafe {
            // Load values
            let va = vld1q_f32(chunk_a.as_ptr());
            let vb = vld1q_f32(chunk_b.as_ptr());

            // Extract signs (top bit)
            let sign_mask = vdupq_n_u32(0x80000000);
            let signs_a = vandq_u32(vreinterpretq_u32_f32(va), sign_mask);
            let signs_b = vandq_u32(vreinterpretq_u32_f32(vb), sign_mask);
            let result_signs = veorq_u32(signs_a, signs_b);

            // Clear signs from inputs for mantissa work
            let va_unsigned = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(va), vdupq_n_u32(!0x80000000))
            );
            let vb_unsigned = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(vb), vdupq_n_u32(!0x80000000))
            );

            // Convert to integers (maintaining exponent and mantissa structure)
            let ma = vreinterpretq_s32_f32(va_unsigned);
            let mb = vreinterpretq_s32_f32(vb_unsigned);

            // Add mantissas and handle implicit 1
            let mut man_result = vaddq_s32(ma, mb);
            man_result = vsubq_s32(man_result, vdupq_n_s32(0x3F800000)); // Subtract 2^23

            // Combine with signs
            let final_result = vorrq_u32(
                vreinterpretq_u32_s32(man_result),
                result_signs
            );

            // Store result
            vst1q_f32(
                result.as_slice_mut().unwrap().as_mut_ptr().add(i * 4),
                vreinterpretq_f32_u32(final_result)
            );
        }
    }

    // Handle remaining elements using scalar multiplication
    let remainder = array_a.len() % 4;
    if remainder != 0 {
        for i in (array_a.len() - remainder)..array_a.len() {
            result[i] = l_mul_scalar(array_a[i], array_b[i])?;
        }
    }

    Ok(result.into_pyarray(py))
}

/* 
#[pymodule]
fn lmul(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(l_mul_scalar, py)?)?;
    m.add_function(wrap_pyfunction!(simd_l_mul_arrays, py)?)?;
    Ok(())
}
*/
// Python Module that exposes the function.
#[pymodule]
fn lmul(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(l_mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(simd_l_mul_arrays, m)?)?;
    Ok(())
}
