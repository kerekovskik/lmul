#include <metal_stdlib>
using namespace metal;

// Kernel for performing L-mul on arrays
kernel void l_mul_kernel(const device float* a [[buffer(0)]],
                        const device float* b [[buffer(1)]],
                        device float* result [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    // Get raw bits of input floats
    uint bits_a = as_type<uint>(a[index]);
    uint bits_b = as_type<uint>(b[index]);
    
    // Extract signs using bitwise operations
    uint sign_a = bits_a & 0x80000000;
    uint sign_b = bits_b & 0x80000000;
    uint result_sign = sign_a ^ sign_b;
    
    // Clear signs from inputs for mantissa work
    uint unsigned_a = bits_a & 0x7FFFFFFF;
    uint unsigned_b = bits_b & 0x7FFFFFFF;
    
    // Add the unsigned values and subtract the bias adjustment
    // Note: 0x3F800000 represents 1.0f in IEEE-754
    uint mantissa_result = unsigned_a + unsigned_b - 0x3F800000;
    
    // Combine with result sign
    uint final_result = (mantissa_result & 0x7FFFFFFF) | result_sign;
    
    // Store result
    result[index] = as_type<float>(final_result);
}