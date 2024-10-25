from .lmul import l_mul_scalar, simd_l_mul_arrays

__all__ = ["l_mul", "simd_l_mul_arrays"]

result = l_mul_scalar(1.5, 2.5)
print(f"L-Mul approximation of 1.5 * 2.5: {result}")
print(f"Expected 1.5 * 2.5: {1.5 * 2.5}")
