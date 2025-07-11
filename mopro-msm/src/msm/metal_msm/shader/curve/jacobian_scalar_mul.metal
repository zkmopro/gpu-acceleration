using namespace metal;
#include "jacobian.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_jacobian_scalar_mul(
    device Jacobian& a [[buffer(0)]],
    device uint* scalar [[buffer(1)]],
    device Jacobian& result [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint s = *scalar;
    result = jacobian_scalar_mul(a, s);
}