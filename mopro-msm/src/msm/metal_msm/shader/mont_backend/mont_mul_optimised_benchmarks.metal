using namespace metal;
#include "mont.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_mont_mul_optimised_benchmarks(
    device BigInt* lhs [[buffer(0)]],
    device BigInt* rhs [[buffer(1)]],
    device array<uint, 1>* cost [[buffer(2)]],
    device BigInt* result [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    BigInt a = *lhs;
    BigInt b = *rhs;
    array<uint, 1> cost_arr = *cost;

    BigInt c = mont_mul_optimised(a, a);
    for (uint i = 1; i < cost_arr[0]; i++) {
        c = mont_mul_optimised(c, a);
    }
    *result = mont_mul_optimised(c, b);
}
