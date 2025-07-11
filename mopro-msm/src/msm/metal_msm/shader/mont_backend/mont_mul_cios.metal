// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include "mont.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_mont_mul_cios(
    device BigInt* lhs [[buffer(0)]],
    device BigInt* rhs [[buffer(1)]],
    device BigInt* result [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    *result = mont_mul_cios(*lhs, *rhs);
}
