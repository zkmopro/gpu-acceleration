// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include "bigint.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_bigint_add_unsafe(
    device BigInt* a [[buffer(0)]],
    device BigInt* b [[buffer(1)]],
    device BigInt* res [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    BigIntResult result = bigint_add_unsafe(*a, *b);
    *res = result.value;
}
