// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "bigint.metal"

kernel void test_bigint_add_wide(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* b [[ buffer(1) ]],
    device BigIntWide* res [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigIntResultWide result = bigint_add_wide(*a, *b);
    *res = result.value;
}
