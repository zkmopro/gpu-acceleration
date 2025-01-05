// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "bigint.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device BigIntWide* result [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt a = *lhs;
    BigInt b = *rhs;
    BigIntWide res = bigint_add_wide(a, b);
    *result = res;
}
