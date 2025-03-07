// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "mont.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device BigInt* result [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *result = mont_mul_optimised(*lhs, *rhs);
}
