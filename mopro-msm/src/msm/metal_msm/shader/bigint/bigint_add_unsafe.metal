// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "bigint.metal"

kernel void run(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* b [[ buffer(1) ]],
    device BigInt* res [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigIntResult result = bigint_add_unsafe(*a, *b);
    *res = result.value;
}
