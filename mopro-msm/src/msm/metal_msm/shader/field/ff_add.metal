// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "ff.metal"

kernel void run(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* b [[ buffer(1) ]],
    device BigInt* prime [[ buffer(2) ]],
    device BigInt* res [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *res = ff_add(*a, *b, *prime);
}
