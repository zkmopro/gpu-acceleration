// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "ff.metal"

kernel void test_ff_sub(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* b [[ buffer(1) ]],
    device BigInt* res [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *res = ff_sub(*a, *b);
}
