// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include "ff.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_ff_reduce(
    device BigInt* a [[buffer(0)]],
    device BigInt* res [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    *res = ff_reduce(*a);
}
