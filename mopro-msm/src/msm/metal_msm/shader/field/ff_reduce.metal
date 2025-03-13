// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "ff.metal"

kernel void run(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* res [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    // wrap into FieldElement
    FieldElement a_field = { *a };
    FieldElement res_field = ff_reduce(a_field);
    *res = res_field.value;
}
