using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "barrett_reduction.metal"

kernel void run(
    device BigInt* a [[ buffer(0) ]],
    device BigInt* b [[ buffer(1) ]],
    device BigInt* res [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *res = field_mul(*a, *b);
}
