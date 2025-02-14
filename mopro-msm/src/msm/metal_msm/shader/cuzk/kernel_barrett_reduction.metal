using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "barrett_reduction.metal"

kernel void run(
    device BigIntExtraWide* a [[ buffer(0) ]],
    device BigInt* res [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *res = barrett_reduce(*a);
}
