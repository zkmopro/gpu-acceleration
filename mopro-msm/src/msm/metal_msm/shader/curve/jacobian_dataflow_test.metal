using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "jacobian.metal"

kernel void run(
    constant Jacobian* points [[ buffer(0) ]],
    device Jacobian* out [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    out[gid] = jacobian_scalar_mul(points[gid], 2);
}
