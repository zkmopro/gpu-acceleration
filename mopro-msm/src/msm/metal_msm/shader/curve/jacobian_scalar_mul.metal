// using namespace metal;
// #include <metal_stdlib>
// #include <metal_math>
// #include "jacobian.metal"

// kernel void run(
//     device Jacobian& a [[ buffer(0) ]],
//     device uint* scalar [[ buffer(1) ]],
//     device Jacobian& result [[ buffer(2) ]],
//     uint gid [[ thread_position_in_grid ]]
// ) {
//     uint s = *scalar;
//     result = jacobian_scalar_mul(a, s);
// }