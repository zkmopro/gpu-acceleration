// // source: https://github.com/geometryxyz/msl-secp256k1

// using namespace metal;
// #include <metal_stdlib>
// #include <metal_math>
// #include "jacobian.metal"

// kernel void run(
//     device BigInt* prime [[ buffer(0) ]],
//     device BigInt* a_xr [[ buffer(1) ]],
//     device BigInt* a_yr [[ buffer(2) ]],
//     device BigInt* a_zr [[ buffer(3) ]],
//     device BigInt* result_xr [[ buffer(4) ]],
//     device BigInt* result_yr [[ buffer(5) ]],
//     device BigInt* result_zr [[ buffer(6) ]],
//     uint gid [[ thread_position_in_grid ]]
// ) {
//     BigInt p = *prime;
//     BigInt x1 = *a_xr;
//     BigInt y1 = *a_yr;
//     BigInt z1 = *a_zr;

//     Jacobian a; a.x = x1; a.y = y1; a.z = z1;

//     Jacobian res = jacobian_dbl_2009_l(a, p);
//     *result_xr = res.x;
//     *result_yr = res.y;
//     *result_zr = res.z;
// }
