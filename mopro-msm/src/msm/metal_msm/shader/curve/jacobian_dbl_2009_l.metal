// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "jacobian.metal"

kernel void test_jacobian_dbl_2009_l(
    device BigInt* a_xr [[ buffer(0) ]],
    device BigInt* a_yr [[ buffer(1) ]],
    device BigInt* a_zr [[ buffer(2) ]],
    device BigInt* result_xr [[ buffer(3) ]],
    device BigInt* result_yr [[ buffer(4) ]],
    device BigInt* result_zr [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt x1 = *a_xr;
    BigInt y1 = *a_yr;
    BigInt z1 = *a_zr;

    Jacobian a; a.x = x1; a.y = y1; a.z = z1;

    Jacobian res = jacobian_dbl_2009_l(a);
    *result_xr = res.x;
    *result_yr = res.y;
    *result_zr = res.z;
}
