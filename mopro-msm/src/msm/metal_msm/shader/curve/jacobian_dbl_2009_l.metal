// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "jacobian.metal"

kernel void run(
    device BigInt* prime [[ buffer(0) ]],
    device BigInt* a_xr [[ buffer(1) ]],
    device BigInt* a_yr [[ buffer(2) ]],
    device BigInt* a_zr [[ buffer(3) ]],
    device BigInt* result_xr [[ buffer(4) ]],
    device BigInt* result_yr [[ buffer(5) ]],
    device BigInt* result_zr [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt p; p.limbs = prime->limbs;
    BigInt x1; x1.limbs = a_xr->limbs;
    BigInt y1; y1.limbs = a_yr->limbs;
    BigInt z1; z1.limbs = a_zr->limbs;

    Jacobian a; a.x = x1; a.y = y1; a.z = z1;

    Jacobian res = jacobian_dbl_2009_l(a, p);
    result_xr->limbs = res.x.limbs;
    result_yr->limbs = res.y.limbs;
    result_zr->limbs = res.z.limbs;
}
