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
    device BigInt* b_xr [[ buffer(4) ]],
    device BigInt* b_yr [[ buffer(5) ]],
    device BigInt* b_zr [[ buffer(6) ]],
    device BigInt* result_xr [[ buffer(7) ]],
    device BigInt* result_yr [[ buffer(8) ]],
    device BigInt* result_zr [[ buffer(9) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt p; p.limbs = prime->limbs;
    BigInt x1; x1.limbs = a_xr->limbs;
    BigInt y1; y1.limbs = a_yr->limbs;
    BigInt z1; z1.limbs = a_zr->limbs;
    BigInt x2; x2.limbs = b_xr->limbs;
    BigInt y2; y2.limbs = b_yr->limbs;
    BigInt z2; z2.limbs = b_zr->limbs;

    Jacobian a; a.x = x1; a.y = y1; a.z = z1;
    Jacobian b; b.x = x2; b.y = y2; b.z = z2;

    Jacobian res = jacobian_add_2007_bl_unsafe(a, b, p);
    result_xr->limbs = res.x.limbs;
    result_yr->limbs = res.y.limbs;
    result_zr->limbs = res.z.limbs;
}
