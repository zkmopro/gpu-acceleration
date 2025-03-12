// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "jacobian.metal"

kernel void run(
    device BigInt* a_xr [[ buffer(0) ]],
    device BigInt* a_yr [[ buffer(1) ]],
    device BigInt* a_zr [[ buffer(2) ]],
    device BigInt* b_xr [[ buffer(3) ]],
    device BigInt* b_yr [[ buffer(4) ]],
    device BigInt* result_xr [[ buffer(5) ]],
    device BigInt* result_yr [[ buffer(6) ]],
    device BigInt* result_zr [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt p = get_p();
    FieldElement x1 = FieldElement{ .value = *a_xr, .modulus = p };
    FieldElement y1 = FieldElement{ .value = *a_yr, .modulus = p };
    FieldElement z1 = FieldElement{ .value = *a_zr, .modulus = p };
    FieldElement x2 = FieldElement{ .value = *b_xr, .modulus = p };
    FieldElement y2 = FieldElement{ .value = *b_yr, .modulus = p };

    Jacobian a = Jacobian{ .x = x1, .y = y1, .z = z1 };
    Affine b = Affine{ .x = x2, .y = y2 };

    Jacobian res = a + b;
    *result_xr = res.x.value;
    *result_yr = res.y.value;
    *result_zr = res.z.value;
}
