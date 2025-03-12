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
    device BigInt* b_zr [[ buffer(5) ]],
    device BigInt* result_xr [[ buffer(6) ]],
    device BigInt* result_yr [[ buffer(7) ]],
    device BigInt* result_zr [[ buffer(8) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    FieldElement x1 = FieldElement{ *a_xr };
    FieldElement y1 = FieldElement{ *a_yr };
    FieldElement z1 = FieldElement{ *a_zr };
    FieldElement x2 = FieldElement{ *b_xr };
    FieldElement y2 = FieldElement{ *b_yr };
    FieldElement z2 = FieldElement{ *b_zr };

    Jacobian a = Jacobian{ .x = x1, .y = y1, .z = z1 };
    Jacobian b = Jacobian{ .x = x2, .y = y2, .z = z2 };

    Jacobian res = a + b;
    *result_xr = res.x.value;
    *result_yr = res.y.value;
    *result_zr = res.z.value;
}
