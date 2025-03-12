// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "jacobian.metal"

kernel void run(
    device BigInt* a_xr [[ buffer(0) ]],
    device BigInt* a_yr [[ buffer(1) ]],
    device BigInt* a_zr [[ buffer(2) ]],
    device BigInt* result_xr [[ buffer(3) ]],
    device BigInt* result_yr [[ buffer(4) ]],
    device BigInt* result_zr [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt p = get_p();
    FieldElement x1 = FieldElement{ .value = *a_xr, .modulus = p };
    FieldElement y1 = FieldElement{ .value = *a_yr, .modulus = p };
    FieldElement z1 = FieldElement{ .value = *a_zr, .modulus = p };

    Jacobian a = Jacobian{ .x = x1, .y = y1, .z = z1 };

    Jacobian res = -a;
    *result_xr = res.x.value;
    *result_yr = res.y.value;
    *result_zr = res.z.value;
}
