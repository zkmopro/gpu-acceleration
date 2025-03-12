// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "mont.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device BigInt* result [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt p = get_p();
    FieldElement lhs_field = { .value = *lhs, .modulus = p };
    FieldElement rhs_field = { .value = *rhs, .modulus = p };
    FieldElement res_field = mont_mul_optimised(lhs_field, rhs_field);
    *result = res_field.value;
}
