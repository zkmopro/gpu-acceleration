using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "mont.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device array<uint, 1>* cost [[ buffer(2) ]],
    device BigInt* result [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    FieldElement lhs_field = { *lhs };
    FieldElement rhs_field = { *rhs };
    array<uint, 1> cost_arr = *cost;

    // calculate lhs^3 * rhs
    FieldElement c = mont_mul_modified(lhs_field, lhs_field);
    for (uint i = 1; i < cost_arr[0]; i ++) {
        c = mont_mul_modified(c, lhs_field);
    }
    *result = mont_mul_modified(c, rhs_field).value;
}
