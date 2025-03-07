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
    BigInt a = *lhs;
    BigInt b = *rhs;
    array<uint, 1> cost_arr = *cost;

    BigInt c = mont_mul_modified(a, a);
    for (uint i = 1; i < cost_arr[0]; i ++) {
        c = mont_mul_modified(c, a);
    }
    *result = mont_mul_modified(c, b);
}
