using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "mont.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device BigInt* prime [[ buffer(2) ]],
    device array<uint, 1>* cost [[ buffer(3) ]],
    device BigInt* result [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt a = *lhs;
    BigInt b = *rhs;
    BigInt p = *prime;
    array<uint, 1> cost_arr = *cost;

    BigInt c = mont_mul_cios(a, a, p);
    for (uint i = 1; i < cost_arr[0]; i ++) {
        c = mont_mul_cios(c, a, p);
    }
    BigInt res = mont_mul_cios(c, b, p);
    *result = res;
}
