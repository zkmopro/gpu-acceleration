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
    BigInt a;
    BigInt b;
    BigInt p;
    a.limbs = lhs->limbs;
    b.limbs = rhs->limbs;
    p.limbs = prime->limbs;
    array<uint, 1> cost_arr = *cost;

    BigInt c = mont_mul_optimised(a, a, p);
    for (uint i = 1; i < cost_arr[0]; i ++) {
        c = mont_mul_optimised(c, a, p);
    }
    BigInt res = mont_mul_optimised(c, b, p);
    result->limbs = res.limbs;
}
