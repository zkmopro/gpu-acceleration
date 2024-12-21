// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "mont.metal"

kernel void run(
    device BigInt* lhs [[ buffer(0) ]],
    device BigInt* rhs [[ buffer(1) ]],
    device BigInt* prime [[ buffer(2) ]],
    device BigInt* result [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    BigInt a;
    BigInt b;
    BigInt p;
    a.limbs = lhs->limbs;
    b.limbs = rhs->limbs;
    p.limbs = prime->limbs;

    BigInt res = mont_mul_cios(a, b, p);
    result->limbs = res.limbs;

}
