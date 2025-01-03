// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

BigInt ff_add(
    BigInt a,
    BigInt b,
    BigInt p
) {
    BigInt sum = bigint_add_unsafe(a, b);

    BigInt res;
    if (bigint_gte(sum, p)) {
        // s = a + b - p
        BigInt s = bigint_sub(sum, p);
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = s.limbs[i];
        }
    }
    else {
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = sum.limbs[i];
        }
    }
    return res;
}

BigInt ff_sub(
    BigInt a,
    BigInt b,
    BigInt p
) {
    // if a >= b
    if (bigint_gte(a, b)) {
        // a - b
        BigInt res = bigint_sub(a, b);
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    } else {
        // p - (b - a)
        BigInt r = bigint_sub(b, a);
        BigInt res = bigint_sub(p, r);
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    }
}
