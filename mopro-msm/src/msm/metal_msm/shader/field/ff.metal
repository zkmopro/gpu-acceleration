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
    // Assign p to p_wide
    BigIntWide p_wide;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        p_wide.limbs[i] = p.limbs[i];
    }

    // a + b
    BigIntWide sum_wide = bigint_add_wide(a, b);

    BigInt res;

    // if (a + b) >= p
    if (bigint_wide_gte(sum_wide, p_wide)) {
        // s = a + b - p
        BigIntWide s = bigint_sub_wide(sum_wide, p_wide);

        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = s.limbs[i];
        }
    } else {
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = sum_wide.limbs[i];
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
