// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

inline BigInt ff_reduce(BigInt a) {
    BigInt p = MODULUS;
    BigIntResult res = bigint_sub(a, p);
    if (res.carry == 1) return a;
    return res.value;
}

inline BigInt ff_add(BigInt a, BigInt b) {
    return ff_reduce(bigint_add_unsafe(a, b).value);
}

inline BigInt ff_sub(BigInt a, BigInt b) {
    bool a_gte_b = bigint_gte(a, b);

    if (a_gte_b) {
        return bigint_sub(a, b).value;
    }
    else {
        // p - (b - a)
        BigInt p = MODULUS;
        BigIntResult diff = bigint_sub(b, a);
        return bigint_sub(p, diff.value).value;
    }
}
