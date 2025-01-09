// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

BigInt ff_add(
    BigInt a,
    BigInt b,
    BigInt p
) {
    BigInt sum = a + b;

    BigInt res;
    if (sum >= p) {
        // s = a + b - p
        BigInt s = sum - p;
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
    if (a >= b) {
        // a - b
        BigInt res = a - b;
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    } else {
        // p - (b - a)
        BigInt r = b - a;
        BigInt res = p - r;
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    }
}
