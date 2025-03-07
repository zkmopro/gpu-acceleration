// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

BigInt ff_reduce(BigInt a) {
    BigInt p = get_p();
    BigIntResult res = bigint_sub(a, p);
    if (res.carry == 1) return a;
    return res.value;
}

BigInt ff_add(BigInt a, BigInt b) {
    BigIntResult res = bigint_add_unsafe(a, b);
    return ff_reduce(res.value);
}

BigInt ff_sub(BigInt a, BigInt b) {
    bool a_gte_b = bigint_gte(a, b);

    if (a_gte_b) {
        BigIntResult res = bigint_sub(a, b);
        return res.value;
    }
    else {
        // p - (b - a)
        BigInt p = get_p();
        BigIntResult diff = bigint_sub(b, a);
        BigIntResult res = bigint_sub(p, diff.value);
        return res.value;
    }
}
