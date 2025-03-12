// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

/// Reduce if the value is greater than the modulus
FieldElement ff_reduce(FieldElement a) {
    BigIntResult sub_res = bigint_sub(a.value, a.modulus);
    if (sub_res.carry == 1) return a;
    return FieldElement{ .value = sub_res.value, .modulus = a.modulus };
}

/// Reduce once if the value is greater than the modulus
FieldElement ff_conditional_reduce(FieldElement a) {
    if (a.value >= a.modulus) {
        a.value = a.value - a.modulus;
    }
    return a;
}

FieldElement ff_add(FieldElement a, FieldElement b) {
    FieldElement res;
    res.value = a.value + b.value;
    res.modulus = a.modulus;
    return ff_reduce(res);
}

FieldElement ff_sub(FieldElement a, FieldElement b) {
    bool a_gte_b = bigint_gte(a.value, b.value);
    BigInt sub_res;
    if (a_gte_b) {
        sub_res = a.value - b.value;
    }
    else {
        // p - (b - a)
        BigInt diff = b.value - a.value;
        sub_res = a.modulus - diff;
    }
    return FieldElement{ .value = sub_res, .modulus = a.modulus };
}

// Overload Operators
constexpr FieldElement operator+(const FieldElement lhs, const FieldElement rhs) {
    return ff_add(lhs, rhs);
}

constexpr FieldElement operator-(const FieldElement lhs, const FieldElement rhs) {
    return ff_sub(lhs, rhs);
}
