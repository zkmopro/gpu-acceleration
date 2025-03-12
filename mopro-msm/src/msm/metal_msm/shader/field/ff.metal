// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"

/// Reduce if the value is greater than the modulus
FieldElement ff_reduce(FieldElement a) {
    BigIntResult sub_res = bigint_sub(a.value, *get_p());
    if (sub_res.carry == 1) return a;
    return FieldElement{ sub_res.value };
}

/// Reduce once if the value is greater than the modulus
FieldElement ff_conditional_reduce(FieldElement a) {
    if (a.value >= *get_p()) {
        a.value = a.value - *get_p();
    }
    return a;
}

FieldElement ff_add(FieldElement a, FieldElement b) {
    FieldElement res;
    res.value = a.value + b.value;
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
        sub_res = *get_p() - diff;
    }
    return FieldElement{ sub_res };
}

// Overload Operators
constexpr FieldElement operator+(const FieldElement lhs, const FieldElement rhs) {
    return ff_add(lhs, rhs);
}

constexpr FieldElement operator-(const FieldElement lhs, const FieldElement rhs) {
    return ff_sub(lhs, rhs);
}
