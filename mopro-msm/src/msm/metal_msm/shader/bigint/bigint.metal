// source: https://github.com/geometryxyz/msl-secp256k1
#pragma once

using namespace metal;
#include "../misc/get_constant.metal"


BigIntResult bigint_add_unsafe(
    BigInt lhs,
    BigInt rhs
) {
    BigIntResult res;
    res.carry = 0;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + res.carry;
        res.value.limbs[i] = c & mask;
        res.carry = c >> LOG_LIMB_SIZE;
    }
    return res;
}

BigIntResultWide bigint_add_wide(
    BigInt lhs,
    BigInt rhs
) {
    BigIntResultWide res;
    res.carry = 0;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    uint carry = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        res.value.limbs[i] = c & mask;
        carry = c >> LOG_LIMB_SIZE;
    }
    res.value.limbs[NUM_LIMBS] = carry;
    res.carry = carry;
    return res;
}

BigIntResult bigint_sub(
    BigInt lhs,
    BigInt rhs
) {
    BigIntResult res;
    res.carry = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        res.value.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - res.carry;
        if (lhs.limbs[i] < rhs.limbs[i] + res.carry) {
            res.value.limbs[i] += TWO_POW_WORD_SIZE;
            res.carry = 1;
        } else {
            res.carry = 0;
        }
    }
    return res;
}


BigIntResultWide bigint_sub_wide(
    BigIntWide lhs,
    BigIntWide rhs
) {
    BigIntResultWide res;
    res.carry = 0;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        res.value.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - res.carry;
        if (lhs.limbs[i] < rhs.limbs[i] + res.carry) {
            res.value.limbs[i] += TWO_POW_WORD_SIZE;
            res.carry = 1;
        } else {
            res.carry = 0;
        }
    }
    return res;
}

bool bigint_gte(
    BigInt lhs,
    BigInt rhs
) {
    // for (uint i = NUM_LIMBS-1; i >= 0; i--) is troublesome from unknown reason
    for (uint idx = 0; idx < NUM_LIMBS; idx++) {
        uint i = NUM_LIMBS - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) return false;
        else if (lhs.limbs[i] > rhs.limbs[i]) return true;
    }
    return true;
}

bool bigint_wide_gte(
    BigIntWide lhs,
    BigIntWide rhs
) {
    for (uint idx = 0; idx < NUM_LIMBS_WIDE; idx++) {
        uint i = NUM_LIMBS_WIDE - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) return false;
        else if (lhs.limbs[i] > rhs.limbs[i]) return true;
    }
    return true;
}

bool bigint_eq(
    BigInt lhs,
    BigInt rhs
) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (lhs.limbs[i] != rhs.limbs[i]) return false;
    }
    return true;
}

bool is_bigint_zero(BigInt x) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (x.limbs[i] != 0) return false;
    }
    return true;
}

bool bigint_eq(
    BigInt lhs,
    BigInt rhs
) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (lhs.limbs[i] != rhs.limbs[i]) {
            return false;
        }
    }
    return true;
}

bool is_bigint_zero(BigInt x) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (x.limbs[i] != 0) {
            return false;
        }
    }
    return true;
}

// Overload Operators
constexpr BigInt operator+(const BigInt lhs, const BigInt rhs) {
    return bigint_add_unsafe(lhs, rhs).value;
}

constexpr BigInt operator-(const BigInt lhs, const BigInt rhs) {
    return bigint_sub(lhs, rhs).value;
}

constexpr bool operator>=(const BigInt lhs, const BigInt rhs) {
    return bigint_gte(lhs, rhs);
}

constexpr bool operator==(const BigInt lhs, const BigInt rhs) {
    return bigint_eq(lhs, rhs);
}