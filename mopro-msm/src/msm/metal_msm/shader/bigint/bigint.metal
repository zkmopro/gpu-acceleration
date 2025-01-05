// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include "../constants.metal"

struct BigInt {
    array<uint, NUM_LIMBS> limbs;
};

struct BigIntWide {
    array<uint, NUM_LIMBS_WIDE> limbs;
};

BigInt bigint_zero() {
    BigInt s;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigInt bigint_add_unsafe(
    BigInt lhs,
    BigInt rhs
) {
    BigInt result;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    uint carry = 0;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & mask;
        carry = c >> LOG_LIMB_SIZE;
    }
    return result;
}

BigIntWide bigint_add_wide(
    BigInt lhs,
    BigInt rhs
) {
    BigIntWide result;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    uint carry = 0;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & mask;
        carry = c >> LOG_LIMB_SIZE;
    }
    result.limbs[NUM_LIMBS] = carry;

    return result;
}

BigInt bigint_sub(
    BigInt lhs,
    BigInt rhs
) {
    uint borrow = 0;

    BigInt res;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;

        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    return res;
}


BigIntWide bigint_sub_wide(
    BigIntWide lhs,
    BigIntWide rhs
) {
    uint borrow = 0;

    BigIntWide res;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;

        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    return res;
}

bool bigint_gte(
    BigInt lhs,
    BigInt rhs
) {
    for (uint idx = 0; idx < NUM_LIMBS; idx ++) {
        uint i = NUM_LIMBS - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) {
            return false;
        } else if (lhs.limbs[i] > rhs.limbs[i]) {
            return true;
        }
    }

    return true;
}

bool bigint_wide_gte(
    BigIntWide lhs,
    BigIntWide rhs
) {
    for (uint idx = 0; idx < NUM_LIMBS_WIDE; idx ++) {
        uint i = NUM_LIMBS_WIDE - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) {
            return false;
        } else if (lhs.limbs[i] > rhs.limbs[i]) {
            return true;
        }
    }

    return true;
}

BigInt get_bn254_basefield_modulus() {
    BigInt modulus;
    modulus.limbs[0] = BN254_BASEFIELD_MODULUS_LIMB_0;
    modulus.limbs[1] = BN254_BASEFIELD_MODULUS_LIMB_1;
    modulus.limbs[2] = BN254_BASEFIELD_MODULUS_LIMB_2;
    modulus.limbs[3] = BN254_BASEFIELD_MODULUS_LIMB_3;
    modulus.limbs[4] = BN254_BASEFIELD_MODULUS_LIMB_4;
    modulus.limbs[5] = BN254_BASEFIELD_MODULUS_LIMB_5;
    modulus.limbs[6] = BN254_BASEFIELD_MODULUS_LIMB_6;
    modulus.limbs[7] = BN254_BASEFIELD_MODULUS_LIMB_7;
    modulus.limbs[8] = BN254_BASEFIELD_MODULUS_LIMB_8;
    modulus.limbs[9] = BN254_BASEFIELD_MODULUS_LIMB_9;
    modulus.limbs[10] = BN254_BASEFIELD_MODULUS_LIMB_10;
    modulus.limbs[11] = BN254_BASEFIELD_MODULUS_LIMB_11;
    modulus.limbs[12] = BN254_BASEFIELD_MODULUS_LIMB_12;
    modulus.limbs[13] = BN254_BASEFIELD_MODULUS_LIMB_13;
    modulus.limbs[14] = BN254_BASEFIELD_MODULUS_LIMB_14;
    modulus.limbs[15] = BN254_BASEFIELD_MODULUS_LIMB_15;
    return modulus;
}

// Overload Operators
constexpr BigInt operator+(const BigInt lhs, const BigInt rhs) {
    return bigint_add_unsafe(lhs, rhs);
}

constexpr BigInt operator-(const BigInt lhs, const BigInt rhs) {
    return bigint_sub(lhs, rhs);
}

constexpr bool operator>=(const BigInt lhs, const BigInt rhs) {
    return bigint_gte(lhs, rhs);
}
