#pragma once

using namespace metal;
#include <metal_stdlib>
#include "../constants.metal"
#include "../misc/types.metal"

BigInt get_mu() {
    BigInt mu;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        mu.limbs[i] = BARRETT_MU[i];
    }
    return mu;
}

BigInt get_n0() {
    BigInt n0;
    uint tmp = N0;
    
    for (uint i = 0; i < NUM_LIMBS; i++) {
        n0.limbs[i] = tmp & MASK;
        tmp >>= LOG_LIMB_SIZE;
    }
    
    return n0;
}

BigIntWide get_r() {
    BigIntWide r;
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        r.limbs[i] = MONT_RADIX[i];
    }
    return r;
}

BigInt get_p() {
    BigInt p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigIntWide get_p_wide() {
    BigIntWide p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigIntExtraWide get_p_extra_wide() {
    BigIntExtraWide p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigInt bigint_zero() {
    BigInt s;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntWide bigint_zero_wide() {
    BigIntWide s;
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntExtraWide bigint_zero_extra_wide() {
    BigIntExtraWide s;
    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

Jacobian get_bn254_zero() {
    Jacobian zero;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_X[i];
        zero.y.limbs[i] = BN254_ZERO_Y[i];
        zero.z.limbs[i] = BN254_ZERO_Z[i];
    }
    return zero;
}

Jacobian get_bn254_one() {
    Jacobian one;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_X[i];
        one.y.limbs[i] = BN254_ONE_Y[i];
        one.z.limbs[i] = BN254_ONE_Z[i];
    }
    return one;
}

Jacobian get_bn254_zero_mont() {
    Jacobian zero;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_XR[i];
        zero.y.limbs[i] = BN254_ZERO_YR[i];
        zero.z.limbs[i] = BN254_ZERO_ZR[i];
    }
    return zero;
}

Jacobian get_bn254_one_mont() {
    Jacobian one;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_XR[i];
        one.y.limbs[i] = BN254_ONE_YR[i];
        one.z.limbs[i] = BN254_ONE_ZR[i];
    }
    return one;
}
