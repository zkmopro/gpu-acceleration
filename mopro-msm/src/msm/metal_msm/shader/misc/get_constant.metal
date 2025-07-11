#pragma once

using namespace metal;
#include "../misc/types.metal"
#include <metal_stdlib>

#define INIT_LIMB(i) BN254_BASEFIELD_MODULUS[i]

// for 16-bit mont_mul_cios (optimised for BN254)
#define LIMBS_16 INIT_LIMB(0), INIT_LIMB(1), INIT_LIMB(2), INIT_LIMB(3),   \
                 INIT_LIMB(4), INIT_LIMB(5), INIT_LIMB(6), INIT_LIMB(7),   \
                 INIT_LIMB(8), INIT_LIMB(9), INIT_LIMB(10), INIT_LIMB(11), \
                 INIT_LIMB(12), INIT_LIMB(13), INIT_LIMB(14), INIT_LIMB(15)
// for 15-bit mont_mul_modified
#define LIMBS_17 INIT_LIMB(0), INIT_LIMB(1), INIT_LIMB(2), INIT_LIMB(3),     \
                 INIT_LIMB(4), INIT_LIMB(5), INIT_LIMB(6), INIT_LIMB(7),     \
                 INIT_LIMB(8), INIT_LIMB(9), INIT_LIMB(10), INIT_LIMB(11),   \
                 INIT_LIMB(12), INIT_LIMB(13), INIT_LIMB(14), INIT_LIMB(15), \
                 INIT_LIMB(16)
// for 13-bit mont_mul_optimised
#define LIMBS_20 INIT_LIMB(0), INIT_LIMB(1), INIT_LIMB(2), INIT_LIMB(3),     \
                 INIT_LIMB(4), INIT_LIMB(5), INIT_LIMB(6), INIT_LIMB(7),     \
                 INIT_LIMB(8), INIT_LIMB(9), INIT_LIMB(10), INIT_LIMB(11),   \
                 INIT_LIMB(12), INIT_LIMB(13), INIT_LIMB(14), INIT_LIMB(15), \
                 INIT_LIMB(16), INIT_LIMB(17), INIT_LIMB(18), INIT_LIMB(19)

// Since modulus is constantly used, we initialize it here
constant BigInt MODULUS = {
#if (NUM_LIMBS == 16)
    LIMBS_16
#elif (NUM_LIMBS == 17)
    LIMBS_17
#elif (NUM_LIMBS == 20)
    LIMBS_20
#endif
};

BigInt get_mu()
{
    BigInt mu;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        mu.limbs[i] = BARRETT_MU[i];
    }
    return mu;
}

BigInt get_n0()
{
    BigInt n0;
    uint tmp = N0;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        n0.limbs[i] = tmp & MASK;
        tmp >>= LOG_LIMB_SIZE;
    }
    return n0;
}

BigIntWide get_r()
{
    BigIntWide r;

#pragma unroll(17)
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        r.limbs[i] = MONT_RADIX[i];
    }
    return r;
}

BigIntWide get_p_wide()
{
    BigIntWide p;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigIntExtraWide get_p_extra_wide()
{
    BigIntExtraWide p;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigInt bigint_zero()
{
    BigInt s;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntWide bigint_zero_wide()
{
    BigIntWide s;

#pragma unroll(17)
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntExtraWide bigint_zero_extra_wide()
{
    BigIntExtraWide s;

#pragma unroll(32)
    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

Jacobian get_bn254_zero()
{
    Jacobian zero;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_X[i];
        zero.y.limbs[i] = BN254_ZERO_Y[i];
        zero.z.limbs[i] = BN254_ZERO_Z[i];
    }
    return zero;
}

Jacobian get_bn254_one()
{
    Jacobian one;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_X[i];
        one.y.limbs[i] = BN254_ONE_Y[i];
        one.z.limbs[i] = BN254_ONE_Z[i];
    }
    return one;
}

Jacobian get_bn254_zero_mont()
{
    Jacobian zero;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_XR[i];
        zero.y.limbs[i] = BN254_ZERO_YR[i];
        zero.z.limbs[i] = BN254_ZERO_ZR[i];
    }
    return zero;
}

Jacobian get_bn254_one_mont()
{
    Jacobian one;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_XR[i];
        one.y.limbs[i] = BN254_ONE_YR[i];
        one.z.limbs[i] = BN254_ONE_ZR[i];
    }
    return one;
}
