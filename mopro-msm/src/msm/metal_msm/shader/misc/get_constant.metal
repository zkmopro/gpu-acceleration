#pragma once

using namespace metal;
#include "../misc/types.metal"
#include <metal_stdlib>

#define INIT_LIMB(i) BN254_BASEFIELD_MODULUS[i]

// 8 x 32-bit limbs for BN254
#define LIMBS_8 INIT_LIMB(0), INIT_LIMB(1), INIT_LIMB(2), INIT_LIMB(3), \
                INIT_LIMB(4), INIT_LIMB(5), INIT_LIMB(6), INIT_LIMB(7)

// Since modulus is constantly used, we initialize it here
constant BigInt MODULUS = {
#if (NUM_LIMBS == 8)
    LIMBS_8
#endif
};

BigInt get_mu()
{
    BigInt mu;

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        mu.limbs[i] = BARRETT_MU[i];
    }
    return mu;
}

BigInt get_n0()
{
    BigInt n0;

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        n0.limbs[i] = 0;
    }
    n0.limbs[0] = N0;
    return n0;
}

BigIntWide get_r()
{
    BigIntWide r;

#pragma unroll(9)
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        r.limbs[i] = MONT_RADIX[i];
    }
    return r;
}

/// R^2 mod p; mont_mul(x, R2) converts x into Montgomery form.
BigInt get_r2()
{
    BigInt r2;

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        r2.limbs[i] = BN254_R2[i];
    }
    return r2;
}

BigIntWide get_p_wide()
{
    BigIntWide p;

#pragma unroll(9)
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        p.limbs[i] = 0;
    }
#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigIntExtraWide get_p_extra_wide()
{
    BigIntExtraWide p;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        p.limbs[i] = 0;
    }
#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigInt bigint_zero()
{
    BigInt s;

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntWide bigint_zero_wide()
{
    BigIntWide s;

#pragma unroll(9)
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigIntExtraWide bigint_zero_extra_wide()
{
    BigIntExtraWide s;

#pragma unroll(16)
    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        s.limbs[i] = 0;
    }
    return s;
}

Jacobian get_bn254_zero()
{
    Jacobian zero;

#pragma unroll(8)
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

#pragma unroll(8)
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

#pragma unroll(8)
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

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_XR[i];
        one.y.limbs[i] = BN254_ONE_YR[i];
        one.z.limbs[i] = BN254_ONE_ZR[i];
    }
    return one;
}
