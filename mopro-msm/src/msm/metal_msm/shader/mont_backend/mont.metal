// Montgomery multiplication for BN254 on 8 x 32-bit limbs (R = 2^256).
// CIOS (Coarsely Integrated Operand Scanning) with 64-bit accumulators:
// each x_j * y_i product is a native 32x32->64 multiply-add on Apple GPUs.
#pragma once

using namespace metal;
#include "../field/ff.metal"
#include <metal_math>
#include <metal_stdlib>

BigInt conditional_reduce(
    BigInt x,
    BigInt y)
{
    if (x >= y) {
        return x - y;
    }
    return x;
}

/// Fused Montgomery multiplication using the LogJumps reduction
/// (Domb / Koh, https://kohweijie.com/articles/25/logjumps.html).
///
/// Instead of Montgomery's per-round quotient q = t0*mu and t += q*p, each of
/// the first NUM_LIMBS-1 rounds directly replaces the low limb t0 by
/// t0 * rho (rho = 2^-32 mod p) added into the shifted remainder — n muls per
/// round instead of n+1, and no serial dependency on the quotient. The final
/// round is a classic Montgomery round to bring the result down to 8 limbs.
/// Total: n^2 + 1 = 65 limb products vs n^2 + n = 72 for CIOS.
inline BigInt mont_mul_logjumps(BigInt x, BigInt y)
{
    BigInt p = MODULUS;
    uint s[NUM_LIMBS + 2] = { 0 };

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        // Row accumulate: s += x * y[i]
        ulong c = 0;

#pragma unroll(8)
        for (uint j = 0; j < NUM_LIMBS; j++) {
            ulong r = (ulong)s[j] + (ulong)x.limbs[j] * (ulong)y.limbs[i] + c;
            s[j] = (uint)r;
            c = r >> LOG_LIMB_SIZE;
        }
        ulong r = (ulong)s[NUM_LIMBS] + c;
        s[NUM_LIMBS] = (uint)r;
        s[NUM_LIMBS + 1] += (uint)(r >> LOG_LIMB_SIZE);

        if (i < NUM_LIMBS - 1) {
            // LogJumps round: s = (s - s0)/2^32 + s0 * rho  (== s * 2^-32 mod p)
            uint s0 = s[0];
            c = 0;

#pragma unroll(8)
            for (uint j = 0; j < NUM_LIMBS; j++) {
                ulong r2 = (ulong)s[j + 1] + (ulong)s0 * (ulong)BN254_RHO[j] + c;
                s[j] = (uint)r2;
                c = r2 >> LOG_LIMB_SIZE;
            }
            r = (ulong)s[NUM_LIMBS + 1] + c;
            s[NUM_LIMBS] = (uint)r;
            s[NUM_LIMBS + 1] = (uint)(r >> LOG_LIMB_SIZE);
        } else {
            // Final Montgomery round: zero the low limb and shift
            uint q = s[0] * N0;
            ulong r2 = (ulong)s[0] + (ulong)q * (ulong)p.limbs[0];
            c = r2 >> LOG_LIMB_SIZE;

#pragma unroll(7)
            for (uint j = 1; j < NUM_LIMBS; j++) {
                r2 = (ulong)s[j] + (ulong)q * (ulong)p.limbs[j] + c;
                s[j - 1] = (uint)r2;
                c = r2 >> LOG_LIMB_SIZE;
            }
            r2 = (ulong)s[NUM_LIMBS] + c;
            s[NUM_LIMBS - 1] = (uint)r2;
            s[NUM_LIMBS] = s[NUM_LIMBS + 1] + (uint)(r2 >> LOG_LIMB_SIZE);
        }
    }

    // Conditional subtraction into [0, p). Because rho ~ 0.89p for BN254 the
    // pre-reduction value can reach ~2.9p, so up to two subtractions of p are
    // needed (verified against the exact worst-case recurrence).
#pragma unroll(2)
    for (uint round = 0; round < 2; round++) {
        bool s_gte_p = s[NUM_LIMBS] != 0;
        if (!s_gte_p) {
            s_gte_p = true;
#pragma unroll(8)
            for (uint idx = 0; idx < NUM_LIMBS; idx++) {
                uint i = NUM_LIMBS - 1 - idx;
                if (s[i] < p.limbs[i]) {
                    s_gte_p = false;
                    break;
                } else if (s[i] > p.limbs[i]) {
                    break;
                }
            }
        }
        if (s_gte_p) {
            uint borrow = 0;

#pragma unroll(8)
            for (uint i = 0; i < NUM_LIMBS; i++) {
                ulong rb = (ulong)p.limbs[i] + borrow;
                uint d = s[i] - (uint)rb;
                borrow = ((ulong)s[i] < rb) ? 1u : 0u;
                s[i] = d;
            }
            s[NUM_LIMBS] -= borrow;
        }
    }

    BigInt result;
#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        result.limbs[i] = s[i];
    }

    return result;
}

/// The CIOS method for Montgomery multiplication from Tolga Acar's thesis:
/// High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
/// https://www.proquest.com/openview/1018972f191afe55443658b28041c118/1
///
/// For p < R/4 (BN254: 254 bits vs R = 2^256) the intermediate value stays
/// below 2p, so t fits in NUM_LIMBS + 1 limbs and a single conditional
/// subtraction at the end suffices.
inline BigInt mont_mul_cios(BigInt x, BigInt y)
{
    BigInt p = MODULUS;
    uint t[NUM_LIMBS + 2] = { 0 };

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        // Step 1: t += x * y[i]
        ulong c = 0;

#pragma unroll(8)
        for (uint j = 0; j < NUM_LIMBS; j++) {
            ulong r = (ulong)t[j] + (ulong)x.limbs[j] * (ulong)y.limbs[i] + c;
            t[j] = (uint)r;
            c = r >> LOG_LIMB_SIZE;
        }
        ulong r = (ulong)t[NUM_LIMBS] + c;
        t[NUM_LIMBS] = (uint)r;
        t[NUM_LIMBS + 1] = (uint)(r >> LOG_LIMB_SIZE);

        // Step 2: reduce (make the lowest limb divisible by 2^32 and shift)
        uint m = t[0] * N0;
        ulong r2 = (ulong)t[0] + (ulong)m * (ulong)p.limbs[0];
        c = r2 >> LOG_LIMB_SIZE;

#pragma unroll(7)
        for (uint j = 1; j < NUM_LIMBS; j++) {
            r2 = (ulong)t[j] + (ulong)m * (ulong)p.limbs[j] + c;
            t[j - 1] = (uint)r2;
            c = r2 >> LOG_LIMB_SIZE;
        }

        r2 = (ulong)t[NUM_LIMBS] + c;
        t[NUM_LIMBS - 1] = (uint)r2;
        t[NUM_LIMBS] = t[NUM_LIMBS + 1] + (uint)(r2 >> LOG_LIMB_SIZE);
    }

    // Final conditional subtraction: t is in [0, 2p), so subtract p iff t >= p.
    bool t_gte_p = t[NUM_LIMBS] != 0;
    if (!t_gte_p) {
        t_gte_p = true;
#pragma unroll(8)
        for (uint idx = 0; idx < NUM_LIMBS; idx++) {
            uint i = NUM_LIMBS - 1 - idx;
            if (t[i] < p.limbs[i]) {
                t_gte_p = false;
                break;
            } else if (t[i] > p.limbs[i]) {
                break;
            }
        }
    }

    BigInt result;
    if (t_gte_p) {
        uint borrow = 0;

#pragma unroll(8)
        for (uint i = 0; i < NUM_LIMBS; i++) {
            ulong rb = (ulong)p.limbs[i] + borrow;
            result.limbs[i] = t[i] - (uint)rb;
            borrow = ((ulong)t[i] < rb) ? 1u : 0u;
        }
    } else {
#pragma unroll(8)
        for (uint i = 0; i < NUM_LIMBS; i++) {
            result.limbs[i] = t[i];
        }
    }

    return result;
}

/// Montgomery multiplication entry point used by the rest of the pipeline.
/// CIOS measured ~8-10% faster end-to-end than mont_mul_logjumps on Apple
/// GPUs: BN254's rho = 2^-32 mod p is ~0.89p, which forces a second
/// conditional subtraction that cancels LogJumps' n-1 saved multiplications.
inline BigInt mont_mul(BigInt x, BigInt y)
{
    return mont_mul_cios(x, y);
}
