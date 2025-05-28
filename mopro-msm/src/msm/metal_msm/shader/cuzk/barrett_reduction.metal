#pragma once

#include <metal_stdlib>
#include <metal_math>
#include "../field/ff.metal"
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    constant os_log logger(/*subsystem=*/"barret_reduction", /*category=*/"metal");
    #define LOG_DEBUG_DUPL(...) logger.log_debug(__VA_ARGS__)
#else
    #define LOG_DEBUG_DUPL(...) ((void)0)
#endif

BigIntExtraWide mul(BigIntWide a, BigIntWide b) {
    BigIntExtraWide res = bigint_zero_extra_wide();
    
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        for (uint j = 0; j < NUM_LIMBS_WIDE; j++) {
            ulong c = (ulong)a.limbs[i] * (ulong)b.limbs[j];
            res.limbs[i+j] += c & MASK;
            res.limbs[i+j+1] += c >> LOG_LIMB_SIZE;
        }
    }

    // Start from 0 and carry the extra over to the next index.
    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        res.limbs[i+1] += res.limbs[i] >> LOG_LIMB_SIZE;
        res.limbs[i] = res.limbs[i] & MASK;
    }
    return res;
}

BigIntResultExtraWide sub_512(BigIntExtraWide a, BigIntExtraWide b) {
    BigIntResultExtraWide res;
    res.value = bigint_zero_extra_wide();
    res.carry = 0;

    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        res.value.limbs[i] = a.limbs[i] - b.limbs[i] - res.carry;
        if (a.limbs[i] < (b.limbs[i] + res.carry)) {
            res.value.limbs[i] += (MASK + 1);
            res.carry = 1;
        }
        else {
            res.carry = 0;
        }
    }
    return res;
}

BigIntResultExtraWide add_512(BigIntExtraWide a, BigIntExtraWide b) {
    BigIntResultExtraWide res;
    res.value = bigint_zero_extra_wide();
    res.carry = 0;

    for (uint i = 0; i < NUM_LIMBS_EXTRA_WIDE; i++) {
        ulong sum = (ulong)a.limbs[i] + (ulong)b.limbs[i] + res.carry;
        res.value.limbs[i] = sum & MASK;
        res.carry = sum >> LOG_LIMB_SIZE;
    }
    return res;
}

BigInt get_higher_with_slack(BigIntExtraWide a) {
    BigInt out = bigint_zero();
    for (uint i = 0; i < NUM_LIMBS; i++) {
        out.limbs[i] = ((a.limbs[i + NUM_LIMBS] << SLACK) + 
                        (a.limbs[i + NUM_LIMBS - 1] >> (LOG_LIMB_SIZE - SLACK))) & MASK;
    }
    return out;
}

BigInt barrett_reduce(BigIntExtraWide a) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        LOG_DEBUG_DUPL("res.limbs[%u] = %u", i, a.limbs[i]);
    }

    BigInt p = MODULUS;
    BigInt mu = get_mu();

    BigInt a_hi = get_higher_with_slack(a);
    BigIntExtraWide l = mul(bigint_to_wide(a_hi), bigint_to_wide(mu));
    BigInt l_hi = get_higher_with_slack(l);
    BigIntExtraWide lp = mul(bigint_to_wide(l_hi), bigint_to_wide(p));

    // Subtract lp from original a
    BigIntResultExtraWide sub_result = sub_512(a, lp);
    BigIntExtraWide r_wide = sub_result.value;

    // Handle underflow by adding back p_wide if needed
    if (sub_result.carry) {
        BigIntExtraWide p_wide = get_p_extra_wide();
        r_wide = add_512(r_wide, p_wide).value;
    }

    BigInt r = bigint_zero();
    for (uint i = 0; i < NUM_LIMBS; i++) {
        r.limbs[i] = r_wide.limbs[i];
    }

    return ff_reduce(r);
}

inline BigInt field_mul(BigIntWide a, BigIntWide b) {
    BigIntExtraWide xy = mul(a, b);
    return barrett_reduce(xy);
}
