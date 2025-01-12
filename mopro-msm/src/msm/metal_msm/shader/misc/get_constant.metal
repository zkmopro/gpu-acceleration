#pragma once

using namespace metal;
#include <metal_stdlib>
#include "../constants.metal"
#include "../misc/types.metal"

BigInt get_mu() {
    BigInt mu;
    uint n0 = N0;
    
    for (uint i = 0; i < NUM_LIMBS; i++) {
        mu.limbs[i] = n0 & MASK;
        n0 >>= LOG_LIMB_SIZE;
    }
    
    return mu;
}

BigIntWide get_r() {
    BigIntWide r;   // 257 bits
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
