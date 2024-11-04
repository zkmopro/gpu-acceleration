// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../field/ff.metal"

BigInt conditional_reduce(
    BigInt x,
    BigInt y
) {
    if (bigint_gte(x, y)) {
        return bigint_sub(x, y);
    }

    return x;
}

/// An optimised variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 12 and 13-bit limbs.
BigInt mont_mul_optimised(
    BigInt x,
    BigInt y,
    BigInt p
) {
    BigInt s = bigint_zero();

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;
        s.limbs[0] = s.limbs[1] + x.limbs[i] * y.limbs[1] + qi * p.limbs[1] + c;

        for (uint j = 2; j < NUM_LIMBS; j ++) {
            s.limbs[j - 1] = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
        }
        s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = s.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        s.limbs[i] = v & MASK;
    }

    return conditional_reduce(s, p);
}

/// An modified variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 14 and 15-bit limbs.
BigInt mont_mul_modified(
    BigInt x,
    BigInt y,
    BigInt p
) {
    BigInt s = bigint_zero();

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;

        for (uint j = 1; j < NUM_LIMBS - 1; j ++) {
            uint t = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
            if ((j - 1) % NSAFE == 0) {
                t = t + c;
            }

            c = t >> LOG_LIMB_SIZE;

            if (j % NSAFE == 0) {
                c = t >> LOG_LIMB_SIZE;
                s.limbs[j - 1] = t & MASK;
            } else {
                s.limbs[j - 1] = t;
            }
        }
        s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = s.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        s.limbs[i] = v & MASK;
    }

    return conditional_reduce(s, p);
}
