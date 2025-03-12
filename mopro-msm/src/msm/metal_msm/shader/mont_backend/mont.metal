// adapted from: https://github.com/geometryxyz/msl-secp256k1
#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdivision-by-zero" // to avoid warning on debug build, but we should always know what NSAFE is

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../field/ff.metal"

/// An optimised variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 12 and 13-bit limbs.
FieldElement mont_mul_optimised(FieldElement x, FieldElement y) {
    FieldElement res = FieldElement{ bigint_zero() };
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = res.value.limbs[0] + x.value.limbs[i] * y.value.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * (*get_p()).limbs[0]) >> LOG_LIMB_SIZE;
        res.value.limbs[0] = res.value.limbs[1] + x.value.limbs[i] * y.value.limbs[1] + qi * (*get_p()).limbs[1] + c;

        for (uint j = 2; j < NUM_LIMBS; j ++) {
            res.value.limbs[j - 1] = res.value.limbs[j] + x.value.limbs[i] * y.value.limbs[j] + qi * (*get_p()).limbs[j];
        }
        res.value.limbs[NUM_LIMBS - 2] = x.value.limbs[i] * y.value.limbs[NUM_LIMBS - 1] + qi * (*get_p()).limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = res.value.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        res.value.limbs[i] = v & MASK;
    }
    return ff_conditional_reduce(res);
}

/// An modified variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 14 and 15-bit limbs.
FieldElement mont_mul_modified(FieldElement x, FieldElement y) {
    FieldElement res = FieldElement{ bigint_zero() };
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = res.value.limbs[0] + x.value.limbs[i] * y.value.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * (*get_p()).limbs[0]) >> LOG_LIMB_SIZE;

        for (uint j = 1; j < NUM_LIMBS - 1; j ++) {
            uint t = res.value.limbs[j] + x.value.limbs[i] * y.value.limbs[j] + qi * (*get_p()).limbs[j];
            if ((j - 1) % NSAFE == 0) {
                t = t + c;
            }

            c = t >> LOG_LIMB_SIZE;

            if (j % NSAFE == 0) {
                c = t >> LOG_LIMB_SIZE;
                res.value.limbs[j - 1] = t & MASK;
            }
            else {
                res.value.limbs[j - 1] = t;
            }
        }
        res.value.limbs[NUM_LIMBS - 2] = x.value.limbs[i] * y.value.limbs[NUM_LIMBS - 1] + qi * (*get_p()).limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = res.value.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        res.value.limbs[i] = v & MASK;
    }
    return ff_conditional_reduce(res);
}

/// The CIOS method for Montgomery multiplication from Tolga Acar's thesis:
/// High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
/// https://www.proquest.com/openview/1018972f191afe55443658b28041c118/1
FieldElement mont_mul_cios(FieldElement x, FieldElement y) {
    FieldElement res = FieldElement{ bigint_zero() };
    uint t[NUM_LIMBS + 2] = {0};  // Extra space for carries

    for (uint i = 0; i < NUM_LIMBS; i++) {
        // Step 1: Multiply and add
        uint c = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            uint r = t[j] + x.value.limbs[j] * y.value.limbs[i] + c;
            c = r >> LOG_LIMB_SIZE;
            t[j] = r & MASK;
        }
        uint r = t[NUM_LIMBS] + c;
        t[NUM_LIMBS + 1] = r >> LOG_LIMB_SIZE;
        t[NUM_LIMBS] = r & MASK;

        // Step 2: Reduce
        uint m = (t[0] * N0) & MASK;
        r = t[0] + m * (*get_p()).limbs[0];
        c = r >> LOG_LIMB_SIZE;

        for (uint j = 1; j < NUM_LIMBS; j++) {
            r = t[j] + m * (*get_p()).limbs[j] + c;
            c = r >> LOG_LIMB_SIZE;
            t[j - 1] = r & MASK;
        }

        r = t[NUM_LIMBS] + c;
        c = r >> LOG_LIMB_SIZE;
        t[NUM_LIMBS - 1] = r & MASK;
        t[NUM_LIMBS] = t[NUM_LIMBS + 1] + c;
    }

    // Final reduction check
    bool t_lt_p = false;
    for (uint idx = 0; idx < NUM_LIMBS; idx++) {
        uint i = NUM_LIMBS - 1 - idx;
        if (t[i] < (*get_p()).limbs[i]) {
            t_lt_p = true;
            break;
        } else if (t[i] > (*get_p()).limbs[i]) {
            break;
        }
    }

    if (t_lt_p) {
        for (uint i = 0; i < NUM_LIMBS; i++) {
            res.value.limbs[i] = t[i];
        }
    } else {
        uint borrow = 0;
        for (uint i = 0; i < NUM_LIMBS; i++) {
            uint diff = t[i] - (*get_p()).limbs[i] - borrow;
            if (t[i] < ((*get_p()).limbs[i] + borrow)) {
                diff += (1 << LOG_LIMB_SIZE);
                borrow = 1;
            } else {
                borrow = 0;
            }
            res.value.limbs[i] = diff;
        }
    }
    return res;
}

// Overload Operators with Default Montgomery Multiplication for FieldElement
constexpr FieldElement operator*(const FieldElement lhs, const FieldElement rhs) {
    return mont_mul_cios(lhs, rhs);
}

#pragma clang diagnostic pop
