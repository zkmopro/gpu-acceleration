// // adapted from: https://github.com/geometryxyz/msl-secp256k1

// #include "mont.h"
// #include <metal_stdlib>
// #include <metal_math>

// #define NUM_LIMBS 16
// #define NUM_LIMBS_WIDE 17
// #define LOG_LIMB_SIZE 16
// #define TWO_POW_WORD_SIZE 65536
// #define MASK 65535
// #define N0 25481
// #define NSAFE 1

// BigInt conditional_reduce(
//     BigInt x,
//     BigInt y
// ) {
//     if (x >= y) {
//         return x - y;
//     }

//     return x;
// }

// /// An optimised variant of the Montgomery product algorithm from
// /// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
// /// Known to work with 12 and 13-bit limbs.
// BigInt mont_mul_optimised(
//     BigInt x,
//     BigInt y,
//     BigInt p
// ) {
//     BigInt s = bigint_zero();

//     for (uint i = 0; i < NUM_LIMBS; i ++) {
//         uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
//         uint tprime = t & MASK;
//         uint qi = (N0 * tprime) & MASK;
//         uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;
//         s.limbs[0] = s.limbs[1] + x.limbs[i] * y.limbs[1] + qi * p.limbs[1] + c;

//         for (uint j = 2; j < NUM_LIMBS; j ++) {
//             s.limbs[j - 1] = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
//         }
//         s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
//     }

//     uint c = 0;
//     for (uint i = 0; i < NUM_LIMBS; i ++) {
//         uint v = s.limbs[i] + c;
//         c = v >> LOG_LIMB_SIZE;
//         s.limbs[i] = v & MASK;
//     }

//     return conditional_reduce(s, p);
// }

// /// An modified variant of the Montgomery product algorithm from
// /// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
// /// Known to work with 14 and 15-bit limbs.
// BigInt mont_mul_modified(
//     BigInt x,
//     BigInt y,
//     BigInt p
// ) {
//     BigInt s = bigint_zero();

//     for (uint i = 0; i < NUM_LIMBS; i ++) {
//         uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
//         uint tprime = t & MASK;
//         uint qi = (N0 * tprime) & MASK;
//         uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;

//         for (uint j = 1; j < NUM_LIMBS - 1; j ++) {
//             uint t = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
//             if ((j - 1) % NSAFE == 0) {
//                 t = t + c;
//             }

//             c = t >> LOG_LIMB_SIZE;

//             if (j % NSAFE == 0) {
//                 c = t >> LOG_LIMB_SIZE;
//                 s.limbs[j - 1] = t & MASK;
//             } else {
//                 s.limbs[j - 1] = t;
//             }
//         }
//         s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
//     }

//     uint c = 0;
//     for (uint i = 0; i < NUM_LIMBS; i ++) {
//         uint v = s.limbs[i] + c;
//         c = v >> LOG_LIMB_SIZE;
//         s.limbs[i] = v & MASK;
//     }

//     return conditional_reduce(s, p);
// }

// /// The CIOS method for Montgomery multiplication from Tolga Acar's thesis:
// /// High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
// /// https://www.proquest.com/openview/1018972f191afe55443658b28041c118/1
// BigInt mont_mul_cios(
//     BigInt x,
//     BigInt y,
//     BigInt p
// ) {
//     uint t[NUM_LIMBS + 2] = {0};  // Extra space for carries
//     BigInt result;
    
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         // Step 1: Multiply and add
//         uint c = 0;
//         for (uint j = 0; j < NUM_LIMBS; j++) {
//             uint r = t[j] + x.limbs[j] * y.limbs[i] + c;
//             c = r >> LOG_LIMB_SIZE;
//             t[j] = r & MASK;
//         }
//         uint r = t[NUM_LIMBS] + c;
//         t[NUM_LIMBS + 1] = r >> LOG_LIMB_SIZE;
//         t[NUM_LIMBS] = r & MASK;

//         // Step 2: Reduce
//         uint m = (t[0] * N0) & MASK;
//         r = t[0] + m * p.limbs[0];
//         c = r >> LOG_LIMB_SIZE;

//         for (uint j = 1; j < NUM_LIMBS; j++) {
//             r = t[j] + m * p.limbs[j] + c;
//             c = r >> LOG_LIMB_SIZE;
//             t[j - 1] = r & MASK;
//         }

//         r = t[NUM_LIMBS] + c;
//         c = r >> LOG_LIMB_SIZE;
//         t[NUM_LIMBS - 1] = r & MASK;
//         t[NUM_LIMBS] = t[NUM_LIMBS + 1] + c;
//     }

//     // Final reduction check
//     bool t_lt_p = false;
//     for (uint idx = 0; idx < NUM_LIMBS; idx++) {
//         uint i = NUM_LIMBS - 1 - idx;
//         if (t[i] < p.limbs[i]) {
//             t_lt_p = true;
//             break;
//         } else if (t[i] > p.limbs[i]) {
//             break;
//         }
//     }

//     if (t_lt_p) {
//         for (uint i = 0; i < NUM_LIMBS; i++) {
//             result.limbs[i] = t[i];
//         }
//     } else {
//         uint borrow = 0;
//         for (uint i = 0; i < NUM_LIMBS; i++) {
//             uint diff = t[i] - p.limbs[i] - borrow;
//             if (t[i] < (p.limbs[i] + borrow)) {
//                 diff += (1 << LOG_LIMB_SIZE);
//                 borrow = 1;
//             } else {
//                 borrow = 0;
//             }
//             result.limbs[i] = diff;
//         }
//     }

//     return result;
// }
