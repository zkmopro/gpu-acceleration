// #include "get_constant.h"
// // #include "../constants.h"

// #define NUM_LIMBS 16
// #define NUM_LIMBS_WIDE 17
// #define LOG_LIMB_SIZE 16
// #define TWO_POW_WORD_SIZE 65536
// #define MASK 65535
// #define N0 25481
// #define NSAFE 1

// constant uint32_t BN254_BASEFIELD_MODULUS[NUM_LIMBS] = {
//     64839,
//     55420,
//     35862,
//     15392,
//     51853,
//     26737,
//     27281,
//     38785,
//     22621,
//     33153,
//     17846,
//     47184,
//     41001,
//     57649,
//     20082,
//     12388
// };
// constant uint32_t MONT_RADIX[NUM_LIMBS_WIDE] = {
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     1
// };
// constant uint32_t BN254_ZERO_X[NUM_LIMBS] = {
//     1,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ZERO_Y[NUM_LIMBS] = {
//     1,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ZERO_Z[NUM_LIMBS] = {
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ONE_X[NUM_LIMBS] = {
//     1,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ONE_Y[NUM_LIMBS] = {
//     2,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ONE_Z[NUM_LIMBS] = {
//     1,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ZERO_XR[NUM_LIMBS] = {
//     3485,
//     50575,
//     17293,
//     54109,
//     2877,
//     62919,
//     60200,
//     2680,
//     17964,
//     30841,
//     41839,
//     26222,
//     57135,
//     39431,
//     30657,
//     3594
// };
// constant uint32_t BN254_ZERO_YR[NUM_LIMBS] = {
//     3485,
//     50575,
//     17293,
//     54109,
//     2877,
//     62919,
//     60200,
//     2680,
//     17964,
//     30841,
//     41839,
//     26222,
//     57135,
//     39431,
//     30657,
//     3594
// };
// constant uint32_t BN254_ZERO_ZR[NUM_LIMBS] = {
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0,
//     0
// };
// constant uint32_t BN254_ONE_XR[NUM_LIMBS] = {
//     3485,
//     50575,
//     17293,
//     54109,
//     2877,
//     62919,
//     60200,
//     2680,
//     17964,
//     30841,
//     41839,
//     26222,
//     57135,
//     39431,
//     30657,
//     3594
// };
// constant uint32_t BN254_ONE_YR[NUM_LIMBS] = {
//     6970,
//     35614,
//     34587,
//     42682,
//     5755,
//     60302,
//     54865,
//     5361,
//     35928,
//     61682,
//     18142,
//     52445,
//     48734,
//     13327,
//     61315,
//     7188
// };
// constant uint32_t BN254_ONE_ZR[NUM_LIMBS] = {
//     3485,
//     50575,
//     17293,
//     54109,
//     2877,
//     62919,
//     60200,
//     2680,
//     17964,
//     30841,
//     41839,
//     26222,
//     57135,
//     39431,
//     30657,
//     3594
// };


// BigInt get_mu() {
//     BigInt mu;
//     uint n0 = N0;
    
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         mu.limbs[i] = n0 & MASK;
//         n0 >>= LOG_LIMB_SIZE;
//     }
    
//     return mu;
// }

// BigIntWide get_r() {
//     BigIntWide r;   // 257 bits
//     for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
//         r.limbs[i] = MONT_RADIX[i];
//     }
//     return r;
// }

// BigInt get_p() {
//     BigInt p;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
//     }
//     return p;
// }

// BigIntWide get_p_wide() {
//     BigIntWide p;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
//     }
//     return p;
// }

// Jacobian get_bn254_zero() {
//     Jacobian zero;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         zero.x.limbs[i] = BN254_ZERO_X[i];
//         zero.y.limbs[i] = BN254_ZERO_Y[i];
//         zero.z.limbs[i] = BN254_ZERO_Z[i];
//     }
//     return zero;
// }

// Jacobian get_bn254_one() {
//     Jacobian one;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         one.x.limbs[i] = BN254_ONE_X[i];
//         one.y.limbs[i] = BN254_ONE_Y[i];
//         one.z.limbs[i] = BN254_ONE_Z[i];
//     }
//     return one;
// }

// Jacobian get_bn254_zero_mont() {
//     Jacobian zero;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         zero.x.limbs[i] = BN254_ZERO_XR[i];
//         zero.y.limbs[i] = BN254_ZERO_YR[i];
//         zero.z.limbs[i] = BN254_ZERO_ZR[i];
//     }
//     return zero;
// }

// Jacobian get_bn254_one_mont() {
//     Jacobian one;
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         one.x.limbs[i] = BN254_ONE_XR[i];
//         one.y.limbs[i] = BN254_ONE_YR[i];
//         one.z.limbs[i] = BN254_ONE_ZR[i];
//     }
//     return one;
// }
