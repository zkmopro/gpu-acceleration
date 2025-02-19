// #pragma once

// #ifndef BIGINT
// #define BIGINT
using namespace metal;
// #include "../constants.h"
#include <metal_stdlib>

#define NUM_LIMBS 16
#define NUM_LIMBS_WIDE 17
#define LOG_LIMB_SIZE 16
#define TWO_POW_WORD_SIZE 65536
#define MASK 65535
#define N0 25481
#define NSAFE 1

constant uint32_t BN254_BASEFIELD_MODULUS[NUM_LIMBS] = {
    64839,
    55420,
    35862,
    15392,
    51853,
    26737,
    27281,
    38785,
    22621,
    33153,
    17846,
    47184,
    41001,
    57649,
    20082,
    12388
};
constant uint32_t MONT_RADIX[NUM_LIMBS_WIDE] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
};
constant uint32_t BN254_ZERO_X[NUM_LIMBS] = {
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ZERO_Y[NUM_LIMBS] = {
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ZERO_Z[NUM_LIMBS] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ONE_X[NUM_LIMBS] = {
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ONE_Y[NUM_LIMBS] = {
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ONE_Z[NUM_LIMBS] = {
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ZERO_XR[NUM_LIMBS] = {
    3485,
    50575,
    17293,
    54109,
    2877,
    62919,
    60200,
    2680,
    17964,
    30841,
    41839,
    26222,
    57135,
    39431,
    30657,
    3594
};
constant uint32_t BN254_ZERO_YR[NUM_LIMBS] = {
    3485,
    50575,
    17293,
    54109,
    2877,
    62919,
    60200,
    2680,
    17964,
    30841,
    41839,
    26222,
    57135,
    39431,
    30657,
    3594
};
constant uint32_t BN254_ZERO_ZR[NUM_LIMBS] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
constant uint32_t BN254_ONE_XR[NUM_LIMBS] = {
    3485,
    50575,
    17293,
    54109,
    2877,
    62919,
    60200,
    2680,
    17964,
    30841,
    41839,
    26222,
    57135,
    39431,
    30657,
    3594
};
constant uint32_t BN254_ONE_YR[NUM_LIMBS] = {
    6970,
    35614,
    34587,
    42682,
    5755,
    60302,
    54865,
    5361,
    35928,
    61682,
    18142,
    52445,
    48734,
    13327,
    61315,
    7188
};
constant uint32_t BN254_ONE_ZR[NUM_LIMBS] = {
    3485,
    50575,
    17293,
    54109,
    2877,
    62919,
    60200,
    2680,
    17964,
    30841,
    41839,
    26222,
    57135,
    39431,
    30657,
    3594
};

typedef struct {
    array<uint, NUM_LIMBS> limbs;
}  BigInt;

typedef struct {
    array<uint, NUM_LIMBS_WIDE> limbs;
}  BigIntWide;

typedef struct {
    BigInt x;
    BigInt y;
    BigInt z;
} Jacobian;

Jacobian jacobian_zero();

typedef struct {
    BigInt x;
    BigInt y;
}  Affine;
// #endif // BIGINT

BigInt ff_add(
    BigInt a,
    BigInt b,
    BigInt p
);

BigInt ff_sub(
    BigInt a,
    BigInt b,
    BigInt p
);

BigInt bigint_zero();
BigInt bigint_add_unsafe(
    BigInt lhs,
    BigInt rhs
);

BigIntWide bigint_add_wide(
    BigInt lhs,
    BigInt rhs
);

BigInt bigint_sub(
    BigInt lhs,
    BigInt rhs
);

BigIntWide bigint_sub_wide(
    BigIntWide lhs,
    BigIntWide rhs
);

bool bigint_gte(
    BigInt lhs,
    BigInt rhs
);

bool bigint_wide_gte(
    BigIntWide lhs,
    BigIntWide rhs
);

bool bigint_eq(
    BigInt lhs,
    BigInt rhs
);

bool is_bigint_zero(BigInt x);

constexpr BigInt operator+(const BigInt lhs, const BigInt rhs);
constexpr BigInt operator-(const BigInt lhs, const BigInt rhs);
constexpr bool operator>=(const BigInt lhs, const BigInt rhs);
constexpr bool operator==(const BigInt lhs, const BigInt rhs);

bool jacobian_eq(
    Jacobian lhs,
    Jacobian rhs
);
bool is_jacobian_zero(Jacobian p);
constexpr bool operator==(const Jacobian lhs, const Jacobian rhs);


BigInt get_mu();
BigIntWide get_r();
BigInt get_p();
BigIntWide get_p_wide();
Jacobian get_bn254_zero();
Jacobian get_bn254_one();
Jacobian get_bn254_zero_mont();
Jacobian get_bn254_one_mont();

// #endif
Jacobian jacobian_dbl_2009_l(
    Jacobian pt,
    BigInt p
);

Jacobian jacobian_add_2007_bl(
    Jacobian a,
    Jacobian b,
    BigInt p
);

Jacobian jacobian_madd_2007_bl(
    Jacobian a,
    Affine b,
    BigInt p
);

Jacobian jacobian_scalar_mul(
    Jacobian point,
    uint scalar
);

Jacobian operator+(Jacobian a, Jacobian b);

BigInt conditional_reduce(
    BigInt x,
    BigInt y
);

BigInt mont_mul_optimised(
    BigInt x,
    BigInt y,
    BigInt p
);

BigInt mont_mul_modified(
    BigInt x,
    BigInt y,
    BigInt p
);

BigInt mont_mul_cios(
    BigInt x,
    BigInt y,
    BigInt p
);
