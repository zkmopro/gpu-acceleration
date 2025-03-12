#pragma once

using namespace metal;
#include <metal_stdlib>
#include "../constants.metal"

struct BigInt {
    array<uint, NUM_LIMBS> limbs;
};

struct BigIntWide {
    array<uint, NUM_LIMBS_WIDE> limbs;
};

struct BigIntExtraWide {
    array<uint, NUM_LIMBS_EXTRA_WIDE> limbs;
};

struct BigIntResult {
    BigInt value;
    bool carry;
};

struct BigIntResultWide {
    BigIntWide value;
    bool carry;
};

struct BigIntResultExtraWide {
    BigIntExtraWide value;
    bool carry;
};

// wrapper around BigInt to avoid having to pass modulus around
struct FieldElement {
    BigInt value;
    BigInt modulus;
};

struct Jacobian {
    FieldElement x;
    FieldElement y;
    FieldElement z;
};

struct Affine {
    FieldElement x;
    FieldElement y;
};
