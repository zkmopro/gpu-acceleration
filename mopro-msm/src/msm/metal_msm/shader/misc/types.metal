#pragma once

using namespace metal;
#include "../constants.metal"
#include <metal_stdlib>

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

struct Jacobian {
    BigInt x;
    BigInt y;
    BigInt z;
};

struct Affine {
    BigInt x;
    BigInt y;
};
