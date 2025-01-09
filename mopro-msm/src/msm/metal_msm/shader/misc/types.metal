#pragma once

using namespace metal;
#include <metal_stdlib>

struct BigInt {
    array<uint, NUM_LIMBS> limbs;
};

struct BigIntWide {
    array<uint, NUM_LIMBS_WIDE> limbs;
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
