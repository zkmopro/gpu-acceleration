#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../bigint/bigint.metal"
#include "../misc/get_constant.metal"

bool is_jacobian_zero(Jacobian a) {
    return is_bigint_zero(a.z.value);
}

bool jacobian_eq(Jacobian lhs, Jacobian rhs) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (lhs.x.value.limbs[i] != rhs.x.value.limbs[i]) return false;
        else if (lhs.y.value.limbs[i] != rhs.y.value.limbs[i]) return false;
        else if (lhs.z.value.limbs[i] != rhs.z.value.limbs[i]) return false;
    }
    return true;
}

constexpr bool operator==(const Jacobian lhs, const Jacobian rhs) {
    return jacobian_eq(lhs, rhs);
}
