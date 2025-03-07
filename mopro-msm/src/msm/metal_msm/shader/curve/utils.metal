#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>

bool jacobian_eq(
    Jacobian lhs,
    Jacobian rhs
) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (lhs.x.limbs[i] != rhs.x.limbs[i]) {
            return false;
        } else if (lhs.y.limbs[i] != rhs.y.limbs[i]) {
            return false;
        } else if (lhs.z.limbs[i] != rhs.z.limbs[i]) {
            return false;
        }
    }
    return true;
}

bool is_jacobian_zero(Jacobian a) {
    return is_bigint_zero(a.z);
}

constexpr bool operator==(const Jacobian lhs, const Jacobian rhs) {
    return jacobian_eq(lhs, rhs);
}
