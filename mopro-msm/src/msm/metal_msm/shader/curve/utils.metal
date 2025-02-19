// using namespace metal;

// #include "utils.h"

// #define NUM_LIMBS 16
// #define NUM_LIMBS_WIDE 17
// #define LOG_LIMB_SIZE 16
// #define TWO_POW_WORD_SIZE 65536
// #define MASK 65535
// #define N0 25481
// #define NSAFE 1

// bool jacobian_eq(
//     Jacobian lhs,
//     Jacobian rhs
// ) {
//     for (uint i = 0; i < NUM_LIMBS; i++) {
//         if (lhs.x.limbs[i] != rhs.x.limbs[i]) {
//             return false;
//         } else if (lhs.y.limbs[i] != rhs.y.limbs[i]) {
//             return false;
//         } else if (lhs.z.limbs[i] != rhs.z.limbs[i]) {
//             return false;
//         }
//     }
//     return true;
// }

// bool is_jacobian_zero(Jacobian p) {
//     return (is_bigint_zero(p.z));
// }

// constexpr bool operator==(const Jacobian lhs, const Jacobian rhs) {
//     return jacobian_eq(lhs, rhs);
// }
