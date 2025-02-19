
#include    "types.h"
// #include "../bigint/bigint.h"
// struct BigInt {
//     array<uint, NUM_LIMBS> limbs;
// };

// struct BigIntWide {
//     array<uint, NUM_LIMBS_WIDE> limbs;
// };

// struct Jacobian {
//     BigInt x;
//     BigInt y;
//     BigInt z;
// };

Jacobian jacobian_zero() {
    Jacobian zero;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = 0;
        zero.y.limbs[i] = 0;
        zero.z.limbs[i] = 0;
    }
    return zero;
}

// struct Affine {
//     BigInt x;
//     BigInt y;
// };


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

bool is_jacobian_zero(Jacobian p) {
    return (is_bigint_zero(p.z));
}

constexpr bool operator==(const Jacobian lhs, const Jacobian rhs) {
    return jacobian_eq(lhs, rhs);
}


BigInt ff_add(
    BigInt a,
    BigInt b,
    BigInt p
) {
    BigInt sum = a + b;

    BigInt res;
    if (sum >= p) {
        // s = a + b - p
        BigInt s = sum - p;
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = s.limbs[i];
        }
    }
    else {
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = sum.limbs[i];
        }
    }
    return res;
}

BigInt ff_sub(
    BigInt a,
    BigInt b,
    BigInt p
) {
    // if a >= b
    if (a >= b) {
        // a - b
        BigInt res = a - b;
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    } else {
        // p - (b - a)
        BigInt r = b - a;
        BigInt res = p - r;
        for (uint i = 0; i < NUM_LIMBS; i ++) {
            res.limbs[i] = res.limbs[i];
        }
        return res;
    }
}


BigInt bigint_zero() {
    BigInt s;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        s.limbs[i] = 0;
    }
    return s;
}

BigInt bigint_add_unsafe(
    BigInt lhs,
    BigInt rhs
) {
    BigInt result;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    uint carry = 0;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & mask;
        carry = c >> LOG_LIMB_SIZE;
    }
    return result;
}

BigIntWide bigint_add_wide(
    BigInt lhs,
    BigInt rhs
) {
    BigIntWide result;
    uint mask = (1 << LOG_LIMB_SIZE) - 1;
    uint carry = 0;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & mask;
        carry = c >> LOG_LIMB_SIZE;
    }
    result.limbs[NUM_LIMBS] = carry;

    return result;
}

BigInt bigint_sub(
    BigInt lhs,
    BigInt rhs
) {
    uint borrow = 0;

    BigInt res;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;

        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    return res;
}


BigIntWide bigint_sub_wide(
    BigIntWide lhs,
    BigIntWide rhs
) {
    uint borrow = 0;

    BigIntWide res;

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;

        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    return res;
}

bool bigint_gte(
    BigInt lhs,
    BigInt rhs
) {
    for (uint idx = 0; idx < NUM_LIMBS; idx ++) {
        uint i = NUM_LIMBS - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) {
            return false;
        } else if (lhs.limbs[i] > rhs.limbs[i]) {
            return true;
        }
    }

    return true;
}

bool bigint_wide_gte(
    BigIntWide lhs,
    BigIntWide rhs
) {
    for (uint idx = 0; idx < NUM_LIMBS_WIDE; idx ++) {
        uint i = NUM_LIMBS_WIDE - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) {
            return false;
        } else if (lhs.limbs[i] > rhs.limbs[i]) {
            return true;
        }
    }

    return true;
}

bool bigint_eq(
    BigInt lhs,
    BigInt rhs
) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (lhs.limbs[i] != rhs.limbs[i]) {
            return false;
        }
    }
    return true;
}

bool is_bigint_zero(BigInt x) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (x.limbs[i] != 0) {
            return false;
        }
    }
    return true;
}

// Overload Operators
constexpr BigInt operator+(const BigInt lhs, const BigInt rhs) {
    return bigint_add_unsafe(lhs, rhs);
}

constexpr BigInt operator-(const BigInt lhs, const BigInt rhs) {
    return bigint_sub(lhs, rhs);
}

constexpr bool operator>=(const BigInt lhs, const BigInt rhs) {
    return bigint_gte(lhs, rhs);
}

constexpr bool operator==(const BigInt lhs, const BigInt rhs) {
    return bigint_eq(lhs, rhs);
}


BigInt get_mu() {
    BigInt mu;
    uint n0 = N0;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        mu.limbs[i] = n0 & MASK;
        n0 >>= LOG_LIMB_SIZE;
    }

    return mu;
}

BigIntWide get_r() {
    BigIntWide r;   // 257 bits
    for (uint i = 0; i < NUM_LIMBS_WIDE; i++) {
        r.limbs[i] = MONT_RADIX[i];
    }
    return r;
}

BigInt get_p() {
    BigInt p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

BigIntWide get_p_wide() {
    BigIntWide p;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.limbs[i] = BN254_BASEFIELD_MODULUS[i];
    }
    return p;
}

Jacobian get_bn254_zero() {
    Jacobian zero;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_X[i];
        zero.y.limbs[i] = BN254_ZERO_Y[i];
        zero.z.limbs[i] = BN254_ZERO_Z[i];
    }
    return zero;
}

Jacobian get_bn254_one() {
    Jacobian one;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_X[i];
        one.y.limbs[i] = BN254_ONE_Y[i];
        one.z.limbs[i] = BN254_ONE_Z[i];
    }
    return one;
}

Jacobian get_bn254_zero_mont() {
    Jacobian zero;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        zero.x.limbs[i] = BN254_ZERO_XR[i];
        zero.y.limbs[i] = BN254_ZERO_YR[i];
        zero.z.limbs[i] = BN254_ZERO_ZR[i];
    }
    return zero;
}

Jacobian get_bn254_one_mont() {
    Jacobian one;
    for (uint i = 0; i < NUM_LIMBS; i++) {
        one.x.limbs[i] = BN254_ONE_XR[i];
        one.y.limbs[i] = BN254_ONE_YR[i];
        one.z.limbs[i] = BN254_ONE_ZR[i];
    }
    return one;
}

BigInt conditional_reduce(
    BigInt x,
    BigInt y
) {
    if (x >= y) {
        return x - y;
    }

    return x;
}

/// An optimised variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 12 and 13-bit limbs.
BigInt mont_mul_optimised(
    BigInt x,
    BigInt y,
    BigInt p
) {
    BigInt s = bigint_zero();

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;
        s.limbs[0] = s.limbs[1] + x.limbs[i] * y.limbs[1] + qi * p.limbs[1] + c;

        for (uint j = 2; j < NUM_LIMBS; j ++) {
            s.limbs[j - 1] = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
        }
        s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = s.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        s.limbs[i] = v & MASK;
    }

    return conditional_reduce(s, p);
}

/// An modified variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
/// Known to work with 14 and 15-bit limbs.
BigInt mont_mul_modified(
    BigInt x,
    BigInt y,
    BigInt p
) {
    BigInt s = bigint_zero();

    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
        uint tprime = t & MASK;
        uint qi = (N0 * tprime) & MASK;
        uint c = (t + qi * p.limbs[0]) >> LOG_LIMB_SIZE;

        for (uint j = 1; j < NUM_LIMBS - 1; j ++) {
            uint t = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
            if ((j - 1) % NSAFE == 0) {
                t = t + c;
            }

            c = t >> LOG_LIMB_SIZE;

            if (j % NSAFE == 0) {
                c = t >> LOG_LIMB_SIZE;
                s.limbs[j - 1] = t & MASK;
            } else {
                s.limbs[j - 1] = t;
            }
        }
        s.limbs[NUM_LIMBS - 2] = x.limbs[i] * y.limbs[NUM_LIMBS - 1] + qi * p.limbs[NUM_LIMBS - 1];
    }

    uint c = 0;
    for (uint i = 0; i < NUM_LIMBS; i ++) {
        uint v = s.limbs[i] + c;
        c = v >> LOG_LIMB_SIZE;
        s.limbs[i] = v & MASK;
    }

    return conditional_reduce(s, p);
}

/// The CIOS method for Montgomery multiplication from Tolga Acar's thesis:
/// High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems
/// https://www.proquest.com/openview/1018972f191afe55443658b28041c118/1
BigInt mont_mul_cios(
    BigInt x,
    BigInt y,
    BigInt p
) {
    uint t[NUM_LIMBS + 2] = {0};  // Extra space for carries
    BigInt result;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        // Step 1: Multiply and add
        uint c = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            uint r = t[j] + x.limbs[j] * y.limbs[i] + c;
            c = r >> LOG_LIMB_SIZE;
            t[j] = r & MASK;
        }
        uint r = t[NUM_LIMBS] + c;
        t[NUM_LIMBS + 1] = r >> LOG_LIMB_SIZE;
        t[NUM_LIMBS] = r & MASK;

        // Step 2: Reduce
        uint m = (t[0] * N0) & MASK;
        r = t[0] + m * p.limbs[0];
        c = r >> LOG_LIMB_SIZE;

        for (uint j = 1; j < NUM_LIMBS; j++) {
            r = t[j] + m * p.limbs[j] + c;
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
        if (t[i] < p.limbs[i]) {
            t_lt_p = true;
            break;
        } else if (t[i] > p.limbs[i]) {
            break;
        }
    }

    if (t_lt_p) {
        for (uint i = 0; i < NUM_LIMBS; i++) {
            result.limbs[i] = t[i];
        }
    } else {
        uint borrow = 0;
        for (uint i = 0; i < NUM_LIMBS; i++) {
            uint diff = t[i] - p.limbs[i] - borrow;
            if (t[i] < (p.limbs[i] + borrow)) {
                diff += (1 << LOG_LIMB_SIZE);
                borrow = 1;
            } else {
                borrow = 0;
            }
            result.limbs[i] = diff;
        }
    }

    return result;
}


Jacobian jacobian_dbl_2009_l(
    Jacobian pt,
    BigInt p
) {
    BigInt x = pt.x;
    BigInt y = pt.y;
    BigInt z = pt.z;

    BigInt a = mont_mul_cios(x, x, p);
    BigInt b = mont_mul_cios(y, y, p);
    BigInt c = mont_mul_cios(b, b, p);
    BigInt x1b = ff_add(x, b, p);
    BigInt x1b2 = mont_mul_cios(x1b, x1b, p);
    BigInt ac = ff_add(a, c, p);
    BigInt x1b2ac = ff_sub(x1b2, ac, p);
    BigInt d = ff_add(x1b2ac, x1b2ac, p);
    BigInt a2 = ff_add(a, a, p);
    BigInt e = ff_add(a2, a, p);
    BigInt f = mont_mul_cios(e, e, p);
    BigInt d2 = ff_add(d, d, p);
    BigInt x3 = ff_sub(f, d2, p);
    BigInt c2 = ff_add(c, c, p);
    BigInt c4 = ff_add(c2, c2, p);
    BigInt c8 = ff_add(c4, c4, p);
    BigInt dx3 = ff_sub(d, x3, p);
    BigInt edx3 = mont_mul_cios(e, dx3, p);
    BigInt y3 = ff_sub(edx3, c8, p);
    BigInt y1z1 = mont_mul_cios(y, z, p);
    BigInt z3 = ff_add(y1z1, y1z1, p);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

Jacobian jacobian_add_2007_bl(
    Jacobian a,
    Jacobian b,
    BigInt p
) {
    if (is_jacobian_zero(a)) {
        return b;
    }
    if (is_jacobian_zero(b)) {
        return a;
    }
    if (a == b) return jacobian_dbl_2009_l(a, p);

    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;
    BigInt z2 = b.z;

    // First compute z coordinates
    BigInt z1z1 = mont_mul_cios(z1, z1, p);
    BigInt z2z2 = mont_mul_cios(z2, z2, p);
    BigInt u1 = mont_mul_cios(x1, z2z2, p);
    BigInt u2 = mont_mul_cios(x2, z1z1, p);
    BigInt y1z2 = mont_mul_cios(y1, z2, p);
    BigInt s1 = mont_mul_cios(y1z2, z2z2, p);

    BigInt y2z1 = mont_mul_cios(y2, z1, p);
    BigInt s2 = mont_mul_cios(y2z1, z1z1, p);
    BigInt h = ff_sub(u2, u1, p);
    BigInt h2 = ff_add(h, h, p);
    BigInt i = mont_mul_cios(h2, h2, p);
    BigInt j = mont_mul_cios(h, i, p);

    BigInt s2s1 = ff_sub(s2, s1, p);
    BigInt r = ff_add(s2s1, s2s1, p);
    BigInt v = mont_mul_cios(u1, i, p);
    BigInt v2 = ff_add(v, v, p);
    BigInt r2 = mont_mul_cios(r, r, p);
    BigInt jv2 = ff_add(j, v2, p);
    BigInt x3 = ff_sub(r2, jv2, p);

    BigInt vx3 = ff_sub(v, x3, p);
    BigInt rvx3 = mont_mul_cios(r, vx3, p);
    BigInt s12 = ff_add(s1, s1, p);
    BigInt s12j = mont_mul_cios(s12, j, p);
    BigInt y3 = ff_sub(rvx3, s12j, p);

    BigInt z1z2 = mont_mul_cios(z1, z2, p);
    BigInt z1z2h = mont_mul_cios(z1z2, h, p);
    BigInt z3 = ff_add(z1z2h, z1z2h, p);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

//http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
Jacobian jacobian_madd_2007_bl(
    Jacobian a,
    Affine b,
    BigInt p
) {
    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;

    // Z1Z1 = Z1^2
    BigInt z1z1 = mont_mul_cios(z1, z1, p);

    // U2 = X2*Z1Z1
    BigInt u2 = mont_mul_cios(x2, z1z1, p);

    // S2 = Y2*Z1*Z1Z1
    BigInt temp_s2 = mont_mul_cios(y2, z1, p);
    BigInt s2 = mont_mul_cios(temp_s2, z1z1, p);

    // H = U2-X1
    BigInt h = ff_sub(u2, x1, p);

    // HH = H^2
    BigInt hh = mont_mul_cios(h, h, p);

    // I = 4*HH
    BigInt i = ff_add(hh, hh, p); // *2
    i = ff_add(i, i, p);          // *4

    // J = H*I
    BigInt j = mont_mul_cios(h, i, p);

    // r = 2*(S2-Y1)
    BigInt s2_minus_y1 = ff_sub(s2, y1, p);
    BigInt r = ff_add(s2_minus_y1, s2_minus_y1, p);

    // V = X1*I
    BigInt v = mont_mul_cios(x1, i, p);

    // X3 = r^2-J-2*V
    BigInt r2 = mont_mul_cios(r, r, p);
    BigInt v2 = ff_add(v, v, p);
    BigInt jv2 = ff_add(j, v2, p);
    BigInt x3 = ff_sub(r2, jv2, p);

    // Y3 = r*(V-X3)-2*Y1*J
    BigInt v_minus_x3 = ff_sub(v, x3, p);
    BigInt r_vmx3 = mont_mul_cios(r, v_minus_x3, p);
    BigInt y1j = mont_mul_cios(y1, j, p);
    BigInt y1j2 = ff_add(y1j, y1j, p);
    BigInt y3 = ff_sub(r_vmx3, y1j2, p);

    // Z3 = (Z1+H)^2-Z1Z1-HH
    BigInt z1_plus_h = ff_add(z1, h, p);
    BigInt z1_plus_h_squared = mont_mul_cios(z1_plus_h, z1_plus_h, p);
    BigInt temp = ff_sub(z1_plus_h_squared, z1z1, p);
    BigInt z3 = ff_sub(temp, hh, p);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

Jacobian jacobian_scalar_mul(
    Jacobian point,
    uint scalar
) {
    // Handle special cases first
    if (scalar == 0 || is_bigint_zero(point.z)) {
        return get_bn254_zero_mont();
    }
    if (scalar == 1) {
        return point;
    }

    BigInt p = get_p();
    Jacobian result = get_bn254_zero_mont();
    Jacobian temp = point;
    uint s = scalar;

    while (s > 0) {
        if (s & 1) {
            result = jacobian_add_2007_bl(result, temp, p);
        }
        temp = jacobian_dbl_2009_l(temp, p);
        s = s >> 1;
    }

    return result;
}

// Override operators in Jacobian
Jacobian operator+(Jacobian a, Jacobian b) {
    BigInt p = get_p();
    return jacobian_add_2007_bl(a, b, p);
}
