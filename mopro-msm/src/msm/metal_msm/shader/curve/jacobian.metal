// source: https://github.com/geometryxyz/msl-secp256k1
// algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
#pragma once

using namespace metal;
#include "../mont_backend/mont.metal"
#include "./utils.metal"
#include <metal_math>
#include <metal_stdlib>

inline Jacobian jacobian_dbl_2009_l(Jacobian pt)
{
    BigInt x = pt.x;
    BigInt y = pt.y;
    BigInt z = pt.z;

    BigInt a = mont_mul(x, x);
    BigInt b = mont_mul(y, y);
    BigInt c = mont_mul(b, b);
    BigInt x1b = ff_add(x, b);
    BigInt x1b2 = mont_mul(x1b, x1b);
    BigInt ac = ff_add(a, c);
    BigInt x1b2ac = ff_sub(x1b2, ac);
    BigInt d = ff_add(x1b2ac, x1b2ac);
    BigInt a2 = ff_add(a, a);
    BigInt e = ff_add(a2, a);
    BigInt f = mont_mul(e, e);
    BigInt d2 = ff_add(d, d);
    BigInt x3 = ff_sub(f, d2);
    BigInt c2 = ff_add(c, c);
    BigInt c4 = ff_add(c2, c2);
    BigInt c8 = ff_add(c4, c4);
    BigInt dx3 = ff_sub(d, x3);
    BigInt edx3 = mont_mul(e, dx3);
    BigInt y3 = ff_sub(edx3, c8);
    BigInt y1z1 = mont_mul(y, z);
    BigInt z3 = ff_add(y1z1, y1z1);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

inline Jacobian jacobian_add_2007_bl(Jacobian a, Jacobian b)
{
    if (is_jacobian_zero(a))
        return b;
    if (is_jacobian_zero(b))
        return a;
    if (a == b)
        return jacobian_dbl_2009_l(a);

    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;
    BigInt z2 = b.z;

    // First compute z coordinates
    BigInt z1z1 = mont_mul(z1, z1);
    BigInt z2z2 = mont_mul(z2, z2);
    BigInt u1 = mont_mul(x1, z2z2);
    BigInt u2 = mont_mul(x2, z1z1);
    BigInt y1z2 = mont_mul(y1, z2);
    BigInt s1 = mont_mul(y1z2, z2z2);

    BigInt y2z1 = mont_mul(y2, z1);
    BigInt s2 = mont_mul(y2z1, z1z1);
    BigInt h = ff_sub(u2, u1);
    BigInt h2 = ff_add(h, h);
    BigInt i = mont_mul(h2, h2);
    BigInt j = mont_mul(h, i);

    BigInt s2s1 = ff_sub(s2, s1);
    BigInt r = ff_add(s2s1, s2s1);
    BigInt v = mont_mul(u1, i);
    BigInt v2 = ff_add(v, v);
    BigInt r2 = mont_mul(r, r);
    BigInt jv2 = ff_add(j, v2);
    BigInt x3 = ff_sub(r2, jv2);

    BigInt vx3 = ff_sub(v, x3);
    BigInt rvx3 = mont_mul(r, vx3);
    BigInt s12 = ff_add(s1, s1);
    BigInt s12j = mont_mul(s12, j);
    BigInt y3 = ff_sub(rvx3, s12j);

    BigInt z1z2 = mont_mul(z1, z2);
    BigInt z1z2h = mont_mul(z1z2, h);
    BigInt z3 = ff_add(z1z2h, z1z2h);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// Notice that this algo only takes standard form instead of Montgomery form
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
inline Jacobian jacobian_madd_2007_bl(Jacobian a, Affine b)
{
    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;

    // Z1Z1 = Z1^2
    BigInt z1z1 = mont_mul(z1, z1);

    // U2 = X2*Z1Z1
    BigInt u2 = mont_mul(x2, z1z1);

    // S2 = Y2*Z1*Z1Z1
    BigInt temp_s2 = mont_mul(y2, z1);
    BigInt s2 = mont_mul(temp_s2, z1z1);

    // H = U2-X1
    BigInt h = ff_sub(u2, x1);

    // HH = H^2
    BigInt hh = mont_mul(h, h);

    // I = 4*HH
    BigInt i = ff_add(hh, hh); // *2
    i = ff_add(i, i); // *4

    // J = H*I
    BigInt j = mont_mul(h, i);

    // r = 2*(S2-Y1)
    BigInt s2_minus_y1 = ff_sub(s2, y1);
    BigInt r = ff_add(s2_minus_y1, s2_minus_y1);

    // V = X1*I
    BigInt v = mont_mul(x1, i);

    // X3 = r^2-J-2*V
    BigInt r2 = mont_mul(r, r);
    BigInt v2 = ff_add(v, v);
    BigInt jv2 = ff_add(j, v2);
    BigInt x3 = ff_sub(r2, jv2);

    // Y3 = r*(V-X3)-2*Y1*J
    BigInt v_minus_x3 = ff_sub(v, x3);
    BigInt r_vmx3 = mont_mul(r, v_minus_x3);
    BigInt y1j = mont_mul(y1, j);
    BigInt y1j2 = ff_add(y1j, y1j);
    BigInt y3 = ff_sub(r_vmx3, y1j2);

    // Z3 = (Z1+H)^2-Z1Z1-HH
    BigInt z1_plus_h = ff_add(z1, h);
    BigInt z1_plus_h_squared = mont_mul(z1_plus_h, z1_plus_h);
    BigInt temp = ff_sub(z1_plus_h_squared, z1z1);
    BigInt z3 = ff_sub(temp, hh);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

/// Mixed addition (madd-2007-bl) with full edge-case handling: `b` is an
/// affine point (implicit z = 1, i.e. Montgomery R). Handles a == infinity,
/// a == b (doubling) and a == -b (infinity). ~11 field muls vs 16 for the
/// general Jacobian addition.
inline Jacobian jacobian_madd_safe(Jacobian a, Affine b)
{
    if (is_jacobian_zero(a)) {
        Jacobian r;
        r.x = b.x;
        r.y = b.y;
        r.z = get_bn254_one_mont().z;
        return r;
    }

    // Z1Z1 = Z1^2, U2 = X2*Z1Z1, S2 = Y2*Z1*Z1Z1
    BigInt z1z1 = mont_mul(a.z, a.z);
    BigInt u2 = mont_mul(b.x, z1z1);
    BigInt z1z1z1 = mont_mul(a.z, z1z1);
    BigInt s2 = mont_mul(b.y, z1z1z1);

    if (bigint_eq(u2, a.x)) {
        if (bigint_eq(s2, a.y)) {
            // Same point: fall back to doubling
            return jacobian_dbl_2009_l(a);
        }
        // Opposite points: result is the point at infinity
        return get_bn254_zero_mont();
    }

    // H = U2-X1, HH = H^2, I = 4*HH, J = H*I
    BigInt h = ff_sub(u2, a.x);
    BigInt hh = mont_mul(h, h);
    BigInt i = ff_add(hh, hh);
    i = ff_add(i, i);
    BigInt j = mont_mul(h, i);

    // r = 2*(S2-Y1), V = X1*I
    BigInt s2y1 = ff_sub(s2, a.y);
    BigInt r = ff_add(s2y1, s2y1);
    BigInt v = mont_mul(a.x, i);

    // X3 = r^2-J-2*V
    BigInt r2 = mont_mul(r, r);
    BigInt v2 = ff_add(v, v);
    BigInt jv2 = ff_add(j, v2);
    BigInt x3 = ff_sub(r2, jv2);

    // Y3 = r*(V-X3)-2*Y1*J
    BigInt vx3 = ff_sub(v, x3);
    BigInt rvx3 = mont_mul(r, vx3);
    BigInt y1j = mont_mul(a.y, j);
    BigInt y1j2 = ff_add(y1j, y1j);
    BigInt y3 = ff_sub(rvx3, y1j2);

    // Z3 = (Z1+H)^2-Z1Z1-HH
    BigInt z1h = ff_add(a.z, h);
    BigInt z1h2 = mont_mul(z1h, z1h);
    BigInt z3 = ff_sub(ff_sub(z1h2, z1z1), hh);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

inline Jacobian jacobian_scalar_mul(
    Jacobian pt,
    uint scalar)
{
    // Handle special cases first
    if (scalar == 0 || is_bigint_zero(pt.z)) {
        return get_bn254_zero_mont();
    }
    if (scalar == 1) {
        return pt;
    }

    Jacobian result = get_bn254_zero_mont();
    Jacobian temp = pt;
    uint s = scalar;

    while (s > 0) {
        if (s & 1) {
            result = jacobian_add_2007_bl(result, temp);
        }
        temp = jacobian_dbl_2009_l(temp);
        s = s >> 1;
    }

    return result;
}

inline Jacobian jacobian_neg(Jacobian pt)
{
    if (is_jacobian_zero(pt)) {
        return pt;
    }

    // Negate Y (mod p): newY = p - Y
    BigInt p = MODULUS;
    BigInt negY = ff_sub(p, pt.y);

    Jacobian result;
    result.x = pt.x;
    result.y = negY;
    result.z = pt.z;
    return result;
}

// Override operators in Jacobian
constexpr Jacobian operator+(const Jacobian lhs, const Jacobian rhs)
{
    return jacobian_add_2007_bl(lhs, rhs);
}

constexpr Jacobian operator+(const Jacobian lhs, const Affine rhs)
{
    return jacobian_madd_2007_bl(lhs, rhs);
}

constexpr Jacobian operator-(const Jacobian pt)
{
    return jacobian_neg(pt);
}