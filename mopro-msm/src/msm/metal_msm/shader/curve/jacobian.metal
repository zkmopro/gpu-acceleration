// source: https://github.com/geometryxyz/msl-secp256k1
// algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../mont_backend/mont.metal"
#include "./utils.metal"

Jacobian jacobian_dbl_2009_l(Jacobian pt) {
    BigInt x = pt.x;
    BigInt y = pt.y;
    BigInt z = pt.z;

    BigInt a = mont_mul_cios(x, x);
    BigInt b = mont_mul_cios(y, y);
    BigInt c = mont_mul_cios(b, b);
    BigInt x1b = ff_add(x, b);
    BigInt x1b2 = mont_mul_cios(x1b, x1b);
    BigInt ac = ff_add(a, c);
    BigInt x1b2ac = ff_sub(x1b2, ac);
    BigInt d = ff_add(x1b2ac, x1b2ac);
    BigInt a2 = ff_add(a, a);
    BigInt e = ff_add(a2, a);
    BigInt f = mont_mul_cios(e, e);
    BigInt d2 = ff_add(d, d);
    BigInt x3 = ff_sub(f, d2);
    BigInt c2 = ff_add(c, c);
    BigInt c4 = ff_add(c2, c2);
    BigInt c8 = ff_add(c4, c4);
    BigInt dx3 = ff_sub(d, x3);
    BigInt edx3 = mont_mul_cios(e, dx3);
    BigInt y3 = ff_sub(edx3, c8);
    BigInt y1z1 = mont_mul_cios(y, z);
    BigInt z3 = ff_add(y1z1, y1z1);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

Jacobian jacobian_add_2007_bl(Jacobian a, Jacobian b) {
    if (is_jacobian_zero(a)) return b;
    if (is_jacobian_zero(b)) return a;
    if (a == b) return jacobian_dbl_2009_l(a);

    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;
    BigInt z2 = b.z;

    // First compute z coordinates
    BigInt z1z1 = mont_mul_cios(z1, z1);
    BigInt z2z2 = mont_mul_cios(z2, z2);
    BigInt u1 = mont_mul_cios(x1, z2z2);
    BigInt u2 = mont_mul_cios(x2, z1z1);
    BigInt y1z2 = mont_mul_cios(y1, z2);
    BigInt s1 = mont_mul_cios(y1z2, z2z2);

    BigInt y2z1 = mont_mul_cios(y2, z1);
    BigInt s2 = mont_mul_cios(y2z1, z1z1);
    BigInt h = ff_sub(u2, u1);
    BigInt h2 = ff_add(h, h);
    BigInt i = mont_mul_cios(h2, h2);
    BigInt j = mont_mul_cios(h, i);

    BigInt s2s1 = ff_sub(s2, s1);
    BigInt r = ff_add(s2s1, s2s1);
    BigInt v = mont_mul_cios(u1, i);
    BigInt v2 = ff_add(v, v);
    BigInt r2 = mont_mul_cios(r, r);
    BigInt jv2 = ff_add(j, v2);
    BigInt x3 = ff_sub(r2, jv2);

    BigInt vx3 = ff_sub(v, x3);
    BigInt rvx3 = mont_mul_cios(r, vx3);
    BigInt s12 = ff_add(s1, s1);
    BigInt s12j = mont_mul_cios(s12, j);
    BigInt y3 = ff_sub(rvx3, s12j);

    BigInt z1z2 = mont_mul_cios(z1, z2);
    BigInt z1z2h = mont_mul_cios(z1z2, h);
    BigInt z3 = ff_add(z1z2h, z1z2h);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// Notice that this algo only takes standard form instead of Montgomery form
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
Jacobian jacobian_madd_2007_bl(Jacobian a, Affine b) {
    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;

    // Z1Z1 = Z1^2
    BigInt z1z1 = mont_mul_cios(z1, z1);
    
    // U2 = X2*Z1Z1
    BigInt u2 = mont_mul_cios(x2, z1z1);
    
    // S2 = Y2*Z1*Z1Z1
    BigInt temp_s2 = mont_mul_cios(y2, z1);
    BigInt s2 = mont_mul_cios(temp_s2, z1z1);
    
    // H = U2-X1
    BigInt h = ff_sub(u2, x1);
    
    // HH = H^2
    BigInt hh = mont_mul_cios(h, h);
    
    // I = 4*HH
    BigInt i = ff_add(hh, hh); // *2
    i = ff_add(i, i);          // *4
    
    // J = H*I
    BigInt j = mont_mul_cios(h, i);
    
    // r = 2*(S2-Y1)
    BigInt s2_minus_y1 = ff_sub(s2, y1);
    BigInt r = ff_add(s2_minus_y1, s2_minus_y1);
    
    // V = X1*I
    BigInt v = mont_mul_cios(x1, i);
    
    // X3 = r^2-J-2*V
    BigInt r2 = mont_mul_cios(r, r);
    BigInt v2 = ff_add(v, v);
    BigInt jv2 = ff_add(j, v2);
    BigInt x3 = ff_sub(r2, jv2);
    
    // Y3 = r*(V-X3)-2*Y1*J
    BigInt v_minus_x3 = ff_sub(v, x3);
    BigInt r_vmx3 = mont_mul_cios(r, v_minus_x3);
    BigInt y1j = mont_mul_cios(y1, j);
    BigInt y1j2 = ff_add(y1j, y1j);
    BigInt y3 = ff_sub(r_vmx3, y1j2);
    
    // Z3 = (Z1+H)^2-Z1Z1-HH
    BigInt z1_plus_h = ff_add(z1, h);
    BigInt z1_plus_h_squared = mont_mul_cios(z1_plus_h, z1_plus_h);
    BigInt temp = ff_sub(z1_plus_h_squared, z1z1);
    BigInt z3 = ff_sub(temp, hh);
    
    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

Jacobian jacobian_scalar_mul(
    Jacobian pt,
    uint scalar
) {
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

Jacobian jacobian_neg(Jacobian pt) {
    if (is_jacobian_zero(pt)) { return pt; }

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
constexpr Jacobian operator+(const Jacobian lhs, const Jacobian rhs) {
    return jacobian_add_2007_bl(lhs, rhs);
}

constexpr Jacobian operator+(const Jacobian lhs, const Affine rhs) {
    return jacobian_madd_2007_bl(lhs, rhs);
}

constexpr Jacobian operator-(const Jacobian pt) {
    return jacobian_neg(pt);
}