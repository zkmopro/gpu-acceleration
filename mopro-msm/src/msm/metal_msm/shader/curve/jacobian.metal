// source: https://github.com/geometryxyz/msl-secp256k1
// algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../mont_backend/mont.metal"
#include "./utils.metal"

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
    if (is_jacobian_zero(a)) return b;
    if (is_jacobian_zero(b)) return a;
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

Jacobian jacobian_neg(
    Jacobian a,
    BigInt p
) {
    if (is_jacobian_zero(a)) { return a; }

    // Negate Y (mod p): newY = p - Y
    BigInt negY = ff_sub(p, a.y, p);

    Jacobian result;
    result.x = a.x;
    result.y = negY;
    result.z = a.z;
    return result;
}

// Override operators in Jacobian
Jacobian operator+(Jacobian a, Jacobian b) {
    BigInt p = get_p();
    return jacobian_add_2007_bl(a, b, p);
}
