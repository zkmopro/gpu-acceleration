// source: https://github.com/geometryxyz/msl-secp256k1
// algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../mont_backend/mont.metal"
#include "./utils.metal"

Jacobian jacobian_dbl_2009_l(Jacobian pt) {
    FieldElement x = pt.x;
    FieldElement y = pt.y;
    FieldElement z = pt.z;

    FieldElement a = x * x;
    FieldElement b = y * y;
    FieldElement c = b * b;
    FieldElement x1b = x + b;
    FieldElement x1b2 = x1b * x1b;
    FieldElement ac = a + c;
    FieldElement x1b2ac = x1b2 - ac;
    FieldElement d = x1b2ac + x1b2ac;
    FieldElement a2 = a + a;
    FieldElement e = a2 + a;
    FieldElement f = e * e;
    FieldElement d2 = d + d;
    FieldElement x3 = f - d2;
    FieldElement c2 = c + c;
    FieldElement c4 = c2 + c2;
    FieldElement c8 = c4 + c4;
    FieldElement dx3 = d - x3;
    FieldElement edx3 = e * dx3;
    FieldElement y3 = edx3 - c8;
    FieldElement y1z1 = y * z;
    FieldElement z3 = y1z1 + y1z1;

    return Jacobian{ .x = x3, .y = y3, .z = z3 };
}

Jacobian jacobian_add_2007_bl(Jacobian a, Jacobian b) {
    if (is_jacobian_zero(a)) return b;
    if (is_jacobian_zero(b)) return a;
    if (a == b) return jacobian_dbl_2009_l(a);

    FieldElement x1 = a.x;
    FieldElement y1 = a.y;
    FieldElement z1 = a.z;
    FieldElement x2 = b.x;
    FieldElement y2 = b.y;
    FieldElement z2 = b.z;

    // First compute z coordinates
    FieldElement z1z1 = z1 * z1;
    FieldElement z2z2 = z2 * z2;
    FieldElement u1 = x1 * z2z2;
    FieldElement u2 = x2 * z1z1;
    FieldElement y1z2 = y1 * z2;
    FieldElement s1 = y1z2 * z2z2;

    FieldElement y2z1 = y2 * z1;
    FieldElement s2 = y2z1 * z1z1;
    FieldElement h = u2 - u1;
    FieldElement h2 = h + h;
    FieldElement i = h2 * h2;
    FieldElement j = h * i;

    FieldElement s2s1 = s2 - s1;
    FieldElement r = s2s1 + s2s1;
    FieldElement v = u1 * i;
    FieldElement v2 = v + v;
    FieldElement r2 = r * r;
    FieldElement jv2 = j + v2;
    FieldElement x3 = r2 - jv2;

    FieldElement vx3 = v - x3;
    FieldElement rvx3 = r * vx3;
    FieldElement s12 = s1 + s1;
    FieldElement s12j = s12 * j;
    FieldElement y3 = rvx3 - s12j;

    FieldElement z1z2 = z1 * z2;
    FieldElement z1z2h = z1z2 * h;
    FieldElement z3 = z1z2h + z1z2h;

    return Jacobian{ .x = x3, .y = y3, .z = z3 };
}

// Notice that this algo only takes standard form instead of Montgomery form
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
Jacobian jacobian_madd_2007_bl(Jacobian a, Affine b) {
    FieldElement x1 = a.x;
    FieldElement y1 = a.y;
    FieldElement z1 = a.z;
    FieldElement x2 = b.x;
    FieldElement y2 = b.y;

    // Z1Z1 = Z1^2
    FieldElement z1z1 = z1 * z1;
    
    // U2 = X2*Z1Z1
    FieldElement u2 = x2 * z1z1;
    
    // S2 = Y2*Z1*Z1Z1
    FieldElement temp_s2 = y2 * z1;
    FieldElement s2 = temp_s2 * z1z1;
    
    // H = U2-X1
    FieldElement h = u2 - x1;
    
    // HH = H^2
    FieldElement hh = h * h;
    
    // I = 4*HH
    FieldElement i = hh + hh; // *2
    i = i + i;  // *4
    
    // J = H*I
    FieldElement j = h * i;
    
    // r = 2*(S2-Y1)
    FieldElement s2_minus_y1 = s2 - y1;
    FieldElement r = s2_minus_y1 + s2_minus_y1;
    
    // V = X1*I
    FieldElement v = x1 * i;
    
    // X3 = r^2-J-2*V
    FieldElement r2 = r * r;
    FieldElement v2 = v + v;
    FieldElement jv2 = j + v2;
    FieldElement x3 = r2 - jv2;
    
    // Y3 = r*(V-X3)-2*Y1*J
    FieldElement v_minus_x3 = v - x3;
    FieldElement r_vmx3 = r * v_minus_x3;
    FieldElement y1j = y1 * j;
    FieldElement y1j2 = y1j + y1j;
    FieldElement y3 = r_vmx3 - y1j2;
    
    // Z3 = (Z1+H)^2-Z1Z1-HH
    FieldElement z1_plus_h = z1 + h;
    FieldElement z1_plus_h_squared = z1_plus_h * z1_plus_h;
    FieldElement temp = z1_plus_h_squared - z1z1;
    FieldElement z3 = temp - hh;
    
    return Jacobian{ .x = x3, .y = y3, .z = z3 };
}

Jacobian jacobian_scalar_mul(Jacobian point, uint scalar) {
    // Handle special cases first
    if (scalar == 0 || is_bigint_zero(point.z.value)) return get_bn254_zero_mont();
    if (scalar == 1) return point;

    Jacobian result = get_bn254_zero_mont();
    Jacobian temp = point;
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

Jacobian jacobian_neg(Jacobian a) {
    if (is_jacobian_zero(a)) return a;

    // Negate Y (mod p): newY = p - Y
    FieldElement p = FieldElement{ .value = a.x.modulus, .modulus = a.x.modulus };  // TODO: refactor this
    FieldElement negY = p - a.y;

    return Jacobian{ .x = a.x, .y = negY, .z = a.z };
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
