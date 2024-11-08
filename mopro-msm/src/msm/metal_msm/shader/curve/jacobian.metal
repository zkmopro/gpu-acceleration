// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "../mont_backend/mont.metal"

struct Jacobian {
    BigInt x;
    BigInt y;
    BigInt z;
};

Jacobian jacobian_add_2007_bl_unsafe(
    Jacobian a,
    Jacobian b,
    BigInt p
) {
    BigInt x1 = a.x;
    BigInt y1 = a.y;
    BigInt z1 = a.z;
    BigInt x2 = b.x;
    BigInt y2 = b.y;
    BigInt z2 = b.z;

    BigInt z1z1 = mont_mul_optimised(z1, z1, p);
    BigInt z2z2 = mont_mul_optimised(z2, z2, p);
    BigInt u1 = mont_mul_optimised(x1, z2z2, p);
    BigInt u2 = mont_mul_optimised(x2, z1z1, p);
    BigInt y1z2 = mont_mul_optimised(y1, z2, p);
    BigInt s1 = mont_mul_optimised(y1z2, z2z2, p);

    BigInt y2z1 = mont_mul_optimised(y2, z1, p);
    BigInt s2 = mont_mul_optimised(y2z1, z1z1, p);
    BigInt h = ff_sub(u2, u1, p);
    BigInt h2 = ff_add(h, h, p);
    BigInt i = mont_mul_optimised(h2, h2, p);
    BigInt j = mont_mul_optimised(h, i, p);

    BigInt s2s1 = ff_sub(s2, s1, p);
    BigInt r = ff_add(s2s1, s2s1, p);
    BigInt v = mont_mul_optimised(u1, i, p);
    BigInt v2 = ff_add(v, v, p);
    BigInt r2 = mont_mul_optimised(r, r, p);
    BigInt jv2 = ff_add(j, v2, p);
    BigInt x3 = ff_sub(r2, jv2, p);

    BigInt vx3 = ff_sub(v, x3, p);
    BigInt rvx3 = mont_mul_optimised(r, vx3, p);
    BigInt s12 = ff_add(s1, s1, p);
    BigInt s12j = mont_mul_optimised(s12, j, p);
    BigInt y3 = ff_sub(rvx3, s12j, p);

    BigInt z1z2 = mont_mul_optimised(z1, z2, p);
    BigInt z1z2h = mont_mul_optimised(z1z2, h, p);
    BigInt z3 = ff_add(z1z2h, z1z2h, p);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

Jacobian jacobian_dbl_2009_l(
    Jacobian pt,
    BigInt p
) {
    BigInt x = pt.x;
    BigInt y = pt.y;
    BigInt z = pt.z;

    BigInt a = mont_mul_optimised(x, x, p);
    BigInt b = mont_mul_optimised(y, y, p);
    BigInt c = mont_mul_optimised(b, b, p);
    BigInt x1b = ff_add(x, b, p);
    BigInt x1b2 = mont_mul_optimised(x1b, x1b, p);
    BigInt ac = ff_add(a, c, p);
    BigInt x1b2ac = ff_sub(x1b2, ac, p);
    BigInt d = ff_add(x1b2ac, x1b2ac, p);
    BigInt a2 = ff_add(a, a, p);
    BigInt e = ff_add(a2, a, p);
    BigInt f = mont_mul_optimised(e, e, p);
    BigInt d2 = ff_add(d, d, p);
    BigInt x3 = ff_sub(f, d2, p);
    BigInt c2 = ff_add(c, c, p);
    BigInt c4 = ff_add(c2, c2, p);
    BigInt c8 = ff_add(c4, c4, p);
    BigInt dx3 = ff_sub(d, x3, p);
    BigInt edx3 = mont_mul_optimised(e, dx3, p);
    BigInt y3 = ff_sub(edx3, c8, p);
    BigInt y1z1 = mont_mul_optimised(y, z, p);
    BigInt z3 = ff_add(y1z1, y1z1, p);

    Jacobian result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}
