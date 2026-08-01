// Extended Jacobian ("XYZZ") coordinates for BN254 short Weierstrass (a = 0):
// (X, Y, ZZ, ZZZ) with ZZ = z^2, ZZZ = z^3 represents affine (X/ZZ, Y/ZZZ).
// The identity is encoded as ZZ = 0, which matches zero-initialized buffers.
//
// Compared to Jacobian, mixed addition costs 8M+2S (vs 7M+4S) with ~5 field
// add/subs (vs ~13), and general addition 12M+2S (vs 11M+5S with ~11
// add/subs) -- fewer total field ops in the pipeline's hottest loops.
// Formulas: https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
#pragma once

using namespace metal;
#include "../mont_backend/mont.metal"
#include "./jacobian.metal"

struct XYZZ {
    BigInt x;
    BigInt y;
    BigInt zz;
    BigInt zzz;
};

inline bool is_xyzz_zero(XYZZ a)
{
    return is_bigint_zero(a.zz);
}

inline XYZZ get_xyzz_zero()
{
    XYZZ r;
    r.x = bigint_zero();
    r.y = bigint_zero();
    r.zz = bigint_zero();
    r.zzz = bigint_zero();
    return r;
}

/// Lift an affine point (Montgomery form) into XYZZ: z = 1 so ZZ = ZZZ = R.
inline XYZZ xyzz_from_affine(Affine b)
{
    XYZZ r;
    r.x = b.x;
    r.y = b.y;
    r.zz = get_bn254_one_mont().z;
    r.zzz = r.zz;
    return r;
}

/// Doubling (dbl-2008-s-1, a = 0): 6M + 3S.
inline XYZZ xyzz_dbl(XYZZ a)
{
    if (is_xyzz_zero(a)) {
        return a;
    }

    BigInt u = ff_add(a.y, a.y); // U = 2*Y1
    BigInt v = mont_mul(u, u); // V = U^2
    BigInt w = mont_mul(u, v); // W = U*V
    BigInt s = mont_mul(a.x, v); // S = X1*V
    BigInt xx = mont_mul(a.x, a.x);
    BigInt m = ff_add(ff_add(xx, xx), xx); // M = 3*X1^2 (a = 0)
    BigInt m2 = mont_mul(m, m);
    BigInt s2 = ff_add(s, s);
    XYZZ r;
    r.x = ff_sub(m2, s2); // X3 = M^2 - 2*S
    BigInt sx3 = ff_sub(s, r.x);
    BigInt msx3 = mont_mul(m, sx3);
    BigInt wy1 = mont_mul(w, a.y);
    r.y = ff_sub(msx3, wy1); // Y3 = M*(S-X3) - W*Y1
    r.zz = mont_mul(v, a.zz); // ZZ3 = V*ZZ1
    r.zzz = mont_mul(w, a.zzz); // ZZZ3 = W*ZZZ1
    return r;
}

/// Mixed addition (madd-2008-s): `b` is affine, z = 1. 8M + 2S.
/// Handles a == identity, a == b (doubling) and a == -b (identity).
inline XYZZ xyzz_madd_safe(XYZZ a, Affine b)
{
    if (is_xyzz_zero(a)) {
        return xyzz_from_affine(b);
    }

    BigInt u2 = mont_mul(b.x, a.zz); // U2 = X2*ZZ1
    BigInt s2 = mont_mul(b.y, a.zzz); // S2 = Y2*ZZZ1

    if (bigint_eq(u2, a.x)) {
        if (bigint_eq(s2, a.y)) {
            return xyzz_dbl(a);
        }
        return get_xyzz_zero();
    }

    BigInt p = ff_sub(u2, a.x); // P = U2-X1
    BigInt r = ff_sub(s2, a.y); // R = S2-Y1
    BigInt pp = mont_mul(p, p); // PP = P^2
    BigInt ppp = mont_mul(p, pp); // PPP = P*PP
    BigInt q = mont_mul(a.x, pp); // Q = X1*PP
    BigInt r2 = mont_mul(r, r);
    BigInt q2 = ff_add(q, q);
    XYZZ out;
    out.x = ff_sub(ff_sub(r2, ppp), q2); // X3 = R^2 - PPP - 2*Q
    BigInt qx3 = ff_sub(q, out.x);
    BigInt rqx3 = mont_mul(r, qx3);
    BigInt y1ppp = mont_mul(a.y, ppp);
    out.y = ff_sub(rqx3, y1ppp); // Y3 = R*(Q-X3) - Y1*PPP
    out.zz = mont_mul(a.zz, pp); // ZZ3 = ZZ1*PP
    out.zzz = mont_mul(a.zzz, ppp); // ZZZ3 = ZZZ1*PPP
    return out;
}

/// General addition (add-2008-s): 12M + 2S.
/// Handles either operand == identity, a == b (doubling) and a == -b.
inline XYZZ xyzz_add_safe(XYZZ a, XYZZ b)
{
    if (is_xyzz_zero(a)) {
        return b;
    }
    if (is_xyzz_zero(b)) {
        return a;
    }

    BigInt u1 = mont_mul(a.x, b.zz); // U1 = X1*ZZ2
    BigInt u2 = mont_mul(b.x, a.zz); // U2 = X2*ZZ1
    BigInt s1 = mont_mul(a.y, b.zzz); // S1 = Y1*ZZZ2
    BigInt s2 = mont_mul(b.y, a.zzz); // S2 = Y2*ZZZ1

    if (bigint_eq(u1, u2)) {
        if (bigint_eq(s1, s2)) {
            return xyzz_dbl(a);
        }
        return get_xyzz_zero();
    }

    BigInt p = ff_sub(u2, u1); // P = U2-U1
    BigInt r = ff_sub(s2, s1); // R = S2-S1
    BigInt pp = mont_mul(p, p);
    BigInt ppp = mont_mul(p, pp);
    BigInt q = mont_mul(u1, pp); // Q = U1*PP
    BigInt r2 = mont_mul(r, r);
    BigInt q2 = ff_add(q, q);
    XYZZ out;
    out.x = ff_sub(ff_sub(r2, ppp), q2); // X3 = R^2 - PPP - 2*Q
    BigInt qx3 = ff_sub(q, out.x);
    BigInt rqx3 = mont_mul(r, qx3);
    BigInt s1ppp = mont_mul(s1, ppp);
    out.y = ff_sub(rqx3, s1ppp); // Y3 = R*(Q-X3) - S1*PPP
    BigInt zz1zz2 = mont_mul(a.zz, b.zz);
    out.zz = mont_mul(zz1zz2, pp); // ZZ3 = ZZ1*ZZ2*PP
    BigInt zzz1zzz2 = mont_mul(a.zzz, b.zzz);
    out.zzz = mont_mul(zzz1zzz2, ppp); // ZZZ3 = ZZZ1*ZZZ2*PPP
    return out;
}

/// Negation: -(X, Y, ZZ, ZZZ) = (X, p - Y, ZZ, ZZZ).
inline XYZZ xyzz_neg(XYZZ a)
{
    if (is_xyzz_zero(a)) {
        return a;
    }
    XYZZ r = a;
    r.y = ff_sub(MODULUS, a.y);
    return r;
}
