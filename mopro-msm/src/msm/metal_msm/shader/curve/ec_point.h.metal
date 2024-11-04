#pragma once

template<typename Fp, const uint64_t A_CURVE>
class ECPoint {
public:
    Fp x;
    Fp y;
    Fp t;
    Fp z;

    /*
    TODO:

    // r is the montgomery radix
    fn get_r() -> BigInt {
        var r: BigInt;
    {{{ r_limbs }}}
        return r;
    }

    fn get_paf() -> Point {
        var result: Point;
        let r = get_r();
        result.y = r;
        result.z = r;
        return result;
    }
    */
    constexpr ECPoint() : ECPoint(ECPoint::point_at_infinity()) {}
    constexpr ECPoint(Fp _x, Fp _y, Fp _t, Fp _z) : x(_x), y(_y), t(_t), z(_z) {}

    constexpr ECPoint operator+(const ECPoint other) const {

    }

    void operator+=(const ECPoint other) {
        *this = *this + other;
    }

    static ECPoint point_at_infinity() {
        return ECPoint(Fp(1), Fp(1), Fp(0)); // Updated to new neutral element (1, 1, 0)
    }

    ECPoint operate_with_self(uint64_t exponent) const {
        ECPoint result = point_at_infinity();
        ECPoint base = ECPoint(x, y, t, z);

        while (exponent > 0) {
            if ((exponent & 1) == 1) {
                result = result + base;
            }
            exponent = exponent >> 1;
            base = base + base;
        }

        return result;
    }

    constexpr ECPoint operator*(uint64_t exponent) const {
        return operate_with_self(exponent);
    }

    constexpr void operator*=(uint64_t exponent) {
        *this = operate_with_self(exponent);
    }

    constexpr ECPoint neg() const {
        return ECPoint(x, y.neg(), t, z);
    }

    constexpr bool is_neutral_element(const ECPoint a_point) const {
        return a_point.z == Fp(0); // Updated to check for (1, 1, 0)
    }

    constexpr ECPoint double_in_place() const {
        if (is_neutral_element(*this)) {
            return *this;
        }

        // Doubling formulas
        Fp a_fp = Fp(A_CURVE).to_montgomery();
        Fp two = Fp(2).to_montgomery();
        Fp three = Fp(3).to_montgomery();

        Fp eight = Fp(8).to_montgomery();

        Fp xx = x * x; // x^2
        Fp yy = y * y; // y^2
        Fp yyyy = yy * yy; // y^4
        Fp zz = z * z; // z^2

        // S = 2 * ((X1 + YY)^2 - XX - YYYY)
        Fp s = two * (((x + yy) * (x + yy)) - xx - yyyy);

        // M = 3 * XX + a * ZZ ^ 2
        Fp m = (three * xx) + (a_fp * (zz * zz));

        // X3 = T = M^2 - 2*S
        Fp x3 = (m * m) - (two * s);

        // Z3 = (Y + Z) ^ 2 - YY - ZZ
        // or Z3 = 2 * Y * Z
        Fp z3 = two * y * z;

        // Y3 = M*(S-X3)-8*YYYY
        Fp y3 = m * (s - x3) - eight * yyyy;

        return ECPoint(x3, y3, z3);
    }
};
