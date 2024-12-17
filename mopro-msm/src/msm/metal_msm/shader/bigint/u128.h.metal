// source: https://github.com/andrewmilson/ministark/blob/875fb385bab9fcbb347d4c69898b56cbeeb71ca1/gpu/src/metal/u256.h.metal

#ifndef u128_h
#define u128_h

class u128
{
public:
    u128() = default;
    constexpr u128(int l) : low(l), high(0) {}
    constexpr u128(unsigned long l) : low(l), high(0) {}
    constexpr u128(bool b) : low(b), high(0) {}
    constexpr u128(unsigned long h, unsigned long l) : low(l), high(h) {}

    constexpr u128 operator+(const u128 rhs) const
    {
        return u128(high + rhs.high + ((low + rhs.low) < low), low + rhs.low);
    }

    constexpr u128 operator+=(const u128 rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    constexpr inline u128 operator-(const u128 rhs) const
    {
        return u128(high - rhs.high - ((low - rhs.low) > low), low - rhs.low);
    }

    constexpr u128 operator-=(const u128 rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr bool operator==(const u128 rhs) const
    {
        return high == rhs.high && low == rhs.low;
    }

    constexpr bool operator!=(const u128 rhs) const
    {
        return !(*this == rhs);
    }

    constexpr bool operator<(const u128 rhs) const
    {
        return ((high == rhs.high) && (low < rhs.low)) || (high < rhs.high);
    }

    constexpr u128 operator&(const u128 rhs) const
    {
        return u128(high & rhs.high, low & rhs.low);
    }

    constexpr u128 operator|(const u128 rhs) const
    {
        return u128(high | rhs.high, low | rhs.low);
    }

    constexpr bool operator>(const u128 rhs) const
    {
        return ((high == rhs.high) && (low > rhs.low)) || (high > rhs.high);
    }

    constexpr bool operator>=(const u128 rhs) const
    {
        return !(*this < rhs);
    }

    constexpr bool operator<=(const u128 rhs) const
    {
        return !(*this > rhs);
    }

    constexpr inline u128 operator>>(unsigned shift) const
    {
        // TODO: reduce branch conditions
        if (shift >= 128)
        {
            return u128(0);
        }
        else if (shift == 64)
        {
            return u128(0, high);
        }
        else if (shift == 0)
        {
            return *this;
        }
        else if (shift < 64)
        {
            return u128(high >> shift, (high << (64 - shift)) | (low >> shift));
        }
        else if ((128 > shift) && (shift > 64))
        {
            return u128(0, (high >> (shift - 64)));
        }
        else
        {
            return u128(0);
        }
    }

    constexpr inline u128 operator<<(unsigned shift) const
    {
        // TODO: reduce branch conditions
        if (shift >= 128)
        {
            return u128(0);
        }
        else if (shift == 64)
        {
            return u128(low, 0);
        }
        else if (shift == 0)
        {
            return *this;
        }
        else if (shift < 64)
        {
            return u128((high << shift) | (low >> (64 - shift)), low << shift);
        }
        else if ((128 > shift) && (shift > 64))
        {
            return u128((low >> (shift - 64)), 0);
        }
        else
        {
            return u128(0);
        }
    }

    constexpr u128 operator>>=(unsigned rhs)
    {
        *this = *this >> rhs;
        return *this;
    }

    u128 operator*(const bool rhs) const
    {
        return u128(high * rhs, low * rhs);
    }

    u128 operator*(const u128 rhs) const
    {
        unsigned long t_low_high = metal::mulhi(low, rhs.high);
        unsigned long t_high = metal::mulhi(low, rhs.low);
        unsigned long t_high_low = metal::mulhi(high, rhs.low);
        unsigned long t_low = low * rhs.low;
        return u128(t_low_high + t_high_low + t_high, t_low);
    }

    u128 operator*=(const u128 rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    // TODO: Could get better performance with  smaller limb size
    // Not sure what word size is for M1 GPU
#ifdef __LITTLE_ENDIAN__
    unsigned long low;
    unsigned long high;
#endif
#ifdef __BIG_ENDIAN__
    unsigned long high;
    unsigned long low;
#endif
};

#endif /* u128_h */
