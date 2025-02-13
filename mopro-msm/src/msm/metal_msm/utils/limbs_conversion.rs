use ark_ff::biginteger::{BigInt, BigInteger};
use std::convert::TryInto;

/// A trait that abstracts "to/from limbs" for *any* BigInteger type
pub trait GenericLimbConversion {
    /// The number of 64-bit words in this BigInteger (e.g., 4 for BigInt<4>)
    const NUM_WORDS: usize;

    /// Convert to big-endian `u32` limbs of length `2 * NUM_WORDS`.
    /// (Because each 64-bit word can be split into two 32-bit limbs.)
    fn to_u32_limbs(&self) -> Vec<u32>;

    /// Convert to `num_limbs` (little-endian) of size `1 << log_limb_size` each.
    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32>;

    /// Construct from big-endian `u32` limbs.
    fn from_u32_limbs(limbs: &[u32]) -> Self;

    /// Construct from little-endian `u128`.
    fn from_u128(num: u128) -> Self;

    /// Construct from little-endian `u32`.
    fn from_u32(num: u32) -> Self;

    /// Construct from variable-size limbs, each of size `1 << log_limb_size`.
    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self;
}

impl GenericLimbConversion for BigInt<4> {
    const NUM_WORDS: usize = 4; // 4 x 64-bit words

    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut limbs = Vec::new();
        // BigInt<4>::to_bytes_be() gives us 32 bytes in BE
        self.to_bytes_be().chunks(8).for_each(|chunk| {
            let high = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
            let low = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
            limbs.push(high);
            limbs.push(low);
        });
        limbs
    }

    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32> {
        let mut result = vec![0u32; num_limbs];
        let limb_size = 1u32 << log_limb_size;
        let mask = limb_size - 1;

        // Convert to LE
        let bytes = self.to_bytes_le();
        let mut val = 0u32;
        let mut bits = 0u32;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            // if we've already filled all limbs, discard the rest and break
            if limb_idx >= num_limbs {
                break;
            }
            val |= (byte as u32) << bits;
            bits += 8;

            while bits >= log_limb_size && limb_idx < num_limbs {
                result[limb_idx] = val & mask;
                val >>= log_limb_size;
                bits -= log_limb_size;
                limb_idx += 1;
            }
        }
        // handle leftover
        if bits > 0 && limb_idx < num_limbs {
            result[limb_idx] = val;
        }
        result
    }

    fn from_u32_limbs(limbs: &[u32]) -> Self {
        let mut big_int = [0u64; 4];
        for (i, limb_pair) in limbs.chunks(2).rev().enumerate() {
            let high = u64::from(limb_pair[0]);
            let low = u64::from(limb_pair[1]);
            big_int[i] = (high << 32) | low;
        }
        BigInt::<4>::new(big_int)
    }

    fn from_u128(num: u128) -> Self {
        let high = (num >> 64) as u64;
        let low = num as u64;
        BigInt::<4>::new([low, high, 0, 0])
    }

    fn from_u32(num: u32) -> Self {
        BigInt::<4>::new([num as u64, 0, 0, 0])
    }

    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self {
        let mut result = [0u64; 4];
        let limb_bits = log_limb_size as usize;
        let mut accumulated_bits = 0;
        let mut current_u64 = 0u64;
        let mut result_idx = 0;

        for &limb in limbs {
            current_u64 |= (limb as u64) << accumulated_bits;
            accumulated_bits += limb_bits;

            while accumulated_bits >= 64 && result_idx < 4 {
                result[result_idx] = current_u64;
                // shift out the stored bits
                current_u64 = (limb as u64) >> (limb_bits - (accumulated_bits - 64));
                accumulated_bits -= 64;
                result_idx += 1;
            }
        }
        if accumulated_bits > 0 && result_idx < 4 {
            result[result_idx] = current_u64;
        }
        BigInt::<4>::new(result)
    }
}

impl GenericLimbConversion for BigInt<6> {
    const NUM_WORDS: usize = 6; // 6 x 64-bit words

    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut limbs = Vec::new();
        self.to_bytes_be().chunks(8).for_each(|chunk| {
            let high = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
            let low = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
            limbs.push(high);
            limbs.push(low);
        });
        limbs
    }

    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32> {
        let mut result = vec![0u32; num_limbs];
        let limb_size = 1u32 << log_limb_size;
        let mask = limb_size - 1;

        let bytes = self.to_bytes_le();
        let mut val = 0u32;
        let mut bits = 0u32;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            // if we've already filled all limbs, discard the rest and break
            if limb_idx >= num_limbs {
                break;
            }
            val |= (byte as u32) << bits;
            bits += 8;

            while bits >= log_limb_size && limb_idx < num_limbs {
                result[limb_idx] = val & mask;
                val >>= log_limb_size;
                bits -= log_limb_size;
                limb_idx += 1;
            }
        }
        if bits > 0 && limb_idx < num_limbs {
            result[limb_idx] = val;
        }
        result
    }

    fn from_u32_limbs(limbs: &[u32]) -> Self {
        let mut big_int = [0u64; 6];
        for (i, limb_pair) in limbs.chunks(2).rev().enumerate() {
            let high = u64::from(limb_pair[0]);
            let low = u64::from(limb_pair[1]);
            big_int[i] = (high << 32) | low;
        }
        BigInt::<6>::new(big_int)
    }

    fn from_u128(num: u128) -> Self {
        let high = (num >> 64) as u64;
        let low = num as u64;
        BigInt::<6>::new([low, high, 0, 0, 0, 0])
    }

    fn from_u32(num: u32) -> Self {
        BigInt::<6>::new([num as u64, 0, 0, 0, 0, 0])
    }

    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self {
        let mut result = [0u64; 6];
        let limb_bits = log_limb_size as usize;
        let mut accumulated_bits = 0;
        let mut current_u64 = 0u64;
        let mut result_idx = 0;

        for &limb in limbs {
            current_u64 |= (limb as u64) << accumulated_bits;
            accumulated_bits += limb_bits;

            while accumulated_bits >= 64 && result_idx < 6 {
                result[result_idx] = current_u64;
                current_u64 = (limb as u64) >> (limb_bits - (accumulated_bits - 64));
                accumulated_bits -= 64;
                result_idx += 1;
            }
        }
        if accumulated_bits > 0 && result_idx < 6 {
            result[result_idx] = current_u64;
        }
        BigInt::<6>::new(result)
    }
}

impl GenericLimbConversion for BigInt<8> {
    const NUM_WORDS: usize = 8; // 8 x 64-bit words

    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut limbs = Vec::new();
        self.to_bytes_be().chunks(8).for_each(|chunk| {
            let high = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
            let low = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
            limbs.push(high);
            limbs.push(low);
        });
        limbs
    }

    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32> {
        let mut result = vec![0u32; num_limbs];
        let limb_size = 1u32 << log_limb_size;
        let mask = limb_size - 1;

        let bytes = self.to_bytes_le();
        let mut val = 0u32;
        let mut bits = 0u32;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            if limb_idx >= num_limbs {
                break;
            }
            val |= (byte as u32) << bits;
            bits += 8;

            while bits >= log_limb_size && limb_idx < num_limbs {
                result[limb_idx] = val & mask;
                val >>= log_limb_size;
                bits -= log_limb_size;
                limb_idx += 1;
            }
        }
        if bits > 0 && limb_idx < num_limbs {
            result[limb_idx] = val;
        }
        result
    }

    fn from_u32_limbs(limbs: &[u32]) -> Self {
        let mut big_int = [0u64; 8];
        for (i, limb_pair) in limbs.chunks(2).rev().enumerate() {
            let high = u64::from(limb_pair[0]);
            let low = u64::from(limb_pair[1]);
            big_int[i] = (high << 32) | low;
        }
        BigInt::<8>::new(big_int)
    }

    fn from_u128(num: u128) -> Self {
        let high = (num >> 64) as u64;
        let low = num as u64;
        BigInt::<8>::new([low, high, 0, 0, 0, 0, 0, 0])
    }

    fn from_u32(num: u32) -> Self {
        BigInt::<8>::new([num as u64, 0, 0, 0, 0, 0, 0, 0])
    }

    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self {
        let mut result = [0u64; 8];
        let limb_bits = log_limb_size as usize;
        let mut accumulated_bits = 0;
        let mut current_u64 = 0u64;
        let mut result_idx = 0;

        for &limb in limbs {
            current_u64 |= (limb as u64) << accumulated_bits;
            accumulated_bits += limb_bits;

            while accumulated_bits >= 64 && result_idx < 8 {
                result[result_idx] = current_u64;
                current_u64 = (limb as u64) >> (limb_bits - (accumulated_bits - 64));
                accumulated_bits -= 64;
                result_idx += 1;
            }
        }
        if accumulated_bits > 0 && result_idx < 8 {
            result[result_idx] = current_u64;
        }
        BigInt::<8>::new(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;
    use ark_bn254::Fq as BaseField;
    use ark_ff::{BigInt, PrimeField};
    use num_bigint::{BigUint, RandBigInt};

    #[test]
    fn test_within_bigint256() {
        let num_limbs = 16;
        let log_limb_size = 16;

        let p_limbs = BaseField::MODULUS.to_limbs(num_limbs, log_limb_size);
        let p_bigint256 = BigInt::<4>::from_limbs(&p_limbs, log_limb_size);

        assert_eq!(BaseField::MODULUS, p_bigint256);
    }

    #[test]
    fn test_within_bigint384() {
        let num_limbs = 16;
        let num_limbs_wide = num_limbs + 1;
        let log_limb_size = 16;

        let r = calc_mont_radix(num_limbs, log_limb_size); // r has 257 bits
        let r_bigint384: BigInt<6> = r.try_into().unwrap();
        let r_limbs = r_bigint384.to_limbs(num_limbs_wide, log_limb_size);
        let r_reconstructed = BigInt::<6>::from_limbs(&r_limbs, log_limb_size);

        // Check if the original and reconstructed values are equal
        assert_eq!(r_bigint384, r_reconstructed);
    }

    #[test]
    fn test_within_bigint512() {
        let num_limbs = 16;
        let num_limbs_extra_wide = num_limbs * 2;
        let log_limb_size = 16;

        let mut rng = rand::thread_rng();
        let p: BigUint = BaseField::MODULUS.try_into().unwrap();
        let a = rng.gen_biguint_below(&p); // a has at most 254 bits
        let r = calc_mont_radix(num_limbs, log_limb_size); // r has 257 bits

        let mont_a = &a * &r; // mont_a has at most 511 bits

        let mont_a_bigint512: BigInt<8> = mont_a.try_into().unwrap();
        let mont_a_limbs = mont_a_bigint512.to_limbs(num_limbs_extra_wide, log_limb_size);
        let mont_a_reconstructed = BigInt::<8>::from_limbs(&mont_a_limbs, log_limb_size);

        // Check if the original and reconstructed values are equal
        assert_eq!(mont_a_bigint512, mont_a_reconstructed);
    }
}
