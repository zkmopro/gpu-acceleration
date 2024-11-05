use ark_bn254::Fq;
use ark_ff::biginteger::{BigInteger, BigInteger256};

use crate::msm::metal::abstraction::mont_reduction;

// implement to_u32_limbs and from_u32_limbs for BigInt<4>
pub trait ToLimbs {
    fn to_u32_limbs(&self) -> Vec<u32>;
    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32>;
}

pub trait FromLimbs {
    fn from_u32_limbs(limbs: &[u32]) -> Self;
    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self;
    fn from_u128(num: u128) -> Self;
    fn from_u32(num: u32) -> Self;
}

// convert from little endian to big endian
impl ToLimbs for BigInteger256 {
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

        // Convert to little-endian representation
        let bytes = self.to_bytes_le();
        let mut val = 0u32;
        let mut bits = 0u32;
        let mut limb_idx = 0;

        for &byte in bytes.iter() {
            val |= (byte as u32) << bits;
            bits += 8;

            while bits >= log_limb_size && limb_idx < num_limbs {
                result[limb_idx] = val & mask;
                val >>= log_limb_size;
                bits -= log_limb_size;
                limb_idx += 1;
            }
        }

        // Handle any remaining bits
        if bits > 0 && limb_idx < num_limbs {
            result[limb_idx] = val;
        }

        result
    }
}

// convert from little endian to big endian
impl ToLimbs for Fq {
    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut limbs = Vec::new();
        self.0.to_bytes_be().chunks(8).for_each(|chunk| {
            let high = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
            let low = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
            limbs.push(high);
            limbs.push(low);
        });
        limbs
    }

    fn to_limbs(&self, num_limbs: usize, log_limb_size: u32) -> Vec<u32> {
        self.0.to_limbs(num_limbs, log_limb_size)
    }
}

impl FromLimbs for BigInteger256 {
    // convert from big endian to little endian for metal
    fn from_u32_limbs(limbs: &[u32]) -> Self {
        let mut big_int = [0u64; 4];
        for (i, limb) in limbs.chunks(2).rev().enumerate() {
            let high = u64::from(limb[0]);
            let low = u64::from(limb[1]);
            big_int[i] = (high << 32) | low;
        }
        BigInteger256::new(big_int)
    }
    // provide little endian u128 since arkworks use this value as well
    fn from_u128(num: u128) -> Self {
        let high = (num >> 64) as u64;
        let low = num as u64;
        BigInteger256::new([low, high, 0, 0])
    }
    // provide little endian u32 since arkworks use this value as well
    fn from_u32(num: u32) -> Self {
        BigInteger256::new([num as u64, 0, 0, 0])
    }

    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self {
        let mut result = [0u64; 4];
        let limb_size = log_limb_size as usize;
        let mut accumulated_bits = 0;
        let mut current_u64 = 0u64;
        let mut result_idx = 0;

        for &limb in limbs {
            // Add the current limb at the appropriate position
            current_u64 |= (limb as u64) << accumulated_bits;
            accumulated_bits += limb_size;

            // If we've accumulated 64 bits or more, store the result
            while accumulated_bits >= 64 && result_idx < 4 {
                result[result_idx] = current_u64;
                current_u64 = limb as u64 >> (limb_size - (accumulated_bits - 64));
                accumulated_bits -= 64;
                result_idx += 1;
            }
        }

        // Handle any remaining bits
        if accumulated_bits > 0 && result_idx < 4 {
            result[result_idx] = current_u64;
        }

        BigInteger256::new(result)
    }
}

impl FromLimbs for Fq {
    // convert from big endian to little endian for metal
    fn from_u32_limbs(limbs: &[u32]) -> Self {
        let mut big_int = [0u64; 4];
        for (i, limb) in limbs.chunks(2).rev().enumerate() {
            let high = u64::from(limb[0]);
            let low = u64::from(limb[1]);
            big_int[i] = (high << 32) | low;
        }
        Fq::new(mont_reduction::raw_reduction(BigInteger256::new(big_int)))
    }
    fn from_u128(num: u128) -> Self {
        let high = (num >> 64) as u64;
        let low = num as u64;
        Fq::new(mont_reduction::raw_reduction(BigInteger256::new([
            low, high, 0, 0,
        ])))
    }
    fn from_u32(num: u32) -> Self {
        Fq::new(mont_reduction::raw_reduction(BigInteger256::new([
            num as u64, 0, 0, 0,
        ])))
    }

    fn from_limbs(limbs: &[u32], log_limb_size: u32) -> Self {
        let bigint = BigInteger256::from_limbs(limbs, log_limb_size);
        Fq::new(mont_reduction::raw_reduction(bigint))
    }
}
