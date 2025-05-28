use ark_bn254::{Fr as ScalarField, G1Affine as Affine};
use ark_ff::{
    biginteger::{BigInt, BigInteger},
    PrimeField,
};
use rayon::prelude::*;
use std::convert::TryInto;

use crate::msm::metal_msm::utils::metal_wrapper::MetalConfig;

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

/// Optimized version for very large datasets that minimizes memory allocations
/// and uses chunked parallel processing for better cache locality
pub fn pack_affine_and_scalars(
    bases: &[Affine],
    scalars: &[ScalarField],
    msm_config: &MetalConfig,
) -> (Vec<u32>, Vec<u32>) {
    let num_elements = bases.len();
    let packed_limbs_per_coord = msm_config.num_limbs / 2;
    let coords_per_point = packed_limbs_per_coord * 2; // x + y

    // Pre-allocate output vectors with exact capacity
    let mut coords = vec![0u32; num_elements * coords_per_point];
    let mut scalars_u32 = vec![0u32; num_elements * packed_limbs_per_coord];

    // Helper function to pack limbs into halfword layout
    let pack_limbs_into = |limbs: &[u32], output: &mut [u32]| {
        for (i, chunk) in limbs.chunks(2).enumerate() {
            output[i] = (chunk[1] << 16) | chunk[0];
        }
    };

    // Process in parallel chunks for better cache locality
    const CHUNK_SIZE: usize = 1024; // Adjust based on cache size

    bases
        .par_chunks(CHUNK_SIZE)
        .zip(scalars.par_chunks(CHUNK_SIZE))
        .zip(coords.par_chunks_mut(CHUNK_SIZE * coords_per_point))
        .zip(scalars_u32.par_chunks_mut(CHUNK_SIZE * packed_limbs_per_coord))
        .for_each(
            |(((base_chunk, scalar_chunk), coord_chunk), scalar_u32_chunk)| {
                for (i, (pt, sc)) in base_chunk.iter().zip(scalar_chunk.iter()).enumerate() {
                    // Process point coordinates
                    let x_limbs =
                        pt.x.into_bigint()
                            .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);
                    let y_limbs =
                        pt.y.into_bigint()
                            .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);

                    // Pack directly into output buffer
                    let coord_start = i * coords_per_point;
                    pack_limbs_into(
                        &x_limbs,
                        &mut coord_chunk[coord_start..coord_start + packed_limbs_per_coord],
                    );
                    pack_limbs_into(
                        &y_limbs,
                        &mut coord_chunk
                            [coord_start + packed_limbs_per_coord..coord_start + coords_per_point],
                    );

                    // Process scalar
                    let sc_limbs = sc
                        .into_bigint()
                        .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);

                    // Pack directly into output buffer
                    let scalar_start = i * packed_limbs_per_coord;
                    pack_limbs_into(
                        &sc_limbs,
                        &mut scalar_u32_chunk[scalar_start..scalar_start + packed_limbs_per_coord],
                    );
                }
            },
        );

    (coords, scalars_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::msm::metal_msm::test_utils::generate_random_bases_and_scalars;
    use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;
    use ark_bn254::Fq as BaseField;
    use ark_ff::{BigInt, PrimeField};
    use num_bigint::{BigUint, RandBigInt};
    use std::time::Instant;

    /// Sequential version for performance comparison
    fn pack_affine_and_scalars_sequential(
        bases: &[Affine],
        scalars: &[ScalarField],
        msm_config: &MetalConfig,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut scalars_u32 = Vec::new();

        let pack_limbs = |limbs: &[u32]| -> Vec<u32> {
            limbs
                .chunks(2)
                .map(|chunk| (chunk[1] << 16) | chunk[0])
                .collect()
        };

        for (pt, sc) in bases.iter().zip(scalars.iter()) {
            let x_limbs =
                pt.x.into_bigint()
                    .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);
            let y_limbs =
                pt.y.into_bigint()
                    .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);

            let x_packed = pack_limbs(&x_limbs);
            let y_packed = pack_limbs(&y_limbs);
            coords.extend_from_slice(&x_packed);
            coords.extend_from_slice(&y_packed);

            let sc_limbs = sc
                .into_bigint()
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);
            let sc_packed = pack_limbs(&sc_limbs);
            scalars_u32.extend_from_slice(&sc_packed);
        }

        (coords, scalars_u32)
    }

    #[test]
    #[serial_test::serial]
    fn test_parallel_packing_correctness() {
        let (bases, scalars) = generate_random_bases_and_scalars(1024);

        let config = MetalConfig {
            log_limb_size: 16,
            num_limbs: 16,
            shader_file: String::new(),
            kernel_name: String::new(),
        };

        // Test all three implementations
        let (coords_seq, scalars_seq) =
            pack_affine_and_scalars_sequential(&bases, &scalars, &config);
        let (coords_opt, scalars_opt) = pack_affine_and_scalars(&bases, &scalars, &config);

        // Verify all implementations produce the same results
        assert_eq!(
            coords_seq, coords_opt,
            "Optimized version should match sequential"
        );
        assert_eq!(
            scalars_seq, scalars_opt,
            "Optimized scalars should match sequential"
        );
    }

    #[test]
    #[serial_test::serial]
    #[ignore]
    fn benchmark_packing_performance() {
        let test_sizes = vec![1 << 10, 1 << 16, 1 << 20, 1 << 22];

        for &size in &test_sizes {
            println!("\n=== Benchmarking with {} elements ===", size);

            let (bases, scalars) = generate_random_bases_and_scalars(size);

            let config = MetalConfig {
                log_limb_size: 16,
                num_limbs: 16,
                shader_file: String::new(),
                kernel_name: String::new(),
            };

            // Benchmark sequential version
            let start = Instant::now();
            let _ = pack_affine_and_scalars_sequential(&bases, &scalars, &config);
            let sequential_time = start.elapsed();
            println!("Sequential: {:?}", sequential_time);

            // Benchmark optimized version
            let start = Instant::now();
            let _ = pack_affine_and_scalars(&bases, &scalars, &config);
            let optimized_time = start.elapsed();
            println!("Optimized: {:?}", optimized_time);

            let optimized_speedup =
                sequential_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

            println!("Optimized speedup: {:.2}x", optimized_speedup);
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_within_bigint256() {
        let num_limbs = 16;
        let log_limb_size = 16;

        let p_limbs = BaseField::MODULUS.to_limbs(num_limbs, log_limb_size);
        let p_bigint256 = BigInt::<4>::from_limbs(&p_limbs, log_limb_size);

        assert_eq!(BaseField::MODULUS, p_bigint256);
    }

    #[test]
    #[serial_test::serial]
    fn test_within_bigint384() {
        let num_limbs = 16;
        let num_limbs_wide = num_limbs + 1;
        let log_limb_size = 16;

        let mut rng = rand::thread_rng();
        let some_257_bit_number = rng.gen_biguint(257);
        let some_257_bit_number_bigint: BigInt<6> = some_257_bit_number.try_into().unwrap();
        let some_257_bit_number_limbs =
            some_257_bit_number_bigint.to_limbs(num_limbs_wide, log_limb_size);
        let some_257_bit_number_reconstructed =
            BigInt::<6>::from_limbs(&some_257_bit_number_limbs, log_limb_size);

        // Check if the original and reconstructed values are equal
        assert_eq!(
            some_257_bit_number_bigint,
            some_257_bit_number_reconstructed
        );
    }

    #[test]
    #[serial_test::serial]
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
