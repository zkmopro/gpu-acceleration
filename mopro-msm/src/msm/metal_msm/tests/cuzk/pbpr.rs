use ark_bn254::{Fq as BaseField, FqConfig, Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
use ark_ff::{
    biginteger::{arithmetic as fa, BigInt},
    fields::models::MontConfig,
    PrimeField,
};
use ark_std::{rand::thread_rng, UniformRand, Zero};
use num_bigint::BigUint;
use rayon::prelude::*;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

const N: usize = 4;

fn closest_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    // if already a power-of-two, return it immediately
    if n.is_power_of_two() {
        return n;
    }
    // lower: the highest power-of-two less than n.
    // upper: the next power-of-two greater than n.
    let lower = 1 << (usize::BITS - n.leading_zeros() - 1);
    let upper = lower << 1;
    if n - lower <= upper - n {
        lower
    } else {
        upper
    }
}

pub fn raw_reduction(a: BigInt<N>) -> BigInt<N> {
    let mut r = a.0; // parse into [u64; N]

    // Montgomery Reduction
    for i in 0..N {
        let k = r[i].wrapping_mul(<FqConfig as MontConfig<N>>::INV);
        let mut carry = 0;

        fa::mac_with_carry(
            r[i],
            k,
            <FqConfig as MontConfig<N>>::MODULUS.0[0],
            &mut carry,
        );
        for j in 1..N {
            r[(j + i) % N] = fa::mac_with_carry(
                r[(j + i) % N],
                k,
                <FqConfig as MontConfig<N>>::MODULUS.0[j],
                &mut carry,
            );
        }
        r[i % N] = carry;
    }
    BigInt::new(r)
}

/// Implement parallel bucket reduction in GPU using separated buffers internally
fn gpu_parallel_bpr(buckets: &Vec<G>) -> G {
    /// Converts a bucket vector into three separated vectors for the x, y, and z coordinates
    fn buckets_to_separated_coords(
        buckets: &Vec<G>,
        num_limbs: usize,
        log_limb_size: u32,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let constants = get_or_calc_constants(num_limbs, log_limb_size);
        let p = &constants.p;
        let r = &constants.r;

        // Pre-allocate vectors for the limbs
        let total_limbs = buckets.len() * num_limbs;
        let mut buckets_x = Vec::with_capacity(total_limbs);
        let mut buckets_y = Vec::with_capacity(total_limbs);
        let mut buckets_z = Vec::with_capacity(total_limbs);

        for point in buckets {
            // Convert to Montgomery form
            let px: BigUint = point.x.into();
            let py: BigUint = point.y.into();
            let pz: BigUint = point.z.into();

            let pxr = (&px * r) % p;
            let pyr = (&py * r) % p;
            let pzr = (&pz * r) % p;

            // Convert to limbs
            let pxr_limbs = BigInt::<4>::try_from(pxr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size);
            let pyr_limbs = BigInt::<4>::try_from(pyr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size);
            let pzr_limbs = BigInt::<4>::try_from(pzr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size);

            // Add limbs to the coordinate vectors
            buckets_x.extend_from_slice(&pxr_limbs);
            buckets_y.extend_from_slice(&pyr_limbs);
            buckets_z.extend_from_slice(&pzr_limbs);
        }

        (buckets_x, buckets_y, buckets_z)
    }

    /// Convert separated coordinate buffers from GPU back to a vector of points
    fn points_from_separated_buffers(
        x_buffer: &[u32],
        y_buffer: &[u32],
        z_buffer: &[u32],
        num_limbs: usize,
        log_limb_size: u32,
    ) -> Vec<G> {
        let coord_size = num_limbs;
        let total_u32s = x_buffer.len() as usize;
        let num_points = total_u32s / coord_size;

        let mut points = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let start_idx = i * coord_size;
            let end_idx = start_idx + coord_size;

            // Extract limbs for each coordinate
            let x_limbs = &x_buffer[start_idx..end_idx];
            let y_limbs = &y_buffer[start_idx..end_idx];
            let z_limbs = &z_buffer[start_idx..end_idx];

            // Convert limbs back to BigInt
            let x_bigint = raw_reduction(BigInt::<4>::from_limbs(x_limbs, log_limb_size));
            let y_bigint = raw_reduction(BigInt::<4>::from_limbs(y_limbs, log_limb_size));
            let z_bigint = raw_reduction(BigInt::<4>::from_limbs(z_limbs, log_limb_size));

            // Convert to field elements
            let x = BaseField::from_bigint(x_bigint).unwrap();
            let y = BaseField::from_bigint(y_bigint).unwrap();
            let z = BaseField::from_bigint(z_bigint).unwrap();

            // Create and add the point
            points.push(G::new_unchecked(x, y, z));
        }

        points
    }

    // Configure Metal test parameters
    let log_limb_size = 16;
    let num_limbs = 16;
    let bucket_size = buckets.len();

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "parallel_bpr".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Convert buckets to separated coordinate arrays
    let (buckets_x, buckets_y, buckets_z) =
        buckets_to_separated_coords(buckets, num_limbs, log_limb_size);

    // Create input buffers
    let buckets_x_buf = helper.create_input_buffer(&buckets_x);
    let buckets_y_buf = helper.create_input_buffer(&buckets_y);
    let buckets_z_buf = helper.create_input_buffer(&buckets_z);

    // Calculate thread dimensions
    let candidate = if bucket_size < 256 { bucket_size } else { 256 }; // Use 256 as default max threads
    let total_threads = closest_power_of_two(candidate);

    // Create zero point buffers for output
    let zero_point = G::zero();
    let (zero_x, zero_y, zero_z) =
        buckets_to_separated_coords(&vec![zero_point], num_limbs, log_limb_size);

    // Initialize output buffers with zero points
    let mut m_shared_x = vec![0u32; total_threads * num_limbs];
    let mut m_shared_y = vec![0u32; total_threads * num_limbs];
    let mut m_shared_z = vec![0u32; total_threads * num_limbs];
    let mut s_shared_x = vec![0u32; total_threads * num_limbs];
    let mut s_shared_y = vec![0u32; total_threads * num_limbs];
    let mut s_shared_z = vec![0u32; total_threads * num_limbs];

    // Fill each position with the zero point
    for i in 0..total_threads {
        for j in 0..num_limbs {
            m_shared_x[i * num_limbs + j] = zero_x[j];
            m_shared_y[i * num_limbs + j] = zero_y[j];
            m_shared_z[i * num_limbs + j] = zero_z[j];
            s_shared_x[i * num_limbs + j] = zero_x[j];
            s_shared_y[i * num_limbs + j] = zero_y[j];
            s_shared_z[i * num_limbs + j] = zero_z[j];
        }
    }

    // Create output buffers
    let m_shared_x_buf = helper.create_input_buffer(&m_shared_x);
    let m_shared_y_buf = helper.create_input_buffer(&m_shared_y);
    let m_shared_z_buf = helper.create_input_buffer(&m_shared_z);
    let s_shared_x_buf = helper.create_input_buffer(&s_shared_x);
    let s_shared_y_buf = helper.create_input_buffer(&s_shared_y);
    let s_shared_z_buf = helper.create_input_buffer(&s_shared_z);

    // Create parameter buffers
    let grid_width = (total_threads as f64).sqrt().ceil() as u64;
    let grid_width_buf = helper.create_input_buffer(&vec![grid_width as u32]);
    let total_threads_buf = helper.create_input_buffer(&vec![total_threads as u32]);
    let num_subtask = (bucket_size + total_threads - 1) / total_threads;
    let num_subtask_buf = helper.create_input_buffer(&vec![num_subtask as u32]);

    // Setup thread group sizes
    let thread_group_width = 32; // Default thread group width
    let thread_group_height = 1;
    let grid_height = (total_threads as u64 + grid_width - 1) / grid_width;

    let threads_per_thread_group =
        helper.create_thread_group_size(thread_group_width, thread_group_height, 1);

    let threads_total = helper.create_thread_group_size(grid_width, grid_height, 1);

    // Execute shader
    helper.execute_shader(
        &config,
        &[
            &buckets_x_buf,
            &buckets_y_buf,
            &buckets_z_buf,
            &m_shared_x_buf,
            &m_shared_y_buf,
            &m_shared_z_buf,
            &s_shared_x_buf,
            &s_shared_y_buf,
            &s_shared_z_buf,
            &grid_width_buf,
            &total_threads_buf,
            &num_subtask_buf,
        ],
        &[], // No separate output buffers - results are in the shared buffers
        &threads_total,
        &threads_per_thread_group,
    );

    // Read results back
    let s_shared_x_result = helper.read_results(&s_shared_x_buf, total_threads * num_limbs);
    let s_shared_y_result = helper.read_results(&s_shared_y_buf, total_threads * num_limbs);
    let s_shared_z_result = helper.read_results(&s_shared_z_buf, total_threads * num_limbs);

    // Convert results back to points
    let s_shared_points = points_from_separated_buffers(
        &s_shared_x_result,
        &s_shared_y_result,
        &s_shared_z_result,
        num_limbs,
        log_limb_size,
    );

    // Clean up
    helper.drop_all_buffers();

    // Sum the points
    s_shared_points.iter().sum::<G>()
}

// This is a very naive way to implement the bucket reduction
// computing sum_{i=1}^{len} i * buckets[i]
fn cpu_naive_bpr(buckets: &Vec<G>) -> G {
    buckets
        .par_iter()
        .enumerate()
        .fold(
            || G::zero(),
            |acc, (i, p)| acc + *p * ScalarField::from((i + 1) as u64),
        )
        .sum()
}

// This immitates the parallel bucket point reduction algortihm in GPU.
// TODO: 1. make total thread dynamic
fn cpu_parallel_bpr(buckets: &Vec<G>) -> G {
    let total_threads = 8 as usize; // TODO: To make this dynamic
    let bucket_size = buckets.len() as usize;
    let r = (bucket_size + total_threads - 1) / total_threads;
    let mut s = vec![G::zero(); total_threads];
    let mut m = vec![G::zero(); total_threads];

    for gid in 0..total_threads {
        for l in 1..=r {
            if l != 1 {
                m[gid] = m[gid] + buckets[(gid + 1) * r - l];
                s[gid] = s[gid] + m[gid];
            } else {
                m[gid] = buckets[(gid + 1) * r - 1];
                s[gid] = m[gid];
            }
        }
    }

    let mut result_arr: Vec<G> = vec![];
    for i in 0..total_threads {
        result_arr.push(s[i] + (m[i] * ScalarField::from((r * i) as u64)));
    }

    result_arr.iter().sum::<G>()
}

#[test]
#[serial_test::serial]
fn test_pbpr_random_inputs() {
    let generator = G::generator();
    let c: u32 = 5;
    let bucket_size = 1 << c;

    let mut rng = thread_rng();
    let buckets = (1..=bucket_size)
        .map(|_| generator * ScalarField::rand(&mut rng))
        .collect::<Vec<G>>();

    let cpu_naive_result = cpu_naive_bpr(&buckets);
    let cpu_pbpr_result = cpu_parallel_bpr(&buckets);
    let gpu_pbpr_result = gpu_parallel_bpr(&buckets);

    assert_eq!(gpu_pbpr_result, cpu_naive_result);
    assert_eq!(gpu_pbpr_result, cpu_pbpr_result);
}
