use crate::msm::metal_msm::host::gpu::{create_buffer, get_default_device};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::data_conversion::raw_reduction;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, MontgomeryParams};
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand::thread_rng, UniformRand, Zero};
use metal::*;
use num_bigint::BigUint;
use rayon::prelude::*;
use std::time::Instant;

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

/// Implement parallel bucket reduction in GPU using separated buffers internally
fn gpu_parallel_bpr(buckets: &Vec<G>) -> G {
    /// Converts a bucket vector into three separated vectors for the x, y, and z coordinates
    /// Each coordinate is represented as a vector of u32 values in Montgomery form
    fn buckets_to_separated_coords(
        buckets: &Vec<G>,
        num_limbs: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let log_limb_size: u32 = 16;
        let p: BigUint = BaseField::MODULUS.try_into().unwrap();
        let r = calc_mont_radix(num_limbs, log_limb_size);

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

            let pxr = (&px * &r) % &p;
            let pyr = (&py * &r) % &p;
            let pzr = (&pz * &r) % &p;

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
    ) -> Vec<G> {
        let log_limb_size = 16;
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

    let mont_params = MontgomeryParams::default();
    let log_limb_size: u32 = 16;
    let num_limbs = mont_params.num_limbs;
    let n0 = mont_params.n0;
    let nsafe = mont_params.nsafe;

    let device = get_default_device();
    let bucket_size = buckets.len();

    // Convert buckets to separated coordinate arrays
    let (buckets_x, buckets_y, buckets_z) = buckets_to_separated_coords(buckets, num_limbs);

    // Create buffers for coordinates
    let buckets_x_buffer = create_buffer(&device, &buckets_x);
    let buckets_y_buffer = create_buffer(&device, &buckets_y);
    let buckets_z_buffer = create_buffer(&device, &buckets_z);

    // Compile shader and create pipeline
    let library_path = compile_metal("../mopro-msm/src/msm/metal_msm/shader/cuzk", "pbpr.metal");
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("parallel_bpr", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    // Get thread dimensions
    let thread_group_width = pipeline_state.thread_execution_width();
    let thread_group_height =
        pipeline_state.max_total_threads_per_threadgroup() / thread_group_width;
    let max_group_threads = pipeline_state.max_total_threads_per_threadgroup();
    let optimal_threads = max_group_threads;

    // Calculate total threads needed
    let candidate = if bucket_size < optimal_threads as usize {
        bucket_size
    } else {
        optimal_threads as usize // This is wrong, since we're workloading just a single TG,
                                 // The thing is that this gives the best performance so far
                                 // But we need to introduce cross kernel synchronisation to improve this.
    };

    let total_threads = closest_power_of_two(candidate);

    // Calculate grid dimensions
    let grid_width = (total_threads as f64).sqrt().ceil() as u64;
    let grid_height = (total_threads as u64 + grid_width - 1) / grid_width;

    let threads_per_thread_group = MTLSize {
        width: thread_group_width,
        height: thread_group_height,
        depth: 1,
    };

    let threads_total = MTLSize {
        width: grid_width,
        height: grid_height,
        depth: 1,
    };

    // Create output buffers
    let zero_point = G::zero();
    let (zero_x, zero_y, zero_z) = buckets_to_separated_coords(&vec![zero_point], num_limbs);

    // Fill the buffers with zero points
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

    let m_shared_x_buffer = create_buffer(&device, &m_shared_x);
    let m_shared_y_buffer = create_buffer(&device, &m_shared_y);
    let m_shared_z_buffer = create_buffer(&device, &m_shared_z);

    let s_shared_x_buffer = create_buffer(&device, &s_shared_x);
    let s_shared_y_buffer = create_buffer(&device, &s_shared_y);
    let s_shared_z_buffer = create_buffer(&device, &s_shared_z);

    // Create parameter buffers
    let grid_width_buffer = create_buffer(&device, &vec![grid_width as u32]);
    let total_threads_buffer = create_buffer(&device, &vec![total_threads as u32]);
    let num_subtask = (bucket_size + total_threads - 1) / total_threads;
    let num_subtask_buffer = create_buffer(&device, &vec![num_subtask as u32]);

    // Set up command queue and encoder
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder =
        command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

    // Write constants
    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );

    encoder.set_compute_pipeline_state(&pipeline_state);

    // Set kernel arguments
    encoder.set_buffer(0, Some(&buckets_x_buffer), 0);
    encoder.set_buffer(1, Some(&buckets_y_buffer), 0);
    encoder.set_buffer(2, Some(&buckets_z_buffer), 0);
    encoder.set_buffer(3, Some(&m_shared_x_buffer), 0);
    encoder.set_buffer(4, Some(&m_shared_y_buffer), 0);
    encoder.set_buffer(5, Some(&m_shared_z_buffer), 0);
    encoder.set_buffer(6, Some(&s_shared_x_buffer), 0);
    encoder.set_buffer(7, Some(&s_shared_y_buffer), 0);
    encoder.set_buffer(8, Some(&s_shared_z_buffer), 0);
    encoder.set_buffer(9, Some(&grid_width_buffer), 0);
    encoder.set_buffer(10, Some(&total_threads_buffer), 0);
    encoder.set_buffer(11, Some(&num_subtask_buffer), 0);

    // Dispatch threads
    encoder.dispatch_threads(threads_total, threads_per_thread_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read results back
    let s_shared_x_ptr = s_shared_x_buffer.contents() as *const u32;
    let s_shared_y_ptr = s_shared_y_buffer.contents() as *const u32;
    let s_shared_z_ptr = s_shared_z_buffer.contents() as *const u32;

    let mut s_shared_x_result = vec![0u32; total_threads * num_limbs];
    let mut s_shared_y_result = vec![0u32; total_threads * num_limbs];
    let mut s_shared_z_result = vec![0u32; total_threads * num_limbs];

    unsafe {
        std::ptr::copy_nonoverlapping(
            s_shared_x_ptr,
            s_shared_x_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
        std::ptr::copy_nonoverlapping(
            s_shared_y_ptr,
            s_shared_y_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
        std::ptr::copy_nonoverlapping(
            s_shared_z_ptr,
            s_shared_z_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
    }

    // Read results back
    let m_shared_x_ptr = m_shared_x_buffer.contents() as *const u32;
    let m_shared_y_ptr = m_shared_y_buffer.contents() as *const u32;
    let m_shared_z_ptr = m_shared_z_buffer.contents() as *const u32;

    let mut m_shared_x_result = vec![0u32; total_threads * num_limbs];
    let mut m_shared_y_result = vec![0u32; total_threads * num_limbs];
    let mut m_shared_z_result = vec![0u32; total_threads * num_limbs];

    unsafe {
        std::ptr::copy_nonoverlapping(
            m_shared_x_ptr,
            m_shared_x_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
        std::ptr::copy_nonoverlapping(
            m_shared_y_ptr,
            m_shared_y_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
        std::ptr::copy_nonoverlapping(
            m_shared_z_ptr,
            m_shared_z_result.as_mut_ptr(),
            total_threads * num_limbs,
        );
    }

    // for debug
    // let m_shared_points = points_from_separated_buffers(&m_shared_x_result, &m_shared_y_result, &m_shared_z_result, num_limbs);

    // Convert results back to points
    let s_shared_points = points_from_separated_buffers(
        &s_shared_x_result,
        &s_shared_y_result,
        &s_shared_z_result,
        num_limbs,
    );

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
pub fn test_pbpr_simple_input() {
    let generator = G::generator();
    let c: u32 = 10;
    let bucket_size = 1 << c;
    let buckets = (1..=bucket_size)
        .map(|i| generator * ScalarField::from(i as u64))
        .collect::<Vec<G>>();

    let cpu_naive_result = cpu_naive_bpr(&buckets);
    let cpu_pbpr_result = cpu_parallel_bpr(&buckets);

    let start = Instant::now();
    let gpu_pbpr_result = gpu_parallel_bpr(&buckets);
    let duration = start.elapsed();
    println!("gpu_parallel_bpr execution time: {:?}", duration);

    assert_eq!(gpu_pbpr_result, cpu_naive_result);
    assert_eq!(gpu_pbpr_result, cpu_pbpr_result);
}

#[test]
#[serial_test::serial]
fn test_pbpr_random_inputs() {
    let generator = G::generator();
    let mut rng = thread_rng();
    let bucket_size = vec![5, 6, 7, 8, 9, 10];

    for size in bucket_size {
        let buckets = (0..(1 << size))
            .map(|_| generator * ScalarField::rand(&mut rng))
            .collect::<Vec<G>>();
        let naive_result = cpu_naive_bpr(&buckets);
        let start = Instant::now();
        let gpu_pbpr_result = gpu_parallel_bpr(&buckets);
        let duration = start.elapsed();
        println!(
            "Size: {}, gpu_parallel_bpr execution time: {:?}",
            size, duration
        );
        assert_eq!(gpu_pbpr_result, naive_result);
    }
}
