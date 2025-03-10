use crate::msm::metal_msm::host::gpu::{create_buffer, get_default_device};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::data_conversion::{points_from_gpu_buffer, points_to_gpu_buffer};
use crate::msm::metal_msm::utils::mont_params::MontgomeryParams;
use ark_bn254::{Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
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

// Implements parallel bucket reduction in GPU
fn gpu_parallel_bpr(buckets: &Vec<G>) -> G {
    let mont_params = MontgomeryParams::default();
    let log_limb_size: u32 = 16;
    let p: BigUint = mont_params.p;
    let num_limbs = mont_params.num_limbs;
    let rinv = mont_params.rinv;
    let n0 = mont_params.n0;
    let nsafe = mont_params.nsafe;

    let device = get_default_device();
    let bucket_size = buckets.len();
    let bucket_buffer = points_to_gpu_buffer(buckets, num_limbs, &device);

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

    let thread_group_width = pipeline_state.thread_execution_width();
    let thread_group_height =
        pipeline_state.max_total_threads_per_threadgroup() / thread_group_width;

    let max_group_threads = pipeline_state.max_total_threads_per_threadgroup();

    let optimal_threads = max_group_threads;

    let candidate = if bucket_size < optimal_threads as usize {
        bucket_size
    } else {
        optimal_threads as usize // This is wrong, since we're workloading just a single TG,
                                 // The thing is that this gives the best performance so far
                                 // But we need to introduce cross kernel synchronisation to improve this.
    };

    let total_threads = closest_power_of_two(candidate) as usize;

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

    let total_threads_buffer = create_buffer(&device, &vec![total_threads as u32]);
    let grid_width_buffer = create_buffer(&device, &vec![grid_width as u32]);

    let m_shared = vec![G::zero(); total_threads];
    let s_shared = vec![G::zero(); total_threads];

    let m_shared_buffer = points_to_gpu_buffer(&m_shared, num_limbs, &device);
    let s_shared_buffer = points_to_gpu_buffer(&s_shared, num_limbs, &device);

    let num_subtask = (bucket_size + total_threads - 1) / total_threads;
    let num_subtask_buffer = create_buffer(&device, &vec![num_subtask as u32]);

    // set up command queue and encoder
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder =
        command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );

    encoder.set_compute_pipeline_state(&pipeline_state);

    encoder.set_buffer(0, Some(&bucket_buffer), 0);
    encoder.set_buffer(1, Some(&m_shared_buffer), 0);
    encoder.set_buffer(2, Some(&s_shared_buffer), 0);
    encoder.set_buffer(3, Some(&grid_width_buffer), 0);
    encoder.set_buffer(4, Some(&total_threads_buffer), 0);
    encoder.set_buffer(5, Some(&num_subtask_buffer), 0);

    encoder.dispatch_threads(threads_total, threads_per_thread_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let s_shared = points_from_gpu_buffer(&s_shared_buffer, num_limbs, p, rinv);

    s_shared.iter().sum::<G>()
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
    let total_threads = 4 as usize; // TODO: To make this dynamic
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
