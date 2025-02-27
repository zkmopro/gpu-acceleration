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

// Implements parallel bucket reduction in GPU
// TODO: 1. current algorithm does not support odd bucket size (should be 2N)
// TODO: 2. current total threads is fixed, we should change it into dynamic
// TODO: 3. we should also dynamically dispatch the threadgroup_size and threads_per_threadgroup
fn gpu_parallel_bpr(buckets: &Vec<G>) -> G {
    let mont_params = MontgomeryParams::default();
    let log_limb_size: u32 = 16;
    let p: BigUint = mont_params.p;
    let num_limbs = mont_params.num_limbs;
    // let r = mont_params.r;
    let rinv = mont_params.rinv;
    let n0 = mont_params.n0;
    let nsafe = mont_params.nsafe;

    let device = get_default_device();
    let bucket_size = buckets.len();
    let bucket_buffer = points_to_gpu_buffer(buckets, num_limbs, &device);

    let total_threads: u32 = 512; // TODO: To make this dynamic

    let total_threads_buffer = create_buffer(&device, &vec![total_threads]);
    let bucket_size_buffer = create_buffer(&device, &vec![bucket_size as u32]);

    let m_shared = vec![G::zero(); total_threads as usize];
    let s_shared = vec![G::zero(); total_threads as usize];

    let m_shared_buffer = points_to_gpu_buffer(&m_shared, num_limbs, &device);
    let s_shared_buffer = points_to_gpu_buffer(&s_shared, num_limbs, &device);

    let num_subtask = (bucket_size + total_threads as usize - 1) / total_threads as usize;
    let num_subtask_buffer = create_buffer(&device, &vec![num_subtask as u32]);
    println!("r: {}", num_subtask);
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
    let library_path = compile_metal("../mopro-msm/src/msm/metal_msm/shader/cuzk", "bpr.metal");
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("parallel_bpr", None).unwrap();

    // set up pipeline
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);

    encoder.set_buffer(0, Some(&bucket_buffer), 0);
    encoder.set_buffer(1, Some(&m_shared_buffer), 0);
    encoder.set_buffer(2, Some(&s_shared_buffer), 0);
    encoder.set_buffer(3, Some(&bucket_size_buffer), 0);
    encoder.set_buffer(4, Some(&total_threads_buffer), 0);
    encoder.set_buffer(5, Some(&num_subtask_buffer), 0);

    // TODO: make this dynamic
    let threads_per_thread_group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    let thread_groups = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_threads(thread_groups, threads_per_thread_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let s_shared = points_from_gpu_buffer(&s_shared_buffer, num_limbs, p, rinv);

    s_shared.iter().sum::<G>()
}

// This is very naive way to implement the bucket reduction
// computing $\sum_{i=1}^{len} i * buckets[i]$
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
// TODO: 1. This algorithm only supports even bucket size (2N), we have to find a way to deal with odd input size.
// TODO: 2. make total thread dynamic
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
pub fn test_pbpr_simple_input() {
    let generator = G::generator();
    let c: u32 = 16;
    let bucket_size = 1 << c;
    let buckets = (1..=bucket_size)
        .map(|i| generator * ScalarField::from(i as u64))
        .collect::<Vec<G>>();

    let cpu_naive_result = cpu_naive_bpr(&buckets);
    let cpu_pbpr_result = cpu_parallel_bpr(&buckets);
    let gpu_pbpr_result = gpu_parallel_bpr(&buckets);

    assert_eq!(gpu_pbpr_result, cpu_naive_result);
    assert_eq!(gpu_pbpr_result, cpu_pbpr_result);
}

#[test]
fn test_pbpr_random_inputs() {
    let generator = G::generator();
    let mut rng = thread_rng();
    let bucket_size = vec![10, 11, 12, 13, 14, 15];

    for size in bucket_size {
        let buckets = (0..(1 << size))
            .map(|_| generator * ScalarField::rand(&mut rng))
            .collect::<Vec<G>>();
        let naive_result = cpu_naive_bpr(&buckets);
        let gpu_pbpr_result = gpu_parallel_bpr(&buckets);

        assert_eq!(gpu_pbpr_result, naive_result);
    }
}
