use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand::thread_rng, UniformRand, Zero};
use metal::*;
use num_bigint::BigUint;

fn gpu_bpr_stage_1(buckets: &Vec<G>) -> (Vec<G>, Vec<G>) {
    let log_limb_size: u32 = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    let device = get_default_device();
    let bucket_size = buckets.len() as usize;

    let bucket_buffer = device.new_buffer(
        (bucket_size * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let bucket_ptr = bucket_buffer.contents() as *mut G;
        for (i, point) in buckets.iter().enumerate() {
            *bucket_ptr.add(i) = *point;
        }
    }

    let total_threads: u32 = 4; // TODO: should make it dynamic

    let total_threads_buffer = device.new_buffer_with_data(
        &total_threads as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let bucket_size_buffer = device.new_buffer_with_data(
        &bucket_size as *const usize as *const _,
        std::mem::size_of::<usize>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let s_shared_buffer = device.new_buffer(
        (total_threads as usize * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let m_shared_buffer = device.new_buffer(
        (bucket_size * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // TODO: remove me: for debugging
    let debug_idx_buffer = device.new_buffer(
        (total_threads as usize * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let debug_bucket_buffer = device.new_buffer(
        (bucket_size * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let debug_r_buffer = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

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
    let kernel = library.get_function("bpr_stage_1", None).unwrap();

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
    encoder.set_buffer(1, Some(&s_shared_buffer), 0);
    encoder.set_buffer(2, Some(&m_shared_buffer), 0);
    encoder.set_buffer(3, Some(&bucket_size_buffer), 0);
    encoder.set_buffer(4, Some(&total_threads_buffer), 0);

    // TODO: remove me:  for debugging
    encoder.set_buffer(5, Some(&debug_idx_buffer), 0);
    encoder.set_buffer(6, Some(&debug_bucket_buffer), 0);
    encoder.set_buffer(7, Some(&debug_r_buffer), 0);

    let thread_group_size = MTLSize {
        width: total_threads as u64,
        height: 1,
        depth: 1,
    };

    let thread_groups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_threads(thread_groups, thread_group_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // TODO: remove me: for debugging
    let debug_idx_ptr = debug_idx_buffer.contents() as *const u32;
    let debug_buckets_ptr = debug_bucket_buffer.contents() as *const G;
    let mut debug_idx = vec![0; total_threads as usize];
    let mut debug_buckets = vec![G::zero(); bucket_size as usize];
    for i in 0..total_threads as usize {
        unsafe {
            debug_idx[i] = *debug_idx_ptr.add(i);
        }
    }
    for i in 0..bucket_size as usize {
        unsafe {
            debug_buckets[i] = *debug_buckets_ptr.add(i);
        }
    }

    // println!("debug idx: {:?}", debug_idx);
    // println!("debug buckets: {:?}", debug_buckets);
    println!("debug r: {:?}", read_buffer(&debug_r_buffer, 1)[0]);

    let s_shared_ptr = s_shared_buffer.contents() as *const G;
    let mut s_shared = vec![G::zero(); total_threads as usize];
    for i in 0..total_threads as usize {
        unsafe {
            let point = *s_shared_ptr.add(i);
            s_shared[i] = point;
        }
    }

    let m_shared_ptr = m_shared_buffer.contents() as *const G;
    let mut m_shared = vec![G::zero(); bucket_size as usize];
    for i in 0..bucket_size as usize {
        unsafe {
            let point = *m_shared_ptr.add(i);
            m_shared[i] = point;
        }
    }

    (s_shared, m_shared)
}

// This is very naive way to implement the bucket reduction
// computing $\sum_{i=1}^{len} i * buckets[i]$
fn cpu_bucket_point_reduction(buckets: &Vec<G>) -> G {
    buckets
        .iter()
        .enumerate()
        .map(|(i, &b)| b * (ScalarField::from((i + 1) as u64)))
        .sum::<G>()
}

// TODO: still need to check if the results are correct
fn cpu_bpr_stage_1(buckets: &Vec<G>) -> (Vec<G>, Vec<G>) {
    let total_threads = 4 as usize;
    let bucket_size = buckets.len() as usize;
    let r = (bucket_size + total_threads as usize - 1) / total_threads as usize; // Ceiling division
    println!("r: {}", r);

    let mut s = vec![G::zero(); total_threads + 1];
    let mut m = vec![G::zero(); bucket_size + 1];

    for tid in 1..=total_threads {
        s[tid - 1] = G::zero();
        for l in 1..=r {
            let m_idx = (tid - 1) * r + l;
            if m_idx > bucket_size {
                break;
            }
            // println!("tid: {}, l: {}, midx: {}, tid*r-l: {}", tid, l, m_idx, tid * r - l);
            if l != 1 {
                m[m_idx] = m[m_idx - 1] + buckets[tid * r - l];
                s[tid - 1] = s[tid - 1] + m[m_idx];
            } else {
                m[m_idx] = buckets[tid * r + 1 - l];
                s[tid - 1] = m[m_idx];
            }
        }
    }

    (s, m)
}

#[test]
pub fn test_bpr_stage_1() {
    let generator = G::generator();
    let c: u32 = 3;
    let bucket_size = (1 << c) - 1; // 7
    let buckets = (1..=bucket_size)
        .map(|_| generator * ScalarField::from(2))
        .collect::<Vec<G>>();

    let r = bucket_size / 4;
    println!("r: {:?}", r);

    let (expected_s, expected_m) = cpu_bpr_stage_1(&buckets);
    let (s, m) = gpu_bpr_stage_1(&buckets);

    println!("expected_s: {:?}\n", expected_s);
    println!("s: {:?}\n", s);

    println!("expected_m: {:?}\n", expected_m);
    println!("m: {:?}\n", m);

    // assert_eq!(expected_s, s, "S_shared: GPU and CPU results don't match");
    // assert_eq!(expected_m, m, "M_shared: GPU and CPU results don't match");
}
