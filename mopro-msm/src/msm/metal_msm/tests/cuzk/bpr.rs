use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand::thread_rng, UniformRand, Zero};
use metal::*;
use num_bigint::BigUint;
use rayon::prelude::*;

fn bucket_point_reduction(buckets: &Vec<G>) -> G {
    let log_limb_size: u32 = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let rinv = res.0;
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    // Convert points to Montgomery form and create buffers
    let device = get_default_device();

    let bucket_size = buckets.len() as u32;

    // Create buffer for buckets
    let bucket_buffer = device.new_buffer(
        (bucket_size as usize * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Copy bucket data to buffer
    unsafe {
        let bucket_ptr = bucket_buffer.contents() as *mut G;
        for (i, point) in buckets.iter().enumerate() {
            *bucket_ptr.add(i) = *point;
        }
    }

    // Calculate optimal number of threads
    // The bucket size should be 2^s - 1, and we want to divide it into t parts
    // Each thread will handle r = (2^s - 1) / t elements
    let target_elements_per_thread = 2;
    let total_threads =
        ((bucket_size + target_elements_per_thread - 1) / target_elements_per_thread) as u32;

    let total_threads_buf = device.new_buffer_with_data(
        &total_threads as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let bucket_size_buf = device.new_buffer_with_data(
        &bucket_size as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result_buffer = device.new_buffer(
        (total_threads as usize * std::mem::size_of::<G>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create shared memory buffers for M and S arrays
    let m_shared_size = (bucket_size as usize * std::mem::size_of::<G>()) as u64;
    let s_shared_size = (total_threads as usize * std::mem::size_of::<G>()) as u64;

    let m_shared_buffer = device.new_buffer(m_shared_size, MTLResourceOptions::StorageModeShared);
    let s_shared_buffer = device.new_buffer(s_shared_size, MTLResourceOptions::StorageModeShared);

    // Set up command queue and encoder
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    // Write constants and compile shader
    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
    let library_path = compile_metal("../mopro-msm/src/msm/metal_msm/shader/cuzk", "bpr.metal");
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("bpr", None).unwrap();

    // Set up pipeline
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);

    // Set buffers
    encoder.set_buffer(0, Some(&bucket_buffer), 0);
    encoder.set_buffer(1, Some(&result_buffer), 0);
    encoder.set_buffer(2, Some(&bucket_size_buf), 0);
    encoder.set_buffer(3, Some(&total_threads_buf), 0);
    encoder.set_buffer(4, Some(&m_shared_buffer), 0);
    encoder.set_buffer(5, Some(&s_shared_buffer), 0);

    // Set up thread counts
    let thread_group_size = MTLSize {
        width: 8 as u64,
        height: 1,
        depth: 1,
    };

    let thread_groups = MTLSize {
        width: 2 as u64, // Round up to next multiple of threadgroup size
        height: 1,
        depth: 1,
    };
    println!("thread_groups: {:?}", thread_groups);
    println!("thread_group_size: {:?}", thread_group_size);

    encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    // TODO: check if m_shared and s_shared have the correct results
    // After the above for loop
    // s_shared[tid] = B_{(tid-1)r+1} + 2*B_{(tid-1)r+2} + ... + r*B_{(tid-1)r+r}
    let r = buckets.len() / total_threads as usize;
    println!("r = {:?}", r);
    let mut sum = G::zero();
    for i in 0..r {
        sum += buckets[(1 - 1) * r + (i + 1)] * ScalarField::from((i + 1) as u64);
    }
    //let s_shared = read_buffer(&s_shared_buffer, num_limbs);
    let s_shared_ptr = s_shared_buffer.contents() as *const G;
    let mut s = Vec::with_capacity(10000);
    for i in 0..total_threads as usize {
        unsafe {
            let point = *s_shared_ptr.add(i);
            s.push(point);
        }
    }
    println!("CPU: {:?}", sum);
    println!("GPU: {:?}", s);
    // m_shared[tid*r] = B_{(tid-1)r+1} + B_{(tid-1)r+2} + ... + B_{(tid-1)r+r}

    // Read results from the result buffer
    let result_ptr = result_buffer.contents() as *const G;
    let mut results = Vec::with_capacity(total_threads as usize);
    for i in 0..total_threads as usize {
        unsafe {
            let point = *result_ptr.add(i);
            //println!("Reading result {}: {:?}", i, point);
            results.push(point);
        }
    }
    results.iter().fold(G::zero(), |acc, &x| acc + x)
    //G::zero()
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

#[test]
pub fn test_bpr_simple_input() {
    // generate a list of points to represent bucket sums
    let generator = G::generator();
    let c: u32 = 4;
    let bucket_size = 1 << c;
    println!("bucket size: {:?}", bucket_size);

    let buckets = (1..=bucket_size).map(|_| generator).collect::<Vec<G>>();
    println!("\n\ninput buckets: {:?}\n\n", buckets);
    println!("input buckets length: {:?}\n", buckets.len());
    let expected = cpu_bucket_point_reduction(&buckets);

    let result = bucket_point_reduction(&buckets);
    println!("\n====================================================");
    println!("- CPU: {:?}\n", expected);
    println!("- GPU: {:?}", result);
    println!("====================================================\n");

    // assert_eq!(expected, result, "GPU and CPU results don't match");
}
