use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use ark_std::rand;
use metal::*;

#[test]
#[serial_test::serial]
pub fn test_ff_reduce_a_less_than_p() {
    let log_limb_size = 16;
    let num_limbs = 16;

    // Scalar field modulus for bn254
    let p = BaseField::MODULUS;

    let mut rng = rand::thread_rng();
    let raw_a = BigInt::<4>::rand(&mut rng);

    // Ensure a is non-negative
    assert!(raw_a >= BigInt::from(0u64), "a must be non-negative");

    // Perform a % p
    let mut a = raw_a.clone();

    // While result >= p, subtract p
    while a >= p {
        a.sub_with_borrow(&p);
    }
    // Ensure expected is non-negative and less than p
    assert!(a >= BigInt::from(0u64), "a must be non-negative");
    assert!(a < p, "a must be less than p");

    let a_limbs = a.to_limbs(num_limbs, log_limb_size);
    let device = get_default_device();
    let a_buf = create_buffer(&device, &a_limbs);
    let result_buf = create_empty_buffer(&device, num_limbs);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        0,
        0,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/field",
        "ff_reduce.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("run", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&result_buf), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&result_buf, num_limbs);
    let result = BigInt::from_limbs(&result_limbs, log_limb_size);

    assert!(result == a);
    assert!(result_limbs == a_limbs);

    // Drop the buffers after reading the results
    drop(a_buf);
    drop(result_buf);
    drop(command_queue);
}

#[test]
#[serial_test::serial]
pub fn test_ff_reduce_a_greater_than_p_less_than_2p() {
    let log_limb_size = 16;
    let num_limbs = 16;

    // Scalar field modulus for bn254
    let p = BaseField::MODULUS;
    let mut two_p = p.clone();
    two_p.add_with_carry(&p);

    let mut rng = rand::thread_rng();
    let raw_a = BigInt::<4>::rand(&mut rng);

    // Ensure a is non-negative
    assert!(raw_a >= BigInt::from(0u64), "a must be non-negative");

    // Perform a % p
    let mut a = raw_a.clone();

    // While result >= p, subtract p
    while a >= p {
        a.sub_with_borrow(&p);
    }
    let expected = a.clone();
    let expected_limbs = a.to_limbs(num_limbs, log_limb_size);

    // Adding p to a to ensure a is in the range [p, 2p)
    a.add_with_carry(&p);

    // Ensure expected is non-negative and less than p
    assert!(a >= BigInt::from(0u64), "a must be non-negative");
    assert!(a < two_p, "a must be less than 2p");
    assert!(a >= p, "a must be greater than or equal to p");

    let a_limbs = a.to_limbs(num_limbs, log_limb_size);
    let device = get_default_device();
    let a_buf = create_buffer(&device, &a_limbs);
    let result_buf = create_empty_buffer(&device, num_limbs);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        0,
        0,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/field",
        "ff_reduce.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("run", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&result_buf), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&result_buf, num_limbs);
    let result = BigInt::from_limbs(&result_limbs, log_limb_size);

    assert!(result == expected);
    assert!(result_limbs == expected_limbs);

    // Drop the buffers after reading the results
    drop(a_buf);
    drop(result_buf);
    drop(command_queue);
}
