// adapted from: https://github.com/geometryxyz/msl-secp256k1

use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::{FromLimbs, ToLimbs};
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use ark_std::rand;
use metal::*;

#[test]
#[serial_test::serial]
pub fn test_ff_sub() {
    let log_limb_size = 16;
    let num_limbs = 16;

    // Scalar field modulus for bn254
    let p = BaseField::MODULUS;

    let mut rng = rand::thread_rng();
    let mut a = BigInt::rand(&mut rng);
    let mut b = BigInt::rand(&mut rng);

    // Reduce a and b if they are greater than or equal to the prime field modulus
    while a >= p {
        a.sub_with_borrow(&p);
    }

    while b >= p {
        b.sub_with_borrow(&p);
    }

    // Ensure a and b are non-negative and less than p
    assert!(a >= BigInt::from(0u64), "a must be non-negative");
    assert!(b >= BigInt::from(0u64), "b must be non-negative");
    assert!(a < p, "a must be less than p");
    assert!(b < p, "b must be less than p");

    let device = get_default_device();
    let a_buf = create_buffer(&device, &a.to_limbs(num_limbs, log_limb_size));
    let b_buf = create_buffer(&device, &b.to_limbs(num_limbs, log_limb_size));
    let p_buf = create_buffer(&device, &p.to_limbs(num_limbs, log_limb_size));
    let result_buf = create_empty_buffer(&device, num_limbs);

    // (a - b) % p
    let mut expected = a.clone();
    if a >= b {
        expected.sub_with_borrow(&b);
    }
    // p - (b - a)
    else {
        let mut p_sub_b = p.clone();
        p_sub_b.sub_with_borrow(&b);
        expected.add_with_carry(&p_sub_b);
    }

    // Ensure expected is non-negative and less than p
    assert!(
        expected >= BigInt::from(0u64),
        "expected must be non-negative"
    );
    assert!(expected < p, "expected must be less than p");

    // Ensure the operation is correct using Arkworks
    let a_field = BaseField::from(a);
    let b_field = BaseField::from(b);
    let expected_field = a_field - b_field;
    assert!(expected_field == expected.into());

    let expected_limbs = expected.to_limbs(num_limbs, log_limb_size);

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
        "ff_sub.metal",
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
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&p_buf), 0);
    encoder.set_buffer(3, Some(&result_buf), 0);

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
}
