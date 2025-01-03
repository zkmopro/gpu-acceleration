// adapted from: https://github.com/geometryxyz/msl-secp256k1

// we avoid using ark_ff::ScalarField here because the mont radix exceeds the range of its field
// and we need to use num_bigint::BigUint for better flexibility

use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::{FromLimbs, ToLimbs};
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_rinv_and_n0};
use ark_bn254::Fr as ScalarField;
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

#[test]
#[serial_test::serial]
pub fn test_mont_mul_12() {
    do_test(12);
}

#[test]
#[serial_test::serial]
pub fn test_mont_mul_13() {
    do_test(13);
}

pub fn do_test(log_limb_size: u32) {
    // Calculate num_limbs based on modulus size and limb size
    let modulus_bits = ScalarField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let p: BigUint = ScalarField::MODULUS.try_into().unwrap();

    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;

    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&p);
    let b = rng.gen_biguint_below(&p);

    let a_r = &a * &r % &p;
    let b_r = &b * &r % &p;
    let expected = (&a * &b * &r) % &p;

    let a_r_in_ark = ScalarField::from_bigint(a_r.clone().try_into().unwrap()).unwrap();
    let b_r_in_ark = ScalarField::from_bigint(b_r.clone().try_into().unwrap()).unwrap();
    let expected_in_ark = ScalarField::from_bigint(expected.clone().try_into().unwrap()).unwrap();
    let expected_limbs = expected_in_ark
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    let device = get_default_device();
    let a_buf = create_buffer(
        &device,
        &a_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size),
    );
    let b_buf = create_buffer(
        &device,
        &b_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size),
    );
    let p_buf = create_buffer(
        &device,
        &ScalarField::MODULUS.to_limbs(num_limbs, log_limb_size),
    );
    let result_buf = create_empty_buffer(&device, num_limbs);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        1,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/mont_backend",
        "mont_mul_optimised.metal",
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

    assert!(result == expected.try_into().unwrap());
    assert!(result_limbs == expected_limbs);
}

#[test]
#[serial_test::serial]
pub fn test_number_conversions() {
    // Setup parameters
    let log_limb_size = 12;
    let modulus_bits = ScalarField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    // Create test values using small numbers for clarity
    let original_biguint = BigUint::parse_bytes(b"123456789", 10).unwrap();

    // Convert BigUint to ScalarField
    let scalar_field_value =
        ScalarField::from_bigint(original_biguint.clone().try_into().unwrap()).unwrap();

    // Convert ScalarField to limbs
    let limbs = scalar_field_value
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    // Convert limbs back to BigUint
    let converted_biguint: BigUint = BigInt::from_limbs(&limbs, log_limb_size)
        .try_into()
        .unwrap();

    // Verify the round trip conversion
    assert_eq!(
        original_biguint, converted_biguint,
        "Round trip conversion failed: original {} != converted {}",
        original_biguint, converted_biguint
    );

    // Test with multiple values to ensure robustness
    let test_values = vec![
        BigUint::parse_bytes(b"1", 10).unwrap(),
        BigUint::parse_bytes(b"12345", 10).unwrap(),
        BigUint::parse_bytes(b"999999999999", 10).unwrap(),
    ];

    for value in test_values {
        let scalar = ScalarField::from_bigint(value.clone().try_into().unwrap()).unwrap();
        let value_limbs = scalar.into_bigint().to_limbs(num_limbs, log_limb_size);
        let converted: BigUint = BigInt::from_limbs(&value_limbs, log_limb_size)
            .try_into()
            .unwrap();

        assert_eq!(value, converted, "Conversion failed for value {}", value);
    }
}
