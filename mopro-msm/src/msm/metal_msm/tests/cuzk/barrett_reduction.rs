use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

#[test]
#[serial_test::serial]
pub fn test_barrett_reduce_with_mont_params() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let num_limbs_extra_wide = num_limbs * 2; // maximum 512 bits
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (_, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&p);

    let mont_a = &a * &r;

    let expected = &mont_a % &p;
    let expected_in_ark: BigInt<4> = expected.clone().try_into().unwrap();

    let mont_a_in_ark: BigInt<8> = mont_a.clone().try_into().unwrap();
    let mont_a_limbs = mont_a_in_ark.to_limbs(num_limbs_extra_wide, log_limb_size);

    let device = get_default_device();
    let mont_a_buf = create_buffer(&device, &mont_a_limbs);
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
        nsafe,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/cuzk",
        "kernel_barrett_reduction.metal",
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
    encoder.set_buffer(0, Some(&mont_a_buf), 0);
    encoder.set_buffer(1, Some(&result_buf), 0);

    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&result_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    assert_eq!(result, expected_in_ark);
}

#[test]
#[serial_test::serial]
pub fn test_field_mul_with_mont_params() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (rinv, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&p);
    let r = calc_mont_radix(num_limbs, log_limb_size);

    let expected = &a * &r % &p;

    // Calculate expected result using Arkworks
    let a_in_ark: BigInt<4> = a.clone().try_into().unwrap();
    let r_in_ark: BigInt<4> = r.clone().try_into().unwrap();

    // Prepare Metal buffers
    let device = get_default_device();
    let a_limbs = a_in_ark.to_limbs(num_limbs, log_limb_size);
    let r_limbs = r_in_ark.to_limbs(num_limbs, log_limb_size);
    let a_buf = create_buffer(&device, &a_limbs);
    let r_buf = create_buffer(&device, &r_limbs);
    let res_buf = create_empty_buffer(&device, num_limbs);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/cuzk",
        "kernel_field_mul.metal",
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
    encoder.set_buffer(1, Some(&r_buf), 0);
    encoder.set_buffer(2, Some(&res_buf), 0);

    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&res_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);
    assert_eq!(result, expected.clone().try_into().unwrap());

    // verify correctness by restoring expected from montgomery form
    let a_in_field = BaseField::from_bigint(a_in_ark).unwrap();
    let expected_restored = (&expected * &rinv) % &p;
    assert_eq!(
        expected_restored,
        a_in_field.into_bigint().try_into().unwrap()
    );
}
