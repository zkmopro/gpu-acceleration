use crate::msm::metal_msm::host::gpu::{create_empty_buffer, get_default_device, read_buffer};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::BigUint;

const NUM_LIMBS: usize = 16;
const NUM_LIMBS_WIDE: usize = 17;
const LOG_LIMB_SIZE: u32 = 16;

#[test]
pub fn test_get_mu() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_mu", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_buf), 0);

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

    let result_limbs: Vec<u32> = read_buffer(&result_buf, NUM_LIMBS);
    let result = BigInt::from_limbs(&result_limbs, LOG_LIMB_SIZE);

    let expected = calc_mu(NUM_LIMBS, LOG_LIMB_SIZE);
    let expected_limbs = expected.to_limbs(NUM_LIMBS, LOG_LIMB_SIZE);

    assert_eq!(result, expected);
    assert_eq!(result_limbs, expected_limbs);
}

#[test]
pub fn test_get_p() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_p", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_buf), 0);

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

    let result_limbs: Vec<u32> = read_buffer(&result_buf, NUM_LIMBS);
    let result = BigInt::from_limbs(&result_limbs, LOG_LIMB_SIZE);

    let expected = BaseField::MODULUS;
    let expected_limbs = expected.to_limbs(NUM_LIMBS, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
}

#[test]
pub fn test_get_r() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_buf = create_empty_buffer(&device, NUM_LIMBS_WIDE);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_r", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_buf), 0);

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

    let result_limbs: Vec<u32> = read_buffer(&result_buf, NUM_LIMBS_WIDE);
    let result = BigInt::from_limbs(&result_limbs, LOG_LIMB_SIZE);

    let expected: BigInt<6> = calc_mont_radix(NUM_LIMBS, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();
    let expected_limbs = expected.to_limbs(NUM_LIMBS_WIDE, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
}

#[test]
pub fn test_get_p_wide() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_buf = create_empty_buffer(&device, NUM_LIMBS_WIDE);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_p_wide", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_buf), 0);

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

    let result_limbs: Vec<u32> = read_buffer(&result_buf, NUM_LIMBS_WIDE);
    let result = BigInt::from_limbs(&result_limbs, LOG_LIMB_SIZE);

    let expected = BaseField::MODULUS;
    let expected_limbs = expected.to_limbs(NUM_LIMBS_WIDE, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
}

// Helper to calculate constants
fn calc_mu(num_limbs: usize, log_limb_size: u32) -> BigInt<4> {
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;
    BigInt::from(n0)
}

fn prepare_constants(num_limbs: usize, log_limb_size: u32) {
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
}
