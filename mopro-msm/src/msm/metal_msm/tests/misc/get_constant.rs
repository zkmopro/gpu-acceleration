use crate::msm::metal_msm::host::gpu::{create_empty_buffer, get_default_device, read_buffer};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::barrett_params::calc_barrett_mu;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField, Zero};
use metal::*;
use num_bigint::BigUint;

const NUM_LIMBS: usize = 16;
const NUM_LIMBS_WIDE: usize = 17;
const LOG_LIMB_SIZE: u32 = 16;

#[test]
pub fn test_get_n0() {
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
    let kernel = library.get_function("test_get_n0", None).unwrap();

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
    let result = BigInt::<4>::from_limbs(&result_limbs, LOG_LIMB_SIZE);

    let (_, _, _, n0, _) = calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let expected = BigInt::<4>::from(n0);
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

/// for 16-bit limbs, r has 257 bits
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
    let result = BigInt::<6>::from_limbs(&result_limbs, LOG_LIMB_SIZE);

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

#[test]
pub fn test_get_bn254_zero() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_x_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_y_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_z_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_bn254_zero", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_x_buf), 0);
    encoder.set_buffer(1, Some(&result_y_buf), 0);
    encoder.set_buffer(2, Some(&result_z_buf), 0);

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

    let result_x_limbs: Vec<u32> = read_buffer(&result_x_buf, NUM_LIMBS);
    let result_y_limbs: Vec<u32> = read_buffer(&result_y_buf, NUM_LIMBS);
    let result_z_limbs: Vec<u32> = read_buffer(&result_z_buf, NUM_LIMBS);
    let result_x = BigInt::<4>::from_limbs(&result_x_limbs, LOG_LIMB_SIZE);
    let result_y = BigInt::<4>::from_limbs(&result_y_limbs, LOG_LIMB_SIZE);
    let result_z = BigInt::<4>::from_limbs(&result_z_limbs, LOG_LIMB_SIZE);

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::zero();
    assert!(result == expected);
}

#[test]
pub fn test_get_bn254_one() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_x_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_y_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_z_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("test_get_bn254_one", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_x_buf), 0);
    encoder.set_buffer(1, Some(&result_y_buf), 0);
    encoder.set_buffer(2, Some(&result_z_buf), 0);

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

    let result_x_limbs: Vec<u32> = read_buffer(&result_x_buf, NUM_LIMBS);
    let result_y_limbs: Vec<u32> = read_buffer(&result_y_buf, NUM_LIMBS);
    let result_z_limbs: Vec<u32> = read_buffer(&result_z_buf, NUM_LIMBS);
    let result_x = BigInt::<4>::from_limbs(&result_x_limbs, LOG_LIMB_SIZE);
    let result_y = BigInt::<4>::from_limbs(&result_y_limbs, LOG_LIMB_SIZE);
    let result_z = BigInt::<4>::from_limbs(&result_z_limbs, LOG_LIMB_SIZE);

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::generator();
    assert!(result == expected);
}

#[test]
pub fn test_get_bn254_zero_mont() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_xr_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_yr_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_zr_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library
        .get_function("test_get_bn254_zero_mont", None)
        .unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_xr_buf), 0);
    encoder.set_buffer(1, Some(&result_yr_buf), 0);
    encoder.set_buffer(2, Some(&result_zr_buf), 0);

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

    let result_xr_limbs: Vec<u32> = read_buffer(&result_xr_buf, NUM_LIMBS);
    let result_yr_limbs: Vec<u32> = read_buffer(&result_yr_buf, NUM_LIMBS);
    let result_zr_limbs: Vec<u32> = read_buffer(&result_zr_buf, NUM_LIMBS);
    let result_xr: BigUint = BigInt::<4>::from_limbs(&result_xr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();
    let result_yr: BigUint = BigInt::<4>::from_limbs(&result_yr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();
    let result_zr: BigUint = BigInt::<4>::from_limbs(&result_zr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();

    let (p, _, rinv, _, _) = calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let result_x = (result_xr * &rinv) % &p;
    let result_y = (result_yr * &rinv) % &p;
    let result_z = (result_zr * &rinv) % &p;

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::zero();
    assert!(result == expected);
}

#[test]
pub fn test_get_bn254_one_mont() {
    prepare_constants(NUM_LIMBS, LOG_LIMB_SIZE);

    let device = get_default_device();
    let result_xr_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_yr_buf = create_empty_buffer(&device, NUM_LIMBS);
    let result_zr_buf = create_empty_buffer(&device, NUM_LIMBS);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/misc",
        "test_get_constant.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library
        .get_function("test_get_bn254_one_mont", None)
        .unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&result_xr_buf), 0);
    encoder.set_buffer(1, Some(&result_yr_buf), 0);
    encoder.set_buffer(2, Some(&result_zr_buf), 0);

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

    let result_xr_limbs: Vec<u32> = read_buffer(&result_xr_buf, NUM_LIMBS);
    let result_yr_limbs: Vec<u32> = read_buffer(&result_yr_buf, NUM_LIMBS);
    let result_zr_limbs: Vec<u32> = read_buffer(&result_zr_buf, NUM_LIMBS);
    let result_xr: BigUint = BigInt::<4>::from_limbs(&result_xr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();
    let result_yr: BigUint = BigInt::<4>::from_limbs(&result_yr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();
    let result_zr: BigUint = BigInt::<4>::from_limbs(&result_zr_limbs, LOG_LIMB_SIZE)
        .try_into()
        .unwrap();

    let (p, _, rinv, _, _) = calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let result_x = (result_xr * &rinv) % &p;
    let result_y = (result_yr * &rinv) % &p;
    let result_z = (result_zr * &rinv) % &p;

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::generator();
    assert!(result == expected);
}

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

    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let expected_mu: BigInt<4> = calc_barrett_mu(&p).try_into().unwrap();
    let expected_limbs = expected_mu.to_limbs(NUM_LIMBS, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected_mu);
}

// Helper to calculate constants
fn calc_constants(num_limbs: usize, log_limb_size: u32) -> (BigUint, BigUint, BigUint, u32, usize) {
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (rinv, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);
    (p, r, rinv, n0, nsafe)
}

fn prepare_constants(num_limbs: usize, log_limb_size: u32) {
    let (_, _, _, n0, nsafe) = calc_constants(num_limbs, log_limb_size);
    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
}
