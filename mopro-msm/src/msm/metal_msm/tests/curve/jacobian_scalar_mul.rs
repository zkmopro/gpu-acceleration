use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::AffineRepr;
use metal::*;

use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_ff::{BigInt, PrimeField};
use num_bigint::BigUint;
use rand::{self, Rng};

fn jacobian_scalar_mul_kernel(point: G, scalar: u32, name: &str) -> G {
    let log_limb_size = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let rinv = res.0;
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    let ax: BigUint = point.x.into();
    let ay: BigUint = point.y.into();
    let az: BigUint = point.z.into();

    let axr = (&ax * &r) % &p;
    let ayr = (&ay * &r) % &p;
    let azr = (&az * &r) % &p;

    let axr_limbs = ark_ff::BigInt::<4>::try_from(axr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let ayr_limbs = ark_ff::BigInt::<4>::try_from(ayr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let azr_limbs = ark_ff::BigInt::<4>::try_from(azr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);

    let device = get_default_device();
    let command_queue = device.new_command_queue();

    let mut point_data = vec![0u32; num_limbs * 3];
    point_data[..num_limbs].copy_from_slice(&axr_limbs);
    point_data[num_limbs..2 * num_limbs].copy_from_slice(&ayr_limbs);
    point_data[2 * num_limbs..].copy_from_slice(&azr_limbs);

    // Create buffers using gpu.rs functions
    let input_buf = create_buffer(&device, &point_data);
    let scalar_data = vec![scalar];
    let scalar_buf = create_buffer(&device, &scalar_data);
    let result_buf = create_empty_buffer(&device, num_limbs * 3);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/curve",
        &format!("{}.metal", name),
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

    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);
    encoder.set_compute_pipeline_state(&pipeline_state);

    encoder.set_buffer(0, Some(&input_buf), 0);
    encoder.set_buffer(1, Some(&scalar_buf), 0);
    encoder.set_buffer(2, Some(&result_buf), 0);

    let grid_size = MTLSize::new(1, 1, 1);
    let threadgroup_size = MTLSize::new(1, 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_slice = read_buffer(&result_buf, num_limbs * 3);

    // Drop the buffers after reading the results
    drop(input_buf);
    drop(scalar_buf);
    drop(result_buf);
    drop(command_queue);

    let result_xr = &result_slice[..num_limbs];
    let result_yr = &result_slice[num_limbs..2 * num_limbs];
    let result_zr = &result_slice[2 * num_limbs..];

    let result_x = (BigUint::try_from(BigInt::<4>::from_limbs(result_xr, log_limb_size)).unwrap()
        * &rinv)
        % &p;
    let result_y = (BigUint::try_from(BigInt::<4>::from_limbs(result_yr, log_limb_size)).unwrap()
        * &rinv)
        % &p;
    let result_z = (BigUint::try_from(BigInt::<4>::from_limbs(result_zr, log_limb_size)).unwrap()
        * &rinv)
        % &p;

    let result_x: BaseField = result_x.try_into().unwrap();
    let result_y: BaseField = result_y.try_into().unwrap();
    let result_z: BaseField = result_z.try_into().unwrap();

    let result = G::new_unchecked(result_x, result_y, result_z);
    result
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_scalar_mul() {
    let base_point = GAffine::generator().into_group();
    let scalar: u32 = 100;
    let expected = base_point * ScalarField::from(scalar as u64);
    let result = jacobian_scalar_mul_kernel(base_point, scalar, "jacobian_scalar_mul");
    assert!(expected == result);
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_scalar_mul_random() {
    // random point
    let base_point = GAffine::generator().into_group();
    let scalar = rand::thread_rng().gen::<u32>();
    let scalar_field = ScalarField::from(scalar);
    let point = base_point * scalar_field;

    // random scalar
    let rand_scalar = rand::thread_rng().gen::<u32>();
    let rand_scalar_field = ScalarField::from(rand_scalar);

    // random point * random scalar
    let expected = point * rand_scalar_field;
    let result = jacobian_scalar_mul_kernel(point, rand_scalar, "jacobian_scalar_mul");
    assert!(expected == result);
}
