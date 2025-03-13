// adapted from https://github.com/geometryxyz/msl-secp256k1

use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::{AffineRepr, CurveGroup};
use metal::*;

use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand::thread_rng, UniformRand};
use num_bigint::BigUint;

#[test]
#[serial_test::serial]
pub fn test_jacobian_add_2007_bl() {
    let log_limb_size = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let rinv = res.0;
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    // Generate 2 random affine points
    let (a, b) = {
        let mut rng = thread_rng();
        let base_point = GAffine::generator().into_group();

        let s1 = ScalarField::rand(&mut rng);
        let mut s2 = ScalarField::rand(&mut rng);

        // Ensure s1 and s2 are different (if s1 == s2, we use pDBL instead of pADD)
        while s1 == s2 {
            s2 = ScalarField::rand(&mut rng);
        }

        (base_point * s1, base_point * s2)
    };

    // Compute the sum in projective form using Arkworks
    let expected = a + b;

    // Set B into Affine form
    let b = b.into_affine();

    let ax: BigUint = a.x.into();
    let ay: BigUint = a.y.into();
    let az: BigUint = a.z.into();
    let bx: BigUint = b.x.into();
    let by: BigUint = b.y.into();

    let axr = (&ax * &r) % &p;
    let ayr = (&ay * &r) % &p;
    let azr = (&az * &r) % &p;
    let bxr = (&bx * &r) % &p;
    let byr = (&by * &r) % &p;

    let axr_limbs = ark_ff::BigInt::<4>::try_from(axr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let ayr_limbs = ark_ff::BigInt::<4>::try_from(ayr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let azr_limbs = ark_ff::BigInt::<4>::try_from(azr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let bxr_limbs = ark_ff::BigInt::<4>::try_from(bxr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let byr_limbs = ark_ff::BigInt::<4>::try_from(byr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);

    let device = get_default_device();
    let axr_buf = create_buffer(&device, &axr_limbs);
    let ayr_buf = create_buffer(&device, &ayr_limbs);
    let azr_buf = create_buffer(&device, &azr_limbs);
    let bxr_buf = create_buffer(&device, &bxr_limbs);
    let byr_buf = create_buffer(&device, &byr_limbs);
    let result_xr_buf = create_empty_buffer(&device, num_limbs);
    let result_yr_buf = create_empty_buffer(&device, num_limbs);
    let result_zr_buf = create_empty_buffer(&device, num_limbs);

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
        "../mopro-msm/src/msm/metal_msm/shader/curve",
        "jacobian_madd_2007_bl.metal",
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
    encoder.set_buffer(0, Some(&axr_buf), 0);
    encoder.set_buffer(1, Some(&ayr_buf), 0);
    encoder.set_buffer(2, Some(&azr_buf), 0);
    encoder.set_buffer(3, Some(&bxr_buf), 0);
    encoder.set_buffer(4, Some(&byr_buf), 0);
    encoder.set_buffer(5, Some(&result_xr_buf), 0);
    encoder.set_buffer(6, Some(&result_yr_buf), 0);
    encoder.set_buffer(7, Some(&result_zr_buf), 0);

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

    let result_xr_limbs: Vec<u32> = read_buffer(&result_xr_buf, num_limbs);
    let result_yr_limbs: Vec<u32> = read_buffer(&result_yr_buf, num_limbs);
    let result_zr_limbs: Vec<u32> = read_buffer(&result_zr_buf, num_limbs);

    // Drop the buffers after reading the results
    drop(axr_buf);
    drop(ayr_buf);
    drop(azr_buf);
    drop(bxr_buf);
    drop(byr_buf);
    drop(result_xr_buf);
    drop(result_yr_buf);
    drop(result_zr_buf);
    drop(command_queue);

    let result_xr: BigUint = BigInt::<4>::from_limbs(&result_xr_limbs, log_limb_size)
        .try_into()
        .unwrap();
    let result_yr: BigUint = BigInt::<4>::from_limbs(&result_yr_limbs, log_limb_size)
        .try_into()
        .unwrap();
    let result_zr: BigUint = BigInt::<4>::from_limbs(&result_zr_limbs, log_limb_size)
        .try_into()
        .unwrap();

    let result_x = (result_xr * &rinv) % &p;
    let result_y = (result_yr * &rinv) % &p;
    let result_z = (result_zr * &rinv) % &p;

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    assert!(result == expected);
}
