use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};
use num_bigint::BigUint;
use rand::{self, Rng};

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

fn jacobian_scalar_mul_kernel(point: G, scalar: u32) -> G {
    let log_limb_size = 16;
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: format!("curve/jacobian_scalar_mul.metal"),
        kernel_name: "test_jacobian_scalar_mul".to_string(),
    };

    let mut helper = MetalHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p = &constants.p;
    let r = &constants.r;
    let rinv = &constants.rinv;

    let ax: BigUint = point.x.into();
    let ay: BigUint = point.y.into();
    let az: BigUint = point.z.into();

    let axr = (ax * r) % p;
    let ayr = (ay * r) % p;
    let azr = (az * r) % p;

    let axr_limbs = ark_ff::BigInt::<4>::try_from(axr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let ayr_limbs = ark_ff::BigInt::<4>::try_from(ayr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let azr_limbs = ark_ff::BigInt::<4>::try_from(azr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);

    // Create input buffers
    let point_buf = helper.create_buffer(
        &[
            axr_limbs.as_slice(),
            ayr_limbs.as_slice(),
            azr_limbs.as_slice(),
        ]
        .concat(),
    );
    let scalar_buf = helper.create_buffer(&vec![scalar]);
    let result_buf = helper.create_empty_buffer(num_limbs * 3);

    // Setup thread group sizes
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&point_buf, &scalar_buf, &result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read results
    let result_data = helper.read_results(&result_buf, num_limbs * 3);

    let result_xr = &result_data[..num_limbs];
    let result_yr = &result_data[num_limbs..2 * num_limbs];
    let result_zr = &result_data[2 * num_limbs..];

    let result_x =
        ((BigUint::try_from(BigInt::<4>::from_limbs(result_xr, log_limb_size)).unwrap() * rinv)
            % p)
            .try_into()
            .unwrap();
    let result_y =
        ((BigUint::try_from(BigInt::<4>::from_limbs(result_yr, log_limb_size)).unwrap() * rinv)
            % p)
            .try_into()
            .unwrap();
    let result_z =
        ((BigUint::try_from(BigInt::<4>::from_limbs(result_zr, log_limb_size)).unwrap() * rinv)
            % p)
            .try_into()
            .unwrap();

    helper.drop_all_buffers();

    G::new_unchecked(result_x, result_y, result_z)
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_scalar_mul() {
    let base_point = GAffine::generator().into_group();
    let scalar: u32 = 100;
    let expected = base_point * ScalarField::from(scalar as u64);
    let result = jacobian_scalar_mul_kernel(base_point, scalar);
    assert!(expected == result);
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_scalar_mul_random() {
    let base_point = GAffine::generator().into_group();
    let scalar = rand::thread_rng().gen::<u32>();
    let scalar_field = ScalarField::from(scalar);
    let point = base_point * scalar_field;

    let rand_scalar = rand::thread_rng().gen::<u32>();
    let rand_scalar_field = ScalarField::from(rand_scalar);

    let expected = point * rand_scalar_field;
    let result = jacobian_scalar_mul_kernel(point, rand_scalar);
    assert!(expected == result);
}
