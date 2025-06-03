use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::{AffineRepr, Group};
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand::thread_rng, UniformRand, Zero};
use num_bigint::BigUint;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

fn jacobian_add_2007_bl_kernel(a: G, b: G, shader_name: &str) -> G {
    let log_limb_size = 16;
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: format!("curve/{}.metal", shader_name),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p = &constants.p;
    let r = &constants.r;
    let rinv = &constants.rinv;

    // Convert inputs to Montgomery domain
    let ax: BigUint = a.x.into();
    let ay: BigUint = a.y.into();
    let az: BigUint = a.z.into();
    let bx: BigUint = b.x.into();
    let by: BigUint = b.y.into();
    let bz: BigUint = b.z.into();

    let axr = (&ax * r) % p;
    let ayr = (&ay * r) % p;
    let azr = (&az * r) % p;
    let bxr = (&bx * r) % p;
    let byr = (&by * r) % p;
    let bzr = (&bz * r) % p;

    // Convert to limbs
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
    let bzr_limbs = ark_ff::BigInt::<4>::try_from(bzr.clone())
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);

    // Create buffers
    let axr_buf = helper.create_buffer(&axr_limbs);
    let ayr_buf = helper.create_buffer(&ayr_limbs);
    let azr_buf = helper.create_buffer(&azr_limbs);
    let bxr_buf = helper.create_buffer(&bxr_limbs);
    let byr_buf = helper.create_buffer(&byr_limbs);
    let bzr_buf = helper.create_buffer(&bzr_limbs);
    let result_xr_buf = helper.create_empty_buffer(num_limbs);
    let result_yr_buf = helper.create_empty_buffer(num_limbs);
    let result_zr_buf = helper.create_empty_buffer(num_limbs);

    // Setup thread group sizes
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &axr_buf,
            &ayr_buf,
            &azr_buf,
            &bxr_buf,
            &byr_buf,
            &bzr_buf,
            &result_xr_buf,
            &result_yr_buf,
            &result_zr_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read results
    let result_xr_limbs = helper.read_results(&result_xr_buf, num_limbs);
    let result_yr_limbs = helper.read_results(&result_yr_buf, num_limbs);
    let result_zr_limbs = helper.read_results(&result_zr_buf, num_limbs);

    // Convert results from Montgomery domain
    let result_xr: BigUint = BigInt::<4>::from_limbs(&result_xr_limbs, log_limb_size)
        .try_into()
        .unwrap();
    let result_yr: BigUint = BigInt::<4>::from_limbs(&result_yr_limbs, log_limb_size)
        .try_into()
        .unwrap();
    let result_zr: BigUint = BigInt::<4>::from_limbs(&result_zr_limbs, log_limb_size)
        .try_into()
        .unwrap();

    let result_x = (result_xr * rinv) % p;
    let result_y = (result_yr * rinv) % p;
    let result_z = (result_zr * rinv) % p;

    // Clean up buffers
    helper.drop_all_buffers();

    G::new(result_x.into(), result_y.into(), result_z.into())
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_add_2007_bl_zero() {
    let a = G::zero();
    let b = G::generator();
    let expected = a + b;
    let result = jacobian_add_2007_bl_kernel(a, b, "jacobian_add_2007_bl");
    assert!(result == expected);

    let c = G::generator();
    let d = G::zero();
    let expected = c + d;
    let result = jacobian_add_2007_bl_kernel(a, b, "jacobian_add_2007_bl");
    assert!(result == expected);
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_add_2007_bl_two_equals() {
    let a = G::zero();
    let expected = a + a;
    let result = jacobian_add_2007_bl_kernel(a, a, "jacobian_add_2007_bl");
    assert!(result == expected);

    let b = G::generator();
    let expected = b + b;
    let result = jacobian_add_2007_bl_kernel(b, b, "jacobian_add_2007_bl");
    assert!(result == expected);

    let mut rng = thread_rng();
    let base_point = G::generator();
    let s = ScalarField::rand(&mut rng);
    let c = base_point * s;
    let expected = c + c;
    let result = jacobian_add_2007_bl_kernel(c, c, "jacobian_add_2007_bl");
    assert!(expected == result);
}

#[test]
#[serial_test::serial]
pub fn test_jacobian_add_2007_bl() {
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
    let expected = a + b;
    let result = jacobian_add_2007_bl_kernel(a, b, "jacobian_add_2007_bl");
    assert!(result == expected);
}
