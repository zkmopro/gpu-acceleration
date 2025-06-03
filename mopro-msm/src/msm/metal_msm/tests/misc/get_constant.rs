use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField, Zero};
use num_bigint::BigUint;

const NUM_LIMBS: usize = 16;
const NUM_LIMBS_WIDE: usize = 17;
const LOG_LIMB_SIZE: u32 = 16;
const SHADER_FILE: &str = "misc/test_get_constant.metal";

#[test]
#[serial_test::serial]
pub fn test_get_n0() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_n0".to_string(),
    };

    let mut helper = MetalHelper::new();
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, config.log_limb_size);

    let constants = get_or_calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let expected = BigInt::<4>::from(constants.n0);
    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);

    assert_eq!(result, expected);
    assert_eq!(result_limbs, expected_limbs);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_get_p() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_p".to_string(),
    };

    let mut helper = MetalHelper::new();
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);

    let expected = BaseField::MODULUS;
    let expected_limbs = expected.to_limbs(NUM_LIMBS, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_get_r() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_r".to_string(),
    };

    let mut helper = MetalHelper::new();
    let result_buf = helper.create_empty_buffer(NUM_LIMBS_WIDE);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, NUM_LIMBS_WIDE);
    let result = BigInt::<6>::from_limbs(&result_limbs, config.log_limb_size);

    let constants = get_or_calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let expected: BigInt<6> = constants.r.try_into().unwrap();
    let expected_limbs = expected.to_limbs(NUM_LIMBS_WIDE, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_get_p_wide() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS_WIDE,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_p_wide".to_string(),
    };

    let mut helper = MetalHelper::new();
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);

    let expected = BaseField::MODULUS;
    let expected_limbs = expected.to_limbs(NUM_LIMBS_WIDE, LOG_LIMB_SIZE);

    assert_eq!(result_limbs, expected_limbs);
    assert_eq!(result, expected);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_get_bn254_zero() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_bn254_zero".to_string(),
    };

    let mut helper = MetalHelper::new();
    let x_buf = helper.create_empty_buffer(config.num_limbs);
    let y_buf = helper.create_empty_buffer(config.num_limbs);
    let z_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&x_buf, &y_buf, &z_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let x_limbs = helper.read_results(&x_buf, config.num_limbs);
    let y_limbs = helper.read_results(&y_buf, config.num_limbs);
    let z_limbs = helper.read_results(&z_buf, config.num_limbs);

    let result_x = BigInt::<4>::from_limbs(&x_limbs, config.log_limb_size);
    let result_y = BigInt::<4>::from_limbs(&y_limbs, config.log_limb_size);
    let result_z = BigInt::<4>::from_limbs(&z_limbs, config.log_limb_size);

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::zero();
    assert!(result == expected);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_get_bn254_one() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_bn254_one".to_string(),
    };

    let mut helper = MetalHelper::new();
    let x_buf = helper.create_empty_buffer(config.num_limbs);
    let y_buf = helper.create_empty_buffer(config.num_limbs);
    let z_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&x_buf, &y_buf, &z_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let x_limbs = helper.read_results(&x_buf, config.num_limbs);
    let y_limbs = helper.read_results(&y_buf, config.num_limbs);
    let z_limbs = helper.read_results(&z_buf, config.num_limbs);

    let result_x = BigInt::<4>::from_limbs(&x_limbs, config.log_limb_size);
    let result_y = BigInt::<4>::from_limbs(&y_limbs, config.log_limb_size);
    let result_z = BigInt::<4>::from_limbs(&z_limbs, config.log_limb_size);
    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::generator();
    assert!(result == expected);
}

#[test]
#[serial_test::serial]
pub fn test_get_bn254_zero_mont() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_bn254_zero_mont".to_string(),
    };

    let mut helper = MetalHelper::new();
    let x_buf = helper.create_empty_buffer(config.num_limbs);
    let y_buf = helper.create_empty_buffer(config.num_limbs);
    let z_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&x_buf, &y_buf, &z_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let x_limbs = helper.read_results(&x_buf, config.num_limbs);
    let y_limbs = helper.read_results(&y_buf, config.num_limbs);
    let z_limbs = helper.read_results(&z_buf, config.num_limbs);

    let result_x: BigUint = BigInt::<4>::from_limbs(&x_limbs, config.log_limb_size)
        .try_into()
        .unwrap();
    let result_y: BigUint = BigInt::<4>::from_limbs(&y_limbs, config.log_limb_size)
        .try_into()
        .unwrap();
    let result_z: BigUint = BigInt::<4>::from_limbs(&z_limbs, config.log_limb_size)
        .try_into()
        .unwrap();

    let constants = get_or_calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let result_x = (result_x * &constants.rinv) % &constants.p;
    let result_y = (result_y * &constants.rinv) % &constants.p;
    let result_z = (result_z * &constants.rinv) % &constants.p;

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::zero();
    assert!(result == expected);
}

#[test]
#[serial_test::serial]
pub fn test_get_bn254_one_mont() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_bn254_one_mont".to_string(),
    };

    let mut helper = MetalHelper::new();
    let x_buf = helper.create_empty_buffer(config.num_limbs);
    let y_buf = helper.create_empty_buffer(config.num_limbs);
    let z_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&x_buf, &y_buf, &z_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let x_limbs = helper.read_results(&x_buf, config.num_limbs);
    let y_limbs = helper.read_results(&y_buf, config.num_limbs);
    let z_limbs = helper.read_results(&z_buf, config.num_limbs);

    let result_x: BigUint = BigInt::<4>::from_limbs(&x_limbs, config.log_limb_size)
        .try_into()
        .unwrap();
    let result_y: BigUint = BigInt::<4>::from_limbs(&y_limbs, config.log_limb_size)
        .try_into()
        .unwrap();
    let result_z: BigUint = BigInt::<4>::from_limbs(&z_limbs, config.log_limb_size)
        .try_into()
        .unwrap();

    let constants = get_or_calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let result_x = (result_x * &constants.rinv) % &constants.p;
    let result_y = (result_y * &constants.rinv) % &constants.p;
    let result_z = (result_z * &constants.rinv) % &constants.p;

    let result = G::new(result_x.into(), result_y.into(), result_z.into());
    let expected = G::generator();
    assert!(result == expected);
}

#[test]
#[serial_test::serial]
pub fn test_get_mu() {
    let config = MetalConfig {
        log_limb_size: LOG_LIMB_SIZE,
        num_limbs: NUM_LIMBS,
        shader_file: SHADER_FILE.to_string(),
        kernel_name: "test_get_mu".to_string(),
    };

    let mut helper = MetalHelper::new();

    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, config.log_limb_size);

    let constants = get_or_calc_constants(NUM_LIMBS, LOG_LIMB_SIZE);
    let expected_mu: BigInt<4> = constants.mu.try_into().unwrap();
    let expected_limbs = expected_mu.to_limbs(NUM_LIMBS, LOG_LIMB_SIZE);

    assert_eq!(*result_limbs, expected_limbs);
    assert_eq!(result, expected_mu);
}
