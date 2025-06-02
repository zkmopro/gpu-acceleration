use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use num_bigint::RandBigInt;
use rand::thread_rng;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

#[test]
#[serial_test::serial]
pub fn test_barrett_reduce_with_mont_params() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let num_limbs_extra_wide = num_limbs * 2; // maximum 512 bits

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/kernel_barrett_reduction.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p = &constants.p;
    let r = &constants.r;

    // Generate test data
    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(p);
    let mont_a = &a * r;
    let expected = &mont_a % p;
    let expected_in_ark: BigInt<4> = expected.clone().try_into().unwrap();

    // Convert to limbs
    let mont_a_in_ark: BigInt<8> = mont_a.clone().try_into().unwrap();
    let mont_a_limbs = mont_a_in_ark.to_limbs(num_limbs_extra_wide, log_limb_size);

    // Create buffers
    let mont_a_buf = helper.create_buffer(&mont_a_limbs);
    let result_buf = helper.create_empty_buffer(num_limbs);

    // Setup thread group sizes
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&mont_a_buf, &result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let result_limbs = helper.read_results(&result_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    assert_eq!(result, expected_in_ark);
    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_field_mul_with_mont_params() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let num_limbs_wide = num_limbs + 1;

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/kernel_field_mul.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p = &constants.p;
    let r = &constants.r;

    // Generate test data
    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(p);
    let expected = &a * r % p;

    // Convert to limbs
    let a_in_ark: BigInt<4> = a.clone().try_into().unwrap();
    let r_in_ark: BigInt<6> = r.clone().try_into().unwrap(); // r has 257 bits when 16-bit limbs are used for BN254
    let a_limbs = a_in_ark.to_limbs(num_limbs_wide, log_limb_size);
    let r_limbs = r_in_ark.to_limbs(num_limbs_wide, log_limb_size);

    // Create buffers
    let a_buf = helper.create_buffer(&a_limbs);
    let r_buf = helper.create_buffer(&r_limbs);
    let res_buf = helper.create_empty_buffer(num_limbs);

    // Setup thread group sizes
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &r_buf, &res_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let result_limbs = helper.read_results(&res_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    assert_eq!(
        result,
        expected.clone().try_into().unwrap(),
        "result is not equal to expected"
    );

    // verify correctness by restoring expected value using Arkworks (Montgomery form)
    let a_in_field = BaseField::from_bigint(a_in_ark).unwrap();
    assert_eq!(
        result, a_in_field.0,
        "result is not equal to arkworks result in Montgomery form"
    );

    helper.drop_all_buffers();
}
