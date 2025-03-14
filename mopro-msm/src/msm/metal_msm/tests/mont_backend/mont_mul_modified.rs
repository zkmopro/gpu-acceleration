use crate::msm::metal_msm::tests::common::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use num_bigint::RandBigInt;
use rand::thread_rng;

#[test]
#[serial_test::serial]
pub fn test_mont_mul_15() {
    do_test(15);
}

pub fn do_test(log_limb_size: u32) {
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let config = MetalTestConfig {
        log_limb_size,
        num_limbs,
        shader_file: "mont_backend/mont_mul_modified.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalTestHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);

    // Generate random values
    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&constants.p);
    let b = rng.gen_biguint_below(&constants.p);

    // Convert to Montgomery domain
    let a_r = (&a * &constants.r) % &constants.p;
    let b_r = (&b * &constants.r) % &constants.p;
    let expected = (&a * &b * &constants.r) % &constants.p;

    // Convert to Arkworks types
    let a_r_in_ark = BaseField::from_bigint(a_r.clone().try_into().unwrap()).unwrap();
    let b_r_in_ark = BaseField::from_bigint(b_r.clone().try_into().unwrap()).unwrap();
    let expected_in_ark = BaseField::from_bigint(expected.clone().try_into().unwrap()).unwrap();
    let expected_limbs = expected_in_ark
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    let a_buf =
        helper.create_input_buffer(&a_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size));
    let b_buf =
        helper.create_input_buffer(&b_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size));
    let result_buf = helper.create_output_buffer(num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &b_buf],
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    assert!(result == expected.try_into().unwrap());
    assert!(result_limbs == expected_limbs);

    helper.drop_all_buffers();
}
