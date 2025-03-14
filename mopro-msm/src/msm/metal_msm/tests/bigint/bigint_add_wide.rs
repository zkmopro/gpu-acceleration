use crate::msm::metal_msm::tests::common::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

use ark_ff::{BigInt, BigInteger, UniformRand};
use ark_std::rand;

#[test]
#[serial_test::serial]
pub fn test_bigint_add_no_overflow() {
    let (a, b, expected) = generate_test_values(false);
    run_bigint_add_test(&a, &b, &expected);
}

#[test]
#[serial_test::serial]
pub fn test_bigint_add_overflow() {
    let (a, b, expected) = generate_test_values(true);
    run_bigint_add_test(&a, &b, &expected);
}

fn generate_test_values(require_overflow: bool) -> (BigInt<4>, BigInt<4>, BigInt<4>) {
    let mut rng = rand::thread_rng();
    loop {
        let a = BigInt::<4>::rand(&mut rng);
        let b = BigInt::<4>::rand(&mut rng);

        let mut expected = a.clone();
        let overflow = expected.add_with_carry(&b);

        if overflow == require_overflow {
            return (a, b, expected);
        }
    }
}

fn run_bigint_add_test(a: &BigInt<4>, b: &BigInt<4>, expected: &BigInt<4>) {
    let config = MetalTestConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "bigint/bigint_add_wide.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalTestHelper::new();

    let a_buf = helper.create_input_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let b_buf = helper.create_input_buffer(&b.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_output_buffer(config.num_limbs + 1);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &b_buf],
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);
    assert_eq!(result_limbs, expected_limbs);

    let result: BigInt<4> = BigInt::from_limbs(&result_limbs, config.log_limb_size);
    assert_eq!(result, *expected);

    helper.drop_all_buffers();
}
