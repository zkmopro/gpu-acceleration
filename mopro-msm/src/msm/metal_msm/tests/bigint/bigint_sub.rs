use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

use ark_ff::{BigInt, BigInteger, UniformRand};
use ark_std::rand;

#[test]
#[serial_test::serial]
pub fn test_bigint_sub_no_underflow() {
    let (a, b, expected) = generate_test_values(false);
    run_bigint_sub_test(a, b, expected);
}

#[test]
#[serial_test::serial]
fn test_bigint_sub_underflow() {
    let (a, b, expected) = generate_test_values(true);
    run_bigint_sub_test(a, b, expected);
}

fn run_bigint_sub_test(a: BigInt<4>, b: BigInt<4>, expected: BigInt<4>) {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "bigint/bigint_sub.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalHelper::new();

    let a_buf = helper.create_input_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let b_buf = helper.create_input_buffer(&b.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_output_buffer(config.num_limbs);

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
    assert_eq!(result_limbs, expected_limbs, "Limb representation mismatch");

    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);
    assert_eq!(result, expected, "BigInt result mismatch");

    helper.drop_all_buffers();
}

fn generate_test_values(require_underflow: bool) -> (BigInt<4>, BigInt<4>, BigInt<4>) {
    let mut rng = rand::thread_rng();

    loop {
        let a = BigInt::<4>::rand(&mut rng);
        let b = BigInt::<4>::rand(&mut rng);

        let mut expected = a.clone();
        let underflow = expected.sub_with_borrow(&b);

        if underflow == require_underflow {
            return (a, b, expected);
        }
    }
}
