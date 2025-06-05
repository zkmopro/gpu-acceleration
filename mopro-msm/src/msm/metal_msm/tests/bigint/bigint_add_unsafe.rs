use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

use ark_ff::{BigInt, BigInteger, UniformRand};

#[test]
#[serial_test::serial]
pub fn test_bigint_add_unsafe() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "bigint/bigint_add_unsafe.metal".to_string(),
        kernel_name: "test_bigint_add_unsafe".to_string(),
    };

    let mut helper = MetalHelper::new();

    let mut rng = rand::thread_rng();
    let (a, b, expected) = loop {
        let a = BigInt::<4>::rand(&mut rng);
        let b = BigInt::<4>::rand(&mut rng);

        let mut expected = a.clone();
        let overflow = expected.add_with_carry(&b);

        // Break the loop if addition does not overflow
        if !overflow {
            break (a, b, expected);
        }
    };

    let a_buf = helper.create_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let b_buf = helper.create_buffer(&b.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let thread_group_size = helper.create_thread_group_size(1, 1, 1);
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &b_buf, &result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);
    assert_eq!(result_limbs, expected_limbs, "Limb representation mismatch");

    let result: BigInt<4> = BigInt::from_limbs(&result_limbs, config.log_limb_size);
    assert_eq!(result, expected, "BigInt result mismatch");

    helper.drop_all_buffers();
}
