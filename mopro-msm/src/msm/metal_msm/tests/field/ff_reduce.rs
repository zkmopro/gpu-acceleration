use crate::msm::metal_msm::host::metal_wrapper::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use ark_std::rand;

#[test]
#[serial_test::serial]
pub fn test_ff_reduce_a_less_than_p() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "field/ff_reduce.metal".to_string(),
        kernel_name: "test_ff_reduce".to_string(),
    };

    let mut helper = MetalHelper::new();

    let (a, expected) = generate_test_values(false);

    let a_buf = helper.create_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);

    assert_eq!(result, expected);
    assert_eq!(result_limbs, expected_limbs);

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
pub fn test_ff_reduce_a_greater_than_p_less_than_2p() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "field/ff_reduce.metal".to_string(),
        kernel_name: "test_ff_reduce".to_string(),
    };

    let mut helper = MetalHelper::new();

    let (a, expected) = generate_test_values(true);

    let a_buf = helper.create_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_empty_buffer(config.num_limbs);

    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&a_buf, &result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let result_limbs = helper.read_results(&result_buf, config.num_limbs);
    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);

    assert_eq!(result, expected);
    assert_eq!(result_limbs, expected_limbs);

    helper.drop_all_buffers();
}

fn generate_test_values(in_range_p_to_2p: bool) -> (BigInt<4>, BigInt<4>) {
    let p = BaseField::MODULUS;

    let mut rng = rand::thread_rng();
    let raw_a = BigInt::<4>::rand(&mut rng);
    assert!(raw_a >= BigInt::from(0u64), "a must be non-negative");

    // a % p
    let mut a = raw_a.clone();
    while a >= p {
        a.sub_with_borrow(&p);
    }

    // At this point, a is in range [0, p)
    let expected = a.clone();

    // add p to a if we want to test the range [p, 2p)
    if in_range_p_to_2p {
        a.add_with_carry(&p);
        let mut two_p = p.clone();
        two_p.add_with_carry(&p);

        assert!(a >= p, "a must be greater than or equal to p");
        assert!(a < two_p, "a must be less than 2p");
    } else {
        assert!(a >= BigInt::from(0u64), "a must be non-negative");
        assert!(a < p, "a must be less than p");
    }

    (a, expected)
}
