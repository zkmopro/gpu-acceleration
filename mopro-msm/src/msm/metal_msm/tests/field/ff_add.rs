use crate::msm::metal_msm::tests::common::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use ark_std::rand;

#[test]
#[serial_test::serial]
pub fn test_ff_add() {
    let config = MetalTestConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "field/ff_add.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalTestHelper::new();

    let p = BaseField::MODULUS;

    let mut rng = rand::thread_rng();
    let mut a = BigInt::<4>::rand(&mut rng);
    let mut b = BigInt::<4>::rand(&mut rng);

    // a % p
    while a >= p {
        a.sub_with_borrow(&p);
    }

    while b >= p {
        b.sub_with_borrow(&p);
    }

    // Ensure a and b are non-negative and less than p
    assert!(a >= BigInt::from(0u64), "a must be non-negative");
    assert!(b >= BigInt::from(0u64), "b must be non-negative");
    assert!(a < p, "a must be less than p");
    assert!(b < p, "b must be less than p");

    let a_buf = helper.create_input_buffer(&a.to_limbs(config.num_limbs, config.log_limb_size));
    let b_buf = helper.create_input_buffer(&b.to_limbs(config.num_limbs, config.log_limb_size));
    let result_buf = helper.create_output_buffer(config.num_limbs);

    // Calculate expected result: (a + b) % p
    let mut expected = a.clone();
    expected.add_with_carry(&b);

    // While result >= p, subtract p
    while expected >= p {
        expected.sub_with_borrow(&p);
    }

    assert!(
        expected >= BigInt::from(0u64),
        "expected must be non-negative"
    );
    assert!(expected < p, "expected must be less than p");

    // Ensure the operation is correct using Arkworks
    let a_field = BaseField::from(a);
    let b_field = BaseField::from(b);
    let expected_field = a_field + b_field;
    assert!(expected_field == expected.into());

    let expected_limbs = expected.to_limbs(config.num_limbs, config.log_limb_size);

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
    let result = BigInt::from_limbs(&result_limbs, config.log_limb_size);

    assert!(result == expected);
    assert!(result_limbs == expected_limbs);

    helper.drop_all_buffers();
}
