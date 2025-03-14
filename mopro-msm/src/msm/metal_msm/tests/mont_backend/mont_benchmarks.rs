use crate::msm::metal_msm::tests::common::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;
use stopwatch::Stopwatch;

#[test]
#[serial_test::serial]
#[ignore]
pub fn all_benchmarks() {
    let benchmarks_to_run = vec![
        (
            "mont_backend/mont_mul_modified_benchmarks.metal",
            "mont_mul_modified",
        ),
        (
            "mont_backend/mont_mul_optimised_benchmarks.metal",
            "mont_mul_optimised",
        ),
        (
            "mont_backend/mont_mul_cios_benchmarks.metal",
            "mont_mul_cios",
        ),
    ];

    for (shader_file, kernel_name) in benchmarks_to_run {
        println!("=== benchmarking {} ===", kernel_name);
        for log_limb_size in 11..17 {
            match benchmark(log_limb_size, shader_file) {
                Ok(elapsed) => println!(
                    "benchmark for {}-bit limbs took {}ms",
                    log_limb_size, elapsed
                ),
                Err(e) => println!("benchmark for {}-bit limbs: {}", log_limb_size, e),
            }
        }
        println!();
    }
}

fn expensive_computation(
    cost: usize,
    a: &BigUint,
    b: &BigUint,
    p: &BigUint,
    r: &BigUint,
) -> BigUint {
    let mut c = (a * a) % p;
    for _ in 1..cost {
        c = (c * a) % p;
    }
    (c * b * r) % p
}

pub fn benchmark(log_limb_size: u32, shader_file: &str) -> Result<i64, String> {
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let config = MetalTestConfig {
        log_limb_size,
        num_limbs,
        shader_file: shader_file.to_string(),
        kernel_name: "run".to_string(),
    };

    // Get constants for this configuration
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let nsafe = constants.nsafe;
    if nsafe == 0 {
        return Err("Benchmark failed: nsafe == 0".to_string());
    }

    // Generate random values
    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&constants.p);
    let b = rng.gen_biguint_below(&constants.p);

    // Convert to Montgomery domain
    let a_r = (&a * &constants.r) % &constants.p;
    let b_r = (&b * &constants.r) % &constants.p;

    // Calculate expected result
    let cost = 2u32.pow(16u32) as usize;
    let expected = expensive_computation(cost, &a, &b, &constants.p, &constants.r);

    // Convert to Arkworks types
    let a_r_in_ark = BaseField::from_bigint(a_r.clone().try_into().unwrap()).unwrap();
    let b_r_in_ark = BaseField::from_bigint(b_r.clone().try_into().unwrap()).unwrap();
    let expected_in_ark = BaseField::from_bigint(expected.clone().try_into().unwrap()).unwrap();
    let expected_limbs = expected_in_ark
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    // Setup Metal helper
    let mut helper = MetalTestHelper::new();

    // Create buffers
    let a_buf =
        helper.create_input_buffer(&a_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size));
    let b_buf =
        helper.create_input_buffer(&b_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size));
    let cost_buf = helper.create_input_buffer(&vec![cost as u32]);
    let result_buf = helper.create_output_buffer(num_limbs);

    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // Time the execution
    let sw = Stopwatch::start_new();

    helper.execute_shader(
        &config,
        &[&a_buf, &b_buf, &cost_buf],
        &[&result_buf],
        &thread_group_count,
        &thread_group_size,
    );

    let elapsed = sw.elapsed_ms();

    // Read and verify results
    let result_limbs = helper.read_results(&result_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    helper.drop_all_buffers();

    if result == expected.try_into().unwrap() && result_limbs == expected_limbs {
        Ok(elapsed)
    } else {
        Err("Benchmark failed: results do not match expected values".to_string())
    }
}
