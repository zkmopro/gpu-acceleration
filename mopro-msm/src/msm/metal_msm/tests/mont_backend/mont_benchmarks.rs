use crate::msm::metal_msm::host::gpu::{create_buffer, create_empty_buffer, get_default_device};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{
    calc_bitwidth, calc_mont_radix, calc_nsafe, calc_num_limbs, calc_rinv_and_n0,
};
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;
use stopwatch::Stopwatch;

#[test]
#[serial_test::serial]
#[ignore]
pub fn all_benchmarks() {
    let benchmarks_to_run = vec![
        ("mont_mul_modified_benchmarks.metal", "mont_mul_modified"),
        ("mont_mul_optimised_benchmarks.metal", "mont_mul_optimised"),
        ("mont_mul_cios_benchmarks.metal", "mont_mul_cios"),
    ];

    for (shader_file, benchmark_name) in benchmarks_to_run {
        println!("=== benchmarking {} ===", benchmark_name);
        for i in 11..17 {
            match benchmark(i, shader_file) {
                Ok(elapsed) => println!("benchmark for {}-bit limbs took {}ms", i, elapsed),
                Err(e) => println!("benchmark for {}-bit limbs: {}", i, e),
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
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let p_bitwidth = calc_bitwidth(&p);
    let num_limbs = calc_num_limbs(log_limb_size, p_bitwidth);

    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&p);
    let b = rng.gen_biguint_below(&p);

    let nsafe = calc_nsafe(log_limb_size);
    if nsafe == 0 {
        return Err("Benchmark failed: nsafe == 0".to_string());
    }

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;

    let a_r = &a * &r % &p;
    let b_r = &b * &r % &p;

    let cost = 2u32.pow(16u32) as usize;
    let expected = expensive_computation(cost, &a, &b, &p, &r);
    let expected_limbs = BaseField::from_bigint(expected.clone().try_into().unwrap())
        .unwrap()
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    let ar_limbs = BaseField::from_bigint(a_r.clone().try_into().unwrap())
        .unwrap()
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);
    let br_limbs = BaseField::from_bigint(b_r.clone().try_into().unwrap())
        .unwrap()
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    let device = get_default_device();

    let a_buf = create_buffer(&device, &ar_limbs);
    let b_buf = create_buffer(&device, &br_limbs);
    let cost_buf = create_buffer(&device, &vec![cost as u32]);
    let result_buf = create_empty_buffer(&device, num_limbs);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );
    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/mont_backend",
        shader_file,
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("run", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&cost_buf), 0);
    encoder.set_buffer(3, Some(&result_buf), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    let sw = Stopwatch::start_new();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let elapsed = sw.elapsed_ms();

    let ptr = result_buf.contents() as *const u32;
    let result_limbs: Vec<u32>;

    // Check if ptr is not null
    if !ptr.is_null() {
        let len = num_limbs;
        result_limbs = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    } else {
        panic!("Pointer is null");
    }

    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);
    if result == expected.try_into().unwrap() && result_limbs == expected_limbs {
        Ok(elapsed)
    } else {
        Err("Benchmark failed: results do not match expected values".to_string())
    }

    // return elapsed;
}
