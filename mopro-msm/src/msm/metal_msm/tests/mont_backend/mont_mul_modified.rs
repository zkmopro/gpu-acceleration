// adapted from: https://github.com/geometryxyz/msl-secp256k1

use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::Fq as BaseField;
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

#[test]
#[serial_test::serial]
pub fn test_mont_mul_15() {
    do_test(15);
}

pub fn do_test(log_limb_size: u32) {
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let nsafe = calc_nsafe(log_limb_size);

    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;

    let mut rng = thread_rng();
    let a = rng.gen_biguint_below(&p);
    let b = rng.gen_biguint_below(&p);

    let a_r = &a * &r % &p;
    let b_r = &b * &r % &p;
    let expected = (&a * &b * &r) % &p;

    let a_r_in_ark = BaseField::from_bigint(a_r.clone().try_into().unwrap()).unwrap();
    let b_r_in_ark = BaseField::from_bigint(b_r.clone().try_into().unwrap()).unwrap();
    let expected_in_ark = BaseField::from_bigint(expected.clone().try_into().unwrap()).unwrap();
    let expected_limbs = expected_in_ark
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    let device = get_default_device();
    let a_buf = create_buffer(
        &device,
        &a_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size),
    );
    let b_buf = create_buffer(
        &device,
        &b_r_in_ark.into_bigint().to_limbs(num_limbs, log_limb_size),
    );
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
        "mont_mul_modified.metal",
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
    encoder.set_buffer(2, Some(&result_buf), 0);

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

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&result_buf, num_limbs);
    let result = BigInt::<4>::from_limbs(&result_limbs, log_limb_size);

    assert!(result == expected.try_into().unwrap());
    assert!(result_limbs == expected_limbs);
}
