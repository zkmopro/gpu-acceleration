use crate::msm::metal_msm::host::gpu::get_default_device;
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::data_conversion::{points_from_gpu_buffer, points_to_gpu_buffer};
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
use ark_ff::PrimeField;
use ark_std::{rand::thread_rng, UniformRand, Zero};
use metal::*;
use num_bigint::BigUint;

#[test]
#[serial_test::serial]
fn test_jacobian_dataflow() {
    let log_limb_size = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let res = calc_rinv_and_n0(&p, &r, log_limb_size);
    let n0 = res.1;
    let nsafe = calc_nsafe(log_limb_size);

    let mut rng = thread_rng();
    let base_point = G::generator();
    let points: Vec<G> = (0..10)
        .map(|_| base_point * ScalarField::rand(&mut rng))
        .collect();
    let output_points: Vec<G> = (0..10).map(|_| G::zero()).collect();
    let device = get_default_device();
    let input_buffer = points_to_gpu_buffer(&points, num_limbs, &device);
    let output_buffer = points_to_gpu_buffer(&output_points, num_limbs, &device);

    let command_queue = device.new_command_queue();

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/curve",
        &format!("{}.metal", "jacobian_dataflow_test"),
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

    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);
    encoder.set_compute_pipeline_state(&pipeline_state);

    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);

    let threads_per_threadgroup = MTLSize::new(10, 1, 1);
    let threadgroup_size = MTLSize::new(1, 1, 1);
    encoder.dispatch_threads(threads_per_threadgroup, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_points: Vec<G> = points_from_gpu_buffer(&output_buffer, num_limbs);
    for i in 0..points.len() {
        assert_eq!(points[i] * ScalarField::from(2 as u64), result_points[i]);
    }
}
