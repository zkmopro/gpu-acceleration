use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
use ark_ff::PrimeField;
use ark_std::{rand::thread_rng, UniformRand, Zero};

use crate::msm::metal_msm::tests::common::*;
use crate::msm::metal_msm::utils::data_conversion::{points_from_gpu_buffer, points_to_gpu_buffer};

#[test]
#[serial_test::serial]
fn test_jacobian_dataflow() {
    let log_limb_size = 16;
    let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
    let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

    let config = MetalTestConfig {
        log_limb_size,
        num_limbs,
        shader_file: "curve/jacobian_dataflow_test.metal".to_string(),
        kernel_name: "run".to_string(),
    };

    let mut helper = MetalTestHelper::new();

    let mut rng = thread_rng();
    let base_point = G::generator();
    let points: Vec<G> = (0..10)
        .map(|_| base_point * ScalarField::rand(&mut rng))
        .collect();
    let output_points: Vec<G> = (0..10).map(|_| G::zero()).collect();

    let input_buffer = points_to_gpu_buffer(&points, num_limbs, &helper.device);
    let output_buffer = points_to_gpu_buffer(&output_points, num_limbs, &helper.device);
    helper.buffers.push(input_buffer.clone());
    helper.buffers.push(output_buffer.clone());

    let threads_per_threadgroup = helper.create_thread_group_size(10, 1, 1);
    let threadgroup_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[&input_buffer],
        &[&output_buffer],
        &threads_per_threadgroup,
        &threadgroup_size,
    );

    let result_points: Vec<G> = points_from_gpu_buffer(&output_buffer, num_limbs);
    for i in 0..points.len() {
        assert_eq!(points[i] * ScalarField::from(2 as u64), result_points[i]);
    }

    helper.drop_all_buffers();
}
