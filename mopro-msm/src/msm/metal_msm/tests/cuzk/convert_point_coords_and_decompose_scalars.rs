use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::{CurveGroup, Group};
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

#[test]
#[serial_test::serial]
fn test_point_coords_conversion() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (_, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    // Generate valid test data (BN254 base field element)
    let mut rng = thread_rng();
    let scalar = rng.gen_biguint(256);
    let scalar_in_scalarfield = ScalarField::from(scalar.clone());
    let point = G::generator().into_affine();

    let x: BigUint = point.x.into_bigint().try_into().unwrap();
    let y: BigUint = point.y.into_bigint().try_into().unwrap();

    let x_mont = (&x * &r) % &p;
    let y_mont = (&y * &r) % &p;

    let x_mont_in_ark: BigInt<4> = x_mont.clone().try_into().unwrap();
    let y_mont_in_ark: BigInt<4> = y_mont.clone().try_into().unwrap();

    let x_mont_in_ark_limbs = x_mont_in_ark.to_limbs(num_limbs, log_limb_size);
    let y_mont_in_ark_limbs = y_mont_in_ark.to_limbs(num_limbs, log_limb_size);
    let x_in_ark: BigInt<4> = x.clone().try_into().unwrap();
    let y_in_ark: BigInt<4> = y.clone().try_into().unwrap();
    let x_limb = x_in_ark.to_limbs(num_limbs, log_limb_size);
    let y_limb = y_in_ark.to_limbs(num_limbs, log_limb_size);

    // Convert coordinates to packed 32-bit limbs (2x16-bit)
    let pack_limbs = |limbs: Vec<u32>| -> Vec<u32> {
        limbs
            .chunks(2)
            .map(|chunk| (chunk[1] << 16) | chunk[0])
            .collect()
    };

    let x_packed = pack_limbs(x_limb);
    let y_packed = pack_limbs(y_limb);

    // Create input buffers with valid data (8 packed u32s per coordinate)
    let coords = [x_packed, y_packed].concat();
    let mut scalars = scalar_in_scalarfield
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);
    scalars.reverse();

    let device = get_default_device();
    let coords_buf = create_buffer(&device, &coords);
    let scalars_buf = create_buffer(&device, &scalars);
    let input_size_buf = create_buffer(&device, &vec![1u32]); // only one point

    // Create output buffers
    let point_x_buf = create_empty_buffer(&device, num_limbs);
    let point_y_buf = create_empty_buffer(&device, num_limbs);
    let chunk_size = if BaseField::MODULUS_BIT_SIZE / 32 >= 65536 {
        16
    } else {
        4
    };
    let num_subtasks = (256f32 / chunk_size as f32).ceil() as usize;
    let chunks_buf = create_empty_buffer(&device, num_subtasks); // Buffer for scalar chunks
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
        "../mopro-msm/src/msm/metal_msm/shader/cuzk",
        "convert_point_coords_and_decompose_scalars.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library
        .get_function("convert_point_coords_and_decompose_scalars", None)
        .unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&coords_buf), 0);
    encoder.set_buffer(1, Some(&scalars_buf), 0);
    encoder.set_buffer(2, Some(&input_size_buf), 0);
    encoder.set_buffer(3, Some(&point_x_buf), 0);
    encoder.set_buffer(4, Some(&point_y_buf), 0);
    encoder.set_buffer(5, Some(&chunks_buf), 0);

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

    let x_result = read_buffer(&point_x_buf, num_limbs);
    let y_result = read_buffer(&point_y_buf, num_limbs);

    assert_eq!(
        x_result, x_mont_in_ark_limbs,
        "X coordinate conversion failed"
    );
    assert_eq!(
        y_result, y_mont_in_ark_limbs,
        "Y coordinate conversion failed"
    );
}
