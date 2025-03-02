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
    // BN254 parameters
    let log_limb_size = 16;
    let num_limbs = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();

    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (_, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    // We only need one scalar for the kernel call, but we won't test it here
    // So let's just supply zeros for the scalar array
    let scalars = vec![0u32; num_limbs];

    // Generate a point on BN254 for testing
    let point = G::generator().into_affine();
    let x: BigUint = point.x.into_bigint().try_into().unwrap();
    let y: BigUint = point.y.into_bigint().try_into().unwrap();

    // Host-side: compute x_mont, y_mont
    let x_mont = (&x * &r) % &p;
    let y_mont = (&y * &r) % &p;

    // Convert them to ark_ff BigInt<4>, then to limbs
    let x_mont_in_ark: BigInt<4> = x_mont.clone().try_into().unwrap();
    let y_mont_in_ark: BigInt<4> = y_mont.clone().try_into().unwrap();
    let x_mont_in_ark_limbs = x_mont_in_ark.to_limbs(num_limbs, log_limb_size);
    let y_mont_in_ark_limbs = y_mont_in_ark.to_limbs(num_limbs, log_limb_size);

    // Convert unreduced x,y into `num_limbs` “halfword” limbs
    let x_in_ark: BigInt<4> = x.clone().try_into().unwrap();
    let y_in_ark: BigInt<4> = y.clone().try_into().unwrap();
    let x_limb = x_in_ark.to_limbs(num_limbs, log_limb_size);
    let y_limb = y_in_ark.to_limbs(num_limbs, log_limb_size);

    // Helper to pack every pair of 16-bit limbs into one 32-bit word
    let pack_limbs = |limbs: &[u32]| -> Vec<u32> {
        limbs
            .chunks(2)
            .map(|chunk| (chunk[1] << 16) | chunk[0])
            .collect()
    };

    let x_packed = pack_limbs(&x_limb);
    let y_packed = pack_limbs(&y_limb);

    // The coords buffer: x + y, each is 8 packed u32 => total 16
    let coords = [x_packed, y_packed].concat();

    // Setup Metal buffers
    let device = get_default_device();
    let coords_buf = create_buffer(&device, &coords);
    let scalars_buf = create_buffer(&device, &scalars); // all zero
    let input_size_buf = create_buffer(&device, &vec![1u32]); // only 1 point

    // Prepare output buffers for the kernel
    let point_x_buf = create_empty_buffer(&device, num_limbs);
    let point_y_buf = create_empty_buffer(&device, num_limbs);

    // We also need a chunks buffer, but we won't actually check it in this test
    let chunks_buf = create_empty_buffer(&device, num_limbs);

    // Build the pipeline
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    // Compile your metal source
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

    // Match buffer indices to kernel signature
    encoder.set_buffer(0, Some(&coords_buf), 0);
    encoder.set_buffer(1, Some(&scalars_buf), 0);
    encoder.set_buffer(2, Some(&input_size_buf), 0);
    encoder.set_buffer(3, Some(&point_x_buf), 0);
    encoder.set_buffer(4, Some(&point_y_buf), 0);
    encoder.set_buffer(5, Some(&chunks_buf), 0);

    // Dispatch
    let tg_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(tg_count, tg_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back X,Y results and compare
    let x_result = read_buffer(&point_x_buf, num_limbs);
    let y_result = read_buffer(&point_y_buf, num_limbs);

    assert_eq!(x_result, x_mont_in_ark_limbs, "X conversion mismatch");
    assert_eq!(y_result, y_mont_in_ark_limbs, "Y conversion mismatch");
}

#[test]
#[serial_test::serial]
fn test_scalar_decomposition() {
    // BN254 parameters
    let log_limb_size = 16;
    let num_limbs = 16;
    let chunk_size = if BaseField::MODULUS_BIT_SIZE / 32 >= 65536 {
        16
    } else {
        4
    };
    let num_subtasks = (256f32 / chunk_size as f32).ceil() as usize;
    let num_columns = 1 << chunk_size; // 2^chunk_size

    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (_, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    // For scalar test, we can provide dummy coords (X=0, Y=0)
    let coords = vec![0u32; 16]; // 16 zeros

    // Generate a random 254-bit scalar
    let mut rng = thread_rng();
    let scalar = rng.gen_biguint(254);
    let scalar_in_scalarfield = ScalarField::from(scalar.clone());

    // Convert scalar to the same limb format the kernel expects
    let scalars = scalar_in_scalarfield
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    // Setup Metal buffers
    let device = get_default_device();
    let coords_buf = create_buffer(&device, &coords); // dummy
    let scalars_buf = create_buffer(&device, &scalars); // real scalar
    let input_size_buf = create_buffer(&device, &vec![1u32]); // only 1 scalar

    // We'll ignore X,Y outputs, but we must pass them
    let point_x_buf = create_empty_buffer(&device, num_limbs);
    let point_y_buf = create_empty_buffer(&device, num_limbs);

    // The important output: chunk decomposition
    let chunks_buf = create_empty_buffer(&device, num_subtasks);

    // Build the pipeline
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    // Compile your metal source
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

    // Buffer indices to match kernel
    encoder.set_buffer(0, Some(&coords_buf), 0);
    encoder.set_buffer(1, Some(&scalars_buf), 0);
    encoder.set_buffer(2, Some(&input_size_buf), 0);
    encoder.set_buffer(3, Some(&point_x_buf), 0);
    encoder.set_buffer(4, Some(&point_y_buf), 0);
    encoder.set_buffer(5, Some(&chunks_buf), 0);

    let tg_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(tg_count, tg_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back the chunk data from GPU
    let gpu_chunks = read_buffer(&chunks_buf, num_subtasks);

    // Now replicate the GPU logic in Rust:
    // (1) build scalar_bytes[16]
    // (2) extract chunk_size bits
    // (3) fix up the last chunk for BN254 (254 bits)
    // (4) do sign logic => range [-s..s-1], then store +s
    let mut scalar_bytes = vec![0u32; 16];
    for (i, &val) in scalars.iter().enumerate().take(8) {
        let lo = val & 0xFFFF;
        let hi = val >> 16;
        // mirrored indexing like the kernel
        scalar_bytes[15 - (i * 2)] = lo;
        scalar_bytes[15 - (i * 2) - 1] = hi;
    }

    fn extract_word_from_bytes_le_mock(bytes: &[u32], word_idx: u32, chunk_size: u32) -> u32 {
        let start_byte_idx = 15 - ((word_idx * chunk_size + chunk_size) / 16);
        let end_byte_idx = 15 - ((word_idx * chunk_size) / 16);
        let start_byte_offset = (word_idx * chunk_size + chunk_size) % 16;
        let end_byte_offset = (word_idx * chunk_size) % 16;

        let mut mask = 0u32;
        if start_byte_offset > 0 {
            mask = (2 << (start_byte_offset - 1)) - 1;
        }

        if start_byte_idx == end_byte_idx {
            (bytes[start_byte_idx as usize] & mask) >> end_byte_offset
        } else {
            let part1 = (bytes[start_byte_idx as usize] & mask) << (16 - end_byte_offset);
            let part2 = bytes[end_byte_idx as usize] >> end_byte_offset;
            part1 + part2
        }
    }

    // (C) Extract chunk_size-bit words for all but the last chunk
    let mut cpu_chunks = vec![0u32; num_subtasks];
    for i in 0..(num_subtasks - 1) {
        cpu_chunks[i] = extract_word_from_bytes_le_mock(&scalar_bytes, i as u32, chunk_size as u32);
    }

    // (D) Last chunk for 254 bits: top 2 bits are unused
    let shift_254 = ((num_subtasks as u32 * chunk_size as u32 - 254) + 16) - chunk_size as u32;
    cpu_chunks[num_subtasks - 1] = scalar_bytes[0] >> shift_254;

    // (E) Sign logic
    let l = num_columns;
    let s = l / 2;
    let mut carry = 0u32;
    let mut cpu_signed_slices = vec![0i32; num_subtasks];

    for i in 0..num_subtasks {
        let raw_val = (cpu_chunks[i] as i32) + (carry as i32);
        if raw_val >= s as i32 {
            cpu_signed_slices[i] = (l as i32 - raw_val) * -1;
            carry = 1;
        } else {
            cpu_signed_slices[i] = raw_val;
            carry = 0;
        }
    }

    for i in 0..num_subtasks {
        cpu_chunks[i] = (cpu_signed_slices[i] + s as i32) as u32;
    }

    assert_eq!(
        gpu_chunks, cpu_chunks,
        "Scalar decomposition mismatch between GPU and CPU!"
    );
}
