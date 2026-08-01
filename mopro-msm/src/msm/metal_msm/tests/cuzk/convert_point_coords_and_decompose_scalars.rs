use ark_bn254::Fr as ScalarField;
use ark_ff::PrimeField;
use num_bigint::RandBigInt;
use rand::thread_rng;

use crate::msm::metal_msm::host::metal_wrapper::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;

#[test]
#[serial_test::serial]
fn test_scalar_decomposition() {
    // Setup test config
    let log_limb_size = 32;
    let num_limbs = 8;
    let window_size = 16;
    let num_subtasks = (256f32 / window_size as f32).ceil() as usize;
    let num_columns = 1 << window_size; // 2^window_size

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/decompose_scalars.metal".to_string(),
        kernel_name: "decompose_scalars".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Generate a random 254-bit scalar
    let mut rng = thread_rng();
    let scalar = rng.gen_biguint(254);
    let scalar_in_scalarfield = ScalarField::from(scalar.clone());

    // Convert scalar to the same limb format the kernel expects
    let scalars = scalar_in_scalarfield
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);
    let packed_scalars = pack_limbs(&scalars);

    // Setup Metal buffers
    let scalars_buf = helper.create_buffer(&packed_scalars);
    let params_buf = helper.create_buffer(&vec![
        1u32,
        window_size as u32,
        num_columns as u32,
        num_subtasks as u32,
    ]);

    let chunks_buf = helper.create_empty_buffer(num_subtasks);

    // Setup thread group sizes
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // Execute the shader
    helper.execute_shader(
        &config,
        &[&scalars_buf, &chunks_buf, &params_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read back the chunk data from GPU
    let gpu_chunks = helper.read_results(&chunks_buf, num_subtasks);

    // Now replicate the GPU logic in Rust:
    // (1) build scalar_bytes[16]
    // (2) extract window_size bits
    // (3) fix up the last chunk for BN254 (254 bits)
    // (4) do sign logic => range [-s..s-1], then store +s
    let mut scalar_bytes = vec![0u32; 16];
    for (i, &val) in packed_scalars.iter().enumerate().take(8) {
        let lo = val & 0xFFFF;
        let hi = val >> 16;
        // mirrored indexing like the kernel
        scalar_bytes[15 - (i * 2)] = lo;
        scalar_bytes[15 - (i * 2) - 1] = hi;
    }

    fn extract_word_from_bytes_le_mock(bytes: &[u32], word_idx: u32, window_size: u32) -> u32 {
        let start_byte_idx = 15 - ((word_idx * window_size + window_size) / 16);
        let end_byte_idx = 15 - ((word_idx * window_size) / 16);
        let start_byte_offset = (word_idx * window_size + window_size) % 16;
        let end_byte_offset = (word_idx * window_size) % 16;

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

    // Calculate expected chunks
    let mut cpu_chunks = vec![0u32; num_subtasks];
    for i in 0..(num_subtasks - 1) {
        cpu_chunks[i] =
            extract_word_from_bytes_le_mock(&scalar_bytes, i as u32, window_size as u32);
    }

    let shift_256 = ((num_subtasks as u32 * window_size as u32 - 256) + 16) - window_size as u32;
    cpu_chunks[num_subtasks - 1] = scalar_bytes[0] >> shift_256;

    // Sign logic
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

    // Clean up resources
    helper.drop_all_buffers();

    // Verify results
    assert_eq!(
        gpu_chunks, cpu_chunks,
        "Scalar decomposition mismatch between GPU and CPU!"
    );
}

/// With 32-bit limbs the GPU buffer layout is simply the little-endian
/// 32-bit words of the value, so packing is the identity.
fn pack_limbs(limbs: &[u32]) -> Vec<u32> {
    limbs.to_vec()
}
