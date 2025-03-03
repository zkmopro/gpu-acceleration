use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use metal::*;
use rand::Rng;

#[test]
#[serial_test::serial]
fn test_sparse_matrix_transposition() {
    let device = get_default_device();

    // Random test parameters
    let mut rng = rand::thread_rng();
    const NUM_TESTS: usize = 10; // Number of random test cases
    const MAX_SUBTASKS: usize = 4;
    const MAX_COLS: u32 = 8;
    const MAX_INPUT_SIZE: u32 = 100;

    for _ in 0..NUM_TESTS {
        let num_subtasks = rng.gen_range(1..=MAX_SUBTASKS);
        let n = rng.gen_range(2..=MAX_COLS);
        let input_size = rng.gen_range(1..=MAX_INPUT_SIZE);

        // Generate random CSR data
        let mut all_csr_col_idx = Vec::new();
        let mut expected_results = Vec::new();

        for _ in 0..num_subtasks {
            // Generate random column indices for this subtask
            let cols = (0..input_size)
                .map(|_| rng.gen_range(0..n))
                .collect::<Vec<u32>>();

            // Compute expected CSC results on CPU
            let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&cols, n);
            expected_results.push((expected_col_ptr, expected_val_idxs));

            // Add to global CSR buffer
            all_csr_col_idx.extend_from_slice(&cols);
        }

        // Create Metal buffers
        let all_csr_col_idx_buf = create_buffer(&device, &all_csr_col_idx);
        let all_csc_col_ptr_buf = create_empty_buffer(&device, num_subtasks * (n as usize + 1));
        let all_csc_val_idxs_buf = create_empty_buffer(&device, num_subtasks * input_size as usize);
        let all_curr_buf = create_empty_buffer(&device, num_subtasks * n as usize);

        let params = vec![n, input_size];
        let params_buf = create_buffer(&device, &params);

        // Compile and setup pipeline (same as before)
        write_constants("../mopro-msm/src/msm/metal_msm/shader", 16, 16, 0, 0);
        let library_path = compile_metal(
            "../mopro-msm/src/msm/metal_msm/shader/cuzk",
            "transpose.metal",
        );
        let library = device.new_library_with_file(library_path).unwrap();
        let kernel = library.get_function("transpose", None).unwrap();

        // Build the pipeline
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        let pipeline_state_descriptor = ComputePipelineDescriptor::new();
        pipeline_state_descriptor.set_compute_function(Some(&kernel));
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(
                pipeline_state_descriptor.compute_function().unwrap(),
            )
            .unwrap();

        encoder.set_compute_pipeline_state(&pipeline_state);

        encoder.set_buffer(0, Some(&all_csr_col_idx_buf), 0);
        encoder.set_buffer(1, Some(&all_csc_col_ptr_buf), 0);
        encoder.set_buffer(2, Some(&all_csc_val_idxs_buf), 0);
        encoder.set_buffer(3, Some(&all_curr_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        encoder.dispatch_threads(
            MTLSize::new(num_subtasks as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Verify results
        let csc_col_ptr: Vec<u32> =
            read_buffer(&all_csc_col_ptr_buf, num_subtasks * (n as usize + 1));
        let csc_val_idxs: Vec<u32> =
            read_buffer(&all_csc_val_idxs_buf, num_subtasks * input_size as usize);

        // Validate each subtask
        for subtask in 0..num_subtasks {
            let offset = subtask * (n as usize + 1);
            let expected_col_ptr = &expected_results[subtask].0;
            let actual_col_ptr = &csc_col_ptr[offset..offset + (n as usize + 1)];

            assert_eq!(
                actual_col_ptr, expected_col_ptr,
                "Subtask {}: Column pointers mismatch\nExpected: {:?}\nActual: {:?}",
                subtask, expected_col_ptr, actual_col_ptr
            );

            let val_offset = subtask * input_size as usize;
            let expected_vals = &expected_results[subtask].1;
            let actual_vals = &csc_val_idxs[val_offset..val_offset + input_size as usize];

            assert_eq!(
                actual_vals, expected_vals,
                "Subtask {}: Value indices mismatch\nExpected: {:?}\nActual: {:?}",
                subtask, expected_vals, actual_vals
            );
        }
    }
}

fn compute_expected_csc(csr_cols: &[u32], n: u32) -> (Vec<u32>, Vec<u32>) {
    // Phase 1: Count column occurrences
    let mut col_counts = vec![0; n as usize + 1];
    for &col in csr_cols {
        col_counts[col as usize + 1] += 1;
    }

    // Phase 2: Prefix sum
    for i in 1..col_counts.len() {
        col_counts[i] += col_counts[i - 1];
    }

    // Phase 3: Build value indices
    let mut curr = vec![0; n as usize];
    let mut val_idxs = vec![0; csr_cols.len()];

    for (val_idx, &col) in csr_cols.iter().enumerate() {
        let pos = col_counts[col as usize] + curr[col as usize];
        val_idxs[pos as usize] = val_idx as u32;
        curr[col as usize] += 1;
    }

    (col_counts, val_idxs)
}
