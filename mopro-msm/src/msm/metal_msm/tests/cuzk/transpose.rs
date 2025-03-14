use crate::msm::metal_msm::tests::common::*;
use rand::Rng;

#[test]
#[serial_test::serial]
fn test_sparse_matrix_transposition() {
    let config = MetalTestConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalTestHelper::new();
    let mut rng = rand::thread_rng();
    const NUM_TESTS: usize = 10;
    const MAX_SUBTASKS: usize = 4;
    const MAX_COLS: u32 = 8;
    const MAX_INPUT_SIZE: u32 = 100;

    for _ in 0..NUM_TESTS {
        let num_subtasks = rng.gen_range(1..=MAX_SUBTASKS);
        let n = rng.gen_range(2..=MAX_COLS);
        let input_size = rng.gen_range(1..=MAX_INPUT_SIZE);

        // Generate test data
        let mut all_csr_col_idx = Vec::new();
        let mut expected_results = Vec::new();

        for _ in 0..num_subtasks {
            let cols = (0..input_size)
                .map(|_| rng.gen_range(0..n))
                .collect::<Vec<u32>>();
            let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&cols, n);
            expected_results.push((expected_col_ptr, expected_val_idxs));
            all_csr_col_idx.extend_from_slice(&cols);
        }

        // Create buffers
        let all_csr_col_idx_buf = helper.create_input_buffer(&all_csr_col_idx);
        let all_csc_col_ptr_buf = helper.create_output_buffer(num_subtasks * (n as usize + 1));
        let all_csc_val_idxs_buf = helper.create_output_buffer(num_subtasks * input_size as usize);
        let all_curr_buf = helper.create_output_buffer(num_subtasks * n as usize);
        let params_buf = helper.create_input_buffer(&vec![n, input_size]);

        // Execute shader
        let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
        let thread_group_size = helper.create_thread_group_size(1, 1, 1);

        helper.execute_shader(
            &config,
            &[&all_csr_col_idx_buf, &params_buf],
            &[&all_csc_col_ptr_buf, &all_csc_val_idxs_buf, &all_curr_buf],
            &thread_group_count,
            &thread_group_size,
        );

        // Read results
        let csc_col_ptr =
            helper.read_results(&all_csc_col_ptr_buf, num_subtasks * (n as usize + 1));
        let csc_val_idxs =
            helper.read_results(&all_csc_val_idxs_buf, num_subtasks * input_size as usize);

        // Validation (same as before)
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
        helper.drop_all_buffers();
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
