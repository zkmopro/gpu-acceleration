use crate::msm::metal_msm::utils::metal_wrapper::*;
use rand::Rng;

#[test]
#[serial_test::serial]
fn test_sparse_matrix_transposition() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();
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
        let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
        let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
        let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
        let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
        let params_buf = helper.create_buffer(&vec![n, input_size]);

        // Execute shader
        let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
        let thread_group_size = helper.create_thread_group_size(1, 1, 1);

        helper.execute_shader(
            &config,
            &[
                &all_csr_col_idx_buf,
                &all_csc_col_ptr_buf,
                &all_csc_val_idxs_buf,
                &all_curr_buf,
                &params_buf,
            ],
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

#[test]
#[serial_test::serial]
fn test_transpose_single_element() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with single element in single column
    let n = 3u32;
    let input_size = 1u32;
    let num_subtasks = 1;

    let all_csr_col_idx = vec![1u32]; // Single element in column 1
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Single element column pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Single element value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_all_same_column() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with all elements in the same column
    let n = 4u32;
    let input_size = 5u32;
    let num_subtasks = 1;

    let all_csr_col_idx = vec![2u32; input_size as usize]; // All elements in column 2
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Same column pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Same column value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_sequential_columns() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with elements in sequential columns (0, 1, 2, 3, ...)
    let n = 5u32;
    let input_size = 5u32;
    let num_subtasks = 1;

    let all_csr_col_idx: Vec<u32> = (0..input_size).collect(); // [0, 1, 2, 3, 4]
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Sequential column pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Sequential column value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_reverse_order() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with elements in reverse order (n-1, n-2, ..., 1, 0)
    let n = 4u32;
    let input_size = 4u32;
    let num_subtasks = 1;

    let all_csr_col_idx: Vec<u32> = (0..input_size).rev().collect(); // [3, 2, 1, 0]
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Reverse order column pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Reverse order value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_empty_columns() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with some empty columns (only use columns 0 and 3, leaving 1 and 2 empty)
    let n = 5u32;
    let input_size = 6u32;
    let num_subtasks = 1;

    let all_csr_col_idx = vec![0u32, 0u32, 3u32, 3u32, 0u32, 4u32]; // Columns 1 and 2 are empty
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Empty columns pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Empty columns value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_multiple_subtasks_edge_cases() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with multiple subtasks having different patterns
    let n = 3u32;
    let input_size = 4u32;
    let num_subtasks = 3;

    let mut all_csr_col_idx = Vec::new();
    let mut expected_results = Vec::new();

    // Subtask 0: All elements in column 0
    let cols0 = vec![0u32; input_size as usize];
    let (expected_col_ptr0, expected_val_idxs0) = compute_expected_csc(&cols0, n);
    expected_results.push((expected_col_ptr0, expected_val_idxs0));
    all_csr_col_idx.extend_from_slice(&cols0);

    // Subtask 1: Sequential pattern
    let cols1: Vec<u32> = vec![0, 1, 2, 1];
    let (expected_col_ptr1, expected_val_idxs1) = compute_expected_csc(&cols1, n);
    expected_results.push((expected_col_ptr1, expected_val_idxs1));
    all_csr_col_idx.extend_from_slice(&cols1);

    // Subtask 2: Reverse pattern
    let cols2: Vec<u32> = vec![2, 2, 1, 0];
    let (expected_col_ptr2, expected_val_idxs2) = compute_expected_csc(&cols2, n);
    expected_results.push((expected_col_ptr2, expected_val_idxs2));
    all_csr_col_idx.extend_from_slice(&cols2);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, num_subtasks * (n as usize + 1));
    let csc_val_idxs =
        helper.read_results(&all_csc_val_idxs_buf, num_subtasks * input_size as usize);

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

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_boundary_columns() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with elements only in first and last columns
    let n = 5u32;
    let input_size = 6u32;
    let num_subtasks = 1;

    let all_csr_col_idx = vec![0u32, 4u32, 0u32, 4u32, 0u32, 4u32]; // Only columns 0 and 4
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&all_csr_col_idx, n);

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read and validate results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, n as usize + 1);
    let csc_val_idxs = helper.read_results(&all_csc_val_idxs_buf, input_size as usize);

    assert_eq!(
        csc_col_ptr, expected_col_ptr,
        "Boundary columns pointers mismatch"
    );
    assert_eq!(
        csc_val_idxs, expected_val_idxs,
        "Boundary columns value indices mismatch"
    );

    helper.drop_all_buffers();
}

#[test]
#[serial_test::serial]
fn test_transpose_large_scale() {
    let config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    let mut helper = MetalHelper::new();

    // Test with larger scale to stress test the shader
    let n = 100u32;
    let input_size = 1000u32;
    let num_subtasks = 8;

    let mut all_csr_col_idx = Vec::new();
    let mut expected_results = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..num_subtasks {
        let cols = (0..input_size)
            .map(|_| rng.gen_range(0..n))
            .collect::<Vec<u32>>();
        let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&cols, n);
        expected_results.push((expected_col_ptr, expected_val_idxs));
        all_csr_col_idx.extend_from_slice(&cols);
    }

    // Create buffers
    let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
    let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
    let all_csc_val_idxs_buf = helper.create_empty_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
    let params_buf = helper.create_buffer(&vec![n, input_size]);

    // Execute shader
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &all_csr_col_idx_buf,
            &all_csc_col_ptr_buf,
            &all_csc_val_idxs_buf,
            &all_curr_buf,
            &params_buf,
        ],
        &thread_group_count,
        &thread_group_size,
    );

    // Read results
    let csc_col_ptr = helper.read_results(&all_csc_col_ptr_buf, num_subtasks * (n as usize + 1));
    let csc_val_idxs =
        helper.read_results(&all_csc_val_idxs_buf, num_subtasks * input_size as usize);

    // Validate each subtask
    for subtask in 0..num_subtasks {
        let offset = subtask * (n as usize + 1);
        let expected_col_ptr = &expected_results[subtask].0;
        let actual_col_ptr = &csc_col_ptr[offset..offset + (n as usize + 1)];

        assert_eq!(
            actual_col_ptr, expected_col_ptr,
            "Large scale subtask {}: Column pointers mismatch",
            subtask
        );

        let val_offset = subtask * input_size as usize;
        let expected_vals = &expected_results[subtask].1;
        let actual_vals = &csc_val_idxs[val_offset..val_offset + input_size as usize];

        assert_eq!(
            actual_vals, expected_vals,
            "Large scale subtask {}: Value indices mismatch",
            subtask
        );
    }

    helper.drop_all_buffers();
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
