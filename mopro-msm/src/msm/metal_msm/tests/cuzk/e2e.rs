use std::error::Error;

use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::{CurveGroup, Group}; // for generator(), etc.
use ark_ff::{BigInt, PrimeField};
use ark_std::{UniformRand, Zero};
use num_bigint::BigUint;
use rand::thread_rng;

use crate::msm::metal_msm::tests::cuzk::transpose::compute_expected_csc;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;
// We'll need montgomery parameters – we use some utility functions for that.
use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;

/// Helper function to pack an array of 16-bit limbs (u32 values assumed to be 16-bit)
/// into 32-bit words.
fn pack_limbs(limbs: &[u32]) -> Vec<u32> {
    limbs
        .chunks(2)
        .map(|chunk| {
            // In our tests each chunk has exactly two limbs.
            (chunk[1] << 16) | chunk[0]
        })
        .collect()
}

/// This function composes the full computational pipeline:
///   1. Conversion of point coordinates into Montgomery form & scalar decomposition
///   2. Transposition of a CSR sparse matrix (dummy example)
///   3. Sparse matrix–vector product (SMVP)
///   4. Parallel bucket point reduction (PBPR)
///
/// The pipeline leverages our metal wrapper functions (via MetalHelper & MetalConfig).
#[test]
fn test_complete_msm_pipeline() {
    // === COMMON PARAMETERS ===
    let log_limb_size = 16;
    let num_limbs = 16;

    let mut helper = MetalHelper::new();

    let (_, _, gpu_scalar_chunks) =
        points_convertion(log_limb_size, num_limbs, &mut helper).unwrap();

    let (gpu_csc_col_ptr, gpu_csc_val_idxs) =
        transpose(&mut helper, gpu_scalar_chunks, log_limb_size, num_limbs).unwrap();

    // === STAGE 3: SPARSE MATRIX VECTOR PRODUCT (SMVP) ===
    let smvp_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/smvp.metal".to_string(),
        kernel_name: "smvp".to_string(),
    };

    // Use test data from the SMVP test.
    let row_ptr_host = vec![0, 2, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]; // length = 16
    let val_idx_host = vec![0, 7, 3, 10, 5, 1, 15];

    // Generate "new point" data for each column (16 columns).
    let mut new_point_x_host = Vec::with_capacity(16);
    let mut new_point_y_host = Vec::with_capacity(16);
    let mut rng = thread_rng();
    for _ in 0..16 {
        let new_point = G::rand(&mut rng).into_affine();
        let x_mont = new_point.x.0;
        let y_mont = new_point.y.0;
        let x_mont_biguint: BigUint = x_mont.try_into().unwrap();
        let y_mont_biguint: BigUint = y_mont.try_into().unwrap();
        new_point_x_host.push(x_mont_biguint);
        new_point_y_host.push(y_mont_biguint);
    }

    // Convert new points into limbs.
    let new_point_x_limbs: Vec<u32> = new_point_x_host
        .into_iter()
        .map(|bi| {
            let ark_form: BigInt<4> = bi.try_into().unwrap();
            ark_form.to_limbs(num_limbs, log_limb_size)
        })
        .flatten()
        .collect();
    let new_point_y_limbs: Vec<u32> = new_point_y_host
        .into_iter()
        .map(|bi| {
            let ark_form: BigInt<4> = bi.try_into().unwrap();
            ark_form.to_limbs(num_limbs, log_limb_size)
        })
        .flatten()
        .collect();

    // Create buffers for SMVP.
    let row_ptr_buf = helper.create_input_buffer(&row_ptr_host);
    let val_idx_buf = helper.create_input_buffer(&val_idx_host);
    let new_point_x_buf = helper.create_input_buffer(&new_point_x_limbs);
    let new_point_y_buf = helper.create_input_buffer(&new_point_y_limbs);

    // Create output buffers for SMVP buckets (8 buckets, each coordinate with num_limbs).
    let bucket_x_buf = helper.create_output_buffer(8 * num_limbs);
    let bucket_y_buf = helper.create_output_buffer(8 * num_limbs);
    let bucket_z_buf = helper.create_output_buffer(8 * num_limbs);

    // SMVP parameters (input_size = 7, etc.)
    let smvp_params = vec![7u32, 1u32, 1u32, 0u32];
    let smvp_params_buf = helper.create_input_buffer(&smvp_params);
    let thread_group_count_smvp = helper.create_thread_group_size(8, 1, 1);
    let thread_group_size_smvp = helper.create_thread_group_size(1, 1, 1);

    // Dispatch the SMVP shader.
    helper.execute_shader(
        &smvp_config,
        &[
            &row_ptr_buf,
            &val_idx_buf,
            &new_point_x_buf,
            &new_point_y_buf,
            &smvp_params_buf,
        ],
        &[&bucket_x_buf, &bucket_y_buf, &bucket_z_buf],
        &thread_group_count_smvp,
        &thread_group_size_smvp,
    );

    let bucket_x_out = helper.read_results(&bucket_x_buf, 8 * num_limbs);
    let bucket_y_out = helper.read_results(&bucket_y_buf, 8 * num_limbs);
    let bucket_z_out = helper.read_results(&bucket_z_buf, 8 * num_limbs);
    println!("Stage 3 – SMVP bucket X: {:?}", bucket_x_out);
    println!("Stage 3 – SMVP bucket Y: {:?}", bucket_y_out);
    println!("Stage 3 – SMVP bucket Z: {:?}", bucket_z_out);

    // ** ASSERTIONS FOR STAGE 3 **
    // Ensure that at least one limb from each bucket output (x, y, z) is nonzero.
    assert!(
        bucket_x_out.iter().any(|&v| v != 0),
        "Stage 3: SMVP bucket X output is all zero!"
    );
    assert!(
        bucket_y_out.iter().any(|&v| v != 0),
        "Stage 3: SMVP bucket Y output is all zero!"
    );
    assert!(
        bucket_z_out.iter().any(|&v| v != 0),
        "Stage 3: SMVP bucket Z output is all zero!"
    );

    // === STAGE 4: PARALLEL BUCKET POINT REDUCTION (PBPR) ===
    let pbpr_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "parallel_bpr".to_string(),
    };

    // Use the SMVP output as inputs.
    let buckets_x_buf = helper.create_input_buffer(&bucket_x_out);
    let buckets_y_buf = helper.create_input_buffer(&bucket_y_out);
    let buckets_z_buf = helper.create_input_buffer(&bucket_z_out);
    // Initialize shared accumulation buffers with the zero point.
    let total_threads = 8; // 8 threads for demonstration.
    let mut m_shared_x = vec![0u32; total_threads * num_limbs];
    let mut m_shared_y = vec![0u32; total_threads * num_limbs];
    let mut m_shared_z = vec![0u32; total_threads * num_limbs];
    let mut s_shared_x = vec![0u32; total_threads * num_limbs];
    let mut s_shared_y = vec![0u32; total_threads * num_limbs];
    let mut s_shared_z = vec![0u32; total_threads * num_limbs];

    // Create the zero point’s limb representation.
    let zero_point = G::zero();
    let zero_x: Vec<u32> = {
        let ark: BigInt<4> = zero_point.x.into_bigint().try_into().unwrap();
        ark.to_limbs(num_limbs, log_limb_size)
    };
    let zero_y: Vec<u32> = {
        let ark: BigInt<4> = zero_point.y.into_bigint().try_into().unwrap();
        ark.to_limbs(num_limbs, log_limb_size)
    };
    let zero_z: Vec<u32> = {
        let ark: BigInt<4> = zero_point.z.into_bigint().try_into().unwrap();
        ark.to_limbs(num_limbs, log_limb_size)
    };

    // Initialize each shared buffer with the zero point.
    for i in 0..total_threads {
        for j in 0..num_limbs {
            m_shared_x[i * num_limbs + j] = zero_x[j];
            m_shared_y[i * num_limbs + j] = zero_y[j];
            m_shared_z[i * num_limbs + j] = zero_z[j];
            s_shared_x[i * num_limbs + j] = zero_x[j];
            s_shared_y[i * num_limbs + j] = zero_y[j];
            s_shared_z[i * num_limbs + j] = zero_z[j];
        }
    }

    let m_shared_x_buf = helper.create_input_buffer(&m_shared_x);
    let m_shared_y_buf = helper.create_input_buffer(&m_shared_y);
    let m_shared_z_buf = helper.create_input_buffer(&m_shared_z);
    let s_shared_x_buf = helper.create_input_buffer(&s_shared_x);
    let s_shared_y_buf = helper.create_input_buffer(&s_shared_y);
    let s_shared_z_buf = helper.create_input_buffer(&s_shared_z);

    // Additional PBPR parameters.
    let grid_width = (total_threads as f64).sqrt().ceil() as u64;
    let grid_width_buf = helper.create_input_buffer(&vec![grid_width as u32]);
    let total_threads_buf = helper.create_input_buffer(&vec![total_threads as u32]);
    // Treat the number of buckets as 8.
    let num_subtask = (8 + total_threads - 1) / total_threads;
    let num_subtask_buf = helper.create_input_buffer(&vec![num_subtask as u32]);

    let thread_group_size_pbpr = helper.create_thread_group_size(32, 1, 1);
    let grid_height = (total_threads as u64 + grid_width - 1) / grid_width;
    let threads_total = helper.create_thread_group_size(grid_width, grid_height, 1);

    // Dispatch the PBPR shader.
    helper.execute_shader(
        &pbpr_config,
        &[
            &buckets_x_buf,
            &buckets_y_buf,
            &buckets_z_buf,
            &m_shared_x_buf,
            &m_shared_y_buf,
            &m_shared_z_buf,
            &s_shared_x_buf,
            &s_shared_y_buf,
            &s_shared_z_buf,
            &grid_width_buf,
            &total_threads_buf,
            &num_subtask_buf,
        ],
        &[], // results are written into the shared buffers
        &threads_total,
        &thread_group_size_pbpr,
    );

    let s_shared_x_result = helper.read_results(&s_shared_x_buf, total_threads * num_limbs);
    let s_shared_y_result = helper.read_results(&s_shared_y_buf, total_threads * num_limbs);
    let s_shared_z_result = helper.read_results(&s_shared_z_buf, total_threads * num_limbs);
    println!("Stage 4 – PBPR s_shared_x: {:?}", s_shared_x_result);
    println!("Stage 4 – PBPR s_shared_y: {:?}", s_shared_y_result);
    println!("Stage 4 – PBPR s_shared_z: {:?}", s_shared_z_result);

    // Decode the shared coordinate buffers back into points.
    let points = crate::msm::metal_msm::tests::cuzk::pbpr::points_from_separated_buffers(
        &s_shared_x_result,
        &s_shared_y_result,
        &s_shared_z_result,
        num_limbs,
        log_limb_size,
    );
    let mut final_result = G::zero();
    for pt in points.into_iter() {
        final_result += pt;
    }
    println!("Final reduced bucket point: {:?}", final_result);

    // ** ASSERTIONS FOR STAGE 4 **
    // We expect the final reduced bucket point not to be the identity (zero).
    assert!(
        !final_result.is_zero(),
        "Stage 4: Final reduced bucket point is zero!"
    );

    // Free resources.
    helper.drop_all_buffers();
}

fn transpose(
    helper: &mut MetalHelper,
    gpu_scalar_chunks: Vec<u32>,
    log_limb_size: u32,
    num_limbs: usize,
) -> Result<(Vec<u32>, Vec<u32>), Box<dyn Error>> {
    // ========= Stage 2: Sparse Matrix Transposition =========

    // We now use the output of stage 1 (gpu_scalar_chunks) as input CSR column indices.
    // In the transpose shader a number of subtasks may be scheduled.
    // For demonstration we assume a single subtask here.
    let num_subtasks = 1;
    const MAX_COLS: u32 = 8;
    // Let MAX_COLS be a fixed constant for the transpose stage.

    // Create the input buffer from gpu_scalar_chunks.
    let csr_cols_buf = helper.create_input_buffer(&gpu_scalar_chunks);

    // Set the transpose parameters: here input_size_trans is the number of CSR entries.
    let input_size_trans = gpu_scalar_chunks.len() as u32;
    let transpose_params_buf = helper.create_input_buffer(&vec![input_size_trans, MAX_COLS]);

    // Create output buffers for the transposed data:
    //   - CSC column pointer array of length = MAX_COLS + 1.
    //   - CSC value indices array of length = input_size_trans.
    //   - Additional buffer for internal state (all_curr) of length = MAX_COLS.
    let csc_col_ptr_buf = helper.create_output_buffer((MAX_COLS as usize) + 1);
    let csc_val_idxs_buf = helper.create_output_buffer(gpu_scalar_chunks.len());
    let all_curr_buf = helper.create_output_buffer(MAX_COLS as usize);

    // Set up the transpose shader configuration.
    let transpose_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    // For simplicity we use 1×1×1 workgroups.
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // Dispatch the transpose shader.
    helper.execute_shader(
        &transpose_config,
        &[&csr_cols_buf, &transpose_params_buf],
        &[&csc_col_ptr_buf, &csc_val_idxs_buf, &all_curr_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read back the GPU output from stage 2.
    let gpu_csc_col_ptr = helper.read_results(&csc_col_ptr_buf, (MAX_COLS as usize) + 1);
    let gpu_csc_val_idxs = helper.read_results(&csc_val_idxs_buf, gpu_scalar_chunks.len());
    println!("Stage 2 – Transposed CSC col_ptr: {:?}", gpu_csc_col_ptr);
    println!("Stage 2 – Transposed CSC val_idxs: {:?}", gpu_csc_val_idxs);

    // ========= CPU Reference: Compute Expected Transposition =========

    // Use the CSR column indices produced from stage 1 (gpu_scalar_chunks) as input to the CPU routine.
    // Here compute_expected_csc is a helper that mimics the GPU’s transpose shader.
    let (expected_col_ptr, expected_val_idxs) = compute_expected_csc(&gpu_scalar_chunks, MAX_COLS);

    // For example, if the transpose was designed to run per subtask,
    // you might loop over each subtask as shown below.
    //
    // (Here we assume one subtask; if multiple subtasks were used then the output buffers
    //  would be laid out contiguously for each subtask—adjust indices accordingly.)
    for subtask in 0..num_subtasks {
        // For the CSC column pointer array, each subtask has (MAX_COLS + 1) entries.
        let offset = subtask * ((MAX_COLS as usize) + 1);
        let actual_col_ptr = &gpu_csc_col_ptr[offset..offset + (MAX_COLS as usize + 1)];

        assert_eq!(
            actual_col_ptr, &expected_col_ptr,
            "Subtask {}: Column pointers mismatch\nExpected: {:?}\nActual: {:?}",
            subtask, expected_col_ptr, actual_col_ptr
        );

        // For the CSC value indices, assume each subtask covers `input_size_trans` elements.
        let val_offset = subtask * gpu_scalar_chunks.len();
        let actual_vals = &gpu_csc_val_idxs[val_offset..val_offset + gpu_scalar_chunks.len()];

        assert_eq!(
            actual_vals, &expected_val_idxs,
            "Subtask {}: Value indices mismatch\nExpected: {:?}\nActual: {:?}",
            subtask, expected_val_idxs, actual_vals
        );
    }

    Ok((gpu_csc_col_ptr, gpu_csc_val_idxs))
}

fn points_convertion(
    log_limb_size: u32,
    num_limbs: usize,
    helper: &mut MetalHelper,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
    // === STAGE 1: CONVERT POINT COORDINATES & DECOMPOSE SCALARS ===
    let conv_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
        kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
    };

    // Generate an example point (using the group's generator)
    let point = G::generator().into_affine();
    // Convert its x and y coordinates to BigUint
    let x: BigUint = point.x.into_bigint().try_into().unwrap();
    let y: BigUint = point.y.into_bigint().try_into().unwrap();

    // Get Montgomery constants.
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p = &constants.p;
    let r = &constants.r;

    // Compute Montgomery representations.
    let x_mont = (&x * r) % p;
    let y_mont = (&y * r) % p;

    // Compute expected Montgomery values as ark_ff BigInt limbs.
    let x_mont_in_ark: BigInt<4> = x_mont.clone().try_into().unwrap();
    let y_mont_in_ark: BigInt<4> = y_mont.clone().try_into().unwrap();
    let expected_x_limbs = x_mont_in_ark.to_limbs(num_limbs, log_limb_size);
    let expected_y_limbs = y_mont_in_ark.to_limbs(num_limbs, log_limb_size);

    // Also compute the raw (non‐Montgomery) x and y as limbs (packed as in the tests).
    let x_in_ark: BigInt<4> = x.clone().try_into().unwrap();
    let y_in_ark: BigInt<4> = y.clone().try_into().unwrap();
    let x_limbs = x_in_ark.to_limbs(num_limbs, log_limb_size);
    let y_limbs = y_in_ark.to_limbs(num_limbs, log_limb_size);
    let x_packed = pack_limbs(&x_limbs);
    let y_packed = pack_limbs(&y_limbs);
    let coords: Vec<u32> = [x_packed, y_packed].concat();

    // In this conversion stage we don’t use the scalar data, so supply zeros.
    let scalars = vec![0u32; num_limbs];

    // Create input buffers.
    let coords_buf = helper.create_input_buffer(&coords);
    let scalars_buf = helper.create_input_buffer(&scalars);
    let input_size_buf = helper.create_input_buffer(&vec![1u32]);
    // one point

    // Create output buffers for the converted point (x and y) and scalar chunks.
    let point_x_buf = helper.create_output_buffer(num_limbs);
    let point_y_buf = helper.create_output_buffer(num_limbs);
    let chunks_buf = helper.create_output_buffer(num_limbs);

    // Set up thread group sizes (we use 1×1×1 workgroup for demonstration).
    let thread_group_count = helper.create_thread_group_size(1, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // Dispatch the conversion & decomposition shader.
    helper.execute_shader(
        &conv_config,
        &[&coords_buf, &scalars_buf, &input_size_buf],
        &[&point_x_buf, &point_y_buf, &chunks_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read back converted point data.
    let gpu_point_x = helper.read_results(&point_x_buf, num_limbs);
    let gpu_point_y = helper.read_results(&point_y_buf, num_limbs);
    let gpu_scalar_chunks = helper.read_results(&chunks_buf, num_limbs);
    println!("Stage 1 – GPU Converted X limbs: {:?}", gpu_point_x);
    println!("Stage 1 – GPU Converted Y limbs: {:?}", gpu_point_y);
    println!("Stage 1 – GPU Scalar chunks: {:?}", gpu_scalar_chunks);
    println!("Stage 1 – Expected X limbs: {:?}", expected_x_limbs);
    println!("Stage 1 – Expected Y limbs: {:?}", expected_y_limbs);

    // ** ASSERTIONS FOR STAGE 1 **
    assert_eq!(
        gpu_point_x, expected_x_limbs,
        "Stage 1: GPU X conversion mismatch!"
    );
    assert_eq!(
        gpu_point_y, expected_y_limbs,
        "Stage 1: GPU Y conversion mismatch!"
    );

    return Ok((gpu_point_x, gpu_point_y, gpu_scalar_chunks));
}
