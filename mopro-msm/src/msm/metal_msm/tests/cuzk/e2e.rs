use std::error::Error;

use ark_bn254::g1::Config;
use ark_bn254::{Fr as ScalarField, G1Projective as G};
use ark_ec::short_weierstrass::Affine;
use ark_ec::{CurveGroup, Group};
use ark_ff::{BigInt, PrimeField};
use ark_std::{UniformRand, Zero};
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

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

    // We now use the output of stage 1 (gpu_scalar_chunks) as input CSR column indices.
    // In the transpose shader a number of subtasks may be scheduled.
    // For demonstration we assume a single subtask here.
    const MAX_COLS: u32 = 8;
    let total_threads = 8;
    let point = G::generator().into_affine();
    // Generate a random 254-bit scalar
    let mut rng = thread_rng();
    let scalar = rng.gen_biguint(254);
    let scalar_in_scalarfield = ScalarField::from(scalar.clone());

    // Convert scalar to the same limb format the kernel expects
    let scalars = scalar_in_scalarfield
        .into_bigint()
        .to_limbs(num_limbs, log_limb_size);

    // === Stage 1: Convert Point Coordinates & Decompose Scalars ===
    let mut helper = MetalHelper::new();
    let (_, _, gpu_scalar_chunks) =
        points_convertion(log_limb_size, num_limbs, &mut helper, point, scalars);
    helper.drop_all_buffers();

    // === Stage 2: Sparse Matrix Transposition ===
    let mut helper2 = MetalHelper::new();
    let (csc_col_ptr, csc_val_idxs, _) = transpose(
        &mut helper2,
        gpu_scalar_chunks,
        log_limb_size,
        num_limbs,
        MAX_COLS,
    );
    helper2.drop_all_buffers();

    // === Stage 3: Sparse Matrix Vector Product (SMVP) ===
    let mut helper3 = MetalHelper::new();
    let (bucket_x_out, bucket_y_out, bucket_z_out) = smvp(
        &mut helper3,
        log_limb_size,
        num_limbs,
        &csc_col_ptr,
        &csc_val_idxs,
        MAX_COLS,
    );
    helper3.drop_all_buffers();

    // === Stage 4: Parallel Bucket Point Reduction (Pbpr) ===
    let mut helper4 = MetalHelper::new();
    let (s_shared_x_result, s_shared_y_result, s_shared_z_result) = pbpr(
        log_limb_size,
        num_limbs,
        &mut helper4,
        bucket_x_out,
        bucket_y_out,
        bucket_z_out,
        total_threads,
    );
    helper4.drop_all_buffers();

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
}

fn points_convertion(
    log_limb_size: u32,
    num_limbs: usize,
    helper: &mut MetalHelper,
    point: Affine<Config>,
    scalars: Vec<u32>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let conv_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
        kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
    };
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

    let coords: Vec<u32> = [x_packed.clone(), y_packed.clone()].concat();

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

    (gpu_point_x, gpu_point_y, gpu_scalar_chunks)
}

fn transpose(
    helper: &mut MetalHelper,
    gpu_scalar_chunks: Vec<u32>,
    log_limb_size: u32,
    num_limbs: usize,
    max_cols: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // Create the input buffer from gpu_scalar_chunks.
    let csr_cols_buf = helper.create_input_buffer(&gpu_scalar_chunks);

    // Set the transpose parameters: here input_size_trans is the number of CSR entries.
    let input_size_trans = gpu_scalar_chunks.len() as u32;
    let transpose_params_buf = helper.create_input_buffer(&vec![max_cols, input_size_trans]);

    // Create output buffers for the transposed data:
    //   - CSC column pointer array of length = MAX_COLS + 1.
    //   - CSC value indices array of length = input_size_trans.
    //   - Additional buffer for internal state (all_curr) of length = MAX_COLS.
    let csc_col_ptr_buf = helper.create_output_buffer((max_cols as usize) + 1);
    let csc_val_idxs_buf = helper.create_output_buffer(gpu_scalar_chunks.len());
    let all_curr_buf = helper.create_output_buffer(max_cols as usize);

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
    let gpu_csc_col_ptr = helper.read_results(&csc_col_ptr_buf, (max_cols as usize) + 1);
    let gpu_csc_val_idxs = helper.read_results(&csc_val_idxs_buf, gpu_scalar_chunks.len());
    let all_curr_buf = helper.read_results(&all_curr_buf, max_cols as usize);
    println!("Stage 2 – Transposed CSC col_ptr: {:?}", gpu_csc_col_ptr);
    println!("Stage 2 – Transposed CSC val_idxs: {:?}", gpu_csc_val_idxs);

    (gpu_csc_col_ptr, gpu_csc_val_idxs, all_curr_buf)
}

fn smvp(
    helper: &mut MetalHelper,
    log_limb_size: u32,
    num_limbs: usize,
    csc_col_ptr: &Vec<u32>,
    csc_val_idxs: &Vec<u32>,
    max_cols: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let smvp_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/smvp.metal".to_string(),
        kernel_name: "smvp".to_string(),
    };

    // Here, num_columns should match the intended number of new points.
    // In your e2e pipeline, you set max_cols to 8 in the transpose stage.
    // Therefore we use that as the number of columns for SMVP.
    let num_columns: u32 = max_cols;
    // The transpose stage creates a row_ptr buffer of length max_cols + 1.
    println!("Stage 3 - Received row_ptr buffer: {:?}", csc_col_ptr);
    assert_eq!(
        csc_col_ptr.len(),
        (num_columns + 1) as usize,
        "Expected row_ptr buffer length to be {}",
        num_columns + 1
    );

    // Generate new points in Montgomery form for each column.
    let mut new_point_x_host = Vec::with_capacity(num_columns as usize);
    let mut new_point_y_host = Vec::with_capacity(num_columns as usize);
    let mut rng = rand::thread_rng();
    for _ in 0..num_columns {
        let new_point = G::rand(&mut rng).into_affine();
        let x_mont = new_point.x.0;
        let y_mont = new_point.y.0;
        let x_mont_biguint: BigUint = x_mont.try_into().unwrap();
        let y_mont_biguint: BigUint = y_mont.try_into().unwrap();
        new_point_x_host.push(x_mont_biguint);
        new_point_y_host.push(y_mont_biguint);
    }

    // Convert each new point to a limbs representation.
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

    // Use the CSR buffers coming from the transpose stage for SMVP.
    let row_ptr_buf = helper.create_input_buffer(csc_col_ptr);
    let val_idx_buf = helper.create_input_buffer(csc_val_idxs);
    let new_point_x_buf = helper.create_input_buffer(&new_point_x_limbs);
    let new_point_y_buf = helper.create_input_buffer(&new_point_y_limbs);

    // Create output buffers – final buckets.
    // Test_smvp uses 8 final buckets (half of 16), each bucket being a Jacobian coordinate with
    // each coordinate represented by num_limbs limbs.
    let bucket_count = num_columns / 2; // 16/2 = 8
    let bucket_x_buf = helper.create_output_buffer((bucket_count as usize * num_limbs) as usize);
    let bucket_y_buf = helper.create_output_buffer((bucket_count as usize * num_limbs) as usize);
    let bucket_z_buf = helper.create_output_buffer((bucket_count as usize * num_limbs) as usize);

    // SMVP parameters as in test_smvp: input_size (here length of val_idx), num_y_workgroups, num_z_workgroups, subtask_offset.
    let input_size = csc_val_idxs.len() as u32; // e.g., 7 if you know that’s the case.
    let smvp_params = vec![input_size, 1u32, 1u32, 0u32];
    let smvp_params_buf = helper.create_input_buffer(&smvp_params);

    // Set thread group dimensions – one thread per final bucket.
    let thread_group_count_smvp = helper.create_thread_group_size(bucket_count as u64, 1, 1);
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

    // Read back results.
    let bucket_y_out =
        helper.read_results(&bucket_y_buf, (bucket_count as usize * num_limbs) as usize);
    let bucket_z_out =
        helper.read_results(&bucket_z_buf, (bucket_count as usize * num_limbs) as usize);
    let bucket_x_out =
        helper.read_results(&bucket_x_buf, (bucket_count as usize * num_limbs) as usize);

    println!("Stage 3 - Bucket X output: {:?}", bucket_x_out);
    // TODO: From time to time this buffer is all zeroes!?
    println!("Stage 3 - Bucket Y output: {:?}", bucket_y_out);
    println!("Stage 3 - Bucket Z output: {:?}", bucket_z_out);

    helper.drop_all_buffers();

    (bucket_x_out, bucket_y_out, bucket_z_out)
}

fn pbpr(
    log_limb_size: u32,
    num_limbs: usize,
    helper: &mut MetalHelper,
    bucket_x_out: Vec<u32>,
    bucket_y_out: Vec<u32>,
    bucket_z_out: Vec<u32>,
    total_threads: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
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

    // 8 threads for demonstration.
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
    (s_shared_x_result, s_shared_y_result, s_shared_z_result)
}
