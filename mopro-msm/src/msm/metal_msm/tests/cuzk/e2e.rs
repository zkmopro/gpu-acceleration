use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, PrimeField};
use ark_std::{UniformRand, Zero};
use num_bigint::BigUint;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::msm::metal_msm::cuzk_cpu_reproduction::*;
use crate::msm::metal_msm::tests::cuzk::pbpr::closest_power_of_two;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

/// This function composes the full computational pipeline:
///   1. Conversion of point coordinates into Montgomery form & scalar decomposition
///   2. Transposition of a CSR sparse matrix (dummy example)
///   3. Sparse matrix–vector product (SMVP)
///   4. Parallel bucket point reduction (PBPR)
///
/// The pipeline leverages our metal wrapper functions (via MetalHelper & MetalConfig).
#[test]
fn test_complete_msm_pipeline() {
    use ark_bn254::G1Projective as G;

    let log_limb_size = 16;
    let num_limbs = 16;

    let input_size = 10;
    let (fixed_point, fixed_scalar) = get_fixed_inputs_cpu_style();

    // Create vectors filled with the same fixed point and scalar.
    let points = vec![fixed_point; input_size];
    let scalars = vec![fixed_scalar; input_size];
    println!("\n===== points =====");

    for (i, pt) in points.iter().enumerate() {
        println!(
            "pt_{}: {:?}",
            i,
            pt.x.into_bigint().to_limbs(num_limbs, log_limb_size as u32)
        );
        println!(
            "pt_{}: {:?}",
            i,
            pt.y.into_bigint().to_limbs(num_limbs, log_limb_size as u32)
        );
    }
    println!("\n===== scalars =====");
    for (i, sc) in scalars.iter().enumerate() {
        println!(
            "sc_{}: {:?}",
            i,
            sc.into_bigint().to_limbs(num_limbs, log_limb_size as u32)
        );
    }

    let chunk_size = if points.len() >= 65536 { 16 } else { 4 };
    let num_subtasks = 256 / chunk_size;
    let num_columns = 1 << chunk_size;

    let points_msm_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
        kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
    };

    // 1) Convert Ark `Affine` and `ScalarField` arrays into the "packed" format that
    //    your GPU code expects: each point => 16 u32 for coords, each scalar => 8 u32.
    let input_size = points.len();
    let (packed_coords, packed_scalars) =
        pack_affine_and_scalars(&points, &scalars, &points_msm_config);

    println!("\n===== INPUT FOR convert_point_coords_and_decompose_scalars shaders =====");
    println!("packed_coords: {:?}", packed_coords);
    println!("packed_scalars: {:?}", packed_scalars);
    println!("input_size: {}", input_size);

    // === Stage 1: Convert Point Coordinates & Decompose Scalars ===
    let mut helper = MetalHelper::new();
    let (gpu_point_x, gpu_point_y, gpu_scalar_chunks) = points_convertion(
        &mut helper,
        points_msm_config.clone(),
        input_size * num_subtasks,
        input_size,
        &packed_coords,
        &packed_scalars,
    );
    helper.drop_all_buffers();

    // === CPU ===
    let mut cpu_point_x = vec![BaseField::zero(); input_size];
    let mut cpu_point_y = vec![BaseField::zero(); input_size];
    let mut cpu_scalar_chunks = vec![0u32; input_size * num_subtasks];

    convert_point_coords_and_decompose_scalars(
        &packed_coords,
        &packed_scalars,
        input_size,
        &mut cpu_point_x,
        &mut cpu_point_y,
        &mut cpu_scalar_chunks,
        &points_msm_config,
        chunk_size as u32,
        num_subtasks,
    )
    .unwrap();

    let cpu_point_x = cpu_point_x
        .iter()
        .flat_map(|f| convert_coord_to_u32(f))
        .collect::<Vec<u32>>();
    let cpu_point_y = cpu_point_y
        .iter()
        .flat_map(|f| convert_coord_to_u32(f))
        .collect::<Vec<u32>>();

    // === CPU END ===
    assert_eq!(gpu_point_x, cpu_point_x);
    assert_eq!(gpu_point_y, cpu_point_y);
    assert_eq!(gpu_scalar_chunks, cpu_scalar_chunks);

    // === Stage 2: Sparse Matrix Transposition ===
    let mut helper2 = MetalHelper::new();
    let (gpu_csc_col_ptr, gpu_csc_val_idxs, _) = transpose(
        &mut helper2,
        gpu_scalar_chunks.clone(),
        points_msm_config.log_limb_size,
        points_msm_config.num_limbs,
        num_subtasks,
        input_size,
        num_columns,
    );
    helper2.drop_all_buffers();

    let (cpu_csc_col_ptr, cpu_csc_val_idxs) = transpose_cpu(
        &gpu_scalar_chunks,
        num_subtasks as u32,
        input_size as u32,
        num_columns as u32,
    );
    let cpu_csc_col_ptr = cpu_csc_col_ptr
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<u32>>();
    let cpu_csc_val_idxs = cpu_csc_val_idxs
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<u32>>();

    assert_eq!(gpu_csc_col_ptr, cpu_csc_col_ptr);
    assert_eq!(gpu_csc_val_idxs, cpu_csc_val_idxs);

    // // === Stage 3: Sparse Matrix Vector Product (SMVP) ===
    // let seed = [42u8; 32];
    // let rng = StdRng::from_seed(seed);
    // let mut helper3 = MetalHelper::new();
    // let (bucket_x_out, bucket_y_out, bucket_z_out) = smvp(
    //     &mut helper3,
    //     points_msm_config.log_limb_size,
    //     points_msm_config.num_limbs,
    //     &csc_col_ptr,
    //     &csc_val_idxs,
    //     num_columns,
    //     rng,
    // );
    // helper3.drop_all_buffers();

    // // === Stage 4: Parallel Bucket Point Reduction (Pbpr) ===
    // let mut helper4 = MetalHelper::new();
    // let (s_shared_x_result, s_shared_y_result, s_shared_z_result) = pbpr(
    //     points_msm_config.log_limb_size,
    //     points_msm_config.num_limbs,
    //     &mut helper4,
    //     bucket_x_out,
    //     bucket_y_out,
    //     bucket_z_out,
    // );
    // helper4.drop_all_buffers();

    // // Decode the shared coordinate buffers back into points.
    // let points = crate::msm::metal_msm::tests::cuzk::pbpr::points_from_separated_buffers(
    //     &s_shared_x_result,
    //     &s_shared_y_result,
    //     &s_shared_z_result,
    //     points_msm_config.num_limbs,
    //     points_msm_config.log_limb_size,
    // );

    // let mut final_result = G::zero();
    // for pt in points.into_iter() {
    //     final_result += pt;
    // }
    // println!("Final reduced bucket point: {:?}", final_result);

    // // ** ASSERTIONS FOR STAGE 4 **
    // // We expect the final reduced bucket point not to be the identity (zero).
    // assert!(
    //     !final_result.is_zero(),
    //     "Stage 4: Final reduced bucket point is zero!"
    // );
}

fn points_convertion(
    helper: &mut MetalHelper,
    points_msm_config: MetalConfig,
    chunk_size: usize,
    input_size: usize,
    coords: &Vec<u32>,
    scalars: &Vec<u32>,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // Create input buffers.
    let coords_buf = helper.create_input_buffer(&coords);
    let scalars_buf = helper.create_input_buffer(&scalars);
    let input_size_buf = helper.create_input_buffer(&vec![input_size as u32]);
    // one point

    // Create output buffers for the converted point (x and y) and scalar chunks.
    let point_x_buf = helper.create_output_buffer(16 * input_size);
    let point_y_buf = helper.create_output_buffer(16 * input_size);
    let chunks_buf = helper.create_output_buffer(16 * input_size);

    // Set up thread group sizes (we use 1×1×1 workgroup for demonstration).
    let thread_group_count = helper.create_thread_group_size(input_size as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(input_size as u64, 1, 1);

    // Dispatch the conversion & decomposition shader.
    helper.execute_shader(
        &points_msm_config,
        &[&coords_buf, &scalars_buf, &input_size_buf],
        &[&point_x_buf, &point_y_buf, &chunks_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read back converted point data.
    let gpu_point_x = helper.read_results(&point_x_buf, 16 * input_size);
    let gpu_point_y = helper.read_results(&point_y_buf, 16 * input_size);
    let gpu_scalar_chunks = helper.read_results(&chunks_buf, chunk_size as usize);
    // println!("\n===== OUTPUT FROM convert_point_coords_and_decompose_scalars shaders =====");
    // println!("point_x: {:?}", gpu_point_x);
    // println!("point_y: {:?}", gpu_point_y);
    // println!("chunks: {:?}", gpu_scalar_chunks);

    (gpu_point_x, gpu_point_y, gpu_scalar_chunks)
}

fn transpose(
    helper: &mut MetalHelper,
    gpu_scalar_chunks: Vec<u32>,
    log_limb_size: u32,
    num_limbs: usize,
    num_subtasks: usize,
    input_size: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // Create the input buffer from gpu_scalar_chunks.
    let csr_cols_buf = helper.create_input_buffer(&gpu_scalar_chunks);
    let params_buf = helper.create_input_buffer(&vec![num_columns, input_size as u32]);

    // Create output buffers for the transposed data:
    //   - CSC column pointer array of length = MAX_COLS + 1.
    //   - CSC value indices array of length = input_size_trans.
    //   - Additional buffer for internal state (all_curr) of length = MAX_COLS.
    let csc_col_ptr_buf = helper.create_output_buffer(num_subtasks * (num_columns as usize + 1));
    let csc_val_idxs_buf = helper.create_output_buffer(num_subtasks * input_size as usize);
    let all_curr_buf = helper.create_output_buffer(num_subtasks * num_columns as usize);

    // Set up the transpose shader configuration.
    let transpose_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(),
    };

    // For simplicity we use 1×1×1 workgroups.
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // Dispatch the transpose shader.
    helper.execute_shader(
        &transpose_config,
        &[&csr_cols_buf, &params_buf],
        &[&csc_col_ptr_buf, &csc_val_idxs_buf, &all_curr_buf],
        &thread_group_count,
        &thread_group_size,
    );

    // Read back the GPU output from stage 2.
    let gpu_csc_col_ptr =
        helper.read_results(&csc_col_ptr_buf, num_subtasks * (num_columns as usize + 1));
    let gpu_csc_val_idxs = helper.read_results(&csc_val_idxs_buf, gpu_scalar_chunks.len());
    let all_curr_buf = helper.read_results(&all_curr_buf, num_columns as usize);
    println!("Stage 2 – Transposed CSC col_ptr: {:?}", gpu_csc_col_ptr);
    println!("Stage 2 – Transposed CSC val_idxs: {:?}", gpu_csc_val_idxs);
    println!("Stage 2 – Transposed CSC all_curr_buf: {:?}", all_curr_buf);

    (gpu_csc_col_ptr, gpu_csc_val_idxs, all_curr_buf)
}

fn smvp(
    helper: &mut MetalHelper,
    log_limb_size: u32,
    num_limbs: usize,
    csc_col_ptr: &Vec<u32>,
    csc_val_idxs: &Vec<u32>,
    max_cols: u32,
    mut rng: StdRng,
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
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let pbpr_config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "parallel_bpr".to_string(),
    };

    let buckets_x_buf = helper.create_input_buffer(&bucket_x_out);
    let buckets_y_buf = helper.create_input_buffer(&bucket_y_out);
    let buckets_z_buf = helper.create_input_buffer(&bucket_z_out);

    // Determine bucket count from SMVP output.
    // In SMVP you created buckets with length = bucket_count * num_limbs.
    // For example, if max_cols was 8 then bucket_count was computed as 8/2 = 4.
    let bucket_count = bucket_x_out.len() / num_limbs;

    // Compute total_threads as in the reference.
    let total_threads = closest_power_of_two(bucket_count);

    // Initialize shared buffers (m_shared and s_shared) with zeros.
    let shared_size = total_threads * num_limbs;
    let mut m_shared_x = vec![0u32; shared_size];
    let mut m_shared_y = vec![0u32; shared_size];
    let mut m_shared_z = vec![0u32; shared_size];
    let mut s_shared_x = vec![0u32; shared_size];
    let mut s_shared_y = vec![0u32; shared_size];
    let mut s_shared_z = vec![0u32; shared_size];

    // Create zero point limb representation.
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

    // Initialize shared buffers with the zero point.
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

    // Create input buffers for the shared arrays.
    let m_shared_x_buf = helper.create_input_buffer(&m_shared_x);
    let m_shared_y_buf = helper.create_input_buffer(&m_shared_y);
    let m_shared_z_buf = helper.create_input_buffer(&m_shared_z);
    let s_shared_x_buf = helper.create_input_buffer(&s_shared_x);
    let s_shared_y_buf = helper.create_input_buffer(&s_shared_y);
    let s_shared_z_buf = helper.create_input_buffer(&s_shared_z);

    // Parameter buffers:
    // grid_width is computed from the total_threads.
    let grid_width = (total_threads as f64).sqrt().ceil() as u64;
    let grid_width_buf = helper.create_input_buffer(&vec![grid_width as u32]);
    let total_threads_buf = helper.create_input_buffer(&vec![total_threads as u32]);
    // Instead of hardcoding 8 buckets, use the actual bucket_count.
    let num_subtask = (bucket_count + total_threads - 1) / total_threads;
    let num_subtask_buf = helper.create_input_buffer(&vec![num_subtask as u32]);

    // Setup thread group sizes. Using a default threadgroup width of 32.
    let thread_group_size_pbpr = helper.create_thread_group_size(32, 1, 1);
    let grid_height = (total_threads as u64 + grid_width - 1) / grid_width;
    let threads_total = helper.create_thread_group_size(grid_width, grid_height, 1);

    // Debug prints for inspection.
    println!("Stage 4 - bucket_count: {}", bucket_count);
    println!(
        "Stage 4 - total_threads (closest power of two): {}",
        total_threads
    );
    println!("Stage 4 - num_subtask: {}", num_subtask);

    // Dispatch the shader.
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
        &[], // No separate output buffers; results come out in shared buffers.
        &threads_total,
        &thread_group_size_pbpr,
    );

    // Read back the results.
    let s_shared_x_result = helper.read_results(&s_shared_x_buf, shared_size);
    let s_shared_y_result = helper.read_results(&s_shared_y_buf, shared_size);
    let s_shared_z_result = helper.read_results(&s_shared_z_buf, shared_size);

    println!("Stage 4 - s_shared_x: {:?}", s_shared_x_result);
    println!("Stage 4 - s_shared_y: {:?}", s_shared_y_result);
    println!("Stage 4 - s_shared_z: {:?}", s_shared_z_result);

    helper.drop_all_buffers();

    // Read the shared memory buffers as points and sum them.
    (s_shared_x_result, s_shared_y_result, s_shared_z_result)
}
