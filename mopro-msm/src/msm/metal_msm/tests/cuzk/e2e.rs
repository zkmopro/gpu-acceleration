use crate::msm::metal_msm::cuzk_cpu_reproduction::*;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;
use crate::msm::metal_msm::utils::mont_reduction::raw_reduction;
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as Affine, G1Projective as G};
use ark_ec::VariableBaseMSM;
use ark_ff::{BigInt, PrimeField};
use ark_std::Zero;
use std::error::Error;

pub fn metal_e2e_msm(bases: &[Affine], scalars: &[ScalarField]) -> Result<G, Box<dyn Error>> {
    let input_size = bases.len();
    let chunk_size = 16;
    let num_subtasks = 256 / chunk_size;
    let num_columns = 1 << chunk_size; // 2^chunk_size

    let msm_config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "".to_string(),
        kernel_name: "".to_string(),
    };
    let msm_constants = get_or_calc_constants(msm_config.num_limbs, msm_config.log_limb_size);

    //-------------------------------------------------------
    // Pack input points + scalars => 16-bit halfwords
    //-------------------------------------------------------
    let (coords, scals) = pack_affine_and_scalars(bases, scalars, &msm_config);

    //-------------------------------------------------------
    // 1) Convert & Decompose (GPU)
    //    => point_x, point_y, scalar_chunks
    //-------------------------------------------------------
    // Per your writeup, we want to launch `input_size` threads total in 1D.
    // E.g. each thread handles exactly 1 (point, scalar).
    let mut helper1 = MetalHelper::new();
    let conv_decomp_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        // The Metal or .metal file that has your coordinate+scalar decomposition kernel:
        shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
        kernel_name: "convert_point_coords_and_decompose_scalars".to_string(), // or whatever entry point you use
    };

    println!("\n--- Stage 1: Convert & Decompose (GPU) ---");
    println!("Input coords len: {}", coords.len());
    println!("Input scals len: {}", scals.len());
    println!("Input input_size: {}", input_size);
    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input chunk_size: {}", chunk_size);

    let (gpu_point_x, gpu_point_y, gpu_scalar_chunks) = convert_decompose_gpu(
        &mut helper1,
        &conv_decomp_config,
        &coords,
        &scals,
        input_size,
        num_subtasks,
    );
    println!("Output gpu_point_x len: {}", gpu_point_x.len());
    println!("Output gpu_point_y len: {}", gpu_point_y.len());
    println!("Output gpu_scalar_chunks len: {}", gpu_scalar_chunks.len());

    helper1.drop_all_buffers();

    //-------------------------------------------------------
    // 2) Transpose (GPU)
    //    => csc_col_ptr, csc_val_idxs
    //-------------------------------------------------------
    // Launch `num_subtasks` threads total (1D). Each subtask does its own transpose.
    let mut helper2 = MetalHelper::new();
    let transpose_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        shader_file: "cuzk/transpose.metal".to_string(),
        kernel_name: "transpose".to_string(), // or your actual entry point
    };

    println!("\n--- Stage 2: Transpose (GPU) ---");
    println!("Input gpu_scalar_chunks len: {}", gpu_scalar_chunks.len());
    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input input_size: {}", input_size);
    println!("Input num_columns: {}", num_columns);

    let (gpu_csc_col_ptr, gpu_csc_val_idxs) = transpose_gpu(
        &mut helper2,
        &transpose_config,
        &gpu_scalar_chunks,
        num_subtasks,
        input_size,
        num_columns,
    );
    println!("Output gpu_csc_col_ptr len: {}", gpu_csc_col_ptr.len());
    println!("Output gpu_csc_val_idxs len: {}", gpu_csc_val_idxs.len());

    helper2.drop_all_buffers();

    //-------------------------------------------------------
    // 3) SMVP (GPU)
    //    => bucket_x, bucket_y, bucket_z
    //-------------------------------------------------------
    // Launch `(input_size/2)*num_subtasks` threads in total (which is half_columns * num_subtasks).
    // Or if you prefer, you can do (num_subtasks, half_columns, 1) as 3D dispatch.
    // Each thread does "two passes" (positive + negative).
    let mut helper3 = MetalHelper::new();
    let smvp_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        shader_file: "cuzk/smvp.metal".to_string(),
        kernel_name: "smvp".to_string(),
    };

    println!("\n--- Stage 3: SMVP (GPU) ---");
    println!("Input gpu_csc_col_ptr len: {}", gpu_csc_col_ptr.len());
    println!("Input gpu_csc_val_idxs len: {}", gpu_csc_val_idxs.len());
    println!("Input gpu_point_x len: {}", gpu_point_x.len());
    println!("Input gpu_point_y len: {}", gpu_point_y.len());
    println!("Input input_size: {}", input_size);
    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input num_columns: {}", num_columns);

    let (gpu_bucket_x, gpu_bucket_y, gpu_bucket_z) = smvp_gpu(
        &mut helper3,
        &smvp_config,
        &gpu_csc_col_ptr,
        &gpu_csc_val_idxs,
        &gpu_point_x,
        &gpu_point_y,
        input_size,
        num_subtasks,
        num_columns,
    );
    println!("Output gpu_bucket_x len: {}", gpu_bucket_x.len());
    println!("Output gpu_bucket_y len: {}", gpu_bucket_y.len());
    println!("Output gpu_bucket_z len: {}", gpu_bucket_z.len());

    helper3.drop_all_buffers();

    //-------------------------------------------------------
    // 4) Parallel Bucket Reduction (PBPR) (GPU)
    //    => g_points_x, g_points_y, g_points_z
    //-------------------------------------------------------
    // Launch `256*num_subtasks` threads total, in two stages (like submission.ts).
    // Each stage processes `num_subtasks_per_bpr` at a time.
    // Typically 256 threads per subtask.
    let mut helper4 = MetalHelper::new();
    let pbpr_stage1_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "bpr_stage_1".to_string(),
    };
    let pbpr_stage2_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "bpr_stage_2".to_string(),
    };

    println!("\n--- Stage 4: PBPR (GPU) ---");
    println!("Input gpu_bucket_x len: {}", gpu_bucket_x.len());
    println!("Input gpu_bucket_y len: {}", gpu_bucket_y.len());
    println!("Input gpu_bucket_z len: {}", gpu_bucket_z.len());
    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input num_columns: {}", num_columns);

    let (gpu_g_points_x, gpu_g_points_y, gpu_g_points_z) = pbpr_gpu(
        &mut helper4,
        &pbpr_stage1_config,
        &pbpr_stage2_config,
        &gpu_bucket_x,
        &gpu_bucket_y,
        &gpu_bucket_z,
        num_subtasks,
        num_columns,
    );
    println!("Output gpu_g_points_x len: {}", gpu_g_points_x.len());
    println!("Output gpu_g_points_y len: {}", gpu_g_points_y.len());
    println!("Output gpu_g_points_z len: {}", gpu_g_points_z.len());

    helper4.drop_all_buffers();

    // At this point, we have `g_points_x, g_points_y, g_points_z` representing the "bucket-sum"
    // partial results for each subtask. We'll decode them to BN254 projective points in CPU.

    println!("\n--- Stage 4.5: PBPR Final Reduction (CPU) ---");
    let pbpr_workgroup_size: usize = 256; // Match b_workgroup_size from GPU
    let mut gpu_points = Vec::with_capacity(num_subtasks);

    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input pbpr_workgroup_size: {}", pbpr_workgroup_size);

    for i in 0..num_subtasks {
        let mut accumulated_point_for_subtask_i = G::zero();
        for j in 0..pbpr_workgroup_size {
            let flat_idx = i * pbpr_workgroup_size + j;
            let limb_start_idx = flat_idx * msm_config.num_limbs;
            let limb_end_idx = (flat_idx + 1) * msm_config.num_limbs;

            let xr_limbs_u32 = &gpu_g_points_x[limb_start_idx..limb_end_idx];
            let yr_limbs_u32 = &gpu_g_points_y[limb_start_idx..limb_end_idx];
            let zr_limbs_u32 = &gpu_g_points_z[limb_start_idx..limb_end_idx];

            // Convert u32 limbs to BigInt<4> for raw_reduction
            let xr_bigint_mont = BigInt::<4>::from_limbs(&xr_limbs_u32, msm_config.log_limb_size);
            let yr_bigint_mont = BigInt::<4>::from_limbs(&yr_limbs_u32, msm_config.log_limb_size);
            let zr_bigint_mont = BigInt::<4>::from_limbs(&zr_limbs_u32, msm_config.log_limb_size);

            let xr_bigint_std = raw_reduction(xr_bigint_mont.clone());
            let yr_bigint_std = raw_reduction(yr_bigint_mont.clone());
            let zr_bigint_std = raw_reduction(zr_bigint_mont.clone());

            let result_xr =
                BaseField::from_bigint(xr_bigint_std).unwrap_or_else(|| BaseField::zero());
            let result_yr =
                BaseField::from_bigint(yr_bigint_std).unwrap_or_else(|| BaseField::zero());
            let result_zr =
                BaseField::from_bigint(zr_bigint_std).unwrap_or_else(|| BaseField::zero());

            let partial_g_point = G::new(result_xr, result_yr, result_zr); // use the unchecked version once we have the correct implementation
                                                                           // let partial_g_point = G::new_unchecked(result_xr, result_yr, result_zr);
            accumulated_point_for_subtask_i += partial_g_point;
        }
        gpu_points.push(accumulated_point_for_subtask_i);
    }

    println!("\n--- Stage 5: Horner's Method (CPU) ---");
    println!("Input decoded gpu_points len: {}", gpu_points.len());
    println!("Input chunk_size: {}", chunk_size);

    //-------------------------------------------------------
    // 5) Horner's Method (CPU)
    //    => final GPU-based MSM result
    //-------------------------------------------------------
    let m = ScalarField::from(1u64 << chunk_size);
    let mut result = gpu_points[gpu_points.len() - 1];
    if gpu_points.len() > 1 {
        for i in (0..gpu_points.len() - 1).rev() {
            result *= m;
            result += gpu_points[i];
        }
    }

    Ok(result)
}

//-------------------------------------------------------------------------------------
// GPU Stage 1: Convert & Decompose
//   total threads => `input_size` (1D dispatch)
//-------------------------------------------------------------------------------------
fn convert_decompose_gpu(
    helper: &mut MetalHelper,
    config: &MetalConfig,
    coords: &Vec<u32>,  // packed X & Y halfwords
    scalars: &Vec<u32>, // packed scalar halfwords
    input_size: usize,
    num_subtasks: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // Create input buffers
    let coords_buf = helper.create_input_buffer(coords);
    let scalars_buf = helper.create_input_buffer(scalars);

    // Output buffers: point_x, point_y, scalar_chunks
    //   - Each X or Y is stored as num_limbs "fullwords"; your kernel may want them in halfwords
    //   - Each scalar_chunk is `input_size * num_subtasks` length (u32).
    let out_point_x = helper.create_output_buffer(input_size * config.num_limbs);
    let out_point_y = helper.create_output_buffer(input_size * config.num_limbs);
    let out_scalar_chunks = helper.create_output_buffer(input_size * num_subtasks);

    let input_size_buf = helper.create_input_buffer(&vec![input_size as u32]);

    // Compute how many total threads, which is `input_size`. We can do 1D:
    let thread_count = helper.create_thread_group_size(input_size as u64, 1, 1);
    let threads_per_group = helper.create_thread_group_size(1, 1, 1);

    // Dispatch
    helper.execute_shader(
        config,
        &[
            &coords_buf,     // buffer(0)
            &scalars_buf,    // buffer(1)
            &input_size_buf, // buffer(2)
        ],
        &[
            &out_point_x,       // buffer(X)
            &out_point_y,       // buffer(Y)
            &out_scalar_chunks, // buffer(chunks)
        ],
        &thread_count,
        &threads_per_group,
    );

    // Read back
    let gpu_point_x = helper.read_results(&out_point_x, input_size * config.num_limbs);
    let gpu_point_y = helper.read_results(&out_point_y, input_size * config.num_limbs);
    let gpu_scalar_chunks = helper.read_results(&out_scalar_chunks, input_size * num_subtasks);

    (gpu_point_x, gpu_point_y, gpu_scalar_chunks)
}

//-------------------------------------------------------------------------------------
// GPU Stage 2: Transpose (CSR -> CSC) for each subtask
//   total threads => num_subtasks
//-------------------------------------------------------------------------------------
fn transpose_gpu(
    helper: &mut MetalHelper,
    config: &MetalConfig,
    scalar_chunks: &Vec<u32>,
    num_subtasks: usize,
    input_size: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>) {
    let in_chunks_buf = helper.create_input_buffer(scalar_chunks);

    let out_csc_col_ptr =
        helper.create_output_buffer(num_subtasks * ((num_columns + 1) as usize) * 4);
    let out_csc_val_idxs = helper.create_output_buffer(scalar_chunks.len());
    let out_curr = helper.create_output_buffer(num_subtasks * (num_columns as usize) * 4);

    let params = vec![
        num_columns,       // params.x  → n ( #columns per sparse matrix )
        input_size as u32, // params.y  → input_size ( #rows )
    ];
    let params_buf = helper.create_input_buffer(&params);

    // 1D dispatch => #threads = num_subtasks
    let thread_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let threads_per_group = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        config,
        &[
            &in_chunks_buf,    // buffer(0)
            &out_csc_col_ptr,  // buffer(1)
            &out_csc_val_idxs, // buffer(2)
            &out_curr,         // buffer(3)
            &params_buf,       // buffer(4)
        ],
        &[],
        &thread_count,
        &threads_per_group,
    );

    // Read results
    let gpu_csc_col_ptr = helper.read_results(
        &out_csc_col_ptr,
        num_subtasks * ((num_columns + 1) as usize) * 4,
    );
    let gpu_csc_val_idxs = helper.read_results(&out_csc_val_idxs, scalar_chunks.len());

    (gpu_csc_col_ptr, gpu_csc_val_idxs)
}

//-------------------------------------------------------------------------------------
// GPU Stage 3: SMVP
//   total threads => half_columns * num_subtasks
//
// This accumulates each "positive/negative bucket."  Our typical kernel does
// a loop (j=0..1) for each thread to handle ± buckets, then merges them in place.
//-------------------------------------------------------------------------------------
fn smvp_gpu(
    helper: &mut MetalHelper,
    config: &MetalConfig,
    gpu_csc_col_ptr: &Vec<u32>,
    gpu_csc_val_idxs: &Vec<u32>,
    gpu_point_x: &Vec<u32>,
    gpu_point_y: &Vec<u32>,
    input_size: usize,
    num_subtasks: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let half_columns = num_columns / 2;

    let mut s_workgroup_size = 256u32;
    let mut s_num_x_workgroups = 64u32;
    let mut s_num_y_workgroups = half_columns / s_workgroup_size / s_num_x_workgroups;
    let mut s_num_z_workgroups = num_subtasks as u32;

    if half_columns < 32768 {
        s_workgroup_size = 32;
        s_num_x_workgroups = 1;
        s_num_y_workgroups = half_columns / s_workgroup_size / s_num_x_workgroups;
    }

    if num_columns < 256 {
        s_workgroup_size = 1;
        s_num_x_workgroups = half_columns;
        s_num_y_workgroups = 1;
        s_num_z_workgroups = 1;
    }

    // This is a dynamic variable that determines the number of CSR
    // matrices processed per invocation of the shader. A safe default is 1.
    let num_subtask_chunk_size = 4u32;

    let bucket_sum_coord_bytelength =
        (num_columns / 2) as usize * config.num_limbs as usize * 4 * num_subtasks as usize;

    // Input buffers
    let row_ptr_buf = helper.create_input_buffer(gpu_csc_col_ptr);
    let val_idx_buf = helper.create_input_buffer(gpu_csc_val_idxs);
    let point_x_buf = helper.create_input_buffer(gpu_point_x);
    let point_y_buf = helper.create_input_buffer(gpu_point_y);

    // Output "bucket" buffers - create once and reuse
    let bucket_x_buf = helper.create_output_buffer(bucket_sum_coord_bytelength);
    let bucket_y_buf = helper.create_output_buffer(bucket_sum_coord_bytelength);
    let bucket_z_buf = helper.create_output_buffer(bucket_sum_coord_bytelength);

    // Debug print the workgroup configuration
    println!("SMVP Debug Info:");
    println!("  half_columns: {}", half_columns);
    println!("  num_subtasks: {}", num_subtasks);
    println!("  s_workgroup_size: {}", s_workgroup_size);
    println!("  s_num_x_workgroups: {}", s_num_x_workgroups);
    println!("  s_num_y_workgroups: {}", s_num_y_workgroups);
    println!("  s_num_z_workgroups: {}", s_num_z_workgroups);
    println!("  num_subtask_chunk_size: {}", num_subtask_chunk_size);

    // Loop through subtask chunks like in the reference
    for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
        // Uniform params => [input_size, num_y_workgroups, num_z_workgroups, offset]
        let params = vec![
            input_size as u32,
            s_num_y_workgroups,
            s_num_z_workgroups,
            offset,
        ];
        let params_buf = helper.create_input_buffer(&params);

        let adjusted_s_num_x_workgroups = if num_columns < 256 {
            s_num_x_workgroups
        } else if num_subtasks as u32 >= num_subtask_chunk_size {
            std::cmp::max(
                1,
                s_num_x_workgroups / (num_subtasks as u32 / num_subtask_chunk_size),
            )
        } else {
            s_num_x_workgroups
        };

        println!("SMVP Execution Debug:");
        println!("  offset: {}", offset);
        println!(
            "  adjusted_s_num_x_workgroups: {}",
            adjusted_s_num_x_workgroups
        );
        println!(
            "  thread_group_count: ({}, {}, {})",
            adjusted_s_num_x_workgroups, s_num_y_workgroups, s_num_z_workgroups
        );
        println!("  threads_per_group: ({}, 1, 1)", s_workgroup_size);

        let thread_group_count = helper.create_thread_group_size(
            adjusted_s_num_x_workgroups as u64,
            s_num_y_workgroups as u64,
            s_num_z_workgroups as u64,
        );
        let threads_per_group = helper.create_thread_group_size(s_workgroup_size as u64, 1, 1);

        println!("executing smvp_gpu, offset: {}", offset);
        helper.execute_shader(
            config,
            &[
                &row_ptr_buf,
                &val_idx_buf,
                &point_x_buf,
                &point_y_buf,
                &bucket_x_buf,
                &bucket_y_buf,
                &bucket_z_buf,
                &params_buf,
            ],
            &[],
            &thread_group_count,
            &threads_per_group,
        );
    }

    let out_x = helper.read_results(&bucket_x_buf, bucket_sum_coord_bytelength);
    let out_y = helper.read_results(&bucket_y_buf, bucket_sum_coord_bytelength);
    let out_z = helper.read_results(&bucket_z_buf, bucket_sum_coord_bytelength);

    (out_x, out_y, out_z)
}

//-------------------------------------------------------------------------------------
// GPU Stage 4: Parallel Bucket Reduction (PBPR)
//   total threads => 256 * num_subtasks
//   done in 2 sub-stages
//-------------------------------------------------------------------------------------
fn pbpr_gpu(
    helper: &mut MetalHelper,
    stage1_config: &MetalConfig,
    stage2_config: &MetalConfig,
    bucket_x: &Vec<u32>,
    bucket_y: &Vec<u32>,
    bucket_z: &Vec<u32>,
    num_subtasks: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let num_subtasks_per_bpr_1 = 16;

    let b_num_x_workgroups = num_subtasks_per_bpr_1;
    let b_num_y_workgroups = 1;
    let b_num_z_workgroups = 1;
    let b_workgroup_size = 256;

    let bucket_sum_x_buf = helper.create_input_buffer(bucket_x);
    let bucket_sum_y_buf = helper.create_input_buffer(bucket_y);
    let bucket_sum_z_buf = helper.create_input_buffer(bucket_z);

    // Buffers that store the bucket points reduction (BPR) output.
    let g_points_coord_bytelength =
        (num_subtasks * b_workgroup_size * stage1_config.num_limbs) as usize * 4;

    let g_points_x_buf = helper.create_output_buffer(g_points_coord_bytelength);
    let g_points_y_buf = helper.create_output_buffer(g_points_coord_bytelength);
    let g_points_z_buf = helper.create_output_buffer(g_points_coord_bytelength);

    // Stage 1
    for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1 as usize) {
        println!("subtask_chunk_idx: {}", subtask_chunk_idx);
        let params = vec![
            subtask_chunk_idx as u32,
            num_columns,
            num_subtasks_per_bpr_1,
        ];
        let params_buf = helper.create_input_buffer(&params);
        let workgroup_size_buf = helper.create_input_buffer(&vec![b_workgroup_size as u32]);

        let stage1_group_count = helper.create_thread_group_size(
            b_num_x_workgroups as u64,
            b_num_y_workgroups as u64,
            b_num_z_workgroups as u64,
        );
        let stage1_group_size = helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

        println!("executing stage1_config");
        helper.execute_shader(
            stage1_config,
            &[
                &bucket_sum_x_buf,
                &bucket_sum_y_buf,
                &bucket_sum_z_buf,
                &g_points_x_buf,
                &g_points_y_buf,
                &g_points_z_buf,
                &params_buf,
                &workgroup_size_buf,
            ],
            &[],
            &stage1_group_count,
            &stage1_group_size,
        );
    }

    // Stage 2
    let num_subtasks_per_bpr_2 = 16;
    let b_2_num_x_workgroups = num_subtasks_per_bpr_2;

    for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_2 as usize) {
        println!("subtask_chunk_idx: {}", subtask_chunk_idx);
        let params = vec![subtask_chunk_idx as u32, num_columns, b_2_num_x_workgroups];
        let params_buf = helper.create_input_buffer(&params);
        let workgroup_size_buf = helper.create_input_buffer(&vec![b_workgroup_size as u32]);

        let stage2_group_count = helper.create_thread_group_size(b_2_num_x_workgroups as u64, 1, 1);
        let stage2_group_size = helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

        println!("executing stage2_config");
        helper.execute_shader(
            stage2_config,
            &[
                &bucket_sum_x_buf,
                &bucket_sum_y_buf,
                &bucket_sum_z_buf,
                &g_points_x_buf,
                &g_points_y_buf,
                &g_points_z_buf,
                &params_buf,
                &workgroup_size_buf,
            ],
            &[],
            &stage2_group_count,
            &stage2_group_size,
        );
    }

    // Read back final
    let out_gx = helper.read_results(&g_points_x_buf, g_points_coord_bytelength);
    let out_gy = helper.read_results(&g_points_y_buf, g_points_coord_bytelength);
    let out_gz = helper.read_results(&g_points_z_buf, g_points_coord_bytelength);

    (out_gx, out_gy, out_gz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    #[test]
    fn test_e2e_msm_pipeline() {
        let log_input_size = 16;
        let input_size = 1 << log_input_size;

        let num_threads = rayon::current_num_threads();
        let thread_chunk_size = (input_size + num_threads - 1) / num_threads;
        println!(
            "Generating {} elements using {} threads",
            input_size, num_threads
        );
        let start = std::time::Instant::now();

        // Generate bases and scalars in parallel
        let (bases, scalars): (Vec<_>, Vec<_>) = (0..num_threads)
            .into_par_iter()
            .flat_map(|thread_id| {
                let mut rng = test_rng();

                let start_idx = thread_id * thread_chunk_size;
                let end_idx = std::cmp::min(start_idx + thread_chunk_size, input_size);
                let current_thread_size = end_idx - start_idx;

                (0..current_thread_size)
                    .map(|_| {
                        let base = G::rand(&mut rng).into_affine();
                        let scalar = ScalarField::rand(&mut rng);
                        (base, scalar)
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        println!("Generated {} elements in {:?}", input_size, start.elapsed());

        println!("running metal_e2e_msm");
        let start = std::time::Instant::now();
        let result = metal_e2e_msm(&bases, &scalars).unwrap();
        println!("metal_e2e_msm took {:?}", start.elapsed());

        println!("running arkworks_msm");
        let start = std::time::Instant::now();
        let arkworks_msm = G::msm(&bases, &scalars).unwrap();
        println!("arkworks_msm took {:?}", start.elapsed());

        assert_eq!(
            result, arkworks_msm,
            "Mismatch between GPU e2e result and Arkworks reference"
        );
        println!("GPU e2e result matches Arkworks reference");
    }
}
