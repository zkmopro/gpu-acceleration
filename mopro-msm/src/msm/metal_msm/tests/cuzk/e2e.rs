use ark_bn254::{G1Affine as Affine, Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::VariableBaseMSM;
use ark_ff::{BigInt, PrimeField};
use ark_std::Zero;
use std::error::Error;
use crate::msm::metal_msm::cuzk_cpu_reproduction::*;
use crate::msm::metal_msm::tests::cuzk::pbpr::closest_power_of_two;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

//-------------------------------------------------------------------------------------
// Example End-to-End Implementation: metal_e2e_msm
//
// This function parallels your "submission.ts" pipeline:
//
//   1) Convert points (X,Y) to Mont form & decompose scalars => (point_x, point_y, scalar_chunks)
//   2) Transpose => build csc_col_ptr + csc_val_idxs
//   3) SMVP => produce per-bucket partial sums (bucket_x, bucket_y, bucket_z)
//   4) PBPR => reduce partial sums within each subtask
//   5) Final CPU Horner => combine subtask results
//
// It compares the final result against an Arkworks reference for correctness.
//-------------------------------------------------------------------------------------

pub fn metal_e2e_msm(
    bases: &[Affine],
    scalars: &[ScalarField],
    force_recompile: bool,
) -> Result<G, Box<dyn Error>> {

    //----------------------------------------
    // Basic Setup: chunk_size, #subtasks, etc.
    //----------------------------------------
    let input_size = bases.len();
    // Decide chunk_size exactly as in submission.ts:
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_subtasks = 256 / chunk_size;
    let num_columns = 1 << chunk_size; // 2^chunk_size
    let half_columns = num_columns / 2;

    // Metal kernel config for BN254 with 13-bit (or 16-bit) limbs, etc.
    let msm_config = MetalConfig {
        // Adjust to your needs:
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "".to_string(),    // Not used in the final e2e, we set these per stage
        kernel_name: "".to_string(),    // Not used in the final e2e, we set these per stage
    };
    let msm_constants = get_or_calc_constants(msm_config.num_limbs, msm_config.log_limb_size);

    //-------------------------------------------------------
    // (A) CPU Reference: Arkworks for final verification
    //-------------------------------------------------------
    // Arkworks "reference" MSM
    let arkworks_msm = G::msm(bases, scalars).unwrap();

    //-------------------------------------------------------
    // (B) Pack input points + scalars => 16-bit halfwords
    //-------------------------------------------------------
    println!("--- Stage B: Packing Inputs ---");
    println!("Input bases len: {}", bases.len());
    println!("Input scalars len: {}", scalars.len());
    let (coords, scals) = pack_affine_and_scalars(bases, scalars, &msm_config);
    println!("Output packed coords len: {}", coords.len());
    println!("Output packed scals len: {}", scals.len());
    println!("Output packed coords sample: {:?}", &coords[0..std::cmp::min(coords.len(), 5)]);
    println!("Output packed scals sample: {:?}", &scals[0..std::cmp::min(scals.len(), 5)]);

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
        force_recompile,
        &coords,
        &scals,
        input_size,
        num_subtasks,
        chunk_size,
    );
    println!("Output gpu_point_x len: {}", gpu_point_x.len());
    println!("Output gpu_point_y len: {}", gpu_point_y.len());
    println!("Output gpu_scalar_chunks len: {}", gpu_scalar_chunks.len());
    println!("Output gpu_point_x sample: {:?}", &gpu_point_x[0..std::cmp::min(gpu_point_x.len(), 5)]);
    println!("Output gpu_point_y sample: {:?}", &gpu_point_y[0..std::cmp::min(gpu_point_y.len(), 5)]);
    println!("Output gpu_scalar_chunks sample: {:?}", &gpu_scalar_chunks[0..std::cmp::min(gpu_scalar_chunks.len(), 5)]);

    // Cleanup
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
        force_recompile,
        &gpu_scalar_chunks,
        num_subtasks,
        input_size,
        num_columns,
    );
    println!("Output gpu_csc_col_ptr len: {}", gpu_csc_col_ptr.len());
    println!("Output gpu_csc_val_idxs len: {}", gpu_csc_val_idxs.len());
    println!("Output gpu_csc_col_ptr sample: {:?}", &gpu_csc_col_ptr[0..std::cmp::min(gpu_csc_col_ptr.len(), 5)]);
    println!("Output gpu_csc_val_idxs sample: {:?}", &gpu_csc_val_idxs[0..std::cmp::min(gpu_csc_val_idxs.len(), 5)]);

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
        force_recompile,
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
    println!("Output gpu_bucket_x sample: {:?}", &gpu_bucket_x[0..std::cmp::min(gpu_bucket_x.len(), 5)]);
    println!("Output gpu_bucket_y sample: {:?}", &gpu_bucket_y[0..std::cmp::min(gpu_bucket_y.len(), 5)]);
    println!("Output gpu_bucket_z sample: {:?}", &gpu_bucket_z[0..std::cmp::min(gpu_bucket_z.len(), 5)]);

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
        shader_file: "cuzk/pbpr_cuzk.metal".to_string(),
        kernel_name: "bpr_stage_1".to_string(),
    };
    let pbpr_stage2_config = MetalConfig {
        log_limb_size: msm_config.log_limb_size,
        num_limbs: msm_config.num_limbs,
        shader_file: "cuzk/pbpr_cuzk.metal".to_string(),
        kernel_name: "bpr_stage_2".to_string(),
    };

    println!("\n--- Stage 4: PBPR (GPU) ---");
    println!("Input gpu_bucket_x len: {}", gpu_bucket_x.len());
    println!("Input gpu_bucket_y len: {}", gpu_bucket_y.len());
    println!("Input gpu_bucket_z len: {}", gpu_bucket_z.len());
    println!("Input num_subtasks: {}", num_subtasks);
    println!("Input num_columns: {}", num_columns);

    let (gpu_g_points_x, gpu_g_points_y, gpu_g_points_z) = pbpr_cuzk_gpu(
        &mut helper4,
        &pbpr_stage1_config,
        &pbpr_stage2_config,
        force_recompile,
        &gpu_bucket_x,
        &gpu_bucket_y,
        &gpu_bucket_z,
        num_subtasks,
        num_columns,
    );
    println!("Output gpu_g_points_x len: {}", gpu_g_points_x.len());
    println!("Output gpu_g_points_y len: {}", gpu_g_points_y.len());
    println!("Output gpu_g_points_z len: {}", gpu_g_points_z.len());
    println!("Output gpu_g_points_x sample: {:?}", &gpu_g_points_x[0..std::cmp::min(gpu_g_points_x.len(), 5)]);
    println!("Output gpu_g_points_y sample: {:?}", &gpu_g_points_y[0..std::cmp::min(gpu_g_points_y.len(), 5)]);
    println!("Output gpu_g_points_z sample: {:?}", &gpu_g_points_z[0..std::cmp::min(gpu_g_points_z.len(), 5)]);

    helper4.drop_all_buffers();

    // At this point, we have `g_points_x, g_points_y, g_points_z` representing the "bucket-sum"
    // partial results for each subtask. We'll decode them to BN254 projective points in CPU.

    // This part could be incorrect
    let gpu_points = crate::msm::metal_msm::tests::cuzk::pbpr::points_from_separated_buffers(
        &gpu_g_points_x,
        &gpu_g_points_y,
        &gpu_g_points_z,
        msm_config.num_limbs,
        msm_config.log_limb_size, // Use the correct log_limb_size
    );

    println!("\n--- Stage 5: Horner's Method (CPU) ---");
    println!("Input decoded gpu_points len: {}", gpu_points.len());
    println!("Input chunk_size: {}", chunk_size);
    println!("Input gpu_points: {:?}", &gpu_points);

    //-------------------------------------------------------
    // 5) Horner's Method (CPU)
    //    => final GPU-based MSM result
    //-------------------------------------------------------
    let m = ScalarField::from(1u64 << chunk_size);
    let mut result = gpu_points[gpu_points.len() - 1];
    for i in (0..gpu_points.len() - 2).rev() {
        result *= m;
        result += gpu_points[i];
    }

    println!("Output final_gpu_result: {:?}", result);

    //-------------------------------------------------------
    // (C) Confirm correctness vs. CPU reference
    //-------------------------------------------------------
    // (C1) We can do a "full CPU pipeline" if we want, or at least check final result:
    // just compare final_gpu_result with Arkworks MSM.
    assert_eq!(
        result, arkworks_msm,
        "Mismatch between GPU e2e result and Arkworks reference"
    );

    println!("✅ e2e MSM pipeline matches Arkworks reference");
    Ok(result)
}

//-------------------------------------------------------------------------------------
// GPU Stage 1: Convert & Decompose
//   total threads => `input_size` (1D dispatch)
//-------------------------------------------------------------------------------------
fn convert_decompose_gpu(
    helper: &mut MetalHelper,
    config: &MetalConfig,
    force_recompile: bool,
    coords: &Vec<u32>,    // packed X & Y halfwords
    scalars: &Vec<u32>,   // packed scalar halfwords
    input_size: usize,
    num_subtasks: usize,
    chunk_size: usize,
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

    // Uniform or params buffer: [input_size, chunk_size, num_subtasks] if needed
    let params = vec![input_size as u32, chunk_size as u32, num_subtasks as u32, 0];
    let params_buf = helper.create_input_buffer(&params);

    // Compute how many total threads, which is `input_size`. We can do 1D:
    let thread_count = helper.create_thread_group_size(input_size as u64, 1, 1);
    let threads_per_group = helper.create_thread_group_size(1, 1, 1);

    // Dispatch
    helper.execute_shader(
        config,
        &[
            &coords_buf,        // buffer(0)
            &scalars_buf,       // buffer(1)
            &params_buf,        // buffer(2) uniform?
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
    force_recompile: bool,
    scalar_chunks: &Vec<u32>,
    num_subtasks: usize,
    input_size: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>) {

    // Input buffer: "scalar_chunks," length = input_size * num_subtasks
    // Output buffers:
    //   - all_csc_col_ptr => length = num_subtasks * (num_columns + 1)
    //   - all_csc_val_idxs => length = input_size * num_subtasks
    let in_chunks_buf = helper.create_input_buffer(scalar_chunks);

    let out_csc_col_ptr = helper.create_output_buffer(num_subtasks * ((num_columns + 1) as usize));
    let out_csc_val_idxs = helper.create_output_buffer(num_subtasks * input_size);

    // We can have a uniform/params buffer with [num_subtasks, input_size, num_columns].
    let params = vec![
        num_subtasks as u32,
        input_size as u32,
        num_columns,
        0,
    ];
    let params_buf = helper.create_input_buffer(&params);

    // 1D dispatch => #threads = num_subtasks
    let thread_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
    let threads_per_group = helper.create_thread_group_size(1, 1, 1);

    // Dispatch
    helper.execute_shader(
        config,
        &[
            &in_chunks_buf,  // buffer(0)
            &params_buf,     // buffer(1) uniform
        ],
        &[
            &out_csc_col_ptr,   // buffer(2)
            &out_csc_val_idxs,  // buffer(3)
        ],
        &thread_count,
        &threads_per_group,
    );

    // Read results
    let gpu_csc_col_ptr = helper.read_results(&out_csc_col_ptr, num_subtasks * ((num_columns + 1) as usize));
    let gpu_csc_val_idxs = helper.read_results(&out_csc_val_idxs, num_subtasks * input_size);

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
    force_recompile: bool,
    gpu_csc_col_ptr: &Vec<u32>,
    gpu_csc_val_idxs: &Vec<u32>,
    gpu_point_x: &Vec<u32>,
    gpu_point_y: &Vec<u32>,
    input_size: usize,
    num_subtasks: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {

    let half_columns = num_columns / 2;
    let total_threads = half_columns as usize * num_subtasks;

    // Input buffers
    let row_ptr_buf = helper.create_input_buffer(gpu_csc_col_ptr);
    let val_idx_buf = helper.create_input_buffer(gpu_csc_val_idxs);
    let point_x_buf = helper.create_input_buffer(gpu_point_x);
    let point_y_buf = helper.create_input_buffer(gpu_point_y);

    // Output "bucket" buffers
    // Each bucket is a Jacobian (X,Y,Z), so 3 * num_limbs each.
    // But in your code, you often store them in separate arrays. We do a separate array for X, Y, Z.
    // total_buckets = half_columns * num_subtasks
    // each X or Y or Z => (total_buckets * num_limbs).
    let total_buckets = half_columns as usize * num_subtasks;
    let bucket_x_buf = helper.create_output_buffer(total_buckets * config.num_limbs);
    let bucket_y_buf = helper.create_output_buffer(total_buckets * config.num_limbs);
    let bucket_z_buf = helper.create_output_buffer(total_buckets * config.num_limbs);

    // Uniform params => [input_size, num_subtasks, (optional) offset...]
    // Exactly as you do in your "smvp.metal".
    let params = vec![
        input_size as u32,         // e.g. needed for indexing
        half_columns,              // for kernel thread offset
        1,                         // sometimes used as num_z_workgroups
        0,                         // subtask_offset=0
    ];
    let params_buf = helper.create_input_buffer(&params);

    // Dispatch => 1D: total_threads, or 2D: (num_subtasks, half_columns)
    let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, half_columns as u64, 1);
    let threads_per_group = helper.create_thread_group_size(1, 1, 1);

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

    let out_x = helper.read_results(&bucket_x_buf, total_buckets * config.num_limbs);
    let out_y = helper.read_results(&bucket_y_buf, total_buckets * config.num_limbs);
    let out_z = helper.read_results(&bucket_z_buf, total_buckets * config.num_limbs);

    (out_x, out_y, out_z)
}

//-------------------------------------------------------------------------------------
// GPU Stage 4: Parallel Bucket Reduction (PBPR)
//   total threads => 256 * num_subtasks
//   done in 2 sub-stages
//-------------------------------------------------------------------------------------
fn pbpr_cuzk_gpu(
    helper: &mut MetalHelper,
    stage1_config: &MetalConfig,
    stage2_config: &MetalConfig,
    force_recompile: bool,
    bucket_x: &Vec<u32>,
    bucket_y: &Vec<u32>,
    bucket_z: &Vec<u32>,
    num_subtasks: usize,
    num_columns: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {

    let workgroup_size: u32 = 256; // from submission.ts
    let num_subtasks_per_bpr_1: u32 = 16;  // or 1, or 16 – your preference
    let num_subtasks_per_bpr_2: u32 = 16;

    // GPU buffers for input bucket sums (read/write):
    println!("creating bucket_sum_x_buf");
    let bucket_sum_x_buf = helper.create_input_buffer(bucket_x);
    println!("creating bucket_sum_y_buf");
    let bucket_sum_y_buf = helper.create_input_buffer(bucket_y);
    println!("creating bucket_sum_z_buf");
    let bucket_sum_z_buf = helper.create_input_buffer(bucket_z);

    // GPU buffers for final G points
    // size = num_subtasks * 256 * num_limbs for each coordinate
    println!("creating g_points_x_buf");
    let g_points_len = num_subtasks * workgroup_size as usize * stage1_config.num_limbs;
    let g_points_x_buf = helper.create_output_buffer(g_points_len);
    println!("creating g_points_y_buf");
    let g_points_y_buf = helper.create_output_buffer(g_points_len);
    println!("creating g_points_z_buf");
    let g_points_z_buf = helper.create_output_buffer(g_points_len);

    let workgroup_size_buf = helper.create_input_buffer(&vec![workgroup_size]);

    // Stage 1
    for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1 as usize) {
        println!("subtask_chunk_idx: {}", subtask_chunk_idx);
        let params = vec![
            subtask_chunk_idx as u32,
            num_columns,                 // e.g. for your kernel
            num_subtasks_per_bpr_1,      // how many subtasks in this chunk
            num_subtasks as u32,
        ];
        let params_buf = helper.create_input_buffer(&params);

        // # threads = workgroup_size * num_subtasks_per_bpr_1
        println!("workgroup_size: {}", workgroup_size);
        println!("num_subtasks_per_bpr_1: {}", num_subtasks_per_bpr_1);
        let stage1_grid_size = workgroup_size * num_subtasks_per_bpr_1;
        println!("stage1_grid_size: {}", stage1_grid_size);
        let stage1_group_count = helper.create_thread_group_size(stage1_grid_size as u64, 1, 1);
        let stage1_group_size = helper.create_thread_group_size(workgroup_size as u64, 1, 1);


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
    for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_2 as usize) {
        println!("subtask_chunk_idx: {}", subtask_chunk_idx);
        let params = vec![
            subtask_chunk_idx as u32,
            num_columns,
            num_subtasks_per_bpr_2,
            num_subtasks as u32,
        ];
        let params_buf = helper.create_input_buffer(&params);

        let stage2_grid_size = workgroup_size * num_subtasks_per_bpr_2;
        let stage2_group_count = helper.create_thread_group_size(stage2_grid_size as u64, 1, 1);
        let stage2_group_size = helper.create_thread_group_size(workgroup_size as u64, 1, 1);

        println!("stage2_grid_size: {}", stage2_grid_size);
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
    let out_gx = helper.read_results(&g_points_x_buf, g_points_len);
    let out_gy = helper.read_results(&g_points_y_buf, g_points_len);
    let out_gz = helper.read_results(&g_points_z_buf, g_points_len);

    (out_gx, out_gy, out_gz)
}

//-------------------------------------------------------------------------------------
// (Optional) Unit test demonstrating usage
//-------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;
    use ark_ec::{Group, CurveGroup};
    use ark_ff::UniformRand;

    #[test]
    fn test_e2e_msm_pipeline() {
        let mut rng = test_rng();
        let input_size = 1;  // or bigger
        let mut bases = Vec::with_capacity(input_size);
        let mut scalars = Vec::with_capacity(input_size);

        // Example: single generator repeated
        for _ in 0..input_size {
            bases.push(G::generator().into_affine());
            scalars.push(ScalarField::from(1u64));
        }
        // Or random
        // for _ in 0..input_size {
        //     bases.push(G::rand(&mut rng).into_affine());
        //     scalars.push(ScalarField::rand(&mut rng));
        // }

        // Full pipeline
        let result = metal_e2e_msm(&bases, &scalars, false).unwrap();
        println!("Final GPU MSM result (input_size={}) => {:?}", input_size, result);
    }
}
