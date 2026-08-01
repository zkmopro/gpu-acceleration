use crate::msm::metal_msm::host::metal_wrapper::{MetalConfig, MetalHelper};
use crate::msm::metal_msm::host::shader_manager::{ShaderManager, ShaderManagerConfig, ShaderType};
use crate::msm::metal_msm::utils::limbs_conversion::{
    pack_affine_and_scalars, GenericLimbConversion,
};
use crate::msm::metal_msm::utils::mont_reduction::raw_reduction;
use crate::msm::metal_msm::utils::window_size_optimizer::fetch_gpu_core_count_and_simd_width_from_device;
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as Affine, G1Projective as G};
use ark_ff::{BigInt, PrimeField};
use ark_std::{vec::Vec, Zero};
use rayon::prelude::*;
use std::error::Error;

/// Configuration for Metal MSM pipeline
#[derive(Clone, Debug)]
pub struct MetalMSMConfig {
    pub num_limbs: usize,
    pub log_limb_size: u32,
}

impl Default for MetalMSMConfig {
    fn default() -> Self {
        Self {
            num_limbs: 8,
            log_limb_size: 32,
        }
    }
}

impl From<MetalMSMConfig> for ShaderManagerConfig {
    fn from(config: MetalMSMConfig) -> Self {
        Self {
            num_limbs: config.num_limbs,
            log_limb_size: config.log_limb_size,
        }
    }
}

/// Main Metal MSM pipeline with pre-compiled shaders
pub struct MetalMSMPipeline {
    config: MetalMSMConfig,
    shader_manager: ShaderManager,
    gpu_cores: usize,
    simd_width: usize,
}

impl MetalMSMPipeline {
    fn new(config: MetalMSMConfig) -> Result<Self, Box<dyn Error>> {
        let shader_config: ShaderManagerConfig = config.clone().into();
        let shader_manager = ShaderManager::new(shader_config)?;

        // Cache GPU core count once during initialization
        let (gpu_cores, simd_width) =
            fetch_gpu_core_count_and_simd_width_from_device(shader_manager.device());

        Ok(Self {
            config,
            shader_manager,
            gpu_cores,
            simd_width,
        })
    }

    fn with_default_config() -> Result<Self, Box<dyn Error>> {
        Self::new(MetalMSMConfig::default())
    }

    /// Get the cached GPU core count
    pub fn gpu_cores(&self) -> usize {
        self.gpu_cores
    }

    /// Execute the complete MSM pipeline on GPU.
    ///
    /// All stages are encoded up front and committed back-to-back; intermediate
    /// results stay in GPU buffers (unified memory) and only the final
    /// per-subtask partial sums are read back to the host.
    fn execute_pipeline(
        &self,
        bases: &[Affine],
        scalars: &[ScalarField],
        input_size: usize,
        window_size: usize,
        scale_factor: usize,
    ) -> Result<G, Box<dyn Error>> {
        use crate::msm::metal_msm::host::metal_wrapper::ShaderOperation;

        // these params are set based on the Alogorithm 3 in the cuZK paper (https://eprint.iacr.org/2022/1321.pdf)
        let num_columns = 1 << window_size;
        let num_subtasks =
            (ScalarField::MODULUS_BIT_SIZE as f32 / window_size as f32).ceil() as usize;
        let num_limbs = self.config.num_limbs;
        let half_columns = num_columns / 2;

        // Stage 0: Pack inputs
        let metal_config = MetalConfig {
            log_limb_size: self.config.log_limb_size,
            num_limbs,
            shader_file: String::new(),
            kernel_name: String::new(),
        };
        let (coords, scals) = pack_affine_and_scalars(bases, scalars, &metal_config);

        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());

        // Input buffers
        let coords_buf = helper.create_buffer(&coords);
        let scalars_buf = helper.create_buffer(&scals);

        // Intermediate buffers (device-resident, never read back)
        let scalar_chunks_buf = helper.create_zeroed_buffer(input_size * num_subtasks);
        let csc_col_ptr_buf = helper.create_zeroed_buffer(num_subtasks * (num_columns + 1));
        let csc_val_idxs_buf = helper.create_zeroed_buffer(input_size * num_subtasks);
        let curr_buf = helper.create_zeroed_buffer(num_subtasks * num_columns);
        let bucket_size = half_columns * num_limbs * num_subtasks;
        let bucket_x_buf = helper.create_zeroed_buffer(bucket_size);
        let bucket_y_buf = helper.create_zeroed_buffer(bucket_size);
        let bucket_zz_buf = helper.create_zeroed_buffer(bucket_size);
        let bucket_zzz_buf = helper.create_zeroed_buffer(bucket_size);

        // Output buffers
        // g_points are reduced to one point per subtask on the GPU
        // (bpr_reduce), so b_workgroup_size no longer adds host-side work.
        let b_workgroup_size = 256usize.min(half_columns);
        let g_points_size = num_subtasks * b_workgroup_size * num_limbs;
        let g_points_x_buf = helper.create_zeroed_buffer(g_points_size);
        let g_points_y_buf = helper.create_zeroed_buffer(g_points_size);
        let g_points_zz_buf = helper.create_zeroed_buffer(g_points_size);
        let g_points_zzz_buf = helper.create_zeroed_buffer(g_points_size);
        let subtask_sum_size = num_subtasks * num_limbs;
        let subtask_x_buf = helper.create_zeroed_buffer(subtask_sum_size);
        let subtask_y_buf = helper.create_zeroed_buffer(subtask_sum_size);
        let subtask_zz_buf = helper.create_zeroed_buffer(subtask_sum_size);
        let subtask_zzz_buf = helper.create_zeroed_buffer(subtask_sum_size);

        let mut ops: Vec<ShaderOperation> = Vec::new();

        // Stage 1: Scalar decomposition (coords pass through to SMVP untouched)
        let convert_shader = self
            .shader_manager
            .get_shader(&ShaderType::ConvertPointAndDecompose)
            .ok_or("ConvertPointAndDecompose shader not found")?;

        let c_workgroup_size = self.simd_width * scale_factor;
        let c_num_x_workgroups = c_workgroup_size;
        let c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;

        let convert_params_buf = helper.create_buffer(&vec![
            input_size as u32,
            window_size as u32,
            num_columns as u32,
            num_subtasks as u32,
        ]);

        ops.push(ShaderOperation {
            pipeline_state: convert_shader.pipeline_state.clone(),
            buffers: vec![&scalars_buf, &scalar_chunks_buf, &convert_params_buf],
            thread_group_count: helper.create_thread_group_size(
                c_num_x_workgroups as u64,
                c_num_y_workgroups.max(1) as u64,
                1,
            ),
            threads_per_threadgroup: helper.create_thread_group_size(c_workgroup_size as u64, 1, 1),
        });

        // Stage 2: Transpose (parallel histogram -> prefix sum -> scatter)
        let histogram_shader = self
            .shader_manager
            .get_shader(&ShaderType::TransposeHistogram)
            .ok_or("TransposeHistogram shader not found")?;
        let prefix_sum_shader = self
            .shader_manager
            .get_shader(&ShaderType::TransposePrefixSum)
            .ok_or("TransposePrefixSum shader not found")?;
        let scatter_shader = self
            .shader_manager
            .get_shader(&ShaderType::TransposeScatter)
            .ok_or("TransposeScatter shader not found")?;

        let transpose_params_buf =
            helper.create_buffer(&vec![num_columns as u32, input_size as u32]);

        let t_threadgroup = 256usize.min(input_size);
        let t_groups_x = (input_size + t_threadgroup - 1) / t_threadgroup;

        ops.push(ShaderOperation {
            pipeline_state: histogram_shader.pipeline_state.clone(),
            buffers: vec![&scalar_chunks_buf, &csc_col_ptr_buf, &transpose_params_buf],
            thread_group_count: helper.create_thread_group_size(
                t_groups_x as u64,
                num_subtasks as u64,
                1,
            ),
            threads_per_threadgroup: helper.create_thread_group_size(t_threadgroup as u64, 1, 1),
        });
        ops.push(ShaderOperation {
            pipeline_state: prefix_sum_shader.pipeline_state.clone(),
            buffers: vec![&csc_col_ptr_buf, &transpose_params_buf],
            thread_group_count: helper.create_thread_group_size(1, 1, 1),
            threads_per_threadgroup: helper.create_thread_group_size(num_subtasks as u64, 1, 1),
        });
        ops.push(ShaderOperation {
            pipeline_state: scatter_shader.pipeline_state.clone(),
            buffers: vec![
                &scalar_chunks_buf,
                &csc_col_ptr_buf,
                &csc_val_idxs_buf,
                &curr_buf,
                &transpose_params_buf,
            ],
            thread_group_count: helper.create_thread_group_size(
                t_groups_x as u64,
                num_subtasks as u64,
                1,
            ),
            threads_per_threadgroup: helper.create_thread_group_size(t_threadgroup as u64, 1, 1),
        });

        // Stage 3: SMVP.
        //
        // Default: segmented, load-balanced accumulation. Threads own
        // fixed-size slices of the bucket-sorted point stream (uniform work),
        // write complete buckets exclusively and spill at most two partial
        // runs each; a merge kernel combines spills and mirror rows into
        // signed buckets. MSM_LEGACY_SMVP=1 selects the classic
        // one-thread-per-bucket kernel; MSM_SMVP_L overrides the slice length.
        let legacy_smvp = std::env::var("MSM_LEGACY_SMVP").map_or(false, |v| v == "1");
        // buffers referenced by ops must outlive the branch
        let mut smvp_params_bufs = Vec::new();
        let mut smvp_dims: Vec<(u64, u64, u64)> = Vec::new();
        let mut seg_buffers: Vec<metal::Buffer> = Vec::new();
        if legacy_smvp {
            let smvp_shader = self
                .shader_manager
                .get_shader(&ShaderType::SMVP)
                .ok_or("SMVP shader not found")?;

            let s_workgroup_size = self.simd_width * scale_factor;
            let num_subtask_chunk_size = num_subtasks as u32; // single dispatch: subtask tails overlap
            for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
                let remaining_subtasks = (num_subtasks as u32 - offset).min(num_subtask_chunk_size);
                let valid_threads = half_columns as u64 * remaining_subtasks as u64;

                let max_y = ((valid_threads as usize)
                    / (s_workgroup_size * remaining_subtasks as usize))
                    .max(1);
                let s_num_y_workgroups = self.simd_width.min(max_y) as u64;
                let s_num_z_workgroups = remaining_subtasks as u64;

                let threads_per_grid =
                    (s_workgroup_size as u64) * s_num_y_workgroups * s_num_z_workgroups;
                let s_num_x_workgroups = (valid_threads + threads_per_grid - 1) / threads_per_grid; // ceil div

                let params_buf = helper.create_buffer(&vec![
                    input_size as u32,
                    num_columns as u32,
                    num_subtasks as u32,
                    offset,
                ]);
                smvp_params_bufs.push(params_buf);
                smvp_dims.push((s_num_x_workgroups, s_num_y_workgroups, s_num_z_workgroups));
            }
            for (params_buf, &(x, y, z)) in smvp_params_bufs.iter().zip(smvp_dims.iter()) {
                ops.push(ShaderOperation {
                    pipeline_state: smvp_shader.pipeline_state.clone(),
                    buffers: vec![
                        &csc_col_ptr_buf,
                        &csc_val_idxs_buf,
                        &coords_buf,
                        &bucket_x_buf,
                        &bucket_y_buf,
                        &bucket_zz_buf,
                        &bucket_zzz_buf,
                        params_buf,
                    ],
                    thread_group_count: helper.create_thread_group_size(x, y, z),
                    threads_per_threadgroup: helper
                        .create_thread_group_size(s_workgroup_size as u64, 1, 1),
                });
            }
        } else {
            let accumulate_shader = self
                .shader_manager
                .get_shader(&ShaderType::SMVPAccumulate)
                .ok_or("SMVPAccumulate shader not found")?;
            let merge_shader = self
                .shader_manager
                .get_shader(&ShaderType::SMVPMerge)
                .ok_or("SMVPMerge shader not found")?;

            let seg_len: usize = std::env::var("MSM_SMVP_L")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128);
            let num_segments = (input_size + seg_len - 1) / seg_len;

            // Process subtasks in chunks, reusing the row-sum/spill buffers
            // across chunks: divides their footprint by num_subtasks/chunk at
            // the cost of a small pipeline bubble (merge of chunk k must
            // finish before accumulate of chunk k+1 reuses the buffers, which
            // Metal's hazard tracking enforces). Stale data is safe: row sums
            // are only read for rows direct-written in the same chunk, and
            // every spill slot is rewritten (value or sentinel) every chunk.
            let subtask_chunk = if num_columns >= 32768 { 4 } else { num_subtasks };

            let row_sums_size = subtask_chunk * num_columns * num_limbs;
            let spill_size = subtask_chunk * num_segments * 2 * num_limbs;
            // 0..=3 row sums, 4..=7 spills, 8 spill rows
            for _ in 0..4 {
                let b = helper.create_zeroed_buffer(row_sums_size);
                seg_buffers.push(b);
            }
            for _ in 0..4 {
                let b = helper.create_zeroed_buffer(spill_size);
                seg_buffers.push(b);
            }
            let b = helper.create_zeroed_buffer(subtask_chunk * num_segments * 2);
            seg_buffers.push(b);
            // one params buffer per chunk, appended after the fixed 9 slots
            for offset in (0..num_subtasks).step_by(subtask_chunk) {
                let b = helper.create_buffer(&vec![
                    input_size as u32,
                    num_columns as u32,
                    seg_len as u32,
                    offset as u32,
                ]);
                seg_buffers.push(b);
            }

            let a_tg = 64usize;
            for (chunk_idx, offset) in (0..num_subtasks).step_by(subtask_chunk).enumerate() {
                let live = subtask_chunk.min(num_subtasks - offset);
                let params_ref = &seg_buffers[9 + chunk_idx];
                ops.push(ShaderOperation {
                    pipeline_state: accumulate_shader.pipeline_state.clone(),
                    buffers: vec![
                        &csc_col_ptr_buf,
                        &csc_val_idxs_buf,
                        &scalar_chunks_buf,
                        &coords_buf,
                        &seg_buffers[0],
                        &seg_buffers[1],
                        &seg_buffers[2],
                        &seg_buffers[3],
                        &seg_buffers[4],
                        &seg_buffers[5],
                        &seg_buffers[6],
                        &seg_buffers[7],
                        &seg_buffers[8],
                        params_ref,
                    ],
                    thread_group_count: helper.create_thread_group_size(
                        ((num_segments + a_tg - 1) / a_tg) as u64,
                        live as u64,
                        1,
                    ),
                    threads_per_threadgroup: helper.create_thread_group_size(a_tg as u64, 1, 1),
                });
                ops.push(ShaderOperation {
                    pipeline_state: merge_shader.pipeline_state.clone(),
                    buffers: vec![
                        &csc_col_ptr_buf,
                        &seg_buffers[0],
                        &seg_buffers[1],
                        &seg_buffers[2],
                        &seg_buffers[3],
                        &seg_buffers[4],
                        &seg_buffers[5],
                        &seg_buffers[6],
                        &seg_buffers[7],
                        &seg_buffers[8],
                        &bucket_x_buf,
                        &bucket_y_buf,
                        &bucket_zz_buf,
                        &bucket_zzz_buf,
                        params_ref,
                    ],
                    thread_group_count: helper.create_thread_group_size(
                        ((half_columns + a_tg - 1) / a_tg) as u64,
                        live as u64,
                        1,
                    ),
                    threads_per_threadgroup: helper.create_thread_group_size(a_tg as u64, 1, 1),
                });
            }
        }

        // Stage 4: Parallel Bucket Reduction (stage 1 then stage 2)
        let bpr1_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage1)
            .ok_or("BPRStage1 shader not found")?;
        let bpr2_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage2)
            .ok_or("BPRStage2 shader not found")?;

        let num_subtasks_per_bpr = num_subtasks;
        let mut bpr_params_bufs = Vec::new();
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr) {
            let params_buf = helper.create_buffer(&vec![
                subtask_chunk_idx as u32,
                num_columns as u32,
                num_subtasks_per_bpr as u32,
                0u32, // dummy 4 bytes to align the Metal's uint3 16-byte style
            ]);
            bpr_params_bufs.push(params_buf);
        }
        for stage_shader in [bpr1_shader, bpr2_shader] {
            for params_buf in bpr_params_bufs.iter() {
                ops.push(ShaderOperation {
                    pipeline_state: stage_shader.pipeline_state.clone(),
                    buffers: vec![
                        &bucket_x_buf,
                        &bucket_y_buf,
                        &bucket_zz_buf,
                        &bucket_zzz_buf,
                        &g_points_x_buf,
                        &g_points_y_buf,
                        &g_points_zz_buf,
                        &g_points_zzz_buf,
                        params_buf,
                    ],
                    thread_group_count: helper.create_thread_group_size(
                        num_subtasks_per_bpr as u64,
                        1,
                        1,
                    ),
                    threads_per_threadgroup: helper
                        .create_thread_group_size(b_workgroup_size as u64, 1, 1),
                });
            }
        }

        // Stage 4b: tree-reduce each subtask's partial sums to a single point
        let reduce_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRReduce)
            .ok_or("BPRReduce shader not found")?;
        let reduce_params_buf =
            helper.create_buffer(&vec![b_workgroup_size as u32, 0u32]);
        ops.push(ShaderOperation {
            pipeline_state: reduce_shader.pipeline_state.clone(),
            buffers: vec![
                &g_points_x_buf,
                &g_points_y_buf,
                &g_points_zz_buf,
                &g_points_zzz_buf,
                &subtask_x_buf,
                &subtask_y_buf,
                &subtask_zz_buf,
                &subtask_zzz_buf,
                &reduce_params_buf,
            ],
            thread_group_count: helper.create_thread_group_size(num_subtasks as u64, 1, 1),
            threads_per_threadgroup: helper.create_thread_group_size(128, 1, 1),
        });

        // Run everything with a single host sync at the end
        helper.execute_shaders_pipelined(ops);

        let subtask_x = helper.read_results(&subtask_x_buf, subtask_sum_size);
        let subtask_y = helper.read_results(&subtask_y_buf, subtask_sum_size);
        let subtask_zz = helper.read_results(&subtask_zz_buf, subtask_sum_size);
        let subtask_zzz = helper.read_results(&subtask_zzz_buf, subtask_sum_size);

        helper.drop_all_buffers();

        // Stage 5 (on CPU): Horner over one point per subtask
        let result = self.final_reduction(
            &subtask_x,
            &subtask_y,
            &subtask_zz,
            &subtask_zzz,
            num_subtasks,
            window_size,
        )?;

        Ok(result)
    }

    /// Final reduction on CPU with Horner's method over one point per
    /// subtask (already tree-reduced on the GPU).
    ///
    /// GPU points arrive in XYZZ coordinates (X, Y, ZZ = z^2, ZZZ = z^3,
    /// Montgomery form). Rather than inverting ZZ/ZZZ, scale to an
    /// equivalent Jacobian representative:
    /// (X*ZZ*ZZZ^2, Y*ZZ^3*ZZZ^2, ZZ*ZZZ).
    fn final_reduction(
        &self,
        subtask_x: &[u32],
        subtask_y: &[u32],
        subtask_zz: &[u32],
        subtask_zzz: &[u32],
        num_subtasks: usize,
        window_size: usize,
    ) -> Result<G, Box<dyn Error>> {
        let field_from_limbs = |limbs: &[u32]| {
            let reduced = raw_reduction(BigInt::<4>::from_limbs(limbs, self.config.log_limb_size));
            BaseField::from_bigint(reduced).unwrap_or_else(BaseField::zero)
        };

        let gpu_points: Vec<G> = (0..num_subtasks)
            .map(|i| {
                let limb_start_idx = i * self.config.num_limbs;
                let limb_end_idx = (i + 1) * self.config.num_limbs;

                let zz = field_from_limbs(&subtask_zz[limb_start_idx..limb_end_idx]);
                if zz.is_zero() {
                    return G::zero();
                }
                let x = field_from_limbs(&subtask_x[limb_start_idx..limb_end_idx]);
                let y = field_from_limbs(&subtask_y[limb_start_idx..limb_end_idx]);
                let zzz = field_from_limbs(&subtask_zzz[limb_start_idx..limb_end_idx]);

                let zzz2 = zzz * zzz;
                let xj = x * zz * zzz2;
                let yj = y * zz * zz * zz * zzz2;
                let zj = zz * zzz;

                G::new(xj, yj, zj)
            })
            .collect();

        // Horner's method
        let m = ScalarField::from(1u64 << window_size);
        let mut result = gpu_points[gpu_points.len() - 1];

        if gpu_points.len() > 1 {
            for i in (0..gpu_points.len() - 1).rev() {
                result *= m;
                result += gpu_points[i];
            }
        }

        Ok(result)
    }
}

thread_local! {
    /// Pipeline (device, compiled pipeline states, GPU info) is expensive to
    /// build, so cache it per thread and reuse it across MSM calls.
    static PIPELINE: std::cell::OnceCell<MetalMSMPipeline> = std::cell::OnceCell::new();
}

/// Convenient wrapper that mimics the Arkworks VariableBaseMSM interface
/// Usage: metal_variable_base_msm(&bases, &scalars)
pub fn metal_variable_base_msm(
    mut bases: &[ark_bn254::G1Affine],
    mut scalars: &[ark_bn254::Fr],
) -> Result<ark_bn254::G1Projective, Box<dyn Error>> {
    // Handle empty input case
    if bases.is_empty() || scalars.is_empty() {
        return Err("Empty input".into());
    }

    // Ensure bases and scalars have the same length
    if bases.len() != scalars.len() {
        let min_len = std::cmp::min(bases.len(), scalars.len());
        bases = &bases[..min_len];
        scalars = &scalars[..min_len];
    }

    let input_size = bases.len();

    // window_size is determined by experiment results
    let window_size = if let Ok(w) = std::env::var("MSM_WINDOW_SIZE") {
        w.parse().unwrap_or(13)
    } else if input_size < 524288 {
        // < 2^19
        13
    } else if input_size <= 16777216 {
        // 2^24
        15
    } else {
        // > 2^24
        16
    };

    // workgroup size will be adjusted based on the scale factor
    // (measured on M-series: smaller workgroups win up to ~2^16, 2x above)
    let scale_factor = if let Ok(s) = std::env::var("MSM_SCALE_FACTOR") {
        s.parse().unwrap_or(1)
    } else if input_size <= 65536 {
        // <= 2^16
        1
    } else if input_size <= 1048576 {
        // 2^20
        2
    } else if input_size <= 16777216 {
        // 2^24
        1 << 2
    } else {
        // > 2^24
        1 << 3
    };

    PIPELINE.with(|cell| {
        if cell.get().is_none() {
            let pipeline = MetalMSMPipeline::with_default_config()?;
            let _ = cell.set(pipeline);
        }
        let pipeline = cell.get().expect("pipeline initialized above");
        pipeline.execute_pipeline(bases, scalars, input_size, window_size, scale_factor)
    })
}

/// Test utilities module - available for both unit tests and integration tests
pub mod test_utils {
    use super::*;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    /// Utility function to generate random bases and scalars for testing
    /// This is made public so it can be used in integration tests
    pub fn generate_random_bases_and_scalars(size: usize) -> (Vec<Affine>, Vec<ScalarField>) {
        let num_threads = rayon::current_num_threads();
        let thread_chunk_size = (size + num_threads - 1) / num_threads;

        let (bases, scalars): (Vec<_>, Vec<_>) = (0..num_threads)
            .into_par_iter()
            .flat_map(|thread_id| {
                let mut rng = test_rng();

                let start_idx = thread_id * thread_chunk_size;
                let end_idx = std::cmp::min(start_idx + thread_chunk_size, size);
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

        (bases, scalars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::VariableBaseMSM;
    use std::time::Instant;

    #[test]
    fn test_metal_msm_pipeline() {
        let log_input_size = 16;
        let input_size = 1 << log_input_size;

        println!("Generating {} elements", input_size);
        let start = Instant::now();
        let (bases, scalars) = test_utils::generate_random_bases_and_scalars(input_size);
        println!("Generated {} elements in {:?}", input_size, start.elapsed());

        println!("running metal_variable_base_msm");
        let start = Instant::now();
        let metal_msm_result = metal_variable_base_msm(&bases, &scalars).unwrap();
        println!("metal_variable_base_msm took {:?}", start.elapsed());

        println!("running arkworks_msm");
        let start = Instant::now();
        let arkworks_msm = G::msm(&bases, &scalars).unwrap();
        println!("arkworks_msm took {:?}", start.elapsed());

        assert_eq!(metal_msm_result, arkworks_msm);
    }
}
