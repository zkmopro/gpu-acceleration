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
            num_limbs: 16,
            log_limb_size: 16,
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
        let point_x_buf = helper.create_zeroed_buffer(input_size * num_limbs);
        let point_y_buf = helper.create_zeroed_buffer(input_size * num_limbs);
        let scalar_chunks_buf = helper.create_zeroed_buffer(input_size * num_subtasks);
        let csc_col_ptr_buf = helper.create_zeroed_buffer(num_subtasks * (num_columns + 1));
        let csc_val_idxs_buf = helper.create_zeroed_buffer(input_size * num_subtasks);
        let curr_buf = helper.create_zeroed_buffer(num_subtasks * num_columns);
        let bucket_size = half_columns * num_limbs * num_subtasks;
        let bucket_x_buf = helper.create_zeroed_buffer(bucket_size);
        let bucket_y_buf = helper.create_zeroed_buffer(bucket_size);
        let bucket_z_buf = helper.create_zeroed_buffer(bucket_size);

        // Output buffers
        let b_workgroup_size = self.simd_width * scale_factor;
        let g_points_size = num_subtasks * b_workgroup_size * num_limbs;
        let g_points_x_buf = helper.create_zeroed_buffer(g_points_size);
        let g_points_y_buf = helper.create_zeroed_buffer(g_points_size);
        let g_points_z_buf = helper.create_zeroed_buffer(g_points_size);

        let mut ops: Vec<ShaderOperation> = Vec::new();

        // Stage 1: Convert Point & Scalar Decomposition
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
            buffers: vec![
                &coords_buf,
                &scalars_buf,
                &point_x_buf,
                &point_y_buf,
                &scalar_chunks_buf,
                &convert_params_buf,
            ],
            thread_group_count: helper.create_thread_group_size(
                c_num_x_workgroups as u64,
                c_num_y_workgroups.max(1) as u64,
                1,
            ),
            threads_per_threadgroup: helper.create_thread_group_size(c_workgroup_size as u64, 1, 1),
        });

        // Stage 2: Transpose
        let transpose_shader = self
            .shader_manager
            .get_shader(&ShaderType::Transpose)
            .ok_or("Transpose shader not found")?;

        let transpose_params_buf =
            helper.create_buffer(&vec![num_columns as u32, input_size as u32]);

        ops.push(ShaderOperation {
            pipeline_state: transpose_shader.pipeline_state.clone(),
            buffers: vec![
                &scalar_chunks_buf,
                &csc_col_ptr_buf,
                &csc_val_idxs_buf,
                &curr_buf,
                &transpose_params_buf,
            ],
            thread_group_count: helper.create_thread_group_size(1, 1, 1),
            threads_per_threadgroup: helper.create_thread_group_size(num_subtasks as u64, 1, 1),
        });

        // Stage 3: SMVP (one dispatch per subtask chunk)
        let smvp_shader = self
            .shader_manager
            .get_shader(&ShaderType::SMVP)
            .ok_or("SMVP shader not found")?;

        let s_workgroup_size = self.simd_width * scale_factor;
        let num_subtask_chunk_size = 4u32;
        let mut smvp_params_bufs = Vec::new();
        let mut smvp_dims = Vec::new();
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
                    &point_x_buf,
                    &point_y_buf,
                    &bucket_x_buf,
                    &bucket_y_buf,
                    &bucket_z_buf,
                    params_buf,
                ],
                thread_group_count: helper.create_thread_group_size(x, y, z),
                threads_per_threadgroup: helper
                    .create_thread_group_size(s_workgroup_size as u64, 1, 1),
            });
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
                        &bucket_z_buf,
                        &g_points_x_buf,
                        &g_points_y_buf,
                        &g_points_z_buf,
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

        // Run everything with a single host sync at the end
        helper.execute_shaders_pipelined(ops);

        let g_points_x = helper.read_results(&g_points_x_buf, g_points_size);
        let g_points_y = helper.read_results(&g_points_y_buf, g_points_size);
        let g_points_z = helper.read_results(&g_points_z_buf, g_points_size);

        helper.drop_all_buffers();

        // Stage 5 (on CPU): read buckets and perform final reduction with Horner's method
        let result = self.final_reduction(
            &g_points_x,
            &g_points_y,
            &g_points_z,
            num_subtasks,
            window_size,
            b_workgroup_size,
        )?;

        Ok(result)
    }

    /// Final reduction on CPU with Horner's method
    fn final_reduction(
        &self,
        g_points_x: &[u32],
        g_points_y: &[u32],
        g_points_z: &[u32],
        num_subtasks: usize,
        window_size: usize,
        pbpr_workgroup_size: usize,
    ) -> Result<G, Box<dyn Error>> {
        // Parallel processing of subtasks
        let gpu_points: Vec<G> = (0..num_subtasks)
            .into_par_iter()
            .map(|i| {
                let mut accumulated_point = G::zero();

                for j in 0..pbpr_workgroup_size {
                    let flat_idx = i * pbpr_workgroup_size + j;
                    let limb_start_idx = flat_idx * self.config.num_limbs;
                    let limb_end_idx = (flat_idx + 1) * self.config.num_limbs;

                    let xr_limbs = &g_points_x[limb_start_idx..limb_end_idx];
                    let yr_limbs = &g_points_y[limb_start_idx..limb_end_idx];
                    let zr_limbs = &g_points_z[limb_start_idx..limb_end_idx];

                    let xr_bigint = BigInt::<4>::from_limbs(xr_limbs, self.config.log_limb_size);
                    let yr_bigint = BigInt::<4>::from_limbs(yr_limbs, self.config.log_limb_size);
                    let zr_bigint = BigInt::<4>::from_limbs(zr_limbs, self.config.log_limb_size);

                    // decoded points from Montgomery form to standard form
                    let xr_reduced = raw_reduction(xr_bigint);
                    let yr_reduced = raw_reduction(yr_bigint);
                    let zr_reduced = raw_reduction(zr_bigint);

                    let x = BaseField::from_bigint(xr_reduced).unwrap_or_else(|| BaseField::zero());
                    let y = BaseField::from_bigint(yr_reduced).unwrap_or_else(|| BaseField::zero());
                    let z = BaseField::from_bigint(zr_reduced).unwrap_or_else(|| BaseField::zero());

                    let point = G::new(x, y, z);
                    accumulated_point += point;
                }

                accumulated_point
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
    let window_size = if input_size < 16384 {
        // 2^14
        8
    } else if input_size < 524288 {
        // 2^19
        13
    } else if input_size <= 16777216 {
        // 2^24
        15
    } else {
        // > 2^24
        16
    };

    // workgroup size will be adjusted based on the scale factor
    let scale_factor = if input_size <= 4096 {
        // 2^12
        1 // 2^0
    } else if input_size <= 65536 {
        // 2^16
        2 // 2^1
    } else if input_size <= 1048576 {
        // 2^20
        1 << 2 // 2^2
    } else if input_size <= 16777216 {
        // 2^24
        1 << 3 // 2^3
    } else {
        // > 2^24
        1 << 4 // 2^4
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
