use crate::msm::metal_msm::utils::limbs_conversion::{
    pack_affine_and_scalars, GenericLimbConversion,
};
use crate::msm::metal_msm::utils::metal_wrapper::{MetalConfig, MetalHelper};
use crate::msm::metal_msm::utils::mont_reduction::raw_reduction;
use crate::msm::metal_msm::utils::shader_manager::{
    ShaderManager, ShaderManagerConfig, ShaderType,
};
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
}

impl MetalMSMPipeline {
    fn new(config: MetalMSMConfig) -> Result<Self, Box<dyn Error>> {
        let shader_config: ShaderManagerConfig = config.clone().into();
        let shader_manager = ShaderManager::new(shader_config)?;

        Ok(Self {
            config,
            shader_manager,
        })
    }

    fn with_default_config() -> Result<Self, Box<dyn Error>> {
        Self::new(MetalMSMConfig::default())
    }

    /// Execute the complete MSM pipeline on GPU
    fn execute(&self, bases: &[Affine], scalars: &[ScalarField]) -> Result<G, Box<dyn Error>> {
        let input_size = bases.len();
        let num_subtasks = 256 / self.config.log_limb_size as usize;
        let num_columns = 1 << self.config.log_limb_size;

        // Stage 0: Pack inputs
        let metal_config = MetalConfig {
            log_limb_size: self.config.log_limb_size,
            num_limbs: self.config.num_limbs,
            shader_file: String::new(),
            kernel_name: String::new(),
        };
        let (coords, scals) = pack_affine_and_scalars(bases, scalars, &metal_config);

        // Stage 1: Convert Point & Scalar Decomposition
        let stage1 = ConvertPointAndScalarDecompose::new(&self.shader_manager);
        let mut c_workgroup_size = 256;
        let mut c_num_x_workgroups = 64;
        let mut c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
        let c_num_z_workgroups = 1;

        if input_size <= 256 {
            c_workgroup_size = input_size;
            c_num_x_workgroups = 1;
            c_num_y_workgroups = 1;
        } else if input_size > 256 && input_size <= 32768 {
            c_workgroup_size = 64;
            c_num_x_workgroups = 4;
            c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
        } else if input_size > 32768 && input_size <= 131072 {
            c_workgroup_size = 256;
            c_num_x_workgroups = 8;
            c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
        } else if input_size > 131072 && input_size < 1048576 {
            c_workgroup_size = 256;
            c_num_x_workgroups = 32;
            c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
        }

        let (point_x, point_y, scalar_chunks) = stage1.execute(
            &coords,
            &scals,
            input_size,
            num_subtasks,
            c_num_x_workgroups,
            c_num_y_workgroups,
            c_num_z_workgroups,
            c_workgroup_size,
        )?;

        // Stage 2: Transpose
        let t_num_x_workgroups = 1;
        let t_num_y_workgroups = 1;
        let t_num_z_workgroups = 1;
        let t_workgroup_size = num_subtasks;

        let stage2 = Transpose::new(&self.shader_manager);
        let (csc_col_ptr, csc_val_idxs) = stage2.execute(
            &scalar_chunks,
            num_subtasks,
            input_size,
            num_columns,
            t_num_x_workgroups,
            t_num_y_workgroups,
            t_num_z_workgroups,
            t_workgroup_size,
        )?;

        // Stage 3: Sparse Matrix-Vector Multiplication
        let s_workgroup_size = 256;
        let s_num_x_workgroups = 64;
        let s_num_y_workgroups = 2;
        let s_num_z_workgroups = num_subtasks;

        let stage3 = SMVP::new(&self.shader_manager);
        let (bucket_x, bucket_y, bucket_z) = stage3.execute(
            &csc_col_ptr,
            &csc_val_idxs,
            &point_x,
            &point_y,
            input_size,
            num_subtasks,
            num_columns,
            s_num_x_workgroups,
            s_num_y_workgroups,
            s_num_z_workgroups,
            s_workgroup_size,
        )?;

        // Stage 4: Parallel Bucket Reduction
        let num_subtasks_per_bpr_1 = 16;
        let num_subtasks_per_bpr_2 = 16;

        let b_num_x_workgroups = num_subtasks_per_bpr_1;
        let b_num_y_workgroups = 1;
        let b_num_z_workgroups = 1;
        let b_workgroup_size = 256;

        let b_2_num_x_workgroups = num_subtasks_per_bpr_2;
        let b_2_num_y_workgroups = 1;
        let b_2_num_z_workgroups = 1;

        let stage4 = PBPR::new(&self.shader_manager);
        let (g_points_x, g_points_y, g_points_z) = stage4.execute(
            &bucket_x,
            &bucket_y,
            &bucket_z,
            num_subtasks,
            num_columns,
            num_subtasks_per_bpr_1,
            num_subtasks_per_bpr_2,
            b_num_x_workgroups,
            b_num_y_workgroups,
            b_num_z_workgroups,
            b_2_num_x_workgroups,
            b_2_num_y_workgroups,
            b_2_num_z_workgroups,
            b_workgroup_size,
        )?;

        // Stage 5: Final reduction and Horner's method
        let result = self.final_reduction(
            &g_points_x,
            &g_points_y,
            &g_points_z,
            num_subtasks,
            self.config.log_limb_size as usize,
            b_workgroup_size,
        )?;

        Ok(result)
    }

    /// Final reduction on CPU
    fn final_reduction(
        &self,
        g_points_x: &[u32],
        g_points_y: &[u32],
        g_points_z: &[u32],
        num_subtasks: usize,
        log_limb_size: usize,
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
        let m = ScalarField::from(1u64 << log_limb_size);
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

/// Stage 1: Convert & Decompose
struct ConvertPointAndScalarDecompose<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> ConvertPointAndScalarDecompose<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        coords: &[u32],
        scalars: &[u32],
        input_size: usize,
        num_subtasks: usize,
        c_num_x_workgroups: usize,
        c_num_y_workgroups: usize,
        c_num_z_workgroups: usize,
        c_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::ConvertPointAndDecompose)
            .ok_or("ConvertPointAndDecompose shader not found")?;

        let coords_buf = helper.create_buffer(&coords.to_vec());
        let scalars_buf = helper.create_buffer(&scalars.to_vec());

        let out_point_x =
            helper.create_empty_buffer(input_size * self.shader_manager.config().num_limbs);
        let out_point_y =
            helper.create_empty_buffer(input_size * self.shader_manager.config().num_limbs);
        let out_scalar_chunks = helper.create_empty_buffer(input_size * num_subtasks);

        let input_size_buf = helper.create_buffer(&vec![input_size as u32]);
        let num_y_workgroups_buf = helper.create_buffer(&vec![c_num_y_workgroups as u32]);

        let thread_group_count = helper.create_thread_group_size(
            c_num_x_workgroups as u64,
            c_num_y_workgroups as u64,
            c_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(c_workgroup_size as u64, 1, 1);

        helper.execute_shader_with_pipeline(
            &shader.pipeline_state,
            &[
                &coords_buf,
                &scalars_buf,
                &input_size_buf,
                &out_point_x,
                &out_point_y,
                &out_scalar_chunks,
                &num_y_workgroups_buf,
            ],
            &thread_group_count,
            &threads_per_threadgroup,
        );

        let point_x = helper.read_results(
            &out_point_x,
            input_size * self.shader_manager.config().num_limbs,
        );
        let point_y = helper.read_results(
            &out_point_y,
            input_size * self.shader_manager.config().num_limbs,
        );
        let scalar_chunks = helper.read_results(&out_scalar_chunks, input_size * num_subtasks);

        helper.drop_all_buffers();

        Ok((point_x, point_y, scalar_chunks))
    }
}

/// Stage 2: Transpose
struct Transpose<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> Transpose<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        scalar_chunks: &[u32],
        num_subtasks: usize,
        input_size: usize,
        num_columns: u32,
        t_num_x_workgroups: usize,
        t_num_y_workgroups: usize,
        t_num_z_workgroups: usize,
        t_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::Transpose)
            .ok_or("Transpose shader not found")?;

        let in_chunks_buf = helper.create_buffer(&scalar_chunks.to_vec());
        let out_csc_col_ptr =
            helper.create_empty_buffer(num_subtasks * ((num_columns + 1) as usize) * 4);
        let out_csc_val_idxs = helper.create_empty_buffer(scalar_chunks.len());
        let out_curr = helper.create_empty_buffer(num_subtasks * (num_columns as usize) * 4);

        let params = vec![num_columns, input_size as u32];
        let params_buf = helper.create_buffer(&params);

        let thread_group_count = helper.create_thread_group_size(
            t_num_x_workgroups as u64,
            t_num_y_workgroups as u64,
            t_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(t_workgroup_size as u64, 1, 1);

        helper.execute_shader_with_pipeline(
            &shader.pipeline_state,
            &[
                &in_chunks_buf,
                &out_csc_col_ptr,
                &out_csc_val_idxs,
                &out_curr,
                &params_buf,
            ],
            &thread_group_count,
            &threads_per_threadgroup,
        );

        let csc_col_ptr = helper.read_results(
            &out_csc_col_ptr,
            num_subtasks * ((num_columns + 1) as usize) * 4,
        );
        let csc_val_idxs = helper.read_results(&out_csc_val_idxs, scalar_chunks.len());

        helper.drop_all_buffers();

        Ok((csc_col_ptr, csc_val_idxs))
    }
}

/// Stage 3: SMVP
struct SMVP<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> SMVP<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        csc_col_ptr: &[u32],
        csc_val_idxs: &[u32],
        point_x: &[u32],
        point_y: &[u32],
        input_size: usize,
        num_subtasks: usize,
        num_columns: u32,
        s_num_x_workgroups: usize,
        s_num_y_workgroups: usize,
        s_num_z_workgroups: usize,
        s_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let shader = self
            .shader_manager
            .get_shader(&ShaderType::SMVP)
            .ok_or("SMVP shader not found")?;

        let bucket_size =
            (num_columns / 2) as usize * self.shader_manager.config().num_limbs * 4 * num_subtasks;

        // Create buffers
        let row_ptr_buf = helper.create_buffer(&csc_col_ptr.to_vec());
        let val_idx_buf = helper.create_buffer(&csc_val_idxs.to_vec());
        let point_x_buf = helper.create_buffer(&point_x.to_vec());
        let point_y_buf = helper.create_buffer(&point_y.to_vec());

        let bucket_x_buf = helper.create_empty_buffer(bucket_size);
        let bucket_y_buf = helper.create_empty_buffer(bucket_size);
        let bucket_z_buf = helper.create_empty_buffer(bucket_size);

        // Execute in chunks
        let num_subtask_chunk_size = 4u32;
        for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
            let params = vec![
                input_size as u32,
                s_num_y_workgroups as u32,
                num_subtasks as u32,
                offset,
            ];
            let params_buf = helper.create_buffer(&params);

            let adjusted_x_workgroups =
                s_num_x_workgroups / (num_subtasks / num_subtask_chunk_size as usize);

            let thread_group_count = helper.create_thread_group_size(
                adjusted_x_workgroups as u64,
                s_num_y_workgroups as u64,
                s_num_z_workgroups as u64,
            );
            let threads_per_threadgroup =
                helper.create_thread_group_size(s_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &shader.pipeline_state,
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
                &thread_group_count,
                &threads_per_threadgroup,
            );
        }

        let bucket_x = helper.read_results(&bucket_x_buf, bucket_size);
        let bucket_y = helper.read_results(&bucket_y_buf, bucket_size);
        let bucket_z = helper.read_results(&bucket_z_buf, bucket_size);

        helper.drop_all_buffers();

        Ok((bucket_x, bucket_y, bucket_z))
    }
}

/// Stage 4: PBPR
struct PBPR<'a> {
    shader_manager: &'a ShaderManager,
}

impl<'a> PBPR<'a> {
    fn new(shader_manager: &'a ShaderManager) -> Self {
        Self { shader_manager }
    }

    fn execute(
        &self,
        bucket_x: &[u32],
        bucket_y: &[u32],
        bucket_z: &[u32],
        num_subtasks: usize,
        num_columns: u32,
        num_subtasks_per_bpr_1: usize,
        num_subtasks_per_bpr_2: usize,
        b_num_x_workgroups: usize,
        b_num_y_workgroups: usize,
        b_num_z_workgroups: usize,
        b_2_num_x_workgroups: usize,
        b_2_num_y_workgroups: usize,
        b_2_num_z_workgroups: usize,
        b_workgroup_size: usize,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), Box<dyn Error>> {
        let mut helper = MetalHelper::with_device(self.shader_manager.device().clone());
        let stage1_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage1)
            .ok_or("BPRStage1 shader not found")?;
        let stage2_shader = self
            .shader_manager
            .get_shader(&ShaderType::BPRStage2)
            .ok_or("BPRStage2 shader not found")?;

        let bucket_sum_x_buf = helper.create_buffer(&bucket_x.to_vec());
        let bucket_sum_y_buf = helper.create_buffer(&bucket_y.to_vec());
        let bucket_sum_z_buf = helper.create_buffer(&bucket_z.to_vec());

        let g_points_size =
            num_subtasks * b_workgroup_size * self.shader_manager.config().num_limbs * 4;
        let g_points_x_buf = helper.create_empty_buffer(g_points_size);
        let g_points_y_buf = helper.create_empty_buffer(g_points_size);
        let g_points_z_buf = helper.create_empty_buffer(g_points_size);

        // Stage 1
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1) {
            let params = vec![
                subtask_chunk_idx as u32,
                num_columns,
                num_subtasks_per_bpr_1 as u32,
            ];
            let params_buf = helper.create_buffer(&params);
            let workgroup_size_buf = helper.create_buffer(&vec![b_workgroup_size as u32]);

            let stage1_thread_group_count = helper.create_thread_group_size(
                b_num_x_workgroups as u64,
                b_num_y_workgroups as u64,
                b_num_z_workgroups as u64,
            );
            let stage1_threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &stage1_shader.pipeline_state,
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
                &stage1_thread_group_count,
                &stage1_threads_per_threadgroup,
            );
        }

        // Stage 2
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_2) {
            let params = vec![
                subtask_chunk_idx as u32,
                num_columns,
                num_subtasks_per_bpr_2 as u32,
            ];
            let params_buf = helper.create_buffer(&params);
            let workgroup_size_buf = helper.create_buffer(&vec![b_workgroup_size as u32]);

            let stage2_thread_group_count = helper.create_thread_group_size(
                b_2_num_x_workgroups as u64,
                b_2_num_y_workgroups as u64,
                b_2_num_z_workgroups as u64,
            );
            let stage2_threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader_with_pipeline(
                &stage2_shader.pipeline_state,
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
                &stage2_thread_group_count,
                &stage2_threads_per_threadgroup,
            );
        }

        let g_points_x = helper.read_results(&g_points_x_buf, g_points_size);
        let g_points_y = helper.read_results(&g_points_y_buf, g_points_size);
        let g_points_z = helper.read_results(&g_points_z_buf, g_points_size);

        helper.drop_all_buffers();

        Ok((g_points_x, g_points_y, g_points_z))
    }
}

/// Convenient wrapper that mimics the Arkworks VariableBaseMSM interface
/// Usage: metal_variable_base_msm(&bases, &scalars)
pub fn metal_variable_base_msm(
    bases: &[ark_bn254::G1Affine],
    scalars: &[ark_bn254::Fr],
) -> Result<ark_bn254::G1Projective, Box<dyn Error>> {
    // Handle empty input case
    if bases.is_empty() || scalars.is_empty() {
        return Err("Empty input".into());
    }

    // Ensure bases and scalars have the same length
    if bases.len() != scalars.len() {
        return Err("Bases and scalars must have the same length".into());
    }

    let pipeline = MetalMSMPipeline::with_default_config()?;
    pipeline.execute(bases, scalars)
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
        let log_input_size = 20;
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
