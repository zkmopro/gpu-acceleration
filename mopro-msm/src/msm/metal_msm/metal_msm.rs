use crate::msm::metal_msm::utils::limbs_conversion::{
    pack_affine_and_scalars, GenericLimbConversion,
};
use crate::msm::metal_msm::utils::metal_wrapper::{MetalConfig, MetalHelper};
use crate::msm::metal_msm::utils::mont_reduction::raw_reduction;
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as Affine, G1Projective as G};
use ark_ff::{BigInt, PrimeField};
use ark_std::{vec::Vec, Zero};
use rayon::prelude::*;
use std::error::Error;

// timing
use std::time::Instant;

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

/// Main Metal MSM pipeline
struct MetalMSMPipeline {
    config: MetalMSMConfig,
}

impl MetalMSMPipeline {
    fn new(config: MetalMSMConfig) -> Self {
        Self { config }
    }

    fn with_default_config() -> Self {
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
        let start = Instant::now();
        let (coords, scals) = pack_affine_and_scalars(bases, scalars, &metal_config);
        println!("0️⃣ Pack points & scalars: {:?}", start.elapsed());

        // Stage 1: Convert Point & Scalar Decomposition
        // 1. Unpack point coordinates and encode them into Montgomery form
        // 2. Decompose scalars into Signed wNAF form
        let start = Instant::now();
        let stage1 = ConvertPointAndScalarDecompose::new(&self.config);
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

        println!(
            "\n=============================== \nc_num_x_workgroups: {}",
            c_num_x_workgroups
        );
        println!("c_num_y_workgroups: {}", c_num_y_workgroups);
        println!("c_num_z_workgroups: {}", c_num_z_workgroups);
        println!("c_workgroup_size: {}", c_workgroup_size);
        println!("===============================\n");

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
        println!("1️⃣ Convert & Decompose: {:?}", start.elapsed());

        // Stage 2: Transpose
        let start = Instant::now();
        let t_num_x_workgroups = 1;
        let t_num_y_workgroups = 1;
        let t_num_z_workgroups = 1;
        let t_workgroup_size = num_subtasks;

        let stage2 = Transpose::new(&self.config);
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
        println!("2️⃣ Transpose: {:?}", start.elapsed());

        // Stage 3: Sparse Matrix-Vector Multiplication
        let start = Instant::now();
        let s_workgroup_size = 256;
        let s_num_x_workgroups = 64;
        let s_num_y_workgroups = 2;
        let s_num_z_workgroups = num_subtasks;

        println!(
            "\n=============================== \ns_workgroup_size: {}",
            s_workgroup_size
        );
        println!("s_num_x_workgroups: {}", s_num_x_workgroups);
        println!("s_num_y_workgroups: {}", s_num_y_workgroups);
        println!("s_num_z_workgroups: {}", s_num_z_workgroups);
        println!(
            "best_fit/threads ratio: {}",
            (input_size as f64 / 2f64 * num_subtasks as f64)
                / (s_num_x_workgroups * s_num_y_workgroups * s_num_z_workgroups * s_workgroup_size)
                    as f64
        );
        println!(
            "total threads: {}, input_size: {}, num_columns: {}\n==============================\n",
            s_num_x_workgroups * s_num_y_workgroups * s_num_z_workgroups * s_workgroup_size,
            input_size,
            num_columns
        );

        let stage3 = SMVP::new(&self.config);
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
        println!("3️⃣ SMVP: {:?}", start.elapsed());

        // Stage 4: Parallel Bucket Reduction
        let start = Instant::now();
        // dynamic variable that determines the number of CSR matrices processed per invocation of the BPR shader. A safe default is 1.
        let num_subtasks_per_bpr_1 = 16;
        let num_subtasks_per_bpr_2 = 16;

        let b_num_x_workgroups = num_subtasks_per_bpr_1;
        let b_num_y_workgroups = 1;
        let b_num_z_workgroups = 1;
        let b_workgroup_size = 256;

        let b_2_num_x_workgroups = num_subtasks_per_bpr_2;
        let b_2_num_y_workgroups = 1;
        let b_2_num_z_workgroups = 1;

        let stage4 = PBPR::new(&self.config);
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
        println!("4️⃣ PBPR: {:?}", start.elapsed());

        // Stage 5: Final reduction and Horner's method
        let start = Instant::now();
        let result = self.final_reduction(
            &g_points_x,
            &g_points_y,
            &g_points_z,
            num_subtasks,
            self.config.log_limb_size as usize,
            b_workgroup_size,
        )?;
        println!("5️⃣ Final Reduction: {:?}", start.elapsed());
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
struct ConvertPointAndScalarDecompose {
    config: MetalConfig,
}

impl ConvertPointAndScalarDecompose {
    fn new(msm_config: &MetalMSMConfig) -> Self {
        Self {
            config: MetalConfig {
                log_limb_size: msm_config.log_limb_size,
                num_limbs: msm_config.num_limbs,
                shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
                kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
            },
        }
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
        let mut helper = MetalHelper::new();

        let coords_buf = helper.create_input_buffer(&coords.to_vec());
        let scalars_buf = helper.create_input_buffer(&scalars.to_vec());

        let out_point_x = helper.create_output_buffer(input_size * self.config.num_limbs);
        let out_point_y = helper.create_output_buffer(input_size * self.config.num_limbs);
        let out_scalar_chunks = helper.create_output_buffer(input_size * num_subtasks);

        let input_size_buf = helper.create_input_buffer(&vec![input_size as u32]);
        let num_y_workgroups_buf = helper.create_input_buffer(&vec![c_num_y_workgroups as u32]);

        let thread_group_count = helper.create_thread_group_size(
            c_num_x_workgroups as u64,
            c_num_y_workgroups as u64,
            c_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(c_workgroup_size as u64, 1, 1);

        helper.execute_shader(
            &self.config,
            &[&coords_buf, &scalars_buf, &input_size_buf],
            &[
                &out_point_x,
                &out_point_y,
                &out_scalar_chunks,
                &num_y_workgroups_buf,
            ],
            &thread_group_count,
            &threads_per_threadgroup,
        );

        let point_x = helper.read_results(&out_point_x, input_size * self.config.num_limbs);
        let point_y = helper.read_results(&out_point_y, input_size * self.config.num_limbs);
        let scalar_chunks = helper.read_results(&out_scalar_chunks, input_size * num_subtasks);

        helper.drop_all_buffers();

        Ok((point_x, point_y, scalar_chunks))
    }
}

/// Stage 2: Transpose
struct Transpose {
    config: MetalConfig,
}

impl Transpose {
    fn new(msm_config: &MetalMSMConfig) -> Self {
        Self {
            config: MetalConfig {
                log_limb_size: msm_config.log_limb_size,
                num_limbs: msm_config.num_limbs,
                shader_file: "cuzk/transpose.metal".to_string(),
                kernel_name: "transpose".to_string(),
            },
        }
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
        let mut helper = MetalHelper::new();

        let in_chunks_buf = helper.create_input_buffer(&scalar_chunks.to_vec());
        let out_csc_col_ptr =
            helper.create_output_buffer(num_subtasks * ((num_columns + 1) as usize) * 4);
        let out_csc_val_idxs = helper.create_output_buffer(scalar_chunks.len());
        let out_curr = helper.create_output_buffer(num_subtasks * (num_columns as usize) * 4);

        let params = vec![num_columns, input_size as u32];
        let params_buf = helper.create_input_buffer(&params);

        let thread_group_count = helper.create_thread_group_size(
            t_num_x_workgroups as u64,
            t_num_y_workgroups as u64,
            t_num_z_workgroups as u64,
        );
        let threads_per_threadgroup =
            helper.create_thread_group_size(t_workgroup_size as u64, 1, 1);

        helper.execute_shader(
            &self.config,
            &[
                &in_chunks_buf,
                &out_csc_col_ptr,
                &out_csc_val_idxs,
                &out_curr,
                &params_buf,
            ],
            &[],
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
struct SMVP {
    config: MetalConfig,
}

impl SMVP {
    fn new(msm_config: &MetalMSMConfig) -> Self {
        Self {
            config: MetalConfig {
                log_limb_size: msm_config.log_limb_size,
                num_limbs: msm_config.num_limbs,
                shader_file: "cuzk/smvp.metal".to_string(),
                kernel_name: "smvp".to_string(),
            },
        }
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
        let mut helper = MetalHelper::new();

        let bucket_size = (num_columns / 2) as usize * self.config.num_limbs * 4 * num_subtasks;

        // Create buffers
        let row_ptr_buf = helper.create_input_buffer(&csc_col_ptr.to_vec());
        let val_idx_buf = helper.create_input_buffer(&csc_val_idxs.to_vec());
        let point_x_buf = helper.create_input_buffer(&point_x.to_vec());
        let point_y_buf = helper.create_input_buffer(&point_y.to_vec());

        let bucket_x_buf = helper.create_output_buffer(bucket_size);
        let bucket_y_buf = helper.create_output_buffer(bucket_size);
        let bucket_z_buf = helper.create_output_buffer(bucket_size);

        // Execute in chunks
        let num_subtask_chunk_size = 4u32;
        for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
            let params = vec![
                input_size as u32,
                s_num_y_workgroups as u32,
                num_subtasks as u32,
                offset,
            ];
            let params_buf = helper.create_input_buffer(&params);

            let adjusted_x_workgroups =
                s_num_x_workgroups / (num_subtasks / num_subtask_chunk_size as usize);

            let thread_group_count = helper.create_thread_group_size(
                adjusted_x_workgroups as u64,
                s_num_y_workgroups as u64,
                s_num_z_workgroups as u64,
            );
            let threads_per_threadgroup =
                helper.create_thread_group_size(s_workgroup_size as u64, 1, 1);

            println!("--- offset: {} ---", offset);
            println!("adjusted_x_workgroups: {}", adjusted_x_workgroups);
            println!("s_num_y_workgroups: {}", s_num_y_workgroups);
            println!("s_num_z_workgroups: {}", s_num_z_workgroups);
            println!("s_workgroup_size: {}", s_workgroup_size);

            helper.execute_shader(
                &self.config,
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
struct PBPR {
    stage1_config: MetalConfig,
    stage2_config: MetalConfig,
}

impl PBPR {
    fn new(msm_config: &MetalMSMConfig) -> Self {
        Self {
            stage1_config: MetalConfig {
                log_limb_size: msm_config.log_limb_size,
                num_limbs: msm_config.num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_1".to_string(),
            },
            stage2_config: MetalConfig {
                log_limb_size: msm_config.log_limb_size,
                num_limbs: msm_config.num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_2".to_string(),
            },
        }
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
        let mut helper = MetalHelper::new();

        let bucket_sum_x_buf = helper.create_input_buffer(&bucket_x.to_vec());
        let bucket_sum_y_buf = helper.create_input_buffer(&bucket_y.to_vec());
        let bucket_sum_z_buf = helper.create_input_buffer(&bucket_z.to_vec());

        let g_points_size = num_subtasks * b_workgroup_size * self.stage1_config.num_limbs * 4;
        let g_points_x_buf = helper.create_output_buffer(g_points_size);
        let g_points_y_buf = helper.create_output_buffer(g_points_size);
        let g_points_z_buf = helper.create_output_buffer(g_points_size);

        // Stage 1
        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1) {
            let params = vec![
                subtask_chunk_idx as u32,
                num_columns,
                num_subtasks_per_bpr_1 as u32,
            ];
            let params_buf = helper.create_input_buffer(&params);
            let workgroup_size_buf = helper.create_input_buffer(&vec![b_workgroup_size as u32]);

            let stage1_thread_group_count = helper.create_thread_group_size(
                b_num_x_workgroups as u64,
                b_num_y_workgroups as u64,
                b_num_z_workgroups as u64,
            );
            let stage1_threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader(
                &self.stage1_config,
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
            let params_buf = helper.create_input_buffer(&params);
            let workgroup_size_buf = helper.create_input_buffer(&vec![b_workgroup_size as u32]);

            let stage2_thread_group_count = helper.create_thread_group_size(
                b_2_num_x_workgroups as u64,
                b_2_num_y_workgroups as u64,
                b_2_num_z_workgroups as u64,
            );
            let stage2_threads_per_threadgroup =
                helper.create_thread_group_size(b_workgroup_size as u64, 1, 1);

            helper.execute_shader(
                &self.stage2_config,
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

    let pipeline = MetalMSMPipeline::with_default_config();
    pipeline.execute(bases, scalars)
}

/// Test utilities module - available for both unit tests and integration tests
pub mod test_utils {
    use super::*;
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use rayon::prelude::*;

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

    #[test]
    fn benchmark_metal_vs_arkworks_msm() {
        println!("\n=== MSM Benchmark: Metal vs Arkworks ===");
        println!(
            "{:<12} {:<15} {:<15} {:<15} {:<10}",
            "Input Size", "Metal Time", "Arkworks Time", "Speedup", "Correct"
        );
        println!("{}", "-".repeat(75));

        for log_input_size in 12..=24 {
            let input_size = 1 << log_input_size;

            // Generate test data
            let generation_start = Instant::now();
            let (bases, scalars) = test_utils::generate_random_bases_and_scalars(input_size);
            let generation_time = generation_start.elapsed();

            // Benchmark Metal MSM
            let metal_start = Instant::now();
            let metal_result = metal_variable_base_msm(&bases, &scalars).unwrap();
            let metal_time = metal_start.elapsed();

            // Benchmark Arkworks MSM
            let arkworks_start = Instant::now();
            let arkworks_result = G::msm(&bases, &scalars).unwrap();
            let arkworks_time = arkworks_start.elapsed();

            // Calculate speedup
            let speedup = arkworks_time.as_secs_f64() / metal_time.as_secs_f64();

            // Verify correctness
            let is_correct = metal_result == arkworks_result;

            // Format output
            let input_size_str = format!("2^{} ({})", log_input_size, input_size);
            let metal_time_str = format!("{:.3}s", metal_time.as_secs_f64());
            let arkworks_time_str = format!("{:.3}s", arkworks_time.as_secs_f64());
            let speedup_str = if speedup > 1.0 {
                format!("{:.2}x faster", speedup)
            } else {
                format!("{:.2}x slower", 1.0 / speedup)
            };
            let correct_str = if is_correct { "✓" } else { "✗" };

            println!(
                "{:<12} {:<15} {:<15} {:<15} {:<10}",
                input_size_str, metal_time_str, arkworks_time_str, speedup_str, correct_str
            );

            // Print generation time for larger sizes
            println!(
                "  └─ Data generation: {:.3}s",
                generation_time.as_secs_f64()
            );

            // Assert correctness for all sizes
            assert_eq!(
                metal_result, arkworks_result,
                "Results don't match for input size 2^{}",
                log_input_size
            );
        }

        println!("{}", "-".repeat(75));
        println!("All benchmarks completed successfully!");
    }
}
