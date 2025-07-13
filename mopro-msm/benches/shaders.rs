use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ff::{BigInt, PrimeField};
use ark_std::{One, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_bigint::BigUint;
use rand::{thread_rng, Rng};

use ark_ec::CurveGroup;

use mopro_msm::msm::metal_msm::host::metal_wrapper::{MetalConfig, MetalHelper};
use mopro_msm::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use mopro_msm::msm::metal_msm::utils::mont_params::calc_mont_radix;

fn bench_shaders(c: &mut Criterion) {
    let mut group = c.benchmark_group("metal_msm_shaders");

    // Points and Scalar Decomposition (varying limb counts)
    {
        const LIMB_COUNTS: &[usize] = &[8, 16, 32];
        for &num_limbs in LIMB_COUNTS {
            let log_limb_size = 16;

            let config = MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
                kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
            };

            let mut helper = MetalHelper::new();

            // We only need one scalar for the kernel call, but we won't test it here
            // So let's just supply zeros for the scalar array
            let scalars = vec![0u32; num_limbs];

            // Generate a random point on BN254 for testing
            let mut rng = thread_rng();
            let point = G::rand(&mut rng).into_affine();
            let x: BigUint = point.x.into_bigint().try_into().unwrap();
            let y: BigUint = point.y.into_bigint().try_into().unwrap();

            // Convert unreduced x,y into `num_limbs` limbs of size `log_limb_size`
            let x_in_ark: BigInt<4> = x.clone().try_into().unwrap();
            let y_in_ark: BigInt<4> = y.clone().try_into().unwrap();
            let x_limb = x_in_ark.to_limbs(num_limbs, log_limb_size);
            let y_limb = y_in_ark.to_limbs(num_limbs, log_limb_size);

            let x_packed = pack_limbs(&x_limb);
            let y_packed = pack_limbs(&y_limb);

            // The coords buffer: x + y
            let coords = [x_packed, y_packed].concat();

            // Setup Metal buffers
            let coords_buf = helper.create_buffer(&coords);
            let scalars_buf = helper.create_buffer(&scalars);
            let params_buf =
                helper.create_buffer(&vec![1u32, log_limb_size, num_limbs as u32, 1u32]);

            // Prepare output buffers for the kernel
            let point_x_buf = helper.create_empty_buffer(num_limbs);
            let point_y_buf = helper.create_empty_buffer(num_limbs);
            let chunks_buf = helper.create_empty_buffer(num_limbs);

            // Setup thread group sizes
            let thread_group_count = helper.create_thread_group_size(1, 1, 1);
            let thread_group_size = helper.create_thread_group_size(1, 1, 1);

            group.throughput(Throughput::Elements(num_limbs as u64));
            group.bench_with_input(
                BenchmarkId::new("Points and Scalar Decomposition", num_limbs),
                &num_limbs,
                |b, &_num_limbs| {
                    b.iter(|| {
                        helper.execute_shader(
                            &config,
                            &[
                                &coords_buf,
                                &scalars_buf,
                                &point_x_buf,
                                &point_y_buf,
                                &chunks_buf,
                                &params_buf,
                            ],
                            &thread_group_count,
                            &thread_group_size,
                        );

                        // Read back X,Y results
                        let _x_result = helper.read_results(&point_x_buf, num_limbs);
                        let _y_result = helper.read_results(&point_y_buf, num_limbs);
                    })
                },
            );
        }
    }

    // Transpose shader (varying tasks, columns, and input size)
    {
        let config = MetalConfig {
            log_limb_size: 16,
            num_limbs: 16,
            shader_file: "cuzk/transpose.metal".to_string(),
            kernel_name: "transpose".to_string(),
        };

        let mut rng = rand::thread_rng();
        const CASES: &[(usize, u32, u32)] = &[(1, 8, 10), (2, 8, 50), (4, 8, 100)];

        for &(num_subtasks, n, input_size) in CASES {
            // Generate test data for this configuration
            let mut all_csr_col_idx = Vec::new();
            for _ in 0..num_subtasks {
                let cols = (0..input_size)
                    .map(|_| rng.gen_range(0..n))
                    .collect::<Vec<u32>>();
                all_csr_col_idx.extend_from_slice(&cols);
            }

            // Setup Metal buffers
            let mut helper = MetalHelper::new();
            let all_csr_col_idx_buf = helper.create_buffer(&all_csr_col_idx);
            let all_csc_col_ptr_buf = helper.create_empty_buffer(num_subtasks * (n as usize + 1));
            let all_csc_val_idxs_buf =
                helper.create_empty_buffer(num_subtasks * input_size as usize);
            let all_curr_buf = helper.create_empty_buffer(num_subtasks * n as usize);
            let params_buf = helper.create_buffer(&vec![n, input_size]);

            let thread_group_count = helper.create_thread_group_size(num_subtasks as u64, 1, 1);
            let thread_group_size = helper.create_thread_group_size(1, 1, 1);

            group.throughput(Throughput::Elements(
                (num_subtasks * input_size as usize) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new(
                    "transpose",
                    format!("s{}_c{}_i{}", num_subtasks, n, input_size),
                ),
                &(num_subtasks, n, input_size),
                |b, &_params| {
                    b.iter(|| {
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
                        let _ = helper
                            .read_results(&all_csc_col_ptr_buf, num_subtasks * (n as usize + 1));
                        let _ = helper.read_results(
                            &all_csc_val_idxs_buf,
                            num_subtasks * input_size as usize,
                        );
                        helper.drop_all_buffers();
                    })
                },
            );
        }
    }

    // SMVP shader (varying input sizes)
    {
        let log_limb_size: u32 = 16;
        let num_limbs: usize = 16;

        // Constants that must match the shader build-time constants
        let num_columns: u32 = 1u32 << 16; // CHUNK_SIZE = 16 -> 65536 columns

        const CASES: &[usize] = &[4, 8, 16];
        for &input_size in CASES {
            let num_subtasks: usize = 1;

            // ---------------------------------------------------------------------------
            // 1. Generate random points and convert coordinates to Montgomery limbs
            // ---------------------------------------------------------------------------
            let mut rng = rand::thread_rng();

            let mut points: Vec<G> = Vec::with_capacity(input_size);
            let mut point_x_limbs: Vec<u32> = Vec::with_capacity(input_size * num_limbs);
            let mut point_y_limbs: Vec<u32> = Vec::with_capacity(input_size * num_limbs);

            // Calculate Montgomery radix for conversion
            let r = calc_mont_radix(num_limbs, log_limb_size);
            let p: BigUint = BaseField::MODULUS.try_into().unwrap();

            for _ in 0..input_size {
                let affine = G::rand(&mut rng).into_affine();
                let proj: G = affine.into();
                points.push(proj);

                let x_std: BigUint = affine.x.into_bigint().try_into().unwrap();
                let y_std: BigUint = affine.y.into_bigint().try_into().unwrap();

                let x_mont = (&x_std * &r) % &p;
                let y_mont = (&y_std * &r) % &p;

                let x_mont_bigint: BigInt<4> = x_mont.try_into().unwrap();
                let y_mont_bigint: BigInt<4> = y_mont.try_into().unwrap();

                let x_limbs = x_mont_bigint.to_limbs(num_limbs, log_limb_size);
                let y_limbs = y_mont_bigint.to_limbs(num_limbs, log_limb_size);

                point_x_limbs.extend(x_limbs);
                point_y_limbs.extend(y_limbs);
            }

            // ---------------------------------------------------------------------------
            // 2. Create a sparse matrix in CSR form (row_ptr / val_idx)
            // ---------------------------------------------------------------------------
            let mut row_to_indices: Vec<Vec<u32>> = vec![Vec::new(); num_columns as usize];
            for (idx, _) in points.iter().enumerate() {
                let row_idx = rng.gen_range(1..num_columns) as usize; // avoid row 0 for variety
                row_to_indices[row_idx].push(idx as u32);
            }

            // Build row_ptr (size = num_columns + 1) and val_idx in row-major order
            let mut row_ptr: Vec<u32> = vec![0u32; (num_columns + 1) as usize];
            let mut val_idx: Vec<u32> = Vec::with_capacity(input_size);
            let mut cumulative: u32 = 0;
            for i in 0..num_columns as usize {
                row_ptr[i] = cumulative;
                val_idx.extend(&row_to_indices[i]);
                cumulative += row_to_indices[i].len() as u32;
            }
            row_ptr[num_columns as usize] = cumulative;
            assert_eq!(cumulative as usize, val_idx.len());

            // ---------------------------------------------------------------------------
            // 3. Execute the SMVP kernel on GPU
            // ---------------------------------------------------------------------------
            let mut helper = MetalHelper::new();
            let smvp_config = MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/smvp.metal".to_string(),
                kernel_name: "smvp".to_string(),
            };

            group.throughput(Throughput::Elements(input_size as u64));
            group.bench_with_input(
                BenchmarkId::new("smvp", input_size),
                &input_size,
                |b, &_input_size| {
                    b.iter(|| {
                        let (_gpu_bucket_x, _gpu_bucket_y, _gpu_bucket_z) = smvp_gpu(
                            &mut helper,
                            &smvp_config,
                            &row_ptr,
                            &val_idx,
                            &point_x_limbs,
                            &point_y_limbs,
                            input_size,
                            num_subtasks,
                            num_columns,
                        );
                        helper.drop_all_buffers();
                    })
                },
            );
        }
    }

    // PBPR shader (varying number of columns)
    {
        let log_limb_size: u32 = 16;
        let num_limbs: usize = 16;

        const NUM_COLUMNS: &[u32] = &[8, 16, 32];
        for &num_columns in NUM_COLUMNS {
            let num_buckets_per_subtask = (num_columns / 2) as usize;
            let num_subtasks = 2;
            let num_subtasks_per_bpr = 2;
            let workgroup_size: u32 = 2;

            // Generate random bucket sums (Jacobian points)
            let mut rng = rand::thread_rng();
            let mut bucket_points = Vec::with_capacity(num_subtasks * num_buckets_per_subtask);
            for _ in 0..(num_subtasks * num_buckets_per_subtask) {
                let rand_pt = G::rand(&mut rng).into_affine();
                let proj = G::new(rand_pt.x, rand_pt.y, BaseField::one());
                bucket_points.push(proj);
            }

            // Convert bucket points to limb representation
            let mut bucket_sum_x_limbs =
                Vec::with_capacity(num_subtasks * num_buckets_per_subtask * num_limbs);
            let mut bucket_sum_y_limbs =
                Vec::with_capacity(num_subtasks * num_buckets_per_subtask * num_limbs);
            let mut bucket_sum_z_limbs =
                Vec::with_capacity(num_subtasks * num_buckets_per_subtask * num_limbs);

            for pt in &bucket_points {
                let x_limbs = pt.x.0.to_limbs(num_limbs, log_limb_size);
                let y_limbs = pt.y.0.to_limbs(num_limbs, log_limb_size);
                let z_limbs = pt.z.0.to_limbs(num_limbs, log_limb_size);
                bucket_sum_x_limbs.extend_from_slice(&x_limbs);
                bucket_sum_y_limbs.extend_from_slice(&y_limbs);
                bucket_sum_z_limbs.extend_from_slice(&z_limbs);
            }

            // g_points buffers (filled with zeros, will be overwritten by GPU)
            let g_points_size = num_subtasks * workgroup_size as usize * num_limbs;
            let g_points_x_limbs = vec![0u32; g_points_size];
            let g_points_y_limbs = vec![0u32; g_points_size];
            let g_points_z_limbs = vec![0u32; g_points_size];

            //----------------------------------------------
            // Create Metal buffers & run stage 1 and 2 in subtask chunks
            //----------------------------------------------
            let mut helper = MetalHelper::new();

            let bucket_sum_x_buf = helper.create_buffer(&bucket_sum_x_limbs);
            let bucket_sum_y_buf = helper.create_buffer(&bucket_sum_y_limbs);
            let bucket_sum_z_buf = helper.create_buffer(&bucket_sum_z_limbs);

            let g_points_x_buf = helper.create_buffer(&g_points_x_limbs);
            let g_points_y_buf = helper.create_buffer(&g_points_y_limbs);
            let g_points_z_buf = helper.create_buffer(&g_points_z_limbs);

            let thread_group_count =
                helper.create_thread_group_size(num_subtasks_per_bpr as u64, 1, 1);
            let thread_group_size = helper.create_thread_group_size(workgroup_size as u64, 1, 1);

            // ----------------------------------------------
            // Stage 1 and Stage 2 kernel launches
            // ----------------------------------------------
            let config_stage1 = MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_1".to_string(),
            };
            let config_stage2 = MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_2".to_string(),
            };

            group.throughput(Throughput::Elements(
                (num_subtasks * num_buckets_per_subtask * num_limbs) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new("pbpr_stage1_and_stage2", num_columns),
                &num_columns,
                |b, &_num_columns| {
                    b.iter(|| {
                        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr) {
                            let params = vec![
                                subtask_chunk_idx as u32,
                                num_columns,
                                num_subtasks_per_bpr as u32,
                            ];
                            let params_buf = helper.create_buffer(&params);
                            helper.execute_shader(
                                &config_stage1,
                                &[
                                    &bucket_sum_x_buf,
                                    &bucket_sum_y_buf,
                                    &bucket_sum_z_buf,
                                    &g_points_x_buf,
                                    &g_points_y_buf,
                                    &g_points_z_buf,
                                    &params_buf,
                                ],
                                &thread_group_count,
                                &thread_group_size,
                            );
                        }
                        for subtask_chunk_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr) {
                            let params = vec![
                                subtask_chunk_idx as u32,
                                num_columns,
                                num_subtasks_per_bpr as u32,
                            ];
                            let params_buf = helper.create_buffer(&params);
                            helper.execute_shader(
                                &config_stage2,
                                &[
                                    &bucket_sum_x_buf,
                                    &bucket_sum_y_buf,
                                    &bucket_sum_z_buf,
                                    &g_points_x_buf,
                                    &g_points_y_buf,
                                    &g_points_z_buf,
                                    &params_buf,
                                ],
                                &thread_group_count,
                                &thread_group_size,
                            );
                        }

                        // ----------------------------------------------
                        // Read GPU results
                        // ----------------------------------------------
                        let _gpu_gx_limbs = helper.read_results(&g_points_x_buf, g_points_size);
                        let _gpu_gy_limbs = helper.read_results(&g_points_y_buf, g_points_size);
                        let _gpu_gz_limbs = helper.read_results(&g_points_z_buf, g_points_size);

                        helper.drop_all_buffers();
                    })
                },
            );
        }
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().configure_from_args() // ← registers the CLI flags
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_shaders
}
criterion_main!(benches);

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

    // Work-group configuration heuristics (same as e2e.rs)
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

    // How many subtasks processed per shader invocation.
    let num_subtask_chunk_size = 4u32;

    let bucket_sum_coord_bytelength =
        (num_columns / 2) as usize * config.num_limbs as usize * 4 * num_subtasks as usize;

    let row_ptr_buf = helper.create_buffer(gpu_csc_col_ptr);
    let val_idx_buf = helper.create_buffer(gpu_csc_val_idxs);
    let point_x_buf = helper.create_buffer(gpu_point_x);
    let point_y_buf = helper.create_buffer(gpu_point_y);
    let bucket_x_buf = helper.create_empty_buffer(bucket_sum_coord_bytelength);
    let bucket_y_buf = helper.create_empty_buffer(bucket_sum_coord_bytelength);
    let bucket_z_buf = helper.create_empty_buffer(bucket_sum_coord_bytelength);

    // Launch shader for each subtask chunk
    for offset in (0..num_subtasks as u32).step_by(num_subtask_chunk_size as usize) {
        // params => [input_size, num_columns, num_subtasks, offset]
        let params = vec![
            input_size as u32,
            num_columns as u32,
            num_subtasks as u32,
            offset,
        ];
        let params_buf = helper.create_buffer(&params);

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

        let thread_group_count = helper.create_thread_group_size(
            adjusted_s_num_x_workgroups as u64,
            s_num_y_workgroups as u64,
            s_num_z_workgroups as u64,
        );
        let threads_per_group = helper.create_thread_group_size(s_workgroup_size as u64, 1, 1);

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
            &thread_group_count,
            &threads_per_group,
        );
    }

    // Read back results
    let out_x = helper.read_results(&bucket_x_buf, bucket_sum_coord_bytelength);
    let out_y = helper.read_results(&bucket_y_buf, bucket_sum_coord_bytelength);
    let out_z = helper.read_results(&bucket_z_buf, bucket_sum_coord_bytelength);

    (out_x, out_y, out_z)
}

fn pack_limbs(limbs: &[u32]) -> Vec<u32> {
    limbs
        .chunks(2)
        .map(|chunk| (chunk[1] << 16) | chunk[0])
        .collect()
}
