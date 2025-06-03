use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand, UniformRand, Zero};
use rand::Rng;
use std::ops::Neg;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;
use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;
use crate::msm::metal_msm::utils::mont_reduction::raw_reduction;
use num_bigint::BigUint;

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
        // params => [input_size, num_y_workgroups, num_z_workgroups, offset]
        let params = vec![
            input_size as u32,
            s_num_y_workgroups,
            s_num_z_workgroups,
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

// ----------------------------------------------------------------------------------
// Unit-test for the SMVP kernel
// ----------------------------------------------------------------------------------
#[test]
#[serial_test::serial]
fn test_smvp() {
    // Ensure we start with the correct constants configuration
    use crate::msm::metal_msm::utils::metal_wrapper::ensure_constants_for_config;
    ensure_constants_for_config(16, 16);

    let log_limb_size: u32 = 16;
    let num_limbs: usize = 16;

    // Constants that must match the shader build-time constants
    let num_columns: u32 = 1u32 << 16; // CHUNK_SIZE = 16 -> 65536 columns
    let half_columns: u32 = num_columns / 2;

    // Use a small input to keep the test lightweight
    let input_size: usize = 8;
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
        // Random affine point then promote to projective (z = 1)
        let affine = G::rand(&mut rng).into_affine();
        let proj: G = affine.into();
        points.push(proj);

        // Convert x/y coordinates to Montgomery form, then to limbs
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
    //    Each row corresponds to a bucket; randomly assign each point to a row.
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

    let (gpu_bucket_x, gpu_bucket_y, gpu_bucket_z) = smvp_gpu(
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

    // ---------------------------------------------------------------------------
    // 4. Convert GPU buckets back to projective points
    // ---------------------------------------------------------------------------
    let mut gpu_buckets: Vec<G> = Vec::with_capacity(half_columns as usize);
    for id in 0..half_columns as usize {
        let limb_start = id * num_limbs;
        let limb_end = (id + 1) * num_limbs;

        let xr_limbs = &gpu_bucket_x[limb_start..limb_end];
        let yr_limbs = &gpu_bucket_y[limb_start..limb_end];
        let zr_limbs = &gpu_bucket_z[limb_start..limb_end];

        let xr_mont = BigInt::<4>::from_limbs(xr_limbs, log_limb_size);
        let yr_mont = BigInt::<4>::from_limbs(yr_limbs, log_limb_size);
        let zr_mont = BigInt::<4>::from_limbs(zr_limbs, log_limb_size);

        // Convert from Montgomery form to standard form
        let xr_std = raw_reduction(xr_mont);
        let yr_std = raw_reduction(yr_mont);
        let zr_std = raw_reduction(zr_mont);

        let x = BaseField::from_bigint(xr_std).unwrap_or_else(|| BaseField::zero());
        let y = BaseField::from_bigint(yr_std).unwrap_or_else(|| BaseField::zero());
        let z = BaseField::from_bigint(zr_std).unwrap_or_else(|| BaseField::zero());

        gpu_buckets.push(G::new(x, y, z));
    }

    // ---------------------------------------------------------------------------
    // 5. CPU reference implementation of the kernel logic
    // ---------------------------------------------------------------------------
    let mut cpu_buckets: Vec<G> = vec![G::zero(); half_columns as usize];

    // helper closure: sum points in a given row
    let accumulate_row = |row_idx: usize, row_to_indices: &Vec<Vec<u32>>, points: &Vec<G>| -> G {
        let mut acc = G::zero();
        for &idx in &row_to_indices[row_idx] {
            acc += points[idx as usize];
        }
        acc
    };

    for id in 0..half_columns as usize {
        let id_mod = id % half_columns as usize;

        // j = 0 (positive bucket mostly)
        let mut row_idx = id_mod + half_columns as usize;
        if id_mod == 0 {
            row_idx = 0;
        }
        let mut sum = accumulate_row(row_idx, &row_to_indices, &points);
        let bucket_idx = if half_columns as usize > row_idx {
            // negative bucket -> negate the sum
            sum = sum.neg();
            half_columns as usize - row_idx
        } else {
            row_idx - half_columns as usize
        };
        if bucket_idx > 0 {
            cpu_buckets[id] = sum;
        }

        // j = 1 (negative bucket counterpart)
        let mut row_idx2 = half_columns as usize - id_mod;
        let mut sum2 = accumulate_row(row_idx2, &row_to_indices, &points);
        let bucket_idx2 = if half_columns as usize > row_idx2 {
            sum2 = sum2.neg();
            half_columns as usize - row_idx2
        } else {
            row_idx2 - half_columns as usize
        };
        if bucket_idx2 > 0 {
            cpu_buckets[id] += sum2;
        }
    }

    // ---------------------------------------------------------------------------
    // 6. Compare GPU vs CPU results on the non-empty buckets
    // ---------------------------------------------------------------------------
    for id in 0..half_columns as usize {
        if cpu_buckets[id].is_zero() {
            continue; // skip empty buckets
        }
        assert_eq!(
            cpu_buckets[id], gpu_buckets[id],
            "Mismatch at bucket {}",
            id
        );
    }
}
