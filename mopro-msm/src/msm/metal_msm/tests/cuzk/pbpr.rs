use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::{CurveGroup, Group};
use ark_ff::{BigInt, PrimeField};
use ark_std::{One, UniformRand, Zero};
use num_bigint::BigUint;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

/// Simple double-and-add helper that mirrors the Metal implementation.
fn cpu_double_and_add(point: G, mut scalar: u32) -> G {
    let mut result = G::zero();
    let mut tmp = point;
    while scalar != 0 {
        if scalar & 1 == 1 {
            result += tmp;
        }
        tmp = tmp.double();
        scalar >>= 1;
    }
    result
}

#[test]
#[serial_test::serial]
fn test_pbpr_stage1_and_stage2() {
    // ----------------------------------------------
    // Parameters (kept small for fast unit testing)
    // ----------------------------------------------
    let log_limb_size = 16;
    let num_limbs = 16;

    let num_columns: u32 = 8; // Must be even
    let num_buckets_per_subtask = (num_columns / 2) as usize; // 4
    let workgroup_size: u32 = 2; // threads / subtask
    let buckets_per_thread = num_buckets_per_subtask as u32 / workgroup_size; // 2

    // Only a single subtask for the test.
    let subtask_idx: u32 = 0;
    let num_subtasks_per_bpr: u32 = 1;

    //------------------------------------------------
    // Generate random bucket sums  (Jacobian points)
    //------------------------------------------------
    let mut rng = rand::thread_rng();
    let mut bucket_points = Vec::with_capacity(num_buckets_per_subtask);
    for _ in 0..num_buckets_per_subtask {
        // Generate random affine point then convert to projective (Z = 1)
        let rand_pt = G::rand(&mut rng).into_affine();
        let proj = G::new(rand_pt.x, rand_pt.y, BaseField::one());
        bucket_points.push(proj);
    }

    // Convert bucket points to limb representation (Montgomery form already)
    let mut bucket_sum_x_limbs = Vec::with_capacity(num_buckets_per_subtask * num_limbs);
    let mut bucket_sum_y_limbs = Vec::with_capacity(num_buckets_per_subtask * num_limbs);
    let mut bucket_sum_z_limbs = Vec::with_capacity(num_buckets_per_subtask * num_limbs);

    for pt in &bucket_points {
        let x_limbs = pt.x.0.to_limbs(num_limbs, log_limb_size);
        let y_limbs = pt.y.0.to_limbs(num_limbs, log_limb_size);
        let z_limbs = pt.z.0.to_limbs(num_limbs, log_limb_size);
        bucket_sum_x_limbs.extend_from_slice(&x_limbs);
        bucket_sum_y_limbs.extend_from_slice(&y_limbs);
        bucket_sum_z_limbs.extend_from_slice(&z_limbs);
    }

    // g_points buffers (filled with zeros, will be overwritten by GPU)
    let g_points_size = workgroup_size as usize * num_limbs;
    let g_points_x_limbs = vec![0u32; g_points_size];
    let g_points_y_limbs = vec![0u32; g_points_size];
    let g_points_z_limbs = vec![0u32; g_points_size];

    //----------------------------------------------
    // Create Metal buffers & run stage 1 and 2
    //----------------------------------------------
    let mut helper = MetalHelper::new();

    let bucket_sum_x_buf = helper.create_input_buffer(&bucket_sum_x_limbs);
    let bucket_sum_y_buf = helper.create_input_buffer(&bucket_sum_y_limbs);
    let bucket_sum_z_buf = helper.create_input_buffer(&bucket_sum_z_limbs);

    let g_points_x_buf = helper.create_input_buffer(&g_points_x_limbs);
    let g_points_y_buf = helper.create_input_buffer(&g_points_y_limbs);
    let g_points_z_buf = helper.create_input_buffer(&g_points_z_limbs);

    // params = [subtask_idx, num_columns, num_subtasks_per_bpr]
    let params = vec![subtask_idx, num_columns, num_subtasks_per_bpr];
    let params_buf = helper.create_input_buffer(&params);

    // workgroup_size as a single u32 uniform
    let wg_size_vec = vec![workgroup_size];
    let wg_size_buf = helper.create_input_buffer(&wg_size_vec);

    // Thread configuration: 1-D grid with `workgroup_size` threads, each threadgroup has 1 thread.
    let thread_group_count = helper.create_thread_group_size(workgroup_size as u64, 1, 1);
    let thread_group_size = helper.create_thread_group_size(1, 1, 1);

    // ----------------------------------------------
    // Stage 1 kernel
    // ----------------------------------------------
    let config_stage1 = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "bpr_stage_1".to_string(),
    };

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
            &wg_size_buf,
        ],
        &[],
        &thread_group_count,
        &thread_group_size,
    );

    // ----------------------------------------------
    // Stage 2 kernel (reads results of stage 1)
    // ----------------------------------------------
    let config_stage2 = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/pbpr.metal".to_string(),
        kernel_name: "bpr_stage_2".to_string(),
    };

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
            &wg_size_buf,
        ],
        &[],
        &thread_group_count,
        &thread_group_size,
    );

    // ----------------------------------------------
    // Read GPU results
    // ----------------------------------------------
    let gpu_gx_limbs = helper.read_results(&g_points_x_buf, g_points_size);
    let gpu_gy_limbs = helper.read_results(&g_points_y_buf, g_points_size);
    let gpu_gz_limbs = helper.read_results(&g_points_z_buf, g_points_size);

    // Drop buffers (no longer needed on GPU side)
    helper.drop_all_buffers();

    // --------------------------------------------------
    // CPU reference implementation for stage1 + stage2
    // --------------------------------------------------
    // We simulate parallel semantics by treating bucket reads as coming
    // from the **original** bucket array, while writes are collected in
    // a separate vector.
    let mut bucket_after_stage1 = bucket_points.clone();
    let mut g_stage1 = vec![G::zero(); workgroup_size as usize];

    for thread_id in 0..workgroup_size {
        // Compute starting bucket index (`idx`) identical to Metal code.
        let idx_start = if thread_id % workgroup_size != 0 {
            (workgroup_size - (thread_id % workgroup_size)) * buckets_per_thread
        } else {
            0
        } as usize;

        // Offset is zero in this single-subtask test.
        let mut m = bucket_points[idx_start];
        let mut g = m;

        for i in 0..buckets_per_thread {
            let idx_local =
                (workgroup_size - (thread_id % workgroup_size)) * buckets_per_thread - 1 - i;
            let bi = idx_local as usize; // offset == 0
            let b = bucket_points[bi];
            m += b;
            g += m;
        }

        bucket_after_stage1[idx_start] = m;
        g_stage1[thread_id as usize] = g;
    }

    // ---------- Stage 2 (CPU) ----------
    let mut g_expected = Vec::with_capacity(workgroup_size as usize);
    for thread_id in 0..workgroup_size {
        let idx_start = if thread_id % workgroup_size != 0 {
            (workgroup_size - (thread_id % workgroup_size)) * buckets_per_thread
        } else {
            0
        } as usize;

        let m = bucket_after_stage1[idx_start];
        let mut g = g_stage1[thread_id as usize];
        let s = buckets_per_thread * (workgroup_size - (thread_id % workgroup_size) - 1);
        g += cpu_double_and_add(m, s);
        g_expected.push(g);
    }

    // ----------------------------------------------
    // Decode GPU limbs back to projective points
    // ----------------------------------------------
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p_biguint = &constants.p;
    let rinv = &constants.rinv;

    let decode_mont = |limbs: &[u32]| {
        let big: BigUint = BigInt::<4>::from_limbs(limbs, log_limb_size)
            .try_into()
            .unwrap();
        let val = (&big * rinv) % p_biguint;
        BaseField::from_bigint(val.try_into().unwrap()).unwrap()
    };

    for thread_id in 0..workgroup_size as usize {
        let start = thread_id * num_limbs;
        let end = (thread_id + 1) * num_limbs;
        let gx = decode_mont(&gpu_gx_limbs[start..end]);
        let gy = decode_mont(&gpu_gy_limbs[start..end]);
        let gz = decode_mont(&gpu_gz_limbs[start..end]);
        let gpu_point = G::new(gx, gy, gz);

        // Convert both GPU & CPU points to affine for comparison
        let gpu_affine = gpu_point.into_affine();
        let cpu_affine = g_expected[thread_id].into_affine();

        assert_eq!(gpu_affine, cpu_affine, "Mismatch at thread {}", thread_id);
    }
}
