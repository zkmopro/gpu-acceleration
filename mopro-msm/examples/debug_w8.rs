// Stage-by-stage debugging harness for the (n=1024, w=8) mismatch.
// Runs the full GPU pipeline with the same dispatch geometry as
// execute_pipeline, reading back every intermediate and validating it
// against a CPU model. First failing stage = culprit.
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, BigInteger, PrimeField};
use ark_std::Zero;
use mopro_msm::msm::metal_msm::host::metal_wrapper::*;
use mopro_msm::msm::metal_msm::host::shader_manager::{ShaderManager, ShaderType};
use mopro_msm::msm::metal_msm::metal_msm::test_utils;
use mopro_msm::msm::metal_msm::utils::limbs_conversion::{
    pack_affine_and_scalars, GenericLimbConversion,
};
use mopro_msm::msm::metal_msm::utils::mont_reduction::raw_reduction;

const NUM_LIMBS: usize = 8;

fn cpu_decompose(scalars: &[ScalarField], window_size: usize, num_subtasks: usize) -> Vec<u32> {
    let l = 1u32 << window_size;
    let s = l / 2;
    let mut chunks = vec![0u32; scalars.len() * num_subtasks];
    for (idx, sc) in scalars.iter().enumerate() {
        let bits = sc.into_bigint().to_bits_le();
        let mut carry = 0u32;
        for i in 0..num_subtasks {
            let mut val = 0u32;
            for b in 0..window_size {
                let bit_idx = i * window_size + b;
                if bit_idx < bits.len() && bits[bit_idx] {
                    val |= 1 << b;
                }
            }
            let mut slice = (val + carry) as i64;
            if slice >= s as i64 {
                slice = -((l as i64) - slice);
                carry = 1;
            } else {
                carry = 0;
            }
            chunks[idx + i * scalars.len()] = (slice + s as i64) as u32;
        }
        assert_eq!(carry, 0);
    }
    chunks
}

fn limbs_to_point(x: &[u32], y: &[u32], z: &[u32]) -> G {
    let to_field = |l: &[u32]| {
        let std = raw_reduction(BigInt::<4>::from_limbs(l, 32));
        BaseField::from_bigint(std).unwrap_or_else(BaseField::zero)
    };
    G::new_unchecked(to_field(x), to_field(y), to_field(z))
}

fn main() {
    let size = 1usize << 10;
    let window_size = 8usize;
    let num_columns = 1usize << window_size;
    let half_columns = num_columns / 2;
    let num_subtasks = (254f32 / window_size as f32).ceil() as usize; // 32

    let (bases, scalars) = test_utils::generate_random_bases_and_scalars(size);
    let config = MetalConfig::default();
    let (coords, scals) = pack_affine_and_scalars(&bases, &scalars, &config);

    let manager = ShaderManager::with_default_config().unwrap();
    let mut helper = MetalHelper::with_device(manager.device().clone());

    let coords_buf = helper.create_buffer(&coords);
    let scalars_buf = helper.create_buffer(&scals);
    let chunks_buf = helper.create_zeroed_buffer(size * num_subtasks);
    let csc_col_ptr_buf = helper.create_zeroed_buffer(num_subtasks * (num_columns + 1));
    let csc_val_idxs_buf = helper.create_zeroed_buffer(size * num_subtasks);
    let curr_buf = helper.create_zeroed_buffer(num_subtasks * num_columns);
    let bucket_size = half_columns * NUM_LIMBS * num_subtasks;
    let bucket_x_buf = helper.create_zeroed_buffer(bucket_size);
    let bucket_y_buf = helper.create_zeroed_buffer(bucket_size);
    let bucket_z_buf = helper.create_zeroed_buffer(bucket_size);

    let params_buf = helper.create_buffer(&vec![
        size as u32,
        window_size as u32,
        num_columns as u32,
        num_subtasks as u32,
    ]);
    let t_params_buf = helper.create_buffer(&vec![num_columns as u32, size as u32]);

    // ---- Stage 1: decompose (same geometry as pipeline, scale 1) ----
    let decompose = manager
        .get_shader(&ShaderType::ConvertPointAndDecompose)
        .unwrap();
    helper.execute_shader_with_pipeline(
        &decompose.pipeline_state,
        &[&scalars_buf, &chunks_buf, &params_buf],
        &helper.create_thread_group_size(32, 1, 1),
        &helper.create_thread_group_size(32, 1, 1),
    );
    let gpu_chunks = helper.read_results(&chunks_buf, size * num_subtasks);
    let cpu_chunks = cpu_decompose(&scalars, window_size, num_subtasks);
    let bad = gpu_chunks
        .iter()
        .zip(&cpu_chunks)
        .filter(|(a, b)| a != b)
        .count();
    println!("stage 1 decompose mismatches: {}", bad);

    // ---- Stage 2: transpose ----
    let hist = manager.get_shader(&ShaderType::TransposeHistogram).unwrap();
    let prefix = manager.get_shader(&ShaderType::TransposePrefixSum).unwrap();
    let scat = manager.get_shader(&ShaderType::TransposeScatter).unwrap();
    let t_tg = 256u64;
    let t_x = (size as u64 + t_tg - 1) / t_tg;
    helper.execute_shader_with_pipeline(
        &hist.pipeline_state,
        &[&chunks_buf, &csc_col_ptr_buf, &t_params_buf],
        &helper.create_thread_group_size(t_x, num_subtasks as u64, 1),
        &helper.create_thread_group_size(t_tg, 1, 1),
    );
    helper.execute_shader_with_pipeline(
        &prefix.pipeline_state,
        &[&csc_col_ptr_buf, &t_params_buf],
        &helper.create_thread_group_size(1, 1, 1),
        &helper.create_thread_group_size(num_subtasks as u64, 1, 1),
    );
    helper.execute_shader_with_pipeline(
        &scat.pipeline_state,
        &[&chunks_buf, &csc_col_ptr_buf, &csc_val_idxs_buf, &curr_buf, &t_params_buf],
        &helper.create_thread_group_size(t_x, num_subtasks as u64, 1),
        &helper.create_thread_group_size(t_tg, 1, 1),
    );
    let col_ptr = helper.read_results(&csc_col_ptr_buf, num_subtasks * (num_columns + 1));
    let val_idx = helper.read_results(&csc_val_idxs_buf, size * num_subtasks);

    // validate: per subtask+column, the set of point indices must equal the
    // set of points whose chunk == column
    let mut t_bad = 0;
    for st in 0..num_subtasks {
        for col in 0..num_columns {
            let start = col_ptr[st * (num_columns + 1) + col] as usize;
            let end = col_ptr[st * (num_columns + 1) + col + 1] as usize;
            let mut got: Vec<u32> = val_idx[st * size + start..st * size + end].to_vec();
            got.sort_unstable();
            let mut expect: Vec<u32> = (0..size as u32)
                .filter(|&j| cpu_chunks[j as usize + st * size] == col as u32)
                .collect();
            expect.sort_unstable();
            if got != expect {
                if t_bad < 5 {
                    println!(
                        "stage 2 mismatch st {} col {}: got {:?} expect {:?}",
                        st, col, got, expect
                    );
                }
                t_bad += 1;
            }
        }
    }
    println!("stage 2 transpose mismatched columns: {}", t_bad);

    // ---- Stage 3: SMVP (same chunked geometry as pipeline, scale 1) ----
    let smvp = manager.get_shader(&ShaderType::SMVP).unwrap();
    let s_wg = 32u64;
    let chunk = 4u32;
    for offset in (0..num_subtasks as u32).step_by(chunk as usize) {
        let remaining = (num_subtasks as u32 - offset).min(chunk);
        let valid = half_columns as u64 * remaining as u64;
        let max_y = ((valid as usize) / (s_wg as usize * remaining as usize)).max(1);
        let y = 32usize.min(max_y) as u64;
        let z = remaining as u64;
        let per_grid = s_wg * y * z;
        let x = (valid + per_grid - 1) / per_grid;
        let sp = helper.create_buffer(&vec![
            size as u32,
            num_columns as u32,
            num_subtasks as u32,
            offset,
        ]);
        helper.execute_shader_with_pipeline(
            &smvp.pipeline_state,
            &[&csc_col_ptr_buf, &csc_val_idxs_buf, &coords_buf, &bucket_x_buf, &bucket_y_buf, &bucket_z_buf, &sp],
            &helper.create_thread_group_size(x, y, z),
            &helper.create_thread_group_size(s_wg, 1, 1),
        );
    }
    let bx = helper.read_results(&bucket_x_buf, bucket_size);
    let by = helper.read_results(&bucket_y_buf, bucket_size);
    let bz = helper.read_results(&bucket_z_buf, bucket_size);

    // CPU model of SMVP semantics: thread id in subtask handles rows
    // (id%half)+half (j=0, forced to 0 when id%half==0) and half-(id%half)
    // (j=1); result goes to slot id with sign depending on row vs half.
    let mut s3_bad = 0;
    for st in 0..num_subtasks {
        for tid in 0..half_columns {
            let mut expect = G::zero();
            for j in 0..2 {
                let mut row = if j == 0 {
                    if tid % half_columns == 0 { 0 } else { (tid % half_columns) + half_columns }
                } else {
                    half_columns - (tid % half_columns)
                };
                if j == 0 && tid == 0 { row = 0; }
                let mut sum = G::zero();
                for jj in 0..size {
                    if cpu_chunks[jj + st * size] == row as u32 {
                        sum += bases[jj];
                    }
                }
                let bucket_idx = if half_columns > row { half_columns - row } else { row - half_columns };
                if half_columns > row {
                    sum = -sum;
                }
                if bucket_idx > 0 {
                    expect += sum;
                }
            }
            let slot = st * half_columns + tid;
            let got = limbs_to_point(
                &bx[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
                &by[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
                &bz[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
            );
            // compare projectively: both zero, or equal points
            let got_zero = got.z.is_zero();
            let ok = if expect.is_zero() {
                got_zero
            } else {
                !got_zero && got.into_affine() == expect.into_affine()
            };
            if !ok {
                if s3_bad < 8 {
                    println!("stage 3 mismatch: subtask {} slot {}", st, tid);
                }
                s3_bad += 1;
            }
        }
    }
    println!("stage 3 smvp mismatched buckets: {}", s3_bad);

    // ---- Stage 4: PBPR (same geometry as pipeline, scale 1) ----
    let bpr1 = manager.get_shader(&ShaderType::BPRStage1).unwrap();
    let bpr2 = manager.get_shader(&ShaderType::BPRStage2).unwrap();
    let b_wg = 32usize;
    let g_points_size = num_subtasks * b_wg * NUM_LIMBS;
    let g_x_buf = helper.create_zeroed_buffer(g_points_size);
    let g_y_buf = helper.create_zeroed_buffer(g_points_size);
    let g_z_buf = helper.create_zeroed_buffer(g_points_size);
    let bpr_params = helper.create_buffer(&vec![0u32, num_columns as u32, num_subtasks as u32, 0u32]);
    for shader in [bpr1, bpr2] {
        helper.execute_shader_with_pipeline(
            &shader.pipeline_state,
            &[&bucket_x_buf, &bucket_y_buf, &bucket_z_buf, &g_x_buf, &g_y_buf, &g_z_buf, &bpr_params],
            &helper.create_thread_group_size(num_subtasks as u64, 1, 1),
            &helper.create_thread_group_size(b_wg as u64, 1, 1),
        );
    }
    let gx = helper.read_results(&g_x_buf, g_points_size);
    let gy = helper.read_results(&g_y_buf, g_points_size);
    let gz = helper.read_results(&g_z_buf, g_points_size);

    // CPU expected per-subtask reduction from the (already validated) GPU buckets
    let mut s4_bad = 0;
    for st in 0..num_subtasks {
        let mut expect = G::zero();
        for d in 1..=half_columns {
            // bucket d of subtask st lives at slot st*half + (d % half)?? slot layout:
            // slot id holds bucket (id % half == 0 ? half : id % half)... derive from smvp:
            // bi = id, bucket_idx = (id % half == 0) ? half : ... j-loop merges into slot id.
            // Slot id's bucket_idx: for j=0 row=(id%half)+half -> bucket id%half (0 -> row 0, no store)
            //                       for j=1 row=half-(id%half) -> bucket id%half==0 ? half : id%half
            // So slot id accumulates bucket (id%half) [positive] and bucket (half - (id%half) ... )
            // Simpler: recompute expected Q_st directly from cpu chunks:
            let _ = d;
        }
        let l = num_columns as i64;
        let s_half = (l / 2) as i64;
        for jj in 0..size {
            let digit = cpu_chunks[jj + st * size] as i64 - s_half;
            if digit != 0 {
                let mut term = bases[jj] * ScalarField::from(digit.unsigned_abs());
                if digit < 0 { term = -term; }
                expect += term;
            }
        }
        let mut got = G::zero();
        for t in 0..b_wg {
            let slot = st * b_wg + t;
            got += limbs_to_point(
                &gx[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
                &gy[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
                &gz[slot * NUM_LIMBS..(slot + 1) * NUM_LIMBS],
            );
        }
        if got != expect {
            if s4_bad < 8 { println!("stage 4 mismatch: subtask {}", st); }
            s4_bad += 1;
        }
    }
    println!("stage 4 pbpr mismatched subtasks: {}", s4_bad);

    helper.drop_all_buffers();
}
