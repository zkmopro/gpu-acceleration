use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand, One, UniformRand, Zero};
use num_bigint::BigUint;
use rand::Rng;

use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;

#[test]
#[serial_test::serial]
fn test_smvp() {
    let log_limb_size = 16;
    let num_limbs = 16;
    let num_columns = 16;
    let half_columns = num_columns / 2;
    let num_rows = 16; // Must match the size used in row_ptr_host
    let max_entries = 20; // Maximum number of nonzero entries to generate

    let config = MetalConfig {
        log_limb_size,
        num_limbs,
        shader_file: "cuzk/smvp.metal".to_string(),
        kernel_name: "smvp".to_string(),
    };

    let mut helper = MetalHelper::new();
    let constants = get_or_calc_constants(num_limbs, log_limb_size);
    let p_biguint = &constants.p;
    let rinv = &constants.rinv;

    // ------------------------------------------------------------------
    // Generate random sparse matrix structure
    // ------------------------------------------------------------------
    let (row_ptr_host, val_idx_host) = generate_random_matrix(num_rows, num_columns, max_entries);

    // We need to ensure input_size is correct (total number of non-zero entries)
    let input_size = *row_ptr_host.last().unwrap() as u32;

    // We thus need new_point_x,y to hold columns from 0..(num_columns-1)
    let mut rng = rand::thread_rng();
    let mut new_point_x_host = Vec::with_capacity(num_columns);
    let mut new_point_y_host = Vec::with_capacity(num_columns);

    // For each column, create random X,Y in Mont form
    for _ in 0..num_columns {
        let new_point = G::rand(&mut rng).into_affine();
        let x_mont = new_point.x.0;
        let y_mont = new_point.y.0;
        let x_mont_biguint: BigUint = x_mont.try_into().unwrap();
        let y_mont_biguint: BigUint = y_mont.try_into().unwrap();
        new_point_x_host.push(x_mont_biguint);
        new_point_y_host.push(y_mont_biguint);
    }

    let new_point_x_limbs = new_point_x_host
        .iter()
        .map(|bi| {
            let ark_form: BigInt<4> = bi.clone().try_into().unwrap();
            ark_form.to_limbs(num_limbs, log_limb_size)
        })
        .flatten()
        .collect::<Vec<u32>>();

    let new_point_y_limbs = new_point_y_host
        .iter()
        .map(|bi| {
            let ark_form: BigInt<4> = bi.clone().try_into().unwrap();
            ark_form.to_limbs(num_limbs, log_limb_size)
        })
        .flatten()
        .collect::<Vec<u32>>();

    // ------------------------------------------------------------------
    // Create Metal buffers
    // ------------------------------------------------------------------
    let row_ptr_buf = helper.create_input_buffer(&row_ptr_host);
    let val_idx_buf = helper.create_input_buffer(&val_idx_host);
    let new_point_x_buf = helper.create_input_buffer(&new_point_x_limbs);
    let new_point_y_buf = helper.create_input_buffer(&new_point_y_limbs);

    // half_columns final buckets => each is a Jacobian coordinate => 3 * num_limbs each
    let bucket_x_buf = helper.create_output_buffer(half_columns * num_limbs);
    let bucket_y_buf = helper.create_output_buffer(half_columns * num_limbs);
    let bucket_z_buf = helper.create_output_buffer(half_columns * num_limbs);

    // params: input_size, num_y_workgroups=1, num_z_workgroups=1, subtask_offset=0
    let params = vec![input_size, 1u32, 1u32, 0u32];
    let params_buf = helper.create_input_buffer(&params);

    // ------------------------------------------------------------------
    // Execute shader
    // ------------------------------------------------------------------
    // Each thread is 1D in x dimension => we have half_columns threads
    let threads_per_grid = helper.create_thread_group_size(half_columns as u64, 1, 1);
    let threads_per_threadgroup = helper.create_thread_group_size(1, 1, 1);

    helper.execute_shader(
        &config,
        &[
            &row_ptr_buf,
            &val_idx_buf,
            &new_point_x_buf,
            &new_point_y_buf,
            &bucket_x_buf,
            &bucket_y_buf,
            &bucket_z_buf,
            &params_buf,
        ],
        &[],
        &threads_per_grid,
        &threads_per_threadgroup,
    );

    // ------------------------------------------------------------------
    // Read back the results from bucket_x,y,z
    // ------------------------------------------------------------------
    // Each bucket coordinate is 16 limbs => the entire array is half_columns buckets * 16 limbs
    let bucket_x_out_limbs = helper.read_results(&bucket_x_buf, half_columns * num_limbs);
    let bucket_y_out_limbs = helper.read_results(&bucket_y_buf, half_columns * num_limbs);
    let bucket_z_out_limbs = helper.read_results(&bucket_z_buf, half_columns * num_limbs);

    // Clean up all Metal resources
    helper.drop_all_buffers();

    // ------------------------------------------------------------------
    // CPU reference check and smvp logic
    //
    // We'll replicate the logic from smvp: for each thread id in [0..half_columns-1],
    // we do j=0..1 => find row_idx => accumulate => maybe negate => store in bucket[id].
    //
    // skipped constants:
    //   subtask_idx= id/half_columns => always 0
    //   rp_offset= 0*(half_columns+1)= 0
    //
    // ------------------------------------------------------------------

    fn decode_mont(
        limbs: &[u32],
        r_inv: &num_bigint::BigUint,
        p: &num_bigint::BigUint,
    ) -> BaseField {
        // Turn the "limbs" back into a BigUint
        let big: BigUint = BigInt::<4>::from_limbs(limbs, 16) // 16 = log_limb_size
            .try_into()
            .unwrap();
        let big = &big * r_inv % p;
        // Then convert to ark_ff::BigInt -> BaseField
        let ark_big: BigInt<4> = big.try_into().unwrap();
        BaseField::from_bigint(ark_big).unwrap()
    }

    let neg = |mut pt: G| {
        pt.y = -pt.y;
        pt
    };

    // Convert an "(x_mont, y_mont) in BN254 base field Mont form" to an Affine G1 with Z=1
    let decode_affine = |xm: &num_bigint::BigUint, ym: &num_bigint::BigUint| {
        let x_big = (xm * rinv) % p_biguint;
        let y_big = (ym * rinv) % p_biguint;
        let x_ark = <BaseField as PrimeField>::from_bigint(x_big.try_into().unwrap()).unwrap();
        let y_ark = <BaseField as PrimeField>::from_bigint(y_big.try_into().unwrap()).unwrap();
        G::new(x_ark, y_ark, BaseField::one())
    };

    // Put new_point_x_host,y_host (already Mont biguint) into an array of G in normal form
    let mut new_points = Vec::with_capacity(num_columns);
    for i in 0..num_columns {
        let gx = decode_affine(&new_point_x_host[i], &new_point_y_host[i]);
        new_points.push(gx);
    }

    let mut cpu_buckets = vec![G::default(); half_columns];
    let identity = G::zero();

    for id in 0..half_columns {
        for j in 0..2 {
            // row_idx = (id % half_columns) + half_columns => j=0
            // row_idx = half_columns - (id % half_columns) => j=1
            let mut row_idx = (id % half_columns) + half_columns;
            if j == 1 {
                row_idx = half_columns - (id % half_columns);
            }
            // special case override if j==0 && id%half_columns==0 => row_idx=0
            if j == 0 && (id % half_columns) == 0 {
                row_idx = 0;
            }

            let row_begin = row_ptr_host[row_idx as usize];
            let row_end = row_ptr_host[row_idx as usize + 1];
            // accumulate
            let mut sum = identity;
            for k in row_begin..row_end {
                let col_idx = val_idx_host[k as usize] as usize;
                sum += new_points[col_idx];
            }
            // check sign
            let mut bucket_idx = 0;
            if half_columns > row_idx {
                // negative => flip sign
                bucket_idx = half_columns - row_idx;
                sum = neg(sum);
            } else {
                bucket_idx = row_idx - half_columns;
            }
            // store
            if bucket_idx > 0 {
                if j == 1 {
                    sum += cpu_buckets[id as usize];
                }
                cpu_buckets[id as usize] = sum;
            }
        }
    }

    for id in 0..half_columns {
        let x_slice = &bucket_x_out_limbs[id * num_limbs..(id + 1) * num_limbs];
        let y_slice = &bucket_y_out_limbs[id * num_limbs..(id + 1) * num_limbs];
        let z_slice = &bucket_z_out_limbs[id * num_limbs..(id + 1) * num_limbs];

        let gx = decode_mont(x_slice, rinv, p_biguint);
        let gy = decode_mont(y_slice, rinv, p_biguint);
        let gz = decode_mont(z_slice, rinv, p_biguint);
        let gpu_point = G::new(gx, gy, gz);

        // Compare to CPU
        let cpu_point = cpu_buckets[id];

        let diff = gpu_point - cpu_point;
        assert!(
            diff.is_zero(),
            "Mismatch at id={}, GPU vs CPU. GPU={:?}, CPU={:?}",
            id,
            gpu_point,
            cpu_point
        );
    }
}

/// Generate random sparse matrix structure
fn generate_random_matrix(
    num_rows: usize,
    num_columns: usize,
    max_nonzero_entries: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut rng = rand::thread_rng();

    let nonzero_entries = rng.gen_range(1..=max_nonzero_entries);
    let mut row_ptr = vec![0; num_rows + 1];
    let mut entries_per_row = vec![0; num_rows];

    // Distribute nonzero entries across rows randomly
    for _ in 0..nonzero_entries {
        let row = rng.gen_range(0..num_rows);
        entries_per_row[row] += 1;
    }

    // Calculate row_ptr values based on entries_per_row
    for i in 0..num_rows {
        row_ptr[i + 1] = row_ptr[i] + entries_per_row[i];
    }

    // Generate random column indices for each nonzero entry
    let mut val_idx = Vec::with_capacity(nonzero_entries);
    for row in 0..num_rows {
        for _ in 0..entries_per_row[row] {
            val_idx.push(rng.gen_range(0..num_columns) as u32);
        }
    }

    (row_ptr, val_idx)
}
