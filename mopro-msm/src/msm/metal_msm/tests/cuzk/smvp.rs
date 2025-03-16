use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};
use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, PrimeField};
use ark_std::{rand, One, UniformRand, Zero};
use metal::*;
use num_bigint::BigUint;

#[test]
#[serial_test::serial]
fn test_smvp() {
    // ------------------------------------------------------------------
    // Set up “BN254” Montgomery parameters
    // ------------------------------------------------------------------
    let log_limb_size = 16;
    let num_limbs = 16;

    let p_biguint = <BaseField as PrimeField>::MODULUS.into();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (rinv, n0) = calc_rinv_and_n0(&p_biguint, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);

    // ------------------------------------------------------------------
    // Create row_ptr, val_idx, new_point_x,y
    //      num_columns=16 => half_columns=8 => 8 threads
    //      do 1 subtask, so id in [0..7]
    //      total number of data entries: row_ptr[16] = 7 => input_size=7
    //
    //    val_idx for those 7 entries:
    //      row0 => [0, 7]
    //      row1 => [3, 10]
    //      row2 => [5]
    //      row3 => [1]
    //      row4 => [15]
    // ------------------------------------------------------------------
    let row_ptr_host = vec![
        0, // row0 starts at index 0
        2, // row1 starts at index 2
        4, // row2 starts at index 4
        5, // row3 starts at index 5
        6, // row4 starts at index 6
        7, // row5 -> still 7 => empty (7..7)
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // up to row15 => all empty
    ];
    let val_idx_host = vec![0, 7, 3, 10, 5, 1, 15];

    // We thus need new_point_x,y to hold columns from 0..15 => length=16
    let mut rng = rand::thread_rng();
    let mut new_point_x_host = Vec::with_capacity(16);
    let mut new_point_y_host = Vec::with_capacity(16);

    // For each column 0..15, create random X,Y in Mont form
    for _ in 0..16 {
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

    // Prepare GPU buffers
    let device = get_default_device();
    let row_ptr_buf = create_buffer(&device, &row_ptr_host);
    let val_idx_buf = create_buffer(&device, &val_idx_host);
    let new_point_x_buf = create_buffer(&device, &new_point_x_limbs);
    let new_point_y_buf = create_buffer(&device, &new_point_y_limbs);

    // 8 final buckets => each is a Jacobian coordinate => 3 * num_limbs each
    let bucket_x_buf = create_empty_buffer(&device, 8 * num_limbs);
    let bucket_y_buf = create_empty_buffer(&device, 8 * num_limbs);
    let bucket_z_buf = create_empty_buffer(&device, 8 * num_limbs);

    // params: input_size=7, num_y_workgroups=1, num_z_workgroups=1, subtask_offset=0
    let params = vec![7u32, 1u32, 1u32, 0u32];
    let params_buf = create_buffer(&device, &params);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader",
        num_limbs,
        log_limb_size,
        n0,
        nsafe,
    );

    let library_path = compile_metal("../mopro-msm/src/msm/metal_msm/shader/cuzk", "smvp.metal");
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("smvp", None).unwrap();

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&row_ptr_buf), 0);
    encoder.set_buffer(1, Some(&val_idx_buf), 0);
    encoder.set_buffer(2, Some(&new_point_x_buf), 0);
    encoder.set_buffer(3, Some(&new_point_y_buf), 0);
    encoder.set_buffer(4, Some(&bucket_x_buf), 0);
    encoder.set_buffer(5, Some(&bucket_y_buf), 0);
    encoder.set_buffer(6, Some(&bucket_z_buf), 0);
    encoder.set_buffer(7, Some(&params_buf), 0);

    // Each thread is 1D in x dimension => we have 8 threads
    let threads_per_grid = MTLSize {
        width: 8,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // ------------------------------------------------------------------
    // Read back the results from bucket_x,y,z
    // ------------------------------------------------------------------
    // Each bucket coordinate is 16 limbs => the entire array is 8 buckets * 16 limbs
    let bucket_x_out_limbs: Vec<u32> = read_buffer(&bucket_x_buf, 8 * num_limbs);
    let bucket_y_out_limbs: Vec<u32> = read_buffer(&bucket_y_buf, 8 * num_limbs);
    let bucket_z_out_limbs: Vec<u32> = read_buffer(&bucket_z_buf, 8 * num_limbs);

    // Drop the buffers after reading the results
    drop(row_ptr_buf);
    drop(val_idx_buf);
    drop(new_point_x_buf);
    drop(new_point_y_buf);
    drop(bucket_x_buf);
    drop(bucket_y_buf);
    drop(bucket_z_buf);
    drop(params_buf);
    drop(command_queue);

    // ------------------------------------------------------------------
    // CPU reference check and smvp logic
    //
    // We’ll replicate the logic from smvp: for each thread id in [0..7],
    // we do j=0..1 => find row_idx => accumulate => maybe negate => store in bucket[id].
    //
    // skipped constants:
    //   subtask_idx= id/8 => always 0
    //   rp_offset= 0*(8+1)= 0
    //
    // ------------------------------------------------------------------

    fn decode_mont(
        limbs: &[u32],
        r_inv: &num_bigint::BigUint,
        p: &num_bigint::BigUint,
    ) -> BaseField {
        // Turn the “limbs” back into a BigUint
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

    // Convert an “(x_mont, y_mont) in BN254 base field Mont form” to an Affine G1 with Z=1
    let decode_affine = |xm: &num_bigint::BigUint, ym: &num_bigint::BigUint| {
        let x_big = (xm * &rinv) % &p_biguint;
        let y_big = (ym * &rinv) % &p_biguint;
        let x_ark = <BaseField as PrimeField>::from_bigint(x_big.try_into().unwrap()).unwrap();
        let y_ark = <BaseField as PrimeField>::from_bigint(y_big.try_into().unwrap()).unwrap();
        G::new(x_ark, y_ark, BaseField::one())
    };

    // Put new_point_x_host,y_host (already Mont biguint) into an array of G in normal form
    let mut new_points = Vec::with_capacity(16);
    for i in 0..16 {
        let gx = decode_affine(&new_point_x_host[i], &new_point_y_host[i]);
        new_points.push(gx);
    }

    let mut cpu_buckets = vec![G::default(); 8];
    let identity = G::zero();

    for id in 0..8 {
        for j in 0..2 {
            let half_columns = 8;
            // row_idx = (id % 8) + 8 => j=0
            // row_idx = 8 - (id % 8) => j=1
            let mut row_idx = (id % half_columns) + half_columns;
            if j == 1 {
                row_idx = half_columns - (id % half_columns);
            }
            // special case override if j==0 && id%8==0 => row_idx=0
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
            let bucket_idx = if row_idx < half_columns as u32 {
                sum = neg(sum);
                half_columns as u32 - row_idx
            } else {
                row_idx - half_columns as u32
            };

            if bucket_idx > 0 {
                if j == 1 {
                    sum += cpu_buckets[id as usize];
                }
                cpu_buckets[id as usize] = sum;
            }
        }
    }

    for id in 0..8 {
        let x_slice = &bucket_x_out_limbs[id * num_limbs..(id + 1) * num_limbs];
        let y_slice = &bucket_y_out_limbs[id * num_limbs..(id + 1) * num_limbs];
        let z_slice = &bucket_z_out_limbs[id * num_limbs..(id + 1) * num_limbs];

        let gx = decode_mont(x_slice, &rinv, &p_biguint);
        let gy = decode_mont(y_slice, &rinv, &p_biguint);
        let gz = decode_mont(z_slice, &rinv, &p_biguint);
        let gpu_point = G::new(gx, gy, gz);

        // Compare to CPU
        let cpu_point = cpu_buckets[id];

        // println!("gpu_point[{}]: {:?}", id, gpu_point);
        // println!("cpu_point[{}]: {:?}", id, cpu_point);

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
