use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::metal_wrapper::*;
use ark_bn254::{Fq as BaseField, Fr as ScalarField, G1Affine as Affine, G1Projective as G};
use ark_ec::{Group, CurveGroup};
use ark_ff::{BigInt, One, PrimeField, Zero};
use ark_std::{rand::thread_rng, UniformRand};
use num_bigint::BigUint;
use std::error::Error;

/// The CPU pipeline for reproducing the MSM result from the GPU code.
pub fn cpu_reproduce_msm(bases: &[Affine], scalars: &[ScalarField]) -> Result<G, Box<dyn Error>> {
    let chunk_size = if bases.len() >= 65536 { 16 } else { 4 };
    let num_subtasks = 256 / chunk_size;
    let num_columns = 1 << chunk_size; // 2^chunk_size
    let half_columns = num_columns / 2;

    let msm_config = MetalConfig {
        log_limb_size: 16,
        num_limbs: 16,
        shader_file: "".to_string(),
        kernel_name: "".to_string(),
    };
    let msm_constants = get_or_calc_constants(msm_config.num_limbs, msm_config.log_limb_size);

    // 1) Convert Ark `Affine` and `ScalarField` arrays into the "packed" format that
    //    GPU code expects: each point => 16 u32 for coords, each scalar => 8 u32.
    let input_size = bases.len();
    let (packed_coords, packed_scalars) = pack_affine_and_scalars(bases, scalars, &msm_config);

    println!("\n===== INPUT FOR convert_point_coords_and_decompose_scalars shaders =====");
    println!("packed_coords: {:?}", packed_coords);
    println!("packed_scalars: {:?}", packed_scalars);
    println!("input_size: {}", input_size);

    // 2) Allocate arrays for the results of convert+decompose.
    let mut point_x = vec![BaseField::zero(); input_size];
    let mut point_y = vec![BaseField::zero(); input_size];
    let mut chunks = vec![0u32; input_size * num_subtasks];

    // 3) Call CPU version of `convert_point_coords_and_decompose_scalars`
    convert_point_coords_and_decompose_scalars(
        &packed_coords,
        &packed_scalars,
        input_size,
        &mut point_x,
        &mut point_y,
        &mut chunks,
        &msm_constants,
        &msm_config,
        chunk_size as u32,
        num_subtasks,
    )?;

    println!("✅ [CPU] convert_point_coords_and_decompose_scalars");
    println!("\n===== OUTPUT FROM convert_point_coords_and_decompose_scalars shaders =====");
    println!("[Affine] point_x:");
    for (i, pt_coord) in point_x.iter().enumerate() {
        println!("point_x[{}] = {:?}", i, convert_coord_to_u32(pt_coord));
    }
    println!("[Affine] point_y:");
    for (i, pt_coord) in point_y.iter().enumerate() {
        println!("point_y[{}] = {:?}", i, convert_coord_to_u32(pt_coord));
    }
    println!("chunks: {:?}", chunks);

    println!("\n===== INPUT FOR transpose_cpu shaders =====");
    println!("chunks: {:?}", chunks);
    println!("num_subtasks: {}", num_subtasks);
    println!("input_size: {}", input_size);
    println!("num_columns: {}", num_columns);

    // 4) Transpose (CSR->CSC). We'll produce all_csc_col_ptr + all_csc_val_idxs.
    let (all_csc_col_ptr, all_csc_val_idxs) = transpose_cpu(
        &chunks, // interpret as all_csr_col_idx
        num_subtasks as u32,
        input_size as u32,
        num_columns as u32,
    );

    println!("✅ [CPU] transpose");
    println!("\n===== OUTPUT FROM transpose_cpu shaders =====");
    println!("all_csc_col_ptr: {:?}", all_csc_col_ptr);
    println!("all_csc_val_idxs: {:?}", all_csc_val_idxs);

    println!("\n===== INPUT FOR smvp_cpu shaders =====");
    println!("all_csc_col_ptr: {:?}", all_csc_col_ptr);
    println!("all_csc_val_idxs: {:?}", all_csc_val_idxs);
    println!("[Affine] point_x:");
    for (i, pt_coord) in point_x.iter().enumerate() {
        println!("point_x[{}] = {:?}", i, convert_coord_to_u32(pt_coord));
    }
    println!("[Affine] point_y:");
    for (i, pt_coord) in point_y.iter().enumerate() {
        println!("point_y[{}] = {:?}", i, convert_coord_to_u32(pt_coord));
    }
    println!("num_subtasks: {}", num_subtasks);
    println!("num_columns: {}", num_columns);

    // 5) SMVP: we'll build final "bucket arrays."
    //    On GPU, you have separate subtask offsets, 2D thread distribution, etc.
    //    On CPU, we can do a simpler for-loops that replicate the logic:
    let (bucket_x, bucket_y, bucket_z) = smvp_cpu(
        &all_csc_col_ptr,
        &all_csc_val_idxs,
        &point_x,
        &point_y,
        num_subtasks,
        num_columns as u32,
        input_size,
        &msm_constants,
        &msm_config,
    );

    println!("✅ [CPU] smvp");
    println!("\n===== OUTPUT FROM smvp_cpu shaders =====");
    println!("[Jacobian] bucket_x:");
    for (i, bucket) in bucket_x.iter().enumerate() {
        println!("bucket_x[{}] = {:?}", i, convert_coord_to_u32(bucket));
    }
    println!("[Jacobian] bucket_y:");
    for (i, bucket) in bucket_y.iter().enumerate() {
        println!("bucket_y[{}] = {:?}", i, convert_coord_to_u32(bucket));
    }
    println!("[Jacobian] bucket_z:");
    for (i, bucket) in bucket_z.iter().enumerate() {
        println!("bucket_z[{}] = {:?}", i, convert_coord_to_u32(bucket));
    }

    println!("\n===== INPUT FOR parallel_bpr_cpu shaders =====");
    println!("bucket_x, bucket_y, bucket_z is the same as previous output");
    println!("num_subtasks: {}", num_subtasks);
    println!("half_columns: {}", half_columns);

    // 6) Parallel Bucket Reduction: combine all buckets
    //    In the GPU code, you do 2-stage partial sums + scalar mul, per subtask.
    //    We'll replicate a simpler approach that yields the same final partial G for each subtask.
    let subtask_pts = parallel_bpr_cpu(
        &bucket_x,
        &bucket_y,
        &bucket_z,
        num_subtasks,
        half_columns as usize,
        &msm_constants,
        &msm_config,
    );

    println!("✅ [CPU] parallel_bpr");
    println!("\n===== OUTPUT FROM parallel_bpr_cpu shaders =====");
    println!("subtask_pts:");
    for (i, pt) in subtask_pts.iter().enumerate() {
        println!("  subtask_pts[{}]:", i);
        println!(
            "    x: {:?}",
            pt.x.0
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size)
        );
        println!(
            "    y: {:?}",
            pt.y.0
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size)
        );
        println!(
            "    z: {:?}",
            pt.z.0
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size)
        );
    }

    println!("\n===== INPUT FOR horner's method shaders =====");
    println!("subtask_pts is the same as previous output");
    println!("chunk_size: {}", chunk_size);

    // 7) Horner's Method: combine the `subtask_pts` in base = 2^chunk_size
    //    (like your final Typescript lines).
    let base = ScalarField::from(1u64 << chunk_size);
    let mut acc = subtask_pts[subtask_pts.len() - 1];
    for i in (0..subtask_pts.len() - 1).rev() {
        acc *= base;
        acc += subtask_pts[i];
    }

    println!("✅ [CPU] horner's method");
    println!("\n===== OUTPUT FROM horner's method shaders =====");
    println!("MSM result: {:?}", acc);

    // 8) Return final projective
    Ok(acc)
}

/// Packs each affine BN254 point into 16 u32 "halfword layout" + each scalar into 8 u32
pub fn pack_affine_and_scalars(
    bases: &[Affine],
    scalars: &[ScalarField],
    msm_config: &MetalConfig,
) -> (Vec<u32>, Vec<u32>) {
    let mut coords = Vec::new();
    let mut scalars_u32 = Vec::new();

    let pack_limbs = |limbs: &[u32]| -> Vec<u32> {
        limbs
            .chunks(2)
            .map(|chunk| (chunk[1] << 16) | chunk[0])
            .collect()
    };

    for (pt, sc) in bases.iter().zip(scalars.iter()) {
        let x_limbs =
            pt.x.into_bigint()
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);
        let y_limbs =
            pt.y.into_bigint()
                .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);

        let x_packed = pack_limbs(&x_limbs);
        let y_packed = pack_limbs(&y_limbs);
        coords.extend_from_slice(&x_packed);
        coords.extend_from_slice(&y_packed);

        let sc_limbs = sc
            .into_bigint()
            .to_limbs(msm_config.num_limbs, msm_config.log_limb_size);
        let sc_packed = pack_limbs(&sc_limbs);
        scalars_u32.extend_from_slice(&sc_packed);
    }

    (coords, scalars_u32)
}

pub fn convert_point_coords_and_decompose_scalars(
    packed_coords: &[u32],
    scalars: &[u32],
    input_size: usize,
    point_x: &mut [BaseField],
    point_y: &mut [BaseField],
    chunks: &mut [u32],
    msm_constants: &MSMConstants,
    _msm_config: &MetalConfig,
    chunk_size: u32,
    num_subtasks: usize,
) -> Result<(), Box<dyn Error>> {
    for i in 0..input_size {
        // -------------------------------------------------------
        // (1) Convert X,Y from 8 u32 => 16 reversed halfwords => BaseField
        // -------------------------------------------------------
        let coord_offset = i * 16;
        let x_32 = &packed_coords[coord_offset..coord_offset + 8];
        let y_32 = &packed_coords[coord_offset + 8..coord_offset + 16];

        // Rebuild halfwords for x
        let mut x_halfs = [0u16; 16];
        for (j, &val) in x_32.iter().enumerate() {
            // GPU logic: x_bytes[15 - 2*j] = low16; x_bytes[15 - (2*j +1)] = hi16
            x_halfs[15 - (j * 2)] = (val & 0xFFFF) as u16;
            x_halfs[15 - (j * 2) - 1] = (val >> 16) as u16;
        }

        // Rebuild halfwords for y
        let mut y_halfs = [0u16; 16];
        for (j, &val) in y_32.iter().enumerate() {
            y_halfs[15 - (j * 2)] = (val & 0xFFFF) as u16;
            y_halfs[15 - (j * 2) - 1] = (val >> 16) as u16;
        }

        let x_big = biguint_from_u16_le(&x_halfs);
        let y_big = biguint_from_u16_le(&y_halfs);

        // Convert BigUint => BaseField (ark_bn254::Fq)
        // Ark's Fq is *already* in Mont form internally, so no extra multiply by R needed.
        let x_fq = BaseField::from(BigInt::<4>::try_from(x_big).unwrap());
        let y_fq = BaseField::from(BigInt::<4>::try_from(y_big).unwrap());

        point_x[i] = x_fq;
        point_y[i] = y_fq;

        // -------------------------------------------------------
        // (2) Decompose the scalar i => wNAF chunk
        // -------------------------------------------------------
        let scalar_offset = i * 8;
        let s_32 = &scalars[scalar_offset..scalar_offset + 8];

        // Rebuild scalar halfwords, reversed
        let mut s_halfs = [0u16; 16];
        for (j, &val) in s_32.iter().enumerate() {
            s_halfs[15 - (j * 2)] = (val & 0xFFFF) as u16;
            s_halfs[15 - (j * 2) - 1] = (val >> 16) as u16;
        }

        // Extract windows of size = chunk_size
        let scalar_chunks = extract_signed_chunks(&s_halfs, chunk_size);

        // Store them => chunks[j*input_size + i]
        for (j, &c) in scalar_chunks.iter().enumerate() {
            chunks[j * input_size + i] = c;
        }
    }

    Ok(())
}

/// Helper: do exactly the GPU chunk extraction with the sign fix
pub fn extract_signed_chunks(halfs: &[u16; 16], chunk_size: u32) -> Vec<u32> {
    let scalar_bytes = halfs;
    let num_subtasks = (256 / chunk_size) as usize;

    // Extract each window from the byte array
    let mut slices = vec![0u32; num_subtasks];
    for i in 0..num_subtasks - 1 {
        slices[i] = extract_word_from_bytes_le(scalar_bytes, i as u32, chunk_size);
    }

    // Special handling for the last chunk (after testing, 256 is the correct value instead of 254)
    let shift_256 = ((num_subtasks as u32 * chunk_size - 256) + 16) - chunk_size;
    slices[num_subtasks - 1] = scalar_bytes[0] as u32 >> shift_256;

    println!("slices       : {:?}", slices);

    // Apply sign logic
    let l = 1u32 << chunk_size;
    let s = l >> 1; // l/2
    let mut carry = 0i32;
    let mut signed_slices = vec![0i32; num_subtasks];

    for i in 0..num_subtasks {
        let raw_val = (slices[i] as i32) + carry;
        if raw_val >= s as i32 {
            signed_slices[i] = (l as i32 - raw_val) * -1;
            carry = 1;
        } else {
            signed_slices[i] = raw_val;
            carry = 0;
        }
    }

    if carry == 1 {
        panic!("carry should always be 0");
    }

    println!("signed_slices: {:?}", signed_slices);

    // Convert back to unsigned representation with offset
    for i in 0..num_subtasks {
        slices[i] = (signed_slices[i] + s as i32) as u32;
    }

    slices
}

/// Helper to extract a chunk of bits from the byte array
pub fn extract_word_from_bytes_le(bytes: &[u16; 16], word_idx: u32, chunk_size: u32) -> u32 {
    let start_byte_idx = 15 - ((word_idx * chunk_size + chunk_size) / 16);
    let end_byte_idx = 15 - ((word_idx * chunk_size) / 16);
    let start_byte_offset = (word_idx * chunk_size + chunk_size) % 16;
    let end_byte_offset = (word_idx * chunk_size) % 16;

    let mut mask = 0u32;
    if start_byte_offset > 0 {
        mask = (2 << (start_byte_offset - 1)) - 1;
    }

    if start_byte_idx == end_byte_idx {
        ((bytes[start_byte_idx as usize] as u32) & mask) >> end_byte_offset
    } else {
        let part1 = ((bytes[start_byte_idx as usize] as u32) & mask) << (16 - end_byte_offset);
        let part2 = (bytes[end_byte_idx as usize] as u32) >> end_byte_offset;
        part1 | part2
    }
}

/// Helper: build BigUint from 16 u16 in little-endian order
pub fn biguint_from_u16_le(halfs: &[u16; 16]) -> BigUint {
    println!("halfs: {:?}", halfs);
    let mut acc = BigUint::zero();
    let mut shift = 0u32;
    for &h in halfs.iter().rev() {
        let val = BigUint::from(h);
        acc |= val << shift;
        shift += 16;
    }
    acc
}

pub fn transpose_cpu(
    chunks: &[u32],
    num_subtasks: u32,
    input_size: u32,
    n: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let num_subtasks_usize = num_subtasks as usize;
    let input_size_usize = input_size as usize;

    // We'll build:
    //   all_csc_col_ptr[subtask][..(n+1)]
    //   all_csc_val_idxs[subtask][..input_size]
    let mut all_csc_col_ptr = vec![vec![0u32; (n + 1) as usize]; num_subtasks_usize];
    let mut all_csc_val_idxs = vec![vec![0u32; input_size_usize]; num_subtasks_usize];

    // Phase 1 + 2 + 3 for each subtask
    for s in 0..num_subtasks_usize {
        let ccp = &mut all_csc_col_ptr[s];
        let cci = &mut all_csc_val_idxs[s];

        // Phase 1: Count
        for i in 0..input_size_usize {
            let col = chunks[s * input_size_usize + i];
            ccp[(col + 1) as usize] += 1;
        }
        // Phase 2: Prefix sum
        for i in 1..=(n as usize) {
            ccp[i] += ccp[i - 1];
        }
        // Phase 3: Build csc_val_idxs
        let mut curr = vec![0u32; n as usize];
        for i in 0..input_size_usize {
            let col = chunks[s * input_size_usize + i];
            let loc = ccp[col as usize];
            let offset = curr[col as usize];
            cci[(loc + offset) as usize] = i as u32;
            curr[col as usize] += 1;
        }
    }

    (all_csc_col_ptr, all_csc_val_idxs)
}

pub fn smvp_cpu(
    all_csc_col_ptr: &[Vec<u32>],
    all_csc_val_idxs: &[Vec<u32>],
    point_x: &[BaseField],
    point_y: &[BaseField],
    num_subtasks: usize,
    num_columns: u32,
    input_size: usize,
    _msm_constants: &MSMConstants,
    _msm_config: &MetalConfig,
) -> (Vec<BaseField>, Vec<BaseField>, Vec<BaseField>) {
    // Each column in [0..num_columns) => one bucket (plus sign logic).
    // We create an array of buckets = (x,y,z) in BaseField form, for all subtasks.
    // total_buckets = num_subtasks * (num_columns/2).
    let half_columns = num_columns / 2;
    let total_buckets = half_columns * (num_subtasks as u32);

    // We'll store each final bucket's G1Projective in (bucket_x, bucket_y, bucket_z).
    // Indices: 0..total_buckets-1.
    // bucket i in subtask s => i = ? We'll figure out the mapping below.
    let mut bucket_x = vec![BaseField::zero(); total_buckets as usize];
    let mut bucket_y = vec![BaseField::zero(); total_buckets as usize];
    let mut bucket_z = vec![BaseField::zero(); total_buckets as usize];

    // For each subtask s:
    for s in 0..num_subtasks {
        // println!("=== subtask: {:?} ===", s);
        let ccp = &all_csc_col_ptr[s]; // csc_col_ptr for subtask s
        let cci = &all_csc_val_idxs[s]; // csc_val_idxs for subtask s

        // For each column col in [0..num_columns):
        for col in 0..num_columns {
            // println!("  --> col: {:?}", col);
            // Gather all the points in that column => sum them
            let row_begin = ccp[col as usize];
            let row_end = ccp[col as usize + 1];

            // We'll accumulate in a G1Projective. Start at identity.
            let mut sum_pt = G::zero();
            for idx in row_begin..row_end {
                // the original point index in [0..input_size)
                let point_idx = cci[idx as usize] as usize;
                // println!("👍 point_idx: {:?}", point_idx);
                // Create an affine point with Z=1
                // Because X=point_x[i], Y=point_y[i], we interpret as an affine point on BN254:
                //   G::new(point_x[i], point_y[i], 1).
                let gx = point_x[point_idx];
                let gy = point_y[point_idx];
                let pt = G::new(gx, gy, BaseField::one());

                sum_pt += pt;
            }

            // In the GPU code, if col < half_columns => negative => sum_pt = -sum_pt
            // Then compute the “bucket index” for that sum. If bucket_idx>0 => store it.
            let bucket_idx;
            if col < half_columns {
                bucket_idx = (half_columns - col) as i32;
                sum_pt = -sum_pt;
            } else {
                bucket_idx = (col - half_columns) as i32;
            }

            // print!("bucket_idx: {:?}, ", bucket_idx);
            // if bucket_idx == 0 {
            //     println!("❌ ignore all points in this col");
            // }

            // If bucket_idx>0 => store in (bucket_x,bucket_y,bucket_z).
            // In the Metal code, we do “bucket_idx-1” for 0-based indexing.
            if bucket_idx > 0 {
                let final_idx = (bucket_idx - 1) as u32 + (s as u32 * half_columns);
                // println!("final_idx: {:?}", final_idx);

                let current_bucket = G::new(
                    bucket_x[final_idx as usize],
                    bucket_y[final_idx as usize],
                    bucket_z[final_idx as usize],
                );
                // println!("---- current_bucket: {:?}", current_bucket);
                let new_bucket = current_bucket + sum_pt;
                // println!("---- new_bucket: {:?}", new_bucket);

                // update the bucket
                bucket_x[final_idx as usize] = new_bucket.x;
                bucket_y[final_idx as usize] = new_bucket.y;
                bucket_z[final_idx as usize] = new_bucket.z;
            }
        }
    }

    (bucket_x, bucket_y, bucket_z)
}

pub fn parallel_bpr_cpu(
    bucket_x: &[BaseField],
    bucket_y: &[BaseField],
    bucket_z: &[BaseField],
    num_subtasks: usize,
    half_columns: usize,
    _msm_constants: &MSMConstants,
    _msm_config: &MetalConfig,
) -> Vec<G> {
    let mut results = vec![G::zero(); num_subtasks];
    let r = half_columns as u32;

    for s in 0..num_subtasks {
        let subtask_start = s as u32 * r;
        let subtask_end = subtask_start + r;

        let mut m_pt = G::zero();
        let mut s_pt = G::zero();

        // accumulate partial sums in reverse
        for l in 1..=r {
            let idx = subtask_end - l;
            let bx = bucket_x[idx as usize];
            let by = bucket_y[idx as usize];
            let bz = bucket_z[idx as usize];

            let b_pt = if bz.is_zero() {
                G::zero()
            } else {
                G::new(bx, by, bz)
            };

            if l == 1 {
                m_pt = b_pt;
                s_pt = m_pt;
            } else {
                m_pt += b_pt;
                s_pt += m_pt;
            }
        }

        // FIX: no extra multiply for subtask offset
        results[s] = s_pt;
    }

    results
}

/// Helper to print a point in Montgomery form.
pub fn convert_coord_to_u32(coords: &BaseField) -> Vec<u32> {
    coords.0.to_limbs(16, 16)
}

/// Helper to get fixed inputs for CPU reproduction test.
pub fn get_fixed_inputs_cpu_style(input_size: usize) -> (Vec<Affine>, Vec<ScalarField>) {
    let mut rng = thread_rng();
    let points = {
        let mut points = vec![G::zero().into_affine(); input_size];
        for i in 0..input_size {
            // points[i] = G::rand(&mut rng).into_affine();
            points[i] = G::generator().into_affine();
        }
        points
    };
    let scalars = {
        let mut scalars = vec![ScalarField::zero(); input_size];
        for i in 0..input_size {
            // scalars[i] = ScalarField::rand(&mut rng);
            scalars[i] = ScalarField::from(1);
        }
        scalars
    };

    (points, scalars)
}

#[test]
fn test_cpu_reproduce_msm() {
    use ark_bn254::G1Projective as G;
    use ark_ec::VariableBaseMSM;
    use ark_ff::BigInteger;

    let input_size = 1;
    let (points, scalars) = get_fixed_inputs_cpu_style(input_size);

    println!("\n===== points =====");
    for (i, pt) in points.iter().enumerate() {
        println!("pt_{}.x: {:?}", i, convert_coord_to_u32(&pt.x));
        println!("pt_{}.y: {:?}", i, convert_coord_to_u32(&pt.y));
    }
    println!("\n===== scalars =====");
    for (i, sc) in scalars.iter().enumerate() {
        println!(
            "{}-bit sc_{}: {:?}",
            sc.into_bigint().num_bits(),
            i,
            sc.into_bigint().to_limbs(16, 16)
        );
    }

    // Arkworks reference
    let arkworks_msm = G::msm(&points[..], &scalars[..]).unwrap();

    // Our CPU pipeline
    let result = cpu_reproduce_msm(&points[..], &scalars[..]).unwrap();
    assert_eq!(result, arkworks_msm);
}
