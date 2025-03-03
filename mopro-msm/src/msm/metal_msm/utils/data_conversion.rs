use crate::msm::metal_msm::host::gpu::create_buffer;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;
use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ff::{BigInt, PrimeField};
use metal::*;
use num_bigint::BigUint;

fn point_to_gpu_form(point: &G, num_limbs: usize) -> Vec<u32> {
    let log_limb_size: u32 = 16;
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);

    let px: BigUint = point.x.into();
    let py: BigUint = point.y.into();
    let pz: BigUint = point.z.into();

    let pxr = (&px * &r) % &p;
    let pyr = (&py * &r) % &p;
    let pzr = (&pz * &r) % &p;

    let pxr_limbs = ark_ff::BigInt::<4>::try_from(pxr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let pyr_limbs = ark_ff::BigInt::<4>::try_from(pyr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);
    let pzr_limbs = ark_ff::BigInt::<4>::try_from(pzr)
        .unwrap()
        .to_limbs(num_limbs, log_limb_size);

    let mut point_data = vec![0u32; num_limbs * 3];
    point_data[..num_limbs].copy_from_slice(&pxr_limbs);
    point_data[num_limbs..2 * num_limbs].copy_from_slice(&pyr_limbs);
    point_data[2 * num_limbs..].copy_from_slice(&pzr_limbs);

    point_data
}

pub fn points_to_gpu_buffer(points: &[G], num_limbs: usize, device: &Device) -> metal::Buffer {
    let total_size = points.len() * num_limbs * 3;
    let mut all_point_data = Vec::with_capacity(total_size);

    for point in points {
        let point_data = point_to_gpu_form(point, num_limbs);
        all_point_data.extend_from_slice(&point_data);
    }
    create_buffer(&device, &all_point_data)
}

pub fn points_from_gpu_buffer(
    buffer: &Buffer,
    num_limbs: usize,
    p: BigUint,
    rinv: BigUint,
) -> Vec<G> {
    let log_limb_size = 16;
    let point_size = num_limbs * 3;
    let total_u32s = buffer.length() as usize / std::mem::size_of::<u32>();
    let num_points = total_u32s / point_size;

    let mut points: Vec<G> = Vec::with_capacity(num_points);

    let ptr = buffer.contents() as *const u32;
    let result_limbs: Vec<u32>;

    if !ptr.is_null() {
        result_limbs = unsafe { std::slice::from_raw_parts(ptr, total_u32s) }.to_vec();
    } else {
        panic!("Pointer is null");
    }

    for i in 0..num_points {
        let start = i * point_size;
        let xr = &result_limbs[start..start + num_limbs];
        let yr = &result_limbs[start + num_limbs..start + 2 * num_limbs];
        let zr = &result_limbs[start + 2 * num_limbs..start + 3 * num_limbs];

        let x =
            (BigUint::try_from(BigInt::<4>::from_limbs(xr, log_limb_size)).unwrap() * &rinv) % &p;
        let y =
            (BigUint::try_from(BigInt::<4>::from_limbs(yr, log_limb_size)).unwrap() * &rinv) % &p;
        let z =
            (BigUint::try_from(BigInt::<4>::from_limbs(zr, log_limb_size)).unwrap() * &rinv) % &p;
        let x: BaseField = x.try_into().unwrap();
        let y: BaseField = y.try_into().unwrap();
        let z: BaseField = z.try_into().unwrap();
        points.push(G::new_unchecked(x, y, z));
    }
    points
}
