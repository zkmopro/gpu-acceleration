// source: https://github.com/geometryxyz/msl-secp256k1

/*
 * It is necessary to hardcode certain constants into MSL source code but dynamically generate the
 * code so that the Rust binary that runs a shader can insert said constants.
 *
 * Shader lifecycle:
 *
 * MSL source -> Compiled .metallib file -> Loaded by program -> Sent to GPU
 *
 * xcrun -sdk macosx metal -c <path.metal> -o <path.ir>
 * xcrun -sdk macosx metallib <path.ir> -o <path.metallib>
 */

use crate::msm::metal_msm::utils::barrett_params::calc_barrett_mu;
use crate::msm::metal_msm::utils::limbs_conversion::GenericLimbConversion;
use crate::msm::metal_msm::utils::mont_params::calc_mont_radix;
use ark_bn254::{Fq as BaseField, G1Projective as G};
use ark_ec::Group;
use ark_ff::{BigInt, PrimeField, Zero};
use num_bigint::BigUint;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::string::String;

macro_rules! write_constant_array {
    ($data:expr, $name:expr, $values:expr, $size:expr) => {
        $data += format!("constant uint32_t {}[{}] = {{\n", $name, $size).as_str();
        $data += &$values
            .iter()
            .map(|v| format!("    {}", v))
            .collect::<Vec<_>>()
            .join(",\n");
        $data += "\n};\n";
    };
}

/// Get the shader directory path using CARGO_MANIFEST_DIR for proper OS file location
/// This function provides robust path resolution that works across different build environments
pub fn get_shader_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_path = manifest_dir
        .join("src")
        .join("msm")
        .join("metal_msm")
        .join("shader");

    // Verify the path exists, if not try alternative locations
    if shader_path.exists() {
        shader_path
    } else {
        // Fallback: try relative path from workspace root
        let workspace_shader_path = manifest_dir
            .parent() // go up one level from mopro-msm to workspace root
            .unwrap_or(&manifest_dir)
            .join("mopro-msm")
            .join("src")
            .join("msm")
            .join("metal_msm")
            .join("shader");

        if workspace_shader_path.exists() {
            workspace_shader_path
        } else {
            // Final fallback: use the original path and let error handling take care of it
            shader_path
        }
    }
}

pub fn compile_metal(path_from_cargo_manifest_dir: &str, input_filename: &str) -> String {
    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(path_from_cargo_manifest_dir)
        .join(input_filename);
    let c = input_path.clone().into_os_string().into_string().unwrap();

    let lib = input_path.clone().into_os_string().into_string().unwrap();
    let lib = format!("{}.lib", lib);

    let exe = if cfg!(target_os = "ios") {
        Command::new("xcrun")
            .args([
                "-sdk",
                "iphoneos",
                "metal",
                "-std=metal3.2",
                "-target",
                "air64-apple-ios18.0",
                "-fmetal-enable-logging",
                "-o",
                lib.as_str(),
                c.as_str(),
            ])
            .output()
            .expect("failed to compile")
    } else if cfg!(target_os = "macos") {
        let macos_version = std::process::Command::new("sw_vers")
            .args(["-productVersion"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|version| {
                version
                    .trim()
                    .split('.')
                    .next()
                    .and_then(|major| major.parse::<u32>().ok())
            })
            .unwrap_or(0);

        let mut args = vec!["-sdk", "macosx", "metal"];

        // Only specify Metal 3.2 for metal logging if macOS version is 15.0 or higher
        if macos_version >= 15 {
            args.extend([
                "-std=metal3.2",
                "-target",
                "air64-apple-macos15.0",
                "-fmetal-enable-logging",
            ]);
        }

        args.extend(["-o", lib.as_str(), c.as_str()]);

        Command::new("xcrun")
            .args(args)
            .output()
            .expect("failed to compile")
    } else {
        panic!("Unsupported architecture");
    };

    if exe.stderr.len() != 0 {
        panic!("{}", String::from_utf8(exe.stderr).unwrap());
    }

    lib
}

pub fn write_constants(
    filepath: &str,
    num_limbs: usize,
    log_limb_size: u32,
    n0: u32,
    nsafe: usize,
) {
    let two_pow_word_size = 2u32.pow(log_limb_size);
    let mask = two_pow_word_size - 1u32;
    let slack = num_limbs as u32 * log_limb_size - BaseField::MODULUS_BIT_SIZE;
    let num_limbs_wide = num_limbs + 1;
    let num_limbs_extra_wide = num_limbs * 2;

    // MSM instance params
    let chunk_size = 16;
    let num_columns = 2u32.pow(chunk_size);
    let num_subtasks = (256 as f32 / chunk_size as f32).ceil() as u32;

    let basefield_modulus = BaseField::MODULUS.to_limbs(num_limbs, log_limb_size);
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let mu_in_ark: BigInt<4> = calc_barrett_mu(&p).try_into().unwrap();
    let mont_radix_limbs: Vec<u32> = {
        let mont_radix_limbs: BigInt<6> = r.clone().try_into().unwrap();
        mont_radix_limbs.to_limbs(num_limbs_wide, log_limb_size) // num_limbs_wide because mont_radix is 257 bits
    };
    let (bn254_zero, bn254_one) = {
        let bn254_zero = G::zero();
        let bn254_one = G::generator();
        (bn254_zero, bn254_one)
    };

    let mut data = "// THIS FILE IS AUTOGENERATED BY shader.rs\n".to_owned();
    data += "#pragma once\n";
    data += format!("#define NUM_LIMBS {}\n", num_limbs).as_str();
    data += format!("#define NUM_LIMBS_WIDE {}\n", num_limbs_wide).as_str();
    data += format!("#define NUM_LIMBS_EXTRA_WIDE {}\n", num_limbs_extra_wide).as_str();
    data += format!("#define LOG_LIMB_SIZE {}\n", log_limb_size).as_str();
    data += format!("#define TWO_POW_WORD_SIZE {}\n", two_pow_word_size).as_str();
    data += format!("#define MASK {}\n", mask).as_str();
    data += format!("#define N0 {}\n", n0).as_str();
    data += format!("#define NSAFE {}\n", nsafe).as_str();
    data += format!("#define SLACK {}\n", slack).as_str();
    data += format!("#define CHUNK_SIZE {}\n", chunk_size).as_str();
    data += format!("#define NUM_COLUMNS {}\n", num_columns).as_str();
    data += format!("#define NUM_SUBTASKS {}\n", num_subtasks).as_str();

    let mu_limbs = mu_in_ark.to_limbs(num_limbs, log_limb_size);
    write_constant_array!(data, "BARRETT_MU", mu_limbs, "NUM_LIMBS");

    write_constant_array!(
        data,
        "BN254_BASEFIELD_MODULUS",
        basefield_modulus,
        "NUM_LIMBS"
    );
    write_constant_array!(data, "MONT_RADIX", mont_radix_limbs, "NUM_LIMBS_WIDE");

    let bn254_zero_limbs = |c: &ark_ff::BigInt<4>| c.to_limbs(num_limbs, log_limb_size);
    write_constant_array!(
        data,
        "BN254_ZERO_X",
        bn254_zero_limbs(&bn254_zero.x.into()),
        "NUM_LIMBS"
    );
    write_constant_array!(
        data,
        "BN254_ZERO_Y",
        bn254_zero_limbs(&bn254_zero.y.into()),
        "NUM_LIMBS"
    );
    write_constant_array!(
        data,
        "BN254_ZERO_Z",
        bn254_zero_limbs(&bn254_zero.z.into()),
        "NUM_LIMBS"
    );

    let bn254_one_limbs = |c: &ark_ff::BigInt<4>| c.to_limbs(num_limbs, log_limb_size);
    write_constant_array!(
        data,
        "BN254_ONE_X",
        bn254_one_limbs(&bn254_one.x.into()),
        "NUM_LIMBS"
    );
    write_constant_array!(
        data,
        "BN254_ONE_Y",
        bn254_one_limbs(&bn254_one.y.into()),
        "NUM_LIMBS"
    );
    write_constant_array!(
        data,
        "BN254_ONE_Z",
        bn254_one_limbs(&bn254_one.z.into()),
        "NUM_LIMBS"
    );

    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let (bn254_zero_xr_limbs, bn254_zero_yr_limbs, bn254_zero_zr_limbs) = {
        let bn254_zero_x: BigUint = bn254_zero.x.into();
        let bn254_zero_y: BigUint = bn254_zero.y.into();
        let bn254_zero_z: BigUint = bn254_zero.z.into();
        let bn254_zero_xr = (bn254_zero_x * &r) % &p;
        let bn254_zero_yr = (bn254_zero_y * &r) % &p;
        let bn254_zero_zr = (bn254_zero_z * &r) % &p;
        (
            ark_ff::BigInt::<4>::try_from(bn254_zero_xr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
            ark_ff::BigInt::<4>::try_from(bn254_zero_yr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
            ark_ff::BigInt::<4>::try_from(bn254_zero_zr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
        )
    };
    write_constant_array!(data, "BN254_ZERO_XR", bn254_zero_xr_limbs, "NUM_LIMBS");
    write_constant_array!(data, "BN254_ZERO_YR", bn254_zero_yr_limbs, "NUM_LIMBS");
    write_constant_array!(data, "BN254_ZERO_ZR", bn254_zero_zr_limbs, "NUM_LIMBS");

    let (bn254_one_xr_limbs, bn254_one_yr_limbs, bn254_one_zr_limbs) = {
        let bn254_one_x: BigUint = bn254_one.x.into();
        let bn254_one_y: BigUint = bn254_one.y.into();
        let bn254_one_z: BigUint = bn254_one.z.into();
        let bn254_one_xr = (bn254_one_x * &r) % &p;
        let bn254_one_yr = (bn254_one_y * &r) % &p;
        let bn254_one_zr = (bn254_one_z * &r) % &p;
        (
            ark_ff::BigInt::<4>::try_from(bn254_one_xr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
            ark_ff::BigInt::<4>::try_from(bn254_one_yr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
            ark_ff::BigInt::<4>::try_from(bn254_one_zr)
                .unwrap()
                .to_limbs(num_limbs, log_limb_size),
        )
    };
    write_constant_array!(data, "BN254_ONE_XR", bn254_one_xr_limbs, "NUM_LIMBS");
    write_constant_array!(data, "BN254_ONE_YR", bn254_one_yr_limbs, "NUM_LIMBS");
    write_constant_array!(data, "BN254_ONE_ZR", bn254_one_zr_limbs, "NUM_LIMBS");

    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(filepath)
        .join("constants.metal");
    fs::write(output_path, data).expect("Unable to write constants file");
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    pub fn test_compile() {
        let lib_filepath = compile_metal(
            "../mopro-msm/src/msm/metal_msm/shader",
            "bigint/bigint_add_unsafe.metal",
        );
        println!("{}", lib_filepath);
    }

    #[test]
    #[serial_test::serial]
    pub fn test_write_constants() {
        write_constants("../mopro-msm/src/msm/metal_msm/shader", 16, 16, 25481, 1);
    }
}
