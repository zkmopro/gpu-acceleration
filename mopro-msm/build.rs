use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> std::io::Result<()> {
    compile_shaders()?;
    Ok(())
}

fn compile_shaders() -> std::io::Result<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_out_dir = out_dir.join("shaders");
    fs::create_dir_all(&shader_out_dir)?;

    // Check if we should compile all shaders for testing
    // Check environment variable first, then check if this is a test build
    let compile_all_shaders = env::var("MSM_COMPILE_ALL_SHADERS")
        .map(|v| v == "1")
        .unwrap_or(false)
        || env::var("CARGO_CFG_TEST").is_ok()
        || env::var("PROFILE").map(|p| p == "test").unwrap_or(false)
        || env::args().any(|arg| arg.contains("test"));

    let shader_root = manifest_dir
        .join("src")
        .join("msm")
        .join("metal_msm")
        .join("shader");

    let mut metal_paths = Vec::new();

    if compile_all_shaders {
        println!("cargo:warning=Compiling ALL Metal shaders for testing");
        // Scan entire shader directory for all .metal files
        visit_dirs(&shader_root, &mut metal_paths)?;
        // Filter out .lib files and keep only .metal files
        metal_paths.retain(|path| {
            let path_str = path.to_string_lossy();
            path.extension().and_then(|e| e.to_str()) == Some("metal")
                && !path_str.contains(".lib")
                && !path_str.contains(".metallib")
        });
    } else {
        println!("cargo:warning=Compiling only MSM production shaders");
        // Only scan cuzk directory for MSM shaders
        let cuzk_root = shader_root.join("cuzk");
        visit_dirs(&cuzk_root, &mut metal_paths)?;
        // Filter only the MSM kernels
        metal_paths.retain(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|name| {
                    name.starts_with("convert_point")
                        || name.starts_with("barrett_reduction")
                        || name.starts_with("pbpr")
                        || name.starts_with("smvp")
                        || name.starts_with("transpose")
                })
                .unwrap_or(false)
        });
    }

    if metal_paths.is_empty() {
        panic!("No Metal shader files found to compile");
    }

    println!(
        "cargo:warning=Found {} Metal shaders to compile",
        metal_paths.len()
    );
    for path in &metal_paths {
        println!("cargo:warning=Including shader: {}", path.display());
    }

    // Combine selected kernels into one .metal file
    let combined = shader_out_dir.join("msm_combined.metal");
    let mut combined_src = String::new();
    combined_src.push_str("#include <metal_stdlib>\n#include <metal_math>\n");
    for path in &metal_paths {
        let inc = path.to_str().unwrap();
        combined_src.push_str(&format!("#include \"{}\"\n", inc));
        println!("cargo:rerun-if-changed={}", inc);
    }
    fs::write(&combined, &combined_src)?;

    // Compile combined source to AIR
    let target = env::var("TARGET").unwrap_or_default();
    let sdk = if target.contains("apple-ios") {
        "iphoneos"
    } else {
        "macosx"
    };
    let air = shader_out_dir.join("msm.air");
    let status = Command::new("xcrun")
        .args(&[
            "-sdk",
            sdk,
            "metal",
            "-c",
            combined.to_str().unwrap(),
            "-o",
            air.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to invoke metal");
    if !status.success() {
        panic!("Metal compile failed");
    }

    // Link AIR into msm.metallib
    let msm_lib = shader_out_dir.join("msm.metallib");
    let status = Command::new("xcrun")
        .args(&[
            "-sdk",
            sdk,
            "metallib",
            air.to_str().unwrap(),
            "-o",
            msm_lib.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to invoke metallib");
    if !status.success() {
        panic!("Metallib linking failed");
    }

    // Emit a single built_shaders.rs with embedded msm.metallib
    let dest = out_dir.join("built_shaders.rs");
    let mut f = fs::File::create(&dest)?;
    writeln!(
        f,
        "pub const MSM_METALLIB: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/shaders/msm.metallib\"));"
    )?;
    Ok(())
}

fn visit_dirs(dir: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let p = entry?.path();
            if p.is_dir() {
                visit_dirs(&p, paths)?;
            } else if p.extension().and_then(|e| e.to_str()) == Some("metal") {
                paths.push(p);
            }
        }
    }
    Ok(())
}
