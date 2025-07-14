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

    build_cpp_header(manifest_dir.clone());

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
        "cargo:warning=Found {} Metal shaders to compile from {}",
        metal_paths.len(),
        shader_root.to_str().unwrap()
    );

    // Combine selected kernels into one .metal file
    let combined = shader_out_dir.join("msm_combined.metal");
    let mut combined_src = String::new();
    combined_src.push_str("#include <metal_stdlib>\n#include <metal_math>\n");
    for path in &metal_paths {
        let inc = path.to_str().unwrap();
        combined_src.push_str(&format!("#include \"{}\"\n", inc));
        println!("cargo:rerun-if-changed={}", inc);
    }

    // Ensure the build script reruns if ANY shader file in the `shader/` tree is
    // touched – even if that shader is currently not selected for inclusion in
    // the combined library.  This makes the `metallib` always stay in sync with
    // the full shader source tree.
    {
        let mut all_shader_paths = Vec::new();
        visit_dirs(&shader_root, &mut all_shader_paths)?;
        for p in all_shader_paths
            .into_iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("metal"))
        {
            println!("cargo:rerun-if-changed={}", p.to_string_lossy());
        }
    }
    fs::write(&combined, &combined_src)?;

    // Determine SDK target
    let target = env::var("TARGET").unwrap_or_default();
    let sdk = if target.contains("apple-ios") {
        "iphoneos"
    } else {
        "macosx"
    };

    // Only detect Metal version if we're not in CI
    let (metal_std, enable_logging) = if env::var("CI").is_err() {
        let (metal_version, metal_std, enable_logging) = detect_metal_version(sdk)?;
        println!("cargo:warning=Detected Metal version: {}", metal_version);
        (metal_std, enable_logging)
    } else {
        println!("cargo:warning=Running in CI - using safe Metal 3.0 standard without logging");
        ("metal3.0".to_string(), false)
    };

    // Compile combined source to AIR
    let air = shader_out_dir.join("msm.air");
    let metal_std_arg = format!("-std={}", metal_std);
    let mut metal_args = vec![
        "-sdk",
        sdk,
        "metal",
        &metal_std_arg,
        "-c",
        combined.to_str().unwrap(),
        "-o",
        air.to_str().unwrap(),
    ];

    // Add logging flag only if enabled (which only happens when not in CI and Metal >= 3.2)
    if enable_logging {
        metal_args.insert(metal_args.len() - 3, "-fmetal-enable-logging");
        println!("cargo:warning=Enabling Metal logging");
    }

    let status = Command::new("xcrun")
        .args(&metal_args)
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

fn build_cpp_header(root_dir: PathBuf) {
    println!("cargo:rerun-if-changed=src/msm/metal_msm/shader/cuzk/Common.h");

    // macOS SDK root for clang
    let sdk_root = String::from_utf8(
        std::process::Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_owned();

    let bindings = bindgen::Builder::default()
        .header(format!(
            "{}/src/msm/metal_msm/shader/cuzk/Common.h",
            root_dir.to_str().unwrap()
        ))
        .clang_arg(format!("-isysroot{}", sdk_root))
        // .clang_arg("-x") // Objective-C dialect so #import works
        // .clang_arg("objective-c")
        .allowlist_type("Uniforms|Params")
        .allowlist_type("BufferIndices|Attributes|TextureIndices")
        .generate()
        .expect("bindgen failed");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_dir.join("common.rs")).unwrap();
}

fn detect_metal_version(sdk: &str) -> std::io::Result<(String, String, bool)> {
    // Use OS version to determine Metal version support
    let os_version = get_os_version();
    let (major, minor) = determine_metal_version_from_os(&os_version);

    // Determine Metal standard and whether to enable logging
    let (metal_std, enable_logging) = if major > 3 || (major == 3 && minor >= 2) {
        ("metal3.2".to_string(), true)
    } else if major == 3 && minor >= 1 {
        ("metal3.1".to_string(), false)
    } else if major == 3 {
        ("metal3.0".to_string(), false)
    } else if major == 2 && minor >= 4 {
        // For Metal 2.x, we need platform-specific prefixes
        let platform_prefix = if sdk == "iphoneos" { "ios" } else { "macos" };
        (format!("{}-metal2.4", platform_prefix), false)
    } else {
        // For Metal 2.x, we need platform-specific prefixes
        let platform_prefix = if sdk == "iphoneos" { "ios" } else { "macos" };
        (format!("{}-metal2.3", platform_prefix), false)
    };

    let version_str = format!("{}.{}", major, minor);
    println!(
        "cargo:warning=Detected OS version: {} -> Metal version: {}",
        os_version, version_str
    );
    Ok((version_str, metal_std, enable_logging))
}

fn get_os_version() -> String {
    // Get OS version using system APIs
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sw_vers").arg("-productVersion").output() {
            if output.status.success() {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }
    }

    #[cfg(target_os = "ios")]
    {
        // For iOS, we can assume modern Metal support
        return "16.0".to_string();
    }

    // Fallback
    "10.15".to_string()
}

fn determine_metal_version_from_os(os_version: &str) -> (u32, u32) {
    let parts: Vec<&str> = os_version.split('.').collect();
    if parts.len() >= 2 {
        if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
            // macOS to Metal version mapping
            #[cfg(target_os = "macos")]
            {
                if major >= 14 || (major == 13) {
                    return (3, 2); // macOS 13.0+ supports Metal 3.2
                } else if major >= 12 || (major == 11) {
                    return (3, 0); // macOS 11.0+ supports Metal 3.0
                } else if major >= 11 || (major == 10 && minor >= 15) {
                    return (2, 4); // macOS 10.15+ supports Metal 2.4
                }
            }

            // iOS to Metal version mapping
            #[cfg(target_os = "ios")]
            {
                if major >= 16 {
                    return (3, 2); // iOS 16+ supports Metal 3.2
                } else if major >= 14 {
                    return (3, 0); // iOS 14+ supports Metal 3.0
                } else if major >= 13 {
                    return (2, 4); // iOS 13+ supports Metal 2.4
                }
            }
        }
    }

    // Default fallback
    (2, 3)
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
