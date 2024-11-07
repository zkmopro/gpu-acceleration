use std::{env, path::Path, process::Command};

fn main() {
    // Compile and link shaders
    compile_shaders();
}

fn compile_shaders() {
    let shader_dir = "src/msm/metal/shader/";
    let out_dir = env::var("OUT_DIR").unwrap();

    // List your Metal shaders here.
    let shaders = vec!["all.metal"];

    let shaders_to_check = vec!["all.metal", "msm.h.metal"];

    let mut air_files = vec![];

    // Step 1: Compile every shader to AIR format
    for shader in &shaders {
        let shader_path = Path::new(shader_dir).join(shader);
        let air_output = Path::new(&out_dir).join(format!("{}.air", shader));

        let mut args = vec![
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-frecord-sources",
            shader_path.to_str().unwrap(),
            "-o",
            air_output.to_str().unwrap(),
        ];

        if std::env::var("PROFILE")
            .map(|profile| profile == "release")
            .unwrap_or(false)
        {
            args.push("-frecord-sources");
        }

        // Compile shader into .air files
        let status = Command::new("xcrun")
            .args(&args)
            .status()
            .expect("Shader compilation failed");

        if !status.success() {
            panic!("Shader compilation failed for {}", shader);
        }

        air_files.push(air_output);
    }

    // Step 2: Link all the .air files into a Metallib archive
    let metallib_output = Path::new(&out_dir).join("msm.metallib");

    let mut metallib_args = vec![
        "-sdk",
        "macosx",
        "metal",
        "-o",
        metallib_output.to_str().unwrap(),
    ];

    if std::env::var("PROFILE")
        .map(|profile| profile == "release")
        .unwrap_or(false)
    {
        metallib_args.push("-frecord-sources");
    }

    for air_file in &air_files {
        metallib_args.push(air_file.to_str().unwrap());
    }

    let status = Command::new("xcrun")
        .args(&metallib_args)
        .status()
        .expect("Failed to link shaders into metallib");

    if !status.success() {
        panic!("Failed to link shaders into metallib");
    }

    let symbols_args = vec![
        "metal-dsymutil",
        "-flat",
        "-remove-source",
        metallib_output.to_str().unwrap(),
    ];

    let status = Command::new("xcrun")
        .args(&symbols_args)
        .status()
        .expect("Failed to extract symbols");

    if !status.success() {
        panic!("Failed to extract symbols");
    }

    // Inform cargo to watch all shader files for changes
    for shader in &shaders_to_check {
        let shader_path = Path::new(shader_dir).join(shader);
        println!("cargo:rerun-if-changed={}", shader_path.to_str().unwrap());
    }
}
