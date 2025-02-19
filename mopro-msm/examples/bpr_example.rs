use ark_bn254::{Fr as ScalarField, G1Projective as G};
use ark_ec::Group;
use std::io::stdin;

// These functions should be defined and exposed in your library.
use mopro_msm::msm::metal_msm::tests::cuzk::bpr::{cpu_bpr_stage_1, gpu_bpr_stage_1};

fn main() {
    // Pause and wait for debugger attach (e.g., Xcode GPU frame capture)
    println!("Waiting for debugger attach... Press Enter to continue");
    let mut pause = String::new();
    let _ = stdin().read_line(&mut pause);

    // Create test buckets; using (2 * generator) as sample data.
    let generator = G::generator();
    let c: u32 = 3;
    let bucket_count = (1 << c) - 1; // For example, 7 buckets.
    let buckets: Vec<G> = (0..bucket_count)
        .map(|_| generator * ScalarField::from(2))
        .collect();

    // Execute the CPU bucket reduction as reference.
    let (cpu_s, cpu_m) = cpu_bpr_stage_1(&buckets);
    println!("CPU Results:");
    println!("s: {:?}", cpu_s);
    println!("m: {:?}", cpu_m);

    // Execute the GPU bucket reduction.
    let (gpu_s, gpu_m) = gpu_bpr_stage_1(&buckets);
    println!("GPU Results:");
    println!("s: {:?}", gpu_s);
    println!("m: {:?}", gpu_m);

    // Optionally, add result comparisons or additional logic here.
}
