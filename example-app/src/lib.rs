// Here we're calling a macro exported with Uniffi. This macro will
// write some functions and bind them to FFI type. These
// functions will invoke the `get_circom_wtns_fn` generated below.
mopro_ffi::app!();

// --- Circom Example of setting up multiplier2 circuit ---
use circom_prover::witness::WitnessFn;

rust_witness::witness!(multiplier2);

mopro_ffi::set_circom_circuits! {
    ("multiplier2_final.zkey", WitnessFn::RustWitness(multiplier2_witness))
}

use mopro_msm::msm::metal_msm::{metal_variable_base_msm, test_utils::generate_random_bases_and_scalars};

#[uniffi::export]
fn metal_msm_benchmark(input_size: u32) -> () {
    let start = std::time::Instant::now();
    let (bases, scalars) = generate_random_bases_and_scalars(input_size as usize);
    println!("Generated bases and scalars in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let result = metal_variable_base_msm(&bases, &scalars).unwrap();
    println!("Metal MSM took {:?}", start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_msm() {
        metal_msm_benchmark(1024);
    }
}