use metal::*;
/// Window size optimizer for MSM computation
///
/// This module implements the cost function described in Section 3.2 to determine
/// the optimal window size for parallel MSM algorithms.
use rayon::prelude::*;
// Embed the precompiled Metal library
include!(concat!(env!("OUT_DIR"), "/built_shaders.rs"));

/// Parameters for MSM computation
#[derive(Debug, Clone)]
pub struct MsmParameters {
    /// size of MSM (n)
    pub n: u64,
    /// Number of scalar bits (λ)
    pub scalar_bits_size: u32,
    /// Number of parallel threads (t)
    pub t: usize,
    /// SIMD width
    pub simd_width: usize,
}

/// Cost calculation result
#[derive(Debug, Clone)]
pub struct CostResult {
    pub window_size: u32,
    pub cost: f64,
}

/// Calculate the computational cost for a given window size
/// Source: https://eprint.iacr.org/2022/1321.pdf Section 4.1
///
/// Formula: ( (λ/s).ceil() * (n + 2^(s+1)) / t + s + log(t) ) PADDs + ( λ + s ) PDBLs
fn calculate_cost(params: &MsmParameters, window_size: u32) -> f64 {
    const PADD_COST: f64 = 1.0;
    const PDBL_COST: f64 = 1.0;

    let n = params.n as f64;
    let lambda = params.scalar_bits_size as f64;
    let t = params.t as f64;
    let s = window_size as f64;

    let padd_cost = (lambda / s).ceil() * (n + (1 << (window_size + 1)) as f64) / t + s + t.log2();
    let pdbl_cost = lambda + s;

    padd_cost * PADD_COST + pdbl_cost * PDBL_COST
}

/// Find the optimal window size for given MSM parameters
///
/// This function evaluates all feasible window sizes in parallel and returns the one
/// with minimum computational cost.
pub fn find_optimal_window_size(params: &MsmParameters) -> CostResult {
    let max_window_size = 28;

    // Evaluate window sizes from 1 to max_window_size in parallel
    (1..=max_window_size)
        .into_par_iter()
        .map(|window_size| {
            let cost = calculate_cost(params, window_size);
            CostResult { window_size, cost }
        })
        .min_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(CostResult {
            window_size: 1,
            cost: f64::INFINITY,
        })
}

/// Find the optimal window size for given MSM parameters
///
/// This function evaluates all feasible window sizes and returns the one
/// with minimum computational cost.
pub fn find_optimal_window_size_serial(params: &MsmParameters) -> CostResult {
    let max_window_size = 30;
    let mut best_result = CostResult {
        window_size: 1,
        cost: f64::INFINITY,
    };

    // Evaluate window sizes from 1 to max_window_size
    for window_size in 1..=max_window_size {
        let cost = calculate_cost(params, window_size);

        if cost < best_result.cost {
            best_result = CostResult { window_size, cost };
        }
    }

    best_result
}

/// Fetch GPU core count from existing Metal device (more efficient)
pub fn fetch_gpu_core_count_and_simd_width_from_device(device: &Device) -> (usize, usize) {
    // fetch the number of parallel cores from the SMVP kernel as example
    let library = device.new_library_with_data(MSM_METALLIB).unwrap();
    let kernel = library.get_function("smvp", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    let simd_width = pipeline.thread_execution_width() as usize; // SIMD width per core
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize; // Max threads per threadgroup
    let estimated_cores = max_threads / simd_width; // ≈ parallel cores

    (estimated_cores, simd_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as ScalarField;
    use ark_ff::PrimeField;
    use std::time::Instant;

    #[test]
    fn test_optimal_window_size_2_16() {
        let msm_size = 1 << 16;
        let start_time = Instant::now();
        let device = Device::system_default().expect("No device found");
        println!("Time taken to fetch device: {:?}", start_time.elapsed());

        let start_time = Instant::now();
        let (gpu_core_count, simd_width) = fetch_gpu_core_count_and_simd_width_from_device(&device);
        println!(
            "Time taken to fetch GPU core count: {:?}",
            start_time.elapsed()
        );

        let params = MsmParameters {
            n: msm_size,
            scalar_bits_size: ScalarField::MODULUS_BIT_SIZE as u32,
            t: gpu_core_count as usize,
            simd_width,
        };

        let start_time = Instant::now();
        let result = find_optimal_window_size(&params);
        println!(
            "Time taken to find optimal window size: {:?}",
            start_time.elapsed()
        );

        println!("Test for n = 2^16:");
        println!("  Input size: {}", params.n);
        println!("  Scalar bits: {}", params.scalar_bits_size);
        println!("  Threads: {}", params.t);
        println!("  Optimal window size: {}", result.window_size);
        println!("  Cost: {:.2}", result.cost);
        println!("  GPU core count: {}", gpu_core_count);

        // Basic sanity checks
        assert!(result.window_size >= 1);
        assert!(result.window_size <= 30);
        assert!(result.cost > 0.0);
        assert!(result.cost < f64::INFINITY);
    }

    #[test]
    #[ignore]
    fn test_optimal_window_size_range_2_10_to_2_26() {
        let device = Device::system_default().expect("No device found");
        let (gpu_core_count, simd_width) = fetch_gpu_core_count_and_simd_width_from_device(&device);
        let scalar_bits_size = ScalarField::MODULUS_BIT_SIZE as u32;

        println!("\n{:=<60}", "");
        println!("{:^60}", "MSM Window Size Optimization Results");
        println!("{:=<60}", "");
        println!(
            "{:<8} {:>12} {:>12} {:>12}",
            "Power", "Input Size", "Window Size", "Cost"
        );
        println!("{:-<60}", "");

        for power in 10..=26 {
            let params = MsmParameters {
                n: 1 << power, // 2^power
                scalar_bits_size,
                t: gpu_core_count,
                simd_width,
            };

            let result = find_optimal_window_size(&params);

            println!(
                "{:<8} {:>12} {:>12} {:>12}",
                format!("2^{}", power),
                format!("{}", params.n),
                result.window_size,
                result.cost
            );
        }

        println!("{:=<60}", "");
        println!(
            "Parameters: {} scalar bits, {} GPU cores",
            scalar_bits_size, gpu_core_count
        );
        println!("{:=<60}", "");
    }
}
