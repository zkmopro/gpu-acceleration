// Quick MSM benchmark harness: runs metal_variable_base_msm at several sizes,
// reports min/median of N runs plus peak GPU memory (MTLDevice
// currentAllocatedSize, sampled by a background thread). Usage:
//   cargo run --release -p mopro-msm --example quick_bench [log_sizes...]
use ark_bn254::G1Projective as G;
use ark_ec::VariableBaseMSM;
use mopro_msm::msm::metal_msm::metal_msm::{metal_variable_base_msm, test_utils};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let log_sizes = if args.is_empty() {
        vec![12, 14, 16, 18, 20]
    } else {
        args
    };
    const RUNS: usize = 5;

    // Background sampler: tracks peak GPU-allocated bytes for this process.
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let sampler = {
        let stop = stop.clone();
        let peak = peak.clone();
        std::thread::spawn(move || {
            let device = metal::Device::system_default().expect("no Metal device");
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(device.current_allocated_size(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        })
    };

    for log_size in log_sizes {
        let size = 1usize << log_size;
        let (bases, scalars) = test_utils::generate_random_bases_and_scalars(size);

        // correctness check once per size
        let metal_res = metal_variable_base_msm(&bases, &scalars).unwrap();
        let ark_res = G::msm(&bases, &scalars).unwrap();
        assert_eq!(metal_res, ark_res, "MISMATCH at 2^{}", log_size);

        // reset peak after warmup so we capture this size's steady-state peak
        peak.store(0, Ordering::Relaxed);

        let mut times_ms: Vec<f64> = (0..RUNS)
            .map(|_| {
                let t = Instant::now();
                let _ = metal_variable_base_msm(&bases, &scalars).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let peak_mb = peak.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);

        let t = Instant::now();
        let _ = G::msm(&bases, &scalars).unwrap();
        let ark_ms = t.elapsed().as_secs_f64() * 1e3;

        println!(
            "2^{:<2} metal min {:>9.2} ms  median {:>9.2} ms | peak GPU mem {:>8.1} MB | arkworks {:>9.2} ms",
            log_size,
            times_ms[0],
            times_ms[RUNS / 2],
            peak_mb,
            ark_ms
        );
    }

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
}
