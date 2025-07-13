use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Re-export commonly used types so we can avoid deep paths inside the
// closures passed to Criterion.  Doing the imports at the top level keeps the
// benchmarking code focused on what is actually being measured rather than
// on verbose module paths.
use mopro_msm::msm::metal_msm::metal_msm::{
    metal_variable_base_msm, test_utils::generate_random_bases_and_scalars,
};

use ark_bn254::G1Projective as G;
use ark_ec::VariableBaseMSM;

/// Benchmark the end-to-end Metal MSM pipeline against the Arkworks reference
/// implementation for a handful of input sizes.
///
/// The logic (random dataset generation followed by the call to
/// `metal_variable_base_msm`) mirrors the `e2e` integration test located under
/// `src/msm/metal_msm/tests/cuzk/e2e.rs`, but here it is wrapped into a
/// Criterion benchmark so that it can be executed with `cargo bench` and
/// provide detailed performance statistics.
fn bench_e2e(c: &mut Criterion) {
    // Test a few logarithmic sizes to keep the benchmark execution time
    // reasonable while still showing how performance scales.  Feel free to
    // tweak these numbers locally if you want a more fine-grained view.
    const LOG_SIZES: &[u32] = &[10, 12, 16]; // 2^10 = 1024, 2^12 = 4096, 2^16 = 65536

    use std::time::Duration;
    let mut group = c.benchmark_group("metal_msm_e2e");
    // Shorten measurement time & sample size so the benchmark can finish
    // quickly in CI environments while still providing meaningful numbers.
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    for &log_n in LOG_SIZES {
        let n = 1usize << log_n;

        // Generate a fresh random dataset for this particular input size once
        // outside of the timed closure so that generation itself does not skew
        // the measurement results.
        let (bases, scalars) = generate_random_bases_and_scalars(n);

        group.throughput(Throughput::Elements(n as u64));

        // Benchmark the Metal implementation.
        group.bench_with_input(BenchmarkId::new("metal_msm", n), &n, |b, &_n| {
            b.iter(|| {
                // For correctness we still *unwrap* the result so that a panic
                // is triggered if the GPU implementation produces an error or
                // a mismatch in input length.
                let _res = metal_variable_base_msm(&bases, &scalars).unwrap();
            });
        });

        // Benchmark the Arkworks CPU reference as a baseline for comparison.
        group.bench_with_input(BenchmarkId::new("arkworks_msm", n), &n, |b, &_n| {
            b.iter(|| {
                let _res = G::msm(&bases, &scalars).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
