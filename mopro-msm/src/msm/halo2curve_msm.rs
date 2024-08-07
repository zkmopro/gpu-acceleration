// use ark_bn254::{Fr as ScalarField, G1Affine as GAffine, G1Projective as G};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::BigInt;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::time::{Duration, Instant};

use ff::Field;
use group::prime::PrimeCurveAffine;
use halo2curves::bn256::{Fr as Scalar, G1Affine as Point};
use halo2curves::msm::{best_multiexp, multiexp_serial};
use halo2curves::serde::SerdeObject;
use rayon::current_thread_index;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::middleware::gpu_explorations::utils::{benchmark::BenchmarkResult, preprocess};

pub fn benchmark_msm<I>(
    instances: I,
    iterations: u32,
) -> Result<Vec<Duration>, preprocess::HarnessError>
where
    I: Iterator<Item = preprocess::Instance>,
{
    let mut instance_durations: Vec<Duration> = Vec::new();

    for instance in instances {
        // parse the instance to halo2curve compatible format
        let points: &Vec<Point> = &instance
            .0
            .iter()
            .map(|p| {
                let mut bytes = Vec::new();
                p.serialize_uncompressed(&mut bytes).unwrap();
                Point::from_raw_bytes_unchecked(&bytes)
            })
            .collect();
        let scalars: &Vec<Scalar> = &instance.1.iter().map(|s| Scalar::from(s.0)).collect();

        let mut instance_total_duration = Duration::ZERO;
        for _i in 0..iterations {
            /* For single-core range */
            // let mut acc = Point::identity().into();
            // let start = Instant::now();
            // let _result = multiexp_serial(&scalars[..], &points[..], &mut acc);

            /* For multi-core range */
            let start = Instant::now();
            let _result = best_multiexp(&scalars[..], &points[..]);

            instance_total_duration += start.elapsed();
        }
        let instance_avg_duration = instance_total_duration / iterations;

        println!(
            "Average time to execute MSM with {} points and {} scalars in {} iterations is: {:?}",
            points.len(),
            scalars.len(),
            iterations,
            instance_avg_duration,
        );
        instance_durations.push(instance_avg_duration);
    }
    Ok(instance_durations)
}

pub fn run_benchmark(
    instance_size: u32,
    num_instance: u32,
    utils_dir: &str,
) -> Result<BenchmarkResult, preprocess::HarnessError> {
    // Check if the vectors have been generated
    match preprocess::FileInputIterator::open(&utils_dir) {
        Ok(_) => {
            println!("Vectors already generated");
        }
        Err(_) => {
            preprocess::gen_vectors(instance_size, num_instance, &utils_dir);
        }
    }

    let benchmark_data = preprocess::FileInputIterator::open(&utils_dir).unwrap();
    let instance_durations = benchmark_msm(benchmark_data, 1).unwrap();
    // in milliseconds
    let avg_processing_time: f64 = instance_durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / instance_durations.len() as f64;

    println!("Done running benchmark.");
    Ok(BenchmarkResult {
        instance_size: instance_size,
        num_instance: num_instance,
        avg_processing_time: avg_processing_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_serialize::Write;
    use std::fs::File;

    const INSTANCE_SIZE: u32 = 16;
    const NUM_INSTANCE: u32 = 10;
    const UTILSPATH: &str = "src/msm/utils/vectors";
    const BENCHMARKSPATH: &str = "benchmark_results";

    #[test]
    fn test_benchmark_msm() {
        let dir = format!(
            "{}/{}/{}x{}",
            preprocess::get_root_path(),
            UTILSPATH,
            INSTANCE_SIZE,
            NUM_INSTANCE
        );

        // Check if the vectors have been generated
        match preprocess::FileInputIterator::open(&dir) {
            Ok(_) => {
                println!("Vectors already generated");
            }
            Err(_) => {
                preprocess::gen_vectors(INSTANCE_SIZE, NUM_INSTANCE, &dir);
            }
        }

        let benchmark_data = preprocess::FileInputIterator::open(&dir).unwrap();
        let result = benchmark_msm(benchmark_data, 1);
        println!("Done running benchmark: {:?}", result);
    }

    #[test]
    fn test_run_benchmark() {
        let utils_path = format!(
            "{}/{}/{}x{}",
            preprocess::get_root_path(),
            &UTILSPATH,
            INSTANCE_SIZE,
            NUM_INSTANCE
        );
        let result = run_benchmark(INSTANCE_SIZE, NUM_INSTANCE, &utils_path).unwrap();
        println!("Benchmark result: {:#?}", result);
    }

    #[test]
    fn test_run_multi_benchmarks() {
        let output_path = format!(
            "{}/{}/{}_benchmark.txt",
            preprocess::get_root_path(),
            &BENCHMARKSPATH,
            "halo2curve_multicore_msm"
        );
        let mut output_file = File::create(output_path).expect("output file creation failed");
        writeln!(output_file, "msm_size,num_msm,avg_processing_time(ms)");

        let instance_size = vec![8, 12, 16, 18, 20, 22];
        let num_instance = vec![10];
        for size in &instance_size {
            for num in &num_instance {
                let utils_path = format!(
                    "{}/{}/{}x{}",
                    preprocess::get_root_path(),
                    &UTILSPATH,
                    *size,
                    *num
                );
                let result = run_benchmark(*size, *num, &utils_path).unwrap();
                println!("{}x{} result: {:#?}", *size, *num, result);
                writeln!(
                    output_file,
                    "{},{},{}",
                    result.instance_size, result.num_instance, result.avg_processing_time
                );
            }
        }
    }
}
