# mopro msm gpu-acceleration

We are researching the methods and implement the methods that can accelerate multi-scalar multiplication (MSM) on IOS mobile device.

## mopro-msm

This is the crate that possess various of implementation of MSM functions, which are integrated in `mopro-core`.

### Run benchmark on the laptop
Currently we support these MSM algorithms on BN254
- arkworks_pippenger
- bucket_wise_msm
- precompute_msm
- metal::msm (GPU)

Replace `MSM_ALGO` with the algorithm name below to get the benchmarks

Benchmarking for <u>single instance size</u>
```sh
cargo test --release --package mopro-msm --lib -- msm::MSM_ALGO::tests::test_run_benchmark --exact --nocapture
```

Benchmarking for <u>multiple instance size</u>
```sh
cargo test --release --package mopro-msm --lib -- msm::MSM_ALGO::tests::test_run_multi_benchmarks --exact --nocapture
```

## gpu-exploration-app

This is a benchmark app to compare the performance of different algorithm on IOS device.

Run this command on the project root directory to compile metal library for OS
```sh
# for macOS
bash mopro-msm/src/msm/metal/compile_metal.sh

# for iphoneOS
bash mopro-msm/src/msm/metal/compile_metal_iphone.sh
```
