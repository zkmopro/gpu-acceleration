# mopro msm gpu-acceleration

We are researching and implementing methods to accelerate multi-scalar multiplication (MSM) on IOS mobile device.

## mopro-msm

This is a of various implementations of MSM functions, which are then integrated in `mopro-core`.

### Run benchmark on the laptop
Currently we support these MSM algorithms on BN254:
- arkworks_pippenger
- bucket_wise_msm
- precompute_msm
- metal::msm (GPU)

Replace `MSM_ALGO` with one of the algorithms above to get the corresponding benchmarks.

Benchmarking for <u>single instance size</u>:
```sh
cargo test --release --package mopro-msm --lib -- msm::MSM_ALGO::tests::test_run_benchmark --exact --nocapture
```

Benchmarking for <u>multiple instance size</u>:
```sh
cargo test --release --package mopro-msm --lib -- msm::MSM_ALGO::tests::test_run_multi_benchmarks --exact --nocapture
```

## gpu-exploration-app

This is a benchmark app to compare the performance of different algorithms on iOS device.

You can run the following commands in the root directory of the project to compile the metal library for a given OS:
```sh
# for macOS
bash mopro-msm/src/msm/metal/compile_metal.sh

# for iphoneOS
bash mopro-msm/src/msm/metal/compile_metal_iphone.sh
```
