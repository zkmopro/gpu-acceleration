# GPU Acceleration App

This is an example for GPU accelerating exploration and a test for benchmarking results for different implemented msm functions, which might also focus on different parts of operations in SNARKs.

## Usage

Clone this repo then run either

```
# CONFIGURATION is either debug or release
CONFIGURATION=release cargo run --bin ios
CONFIGURATION=debug cargo run --bin ios
```

### running Benchmarks on IOS devices

1. Make sure you are in `./gpu-acceleartion-app/`.
2. Run `CONFIGURATION=release cargo run --bin ios` for release building.


The result would be like:

![benchmark result on simulator](image/simulator_benchmark_result.png)
