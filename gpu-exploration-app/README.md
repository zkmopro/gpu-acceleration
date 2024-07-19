# GPU Exploration App

This the a example for GPU exploration and a test for benchmarking results.

## Documentation

See [here](https://github.com/zkmopro/mopro/blob/main/mopro-core/gpu_explorations/README.md)

## Usage

Clone this repo then run either
```
# CONFIGURATION is either debug or release
CONFIGURATION=release cargo run --bin ios
CONFIGURATION=debug cargo run --bin ios
```

### running Benchmarks on IOS devices
 
1. modify the deafult feature in the folder `mopro-ffi` in your `MOPRO_ROOT`
    * original: ~~`default=[]`~~ => `default=["gpu-benchmarks"]`
2. open the `ExampleApp.xcworkspace` in the `ios/` with Xcode on your Mac/ device
3. modify `mopro-config.toml` in the root directory:
    * ios_device_type = "simulator" to run on the simulator
    * ios_device_type = "device" to run on a iphone or a ipad
4. Build the project using `cmd + R`, your device will redirect to the app
5. Then, select the algorithms you want to run on your device, click `Generate Benchmarks`.

The result would be like:

![benchmark result on simulator](image/simulator_benchmark_result.png)
