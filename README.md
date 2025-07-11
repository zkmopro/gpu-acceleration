# Metal MSM

Metal-MSM v2 executes MSM on [BN254](https://hackmd.io/@jpw/bn254) curve on Apple GPUs using Metal Shading Language (MSL). Unlike v1, which naively split the work into smaller tasks, v2 takes [Tal and Koh’s WebGPU MSM](https://github.com/z-prize/2023-entries/tree/main/prize-2-msm-wasm/webgpu-only/tal-derei-koh-wei-jie) in ZPrize2023 and the cuZK [[LWY+23](https://eprint.iacr.org/2022/1321)] approach as reference.

By adopting sparse matrices, it improves the Pippenger algorithm [Pip76](https://dl.acm.org/doi/10.1109/SFCS.1976.21) with a more memory-efficient storage format and uses well-studied sparse matrix algorithms, such as sparse matrix–vector multiplication and sparse matrix transposition, in both the preprocessing phase (e.g., radix sort via sparse matrix transpose) and the bucket-accumulation phase to achieve high parallelism.

We took the WebGPU MSM reference and tuned it for all scales by auto-adjusting workgroup sizes for each cuZK shaders with SIMD width and the amount of GPU cores, squeezing out better GPU utilization. Plus, with dynamic window sizes, we speed up small and medium inputs (2^14 – 2^18) by eliminating unused sparse-matrix columns.

One thing to highlight is that our implementation runs most computations on the GPU, but it’s still slower than the CPU-only solution like [Arkworks](https://github.com/arkworks-rs). However, because we target client-side devices with limited resources, applying a hybrid approach, leveraging both CPU and GPU for MSM tasks and combining the results at the end, can yield an implementation slightly faster than a pure-CPU one. Check the write-up below for estimated speedups with this hybrid method.

## How to use

Metal MSM v2 works with `arkworks v0.4.x`; just include the crate in your `Cargo.toml`.
```toml
mopro-msm = { git = "https://github.com/zkmopro/gpu-acceleration.git", tag = "v0.2.0" }
```

Next, invoke MSM within your Rust code.
```rust
use mopro_msm::msm::metal_msm::{
    metal_variable_base_msm,
    test_utils::generate_random_bases_and_scalars,    // optional
};

fn main() {
    let input_size = 1 << 16;
    let (bases, scalars) = generate_random_bases_and_scalars(input_size);
    let msm_result = metal_variable_base_msm(&bases, &scalars);

    println!("Result: {:?}", msm_result);
}
```

Because it’s compatible with Arkworks, you can seamlessly swap between Metal MSM and the Arkworks MSM implementation.
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr as ScalarField, G1Projective as G};
    use ark_ec::{CurveGroup, VariableBaseMSM};
    use ark_std::{UniformRand, test_rng};

    #[test]
    fn test_msm() {
        let input_size = 1 << 10;

        // Generate random EC points and scalars with Arkworks
        let mut rng = test_rng();
        let bases = (0..input_size)
            .map(|_| G::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();
        let scalars = (0..input_size)
            .map(|_| ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();

        let metal_msm_result = metal_variable_base_msm(&bases, &scalars).unwrap();
        let arkworks_msm_result = G::msm(&bases, &scalars).unwrap();

        assert_eq!(metal_msm_result, arkworks_msm_result);    // the result is the same
    }
}
```

## Benchmark

Benchmarking on BN254 curve ran on a MacBook Air with M3 chips, with test case setup time excluded.

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center">Scheme</th>
      <th colspan="7" style="text-align:center">Input Size (ms)</th>
    </tr>
    <tr>
      <th style="text-align:center">2<sup>12</sup></th>
      <th style="text-align:center">2<sup>14</sup></th>
      <th style="text-align:center">2<sup>16</sup></th>
      <th style="text-align:center">2<sup>18</sup></th>
      <th style="text-align:center">2<sup>20</sup></th>
      <th style="text-align:center">2<sup>22</sup></th>
      <th style="text-align:center">2<sup>24</sup></th>
    </tr>
  </thead>
  <tbody style="text-align:center">
    <tr>
      <th style="text-align:center"><a href="https://github.com/arkworks-rs">Arkworks v0.4.x</a><br>(CPU, Baseline)</br></th>
      <td>6</td>
      <td>19</td>
      <td>69</td>
      <td>245</td>
      <td>942</td>
      <td>3,319</td>
      <td>14,061</td>
    </tr>
    <tr>
      <th style="text-align:center"><a href="https://github.com/zkmopro/gpu-acceleration/tree/v0.1.0">Metal MSM v0.1.0</a><br>(GPU)</br></th>
      <td>143<br>(-23.8x)</br></td>
      <td>273<br>(-14.4x)</br></td>
      <td>1,730<br>(-25.1x)</br></td>
      <td>10,277<br>(-41.9x)</br></td>
      <td>41,019<br>(-43.5x)</br></td>
      <td>555,877<br>(-167.5x)</br></td>
      <td>N/A</td>
    </tr>
    <tr>
      <th style="text-align:center"><a href="https://github.com/zkmopro/gpu-acceleration/tree/v0.2.0">Metal MSM v0.2.0</a><br>(GPU)</br></th>
      <td>134<br>(-22.3x)</br></td>
      <td>124<br>(-6.5x)</br></td>
      <td>253<br>(-3.7x)</br></td>
      <td>678<br>(-2.8x)</br></td>
      <td>1,702<br>(-1.8x)</br></td>
      <td>5,390<br>(-1.6x)</br></td>
      <td>22,241<br>(-1.6x)</br></td>
    </tr>
    <tr>
      <th style="text-align:center"><a href="https://github.com/ICME-Lab/msm-webgpu">ICME WebGPU MSM</a><br>(GPU)</br></th>
      <td>N/A</td>
      <td>N/A</td>
      <td>2,719<br>(-39.4x)</br></td>
      <td>5,418<br>(-22.1x)</br></td>
      <td>17,475<br>(-18.6x)</br></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th style="text-align:center"><a href="https://github.com/moven0831/icicle/tree/bn254-metal-benchmark">ICICLE-Metal v3.8.0</a><br>(GPU)</br></th>
      <td>59<br>(-9.8x)</br></td>
      <td>54<br>(-2.8x)</br></td>
      <td>89<br>(-1.3x)</br></td>
      <td>149<br>(+1.6x)</br></td>
      <td>421<br>(+2.2x)</br></td>
      <td>1,288<br>(+2.6x)</br></td>
      <td>4,945<br>(+2.8x)</br></td>
    </tr>
  </tbody>
</table>

> side note:
> - for ICME WebGPU MSM, input size 2^12 causes M3 chip machines to crash; any sizes not listed on the project’s GitHub page are shown as "N/A"
> - for Metal MSM v0.1.0, the 2^24 benchmark was abandoned because it exceeded practical runtime

## Profiling summary (v1 vs v2)

Environment: M1 Pro, macOS 15.2, curve `ark_bn254`, dataset 2^20 unless stated. Medians of 5 runs.

### v2 → v1

| metric | v1[^1] | v2[^2] | gain |
|---|---|---|---|
| end-to-end latency | 10.3 s | **0.42 s** | **×24** |
| GPU occupancy | 32 % | 76 % | +44 pp |
| CPU share | 19 % | **<3 %** | –16 pp |
| peak VRAM | 1.6 GB | **220 MB** | –7.3× |

Key changes:

* single sparse-matrix kernel eliminates most launches and memory thrash  
* CSR buckets keep data on-device → near-zero host↔GPU traffic  
* on-GPU radix sort makes preprocessing parallel

## Future

### Technical Improvements
- **Modern Dependencies**: Update to `objc2` and `objc2-metal` ([objc2](https://github.com/madsmtm/objc2))
- **Metal 4**: Adopt latest [Metal 4](https://developer.apple.com/metal/whats-new/) features
- **Refactor with SIMD in mind**:
  - Instruction-level parallelism using vector types for faster FMA within SIMD groups
  - Memory coalescing to increase locality (e.g., structure of array instead of array of structure)
  - Optimized input reading patterns (e.g. `[X_i || Y_i]_0^{n-1}` instead of separate arrays)
  - Latency hiding and occupancy fine-tuning
  - Minimize thread divergence

### Algorithm & Integration
- **CPU-GPU Hybrid**: Research interleaving with CPU MSM crate and update to `arkworks 0.5`
- **Advanced Algorithms**:
  - Elastic MSM [[ZHY+24](https://eprint.iacr.org/2024/057.pdf)] implementation
  - Faster modular reduction with LogJump ([article by Wei Jie](https://kohweijie.com/articles/25/logjumps.html), [Barret-Montgomery](https://hackmd.io/@Ingonyama/Barret-Montgomery))

### Platform Expansion
- **Cross-platform**: WGSL support with native execution environment
- **Crypto Math Library**: Maintain a Metal/WebGPU crypto math library

## Community

-   X account: <a href="https://twitter.com/zkmopro"><img src="https://img.shields.io/twitter/follow/zkmopro?style=flat-square&logo=x&label=zkmopro"></a>
-   Telegram group: <a href="https://t.me/zkmopro"><img src="https://img.shields.io/badge/telegram-@zkmopro-blue.svg?style=flat-square&logo=telegram"></a>

## Acknowledgements

This work was initially sponsored by a joint grant from [PSE](https://pse.dev/) and [0xPARC](https://0xparc.org/). It is currently incubated by PSE.

[^1]: https://hackmd.io/@yaroslav-ya/rJkpqc_Nke
[^2]: https://hackmd.io/@yaroslav-ya/HyFA7XAQll