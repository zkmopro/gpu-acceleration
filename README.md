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

## Benchmarking (how-to)

Metal-MSM v0.2.0 ships two Criterion benches:

| Command | What it measures |
|---------|------------------|
| `cargo bench -p mopro-msm --bench shaders -- --sample-size 100 --warm-up-time 3 --measurement-time 10` | Pure GPU time of each individual Metal shader (decompose, transpose, SMVP, PBPR …) across several input sizes.  Host work is minimal so numbers reflect kernel performance. |
| `cargo bench -p mopro-msm --bench e2e -- --sample-size 100 --warm-up-time 3 --measurement-time 10` | Full `metal_variable_base_msm` pipeline – includes host-side preprocessing and postprocessing.  Good for an overall sanity check, less ideal for micro-optimisation. |

Flags after the double dash (`--`) are passed straight to Criterion, letting you tweak sample count or timing budget.  Typical knobs:

* `--sample-size <n>` – how many timed iterations to record (default 50).
* `--warm-up-time <seconds>` – CPU/GPU warm-up before sampling starts.
* `--measurement-time <seconds>` – total timed run per benchmark case.

> The *shaders* bench uses the same logarithmic dataset sizes as the `e2e` bench (`2^10`, `2^12`, `2^16`).  Modify `LOG_SIZES` inside `benches/shaders.rs` if you need different scales.

When tuning kernels focus on the **shaders** bench first; improvements there should propagate to the end-to-end numbers automatically.

## Optimization notes (2026-08)

The following optimizations were applied on top of v0.2.0 (measured on an
M-series MacBook, medians of 5 runs; cumulative speedup vs v0.2.0 on the same
machine: **~2.9x at 2^12, ~4.4x at 2^16, ~5x at 2^20-2^22**, making the GPU
~2.3-3.2x faster than Arkworks CPU MSM at 2^20-2^22):

| Change | Effect |
|---|---|
| Single-sync GPU pipeline: all stages encoded up-front, buffers stay on device, one host wait; pipeline-state cache across calls | ~12-14% e2e |
| 32-bit limbs (8x32 instead of 16x16) with `ulong` fused multiply-add CIOS | ~1.4-1.7x e2e |
| Mixed addition (`madd-2007-bl` with edge cases) in SMVP bucket accumulation | ~16% e2e |
| Parallel sparse-matrix transpose (histogram → prefix sum → atomic scatter) instead of ~20 serial threads | up to 2.1x at 2^20 |
| Zero-copy Montgomery: pass arkworks' internal Montgomery representation (same R = 2^256) straight to the GPU; deletes host-side reductions, the GPU to-Montgomery conversion and the Barrett reduction path entirely | less host work + memory |
| Window-size and workgroup-scale retuning (window 13 down to 2^12; smaller workgroups) | ~4-13% e2e |
| Single SMVP dispatch for all subtasks (drop WebGPU-era 4-subtask chunking) | ~5-11% e2e |
| XYZZ (extended Jacobian) bucket accumulation: mixed add 8M+2S vs 7M+4S and ~5 vs ~13 field add/subs; general add 12M+2S vs 11M+5S | ~7-11% e2e |

Findings on the previously-listed future items:

- **LogJumps** ([Wei Jie's article](https://kohweijie.com/articles/25/logjumps.html)):
  implemented, benchmarked and **removed**: for BN254, rho = 2^-32 mod p
  ≈ 0.89p, so the pre-reduction value reaches ~2.9p and needs a second
  conditional subtraction, which cancels the n-1 saved multiplications.
  Measured ~8-10% slower than CIOS end-to-end on Apple GPUs (the
  implementation lives in git history if rho-friendly parameters ever
  make it worth revisiting).
- **Elastic MSM** [[ZHY+24](https://eprint.iacr.org/2024/057.pdf)]: a
  fixed-base preprocessing technique (stores w-1 shifted copies of each base,
  runs the same pipeline on a (wn, ks)-instance with k windows). It only pays
  off when the same bases are reused across many MSMs (e.g. a proving key);
  for the one-shot `metal_variable_base_msm` API the precompute cost exceeds
  the savings. Worth adding later as an opt-in fixed-base API; expected
  benefit is largest at 2^12-2^18.
- **Metal 4**: unreachable from the deprecated `metal-rs` crate; requires
  migrating to `objc2-metal` >= 0.3. Its wins (argument tables, command
  allocators, untracked resources) mostly reduce CPU encoding overhead, which
  is now a small fraction of runtime after the single-sync pipeline rework —
  worthwhile mainly as part of the general `objc2` migration.

## Future

### Technical Improvements
- **Modern Dependencies**: Update to `objc2` and `objc2-metal` ([objc2](https://github.com/madsmtm/objc2)), then adopt [Metal 4](https://developer.apple.com/metal/whats-new/) command encoding
- **SMVP load balancing**: bucket accumulation is still one-thread-per-bucket; sorting into fixed-size segments with a segmented reduction (ICICLE/Yrrid-style) would remove the slowest-bucket tail and make every window size safe
- **Batched-affine accumulation**: amortized ~6-7 muls per bucket addition via Montgomery batch inversion, plus smaller per-thread state

### Algorithm & Integration
- **CPU-GPU Hybrid**: Research interleaving with CPU MSM crate and update to `arkworks 0.5`
- **Elastic MSM** [[ZHY+24](https://eprint.iacr.org/2024/057.pdf)] as an opt-in fixed-base API (see notes above)

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
