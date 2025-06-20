# Metal MSM

Metal-MSM v2 executes MSM on Apple's GPU using Metal Shading Language (MSL). Unlike v1, which naively split the work into smaller tasks, v2 takes [Tal and Koh’s WebGPU MSM](https://github.com/z-prize/2023-entries/tree/main/prize-2-msm-wasm/webgpu-only/tal-derei-koh-wei-jie) in ZPrize2023 and the [LWY+23](https://eprint.iacr.org/2022/1321) (cuZK) approach as reference.

By adopting sparse matrices, it improves the Pippenger algorithm [Pip76](https://dl.acm.org/doi/10.1109/SFCS.1976.21) with a more memory-efficient storage format and uses well-studied sparse matrix algorithms, such as sparse matrix–vector multiplication and sparse matrix transposition, in both the preprocessing phase (e.g., radix sort via sparse matrix transpose) and the bucket-accumulation phase to achieve high parallelism.

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
// TODO

## Community

-   X account: <a href="https://twitter.com/zkmopro"><img src="https://img.shields.io/twitter/follow/zkmopro?style=flat-square&logo=x&label=zkmopro"></a>
-   Telegram group: <a href="https://t.me/zkmopro"><img src="https://img.shields.io/badge/telegram-@zkmopro-blue.svg?style=flat-square&logo=telegram"></a>

## Acknowledgements

This work was initially sponsored by a joint grant from [PSE](https://pse.dev/) and [0xPARC](https://0xparc.org/). It is currently incubated by PSE.
