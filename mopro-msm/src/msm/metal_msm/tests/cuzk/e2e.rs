use crate::msm::metal_msm::metal_msm::metal_variable_base_msm;
use ark_bn254::{Fr as ScalarField, G1Projective as G};
use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::UniformRand;
use ark_std::test_rng;
use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_metal_msm_pipeline() {
        let log_input_size = 16;
        let input_size = 1 << log_input_size;

        let num_threads = rayon::current_num_threads();
        let thread_chunk_size = (input_size + num_threads - 1) / num_threads;
        println!(
            "Generating {} elements using {} threads",
            input_size, num_threads
        );
        let start = std::time::Instant::now();

        // Generate bases and scalars in parallel
        let (bases, scalars): (Vec<_>, Vec<_>) = (0..num_threads)
            .into_par_iter()
            .flat_map(|thread_id| {
                let mut rng = test_rng();

                let start_idx = thread_id * thread_chunk_size;
                let end_idx = std::cmp::min(start_idx + thread_chunk_size, input_size);
                let current_thread_size = end_idx - start_idx;

                (0..current_thread_size)
                    .map(|_| {
                        let base = G::rand(&mut rng).into_affine();
                        let scalar = ScalarField::rand(&mut rng);
                        (base, scalar)
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        println!("Generated {} elements in {:?}", input_size, start.elapsed());

        println!("running metal_variable_base_msm");
        let start = std::time::Instant::now();
        let result = metal_variable_base_msm(&bases, &scalars).unwrap();
        println!("metal_variable_base_msm took {:?}", start.elapsed());

        println!("running arkworks_msm");
        let start = std::time::Instant::now();
        let arkworks_msm = G::msm(&bases, &scalars).unwrap();
        println!("arkworks_msm took {:?}", start.elapsed());

        assert_eq!(
            result, arkworks_msm,
            "Mismatch between GPU e2e result and Arkworks reference"
        );
        println!("GPU e2e result matches Arkworks reference");
    }
}
