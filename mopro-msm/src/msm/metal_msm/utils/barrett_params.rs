use num_bigint::BigUint;

pub fn calc_barrett_mu(p: &BigUint) -> BigUint {
    let k = p.bits() as u32;
    let numerator = BigUint::from(1u32) << (2 * k);
    numerator / p
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use ark_bn254::Fq as BaseField;
    use ark_ff::PrimeField;
    use num_bigint::BigUint;

    #[test]
    #[serial_test::serial]
    fn test_calc_barrett_mu_bn254() {
        let p: BigUint = BaseField::MODULUS.try_into().unwrap();
        let mu = calc_barrett_mu(&p);

        // Precomputed value for BN254's 254-bit modulus
        let expected_mu = BigUint::from_str(
            "38284845454613504619394467267190322316455732053192006567598327834621704638693",
        )
        .unwrap();

        assert_eq!(mu, expected_mu, "Calculated μ doesn't match expected value");

        // Verify 2^(2k) / p relationship
        let k = p.bits() as u32;
        let two_to_2k = BigUint::from(1u32) << (2 * k);
        let product = &mu * &p;
        assert!(product <= two_to_2k, "μ should satisfy μ = floor(2²ᵏ/p)");
        assert!(
            &product + &p > two_to_2k,
            "μ should be the largest integer satisfying μ*p ≤ 2²ᵏ"
        );
    }
}
