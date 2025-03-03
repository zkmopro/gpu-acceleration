// adapted from https://github.com/geometers/multiprecision/master/src/mont.rs

/* Constants for bn254 field operations
 * N: scalar field modulus
 * R_SQUARED: R^2 mod N
 * R_SUB_N: R - N
 * MU: Montgomery Multiplication Constant = -N^{-1} mod (2^32)
 *
 * For bn254, the modulus is "21888242871839275222246405745257275088548364400416034343698204186575808495617" [1, 2]
 * We use 8 limbs of 32 bits unsigned integers to represent the constanst
 *
 * References:
 * [1] https://github.com/arkworks-rs/algebra/blob/065cd24fc5ae17e024c892cee126ad3bd885f01c/curves/bn254/src/lib.rs
 * [2] https://github.com/scipr-lab/libff/blob/develop/libff/algebra/curves/alt_bn128/alt_bn128.sage
 */

use ark_bn254::Fq as BaseField;
use ark_ff::PrimeField;
use num_bigint::{BigInt, BigUint, Sign};

pub fn calc_nsafe(log_limb_size: u32) -> usize {
    let max_int_width = 32;
    let rhs = 2u64.pow(max_int_width);
    let mut k = 1usize;
    let x = 2u64.pow(2u32 * log_limb_size);
    while (k as u64) * x <= rhs {
        k += 1;
    }

    k / 2
}

pub fn calc_mont_radix(num_limbs: usize, log_limb_size: u32) -> BigUint {
    BigUint::from(2u32).pow(num_limbs as u32 * log_limb_size)
}

fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if *a == BigInt::from(0u32) {
        return (b.clone(), BigInt::from(0u32), BigInt::from(1u32));
    }
    let (g, x, y) = egcd(&(b % a), a);

    (g, y - (b / a) * x.clone(), x.clone())
}

pub fn calc_inv_and_pprime(p: &BigUint, r: &BigUint) -> (BigUint, BigUint) {
    assert!(*r != BigUint::from(0u32));

    let p_bigint = BigInt::from_biguint(Sign::Plus, p.clone());
    let r_bigint = BigInt::from_biguint(Sign::Plus, r.clone());
    let one = BigInt::from(1u32);
    let (_, mut rinv, mut pprime) = egcd(
        &BigInt::from_biguint(Sign::Plus, r.clone()),
        &BigInt::from_biguint(Sign::Plus, p.clone()),
    );

    if rinv.sign() == Sign::Minus {
        rinv = BigInt::from_biguint(Sign::Plus, p.clone()) + rinv;
    }

    if pprime.sign() == Sign::Minus {
        pprime = BigInt::from_biguint(Sign::Plus, r.clone()) + pprime;
    }

    // r * rinv - p * pprime == 1
    assert!(
        (BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint)
            - (&p_bigint * &pprime % &p_bigint)
            == one
    );

    // r * rinv % p == 1
    assert!((BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint) == one);

    // p * pprime % r == 1
    assert!((&p_bigint * &pprime % &r_bigint) == one);

    (rinv.to_biguint().unwrap(), pprime.to_biguint().unwrap())
}

pub fn calc_rinv_and_n0(p: &BigUint, r: &BigUint, log_limb_size: u32) -> (BigUint, u32) {
    let (rinv, pprime) = calc_inv_and_pprime(p, r);
    let pprime = BigInt::from_biguint(Sign::Plus, pprime);

    let neg_n_inv = BigInt::from_biguint(Sign::Plus, r.clone()) - pprime;
    let n0 = neg_n_inv % BigInt::from(2u32.pow(log_limb_size as u32));
    let n0 = n0.to_biguint().unwrap().to_u32_digits()[0];

    (rinv, n0)
}

#[cfg(test)]
pub mod tests {
    use std::str::FromStr;

    use super::{calc_mont_radix, calc_rinv_and_n0};
    use ark_bn254::Fq as BaseField;
    use ark_ff::PrimeField;
    use num_bigint::BigUint;

    #[test]
    pub fn test_calc_rinv_and_n0() {
        // Use the BN254 base field as a known example
        let p: BigUint = BaseField::MODULUS.try_into().unwrap();
        let num_limbs = 16;
        let log_limb_size = 16;
        let r = calc_mont_radix(num_limbs, log_limb_size);
        let res = calc_rinv_and_n0(&p, &r, log_limb_size);
        let rinv = res.0;
        let n0 = res.1;

        println!("p: {}", p);
        println!("r: {}", r);
        println!("rinv: {}", rinv);
        println!("n0: {}", n0);

        assert!(
            rinv == BigUint::from_str(
                "20988524275117001072002809824448087578619730785600314334253784976379291040311"
            )
            .unwrap()
        );
        assert!(n0 == 25481u32);
    }
}

pub fn calc_num_limbs(log_limb_size: u32, p_bitwidth: usize) -> usize {
    let l = log_limb_size as usize;
    let mut num_limbs = p_bitwidth / l;
    while num_limbs * l <= p_bitwidth {
        num_limbs += 1;
    }
    num_limbs
}

pub fn calc_bitwidth(p: &BigUint) -> usize {
    if *p == BigUint::from(0u32) {
        return 0;
    }

    p.to_radix_le(2).len()
}

pub struct MontgomeryParams {
    pub log_limb_size: u32,
    pub p: BigUint,
    pub modulus_bits: u32,
    pub num_limbs: usize,
    pub r: BigUint,
    pub rinv: BigUint,
    pub n0: u32,
    pub nsafe: usize,
}

impl Default for MontgomeryParams {
    fn default() -> Self {
        let log_limb_size: u32 = 16;
        let p: BigUint = BaseField::MODULUS.try_into().unwrap();
        let modulus_bits = BaseField::MODULUS_BIT_SIZE as u32;
        let num_limbs = ((modulus_bits + log_limb_size - 1) / log_limb_size) as usize;

        let r = calc_mont_radix(num_limbs, log_limb_size);
        let (rinv, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
        let nsafe = calc_nsafe(log_limb_size);

        Self {
            log_limb_size,
            p,
            modulus_bits,
            num_limbs,
            r,
            rinv,
            n0,
            nsafe,
        }
    }
}
