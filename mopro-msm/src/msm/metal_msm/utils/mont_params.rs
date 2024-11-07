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
    use super::{calc_mont_radix, calc_rinv_and_n0};
    use num_bigint::BigUint;

    #[test]
    pub fn test_calc_rinv_and_n0() {
        // Use the BN254 scalar field as a known example
        let p = BigUint::parse_bytes(
            b"30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001",
            16,
        )
        .unwrap();
        let num_limbs = 20;
        let log_limb_size = 13;
        let r = calc_mont_radix(num_limbs, log_limb_size);
        let res = calc_rinv_and_n0(&p, &r, log_limb_size);
        let rinv = res.0;
        let n0 = res.1;

        println!("p: {}", p);
        println!("r: {}", r);
        println!("rinv: {}", rinv);
        println!("n0: {}", n0);

        assert!(
            rinv == BigUint::parse_bytes(
                b"3355749084782366974633145829281540770527962706798299046061554422584528540053",
                10
            )
            .unwrap()
        );
        assert!(n0 == 8191u32);
    }
}
