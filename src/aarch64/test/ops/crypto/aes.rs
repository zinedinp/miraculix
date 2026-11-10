//! Algebraic identities, as in `aarch32/test/ops/crypto.rs`: no S-box table
//! is transcribed, so there is nothing to transcribe wrongly.

use super::Aes;
use crate::aarch64::ops::macros::mix;

fn block(i: u64) -> [u8; 16] {
    let (lo, hi) = (mix(i), mix(i ^ 0xABCD));
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&lo.to_le_bytes());
    b[8..].copy_from_slice(&hi.to_le_bytes());
    b
}

#[test]
fn aesmc_aesimc_are_exact_inverses() {
    let Some(t) = Aes::detect() else { return };
    for i in 0..64 {
        let x = block(i);
        assert_eq!(t.aesimc(t.aesmc(x)), x);
        assert_eq!(t.aesmc(t.aesimc(x)), x);
    }
}

#[test]
fn aesd_undoes_aese_with_a_zero_key() {
    let Some(t) = Aes::detect() else { return };
    for i in 0..64 {
        let x = block(i);
        assert_eq!(t.aesd(t.aese(x, [0; 16]), [0; 16]), x);
    }
}

#[test]
fn aese_key_is_a_plain_xor_before_the_round() {
    let Some(t) = Aes::detect() else { return };
    // AESE(x, k) = SubBytes(ShiftRows(x ^ k)) = AESE(x ^ k, 0).
    for i in 0..64 {
        let (x, k) = (block(i), block(i + 1000));
        let xk: [u8; 16] = core::array::from_fn(|j| x[j] ^ k[j]);
        assert_eq!(t.aese(x, k), t.aese(xk, [0; 16]));
        assert_eq!(t.aesd(x, k), t.aesd(xk, [0; 16]));
    }
}

#[test]
fn aese_of_zero_is_the_sbox_of_zero() {
    let Some(t) = Aes::detect() else { return };
    // S(0x00) = 0x63 is the S-box's defining affine constant (FIPS-197 5.1.1),
    // and ShiftRows of a constant block is the same block.
    assert_eq!(t.aese([0; 16], [0; 16]), [0x63; 16]);
}
