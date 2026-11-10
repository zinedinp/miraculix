//! `Sve2Aes` / `Sve2Pmull` against the Neon `Aes` / `Pmull` tokens, block by
//! block. Both are present on every QEMU row that has SVE2-AES, so the Neon
//! instruction is the oracle for the same maths on `z` registers.

use super::{Sve2Aes, Sve2Pmull};
use crate::aarch64::ops::crypto::aes::Aes;
use crate::aarch64::ops::crypto::pmull::Pmull;
use crate::aarch64::ops::macros::mix;

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    (0..n).map(|i| mix(i as u64 ^ seed) as u8).collect()
}

#[test]
fn aes_blocks_match_neon() {
    let (Some(t), Some(n)) = (Sve2Aes::detect(), Aes::detect()) else {
        return;
    };
    for blocks in [0usize, 1, 2, 3, 4, 5, 16, 17, 33] {
        let len = blocks * 16;
        let (d, k) = (bytes(len, 1), bytes(len, 2));
        let block = |v: &[u8], i: usize| -> [u8; 16] { v[16 * i..16 * i + 16].try_into().unwrap() };
        let mut out = vec![0u8; len];
        t.aese_slice(&d, &k, &mut out);
        for i in 0..blocks {
            assert_eq!(
                block(&out, i),
                n.aese(block(&d, i), block(&k, i)),
                "aese block {i}/{blocks}"
            );
        }
        t.aesd_slice(&d, &k, &mut out);
        for i in 0..blocks {
            assert_eq!(
                block(&out, i),
                n.aesd(block(&d, i), block(&k, i)),
                "aesd block {i}/{blocks}"
            );
        }
        t.aesmc_slice(&d, &mut out);
        for i in 0..blocks {
            assert_eq!(
                block(&out, i),
                n.aesmc(block(&d, i)),
                "aesmc block {i}/{blocks}"
            );
        }
        t.aesimc_slice(&d, &mut out);
        for i in 0..blocks {
            assert_eq!(
                block(&out, i),
                n.aesimc(block(&d, i)),
                "aesimc block {i}/{blocks}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "multiple of 16")]
fn aes_rejects_a_partial_block() {
    let Some(t) = Sve2Aes::detect() else {
        panic!("multiple of 16 (no SVE2-AES here; vacuous)")
    };
    let mut out = [0u8; 20];
    t.aesmc_slice(&[0u8; 20], &mut out);
}

#[test]
fn pmull_pairs_match_neon() {
    let (Some(t), Some(n)) = (Sve2Pmull::detect(), Pmull::detect()) else {
        return;
    };
    for pairs in [0usize, 1, 2, 3, 8, 9, 64] {
        let len = pairs * 2;
        let a: Vec<u64> = (0..len).map(|i| mix(i as u64 ^ 3)).collect();
        let b: Vec<u64> = (0..len).map(|i| mix(i as u64 ^ 4)).collect();
        let mut out = vec![0u64; len];
        let got = |out: &[u64], k: usize| out[2 * k] as u128 | (out[2 * k + 1] as u128) << 64;
        t.pmull_u64_slice(&a, &b, &mut out);
        for k in 0..pairs {
            assert_eq!(
                got(&out, k),
                n.pmull_u64(a[2 * k], b[2 * k]),
                "pmullb pair {k}"
            );
        }
        t.pmull_high_u64_slice(&a, &b, &mut out);
        for k in 0..pairs {
            let (x, y) = ([a[2 * k], a[2 * k + 1]], [b[2 * k], b[2 * k + 1]]);
            assert_eq!(got(&out, k), n.pmull_high_u64x2(x, y), "pmullt pair {k}");
        }
    }
}
