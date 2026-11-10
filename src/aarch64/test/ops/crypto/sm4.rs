//! SM4's round is an unbalanced Feistel step,
//! `X[i+4] = X[i] ^ T(X[i+1] ^ X[i+2] ^ X[i+3] ^ rk[i])`, so running it on the
//! reversed output with reversed keys walks back to the input - for any `T`.
//! That inverts `SM4E` (and, with `T'`/`CK`, `SM4EKEY`) exactly, without
//! transcribing the S-box.

use super::Sm4;
use crate::aarch64::ops::macros::mix;

fn v(i: u64) -> [u32; 4] {
    core::array::from_fn(|k| mix(i * 4 + k as u64) as u32)
}
fn rev(x: [u32; 4]) -> [u32; 4] {
    [x[3], x[2], x[1], x[0]]
}

#[test]
fn sm4e_is_inverted_by_reversal() {
    let Some(t) = Sm4::detect() else { return };
    for i in 0..300 {
        let (x, rk) = (v(i), v(i + 5000));
        let y = t.sm4e(x, rk);
        assert_ne!(y, x);
        assert_eq!(rev(t.sm4e(rev(y), rev(rk))), x);
    }
}

#[test]
fn sm4ekey_is_inverted_by_reversal() {
    let Some(t) = Sm4::detect() else { return };
    for i in 0..300 {
        let (k, ck) = (v(i), v(i + 9000));
        let y = t.sm4ekey(k, ck);
        assert_eq!(rev(t.sm4ekey(rev(y), rev(ck))), k);
    }
}
