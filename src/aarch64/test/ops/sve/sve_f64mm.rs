//! `SveF64mm` against a plain 2×2 product on exactly representable data
//! (`k / 8`), so every rounding order gives the same bits.

use super::SveF64mm;
use crate::aarch64::ops::macros::{SVE_LENS, TestElem, assert_bits_eq};

#[test]
fn mmla_f64_slice_matches_scalar() {
    let Some(t) = SveF64mm::detect() else { return };
    for n in SVE_LENS.map(|n| n / 4 * 4) {
        let acc: Vec<f64> = (0..n).map(|i| f64::sample(i + 1)).collect();
        let a: Vec<f64> = (0..n).map(|i| f64::sample(i + 200)).collect();
        let b: Vec<f64> = (0..n).map(|i| f64::sample(i + 400)).collect();
        let mut out = vec![0.0f64; n];
        t.mmla_f64_slice(&acc, &a, &b, &mut out);
        let want: Vec<f64> = (0..n)
            .map(|x| {
                let (s, i, j) = (x / 4 * 4, x % 4 / 2, x % 2);
                acc[x] + a[s + 2 * i] * b[s + 2 * j] + a[s + 2 * i + 1] * b[s + 2 * j + 1]
            })
            .collect();
        assert_bits_eq(&out, &want, &format!("mmla_f64_slice n={n}"));
    }
}
