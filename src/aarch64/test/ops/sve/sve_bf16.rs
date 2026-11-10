//! `SveBf16` against scalar oracles on exact data (small integers as `bf16`,
//! where `BFDOT`/`BFMMLA`'s odd rounding cannot show), and `cvt` against the
//! Neon `Bf16` token's `BFCVTN`.

use super::SveBf16;
use crate::aarch64::ops::macros::{SVE_LENS, assert_bits_eq, mix};
use crate::aarch64::ops::neon::bf16::Bf16;

/// Small integer in -8..=8 as a `bf16` bit pattern (exact in 8 bits).
fn small(i: usize) -> u16 {
    (((mix(i as u64) % 17) as i32 - 8) as f32)
        .to_bits()
        .wrapping_shr(16) as u16
}

fn f(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[test]
fn bf16_accumulate_ops_match_scalar() {
    let Some(t) = SveBf16::detect() else { return };
    for n in SVE_LENS.map(|n| n / 4 * 4) {
        let acc: Vec<f32> = (0..n).map(|i| f(small(i + 1000))).collect();
        let a: Vec<u16> = (0..2 * n).map(|i| small(i + 2000)).collect();
        let b: Vec<u16> = (0..2 * n).map(|i| small(i + 3000)).collect();
        let mut out = vec![0.0f32; n];
        let run = |name: &str, want: &dyn Fn(usize) -> f32, got: &[f32]| {
            let want: Vec<f32> = (0..n).map(want).collect();
            assert_bits_eq(got, &want, &format!("{name} n={n}"));
        };
        t.bfdot_f32_slice(&acc, &a, &b, &mut out);
        run(
            "bfdot",
            &|i| acc[i] + f(a[2 * i]) * f(b[2 * i]) + f(a[2 * i + 1]) * f(b[2 * i + 1]),
            &out,
        );
        t.bfmlalb_f32_slice(&acc, &a, &b, &mut out);
        run("bfmlalb", &|i| acc[i] + f(a[2 * i]) * f(b[2 * i]), &out);
        t.bfmlalt_f32_slice(&acc, &a, &b, &mut out);
        run(
            "bfmlalt",
            &|i| acc[i] + f(a[2 * i + 1]) * f(b[2 * i + 1]),
            &out,
        );
        t.bfmmla_f32_slice(&acc, &a, &b, &mut out);
        run(
            "bfmmla",
            &|x| {
                // Segment = 4 f32 lanes of `acc`, 8 bf16 of `a` and of `b`.
                let (s, i, j) = (x / 4, x % 4 / 2, x % 2);
                (0..4).fold(acc[x], |sum, k| {
                    sum + f(a[8 * s + 4 * i + k]) * f(b[8 * s + 4 * j + k])
                })
            },
            &out,
        );
    }
}

#[test]
fn cvt_f32_to_bf16_matches_neon_bfcvtn() {
    let (Some(t), Some(n)) = (SveBf16::detect(), Bf16::detect()) else {
        return;
    };
    for len in SVE_LENS.map(|l| l / 4 * 4) {
        // Random bit patterns: covers ties, NaN, infinities and denormals.
        let a: Vec<f32> = (0..len)
            .map(|i| f32::from_bits(mix(i as u64 ^ 77) as u32))
            .collect();
        let mut out = vec![0u16; len];
        t.cvt_f32_to_bf16_slice(&a, &mut out);
        for c in 0..len / 4 {
            let want = n.cvt_f32x4_to_bf16x4(a[4 * c..4 * c + 4].try_into().unwrap());
            assert_eq!(out[4 * c..4 * c + 4], want, "chunk {c} of {len}");
        }
    }
}
