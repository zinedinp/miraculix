//! `half::bf16` is the oracle for the IEEE ops. `BFDOT`/`BFMMLA` use
//! non-IEEE intermediate rounding, so they are checked only on small-integer
//! data where every intermediate is exact.

use super::Bf16;
use crate::aarch64::ops::macros::mix;
use half::bf16;

fn b(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}
fn f(x: u16) -> f32 {
    bf16::from_bits(x).to_f32()
}
fn small(i: usize) -> u16 {
    b((mix(i as u64) % 33) as f32 - 16.0)
}

#[test]
fn bfdot_matches_exact_sum() {
    let Some(t) = Bf16::detect() else { return };
    for r in 0..16usize {
        let acc: [f32; 4] =
            core::array::from_fn(|i| ((mix((r * 7 + i) as u64) % 200) as f32) - 100.0);
        let x: [u16; 8] = core::array::from_fn(|i| small(r * 97 + i));
        let y: [u16; 8] = core::array::from_fn(|i| small(r * 97 + i + 50));
        let want: [f32; 4] = core::array::from_fn(|i| {
            acc[i] + f(x[2 * i]) * f(y[2 * i]) + f(x[2 * i + 1]) * f(y[2 * i + 1])
        });
        assert_eq!(t.bfdot_f32x4(acc, x, y), want, "round {r}");
    }
}

#[test]
fn bfmmla_matches_exact_matrix_product() {
    let Some(t) = Bf16::detect() else { return };
    for r in 0..16usize {
        let acc: [f32; 4] =
            core::array::from_fn(|i| ((mix((r * 7 + i) as u64) % 200) as f32) - 100.0);
        let x: [u16; 8] = core::array::from_fn(|i| small(r * 97 + i));
        let y: [u16; 8] = core::array::from_fn(|i| small(r * 97 + i + 50));
        let want: [f32; 4] = core::array::from_fn(|n| {
            let (i, j) = (n / 2, n % 2);
            (0..4).fold(acc[n], |s, k| s + f(x[4 * i + k]) * f(y[4 * j + k]))
        });
        assert_eq!(t.bfmmla_f32x4(acc, x, y), want, "round {r}");
    }
}

#[test]
fn bfmlal_bottom_and_top_are_fused_on_their_lanes() {
    let Some(t) = Bf16::detect() else { return };
    for r in 0..16usize {
        let acc: [f32; 4] = core::array::from_fn(|i| {
            f32::from_bits(0x3F80_0000 | (mix((r + i) as u64) as u32 & 0x7F_FFFF))
        });
        let x: [u16; 8] =
            core::array::from_fn(|i| b(1.0 + (mix((r * 97 + i) as u64) % 127) as f32 / 128.0));
        let y: [u16; 8] =
            core::array::from_fn(|i| b(-1.0 - (mix((r * 97 + i + 9) as u64) % 127) as f32 / 128.0));
        // bf16 × bf16 is exact in f32 (16 significand bits), so mul_add is the oracle.
        let lo: [f32; 4] = core::array::from_fn(|i| f(x[2 * i]).mul_add(f(y[2 * i]), acc[i]));
        let hi: [f32; 4] =
            core::array::from_fn(|i| f(x[2 * i + 1]).mul_add(f(y[2 * i + 1]), acc[i]));
        assert_eq!(
            t.bfmlalb_f32x4(acc, x, y).map(f32::to_bits),
            lo.map(f32::to_bits),
            "bottom round {r}"
        );
        assert_eq!(
            t.bfmlalt_f32x4(acc, x, y).map(f32::to_bits),
            hi.map(f32::to_bits),
            "top round {r}"
        );
    }
}

#[test]
fn cvt_f32x4_to_bf16x4_rounds_to_nearest_even() {
    let Some(t) = Bf16::detect() else { return };
    for i in 0..2000u64 {
        let a: [f32; 4] = core::array::from_fn(|k| f32::from_bits(mix(i * 4 + k as u64) as u32));
        let want = a.map(|x| if x.is_nan() { u16::MAX } else { b(x) });
        let got = t.cvt_f32x4_to_bf16x4(a);
        for k in 0..4 {
            if a[k].is_nan() {
                // NaN payload handling is not what this test pins; only NaN-ness.
                assert!(f(got[k]).is_nan(), "{:?} -> {:#06x}", a[k], got[k]);
            } else {
                assert_eq!(got[k], want[k], "{:e} ({:#010x})", a[k], a[k].to_bits());
            }
        }
    }
    // Tie: 1 + 2^-8 sits exactly between two bf16 values; even mantissa wins.
    assert_eq!(t.cvt_f32x4_to_bf16x4([1.0 + 2f32.powi(-8); 4]), [b(1.0); 4]);
}
