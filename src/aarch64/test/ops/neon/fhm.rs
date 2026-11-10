//! Oracle: an `f16 × f16` product is exact in `f32` (22 significand bits), so
//! `acc.mul_add(±1, a·b)` - one `f32` rounding - is exactly what `FMLAL` computes.

use super::Fhm;
use crate::aarch64::ops::macros::{TestElem, mix};
use half::f16;

fn h(i: usize) -> u16 {
    // Wider spread than the k/8 grid, so the accumulate actually rounds.
    let k = (mix(i as u64) % 60001) as f32 - 30000.0;
    f16::from_f32(k / 1024.0).to_bits()
}

macro_rules! fhm_test {
    ($test:ident, $method:ident, $off:literal, $sub:literal) => {
        #[test]
        fn $test() {
            let Some(t) = Fhm::detect() else { return };
            for r in 0..16usize {
                let acc: [f32; 4] = core::array::from_fn(|i| f32::sample(r * 31 + i) * 1.000_123);
                let a: [u16; 8] = core::array::from_fn(|i| h(r * 97 + i));
                let b: [u16; 8] = core::array::from_fn(|i| h(r * 97 + i + 50));
                let want: [f32; 4] = core::array::from_fn(|i| {
                    let p =
                        f16::from_bits(a[i + $off]).to_f32() * f16::from_bits(b[i + $off]).to_f32();
                    (if $sub { -p } else { p }).mul_add(1.0, acc[i])
                });
                let got = t.$method(acc, a, b);
                assert_eq!(
                    got.map(f32::to_bits),
                    want.map(f32::to_bits),
                    "{} round {r}",
                    stringify!($method)
                );
            }
        }
    };
}

fhm_test!(fmlal_low_f32x4_is_one_rounding, fmlal_low_f32x4, 0, false);
fhm_test!(fmlal_high_f32x4_is_one_rounding, fmlal_high_f32x4, 4, false);
fhm_test!(fmlsl_low_f32x4_is_one_rounding, fmlsl_low_f32x4, 0, true);
fhm_test!(fmlsl_high_f32x4_is_one_rounding, fmlsl_high_f32x4, 4, true);
