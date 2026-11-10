//! `half` is the oracle (as for `aarch32::ops::fp16`): IEEE binary16 rounding
//! is not re-implemented here. `half`'s arithmetic goes through `f32` and
//! rounds back once, which is exact for `+ - * / sqrt` on binary16 inputs
//! (binary32 has at least `2·11 + 2` significand bits, so the double rounding
//! is innocuous). Inputs are `k / 8`, exactly representable.

use super::Fp16Neon;
use crate::aarch64::ops::macros::{LENS, mix};
use half::f16;

fn h(i: usize) -> u16 {
    let k = (mix(i as u64) % 2001) as i64 - 1000;
    let k = if k == 0 { 3 } else { k };
    f16::from_f32(k as f32 / 8.0).to_bits()
}

fn f(x: u16) -> f32 {
    f16::from_bits(x).to_f32()
}

fn r(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

macro_rules! fp16_binop_test {
    ($test:ident, $fixed:ident, $slice:ident, $oracle:expr) => {
        #[test]
        fn $test() {
            let Some(t) = Fp16Neon::detect() else { return };
            let op: fn(f32, f32) -> f32 = $oracle;
            for round in 0..8usize {
                let a: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i));
                let b: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i + 50));
                let expect: [u16; 8] = core::array::from_fn(|i| r(op(f(a[i]), f(b[i]))));
                assert_eq!(
                    t.$fixed(a, b),
                    expect,
                    "{} round {round}",
                    stringify!($fixed)
                );
            }
            for len in LENS {
                let a: Vec<u16> = (0..len).map(|i| h(i + 7)).collect();
                let b: Vec<u16> = (0..len).map(|i| h(i + 300)).collect();
                let mut out = vec![0xDEAD; len];
                t.$slice(&a, &b, &mut out);
                let expect: Vec<u16> = a.iter().zip(&b).map(|(&x, &y)| r(op(f(x), f(y)))).collect();
                assert_eq!(out, expect, "{} len={len}", stringify!($slice));
            }
        }
    };
}

macro_rules! fp16_unop_test {
    ($test:ident, $fixed:ident, $slice:ident, $oracle:expr) => {
        #[test]
        fn $test() {
            let Some(t) = Fp16Neon::detect() else { return };
            let op: fn(f32) -> f32 = $oracle;
            for round in 0..8usize {
                let a: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i));
                let expect: [u16; 8] = core::array::from_fn(|i| r(op(f(a[i]))));
                assert_eq!(t.$fixed(a), expect, "{} round {round}", stringify!($fixed));
            }
            for len in LENS {
                let a: Vec<u16> = (0..len).map(|i| h(i + 7)).collect();
                let mut out = vec![0xDEAD; len];
                t.$slice(&a, &mut out);
                let expect: Vec<u16> = a.iter().map(|&x| r(op(f(x)))).collect();
                assert_eq!(out, expect, "{} len={len}", stringify!($slice));
            }
        }
    };
}

macro_rules! fp16_cmp_test {
    ($test:ident, $fixed:ident, $slice:ident, $oracle:expr) => {
        #[test]
        fn $test() {
            let Some(t) = Fp16Neon::detect() else { return };
            let op: fn(f32, f32) -> bool = $oracle;
            let m = |p: bool| if p { u16::MAX } else { 0 };
            for round in 0..8usize {
                let a: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i));
                let b: [u16; 8] = core::array::from_fn(|i| {
                    if i % 2 == 0 {
                        a[i]
                    } else {
                        h(round * 97 + i + 50)
                    }
                });
                let expect: [u16; 8] = core::array::from_fn(|i| m(op(f(a[i]), f(b[i]))));
                assert_eq!(
                    t.$fixed(a, b),
                    expect,
                    "{} round {round}",
                    stringify!($fixed)
                );
            }
            for len in LENS {
                let a: Vec<u16> = (0..len).map(|i| h(i + 7)).collect();
                let b: Vec<u16> = (0..len)
                    .map(|i| if i % 3 == 0 { a[i] } else { h(i + 300) })
                    .collect();
                let mut out = vec![0x5A5A; len];
                t.$slice(&a, &b, &mut out);
                let expect: Vec<u16> = a.iter().zip(&b).map(|(&x, &y)| m(op(f(x), f(y)))).collect();
                assert_eq!(out, expect, "{} len={len}", stringify!($slice));
            }
        }
    };
}

fp16_binop_test!(add_f16x8_matches_half, add_f16x8, add_f16_slice, |x, y| x
    + y);
fp16_binop_test!(sub_f16x8_matches_half, sub_f16x8, sub_f16_slice, |x, y| x
    - y);
fp16_binop_test!(mul_f16x8_matches_half, mul_f16x8, mul_f16_slice, |x, y| x
    * y);
fp16_binop_test!(div_f16x8_matches_half, div_f16x8, div_f16_slice, |x, y| x
    / y);
fp16_binop_test!(max_f16x8_matches_half, max_f16x8, max_f16_slice, |x, y| x
    .max(y));
fp16_binop_test!(min_f16x8_matches_half, min_f16x8, min_f16_slice, |x, y| x
    .min(y));
fp16_unop_test!(abs_f16x8_matches_half, abs_f16x8, abs_f16_slice, |x| x
    .abs());
fp16_unop_test!(neg_f16x8_matches_half, neg_f16x8, neg_f16_slice, |x| -x);
fp16_unop_test!(sqrt_f16x8_matches_half, sqrt_f16x8, sqrt_f16_slice, |x| x
    .sqrt());
fp16_cmp_test!(
    cmpeq_f16x8_matches_half,
    cmpeq_f16x8,
    cmpeq_f16_slice,
    |x, y| x == y
);
fp16_cmp_test!(
    cmpgt_f16x8_matches_half,
    cmpgt_f16x8,
    cmpgt_f16_slice,
    |x, y| x > y
);
fp16_cmp_test!(
    cmpge_f16x8_matches_half,
    cmpge_f16x8,
    cmpge_f16_slice,
    |x, y| x >= y
);
fp16_cmp_test!(
    cmplt_f16x8_matches_half,
    cmplt_f16x8,
    cmplt_f16_slice,
    |x, y| x < y
);
fp16_cmp_test!(
    cmple_f16x8_matches_half,
    cmple_f16x8,
    cmple_f16_slice,
    |x, y| x <= y
);

#[test]
fn cmpeq_f16x8_nan_is_never_equal() {
    let Some(t) = Fp16Neon::detect() else { return };
    let nan = f16::NAN.to_bits();
    assert_eq!(t.cmpeq_f16x8([nan; 8], [nan; 8]), [0; 8]);
}

#[test]
fn fmadd_f16x8_matches_exact_and_is_fused() {
    let Some(t) = Fp16Neon::detect() else { return };
    // k/8 inputs: a*b+c is exact in f32, so one f16 rounding is the oracle.
    for round in 0..8usize {
        let a: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i));
        let b: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i + 50));
        let c: [u16; 8] = core::array::from_fn(|i| h(round * 97 + i + 90));
        let expect: [u16; 8] = core::array::from_fn(|i| r(f(a[i]) * f(b[i]) + f(c[i])));
        assert_eq!(t.fmadd_f16x8(a, b, c), expect, "round {round}");
    }
    // (1 + 2^-10)(1 - 2^-10) - 1 = -2^-20: rounds to 0 unless fused.
    let e = 2f32.powi(-10);
    let got = t.fmadd_f16x8([r(1.0 + e); 8], [r(1.0 - e); 8], [r(-1.0); 8]);
    assert_eq!(got, [r(-2f32.powi(-20)); 8], "fmla did not fuse");
    // Slice: the padded tail is fused as well.
    for len in LENS {
        let a = vec![r(1.0 + e); len];
        let b = vec![r(1.0 - e); len];
        let c = vec![r(-1.0); len];
        let mut out = vec![0; len];
        t.fmadd_f16_slice(&a, &b, &c, &mut out);
        assert_eq!(out, vec![r(-2f32.powi(-20)); len], "len={len}");
    }
}

#[test]
fn fma_variants_are_fused_with_the_right_signs() {
    let Some(t) = Fp16Neon::detect() else { return };
    // p = (1 + 2^-10)(1 - 2^-10) = 1 - 2^-20: only a fused op keeps the 2^-20.
    let e = 2f32.powi(-10);
    let (a, b) = ([r(1.0 + e); 8], [r(1.0 - e); 8]);
    let tiny = 2f32.powi(-20);
    assert_eq!(t.fmsub_f16x8(a, b, [r(1.0); 8]), [r(-tiny); 8], "p - 1");
    assert_eq!(t.fnmadd_f16x8(a, b, [r(1.0); 8]), [r(tiny); 8], "-p + 1");
    assert_eq!(
        t.fnmsub_f16x8(a, b, [r(-1.0); 8]),
        [r(tiny); 8],
        "-p - (-1)"
    );
    for len in LENS {
        let (va, vb, vc) = (
            vec![r(1.0 + e); len],
            vec![r(1.0 - e); len],
            vec![r(1.0); len],
        );
        let mut out = vec![0; len];
        t.fmsub_f16_slice(&va, &vb, &vc, &mut out);
        assert_eq!(out, vec![r(-tiny); len], "fmsub len={len}");
        t.fnmadd_f16_slice(&va, &vb, &vc, &mut out);
        assert_eq!(out, vec![r(tiny); len], "fnmadd len={len}");
        t.fnmsub_f16_slice(&va, &vb, &vc, &mut out);
        assert_eq!(out, vec![r(-2.0 + tiny); len], "fnmsub len={len}");
    }
}
