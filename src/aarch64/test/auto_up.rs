//! `auto` layer: name coverage, argument order, and the `f16` bottom rung.
//! Per-op lane semantics are already pinned by the token tests; these check
//! what the `auto` wrappers themselves add.

use super::super::ops::neon::fp16::Fp16Neon;
use super::super::ops::neon::neon::Neon;
use crate::aarch64::ops::macros::{LENS, mix};
use crate::aarch64::{auto, auto_down, auto_up};
use half::f16;

/// Compile-time proof that every one of `aarch32::auto_up`'s 62 names exists
/// here with the same signature, so aarch32 call sites port unchanged.
#[test]
fn aarch32_auto_names_are_a_subset() {
    type B<T> = fn(&[T], &[T], &mut [T]);
    type U<T> = fn(&[T], &mut [T]);
    type C<T, M> = fn(&[T], &[T], &mut [M]);
    type S<T> = fn(&[T], &[T], &[u32], &mut [T]);
    type F<T> = fn(&[T], &[T], &[T], &mut [T]);
    let _: [B<i32>; 7] = [
        auto::add_i32,
        auto::sub_i32,
        auto::mul_i32,
        auto::max_i32,
        auto::min_i32,
        auto::shl_i32,
        auto::add_i32,
    ];
    let _: [B<f32>; 5] = [
        auto::add_f32,
        auto::sub_f32,
        auto::mul_f32,
        auto::max_f32,
        auto::min_f32,
    ];
    let _: [B<u32>; 4] = [auto::and_u32, auto::or_u32, auto::xor_u32, auto::andnot_u32];
    let _: [U<i32>; 2] = [auto::abs_i32, auto::neg_i32];
    let _: [U<f32>; 2] = [auto::abs_f32, auto::neg_f32];
    let _: U<u32> = auto::not_u32;
    let _: [C<i32, u32>; 5] = [
        auto::cmpeq_i32,
        auto::cmpgt_i32,
        auto::cmpge_i32,
        auto::cmplt_i32,
        auto::cmple_i32,
    ];
    let _: [C<f32, u32>; 5] = [
        auto::cmpeq_f32,
        auto::cmpgt_f32,
        auto::cmpge_f32,
        auto::cmplt_f32,
        auto::cmple_f32,
    ];
    let _: S<i32> = auto::select_i32;
    let _: S<u32> = auto::select_u32;
    let _: S<f32> = auto::select_f32;
    let _: F<f32> = auto::fmadd_f32;
    let _: [B<u16>; 11] = [
        auto::add_f16,
        auto::sub_f16,
        auto::mul_f16,
        auto::max_f16,
        auto::min_f16,
        auto::cmpeq_f16,
        auto::cmpgt_f16,
        auto::cmpge_f16,
        auto::cmplt_f16,
        auto::cmple_f16,
        auto::add_f16,
    ];
    let _: [U<u16>; 2] = [auto::abs_f16, auto::neg_f16];
    let _: F<u16> = auto::fmadd_f16;
    let _: [B<i8>; 6] = [
        auto::qadd_i8,
        auto::qsub_i8,
        auto::sadd_i8,
        auto::ssub_i8,
        auto::shadd_i8,
        auto::shsub_i8,
    ];
    let _: [B<i16>; 8] = [
        auto::qadd_i16,
        auto::qsub_i16,
        auto::sadd_i16,
        auto::shadd_i16,
        auto::shsub_i16,
        auto::qasx_i16,
        auto::qsax_i16,
        auto::sasx_i16,
    ];
    let _: B<u8> = auto::usub_u8;
}

#[test]
fn auto_is_an_alias_of_auto_up() {
    let (a, b) = ([1i32, -2, 3, -4, 5], [10, 20, -30, 40, 50]);
    let (mut x, mut y) = ([0i32; 5], [0i32; 5]);
    auto::add_i32(&a, &b, &mut x);
    auto_up::add_i32(&a, &b, &mut y);
    assert_eq!(x, y);
    assert_eq!(x, [11, 18, -27, 36, 55]);
}

#[test]
fn select_takes_b_where_the_mask_is_set() {
    let a = [1i32, 2, 3, 4, 0x0007_0005];
    let b = [10i32, 20, 30, 40, 0x0009_0032];
    let mask = [u32::MAX, 0, u32::MAX, 0, 0x0000_FFFF];
    let mut out = [0i32; 5];
    auto::select_i32(&a, &b, &mask, &mut out);
    // Per bit: the last lane mixes b's low half with a's high half.
    assert_eq!(out, [10, 2, 30, 4, 0x0007_0032]);
}

#[test]
fn andnot_is_a_and_not_b() {
    let mut out = [0u32; 3];
    auto::andnot_u32(&[0b1100, 0b1010, u32::MAX], &[0b1010, 0b1010, 0], &mut out);
    assert_eq!(out, [0b0100, 0, u32::MAX]);
}

#[test]
fn x86_aliases_are_the_same_functions() {
    let (a, b) = ([i8::MAX, i8::MIN, 3], [1i8, -1, 4]);
    let (mut x, mut y) = ([0i8; 3], [0i8; 3]);
    auto::adds_i8(&a, &b, &mut x);
    auto::qadd_i8(&a, &b, &mut y);
    assert_eq!(x, y);
    assert_eq!(x, [i8::MAX, i8::MIN, 7]);
    let mut m = [0i64; 2];
    auto::mullo_i64(&[i64::MAX, -3], &[2, 5], &mut m);
    assert_eq!(m, [-2, -15]);
}

#[test]
fn reductions_and_shifts_reach_the_token() {
    let a: Vec<i32> = (1..=100).collect();
    assert_eq!(auto::sum_i32(&a), 5050);
    assert_eq!(auto::maxv_i32(&a), 100);
    assert_eq!(auto::minv_i32(&[]), i32::MAX);
    let mut out = vec![0i32; a.len()];
    auto::slli_i32::<3>(&a, &mut out);
    assert_eq!(out, a.iter().map(|x| x << 3).collect::<Vec<_>>());
}

// ---- f16 bottom rung --------------------------------------------------------

fn h(i: u64) -> u16 {
    mix(i) as u16
}

/// Compare as values, NaN only by NaN-ness (payloads may legitimately differ
/// between the hardware and the widen/narrow path).
fn same_f16(x: u16, y: u16) -> bool {
    let (a, b) = (f16::from_bits(x), f16::from_bits(y));
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else {
        x == y
    }
}

/// The `f32` path must agree bit-for-bit with native `f16` hardware on
/// arbitrary bit patterns - subnormals, infinities, huge/tiny mixes.
#[test]
fn f16_bottom_rung_matches_fp16_hardware() {
    let Some(t) = Fp16Neon::detect() else { return };
    type Pair = (
        fn(&[u16], &[u16], &mut [u16]),
        fn(Fp16Neon, &[u16], &[u16], &mut [u16]),
        &'static str,
    );
    let ops: [Pair; 11] = [
        (auto_down::add_f16, Fp16Neon::add_f16_slice, "add"),
        (auto_down::sub_f16, Fp16Neon::sub_f16_slice, "sub"),
        (auto_down::mul_f16, Fp16Neon::mul_f16_slice, "mul"),
        (auto_down::div_f16, Fp16Neon::div_f16_slice, "div"),
        (auto_down::max_f16, Fp16Neon::max_f16_slice, "max"),
        (auto_down::min_f16, Fp16Neon::min_f16_slice, "min"),
        (auto_down::cmpeq_f16, Fp16Neon::cmpeq_f16_slice, "cmpeq"),
        (auto_down::cmpgt_f16, Fp16Neon::cmpgt_f16_slice, "cmpgt"),
        (auto_down::cmpge_f16, Fp16Neon::cmpge_f16_slice, "cmpge"),
        (auto_down::cmplt_f16, Fp16Neon::cmplt_f16_slice, "cmplt"),
        (auto_down::cmple_f16, Fp16Neon::cmple_f16_slice, "cmple"),
    ];
    let n = 4003;
    let a: Vec<u16> = (0..n).map(h).collect();
    let b: Vec<u16> = (0..n).map(|i| h(i + 99_999)).collect();
    for (down, hw, name) in ops {
        let (mut x, mut y) = (vec![0; n as usize], vec![0; n as usize]);
        down(&a, &b, &mut x);
        hw(t, &a, &b, &mut y);
        for i in 0..n as usize {
            assert!(
                same_f16(x[i], y[i]),
                "{name}({:#06x}, {:#06x}): widen {:#06x} vs fp16 {:#06x}",
                a[i],
                b[i],
                x[i],
                y[i]
            );
        }
    }
}

/// Exact `a·b + c` for binary16 inputs, rounded once to binary16. Independent
/// of the implementation: every finite binary16 value is a multiple of 2^-24,
/// so `(a·b + c)·2^48` is an integer in `i128`; round it to odd at 24 bits
/// (exact `f32`) and let `half` do the single RNE step to binary16.
///
/// Not via `half::f16::from_f64`: in half 2.7.1 its software fallback (used
/// when the CPU lacks FEAT_FP16, e.g. QEMU `cortex-a53`) truncates the low 32
/// mantissa bits first and so loses the sticky bit - `from_f64(26888.0146…)`
/// gives `0x7690` instead of `0x7691`. The `f32` fallback keeps it.
fn fma_f16_exact(a: u16, b: u16, c: u16) -> Option<u16> {
    let (fa, fb, fc) = (f16::from_bits(a), f16::from_bits(b), f16::from_bits(c));
    if !(fa.is_finite() && fb.is_finite() && fc.is_finite()) {
        return None;
    }
    let q = |x: f16| (x.to_f64() * 16_777_216.0) as i128; // exact: x · 2^24
    let n = q(fa) * q(fb) + (q(fc) << 24);
    let (neg, mut m) = (n < 0, n.unsigned_abs());
    let mut shift = 0;
    while m >> 24 != 0 {
        let sticky = m & 1;
        m = (m >> 1) | sticky;
        shift += 1;
    }
    let v = (m as f32) * 2f32.powi(shift - 48);
    Some(f16::from_f32(if neg { -v } else { v }).to_bits())
}

#[test]
fn fmadd_f16_bottom_rung_rounds_once() {
    let mut checked = 0;
    for i in 0..20_000u64 {
        let (a, b, c) = (h(3 * i), h(3 * i + 1), h(3 * i + 2));
        let Some(want) = fma_f16_exact(a, b, c) else {
            continue;
        };
        let mut got = [0u16];
        auto_down::fmadd_f16(&[a], &[b], &[c], &mut got);
        assert!(
            same_f16(got[0], want),
            "fma({a:#06x}, {b:#06x}, {c:#06x}) = {:#06x}, want {want:#06x}",
            got[0]
        );
        checked += 1;
    }
    assert!(checked > 10_000);
    // Double-rounding trap: a·b + c = 1 + 2^-11 + 2^-24, just above the f16
    // tie at 1 + 2^-11. Rounding to f32 first lands exactly on the tie (then
    // to even: 1.0); the correct single rounding is up, to 1 + 2^-10.
    let a = f16::from_f32(1.0 + 2f32.powi(-10)).to_bits();
    let b = f16::from_f32(1.0 - 2f32.powi(-11)).to_bits();
    let c = f16::from_f32(9.0 * 2f32.powi(-24)).to_bits();
    assert_eq!(
        fma_f16_exact(a, b, c),
        Some(f16::from_f32(1.0 + 2f32.powi(-10)).to_bits()),
        "oracle"
    );
    let naive = f16::from_f32(
        f16::from_bits(a)
            .to_f32()
            .mul_add(f16::from_bits(b).to_f32(), f16::from_bits(c).to_f32()),
    );
    assert_eq!(
        naive,
        f16::from_f32(1.0),
        "the trap really traps an f32 intermediate"
    );
    let mut got = [0u16];
    auto_down::fmadd_f16(&[a], &[b], &[c], &mut got);
    assert_eq!(got[0], f16::from_f32(1.0 + 2f32.powi(-10)).to_bits());
}

#[test]
fn fmadd_f16_auto_matches_hardware_when_present() {
    let Some(t) = Fp16Neon::detect() else { return };
    let n = 3001;
    let a: Vec<u16> = (0..n).map(h).collect();
    let b: Vec<u16> = (0..n).map(|i| h(i + 7)).collect();
    let c: Vec<u16> = (0..n).map(|i| h(i + 13)).collect();
    let (mut x, mut y) = (vec![0; n as usize], vec![0; n as usize]);
    auto_down::fmadd_f16(&a, &b, &c, &mut x);
    t.fmadd_f16_slice(&a, &b, &c, &mut y);
    for i in 0..n as usize {
        assert!(
            same_f16(x[i], y[i]),
            "fma({:#06x}, {:#06x}, {:#06x}): {:#06x} vs {:#06x}",
            a[i],
            b[i],
            c[i],
            x[i],
            y[i]
        );
    }
}

#[test]
fn f16_family_handles_every_length() {
    for len in LENS {
        let a: Vec<u16> = (0..len)
            .map(|i| f16::from_f32(i as f32 * 0.5 - 7.0).to_bits())
            .collect();
        let b: Vec<u16> = (0..len)
            .map(|i| f16::from_f32(3.25 - i as f32 * 0.25).to_bits())
            .collect();
        let mut out = vec![0u16; len];
        auto::add_f16(&a, &b, &mut out);
        let want: Vec<u16> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                f16::from_f32(f16::from_bits(x).to_f32() + f16::from_bits(y).to_f32()).to_bits()
            })
            .collect();
        assert_eq!(out, want, "len={len}");
        let mut wide = vec![0f32; len];
        auto::f16_to_f32(&a, &mut wide);
        let mut back = vec![0u16; len];
        auto::f32_to_f16(&wide, &mut back);
        assert_eq!(back, a, "len={len}");
    }
    let _ = Neon::assume_baseline();
}

// ---- f16 fused variants -----------------------------------------------------

type Fma16 = fn(&[u16], &[u16], &[u16], &mut [u16]);
type Fma16Hw = fn(Fp16Neon, &[u16], &[u16], &[u16], &mut [u16]);
type Fma16Form = (Fma16, Fma16Hw, bool, bool, &'static str);

/// `(auto_down fn, Fp16Neon slice fn, sign of the product, sign of the addend, name)`.
fn f16_fma_forms() -> [Fma16Form; 3] {
    [
        (
            auto_down::fmsub_f16,
            Fp16Neon::fmsub_f16_slice,
            false,
            true,
            "fmsub",
        ),
        (
            auto_down::fnmadd_f16,
            Fp16Neon::fnmadd_f16_slice,
            true,
            false,
            "fnmadd",
        ),
        (
            auto_down::fnmsub_f16,
            Fp16Neon::fnmsub_f16_slice,
            true,
            true,
            "fnmsub",
        ),
    ]
}

#[test]
fn f16_fma_variants_bottom_rung_round_once() {
    for (down, _, neg_p, neg_c, name) in f16_fma_forms() {
        let mut checked = 0;
        for i in 0..10_000u64 {
            let (a, b, c) = (h(3 * i), h(3 * i + 1), h(3 * i + 2));
            // Negating an operand is a sign-bit flip, so the oracle is fmadd's.
            let (oa, oc) = (
                if neg_p { a ^ 0x8000 } else { a },
                if neg_c { c ^ 0x8000 } else { c },
            );
            let Some(want) = fma_f16_exact(oa, b, oc) else {
                continue;
            };
            let mut got = [0u16];
            down(&[a], &[b], &[c], &mut got);
            assert!(
                same_f16(got[0], want),
                "{name}({a:#06x}, {b:#06x}, {c:#06x}) = {:#06x}, want {want:#06x}",
                got[0]
            );
            checked += 1;
        }
        assert!(checked > 5_000, "{name}");
    }
}

#[test]
fn f16_fma_variants_match_hardware_when_present() {
    let Some(t) = Fp16Neon::detect() else { return };
    let n = 3001;
    let a: Vec<u16> = (0..n).map(h).collect();
    let b: Vec<u16> = (0..n).map(|i| h(i + 7)).collect();
    let c: Vec<u16> = (0..n).map(|i| h(i + 13)).collect();
    for (down, hw, _, _, name) in f16_fma_forms() {
        let (mut x, mut y) = (vec![0; n as usize], vec![0; n as usize]);
        down(&a, &b, &c, &mut x);
        hw(t, &a, &b, &c, &mut y);
        for i in 0..n as usize {
            assert!(
                same_f16(x[i], y[i]),
                "{name}({:#06x}, {:#06x}, {:#06x}): {:#06x} vs {:#06x}",
                a[i],
                b[i],
                c[i],
                x[i],
                y[i]
            );
        }
    }
}

#[test]
fn f16_fma_variants_auto_signs() {
    let r = |x: f32| f16::from_f32(x).to_bits();
    let (a, b, c) = (vec![r(2.0); 9], vec![r(3.0); 9], vec![r(1.0); 9]);
    let mut out = vec![0u16; 9];
    auto::fmsub_f16(&a, &b, &c, &mut out);
    assert_eq!(out, vec![r(5.0); 9]);
    auto::fnmadd_f16(&a, &b, &c, &mut out);
    assert_eq!(out, vec![r(-5.0); 9]);
    auto::fnmsub_f16(&a, &b, &c, &mut out);
    assert_eq!(out, vec![r(-7.0); 9]);
}

// ---- float bitwise ops and clamp (x86 names) -----------------------------

/// Lengths around `CLAMP_CHUNK` (256) on top of the usual sweep.
const CHUNK_LENS: [usize; 4] = [255, 256, 257, 1000];

#[test]
fn x86_float_bitop_and_clamp_names_exist() {
    type B<T> = fn(&[T], &[T], &mut [T]);
    type F<T> = fn(&[T], &[T], &[T], &mut [T]);
    let _: [B<f32>; 4] = [auto::and_f32, auto::or_f32, auto::xor_f32, auto::andnot_f32];
    let _: [B<f64>; 4] = [auto::and_f64, auto::or_f64, auto::xor_f64, auto::andnot_f64];
    let _: [F<f32>; 1] = [auto::clamp_f32];
    let _: [F<f64>; 1] = [auto::clamp_f64];
    let _: [F<i32>; 1] = [auto::clamp_i32];
    let _: [F<u32>; 1] = [auto::clamp_u32];
    let _: [F<i64>; 1] = [auto::clamp_i64];
    let _: [F<u64>; 1] = [auto::clamp_u64];
    let _: [F<u16>; 3] = [auto::fmsub_f16, auto::fnmadd_f16, auto::fnmsub_f16];
}

#[test]
fn float_bitops_act_on_bit_patterns() {
    type Op32 = (fn(&[f32], &[f32], &mut [f32]), fn(u32, u32) -> u32);
    type Op64 = (fn(&[f64], &[f64], &mut [f64]), fn(u64, u64) -> u64);
    let ops32: [Op32; 4] = [
        (auto::and_f32, |x, y| x & y),
        (auto::or_f32, |x, y| x | y),
        (auto::xor_f32, |x, y| x ^ y),
        (auto::andnot_f32, |x, y| x & !y),
    ];
    let ops64: [Op64; 4] = [
        (auto::and_f64, |x, y| x & y),
        (auto::or_f64, |x, y| x | y),
        (auto::xor_f64, |x, y| x ^ y),
        (auto::andnot_f64, |x, y| x & !y),
    ];
    for len in LENS.into_iter().chain(CHUNK_LENS) {
        // Arbitrary bit patterns, NaNs and signed zeros included.
        let a: Vec<f32> = (0..len as u64)
            .map(|i| f32::from_bits(mix(i) as u32))
            .collect();
        let b: Vec<f32> = (0..len as u64)
            .map(|i| f32::from_bits(mix(i + 777) as u32))
            .collect();
        for (op, want) in ops32 {
            let mut out = vec![0f32; len];
            op(&a, &b, &mut out);
            let got: Vec<u32> = out.iter().map(|x| x.to_bits()).collect();
            let exp: Vec<u32> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| want(x.to_bits(), y.to_bits()))
                .collect();
            assert_eq!(got, exp, "f32 len={len}");
        }
        let a: Vec<f64> = (0..len as u64).map(|i| f64::from_bits(mix(i))).collect();
        let b: Vec<f64> = (0..len as u64)
            .map(|i| f64::from_bits(mix(i + 777)))
            .collect();
        for (op, want) in ops64 {
            let mut out = vec![0f64; len];
            op(&a, &b, &mut out);
            let got: Vec<u64> = out.iter().map(|x| x.to_bits()).collect();
            let exp: Vec<u64> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| want(x.to_bits(), y.to_bits()))
                .collect();
            assert_eq!(got, exp, "f64 len={len}");
        }
    }
    // andnot keeps the ARM order: clear the sign bit of -1.5 with -0.0 as mask.
    let mut out = [0f32];
    auto::andnot_f32(&[-1.5], &[-0.0], &mut out);
    assert_eq!(out, [1.5]);
}

macro_rules! clamp_case {
    ($fn_name:ident, $T:ty, $gen:expr) => {
        for len in LENS.into_iter().chain(CHUNK_LENS) {
            let g = $gen;
            let a: Vec<$T> = (0..len as u64).map(|i| g(i)).collect();
            let x: Vec<$T> = (0..len as u64).map(|i| g(i + 1000)).collect();
            let y: Vec<$T> = (0..len as u64).map(|i| g(i + 2000)).collect();
            // lo <= hi per lane, so the oracle is unambiguous.
            let lo: Vec<$T> = x
                .iter()
                .zip(&y)
                .map(|(&p, &q)| if p < q { p } else { q })
                .collect();
            let hi: Vec<$T> = x
                .iter()
                .zip(&y)
                .map(|(&p, &q)| if p < q { q } else { p })
                .collect();
            let mut out = vec![<$T>::default(); len];
            auto::$fn_name(&a, &lo, &hi, &mut out);
            let want: Vec<$T> = (0..len)
                .map(|i| {
                    let v = if a[i] < lo[i] { lo[i] } else { a[i] };
                    if v > hi[i] { hi[i] } else { v }
                })
                .collect();
            assert_eq!(out, want, "{} len={len}", stringify!($fn_name));
        }
    };
}

#[test]
fn clamp_matches_scalar_oracle() {
    clamp_case!(clamp_i32, i32, |i: u64| mix(i) as i32);
    clamp_case!(clamp_u32, u32, |i: u64| mix(i) as u32);
    clamp_case!(clamp_i64, i64, |i: u64| mix(i) as i64);
    clamp_case!(clamp_u64, u64, |i: u64| mix(i));
    // Finite floats only; NaN is the documented divergence from x86.
    clamp_case!(clamp_f32, f32, |i: u64| (mix(i) as i32 as f32) * 1e-6);
    clamp_case!(clamp_f64, f64, |i: u64| (mix(i) as i64 as f64) * 1e-12);
}
