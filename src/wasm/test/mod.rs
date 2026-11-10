//! wasm SIMD tests. `auto.rs` (generated) holds one test per op; this file
//! holds the shared oracles plus the hand-written relaxed and token-only
//! checks.
//!
//! **Rungs.** The runner (`scripts/wasm-runner.sh`) gives the engine simd128
//! and relaxed-simd, so every test can switch rungs at run time with the
//! pulp-style flags: scalar (`disable_simd128`), `Simd128`, and
//! `RelaxedSimd` when its code is compiled in. In a `+simd128` build the
//! flag cannot switch simd128 off, so "scalar" then runs `Simd128` again.
//! libtest runs tests one after another on wasm (no threads), so the global
//! flags do not race.
//!
//! **Data.** Integers are full-range with `MIN`/`MAX`/`0` forced in. Floats
//! are `k / 8` (exact sums and products, so fused and unfused FMA agree)
//! plus NaN, `±0.0` and `±inf`. NaN results compare as NaN, not by payload:
//! wasm leaves the payload open.
//!
//! wasm has `panic = "abort"`, so the length-mismatch panics are not tested
//! here (a `#[should_panic]` test would abort the whole run).

use crate::wasm::ops::simd128::Simd128;
use crate::wasm::{disable_simd128, enable_simd128};

mod auto;

const LENS: [usize; 14] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 100];

/// splitmix64: deterministic, full-period, no dependency.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

trait TestElem: Copy + core::fmt::Debug + Default {
    fn sample(i: usize) -> Self;
    /// Bit-equal, or both NaN.
    fn same(self, other: Self) -> bool;
}

macro_rules! test_elem_int {
	($($T:ty),*) => {$(
		impl TestElem for $T {
			fn sample(i: usize) -> Self {
				match i % 16 {
					0 => <$T>::MIN,
					1 => <$T>::MAX,
					2 => 0,
					_ => mix(i as u64) as $T,
				}
			}
			fn same(self, other: Self) -> bool {
				self == other
			}
		}
	)*};
}
test_elem_int!(i8, u8, i16, u16, i32, u32, i64, u64);

macro_rules! test_elem_float {
	($($T:ty),*) => {$(
		impl TestElem for $T {
			fn sample(i: usize) -> Self {
				match i % 23 {
					0 => <$T>::NAN,
					1 => -0.0,
					2 => 0.0,
					3 => <$T>::INFINITY,
					4 => <$T>::NEG_INFINITY,
					_ => {
						// k / 8, k in -1000..=1000 without 0: exact in f32 and f64,
						// products fit the f32 mantissa.
						let k = (mix(i as u64) % 2001) as i64 - 1000;
						let k = if k == 0 { 1 } else { k };
						k as $T / 8.0
					}
				}
			}
			fn same(self, other: Self) -> bool {
				(self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
			}
		}
	)*};
}
test_elem_float!(f32, f64);

fn gen_vec<T: TestElem>(len: usize, seed: usize) -> Vec<T> {
    (0..len).map(|i| T::sample(i + seed)).collect()
}

fn gen_arr<T: TestElem, const N: usize>(seed: usize) -> [T; N] {
    core::array::from_fn(|i| T::sample(i + seed))
}

fn assert_same<T: TestElem>(got: &[T], expect: &[T], what: &str) {
    let ok = got.len() == expect.len() && got.iter().zip(expect).all(|(&g, &e)| g.same(e));
    assert!(ok, "{what}: got {got:?}, expected {expect:?}");
}

/// Run `f` once per rung the build can reach, switching with the runtime
/// flags. Leaves simd128 enabled (and relaxed-simd as the build set it).
fn rungs(mut f: impl FnMut(&'static str)) {
    disable_simd128();
    f("scalar");
    enable_simd128();
    f("simd128");
    #[cfg(any(feature = "wasm-relaxed-simd", target_feature = "relaxed-simd"))]
    {
        crate::wasm::enable_relaxed_simd();
        f("relaxed");
        crate::wasm::disable_relaxed_simd();
    }
}

fn simd128() -> Simd128 {
    enable_simd128();
    Simd128::detect().expect("enable_simd128 makes Simd128 detectable")
}

fn map_un<T: Copy>(a: &[T], out: &mut [T], f: fn(T) -> T) {
    for (o, &x) in out.iter_mut().zip(a) {
        *o = f(x);
    }
}

// ------------------------------------------------------------ slice oracles

fn check_bin<T: TestElem, O: TestElem>(
    auto: impl Fn(&[T], &[T], &mut [O]),
    down: impl Fn(&[T], &[T], &mut [O]),
    what: &str,
) {
    rungs(|rung| {
        for len in LENS {
            let (a, b) = (gen_vec::<T>(len, 7), gen_vec::<T>(len, 300));
            let mut expect = vec![O::default(); len];
            down(&a, &b, &mut expect);
            let mut got = vec![O::default(); len];
            auto(&a, &b, &mut got);
            assert_same(&got, &expect, &format!("{what} [{rung}] len={len}"));
        }
    });
}

fn check_un<T: TestElem, O: TestElem>(
    auto: impl Fn(&[T], &mut [O]),
    down: impl Fn(&[T], &mut [O]),
    what: &str,
) {
    rungs(|rung| {
        for len in LENS {
            let a = gen_vec::<T>(len, 11);
            let mut expect = vec![O::default(); len];
            down(&a, &mut expect);
            let mut got = vec![O::default(); len];
            auto(&a, &mut got);
            assert_same(&got, &expect, &format!("{what} [{rung}] len={len}"));
        }
    });
}

fn check_sel<T: TestElem, M: TestElem>(
    auto: impl Fn(&[T], &[T], &[M], &mut [T]),
    down: impl Fn(&[T], &[T], &[M], &mut [T]),
    what: &str,
) {
    rungs(|rung| {
        for len in LENS {
            let (a, b, m) = (
                gen_vec::<T>(len, 7),
                gen_vec::<T>(len, 300),
                gen_vec::<M>(len, 41),
            );
            let mut expect = vec![T::default(); len];
            down(&a, &b, &m, &mut expect);
            let mut got = vec![T::default(); len];
            auto(&a, &b, &m, &mut got);
            assert_same(&got, &expect, &format!("{what} [{rung}] len={len}"));
        }
    });
}

fn check_tern<T: TestElem>(
    auto: impl Fn(&[T], &[T], &[T], &mut [T]),
    down: impl Fn(&[T], &[T], &[T], &mut [T]),
    what: &str,
) {
    rungs(|rung| {
        for len in LENS {
            let (a, b, c) = (
                gen_vec::<T>(len, 7),
                gen_vec::<T>(len, 300),
                gen_vec::<T>(len, 555),
            );
            let mut expect = vec![T::default(); len];
            down(&a, &b, &c, &mut expect);
            let mut got = vec![T::default(); len];
            auto(&a, &b, &c, &mut got);
            assert_same(&got, &expect, &format!("{what} [{rung}] len={len}"));
        }
    });
}

fn check_sum<T: TestElem>(auto: impl Fn(&[T]) -> T, down: impl Fn(&[T]) -> T, what: &str) {
    rungs(|rung| {
        for len in LENS {
            // Seed 5 keeps NaN/inf out of the short lengths, so most sums are
            // finite and the grouping is actually compared.
            let a = gen_vec::<T>(len, 5);
            assert_same(
                &[auto(&a)],
                &[down(&a)],
                &format!("{what} [{rung}] len={len}"),
            );
        }
    });
}

// ------------------------------------------------------ fixed-width oracles

const ROUNDS: usize = 16;

fn check_fixed_bin<T: TestElem, O: TestElem, const N: usize>(
    f: impl Fn([T; N], [T; N]) -> [O; N],
    down: impl Fn(&[T], &[T], &mut [O]),
    what: &str,
) {
    for r in 0..ROUNDS {
        let (a, b) = (gen_arr::<T, N>(r * 97), gen_arr::<T, N>(r * 97 + 50));
        let mut expect = [O::default(); N];
        down(&a, &b, &mut expect);
        assert_same(&f(a, b), &expect, &format!("{what} round {r}"));
    }
}

fn check_fixed_un<T: TestElem, O: TestElem, const N: usize>(
    f: impl Fn([T; N]) -> [O; N],
    down: impl Fn(&[T], &mut [O]),
    what: &str,
) {
    for r in 0..ROUNDS {
        let a = gen_arr::<T, N>(r * 97);
        let mut expect = [O::default(); N];
        down(&a, &mut expect);
        assert_same(&f(a), &expect, &format!("{what} round {r}"));
    }
}

fn check_fixed_sel<T: TestElem, M: TestElem, const N: usize>(
    f: impl Fn([T; N], [T; N], [M; N]) -> [T; N],
    down: impl Fn(&[T], &[T], &[M], &mut [T]),
    what: &str,
) {
    for r in 0..ROUNDS {
        let (a, b, m) = (
            gen_arr::<T, N>(r * 97),
            gen_arr::<T, N>(r * 97 + 50),
            gen_arr::<M, N>(r * 31),
        );
        let mut expect = [T::default(); N];
        down(&a, &b, &m, &mut expect);
        assert_same(&f(a, b, m), &expect, &format!("{what} round {r}"));
    }
}

fn check_fixed_tern<T: TestElem, const N: usize>(
    f: impl Fn([T; N], [T; N], [T; N]) -> [T; N],
    down: impl Fn(&[T], &[T], &[T], &mut [T]),
    what: &str,
) {
    for r in 0..ROUNDS {
        let (a, b, c) = (
            gen_arr::<T, N>(r * 97),
            gen_arr::<T, N>(r * 97 + 50),
            gen_arr::<T, N>(r * 13),
        );
        let mut expect = [T::default(); N];
        down(&a, &b, &c, &mut expect);
        assert_same(&f(a, b, c), &expect, &format!("{what} round {r}"));
    }
}

// --------------------------------------------------------- hand-written

/// The float min/max contract on the edge cases `TestElem` mixes in only
/// now and then: NaN wins, `-0.0 < +0.0`.
#[test]
fn float_min_max_edge_cases() {
    use crate::wasm::auto;
    let a = [f32::NAN, 1.0, -0.0, 0.0, -0.0, f32::NEG_INFINITY];
    let b = [1.0, f32::NAN, 0.0, -0.0, -0.0, 3.0];
    rungs(|rung| {
        let mut mn = [0f32; 6];
        let mut mx = [0f32; 6];
        auto::min_f32(&a, &b, &mut mn);
        auto::max_f32(&a, &b, &mut mx);
        assert!(
            mn[0].is_nan() && mn[1].is_nan() && mx[0].is_nan() && mx[1].is_nan(),
            "[{rung}] NaN wins"
        );
        assert_eq!(
            mn[2..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [(-0f32).to_bits(); 3]
                .iter()
                .copied()
                .chain([f32::NEG_INFINITY.to_bits()])
                .collect::<Vec<_>>(),
            "[{rung}] min"
        );
        assert_eq!(mx[2].to_bits(), 0f32.to_bits(), "[{rung}] max(-0, +0)");
        assert_eq!(mx[3].to_bits(), 0f32.to_bits(), "[{rung}] max(+0, -0)");
        assert_eq!(mx[4].to_bits(), (-0f32).to_bits(), "[{rung}] max(-0, -0)");
        assert_eq!(mx[5], 3.0, "[{rung}] max(-inf, 3)");
    });
}

/// `TestElem` floats are exact, so any summation order agrees there. On
/// inexact data the lane grouping shows, and every rung must still match
/// `auto_down` bit for bit. The plain fold must differ somewhere, or this
/// test would not see a wrong grouping either.
#[test]
fn float_sum_grouping_on_inexact_data() {
    use crate::wasm::{auto, auto_down};
    let a32: Vec<f32> = (0..103).map(|i| 0.1 + i as f32 * 1.37e-3).collect();
    let a64: Vec<f64> = (0..103).map(|i| 0.1 + i as f64 * 1.37e-11).collect();
    let (mut fold_differs32, mut fold_differs64) = (false, false);
    rungs(|rung| {
        for len in 0..a32.len() {
            let (x, y) = (&a32[..len], &a64[..len]);
            assert_eq!(
                auto::sum_f32(x).to_bits(),
                auto_down::sum_f32(x).to_bits(),
                "sum_f32 [{rung}] len={len}"
            );
            assert_eq!(
                auto::sum_f64(y).to_bits(),
                auto_down::sum_f64(y).to_bits(),
                "sum_f64 [{rung}] len={len}"
            );
            fold_differs32 |= x.iter().fold(-0.0f32, |s, &v| s + v) != auto_down::sum_f32(x);
            fold_differs64 |= y.iter().fold(-0.0f64, |s, &v| s + v) != auto_down::sum_f64(y);
        }
    });
    assert!(
        fold_differs32 && fold_differs64,
        "data too exact to pin the grouping"
    );
}

/// The flags really switch the token: off gives `None`, on gives `Some`
/// (unless the build has `+simd128`, where off is impossible).
#[test]
fn flags_gate_the_token() {
    disable_simd128();
    assert_eq!(
        Simd128::detect().is_some(),
        cfg!(target_feature = "simd128")
    );
    enable_simd128();
    assert!(Simd128::detect().is_some());
}

#[test]
fn simd128_token_only_ops() {
    let t = simd128();
    let a: [u8; 16] = core::array::from_fn(|i| i as u8 * 3);
    let idx: [u8; 16] = core::array::from_fn(|i| {
        if i % 5 == 4 {
            16 + i as u8 * 9
        } else {
            15 - i as u8
        }
    });
    let expect: [u8; 16] =
        core::array::from_fn(|i| if idx[i] < 16 { a[idx[i] as usize] } else { 0 });
    assert_eq!(t.swizzle_u8x16(a, idx), expect);

    let x: [i16; 8] = [1, -2, 300, 4, i16::MIN, i16::MIN, 7, -8];
    let y: [i16; 8] = [5, 6, -7, 8, i16::MIN, i16::MIN, i16::MAX, 2];
    let dot: [i32; 4] = core::array::from_fn(|i| {
        (x[2 * i] as i32 * y[2 * i] as i32).wrapping_add(x[2 * i + 1] as i32 * y[2 * i + 1] as i32)
    });
    assert_eq!(t.dot_i16x8(x, y), dot);

    let q: [i16; 8] = core::array::from_fn(|i| {
        (((x[i] as i32 * y[i] as i32) + 0x4000) >> 15).clamp(i16::MIN as i32, i16::MAX as i32)
            as i16
    });
    assert_eq!(t.q15mulr_sat_i16x8(x, y), q);
}

#[cfg(any(feature = "wasm-relaxed-simd", target_feature = "relaxed-simd"))]
mod relaxed {
    use crate::wasm::ops::relaxed::RelaxedSimd;

    fn token() -> RelaxedSimd {
        crate::wasm::enable_relaxed_simd();
        RelaxedSimd::detect().expect("enable_relaxed_simd makes RelaxedSimd detectable")
    }

    /// On inexact data the engine may fuse or not: each lane must be one of
    /// the two allowed results.
    #[test]
    fn fmadd_is_fused_or_unfused() {
        let t = token();
        let a: Vec<f32> = (0..37).map(|i| 1.0 + i as f32 * 1.0e-7).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 3.0e-7).collect();
        let c: Vec<f32> = (0..37).map(|i| -1.0 + i as f32 * 1.0e-8).collect();
        let mut out = vec![0f32; 37];
        t.fmadd_f32_slice(&a, &b, &c, &mut out);
        for i in 0..37 {
            let fused = a[i].mul_add(b[i], c[i]);
            let unfused = a[i] * b[i] + c[i];
            assert!(
                out[i] == fused || out[i] == unfused,
                "lane {i}: {} not in {{{fused}, {unfused}}}",
                out[i]
            );
        }
        let x = [1.0f64 + 1e-15, 3.0];
        let y = [1.0f64 - 1e-15, 0.5];
        let z = [-1.0f64, 0.25];
        let r = t.fmadd_f64x2(x, y, z);
        for i in 0..2 {
            assert!(
                r[i] == x[i].mul_add(y[i], z[i]) || r[i] == x[i] * y[i] + z[i],
                "f64 lane {i}"
            );
        }
    }

    /// Relaxed ops on inputs where every engine must agree.
    #[test]
    fn relaxed_token_only_ops_on_defined_inputs() {
        let t = token();
        assert_eq!(
            t.relaxed_min_f32x4([1.0, -2.0, 3.0, 4.0], [0.5, 2.0, 3.5, -4.0]),
            [0.5, -2.0, 3.0, -4.0]
        );
        assert_eq!(
            t.relaxed_max_f32x4([1.0, -2.0, 3.0, 4.0], [0.5, 2.0, 3.5, -4.0]),
            [1.0, 2.0, 3.5, 4.0]
        );

        let a: [u8; 16] = core::array::from_fn(|i| i as u8);
        let b: [u8; 16] = core::array::from_fn(|i| 100 + i as u8);
        let m: [u8; 16] = core::array::from_fn(|i| if i % 3 == 0 { 0xff } else { 0 });
        let sel: [u8; 16] = core::array::from_fn(|i| if m[i] != 0 { a[i] } else { b[i] });
        assert_eq!(t.relaxed_laneselect_u8x16(a, b, m), sel);

        let idx: [u8; 16] = core::array::from_fn(|i| 15 - i as u8);
        assert_eq!(
            t.relaxed_swizzle_u8x16(b, idx),
            core::array::from_fn(|i| b[15 - i])
        );

        assert_eq!(
            t.relaxed_trunc_f32x4([1.9, -1.9, 1e6, -0.5]),
            [1, -1, 1_000_000, 0]
        );

        let x: [i16; 8] = [1, -2, 300, 4, 16384, -16384, 7, -8];
        let y: [i16; 8] = [5, 6, -7, 8, 16384, 16384, i16::MAX, 2];
        let q: [i16; 8] =
            core::array::from_fn(|i| (((x[i] as i32 * y[i] as i32) + 0x4000) >> 15) as i16);
        assert_eq!(t.relaxed_q15mulr_i16x8(x, y), q);

        let p: [i8; 16] = core::array::from_fn(|i| (i as i8 - 8) * 15);
        let s: [i8; 16] = core::array::from_fn(|i| (i as i8) * 8);
        let dot: [i16; 8] = core::array::from_fn(|i| {
            p[2 * i] as i16 * s[2 * i] as i16 + p[2 * i + 1] as i16 * s[2 * i + 1] as i16
        });
        assert_eq!(t.relaxed_dot_i8x16_i7x16(p, s), dot);
    }
}
