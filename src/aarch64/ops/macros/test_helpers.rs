//! Test-only oracles for aarch64 token tests. Same idea as x86: `detect()`,
//! then SIMD vs scalar over `[0,1,3,4,5,7,8,9,15,16,17,100]`. Two differences:
//!
//! 1. **Harder data.** Integers are full-range with `MIN`/`MAX`/`0` forced
//!    in, so wrap/sat/sign actually fire. x86's `i + 1` ramp stays small.
//! 2. **Exact float.** Inputs are `k / 8` (small non-zero `k`), so sums and
//!    products are exact in any order. Reductions match a plain fold with no
//!    ulp slop. FMA equals the unfused oracle on these inputs; fusion itself
//!    is checked separately against `mul_add`.
//!
//! Every generated test opens with `let Some(t) = Token::detect() else {
//! return };`. On a CPU without the extension (QEMU `cortex-a53`) it passes
//! vacuously instead of executing an illegal instruction.

pub(crate) const LENS: [usize; 12] = [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100];

/// splitmix64: deterministic, full-period, no dependency.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Element types the helpers can generate and compare bit-exactly.
pub(crate) trait TestElem: Copy + core::fmt::Debug {
    /// Deterministic test value for position `i`.
    fn sample(i: usize) -> Self;
    /// Bit pattern, so `-0.0 != 0.0` and NaN payloads are compared exactly.
    fn bits(self) -> u64;
}

macro_rules! test_elem_int {
	($($T:ty => $U:ty),*) => {$(
		impl TestElem for $T {
			fn sample(i: usize) -> Self {
				match i % 16 {
					0 => <$T>::MIN,
					1 => <$T>::MAX,
					2 => 0,
					_ => mix(i as u64) as $T,
				}
			}
			// Zero-extended from the lane width, so `-1i8` and `255u8` agree.
			fn bits(self) -> u64 {
				self as $U as u64
			}
		}
	)*};
}
test_elem_int!(i8 => u8, u8 => u8, i16 => u16, u16 => u16, i32 => u32, u32 => u32, i64 => u64, u64 => u64);

macro_rules! test_elem_float {
	($($T:ty),*) => {$(
		impl TestElem for $T {
			fn sample(i: usize) -> Self {
				// k / 8 with k in -1000..=1000, never 0: exact in both f32 and
				// f64, products fit the f32 mantissa, and no 0/0 NaN.
				let k = (mix(i as u64) % 2001) as i64 - 1000;
				let k = if k == 0 { 1 } else { k };
				k as $T / 8.0
			}
			fn bits(self) -> u64 {
				self.to_bits() as u64
			}
		}
	)*};
}
test_elem_float!(f32, f64);

/// Signed shift count in `-(bits + 8)..=(bits + 8)`: covers left, right, zero
/// and both saturating ends of `SSHL`.
pub(crate) fn shift_count<T: TryFrom<i64>>(i: usize, bits: i64) -> T {
    let span = 2 * (bits + 8) + 1;
    let v = (mix(i as u64 ^ 0x5eed) % span as u64) as i64 - (bits + 8);
    T::try_from(v).ok().expect("shift count fits the lane type")
}

pub(crate) fn assert_bits_eq<T: TestElem>(got: &[T], expect: &[T], what: &str) {
    let g: Vec<u64> = got.iter().map(|x| x.bits()).collect();
    let e: Vec<u64> = expect.iter().map(|x| x.bits()).collect();
    assert_eq!(g, e, "{what}: got {got:?}, expected {expect:?}");
}

/// 2-op: fixed-width over 8 rounds, then (unless `fixed_only`) the slice sweep.
macro_rules! neon_binop_test {
	($test:ident, $Token:ty, $fixed:ident, $Elem:ty, $lanes:literal, gen_b = $gen_b:expr, $oracle:expr $(, slice = $slice:ident)?) => {
		#[test]
		fn $test() {
			use $crate::aarch64::ops::macros::{assert_bits_eq, TestElem};
			let Some(t) = <$Token>::detect() else { return };
			let op: fn($Elem, $Elem) -> $Elem = $oracle;
			let gb: fn(usize) -> $Elem = $gen_b;
			for r in 0..8usize {
				let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
				let b: [$Elem; $lanes] = core::array::from_fn(|i| gb(r * 97 + i + 50));
				let expect: [$Elem; $lanes] = core::array::from_fn(|i| op(a[i], b[i]));
				assert_bits_eq(&t.$fixed(a, b), &expect, stringify!($fixed));
			}
			$(
				for len in $crate::aarch64::ops::macros::LENS {
					let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
					let b: Vec<$Elem> = (0..len).map(|i| gb(i + 300)).collect();
					let mut out: Vec<$Elem> = a.clone();
					t.$slice(&a, &b, &mut out);
					let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
					assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
				}
			)?
		}
	};
}

/// 1-op, same shape as [`neon_binop_test`].
macro_rules! neon_unop_test {
	($test:ident, $Token:ty, $fixed:ident, $Elem:ty, $lanes:literal, $oracle:expr $(, slice = $slice:ident)?) => {
		#[test]
		fn $test() {
			use $crate::aarch64::ops::macros::{assert_bits_eq, TestElem};
			let Some(t) = <$Token>::detect() else { return };
			let op: fn($Elem) -> $Elem = $oracle;
			for r in 0..8usize {
				let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
				let expect: [$Elem; $lanes] = core::array::from_fn(|i| op(a[i]));
				assert_bits_eq(&t.$fixed(a), &expect, stringify!($fixed));
			}
			$(
				for len in $crate::aarch64::ops::macros::LENS {
					let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
					let mut out: Vec<$Elem> = a.clone();
					t.$slice(&a, &mut out);
					let expect: Vec<$Elem> = a.iter().map(|&x| op(x)).collect();
					assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
				}
			)?
		}
	};
}

/// Compare: `$Elem` in, `$Mask` lanes out. `b` shares half its lanes with `a`
/// so the equal / not-equal split is exercised, not just "never equal".
macro_rules! neon_cmp_test {
	($test:ident, $Token:ty, $fixed:ident, $Elem:ty, $Mask:ty, $lanes:literal, $oracle:expr $(, slice = $slice:ident)?) => {
		#[test]
		fn $test() {
			use $crate::aarch64::ops::macros::{assert_bits_eq, TestElem};
			let Some(t) = <$Token>::detect() else { return };
			let op: fn($Elem, $Elem) -> bool = $oracle;
			let mask = |p: bool| -> $Mask { if p { <$Mask>::MAX } else { 0 } };
			for r in 0..8usize {
				let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
				let b: [$Elem; $lanes] =
					core::array::from_fn(|i| if i % 2 == 0 { a[i] } else { <$Elem>::sample(r * 97 + i + 50) });
				let expect: [$Mask; $lanes] = core::array::from_fn(|i| mask(op(a[i], b[i])));
				assert_bits_eq(&t.$fixed(a, b), &expect, stringify!($fixed));
			}
			$(
				for len in $crate::aarch64::ops::macros::LENS {
					let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
					let b: Vec<$Elem> =
						(0..len).map(|i| if i % 3 == 0 { a[i] } else { <$Elem>::sample(i + 300) }).collect();
					let mut out: Vec<$Mask> = vec![0x5a as $Mask; len];
					t.$slice(&a, &b, &mut out);
					let expect: Vec<$Mask> = a.iter().zip(&b).map(|(&x, &y)| mask(op(x, y))).collect();
					assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
				}
			)?
		}
	};
}

/// Bit-select. Mask lanes are random bit patterns, not just all-ones/zero:
/// `BSL` is per **bit**, and that is the property worth pinning down.
macro_rules! neon_select_test {
	($test:ident, $Token:ty, $fixed:ident, $Elem:ty, $Mask:ty, $lanes:literal $(, slice = $slice:ident)?) => {
		#[test]
		fn $test() {
			use $crate::aarch64::ops::macros::TestElem;
			let Some(t) = <$Token>::detect() else { return };
			// Built from bits, so a float `$Elem` is compared by pattern too.
			let sel = |m: $Mask, x: $Elem, y: $Elem| -> u64 {
				let m = m as u64;
				((m & x.bits()) | (!m & y.bits())) & (<$Mask>::MAX as u64)
			};
			for r in 0..8usize {
				let m: [$Mask; $lanes] = core::array::from_fn(|i| <$Mask>::sample(r * 97 + i + 11));
				let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
				let b: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i + 50));
				let got = t.$fixed(m, a, b);
				for i in 0..$lanes {
					assert_eq!(got[i].bits(), sel(m[i], a[i], b[i]), "{} lane {i}", stringify!($fixed));
				}
			}
			$(
				for len in $crate::aarch64::ops::macros::LENS {
					let m: Vec<$Mask> = (0..len).map(|i| <$Mask>::sample(i + 3)).collect();
					let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
					let b: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 300)).collect();
					let mut out: Vec<$Elem> = a.clone();
					t.$slice(&m, &a, &b, &mut out);
					for i in 0..len {
						assert_eq!(out[i].bits(), sel(m[i], a[i], b[i]), "{} len={len} i={i}", stringify!($slice));
					}
				}
			)?
		}
	};
}

/// Fused multiply-add family (`fmadd`/`fmsub`/`fnmadd`/`fnmsub`). Two
/// checks: the `k/8` data against the `fused` oracle (exact, so it also
/// equals `plain` and the slice tail is comparable), and a fusion probe where
/// one rounding and two roundings give different bits.
macro_rules! neon_fma_test {
	($test:ident, $Token:ty, $fixed:ident, $Elem:ty, $lanes:literal, fused = $fused:expr, plain = $plain:expr $(, slice = $slice:ident)?) => {
		#[test]
		fn $test() {
			use $crate::aarch64::ops::macros::{assert_bits_eq, TestElem};
			let Some(t) = <$Token>::detect() else { return };
			let fused: fn($Elem, $Elem, $Elem) -> $Elem = $fused;
			let plain: fn($Elem, $Elem, $Elem) -> $Elem = $plain;
			for r in 0..8usize {
				let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
				let b: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i + 50));
				let c: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i + 90));
				let expect: [$Elem; $lanes] = core::array::from_fn(|i| fused(a[i], b[i], c[i]));
				assert_bits_eq(&t.$fixed(a, b, c), &expect, stringify!($fixed));
			}
			// (1 + e)(1 - e) = 1 - e^2: with c = ±1 chosen so the unfused
			// result cancels to 0, only a fused op keeps the e^2.
			let e = <$Elem>::EPSILON;
			let c1 = if plain(1.0 + e, 1.0 - e, 1.0) == 0.0 { 1.0 } else { -1.0 };
			let got = t.$fixed([1.0 + e; $lanes], [1.0 - e; $lanes], [c1; $lanes]);
			assert_ne!(got[0], 0.0, "{} did not fuse", stringify!($fixed));
			assert_bits_eq(&got, &[fused(1.0 + e, 1.0 - e, c1); $lanes], "fusion probe");
			$(
				for len in $crate::aarch64::ops::macros::LENS {
					let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
					let b: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 300)).collect();
					let c: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 600)).collect();
					let mut out: Vec<$Elem> = a.clone();
					t.$slice(&a, &b, &c, &mut out);
					let expect: Vec<$Elem> = (0..len).map(|i| plain(a[i], b[i], c[i])).collect();
					assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
				}
			)?
		}
	};
}

/// Immediate shift at one `IMM`.
macro_rules! neon_shift_imm_test {
    ($test:ident, $Token:ty, $fixed:ident, $slice:ident, $Elem:ty, $lanes:literal, $imm:literal, $oracle:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{TestElem, assert_bits_eq};
            let Some(t) = <$Token>::detect() else { return };
            let op: fn($Elem) -> $Elem = $oracle;
            for r in 0..8usize {
                let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
                let expect: [$Elem; $lanes] = core::array::from_fn(|i| op(a[i]));
                assert_bits_eq(&t.$fixed::<$imm>(a), &expect, stringify!($fixed));
            }
            for len in $crate::aarch64::ops::macros::LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let mut out: Vec<$Elem> = a.clone();
                t.$slice::<$imm>(&a, &mut out);
                let expect: Vec<$Elem> = a.iter().map(|&x| op(x)).collect();
                assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
            }
        }
    };
}

/// Across-lane reduction, fixed and slice, against a plain fold.
macro_rules! neon_reduce_test {
    ($test:ident, $Token:ty, $fixed:ident, $slice:ident, $Elem:ty, $lanes:literal, init = $init:expr, $fold:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::TestElem;
            let Some(t) = <$Token>::detect() else { return };
            let f: fn($Elem, $Elem) -> $Elem = $fold;
            for r in 0..8usize {
                let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
                let expect = a.iter().copied().fold($init, f);
                assert_eq!(
                    t.$fixed(a).bits(),
                    expect.bits(),
                    "{} round {r}",
                    stringify!($fixed)
                );
            }
            for len in $crate::aarch64::ops::macros::LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let expect = a.iter().copied().fold($init, f);
                assert_eq!(
                    t.$slice(&a).bits(),
                    expect.bits(),
                    "{} len={len}",
                    stringify!($slice)
                );
            }
        }
    };
}

/// Lane-preserving conversion. `$gen` supplies source values (the float
/// sources include out-of-range and NaN lanes to pin down saturation).
macro_rules! neon_cvt_test {
    ($test:ident, $Token:ty, $fixed:ident, $slice:ident, $Src:ty, $Dst:ty, $lanes:literal, gen = $gen:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::assert_bits_eq;
            let Some(t) = <$Token>::detect() else { return };
            let g: fn(usize) -> $Src = $gen;
            for r in 0..8usize {
                let a: [$Src; $lanes] = core::array::from_fn(|i| g(r * 97 + i));
                let expect: [$Dst; $lanes] = core::array::from_fn(|i| a[i] as $Dst);
                assert_bits_eq(&t.$fixed(a), &expect, stringify!($fixed));
            }
            for len in $crate::aarch64::ops::macros::LENS {
                let a: Vec<$Src> = (0..len).map(|i| g(i + 7)).collect();
                let mut out: Vec<$Dst> = vec![Default::default(); len];
                t.$slice(&a, &mut out);
                let expect: Vec<$Dst> = a.iter().map(|&x| x as $Dst).collect();
                assert_bits_eq(&out, &expect, &format!("{} len={len}", stringify!($slice)));
            }
        }
    };
}

pub(crate) use {
    neon_binop_test, neon_cmp_test, neon_cvt_test, neon_fma_test, neon_reduce_test,
    neon_select_test, neon_shift_imm_test, neon_unop_test,
};

// ---- SVE: slice-first test macros
//
// SVE ops have no Neon-style fixed-width twin on the base token, so these
// test the `_slice` form alone, over lengths that cross many whole vectors
// even at VL 2048 (256 `u8` lanes). Every output buffer carries a canary
// past `out.len()`: a predicate bug that stores one vector too far shows up
// as a clobbered canary rather than silent heap damage.

/// `LENS` plus lengths around 64/256 lanes and one long run.
pub(crate) const SVE_LENS: [usize; 19] = [
    0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 100, 255, 256, 257, 1000,
];

/// Canary lanes appended past every `out`.
pub(crate) const CANARY: usize = 300;

/// `len` outputs followed by `CANARY` copies of `fill`; returns the buffer.
pub(crate) fn with_canary<T: Copy>(len: usize, fill: T) -> Vec<T> {
    vec![fill; len + CANARY]
}

/// Asserts the canary tail is still `fill`, bit for bit.
pub(crate) fn assert_canary<T: TestElem>(buf: &[T], len: usize, fill: T, what: &str) {
    assert!(
        buf[len..].iter().all(|x| x.bits() == fill.bits()),
        "{what} len={len}: wrote past the end of `out`"
    );
}

macro_rules! sve_binop_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, gen_b = $gen_b:expr, $oracle:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let op: fn($Elem, $Elem) -> $Elem = $oracle;
            let gb: fn(usize) -> $Elem = $gen_b;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let b: Vec<$Elem> = (0..len).map(|i| gb(i + 300)).collect();
                let fill = <$Elem>::sample(5);
                let mut buf = with_canary(len, fill);
                t.$slice(&a, &b, &mut buf[..len]);
                let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, fill, stringify!($slice));
            }
        }
    };
}

macro_rules! sve_unop_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, gen = $gen:expr, $oracle:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let op: fn($Elem) -> $Elem = $oracle;
            let g: fn(usize) -> $Elem = $gen;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| g(i + 7)).collect();
                let fill = <$Elem>::sample(5);
                let mut buf = with_canary(len, fill);
                t.$slice(&a, &mut buf[..len]);
                let expect: Vec<$Elem> = a.iter().map(|&x| op(x)).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, fill, stringify!($slice));
            }
        }
    };
}

macro_rules! sve_cmp_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, $Mask:ty, $oracle:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let op: fn($Elem, $Elem) -> bool = $oracle;
            let mask = |p: bool| -> $Mask { if p { <$Mask>::MAX } else { 0 } };
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let b: Vec<$Elem> = (0..len)
                    .map(|i| {
                        if i % 3 == 0 {
                            a[i]
                        } else {
                            <$Elem>::sample(i + 300)
                        }
                    })
                    .collect();
                let mut buf: Vec<$Mask> = with_canary(len, 0x5a as $Mask);
                t.$slice(&a, &b, &mut buf[..len]);
                let expect: Vec<$Mask> = a.iter().zip(&b).map(|(&x, &y)| mask(op(x, y))).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, 0x5a as $Mask, stringify!($slice));
            }
        }
    };
}

macro_rules! sve_select_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, $Mask:ty) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{TestElem, assert_canary, with_canary};
            let Some(t) = <$Token>::detect() else { return };
            let sel = |m: $Mask, x: $Elem, y: $Elem| -> u64 {
                let m = m as u64;
                ((m & x.bits()) | (!m & y.bits())) & (<$Mask>::MAX as u64)
            };
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let m: Vec<$Mask> = (0..len).map(|i| <$Mask>::sample(i + 3)).collect();
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let b: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 300)).collect();
                let fill = <$Elem>::sample(5);
                let mut buf = with_canary(len, fill);
                t.$slice(&m, &a, &b, &mut buf[..len]);
                for i in 0..len {
                    assert_eq!(
                        buf[i].bits(),
                        sel(m[i], a[i], b[i]),
                        "{} len={len} i={i}",
                        stringify!($slice)
                    );
                }
                assert_canary(&buf, len, fill, stringify!($slice));
            }
        }
    };
}

/// Fused in every lane (no scalar tail on SVE), so the oracle is `mul_add`
/// throughout, plus the same one-rounding probe as `neon_fma_test`.
macro_rules! sve_fma_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, $fused:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let fused: fn($Elem, $Elem, $Elem) -> $Elem = $fused;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let b: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 300)).collect();
                let c: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 600)).collect();
                let fill = <$Elem>::sample(5);
                let mut buf = with_canary(len, fill);
                t.$slice(&a, &b, &c, &mut buf[..len]);
                let expect: Vec<$Elem> = (0..len).map(|i| fused(a[i], b[i], c[i])).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, fill, stringify!($slice));
            }
            // (1 + e)(1 - e) + c: only one rounding keeps the e^2, in every
            // lane including the last partial vector.
            let e = <$Elem>::EPSILON;
            for c1 in [1.0, -1.0] {
                let n = 37;
                let (a, b, c) = (vec![1.0 + e; n], vec![1.0 - e; n], vec![c1; n]);
                let mut out = vec![0.0; n];
                t.$slice(&a, &b, &c, &mut out);
                let want = fused(1.0 + e, 1.0 - e, c1);
                assert_bits_eq(
                    &out,
                    &vec![want; n],
                    concat!(stringify!($slice), " fusion probe"),
                );
            }
        }
    };
}

macro_rules! sve_shift_imm_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, $imm:literal, $oracle:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let op: fn($Elem) -> $Elem = $oracle;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let fill = <$Elem>::sample(5);
                let mut buf = with_canary(len, fill);
                t.$slice::<$imm>(&a, &mut buf[..len]);
                let expect: Vec<$Elem> = a.iter().map(|&x| op(x)).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, fill, stringify!($slice));
            }
        }
    };
}

macro_rules! sve_reduce_test {
    ($test:ident, $Token:ty, $slice:ident, $Elem:ty, init = $init:expr, $fold:expr) => {
        #[test]
        fn $test() {
            use $crate::aarch64::ops::macros::TestElem;
            let Some(t) = <$Token>::detect() else { return };
            let f: fn($Elem, $Elem) -> $Elem = $fold;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Elem> = (0..len).map(|i| <$Elem>::sample(i + 7)).collect();
                let expect = a.iter().copied().fold($init, f);
                assert_eq!(
                    t.$slice(&a).bits(),
                    expect.bits(),
                    "{} len={len}",
                    stringify!($slice)
                );
            }
        }
    };
}

macro_rules! sve_cvt_test {
    ($test:ident, $Token:ty, $slice:ident, $Src:ty, $Dst:ty, gen = $gen:expr) => {
        #[test]
        fn $test() {
            #[allow(unused_imports)]
            use $crate::aarch64::ops::macros::{
                TestElem, assert_bits_eq, assert_canary, with_canary,
            };
            let Some(t) = <$Token>::detect() else { return };
            let g: fn(usize) -> $Src = $gen;
            for len in $crate::aarch64::ops::macros::SVE_LENS {
                let a: Vec<$Src> = (0..len).map(|i| g(i + 7)).collect();
                let fill: $Dst = 0x5a as $Dst;
                let mut buf = with_canary(len, fill);
                t.$slice(&a, &mut buf[..len]);
                let expect: Vec<$Dst> = a.iter().map(|&x| x as $Dst).collect();
                assert_bits_eq(
                    &buf[..len],
                    &expect,
                    &format!("{} len={len}", stringify!($slice)),
                );
                assert_canary(&buf, len, fill, stringify!($slice));
            }
        }
    };
}

pub(crate) use {
    sve_binop_test, sve_cmp_test, sve_cvt_test, sve_fma_test, sve_reduce_test, sve_select_test,
    sve_shift_imm_test, sve_unop_test,
};
