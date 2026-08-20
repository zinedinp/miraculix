//! # Auto slice ops (aarch32)
//!
//! Elementwise slice work without picking a token by hand. Each fn uses
//! [`super::detect_features`] (cached), picks the best token, and falls
//! back to scalar. Pure scalar lives in private `auto_down`. Also as
//! [`super::auto`] (alias; same naming as [`crate::x86::auto_up`]).
//!
//! ```
//! # #[cfg(all(target_arch = "arm", feature = "nightly-arm-neon"))] {
//! use miraculix::aarch32::auto_up;
//! let a = [1i32, 2, 3, 4];
//! let b = [10, 20, 30, 40];
//! let mut out = [0i32; 4];
//! auto_up::add_i32(&a, &b, &mut out);
//! assert_eq!(out, [11, 22, 33, 44]);
//! # }
//! ```
//!
//! ## Scope
//!
//! One hardware rung per family so far (`Neon`+`Vfpv4Neon` for `i32`/`f32`/
//! `u32`, `Fp16Neon`+`Fp16Fma` for `f16` bit patterns, `Dsp` for `i8`/`i16`):
//! each fn is a 2-rung "token -> portable scalar" cascade. Breadth is meant
//! to track x86's `auto_up.rs` for the equivalent instruction classes -
//! `add`/`sub`/`mul`/`min`/`max`/`abs`/`neg`/`and`/`or`/`xor`/`andnot`/5-way
//! `cmp`/`select`/`fmadd`, plus DSP's saturating/halving/wrapping/cross
//! add-subtract families. Ops without independent-lane slice shape
//! (`smulbb`, `smuad`, `usada8`, `crc32*`, AES/SHA, Dotprod's `dot_*`,
//! I8mm's `dot_us32`/`mmla_*`, etc.) stay token-only (same "tier-unique
//! stays off `auto`" policy as x86).
//!
//! ## Shared contract
//!
//! - Lengths of inputs and `out` must match. Mismatch **panics**.
//! - `qadd`/`qsub` (Dsp) are **saturating**; `sadd`/`ssub`/`shadd`/`shsub`
//!   are **wrapping**/**halving** (no saturation); `i32`/`f32`/`u32`/`f16`
//!   (Neon/`Fp16Neon`) arithmetic is wrapping/IEEE as usual.
//! - Compare functions (`cmp*_i32`/`f32`/`f16`) always produce a lane mask
//!   typed `u32`/`u16` (all-1s or 0), **regardless of the input element
//!   type** - unlike x86, where the mask is typed like the input. `select_*`
//!   takes that same `u32` mask type for all three of `select_i32`/`u32`/
//!   `f32`.
//! - `fmadd_f32`/`fmadd_f16` are HW-fused when a token is available; the
//!   scalar remainder is **not** fused (`a*b+c` computed directly),
//!   matching x86's `fmadd_f32` convention.
//! - `qasx`/`qsax`/`sasx_i16` operate on whole `[i16; 2]` pairs (cross
//!   add-subtract has no single-element meaning) - odd-length input
//!   **panics** instead of silently dropping or misreading a trailing
//!   element.

use super::ops::dsp::Dsp;
use super::ops::fp16::{Fp16Fma, Fp16Neon};
use super::ops::neon::Neon;
use super::ops::vfpv4neon::Vfpv4Neon;

/// One 4-lane token method lifted to a slice: chunk `a`/`b`/`out` by 4,
/// call the token per chunk, hand the (`< 4`-element) remainder to
/// `auto_down`. Falls back to `auto_down` entirely if the token isn't
/// available.
macro_rules! auto_binop4 {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, b, out);
			};
			for ((ac, bc), oc) in a.chunks_exact(4).zip(b.chunks_exact(4)).zip(out.chunks_exact_mut(4)) {
				let av: [$Elem; 4] = ac.try_into().unwrap();
				let bv: [$Elem; 4] = bc.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av, bv));
			}
			let rem = a.len() - a.len() % 4;
			$scalar_fn(&a[rem..], &b[rem..], &mut out[rem..]);
		}
	};
}

/// One 4-lane token unop lifted to a slice: chunk `a`/`out` by 4, call the
/// token per chunk, hand the remainder to `auto_down`. Same shape as
/// [`auto_binop4`], one operand.
macro_rules! auto_unop4 {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, out);
			};
			for (ac, oc) in a.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
				let av: [$Elem; 4] = ac.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av));
			}
			let rem = a.len() - a.len() % 4;
			$scalar_fn(&a[rem..], &mut out[rem..]);
		}
	};
}

/// One 4-lane token compare lifted to a slice: `$Elem` in, always `u32` lane
/// masks out (Neon's `neon_cmp_x4!` always returns `[u32; 4]` regardless of
/// `$Elem` - unlike x86, whose compare masks are typed like the input).
macro_rules! auto_cmp4 {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [u32]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, b, out);
			};
			for ((ac, bc), oc) in a.chunks_exact(4).zip(b.chunks_exact(4)).zip(out.chunks_exact_mut(4)) {
				let av: [$Elem; 4] = ac.try_into().unwrap();
				let bv: [$Elem; 4] = bc.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av, bv));
			}
			let rem = a.len() - a.len() % 4;
			$scalar_fn(&a[rem..], &b[rem..], &mut out[rem..]);
		}
	};
}

/// As [`auto_binop4`], but for [`super::ops::fp16`]'s `[u16; 8]` FullFP16
/// bit-pattern shape (8 lanes, `u16` fixed rather than generic - both
/// `Fp16Neon`'s arithmetic and its `neon_cmp_f16x8!`-based compares return
/// `[u16; 8]`, so one macro covers both).
macro_rules! auto_binop8 {
	($(#[$doc:meta])* $fn_name:ident, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[u16], b: &[u16], out: &mut [u16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, b, out);
			};
			for ((ac, bc), oc) in a.chunks_exact(8).zip(b.chunks_exact(8)).zip(out.chunks_exact_mut(8)) {
				let av: [u16; 8] = ac.try_into().unwrap();
				let bv: [u16; 8] = bc.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av, bv));
			}
			let rem = a.len() - a.len() % 8;
			$scalar_fn(&a[rem..], &b[rem..], &mut out[rem..]);
		}
	};
}

/// As [`auto_unop4`], for [`super::ops::fp16`]'s 8-lane `[u16; 8]` shape.
macro_rules! auto_unop8 {
	($(#[$doc:meta])* $fn_name:ident, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[u16], out: &mut [u16]) {
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, out);
			};
			for (ac, oc) in a.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
				let av: [u16; 8] = ac.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av));
			}
			let rem = a.len() - a.len() % 8;
			$scalar_fn(&a[rem..], &mut out[rem..]);
		}
	};
}

auto_binop4!(
	/// `VADD.S32`/scalar: per-lane wrapping `i32` addition.
	add_i32,
	i32,
	Neon,
	add_i32x4,
	super::auto_down::add_i32
);
auto_binop4!(
	/// `VSUB.S32`/scalar: per-lane wrapping `i32` subtraction.
	sub_i32,
	i32,
	Neon,
	sub_i32x4,
	super::auto_down::sub_i32
);
auto_binop4!(
	/// `VMUL.S32`/scalar: per-lane wrapping `i32` multiplication.
	mul_i32,
	i32,
	Neon,
	mul_i32x4,
	super::auto_down::mul_i32
);

auto_binop4!(
	/// `VADD.F32`/scalar: per-lane `f32` addition.
	add_f32,
	f32,
	Neon,
	add_f32x4,
	super::auto_down::add_f32
);
auto_binop4!(
	/// `VSUB.F32`/scalar: per-lane `f32` subtraction.
	sub_f32,
	f32,
	Neon,
	sub_f32x4,
	super::auto_down::sub_f32
);
auto_binop4!(
	/// `VMUL.F32`/scalar: per-lane `f32` multiplication.
	mul_f32,
	f32,
	Neon,
	mul_f32x4,
	super::auto_down::mul_f32
);

auto_binop4!(
	/// `VAND`/scalar: per-lane `u32` bitwise AND.
	and_u32,
	u32,
	Neon,
	and_u32x4,
	super::auto_down::and_u32
);
auto_binop4!(
	/// `VORR`/scalar: per-lane `u32` bitwise OR.
	or_u32,
	u32,
	Neon,
	or_u32x4,
	super::auto_down::or_u32
);
auto_binop4!(
	/// `VEOR`/scalar: per-lane `u32` bitwise XOR.
	xor_u32,
	u32,
	Neon,
	xor_u32x4,
	super::auto_down::xor_u32
);
auto_binop4!(
	/// `VBIC`/scalar: per-lane `u32` bitwise AND-NOT, `a[i] & !b[i]` (native
	/// `vbicq_u32` operand order; this is the mirror image of x86's `andnot`
	/// convention, `!a[i] & b[i]` - swap the arguments at the call site if
	/// x86 parity is needed).
	andnot_u32,
	u32,
	Neon,
	andnot_u32x4,
	super::auto_down::andnot_u32
);

auto_binop4!(
	/// `VMAX.S32`/scalar: per-lane `i32` maximum.
	max_i32,
	i32,
	Neon,
	max_i32x4,
	super::auto_down::max_i32
);
auto_binop4!(
	/// `VMIN.S32`/scalar: per-lane `i32` minimum.
	min_i32,
	i32,
	Neon,
	min_i32x4,
	super::auto_down::min_i32
);
auto_binop4!(
	/// `VMAX.F32`/scalar: per-lane `f32` maximum. NaN follows the `VMAX`
	/// instruction, not Rust `f32::max` (same caveat as
	/// [`super::ops::neon::Neon::max_f32x4`]).
	max_f32,
	f32,
	Neon,
	max_f32x4,
	super::auto_down::max_f32
);
auto_binop4!(
	/// `VMIN.F32`/scalar: per-lane `f32` minimum. Same NaN caveat as [`max_f32`].
	min_f32,
	f32,
	Neon,
	min_f32x4,
	super::auto_down::min_f32
);
auto_binop4!(
	/// `VSHL.S32`/scalar: per-lane variable `i32` shift (see
	/// [`super::ops::neon::Neon::shl_i32x4`] for the exact saturating-shift
	/// semantics the scalar remainder must match).
	shl_i32,
	i32,
	Neon,
	shl_i32x4,
	super::auto_down::shl_i32
);

auto_unop4!(
	/// `VABS.S32`/scalar: per-lane `i32` absolute value.
	abs_i32,
	i32,
	Neon,
	abs_i32x4,
	super::auto_down::abs_i32
);
auto_unop4!(
	/// `VNEG.S32`/scalar: per-lane `i32` negation.
	neg_i32,
	i32,
	Neon,
	neg_i32x4,
	super::auto_down::neg_i32
);
auto_unop4!(
	/// `VABS.F32`/scalar: per-lane `f32` absolute value.
	abs_f32,
	f32,
	Neon,
	abs_f32x4,
	super::auto_down::abs_f32
);
auto_unop4!(
	/// `VNEG.F32`/scalar: per-lane `f32` negation.
	neg_f32,
	f32,
	Neon,
	neg_f32x4,
	super::auto_down::neg_f32
);
auto_unop4!(
	/// `VMVN`/scalar: per-lane `u32` bitwise NOT.
	not_u32,
	u32,
	Neon,
	not_u32x4,
	super::auto_down::not_u32
);

auto_cmp4!(
	/// `VCEQ.S32`/scalar: per-lane `i32` equality, `[u32]` lane mask
	/// (all-1s or 0, not `bool`).
	cmpeq_i32,
	i32,
	Neon,
	cmpeq_i32x4,
	super::auto_down::cmpeq_i32
);
auto_cmp4!(
	/// `VCGT.S32`/scalar: per-lane `i32` greater-than, `[u32]` lane mask.
	cmpgt_i32,
	i32,
	Neon,
	cmpgt_i32x4,
	super::auto_down::cmpgt_i32
);
auto_cmp4!(
	/// `VCGE.S32`/scalar: per-lane `i32` greater-or-equal, `[u32]` lane mask.
	cmpge_i32,
	i32,
	Neon,
	cmpge_i32x4,
	super::auto_down::cmpge_i32
);
auto_cmp4!(
	/// `VCLT.S32`/scalar: per-lane `i32` less-than, `[u32]` lane mask.
	cmplt_i32,
	i32,
	Neon,
	cmplt_i32x4,
	super::auto_down::cmplt_i32
);
auto_cmp4!(
	/// `VCLE.S32`/scalar: per-lane `i32` less-or-equal, `[u32]` lane mask.
	cmple_i32,
	i32,
	Neon,
	cmple_i32x4,
	super::auto_down::cmple_i32
);
auto_cmp4!(
	/// `VCEQ.F32`/scalar: per-lane `f32` equality, `[u32]` lane mask. NaN
	/// never equals (mask 0).
	cmpeq_f32,
	f32,
	Neon,
	cmpeq_f32x4,
	super::auto_down::cmpeq_f32
);
auto_cmp4!(
	/// `VCGT.F32`/scalar: per-lane `f32` greater-than (ordered; false if
	/// either lane is NaN), `[u32]` lane mask.
	cmpgt_f32,
	f32,
	Neon,
	cmpgt_f32x4,
	super::auto_down::cmpgt_f32
);
auto_cmp4!(
	/// `VCGE.F32`/scalar: per-lane `f32` greater-or-equal (ordered), `[u32]` lane mask.
	cmpge_f32,
	f32,
	Neon,
	cmpge_f32x4,
	super::auto_down::cmpge_f32
);
auto_cmp4!(
	/// `VCLT.F32`/scalar: per-lane `f32` less-than (ordered), `[u32]` lane mask.
	cmplt_f32,
	f32,
	Neon,
	cmplt_f32x4,
	super::auto_down::cmplt_f32
);
auto_cmp4!(
	/// `VCLE.F32`/scalar: per-lane `f32` less-or-equal (ordered), `[u32]` lane mask.
	cmple_f32,
	f32,
	Neon,
	cmple_f32x4,
	super::auto_down::cmple_f32
);

/// `out[i] = a[i] * b[i] + c[i]` (`VFMA.F32`/scalar). HW-fused via
/// [`Vfpv4Neon::fma_f32x4`] (`(acc, b, c) -> b*c + acc`, so `c` here plays
/// the accumulator role); scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let Some(t) = Vfpv4Neon::from_features(super::detect_features()) else {
		return super::auto_down::fmadd_f32(a, b, c, out);
	};
	let (a_chunks, a_rem) = a.as_chunks::<4>();
	let (b_chunks, _) = b.as_chunks::<4>();
	let (c_chunks, _) = c.as_chunks::<4>();
	let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
	for (((av, bv), cv), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(out_chunks) {
		*oc = t.fma_f32x4(*cv, *av, *bv);
	}
	let rem = a.len() - a_rem.len();
	super::auto_down::fmadd_f32(a_rem, &b[rem..], &c[rem..], out_rem);
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`VBSL`/scalar), x86
/// `select_i32` semantics/argument order. `mask` is always `&[u32]` here
/// (Neon compares always produce `u32` lane masks, see [`cmpeq_i32`] etc.),
/// unlike x86 where the mask is typed like `a`/`b`.
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_i32(a: &[i32], b: &[i32], mask: &[u32], out: &mut [i32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let Some(t) = Neon::from_features(super::detect_features()) else {
		return super::auto_down::select_i32(a, b, mask, out);
	};
	let (a_chunks, a_rem) = a.as_chunks::<4>();
	let (b_chunks, _) = b.as_chunks::<4>();
	let (mask_chunks, _) = mask.as_chunks::<4>();
	let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
	for (((av, bv), mv), oc) in a_chunks.iter().zip(b_chunks).zip(mask_chunks).zip(out_chunks) {
		let av32: [u32; 4] = core::array::from_fn(|i| av[i] as u32);
		let bv32: [u32; 4] = core::array::from_fn(|i| bv[i] as u32);
		let rv = t.select_u32x4(*mv, bv32, av32);
		for i in 0..4 {
			oc[i] = rv[i] as i32;
		}
	}
	let rem = a.len() - a_rem.len();
	super::auto_down::select_i32(a_rem, &b[rem..], &mask[rem..], out_rem);
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u32` view). Same
/// cascade as [`select_i32`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_u32(a: &[u32], b: &[u32], mask: &[u32], out: &mut [u32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let Some(t) = Neon::from_features(super::detect_features()) else {
		return super::auto_down::select_u32(a, b, mask, out);
	};
	let (a_chunks, a_rem) = a.as_chunks::<4>();
	let (b_chunks, _) = b.as_chunks::<4>();
	let (mask_chunks, _) = mask.as_chunks::<4>();
	let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
	for (((av, bv), mv), oc) in a_chunks.iter().zip(b_chunks).zip(mask_chunks).zip(out_chunks) {
		*oc = t.select_u32x4(*mv, *bv, *av);
	}
	let rem = a.len() - a_rem.len();
	super::auto_down::select_u32(a_rem, &b[rem..], &mask[rem..], out_rem);
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`f32` view, bitcast
/// through [`Neon::select_u32x4`]). Same cascade as [`select_i32`]. Unlike
/// x86's `select_f32`, the mask test here is a plain nonzero test on the
/// `u32` mask lane, not a float sign-bit test.
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_f32(a: &[f32], b: &[f32], mask: &[u32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let Some(t) = Neon::from_features(super::detect_features()) else {
		return super::auto_down::select_f32(a, b, mask, out);
	};
	let (a_chunks, a_rem) = a.as_chunks::<4>();
	let (b_chunks, _) = b.as_chunks::<4>();
	let (mask_chunks, _) = mask.as_chunks::<4>();
	let (out_chunks, out_rem) = out.as_chunks_mut::<4>();
	for (((av, bv), mv), oc) in a_chunks.iter().zip(b_chunks).zip(mask_chunks).zip(out_chunks) {
		let av32: [u32; 4] = core::array::from_fn(|i| av[i].to_bits());
		let bv32: [u32; 4] = core::array::from_fn(|i| bv[i].to_bits());
		let rv = t.select_u32x4(*mv, bv32, av32);
		for i in 0..4 {
			oc[i] = f32::from_bits(rv[i]);
		}
	}
	let rem = a.len() - a_rem.len();
	super::auto_down::select_f32(a_rem, &b[rem..], &mask[rem..], out_rem);
}

auto_binop8!(
	/// `VADD.F16`/scalar: per-lane FullFP16 addition (`[u16; 8]` bit pattern,
	/// see [`super::ops::fp16`]).
	add_f16,
	Fp16Neon,
	add_f16x8,
	super::auto_down::add_f16
);
auto_binop8!(
	/// `VSUB.F16`/scalar: per-lane FullFP16 subtraction.
	sub_f16,
	Fp16Neon,
	sub_f16x8,
	super::auto_down::sub_f16
);
auto_binop8!(
	/// `VMUL.F16`/scalar: per-lane FullFP16 multiplication.
	mul_f16,
	Fp16Neon,
	mul_f16x8,
	super::auto_down::mul_f16
);
auto_binop8!(
	/// `VMAX.F16`/scalar: per-lane FullFP16 maximum. NaN follows the `VMAX`
	/// instruction, same caveat as [`max_f32`].
	max_f16,
	Fp16Neon,
	max_f16x8,
	super::auto_down::max_f16
);
auto_binop8!(
	/// `VMIN.F16`/scalar: per-lane FullFP16 minimum. Same NaN caveat as [`max_f16`].
	min_f16,
	Fp16Neon,
	min_f16x8,
	super::auto_down::min_f16
);

auto_unop8!(
	/// `VABS.F16`/scalar: per-lane FullFP16 absolute value.
	abs_f16,
	Fp16Neon,
	abs_f16x8,
	super::auto_down::abs_f16
);
auto_unop8!(
	/// `VNEG.F16`/scalar: per-lane FullFP16 negation.
	neg_f16,
	Fp16Neon,
	neg_f16x8,
	super::auto_down::neg_f16
);

auto_binop8!(
	/// `VCEQ.F16`/scalar: per-lane FullFP16 equality, `[u16]` lane mask
	/// (all-1s or 0, not `bool`). NaN never equals (mask 0).
	cmpeq_f16,
	Fp16Neon,
	cmpeq_f16x8,
	super::auto_down::cmpeq_f16
);
auto_binop8!(
	/// `VCGT.F16`/scalar: per-lane FullFP16 greater-than (ordered; false if
	/// either lane is NaN), `[u16]` lane mask.
	cmpgt_f16,
	Fp16Neon,
	cmpgt_f16x8,
	super::auto_down::cmpgt_f16
);
auto_binop8!(
	/// `VCGE.F16`/scalar: per-lane FullFP16 greater-or-equal (ordered), `[u16]` lane mask.
	cmpge_f16,
	Fp16Neon,
	cmpge_f16x8,
	super::auto_down::cmpge_f16
);
auto_binop8!(
	/// `VCLT.F16`/scalar: per-lane FullFP16 less-than (ordered), `[u16]` lane mask.
	cmplt_f16,
	Fp16Neon,
	cmplt_f16x8,
	super::auto_down::cmplt_f16
);
auto_binop8!(
	/// `VCLE.F16`/scalar: per-lane FullFP16 less-or-equal (ordered), `[u16]` lane mask.
	cmple_f16,
	Fp16Neon,
	cmple_f16x8,
	super::auto_down::cmple_f16
);

/// `out[i] = a[i] * b[i] + c[i]` (`VFMA.F16`/scalar, `f16` bit patterns as
/// `[u16]`). HW-fused via [`Fp16Fma::fma_f16x8`] (`(acc, b, c) -> b*c + acc`,
/// `c` plays the accumulator role, same convention as [`fmadd_f32`]); scalar
/// remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmadd_f16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let Some(t) = Fp16Fma::from_features(super::detect_features()) else {
		return super::auto_down::fmadd_f16(a, b, c, out);
	};
	let (a_chunks, a_rem) = a.as_chunks::<8>();
	let (b_chunks, _) = b.as_chunks::<8>();
	let (c_chunks, _) = c.as_chunks::<8>();
	let (out_chunks, out_rem) = out.as_chunks_mut::<8>();
	for (((av, bv), cv), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(out_chunks) {
		*oc = t.fma_f16x8(*cv, *av, *bv);
	}
	let rem = a.len() - a_rem.len();
	super::auto_down::fmadd_f16(a_rem, &b[rem..], &c[rem..], out_rem);
}

/// As [`auto_binop4`], but for [`super::ops::dsp::Dsp`]'s packed `[i16; 2]`
/// shape (2 lanes in one GPR, e.g. `qadd16`/`qasx`).
macro_rules! auto_binop2 {
	($(#[$doc:meta])* $fn_name:ident, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[i16], b: &[i16], out: &mut [i16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, b, out);
			};
			for ((ac, bc), oc) in a.chunks_exact(2).zip(b.chunks_exact(2)).zip(out.chunks_exact_mut(2)) {
				let av: [i16; 2] = ac.try_into().unwrap();
				let bv: [i16; 2] = bc.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av, bv));
			}
			let rem = a.len() - a.len() % 2;
			$scalar_fn(&a[rem..], &b[rem..], &mut out[rem..]);
		}
	};
}

auto_binop4!(
	/// `QADD8`/scalar: per-lane saturating `i8` addition.
	qadd_i8,
	i8,
	Dsp,
	qadd8,
	super::auto_down::qadd_i8
);
auto_binop4!(
	/// `QSUB8`/scalar: per-lane saturating `i8` subtraction.
	qsub_i8,
	i8,
	Dsp,
	qsub8,
	super::auto_down::qsub_i8
);
auto_binop4!(
	/// `SADD8`/scalar: per-lane wrapping `i8` addition.
	sadd_i8,
	i8,
	Dsp,
	sadd8,
	super::auto_down::sadd_i8
);
auto_binop4!(
	/// `SSUB8`/scalar: per-lane wrapping `i8` subtraction.
	ssub_i8,
	i8,
	Dsp,
	ssub8,
	super::auto_down::ssub_i8
);
auto_binop4!(
	/// `SHADD8`/scalar: per-lane halving signed `i8` addition (`(a+b)/2`, no
	/// saturation).
	shadd_i8,
	i8,
	Dsp,
	shadd8,
	super::auto_down::shadd_i8
);
auto_binop4!(
	/// `SHSUB8`/scalar: per-lane halving signed `i8` subtraction (`(a-b)/2`,
	/// no saturation).
	shsub_i8,
	i8,
	Dsp,
	shsub8,
	super::auto_down::shsub_i8
);
auto_binop4!(
	/// `USUB8`/scalar: per-lane wrapping unsigned `u8` subtraction.
	usub_u8,
	u8,
	Dsp,
	usub8,
	super::auto_down::usub_u8
);

auto_binop2!(
	/// `QADD16`/scalar: per-lane saturating `i16` addition, packed 2-lane.
	qadd_i16,
	Dsp,
	qadd16,
	super::auto_down::qadd_i16
);
auto_binop2!(
	/// `QSUB16`/scalar: per-lane saturating `i16` subtraction, packed 2-lane.
	qsub_i16,
	Dsp,
	qsub16,
	super::auto_down::qsub_i16
);
auto_binop2!(
	/// `SADD16`/scalar: per-lane wrapping `i16` addition, packed 2-lane.
	sadd_i16,
	Dsp,
	sadd16,
	super::auto_down::sadd_i16
);
auto_binop2!(
	/// `SHADD16`/scalar: per-lane halving signed `i16` addition (no
	/// saturation), packed 2-lane.
	shadd_i16,
	Dsp,
	shadd16,
	super::auto_down::shadd_i16
);
auto_binop2!(
	/// `SHSUB16`/scalar: per-lane halving signed `i16` subtraction (no
	/// saturation), packed 2-lane.
	shsub_i16,
	Dsp,
	shsub16,
	super::auto_down::shsub_i16
);
/// One 2-lane token cross-op (`QASX`/`QSAX`/`SASX`-shaped) lifted to a
/// slice. Unlike [`auto_binop2`]'s independent lanes, a cross op needs both
/// elements of a pair together, so an odd-length input has no defined
/// remainder pairing - **panics** on odd length rather than silently
/// dropping or misreading the trailing element.
macro_rules! auto_cross2 {
	($(#[$doc:meta])* $fn_name:ident, $Token:ty, $token_method:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len() || a.len() % 2 != 0`.
		#[inline]
		pub fn $fn_name(a: &[i16], b: &[i16], out: &mut [i16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			assert_eq!(a.len() % 2, 0, "cross add-subtract needs pairs; odd length has no defined remainder");
			let Some(t) = <$Token>::from_features(super::detect_features()) else {
				return $scalar_fn(a, b, out);
			};
			for ((ac, bc), oc) in a.chunks_exact(2).zip(b.chunks_exact(2)).zip(out.chunks_exact_mut(2)) {
				let av: [i16; 2] = ac.try_into().unwrap();
				let bv: [i16; 2] = bc.try_into().unwrap();
				oc.copy_from_slice(&t.$token_method(av, bv));
			}
		}
	};
}

auto_cross2!(
	/// `QASX`/scalar: saturating cross add-subtract, packed pairs
	/// (`out[2k]=a[2k]-b[2k+1], out[2k+1]=a[2k+1]+b[2k]`).
	qasx_i16,
	Dsp,
	qasx,
	super::auto_down::qasx_i16
);
auto_cross2!(
	/// `QSAX`/scalar: saturating cross subtract-add, packed pairs
	/// (`out[2k]=a[2k]+b[2k+1], out[2k+1]=a[2k+1]-b[2k]`).
	qsax_i16,
	Dsp,
	qsax,
	super::auto_down::qsax_i16
);
auto_cross2!(
	/// `SASX`/scalar: wrapping cross add-subtract, packed pairs
	/// (`out[2k]=a[2k]-b[2k+1], out[2k+1]=a[2k+1]+b[2k]`).
	sasx_i16,
	Dsp,
	sasx,
	super::auto_down::sasx_i16
);

#[cfg(test)]
#[path = "test/auto_up.rs"]
mod tests;
