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
//! each fn is a 2-rung "token -> portable scalar" cascade. The chunking
//! lives in the tokens' `_slice` methods (vector body plus a scalar tail with
//! the same formulas as `auto_down`), as on aarch64 and x86; this file only
//! checks lengths and picks the rung. Breadth is meant
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

/// One token's `_slice` method as an `auto` fn: lengths checked here, then
/// the token's slice form (vector body + scalar tail), or `auto_down` when the
/// token isn't available. `$Out` is the output element: the input type for
/// arithmetic, `u32`/`u16` for compare masks.
macro_rules! auto_binop {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty => $Out:ty, $Token:ty, $slice_fn:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Out]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			match <$Token>::from_features(super::detect_features()) {
				Some(t) => t.$slice_fn(a, b, out),
				None => $scalar_fn(a, b, out),
			}
		}
	};
}

/// As [`auto_binop`], one operand.
macro_rules! auto_unop {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $Token:ty, $slice_fn:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(out.len(), a.len());
			match <$Token>::from_features(super::detect_features()) {
				Some(t) => t.$slice_fn(a, out),
				None => $scalar_fn(a, out),
			}
		}
	};
}

/// `out = a * b + c` from a token's `fmadd_*_slice` (fused body, unfused
/// scalar tail), else `auto_down`.
macro_rules! auto_fmadd {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $Token:ty, $slice_fn:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// Length mismatch among `a`, `b`, `c`, `out`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(a.len(), c.len());
			assert_eq!(out.len(), a.len());
			match <$Token>::from_features(super::detect_features()) {
				Some(t) => t.$slice_fn(a, b, c, out),
				None => $scalar_fn(a, b, c, out),
			}
		}
	};
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` from Neon's
/// `select_*_slice` (which takes `(mask, if_set, if_clear)`), else `auto_down`.
macro_rules! auto_select {
	($(#[$doc:meta])* $fn_name:ident, $Elem:ty, $slice_fn:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// Length mismatch among `a`, `b`, `mask`, `out`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], mask: &[u32], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(a.len(), mask.len());
			assert_eq!(out.len(), a.len());
			match Neon::from_features(super::detect_features()) {
				Some(t) => t.$slice_fn(mask, b, a, out),
				None => $scalar_fn(a, b, mask, out),
			}
		}
	};
}

auto_binop!(
    /// `VADD.S32`/scalar: per-lane wrapping `i32` addition.
    add_i32,
    i32 => i32,
    Neon,
    add_i32_slice,
    super::auto_down::add_i32
);
auto_binop!(
    /// `VSUB.S32`/scalar: per-lane wrapping `i32` subtraction.
    sub_i32,
    i32 => i32,
    Neon,
    sub_i32_slice,
    super::auto_down::sub_i32
);
auto_binop!(
    /// `VMUL.S32`/scalar: per-lane wrapping `i32` multiplication.
    mul_i32,
    i32 => i32,
    Neon,
    mul_i32_slice,
    super::auto_down::mul_i32
);

auto_binop!(
    /// `VADD.F32`/scalar: per-lane `f32` addition.
    add_f32,
    f32 => f32,
    Neon,
    add_f32_slice,
    super::auto_down::add_f32
);
auto_binop!(
    /// `VSUB.F32`/scalar: per-lane `f32` subtraction.
    sub_f32,
    f32 => f32,
    Neon,
    sub_f32_slice,
    super::auto_down::sub_f32
);
auto_binop!(
    /// `VMUL.F32`/scalar: per-lane `f32` multiplication.
    mul_f32,
    f32 => f32,
    Neon,
    mul_f32_slice,
    super::auto_down::mul_f32
);

auto_binop!(
    /// `VAND`/scalar: per-lane `u32` bitwise AND.
    and_u32,
    u32 => u32,
    Neon,
    and_u32_slice,
    super::auto_down::and_u32
);
auto_binop!(
    /// `VORR`/scalar: per-lane `u32` bitwise OR.
    or_u32,
    u32 => u32,
    Neon,
    or_u32_slice,
    super::auto_down::or_u32
);
auto_binop!(
    /// `VEOR`/scalar: per-lane `u32` bitwise XOR.
    xor_u32,
    u32 => u32,
    Neon,
    xor_u32_slice,
    super::auto_down::xor_u32
);
auto_binop!(
    /// `VBIC`/scalar: per-lane `u32` bitwise AND-NOT, `a[i] & !b[i]` (native
    /// `vbicq_u32` operand order; this is the mirror image of x86's `andnot`
    /// convention, `!a[i] & b[i]` - swap the arguments at the call site if
    /// x86 parity is needed).
    andnot_u32,
    u32 => u32,
    Neon,
    andnot_u32_slice,
    super::auto_down::andnot_u32
);

auto_binop!(
    /// `VMAX.S32`/scalar: per-lane `i32` maximum.
    max_i32,
    i32 => i32,
    Neon,
    max_i32_slice,
    super::auto_down::max_i32
);
auto_binop!(
    /// `VMIN.S32`/scalar: per-lane `i32` minimum.
    min_i32,
    i32 => i32,
    Neon,
    min_i32_slice,
    super::auto_down::min_i32
);
auto_binop!(
    /// `VMAX.F32`/scalar: per-lane `f32` maximum. NaN follows the `VMAX`
    /// instruction, not Rust `f32::max` (same caveat as
    /// [`super::ops::neon::Neon::max_f32x4`]).
    max_f32,
    f32 => f32,
    Neon,
    max_f32_slice,
    super::auto_down::max_f32
);
auto_binop!(
    /// `VMIN.F32`/scalar: per-lane `f32` minimum. Same NaN caveat as [`max_f32`].
    min_f32,
    f32 => f32,
    Neon,
    min_f32_slice,
    super::auto_down::min_f32
);
auto_binop!(
    /// `VSHL.S32`/scalar: per-lane variable `i32` shift (see
    /// [`super::ops::neon::Neon::shl_i32x4`] for the exact saturating-shift
    /// semantics the scalar remainder must match).
    shl_i32,
    i32 => i32,
    Neon,
    shl_i32_slice,
    super::auto_down::shl_i32
);

auto_unop!(
    /// `VABS.S32`/scalar: per-lane `i32` absolute value.
    abs_i32,
    i32,
    Neon,
    abs_i32_slice,
    super::auto_down::abs_i32
);
auto_unop!(
    /// `VNEG.S32`/scalar: per-lane `i32` negation.
    neg_i32,
    i32,
    Neon,
    neg_i32_slice,
    super::auto_down::neg_i32
);
auto_unop!(
    /// `VABS.F32`/scalar: per-lane `f32` absolute value.
    abs_f32,
    f32,
    Neon,
    abs_f32_slice,
    super::auto_down::abs_f32
);
auto_unop!(
    /// `VNEG.F32`/scalar: per-lane `f32` negation.
    neg_f32,
    f32,
    Neon,
    neg_f32_slice,
    super::auto_down::neg_f32
);
auto_unop!(
    /// `VMVN`/scalar: per-lane `u32` bitwise NOT.
    not_u32,
    u32,
    Neon,
    not_u32_slice,
    super::auto_down::not_u32
);

auto_binop!(
    /// `VCEQ.S32`/scalar: per-lane `i32` equality, `[u32]` lane mask
    /// (all-1s or 0, not `bool`).
    cmpeq_i32,
    i32 => u32,
    Neon,
    cmpeq_i32_slice,
    super::auto_down::cmpeq_i32
);
auto_binop!(
    /// `VCGT.S32`/scalar: per-lane `i32` greater-than, `[u32]` lane mask.
    cmpgt_i32,
    i32 => u32,
    Neon,
    cmpgt_i32_slice,
    super::auto_down::cmpgt_i32
);
auto_binop!(
    /// `VCGE.S32`/scalar: per-lane `i32` greater-or-equal, `[u32]` lane mask.
    cmpge_i32,
    i32 => u32,
    Neon,
    cmpge_i32_slice,
    super::auto_down::cmpge_i32
);
auto_binop!(
    /// `VCLT.S32`/scalar: per-lane `i32` less-than, `[u32]` lane mask.
    cmplt_i32,
    i32 => u32,
    Neon,
    cmplt_i32_slice,
    super::auto_down::cmplt_i32
);
auto_binop!(
    /// `VCLE.S32`/scalar: per-lane `i32` less-or-equal, `[u32]` lane mask.
    cmple_i32,
    i32 => u32,
    Neon,
    cmple_i32_slice,
    super::auto_down::cmple_i32
);
auto_binop!(
    /// `VCEQ.F32`/scalar: per-lane `f32` equality, `[u32]` lane mask. NaN
    /// never equals (mask 0).
    cmpeq_f32,
    f32 => u32,
    Neon,
    cmpeq_f32_slice,
    super::auto_down::cmpeq_f32
);
auto_binop!(
    /// `VCGT.F32`/scalar: per-lane `f32` greater-than (ordered; false if
    /// either lane is NaN), `[u32]` lane mask.
    cmpgt_f32,
    f32 => u32,
    Neon,
    cmpgt_f32_slice,
    super::auto_down::cmpgt_f32
);
auto_binop!(
    /// `VCGE.F32`/scalar: per-lane `f32` greater-or-equal (ordered), `[u32]` lane mask.
    cmpge_f32,
    f32 => u32,
    Neon,
    cmpge_f32_slice,
    super::auto_down::cmpge_f32
);
auto_binop!(
    /// `VCLT.F32`/scalar: per-lane `f32` less-than (ordered), `[u32]` lane mask.
    cmplt_f32,
    f32 => u32,
    Neon,
    cmplt_f32_slice,
    super::auto_down::cmplt_f32
);
auto_binop!(
    /// `VCLE.F32`/scalar: per-lane `f32` less-or-equal (ordered), `[u32]` lane mask.
    cmple_f32,
    f32 => u32,
    Neon,
    cmple_f32_slice,
    super::auto_down::cmple_f32
);

auto_fmadd!(
    /// `out[i] = a[i] * b[i] + c[i]` (`VFMA.F32`/scalar). HW-fused via
    /// [`Vfpv4Neon`]'s `fmadd_f32_slice`; the scalar remainder is not fused.
    fmadd_f32,
    f32,
    Vfpv4Neon,
    fmadd_f32_slice,
    super::auto_down::fmadd_f32
);

auto_select!(
    /// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`VBSL`/scalar), x86
    /// `select_i32` semantics/argument order. `mask` is always `&[u32]` here
    /// (Neon compares always produce `u32` lane masks, see [`cmpeq_i32`] etc.),
    /// unlike x86 where the mask is typed like `a`/`b`.
    select_i32,
    i32,
    select_i32_slice,
    super::auto_down::select_i32
);

auto_select!(
    /// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u32` view). Same
    /// cascade as [`select_i32`].
    select_u32,
    u32,
    select_u32_slice,
    super::auto_down::select_u32
);

auto_select!(
    /// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`f32` view). Same
    /// cascade as [`select_i32`]. Unlike x86's `select_f32`, the mask test
    /// here is a plain nonzero test on the `u32` mask lane, not a float
    /// sign-bit test.
    select_f32,
    f32,
    select_f32_slice,
    super::auto_down::select_f32
);

auto_binop!(
    /// `VADD.F16`/scalar: per-lane FullFP16 addition (`[u16; 8]` bit pattern,
    /// see [`super::ops::fp16`]).
    add_f16,
    u16 => u16,
    Fp16Neon,
    add_f16_slice,
    super::auto_down::add_f16
);
auto_binop!(
    /// `VSUB.F16`/scalar: per-lane FullFP16 subtraction.
    sub_f16,
    u16 => u16,
    Fp16Neon,
    sub_f16_slice,
    super::auto_down::sub_f16
);
auto_binop!(
    /// `VMUL.F16`/scalar: per-lane FullFP16 multiplication.
    mul_f16,
    u16 => u16,
    Fp16Neon,
    mul_f16_slice,
    super::auto_down::mul_f16
);
auto_binop!(
    /// `VMAX.F16`/scalar: per-lane FullFP16 maximum. NaN follows the `VMAX`
    /// instruction, same caveat as [`max_f32`].
    max_f16,
    u16 => u16,
    Fp16Neon,
    max_f16_slice,
    super::auto_down::max_f16
);
auto_binop!(
    /// `VMIN.F16`/scalar: per-lane FullFP16 minimum. Same NaN caveat as [`max_f16`].
    min_f16,
    u16 => u16,
    Fp16Neon,
    min_f16_slice,
    super::auto_down::min_f16
);

auto_unop!(
    /// `VABS.F16`/scalar: per-lane FullFP16 absolute value.
    abs_f16,
    u16,
    Fp16Neon,
    abs_f16_slice,
    super::auto_down::abs_f16
);
auto_unop!(
    /// `VNEG.F16`/scalar: per-lane FullFP16 negation.
    neg_f16,
    u16,
    Fp16Neon,
    neg_f16_slice,
    super::auto_down::neg_f16
);

auto_binop!(
    /// `VCEQ.F16`/scalar: per-lane FullFP16 equality, `[u16]` lane mask
    /// (all-1s or 0, not `bool`). NaN never equals (mask 0).
    cmpeq_f16,
    u16 => u16,
    Fp16Neon,
    cmpeq_f16_slice,
    super::auto_down::cmpeq_f16
);
auto_binop!(
    /// `VCGT.F16`/scalar: per-lane FullFP16 greater-than (ordered; false if
    /// either lane is NaN), `[u16]` lane mask.
    cmpgt_f16,
    u16 => u16,
    Fp16Neon,
    cmpgt_f16_slice,
    super::auto_down::cmpgt_f16
);
auto_binop!(
    /// `VCGE.F16`/scalar: per-lane FullFP16 greater-or-equal (ordered), `[u16]` lane mask.
    cmpge_f16,
    u16 => u16,
    Fp16Neon,
    cmpge_f16_slice,
    super::auto_down::cmpge_f16
);
auto_binop!(
    /// `VCLT.F16`/scalar: per-lane FullFP16 less-than (ordered), `[u16]` lane mask.
    cmplt_f16,
    u16 => u16,
    Fp16Neon,
    cmplt_f16_slice,
    super::auto_down::cmplt_f16
);
auto_binop!(
    /// `VCLE.F16`/scalar: per-lane FullFP16 less-or-equal (ordered), `[u16]` lane mask.
    cmple_f16,
    u16 => u16,
    Fp16Neon,
    cmple_f16_slice,
    super::auto_down::cmple_f16
);

auto_fmadd!(
    /// `out[i] = a[i] * b[i] + c[i]` (`VFMA.F16`/scalar, `f16` bit patterns as
    /// `[u16]`). HW-fused via [`Fp16Fma`]'s `fmadd_f16_slice`; the scalar
    /// remainder is not fused.
    fmadd_f16,
    u16,
    Fp16Fma,
    fmadd_f16_slice,
    super::auto_down::fmadd_f16
);

auto_binop!(
    /// `QADD8`/scalar: per-lane saturating `i8` addition.
    qadd_i8,
    i8 => i8,
    Dsp,
    qadd_i8_slice,
    super::auto_down::qadd_i8
);
auto_binop!(
    /// `QSUB8`/scalar: per-lane saturating `i8` subtraction.
    qsub_i8,
    i8 => i8,
    Dsp,
    qsub_i8_slice,
    super::auto_down::qsub_i8
);
auto_binop!(
    /// `SADD8`/scalar: per-lane wrapping `i8` addition.
    sadd_i8,
    i8 => i8,
    Dsp,
    sadd_i8_slice,
    super::auto_down::sadd_i8
);
auto_binop!(
    /// `SSUB8`/scalar: per-lane wrapping `i8` subtraction.
    ssub_i8,
    i8 => i8,
    Dsp,
    ssub_i8_slice,
    super::auto_down::ssub_i8
);
auto_binop!(
    /// `SHADD8`/scalar: per-lane halving signed `i8` addition (`(a+b)/2`, no
    /// saturation).
    shadd_i8,
    i8 => i8,
    Dsp,
    shadd_i8_slice,
    super::auto_down::shadd_i8
);
auto_binop!(
    /// `SHSUB8`/scalar: per-lane halving signed `i8` subtraction (`(a-b)/2`,
    /// no saturation).
    shsub_i8,
    i8 => i8,
    Dsp,
    shsub_i8_slice,
    super::auto_down::shsub_i8
);
auto_binop!(
    /// `USUB8`/scalar: per-lane wrapping unsigned `u8` subtraction.
    usub_u8,
    u8 => u8,
    Dsp,
    usub_u8_slice,
    super::auto_down::usub_u8
);

auto_binop!(
    /// `QADD16`/scalar: per-lane saturating `i16` addition, packed 2-lane.
    qadd_i16,
    i16 => i16,
    Dsp,
    qadd_i16_slice,
    super::auto_down::qadd_i16
);
auto_binop!(
    /// `QSUB16`/scalar: per-lane saturating `i16` subtraction, packed 2-lane.
    qsub_i16,
    i16 => i16,
    Dsp,
    qsub_i16_slice,
    super::auto_down::qsub_i16
);
auto_binop!(
    /// `SADD16`/scalar: per-lane wrapping `i16` addition, packed 2-lane.
    sadd_i16,
    i16 => i16,
    Dsp,
    sadd_i16_slice,
    super::auto_down::sadd_i16
);
auto_binop!(
    /// `SHADD16`/scalar: per-lane halving signed `i16` addition (no
    /// saturation), packed 2-lane.
    shadd_i16,
    i16 => i16,
    Dsp,
    shadd_i16_slice,
    super::auto_down::shadd_i16
);
auto_binop!(
    /// `SHSUB16`/scalar: per-lane halving signed `i16` subtraction (no
    /// saturation), packed 2-lane.
    shsub_i16,
    i16 => i16,
    Dsp,
    shsub_i16_slice,
    super::auto_down::shsub_i16
);
/// Dsp cross add-subtract (`QASX`/`QSAX`/`SASX`) as an `auto` fn. A cross
/// op needs both elements of a pair together, so an odd-length input has no
/// defined remainder pairing - **panics** on odd length (on every path, not
/// only when the token is present) rather than silently dropping or
/// misreading the trailing element.
macro_rules! auto_cross {
	($(#[$doc:meta])* $fn_name:ident, $slice_fn:ident, $scalar_fn:path) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len() || a.len() % 2 != 0`.
		#[inline]
		pub fn $fn_name(a: &[i16], b: &[i16], out: &mut [i16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			assert_eq!(a.len() % 2, 0, "cross add-subtract needs pairs; odd length has no defined remainder");
			match Dsp::from_features(super::detect_features()) {
				Some(t) => t.$slice_fn(a, b, out),
				None => $scalar_fn(a, b, out),
			}
		}
	};
}

auto_cross!(
    /// `QASX`/scalar: saturating cross add-subtract, packed pairs
    /// (`out[2k]=a[2k]-b[2k+1], out[2k+1]=a[2k+1]+b[2k]`).
    qasx_i16,
    qasx_i16_slice,
    super::auto_down::qasx_i16
);
auto_cross!(
    /// `QSAX`/scalar: saturating cross subtract-add, packed pairs
    /// (`out[2k]=a[2k]+b[2k+1], out[2k+1]=a[2k+1]-b[2k]`).
    qsax_i16,
    qsax_i16_slice,
    super::auto_down::qsax_i16
);
auto_cross!(
    /// `SASX`/scalar: wrapping cross add-subtract, packed pairs
    /// (`out[2k]=a[2k]-b[2k+1], out[2k+1]=a[2k+1]+b[2k]`).
    sasx_i16,
    sasx_i16_slice,
    super::auto_down::sasx_i16
);

#[cfg(test)]
#[path = "test/auto_up.rs"]
mod tests;
