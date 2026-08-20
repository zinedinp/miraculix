//! # Auto slice ops
//!
//! Elementwise slice work without picking a SIMD tier. Each fn uses
//! [`super::detect_features`] (not the coarse V1..V4 level), picks the best
//! path, and falls back to scalar when needed. Lower SSE/scalar rungs live in
//! private `auto_down`. Also reachable as [`super::auto`] (alias).
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::auto_up;
//! let a = [1i32, 2, 3, 4];
//! let b = [10, 20, 30, 40];
//! let mut out = [0i32; 4];
//! auto_up::add_i32(&a, &b, &mut out);
//! assert_eq!(out, [11, 22, 33, 44]);
//! # }
//! ```
//!
//! ## Shared contract
//!
//! - Lengths of inputs and `out` must match (or the op's documented ratio).
//!   Mismatch **panics**.
//! - Integer `add`/`sub`/`mul` are **wrapping** unless the name is `adds`/`subs`.
//! - `cmpeq`/`cmpgt`/... write **lane masks** (all-1s or 0), not `bool`.
//! - Float `select_*` uses the **sign bit** of the mask, not nonzero.
//! - `f16` values are raw `u16` IEEE half bit patterns.
//! - Names: `verb_type` (`add_i32`, `fmadd_f32`, `select_u8`).
//!
//! ## Float NaN
//!
//! - Ordered compares (`cmplt`/`cmpgt`/...): false if either lane is NaN.
//! - `cmpeq`: NaN never equals (mask 0).
//! - SIMD `min`/`max` (SSE/AVX/AVX-512 rungs): x86 **second-operand-on-NaN**
//!   (`minps`/`maxps` family), not Rust `f32::min` / `f64::min`.
//! - Scalar remainder and pure-scalar path: Rust `f32::min` / `f64::min`
//!   (and `max`). A short tail after a SIMD body can therefore disagree with
//!   the SIMD lanes on NaN inputs.
//! - `f16` min/max scalar path widens to `f32`, applies the rule above, narrows.

/// Must be used in tail position inside a `pub fn` (expands to `return`).
/// `$lifted` is a closure `|proof: Avx512FVl| -> _` (not a plain `expr`):
/// the `_lifted` token methods take the `Avx512FVl` proof as an explicit
/// parameter (they need more than the `Avx2`/etc. token they're an inherent
/// method on proves), and macro hygiene means `auto_up!` can't bind a name
/// into `$lifted`'s own scope
#[cfg(feature = "wider-bus-lift")]
macro_rules! auto_up {
	($lifted:expr, $plain:expr) => {{
		if let Some(proof) = crate::x86::ops::avx512::avx512vl::Avx512FVl::from_features(super::detect_features()) {
			return ($lifted)(proof);
		}
		return $plain;
	}};
}
#[cfg(not(feature = "wider-bus-lift"))]
macro_rules! auto_up {
	($lifted:expr, $plain:expr) => {{
		return $plain;
	}};
}
use super::ops::{
	avx::avx2::Avx2, avx::avx_ifma::AvxIfma, avx::avx_vnni::AvxVnni, avx::avx_vnni::vnni_acc_saturating,
	avx::avx_vnni::vnni_acc_wrapping, avx::f16c::F16c, avx::f16c::f16_to_f32_scalar, avx::f16c::f32_to_f16_scalar,
	avx::fma::Fma, avx512::avx512bf16::Avx512Bf16, avx512::avx512bf16::bf16_to_f32_scalar,
	avx512::avx512bf16::f32_to_bf16_scalar, avx512::avx512bitalg::Avx512Bitalg, avx512::avx512bw::Avx512Bw,
	avx512::avx512dq::Avx512Dq, avx512::avx512f::Avx512f, avx512::avx512fp16::Avx512Fp16, avx512::avx512fp16::Avx512Fp16Vl,
	avx512::avx512ifma::Avx512Ifma, avx512::avx512ifma::madd52hi_scalar, avx512::avx512ifma::madd52lo_scalar,
	avx512::avx512vnni::Avx512Vnni, avx512::avx512vpopcntdq::Avx512Vpopcntdq, other::popcnt::Popcnt,
};

/// Regular int binop, both `baseline` and `probed` bottom families share this
/// exact "up" shape (`Avx512f` -> `Avx2`[+lift]): only their bottom tier
/// (in `auto_down.rs`) differs. `lifted = ...,` arm for the 18 i32/u32 ops the
/// wider-bus lift covers; the plain arm for everything else.
macro_rules! auto_binop_up {
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal, lifted = $lifted:ident,) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512f::from_features(features) {
				return t.$slice_method(a, b, out);
			}
			if let Some(t) = Avx2::from_features(features) {
				auto_up!(|proof| t.$lifted(proof, a, b, out), t.$slice_method(a, b, out));
			}
			super::auto_down::$fn_name(a, b, out)
		}
	};
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512f::from_features(features) {
				return t.$slice_method(a, b, out);
			}
			if let Some(t) = Avx2::from_features(features) {
				return t.$slice_method(a, b, out);
			}
			super::auto_down::$fn_name(a, b, out)
		}
	};
}

/// f32/f64: only `Avx512f` up here (no `Avx2` rung: float width doesn't grow
/// from AVX to AVX2). `Avx`/SSE/scalar all live in `auto_down.rs`.
macro_rules! auto_f32f64_up {
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			if let Some(t) = Avx512f::from_features(super::detect_features()) {
				return t.$slice_method(a, b, out);
			}
			super::auto_down::$fn_name(a, b, out)
		}
	};
}

/// Narrow ints (i8/u8/i16/u16): `Avx512Bw` -> `Avx2`, no lift wired for this
/// family yet (tracked follow-up).
/// Shared by both the `baseline` and `probed` bottom families in `auto_down.rs`.
macro_rules! auto_binop_up_bw {
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512Bw::from_features(features) {
				return t.$slice_method(a, b, out);
			}
			if let Some(t) = Avx2::from_features(features) {
				return t.$slice_method(a, b, out);
			}
			super::auto_down::$fn_name(a, b, out)
		}
	};
}

/// Const-imm shift, `Avx512f` top (i32/u32 family).
macro_rules! auto_shift_imm_up {
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name<const IMM: u32>(a: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512f::from_features(features) {
				return t.$slice_method::<IMM>(a, out);
			}
			if let Some(t) = Avx2::from_features(features) {
				return t.$slice_method::<IMM>(a, out);
			}
			super::auto_down::$fn_name::<IMM>(a, out)
		}
	};
}

auto_f32f64_up!(add_f32, add_f32_slice, f32, "`out[i] = a[i] + b[i]`, best tier (AVX-512F / AVX / SSE / scalar).");
auto_f32f64_up!(sub_f32, sub_f32_slice, f32, "`out[i] = a[i] - b[i]`, best tier (AVX-512F / AVX / SSE / scalar).");
auto_f32f64_up!(mul_f32, mul_f32_slice, f32, "`out[i] = a[i] * b[i]`, best tier (AVX-512F / AVX / SSE / scalar).");
auto_f32f64_up!(div_f32, div_f32_slice, f32, "`out[i] = a[i] / b[i]`, best tier (AVX-512F / AVX / SSE / scalar).");
auto_f32f64_up!(min_f32, min_f32_slice, f32,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512F / AVX / SSE / scalar). NaN: see module doc.");
auto_f32f64_up!(max_f32, max_f32_slice, f32,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512F / AVX / SSE / scalar). NaN: see module doc.");
auto_f32f64_up!(and_f32, and_f32_slice, f32,
	"`out[i] = a[i] & b[i]` bitwise, best tier (AVX-512F / AVX / SSE / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(or_f32, or_f32_slice, f32,
	"`out[i] = a[i] | b[i]` bitwise, best tier (AVX-512F / AVX / SSE / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(xor_f32, xor_f32_slice, f32,
	"`out[i] = a[i] ^ b[i]` bitwise, best tier (AVX-512F / AVX / SSE / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(andnot_f32, andnot_f32_slice, f32,
	"`out[i] = !a[i] & b[i]` bitwise, best tier (AVX-512F / AVX / SSE / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(cmpeq_f32, cmpeq_f32_slice, f32,
	"`out[i] = all-1s bits if a[i]==b[i] else 0` (lane mask). Best tier. NaN never equals.");
auto_f32f64_up!(cmplt_f32, cmplt_f32_slice, f32, "`out[i] = all-1s bits if a[i]<b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmple_f32, cmple_f32_slice, f32, "`out[i] = all-1s bits if a[i]<=b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmpgt_f32, cmpgt_f32_slice, f32, "`out[i] = all-1s bits if a[i]>b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmpge_f32, cmpge_f32_slice, f32, "`out[i] = all-1s bits if a[i]>=b[i] else 0`. Best tier. False if either NaN.");

auto_f32f64_up!(add_f64, add_f64_slice, f64, "`out[i] = a[i] + b[i]`, best tier (AVX-512F / AVX / SSE2 / scalar).");
auto_f32f64_up!(sub_f64, sub_f64_slice, f64, "`out[i] = a[i] - b[i]`, best tier (AVX-512F / AVX / SSE2 / scalar).");
auto_f32f64_up!(mul_f64, mul_f64_slice, f64, "`out[i] = a[i] * b[i]`, best tier (AVX-512F / AVX / SSE2 / scalar).");
auto_f32f64_up!(div_f64, div_f64_slice, f64, "`out[i] = a[i] / b[i]`, best tier (AVX-512F / AVX / SSE2 / scalar).");
auto_f32f64_up!(min_f64, min_f64_slice, f64,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512F / AVX / SSE2 / scalar). NaN: see module doc.");
auto_f32f64_up!(max_f64, max_f64_slice, f64,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512F / AVX / SSE2 / scalar). NaN: see module doc.");
auto_f32f64_up!(and_f64, and_f64_slice, f64,
	"`out[i] = a[i] & b[i]` bitwise, best tier (AVX-512F / AVX / SSE2 / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(or_f64, or_f64_slice, f64,
	"`out[i] = a[i] | b[i]` bitwise, best tier (AVX-512F / AVX / SSE2 / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(xor_f64, xor_f64_slice, f64,
	"`out[i] = a[i] ^ b[i]` bitwise, best tier (AVX-512F / AVX / SSE2 / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(andnot_f64, andnot_f64_slice, f64,
	"`out[i] = !a[i] & b[i]` bitwise, best tier (AVX-512F / AVX / SSE2 / scalar). 512-bit uses `si512` bitcast, no AVX-512DQ needed.");
auto_f32f64_up!(cmpeq_f64, cmpeq_f64_slice, f64,
	"`out[i] = all-1s bits if a[i]==b[i] else 0` (lane mask). Best tier. NaN never equals.");
auto_f32f64_up!(cmplt_f64, cmplt_f64_slice, f64, "`out[i] = all-1s bits if a[i]<b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmple_f64, cmple_f64_slice, f64, "`out[i] = all-1s bits if a[i]<=b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmpgt_f64, cmpgt_f64_slice, f64, "`out[i] = all-1s bits if a[i]>b[i] else 0`. Best tier. False if either NaN.");
auto_f32f64_up!(cmpge_f64, cmpge_f64_slice, f64, "`out[i] = all-1s bits if a[i]>=b[i] else 0`. Best tier. False if either NaN.");

auto_binop_up!(add_i32, add_i32_slice, i32, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = add_i32_slice_lifted,);
auto_binop_up!(sub_i32, sub_i32_slice, i32, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = sub_i32_slice_lifted,);
auto_binop_up!(div_i32, div_i32_slice, i32,
	"`out[i] = a[i] / b[i]`, best tier for the surrounding buffer op (AVX-512F / AVX2 / SSE2 / scalar), but no tier actually vectorizes the divide itself: x86 SIMD has no integer divide instruction at any width. Panics on zero divisor or `i32::MIN / -1`, matching Rust's `/`.");
auto_binop_up!(mul_i32, mul_i32_slice, i32,
	"`out[i] = a[i].wrapping_mul(b[i])`, best tier (AVX-512F / AVX2 / SSE4.1 / scalar). SSE4.1 rung is a real probe, not ABI-baseline.",
	lifted = mul_i32_slice_lifted,);
auto_binop_up!(min_i32, min_i32_slice, i32,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512F / AVX2 / SSE2 composed / scalar). SSE2: `cmpgt`+and/or, not `pminsd`.");
auto_binop_up!(max_i32, max_i32_slice, i32,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512F / AVX2 / SSE2 composed / scalar). SSE2: `cmpgt`+and/or, not `pmaxsd`.");
auto_binop_up!(and_i32, and_i32_slice, i32, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = and_i32_slice_lifted,);
auto_binop_up!(or_i32, or_i32_slice, i32, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = or_i32_slice_lifted,);
auto_binop_up!(xor_i32, xor_i32_slice, i32, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = xor_i32_slice_lifted,);
auto_binop_up!(andnot_i32, andnot_i32_slice, i32, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = andnot_i32_slice_lifted,);
auto_binop_up!(cmpeq_i32, cmpeq_i32_slice, i32,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier.", lifted = cmpeq_i32_slice_lifted,);
auto_binop_up!(cmpgt_i32, cmpgt_i32_slice, i32,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask, not bool), best tier.", lifted = cmpgt_i32_slice_lifted,);
auto_binop_up!(cmplt_i32, cmplt_i32_slice, i32, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask, not bool), best tier.");
auto_binop_up!(cmple_i32, cmple_i32_slice, i32, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask, not bool), best tier.");
auto_binop_up!(cmpge_i32, cmpge_i32_slice, i32, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask, not bool), best tier.");

auto_binop_up!(add_u32, add_u32_slice, u32, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = add_u32_slice_lifted,);
auto_binop_up!(sub_u32, sub_u32_slice, u32, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = sub_u32_slice_lifted,);
auto_binop_up!(div_u32, div_u32_slice, u32,
	"`out[i] = a[i] / b[i]`, best tier for the surrounding buffer op (AVX-512F / AVX2 / SSE2 / scalar), but no tier actually vectorizes the divide itself: x86 SIMD has no integer divide instruction at any width. Panics on zero divisor, matching Rust's `/`.");
auto_binop_up!(mul_u32, mul_u32_slice, u32,
	"`out[i] = a[i].wrapping_mul(b[i])`, best tier (AVX-512F / AVX2 / SSE4.1 / scalar). SSE4.1 rung is a real probe, not ABI-baseline.",
	lifted = mul_u32_slice_lifted,);
auto_binop_up!(min_u32, min_u32_slice, u32,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512F / AVX2 / SSE2 composed / scalar). SSE2: sign-flip `cmpgt`+and/or.");
auto_binop_up!(max_u32, max_u32_slice, u32,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512F / AVX2 / SSE2 composed / scalar). SSE2: sign-flip `cmpgt`+and/or.");
auto_binop_up!(and_u32, and_u32_slice, u32, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = and_u32_slice_lifted,);
auto_binop_up!(or_u32, or_u32_slice, u32, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = or_u32_slice_lifted,);
auto_binop_up!(xor_u32, xor_u32_slice, u32, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = xor_u32_slice_lifted,);
auto_binop_up!(andnot_u32, andnot_u32_slice, u32, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).",
	lifted = andnot_u32_slice_lifted,);
auto_binop_up!(cmpeq_u32, cmpeq_u32_slice, u32,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier.", lifted = cmpeq_u32_slice_lifted,);
auto_binop_up!(cmpgt_u32, cmpgt_u32_slice, u32, "`out[i] = all-1s if a[i]>b[i] else 0` (lane mask, not bool), best tier.");
auto_binop_up!(cmplt_u32, cmplt_u32_slice, u32, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask, not bool), best tier.");
auto_binop_up!(cmple_u32, cmple_u32_slice, u32, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask, not bool), best tier.");
auto_binop_up!(cmpge_u32, cmpge_u32_slice, u32, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask, not bool), best tier.");

auto_binop_up!(add_i64, add_i64_slice, i64, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(sub_i64, sub_i64_slice, i64, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(add_u64, add_u64_slice, u64, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(sub_u64, sub_u64_slice, u64, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(and_i64, and_i64_slice, i64, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_i64, or_i64_slice, i64, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_i64, xor_i64_slice, i64, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_i64, andnot_i64_slice, i64, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(and_u64, and_u64_slice, u64, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_u64, or_u64_slice, u64, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_u64, xor_u64_slice, u64, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_u64, andnot_u64_slice, u64, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");

auto_binop_up!(cmpeq_i64, cmpeq_i64_slice, i64,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask). AVX-512F / AVX2 / SSE4.1 / scalar.");
auto_binop_up!(cmpeq_u64, cmpeq_u64_slice, u64,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask). AVX-512F / AVX2 / SSE4.1 / scalar.");
auto_binop_up!(cmpgt_i64, cmpgt_i64_slice, i64,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask). AVX-512F / AVX2 / SSE4.2 / scalar.");
auto_binop_up!(cmpgt_u64, cmpgt_u64_slice, u64,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask). AVX-512F / AVX2 / SSE4.2 / scalar.");
auto_binop_up!(cmplt_i64, cmplt_i64_slice, i64, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up!(cmple_i64, cmple_i64_slice, i64, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up!(cmpge_i64, cmpge_i64_slice, i64, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");
auto_binop_up!(cmplt_u64, cmplt_u64_slice, u64, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up!(cmple_u64, cmple_u64_slice, u64, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up!(cmpge_u64, cmpge_u64_slice, u64, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");

/// Const-imm shift, `Avx512Bw` top (i16/u16 family).
macro_rules! auto_shift_imm_up_bw {
	($fn_name:ident, $slice_method:ident, $Elem:ty, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name<const IMM: u32>(a: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512Bw::from_features(features) {
				return t.$slice_method::<IMM>(a, out);
			}
			if let Some(t) = Avx2::from_features(features) {
				return t.$slice_method::<IMM>(a, out);
			}
			super::auto_down::$fn_name::<IMM>(a, out)
		}
	};
}

auto_shift_imm_up!(shl_i32, shl_i32_slice, i32, "`out[i] = a[i] << IMM`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_shift_imm_up!(shr_i32, shr_i32_slice, i32, "`out[i] = a[i] logical >> IMM`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_shift_imm_up!(sra_i32, sra_i32_slice, i32, "`out[i] = a[i] arithmetic >> IMM`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_shift_imm_up!(shl_u32, shl_u32_slice, u32, "`out[i] = a[i] << IMM`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_shift_imm_up!(shr_u32, shr_u32_slice, u32, "`out[i] = a[i] >> IMM`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");

// i8: add/sub/adds/subs/cmpeq/bitwise bottom SSE2 (ABI-baseline); min/max bottom
// SSE4.1 (real probe: SSE2 only has u8/i16 min/max natively, not i8).
auto_binop_up_bw!(add_i8, add_i8_slice, i8, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(sub_i8, sub_i8_slice, i8, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(adds_i8, adds_i8_slice, i8,
	"`out[i] = a[i].saturating_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(subs_i8, subs_i8_slice, i8,
	"`out[i] = a[i].saturating_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpeq_i8, cmpeq_i8_slice, i8,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpgt_i8, cmpgt_i8_slice, i8,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmplt_i8, cmplt_i8_slice, i8, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up_bw!(cmple_i8, cmple_i8_slice, i8, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up_bw!(cmpge_i8, cmpge_i8_slice, i8, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");
auto_binop_up!(and_i8, and_i8_slice, i8, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_i8, or_i8_slice, i8, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_i8, xor_i8_slice, i8, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_i8, andnot_i8_slice, i8, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(min_i8, min_i8_slice, i8,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE4.1 / scalar). SSE4.1 is a real probe.");
auto_binop_up_bw!(max_i8, max_i8_slice, i8,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE4.1 / scalar). SSE4.1 is a real probe.");

// u8: same shape, but min/max also bottom SSE2 (pminub/pmaxub are native SSE2, unlike i8's pminsb).
auto_binop_up_bw!(add_u8, add_u8_slice, u8, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(sub_u8, sub_u8_slice, u8, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(adds_u8, adds_u8_slice, u8,
	"`out[i] = a[i].saturating_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(subs_u8, subs_u8_slice, u8,
	"`out[i] = a[i].saturating_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpeq_u8, cmpeq_u8_slice, u8,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpgt_u8, cmpgt_u8_slice, u8,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask). SSE2/AVX2: sign-bit flip; BW: native.");
auto_binop_up_bw!(cmplt_u8, cmplt_u8_slice, u8, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up_bw!(cmple_u8, cmple_u8_slice, u8, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up_bw!(cmpge_u8, cmpge_u8_slice, u8, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");
auto_binop_up!(and_u8, and_u8_slice, u8, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_u8, or_u8_slice, u8, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_u8, xor_u8_slice, u8, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_u8, andnot_u8_slice, u8, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(min_u8, min_u8_slice, u8, "`out[i] = min(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(max_u8, max_u8_slice, u8, "`out[i] = max(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(avg_u8, avg_u8_slice, u8,
	"`out[i] = (a[i] as u16 + b[i] as u16 + 1) / 2`, best tier (AVX-512BW / AVX2 / SSE2 / scalar). No signed form exists in the ISA.");

// i16: add/sub/adds/subs/cmpeq/bitwise/mul/min/max all bottom SSE2 (all native there).
auto_binop_up_bw!(add_i16, add_i16_slice, i16, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(sub_i16, sub_i16_slice, i16, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(adds_i16, adds_i16_slice, i16,
	"`out[i] = a[i].saturating_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(subs_i16, subs_i16_slice, i16,
	"`out[i] = a[i].saturating_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpeq_i16, cmpeq_i16_slice, i16,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpgt_i16, cmpgt_i16_slice, i16,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmplt_i16, cmplt_i16_slice, i16, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up_bw!(cmple_i16, cmple_i16_slice, i16, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up_bw!(cmpge_i16, cmpge_i16_slice, i16, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");
auto_binop_up!(and_i16, and_i16_slice, i16, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_i16, or_i16_slice, i16, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_i16, xor_i16_slice, i16, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_i16, andnot_i16_slice, i16, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(mul_i16, mul_i16_slice, i16, "`out[i] = a[i].wrapping_mul(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(min_i16, min_i16_slice, i16, "`out[i] = min(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(max_i16, max_i16_slice, i16, "`out[i] = max(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");

// u16: same shape, but min/max bottom SSE4.1 (pminuw is not native SSE2, unlike i16's pminsw).
auto_binop_up_bw!(add_u16, add_u16_slice, u16, "`out[i] = a[i].wrapping_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(sub_u16, sub_u16_slice, u16, "`out[i] = a[i].wrapping_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(adds_u16, adds_u16_slice, u16,
	"`out[i] = a[i].saturating_add(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(subs_u16, subs_u16_slice, u16,
	"`out[i] = a[i].saturating_sub(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpeq_u16, cmpeq_u16_slice, u16,
	"`out[i] = all-1s if a[i]==b[i] else 0` (lane mask, not bool), best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(cmpgt_u16, cmpgt_u16_slice, u16,
	"`out[i] = all-1s if a[i]>b[i] else 0` (lane mask). SSE2/AVX2: sign-bit flip; BW: native.");
auto_binop_up_bw!(cmplt_u16, cmplt_u16_slice, u16, "`out[i] = all-1s if a[i]<b[i] else 0` (lane mask). Derived swap of cmpgt.");
auto_binop_up_bw!(cmple_u16, cmple_u16_slice, u16, "`out[i] = all-1s if a[i]<=b[i] else 0` (lane mask). Derived NOT of cmpgt.");
auto_binop_up_bw!(cmpge_u16, cmpge_u16_slice, u16, "`out[i] = all-1s if a[i]>=b[i] else 0` (lane mask). Derived NOT of cmplt.");
auto_binop_up!(and_u16, and_u16_slice, u16, "`out[i] = a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(or_u16, or_u16_slice, u16, "`out[i] = a[i] | b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(xor_u16, xor_u16_slice, u16, "`out[i] = a[i] ^ b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up!(andnot_u16, andnot_u16_slice, u16, "`out[i] = !a[i] & b[i]`, best tier (AVX-512F / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(mul_u16, mul_u16_slice, u16, "`out[i] = a[i].wrapping_mul(b[i])`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_binop_up_bw!(min_u16, min_u16_slice, u16,
	"`out[i] = min(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE4.1 / scalar). SSE4.1 is a real probe.");
auto_binop_up_bw!(max_u16, max_u16_slice, u16,
	"`out[i] = max(a[i], b[i])`, best tier (AVX-512BW / AVX2 / SSE4.1 / scalar). SSE4.1 is a real probe.");
auto_binop_up_bw!(avg_u16, avg_u16_slice, u16,
	"`out[i] = (a[i] as u32 + b[i] as u32 + 1) / 2`, best tier (AVX-512BW / AVX2 / SSE2 / scalar). No signed form exists in the ISA.");

// i16/u16 shifts, bottom SSE2 (native there; no i8/u8 shift cascade, no HW path below AVX2 for those).
auto_shift_imm_up_bw!(shl_i16, shl_i16_slice, i16, "`out[i] = a[i] << IMM`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_shift_imm_up_bw!(shr_i16, shr_i16_slice, i16, "`out[i] = a[i] logical >> IMM`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_shift_imm_up_bw!(sra_i16, sra_i16_slice, i16, "`out[i] = a[i] arithmetic >> IMM`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_shift_imm_up_bw!(shl_u16, shl_u16_slice, u16, "`out[i] = a[i] << IMM`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");
auto_shift_imm_up_bw!(shr_u16, shr_u16_slice, u16, "`out[i] = a[i] >> IMM`, best tier (AVX-512BW / AVX2 / SSE2 / scalar).");

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (AVX-512F / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// No SSE2 rung (`blendv` is SSE4.1+). `mask`: all-0/all-1 lanes (e.g. [`cmpeq_i32`]/[`cmpgt_i32`]).
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_i32(a: &[i32], b: &[i32], mask: &[i32], out: &mut [i32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_i32_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_i32_slice(a, b, mask, out);
	}
	super::auto_down::select_i32(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u32` view). Same cascade as [`select_i32`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_u32(a: &[u32], b: &[u32], mask: &[u32], out: &mut [u32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_u32_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_u32_slice(a, b, mask, out);
	}
	super::auto_down::select_u32(a, b, mask, out)
}

/// `out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }` (AVX-512F / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// Float: sign bit selects, not a nonzero test.
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_f32(a: &[f32], b: &[f32], mask: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_f32_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_f32_slice(a, b, mask, out);
	}
	super::auto_down::select_f32(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (AVX-512F / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// No SSE2 rung. `mask`: all-0/all-1 lanes (e.g. [`cmpeq_i64`]/[`cmpgt_i64`]).
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_i64(a: &[i64], b: &[i64], mask: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_i64_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_i64_slice(a, b, mask, out);
	}
	super::auto_down::select_i64(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u64` view). Same cascade as [`select_i64`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_u64(a: &[u64], b: &[u64], mask: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_u64_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_u64_slice(a, b, mask, out);
	}
	super::auto_down::select_u64(a, b, mask, out)
}

/// `out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }` (AVX-512F / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// Float: sign bit selects, not a nonzero test.
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_f64(a: &[f64], b: &[f64], mask: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.select_f64_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_f64_slice(a, b, mask, out);
	}
	super::auto_down::select_f64(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (AVX-512BW / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// No SSE2 rung (`blendv` is SSE4.1+). `mask`: all-0/all-1 lanes (e.g. [`cmpeq_i8`]).
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_i8(a: &[i8], b: &[i8], mask: &[i8], out: &mut [i8]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.select_i8_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_i8_slice(a, b, mask, out);
	}
	super::auto_down::select_i8(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u8` view). Same cascade as [`select_i8`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_u8(a: &[u8], b: &[u8], mask: &[u8], out: &mut [u8]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.select_u8_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_u8_slice(a, b, mask, out);
	}
	super::auto_down::select_u8(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (AVX-512BW / AVX2 / [`super::ops::sse::sse41::Sse41`] / scalar).
/// No SSE2 rung. `mask`: all-0/all-1 lanes (e.g. [`cmpeq_i16`]).
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_i16(a: &[i16], b: &[i16], mask: &[i16], out: &mut [i16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.select_i16_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_i16_slice(a, b, mask, out);
	}
	super::auto_down::select_i16(a, b, mask, out)
}

/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u16` view). Same cascade as [`select_i16`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `mask`, `out`.
#[inline]
pub fn select_u16(a: &[u16], b: &[u16], mask: &[u16], out: &mut [u16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), mask.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.select_u16_slice(a, b, mask, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.select_u16_slice(a, b, mask, out);
	}
	super::auto_down::select_u16(a, b, mask, out)
}

/// `out[i] = min(a[i], b[i])`. Best-tier cascade: AVX-512F -> AVX2 -> SSE4.2 -> scalar.
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn min_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.min_i64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.min_i64_slice(a, b, out);
	}
	super::auto_down::min_i64(a, b, out)
}

/// `out[i] = max(a[i], b[i])`: same cascade as [`min_i64`] (`max(a,b) = blendv(a, b, cmpgt(b,a))`).
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn max_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.max_i64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.max_i64_slice(a, b, out);
	}
	super::auto_down::max_i64(a, b, out)
}

/// `out[i] = min(a[i], b[i])` (`u64`). Same cascade as `min_i64`, with a sign-bit flip
/// to implement unsigned ordering.
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn min_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.min_u64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.min_u64_slice(a, b, out);
	}
	super::auto_down::min_u64(a, b, out)
}

/// `out[i] = max(a[i], b[i])` (`u64`): same cascade as [`min_u64`].
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn max_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.max_u64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.max_u64_slice(a, b, out);
	}
	super::auto_down::max_u64(a, b, out)
}

/// `out[i] = a[i].wrapping_mul(b[i])` (low 64 bits). Cascade: AVX-512DQ -> AVX2 -> SSE2 -> scalar.
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn mullo_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Dq::from_features(features) {
		return t.mullo_i64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.mullo_i64_slice(a, b, out);
	}
	super::auto_down::mullo_i64(a, b, out)
}

/// `out[i] = a[i].wrapping_mul(b[i])`, low 64 bits: AVX-512DQ -> AVX2 -> SSE2 -> scalar.
/// Same cascade as [`mullo_i64`].
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn mullo_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Dq::from_features(features) {
		return t.mullo_u64_slice(a, b, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.mullo_u64_slice(a, b, out);
	}
	super::auto_down::mullo_u64(a, b, out)
}

/// `out[i] = a[i].wrapping_abs()`: best tier (AVX-512F / AVX2 / SSSE3 / scalar). SSSE3 is a real probe.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn abs_i32(a: &[i32], out: &mut [i32]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.abs_i32_slice(a, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.abs_i32_slice(a, out);
	}
	super::auto_down::abs_i32(a, out)
}

/// `out[i] = a[i].wrapping_abs()`: AVX-512F (native `vpabsq`) -> AVX2 -> SSE2 (both
/// composed: branchless sign-broadcast, `shuffle`+`srai`) -> scalar. No native 64-bit
/// arithmetic shift exists below AVX-512, but the composed form reconstructs abs at
/// every lower tier: not actually tier-unique, unlike the F-native instruction itself.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn abs_i64(a: &[i64], out: &mut [i64]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.abs_i64_slice(a, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.abs_i64_slice(a, out);
	}
	super::auto_down::abs_i64(a, out)
}

/// `out[i] = a[i].wrapping_abs()`: best tier (AVX-512BW / AVX2 / SSSE3 / scalar). SSSE3 is a real probe.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn abs_i8(a: &[i8], out: &mut [i8]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.abs_i8_slice(a, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.abs_i8_slice(a, out);
	}
	super::auto_down::abs_i8(a, out)
}

/// `out[i] = a[i].wrapping_abs()`: best tier (AVX-512BW / AVX2 / SSSE3 / scalar). SSSE3 is a real probe.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn abs_i16(a: &[i16], out: &mut [i16]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bw::from_features(features) {
		return t.abs_i16_slice(a, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.abs_i16_slice(a, out);
	}
	super::auto_down::abs_i16(a, out)
}

/// `out[i] = a[i].wrapping_mul(b[i])`: AVX2 -> SSE2, both composed (zero-extend,
/// `pmullw`/`vpmullw`, `packuswb`/`vpackuswb`; no native 8-bit SIMD multiply
/// exists on x86
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn mul_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.mul_i8_slice(a, b, out);
	}
	super::auto_down::mul_i8(a, b, out)
}

/// `out[i] = a[i].wrapping_mul(b[i])`
///
/// # Panics
/// `a.len() != b.len() || out.len() != a.len()`.
#[inline]
pub fn mul_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.mul_u8_slice(a, b, out);
	}
	super::auto_down::mul_u8(a, b, out)
}

/// `out[i] = a[i].wrapping_shl(IMM)`: AVX2 -> SSE2, both composed (widen to
/// 16-bit lanes + a byte-repeated mask; no native byte-granularity shift
/// exists on x86
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn shl_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.shl_i8_slice::<IMM>(a, out);
	}
	super::auto_down::shl_i8::<IMM>(a, out)
}

/// `out[i] = a[i].wrapping_shl(IMM)`
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn shl_u8<const IMM: u32>(a: &[u8], out: &mut [u8]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.shl_u8_slice::<IMM>(a, out);
	}
	super::auto_down::shl_u8::<IMM>(a, out)
}

/// `out[i] = a[i].wrapping_shr(IMM)` (logical
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn shr_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.shr_i8_slice::<IMM>(a, out);
	}
	super::auto_down::shr_i8::<IMM>(a, out)
}

/// `out[i] = a[i].wrapping_shr(IMM)`
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn shr_u8<const IMM: u32>(a: &[u8], out: &mut [u8]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.shr_u8_slice::<IMM>(a, out);
	}
	super::auto_down::shr_u8::<IMM>(a, out)
}

/// `out[i] = a[i].wrapping_shr(IMM)` (arithmetic): AVX2 -> SSE2, both composed
/// (per-byte sign extension + `psraw`/`vpsraw` + `packsswb`/`vpacksswb`). No
/// AVX-512
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn sra_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx2::from_features(super::detect_features()) {
		return t.sra_i8_slice::<IMM>(a, out);
	}
	super::auto_down::sra_i8::<IMM>(a, out)
}

/// `out[i] = a[i] << count[i]` (per-lane shift). Cascade: AVX-512F -> AVX2 -> scalar.
/// No SSE2 rung; counts >= machine-width zero the lane.
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn sllv_i64(a: &[i64], count: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.sllv_i64_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.sllv_i64_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u64) >= 64 { 0 } else { x.wrapping_shl(c as u32) };
	}
}

/// `out[i] = a[i] >> count[i]` logical. Same cascade as [`sllv_i64`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srlv_i64(a: &[i64], count: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.srlv_i64_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.srlv_i64_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u64) >= 64 {
			0
		} else {
			((x as u64).wrapping_shr(c as u32)) as i64
		};
	}
}

/// `out[i] = a[i] >> count[i]` arithmetic. AVX-512F only (no `vpsravq` at AVX2) -> scalar.
/// `count>=64` sign-fills.
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srav_i64(a: &[i64], count: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx512f::from_features(super::detect_features()) {
		return t.srav_i64_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u64) >= 64 {
			x >> 63
		} else {
			x.wrapping_shr(c as u32)
		};
	}
}

/// `out[i] = a[i] << count[i]` (`u64` view). Same cascade as [`sllv_i64`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn sllv_u64(a: &[u64], count: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.sllv_u64_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.sllv_u64_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if c >= 64 { 0 } else { x.wrapping_shl(c as u32) };
	}
}

/// `out[i] = a[i] >> count[i]` logical (`u64` view). Same cascade as [`sllv_i64`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srlv_u64(a: &[u64], count: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.srlv_u64_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.srlv_u64_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if c >= 64 { 0 } else { x.wrapping_shr(c as u32) };
	}
}

/// `out[i] = a[i] << count[i]`, `count` a per-lane vector (AVX-512F / AVX2 / scalar).
/// No SSE2 rung: `sllv`/`srlv`/`srav` need AVX2 CPUID even at 128-bit. `count>=32` zeroes,
/// matching x86 semantics (not Rust's wrapping-count `wrapping_shl`).
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn sllv_i32(a: &[i32], count: &[i32], out: &mut [i32]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.sllv_i32_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.sllv_i32_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u32) >= 32 { 0 } else { x.wrapping_shl(c as u32) };
	}
}

/// `out[i] = a[i] >> count[i]` logical, `count` a per-lane vector. Same cascade as [`sllv_i32`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srlv_i32(a: &[i32], count: &[i32], out: &mut [i32]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.srlv_i32_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.srlv_i32_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u32) >= 32 { 0 } else { ((x as u32).wrapping_shr(c as u32)) as i32 };
	}
}

/// `out[i] = a[i] >> count[i]` arithmetic, `count` a per-lane vector. `count>=32` sign-fills
/// (no unsigned form exists: arithmetic shift is meaningless for `u32`).
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srav_i32(a: &[i32], count: &[i32], out: &mut [i32]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.srav_i32_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.srav_i32_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if (c as u32) >= 32 { x >> 31 } else { x.wrapping_shr(c as u32) };
	}
}

/// `out[i] = a[i] << count[i]`, `count` a per-lane vector (`u32` view). Same cascade as [`sllv_i32`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn sllv_u32(a: &[u32], count: &[u32], out: &mut [u32]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.sllv_u32_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.sllv_u32_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if c >= 32 { 0 } else { x.wrapping_shl(c) };
	}
}

/// `out[i] = a[i] >> count[i]`, `count` a per-lane vector (`u32` view). Same cascade as [`sllv_i32`].
///
/// # Panics
/// `a.len() != count.len() || out.len() != a.len()`.
#[inline]
pub fn srlv_u32(a: &[u32], count: &[u32], out: &mut [u32]) {
	assert_eq!(a.len(), count.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.srlv_u32_slice(a, count, out);
	}
	if let Some(t) = Avx2::from_features(features) {
		return t.srlv_u32_slice(a, count, out);
	}
	for ((&x, &c), o) in a.iter().zip(count).zip(out.iter_mut()) {
		*o = if c >= 32 { 0 } else { x.wrapping_shr(c) };
	}
}

/// `out[i] = a[i] * b[i] + c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fmadd_f32_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		// FMA3 implies AVX; use 256-bit path.
		return t.fmadd_f32x8_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = x * y + z;
	}
}

/// `out[i] = a[i] * b[i] + c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmadd_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fmadd_f64_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fmadd_f64x4_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = x * y + z;
	}
}

/// `out[i] = a[i] * b[i] - c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmsub_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fmsub_f32_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fmsub_f32x8_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = x * y - z;
	}
}

/// `out[i] = a[i] * b[i] - c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmsub_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fmsub_f64_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fmsub_f64x4_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = x * y - z;
	}
}

/// `out[i] = -(a[i] * b[i]) + c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fnmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fnmadd_f32_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fnmadd_f32x8_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = -(x * y) + z;
	}
}

/// `out[i] = -(a[i] * b[i]) + c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fnmadd_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fnmadd_f64_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fnmadd_f64x4_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = -(x * y) + z;
	}
}

/// `out[i] = -(a[i] * b[i]) - c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fnmsub_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fnmsub_f32_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fnmsub_f32x8_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = -(x * y) - z;
	}
}

/// `out[i] = -(a[i] * b[i]) - c[i]` (HW fused when FMA/AVX-512F). Scalar remainder is not fused.
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fnmsub_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512f::from_features(features) {
		return t.fnmsub_f64_slice(a, b, c, out);
	}
	if let Some(t) = Fma::from_features(features) {
		return t.fnmsub_f64x4_slice(a, b, c, out);
	}
	for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
		*o = -(x * y) - z;
	}
}

/// `out[i] = a[i] + sum_n(b[n][i] * c[n])`: Xeon-Phi-only AVX512_4FMAPS
/// (`VP4FMADDPS`), software-composed as 4 folded `Avx512f` `fmadd_f32`
/// calls, no `avx512_4fmaps` CPUID bit needed. Scalar fallback below
/// `Avx512f` (no lower-tier HW form to cascade through).
///
/// # Panics
/// `a`/`out`/every `b[n]` length mismatch.
#[inline]
pub fn p4fmadd_f32(a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
	assert_eq!(out.len(), a.len());
	for bn in &b {
		assert_eq!(bn.len(), a.len());
	}
	if let Some(t) = Avx512f::from_features(super::detect_features()) {
		return t.p4fmadd_f32_slice(a, b, c, out);
	}
	for (i, (&x, o)) in a.iter().zip(out.iter_mut()).enumerate() {
		let mut acc = x;
		for n in 0..4 {
			acc += b[n][i] * c[n];
		}
		*o = acc;
	}
}

/// `out[i] = a[i] - sum_n(b[n][i] * c[n])`: Xeon-Phi-only AVX512_4FMAPS
/// (`VP4FNMADDPS`), software-composed as 4 folded `Avx512f` `fnmadd_f32`
/// calls, no `avx512_4fmaps` CPUID bit needed. Scalar fallback below
/// `Avx512f` (no lower-tier HW form to cascade through).
///
/// # Panics
/// `a`/`out`/every `b[n]` length mismatch.
#[inline]
pub fn p4fnmadd_f32(a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
	assert_eq!(out.len(), a.len());
	for bn in &b {
		assert_eq!(bn.len(), a.len());
	}
	if let Some(t) = Avx512f::from_features(super::detect_features()) {
		return t.p4fnmadd_f32_slice(a, b, c, out);
	}
	for (i, (&x, o)) in a.iter().zip(out.iter_mut()).enumerate() {
		let mut acc = x;
		for n in 0..4 {
			acc -= b[n][i] * c[n];
		}
		*o = acc;
	}
}

// FP16: AVX512FP16 (512-bit) -> AVX512FP16+VL (128/256-bit) -> scalar via f32.
// No pre-AVX-512 HW form at any width (F16C only converts FP16<->f32, it
// doesn't compute in FP16), same gap as `dpbf16_ps_f32`/`mullo_i64`.
macro_rules! auto_fp16_binop {
	($fn_name:ident, $slice_512:ident, $slice_256:ident, $scalar:expr, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[u16], b: &[u16], out: &mut [u16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512Fp16::from_features(features) {
				return t.$slice_512(a, b, out);
			}
			if let Some(t) = Avx512Fp16Vl::from_features(features) {
				return t.$slice_256(a, b, out);
			}
			let op: fn(u16, u16) -> u16 = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
}
macro_rules! auto_fp16_unop {
	($fn_name:ident, $slice_512:ident, $slice_256:ident, $scalar:expr, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// `out.len() != a.len()`.
		#[inline]
		pub fn $fn_name(a: &[u16], out: &mut [u16]) {
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512Fp16::from_features(features) {
				return t.$slice_512(a, out);
			}
			if let Some(t) = Avx512Fp16Vl::from_features(features) {
				return t.$slice_256(a, out);
			}
			let op: fn(u16) -> u16 = $scalar;
			for (&x, o) in a.iter().zip(out.iter_mut()) {
				*o = op(x);
			}
		}
	};
}
macro_rules! auto_fp16_ternop {
	($fn_name:ident, $slice_512:ident, $slice_256:ident, $scalar:expr, $doc:literal) => {
		#[doc = $doc]
		///
		/// # Panics
		/// Length mismatch among `a`, `b`, `c`, `out`.
		#[inline]
		pub fn $fn_name(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(a.len(), c.len());
			assert_eq!(out.len(), a.len());
			let features = super::detect_features();
			if let Some(t) = Avx512Fp16::from_features(features) {
				return t.$slice_512(a, b, c, out);
			}
			if let Some(t) = Avx512Fp16Vl::from_features(features) {
				return t.$slice_256(a, b, c, out);
			}
			let op: fn(u16, u16, u16) -> u16 = $scalar;
			for (((&x, &y), &z), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
				*o = op(x, y, z);
			}
		}
	};
}

auto_fp16_binop!(add_f16, add_ph_u16x32_slice, add_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x) + f16_to_f32_scalar(y)),
	"`out[i] = a[i] + b[i]` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_binop!(sub_f16, sub_ph_u16x32_slice, sub_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x) - f16_to_f32_scalar(y)),
	"`out[i] = a[i] - b[i]` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_binop!(mul_f16, mul_ph_u16x32_slice, mul_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y)),
	"`out[i] = a[i] * b[i]` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_binop!(div_f16, div_ph_u16x32_slice, div_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x) / f16_to_f32_scalar(y)),
	"`out[i] = a[i] / b[i]` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_binop!(min_f16, min_ph_u16x32_slice, min_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x).min(f16_to_f32_scalar(y))),
	"`out[i] = min(a[i], b[i])` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`). NaN: see module doc.");
auto_fp16_binop!(max_f16, max_ph_u16x32_slice, max_ph_u16x16_slice,
	|x, y| f32_to_f16_scalar(f16_to_f32_scalar(x).max(f16_to_f32_scalar(y))),
	"`out[i] = max(a[i], b[i])` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`). NaN: see module doc.");
auto_fp16_unop!(abs_f16, abs_ph_u16x32_slice, abs_ph_u16x16_slice, |x| f32_to_f16_scalar(f16_to_f32_scalar(x).abs()),
	"`out[i] = |a[i]|` (FP16 bit patterns), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_ternop!(fmadd_f16, fmadd_ph_u16x32_slice, fmadd_ph_u16x16_slice,
	|x, y, z| f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y) + f16_to_f32_scalar(z)),
	"`out[i] = a[i]*b[i] + c[i]` (FP16 bit patterns, HW fused above scalar), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_ternop!(fmsub_f16, fmsub_ph_u16x32_slice, fmsub_ph_u16x16_slice,
	|x, y, z| f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y) - f16_to_f32_scalar(z)),
	"`out[i] = a[i]*b[i] - c[i]` (FP16 bit patterns, HW fused above scalar), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_ternop!(fnmadd_f16, fnmadd_ph_u16x32_slice, fnmadd_ph_u16x16_slice,
	|x, y, z| f32_to_f16_scalar(-(f16_to_f32_scalar(x) * f16_to_f32_scalar(y)) + f16_to_f32_scalar(z)),
	"`out[i] = -(a[i]*b[i]) + c[i]` (FP16 bit patterns, HW fused above scalar), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");
auto_fp16_ternop!(fnmsub_f16, fnmsub_ph_u16x32_slice, fnmsub_ph_u16x16_slice,
	|x, y, z| f32_to_f16_scalar(-(f16_to_f32_scalar(x) * f16_to_f32_scalar(y)) - f16_to_f32_scalar(z)),
	"`out[i] = -(a[i]*b[i]) - c[i]` (FP16 bit patterns, HW fused above scalar), best tier (AVX-512FP16 / AVX-512FP16+VL / scalar via `f32`).");

/// `out[j] = a[j]*b[j] -/+ c[j]` alternating by lane parity: AVX-512FP16 ->
/// scalar via `f32` (512-bit only: no 128/256-bit wrapper yet, unlike the
/// rest of this family). Not `auto_fp16_ternop!`-shaped (the scalar closure
/// needs the lane index for the alternation), same gap as
/// [`super::ops::avx512::avx512fp16::Avx512Fp16::fmaddsub_ph_u16_slice`].
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmaddsub_f16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx512Fp16::from_features(super::detect_features()) {
		return t.fmaddsub_ph_u16_slice(a, b, c, out);
	}
	for (j, (((&x, &y), &z), o)) in a.iter().zip(b).zip(c).zip(out.iter_mut()).enumerate() {
		let prod = f16_to_f32_scalar(x) * f16_to_f32_scalar(y);
		let cc = f16_to_f32_scalar(z);
		*o = f32_to_f16_scalar(if j & 1 == 0 { prod - cc } else { prod + cc });
	}
}

/// `out[j] = a[j]*b[j] +/- c[j]` alternating by lane parity: same cascade
/// shape as [`fmaddsub_f16`] (512-bit only).
///
/// # Panics
/// Length mismatch among `a`, `b`, `c`, `out`.
#[inline]
pub fn fmsubadd_f16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), c.len());
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx512Fp16::from_features(super::detect_features()) {
		return t.fmsubadd_ph_u16_slice(a, b, c, out);
	}
	for (j, (((&x, &y), &z), o)) in a.iter().zip(b).zip(c).zip(out.iter_mut()).enumerate() {
		let prod = f16_to_f32_scalar(x) * f16_to_f32_scalar(y);
		let cc = f16_to_f32_scalar(z);
		*o = f32_to_f16_scalar(if j & 1 == 0 { prod + cc } else { prod - cc });
	}
}

/// `out[i] = a[i].count_ones()`: AVX512VPOPCNTDQ -> POPCNT GPR -> portable.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn popcnt_u32(a: &[u32], out: &mut [u32]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vpopcntdq::from_features(features) {
		return t.popcnt_u32_slice(a, out);
	}
	if let Some(t) = Popcnt::from_features(features) {
		return t.popcnt_u32_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.count_ones();
	}
}

/// `out[i] = a[i].count_ones()` as `u64`: AVX512VPOPCNTDQ -> POPCNT GPR -> portable.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn popcnt_u64(a: &[u64], out: &mut [u64]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vpopcntdq::from_features(features) {
		return t.popcnt_u64_slice(a, out);
	}
	if let Some(t) = Popcnt::from_features(features) {
		return t.popcnt_u64_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.count_ones() as u64;
	}
}

/// `out[i] = a[i].count_ones()`: AVX512BITALG -> POPCNT via `u32` widen -> portable.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn popcnt_u8(a: &[u8], out: &mut [u8]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bitalg::from_features(features) {
		return t.popcnt_u8_slice(a, out);
	}
	if let Some(t) = Popcnt::from_features(features) {
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = t.popcnt_u32(x as u32) as u8;
		}
		return;
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.count_ones() as u8;
	}
}

/// `out[i] = a[i].count_ones()`: AVX512BITALG -> POPCNT via `u32` widen -> portable.
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn popcnt_u16(a: &[u16], out: &mut [u16]) {
	assert_eq!(out.len(), a.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Bitalg::from_features(features) {
		return t.popcnt_u16_slice(a, out);
	}
	if let Some(t) = Popcnt::from_features(features) {
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = t.popcnt_u32(x as u32) as u16;
		}
		return;
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.count_ones() as u16;
	}
}

/// `out[i] = f16_to_f32(a[i])`: F16C -> portable scalar. No lower-tier SIMD
/// form exists (F16C is the whole story on x86).
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn f16_to_f32(a: &[u16], out: &mut [f32]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = F16c::from_features(super::detect_features()) {
		return t.f16_to_f32_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = f16_to_f32_scalar(x);
	}
}

/// `out[i] = f32_to_f16(a[i])`, RNE: F16C -> portable scalar. Same gap as
/// [`f16_to_f32`]; always RNE (matches the scalar fallback's fixed rounding,
/// unlike [`super::ops::avx::f16c::F16c::f32_to_f16x8`]'s `ROUNDING` param).
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn f32_to_f16(a: &[f32], out: &mut [u16]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = F16c::from_features(super::detect_features()) {
		const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
		return t.f32_to_f16_slice::<ROUNDING>(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = f32_to_f16_scalar(x);
	}
}

/// `out[j] = src[j] + bf16(a[2j+1])*bf16(b[2j+1]) + bf16(a[2j])*bf16(b[2j])`:
/// AVX512BF16 -> portable scalar. No lower-tier SIMD form exists (same shape
/// as [`mullo_i64`]'s `Avx512Dq`->scalar gap).
///
/// # Panics
/// `a.len() != b.len() || a.len() != 2*src.len() || out.len() != src.len()`.
#[inline]
pub fn dpbf16_ps_f32(src: &[f32], a: &[u16], b: &[u16], out: &mut [f32]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), 2 * src.len());
	assert_eq!(out.len(), src.len());
	if let Some(t) = Avx512Bf16::from_features(super::detect_features()) {
		return t.dpbf16_ps_f32_slice(src, a, b, out);
	}
	for (j, o) in out.iter_mut().enumerate() {
		let mut acc = src[j];
		acc += bf16_to_f32_scalar(a[2 * j + 1]) * bf16_to_f32_scalar(b[2 * j + 1]);
		acc += bf16_to_f32_scalar(a[2 * j]) * bf16_to_f32_scalar(b[2 * j]);
		*o = acc;
	}
}

/// `out[i] = f32_to_bf16(a[i])`, RNE: AVX512BF16 -> portable scalar. Same
/// gap as [`dpbf16_ps_f32`].
///
/// # Panics
/// `out.len() != a.len()`.
#[inline]
pub fn cvtneps_pbh_u16(a: &[f32], out: &mut [u16]) {
	assert_eq!(out.len(), a.len());
	if let Some(t) = Avx512Bf16::from_features(super::detect_features()) {
		return t.cvtneps_pbh_u16_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = f32_to_bf16_scalar(x);
	}
}

/// `out[j] = f32_to_bf16(b[j])` for `j < a.len()`, `f32_to_bf16(a[j-a.len()])`
/// for `j >= a.len()`: AVX512BF16 -> portable scalar. Same gap as
/// [`dpbf16_ps_f32`]; lane order matches
/// [`super::ops::avx512::avx512bf16::Avx512Bf16::cvtne2ps_pbh_u16x32`].
///
/// # Panics
/// `a.len() != b.len() || out.len() != 2*a.len()`.
#[inline]
pub fn cvtne2ps_pbh_u16(a: &[f32], b: &[f32], out: &mut [u16]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(out.len(), 2 * a.len());
	if let Some(t) = Avx512Bf16::from_features(super::detect_features()) {
		return t.cvtne2ps_pbh_u16_slice(a, b, out);
	}
	let n = a.len();
	for i in 0..n {
		out[i] = f32_to_bf16_scalar(b[i]);
		out[n + i] = f32_to_bf16_scalar(a[i]);
	}
}

/// `out[i] = src[i] + low52(a[i] * b[i])`, wrapping: `Avx512Ifma` (EVEX,
/// 512-bit) -> `AvxIfma` (VEX, 256-bit) -> portable scalar. Unlike
/// [`mullo_i64`]'s single-tier gap, IFMA has a genuine narrower-width HW
/// rung: just gated by a different CPUID bit than the EVEX form, since
/// `AvxIfma` is a separate (Sierra-Forest-class, no-AVX-512) instruction
/// encoding of the identical `vpmadd52luq` math, not a lower SIMD tier of
/// the same encoding family.
///
/// # Panics
/// `a.len() != b.len() || a.len() != src.len() || out.len() != src.len()`.
#[inline]
pub fn madd52lo_u64(src: &[u64], a: &[u64], b: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), src.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Ifma::from_features(features) {
		return t.madd52lo_u64_slice(src, a, b, out);
	}
	if let Some(t) = AvxIfma::from_features(features) {
		return t.madd52lo_u64_slice_wide(src, a, b, out);
	}
	for (((&s, &x), &y), o) in src.iter().zip(a).zip(b).zip(out.iter_mut()) {
		*o = madd52lo_scalar(s, x, y);
	}
}

/// `out[i] = src[i] + high52(a[i] * b[i])`, wrapping: same cascade shape as
/// [`madd52lo_u64`].
///
/// # Panics
/// `a.len() != b.len() || a.len() != src.len() || out.len() != src.len()`.
#[inline]
pub fn madd52hi_u64(src: &[u64], a: &[u64], b: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), b.len());
	assert_eq!(a.len(), src.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Ifma::from_features(features) {
		return t.madd52hi_u64_slice(src, a, b, out);
	}
	if let Some(t) = AvxIfma::from_features(features) {
		return t.madd52hi_u64_slice_wide(src, a, b, out);
	}
	for (((&s, &x), &y), o) in src.iter().zip(a).zip(b).zip(out.iter_mut()) {
		*o = madd52hi_scalar(s, x, y);
	}
}

/// `out[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)`, wrapping
/// (`u8`x`i8`): `Avx512Vnni` (EVEX, 512-bit) -> `AvxVnni` (VEX, 256-bit) ->
/// portable scalar. Same cross-encoding shape as [`madd52lo_u64`] (EVEX/VEX
/// disjoint CPUID bits, same math, not a width hierarchy of one encoding).
///
/// # Panics
/// `a.len() != src.len() * 4 || b.len() != a.len() || out.len() != src.len()`.
#[inline]
pub fn dpbusd_i32(src: &[i32], a: &[u8], b: &[i8], out: &mut [i32]) {
	assert_eq!(a.len(), src.len() * 4);
	assert_eq!(b.len(), a.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vnni::from_features(features) {
		return t.dpbusd_i32_slice(src, a, b, out);
	}
	if let Some(t) = AvxVnni::from_features(features) {
		return t.dpbusd_i32_slice_wide(src, a, b, out);
	}
	dpbusd_scalar(src, a, b, out, vnni_acc_wrapping);
}

/// Saturating [`dpbusd_i32`].
///
/// # Panics
/// Same as [`dpbusd_i32`].
#[inline]
pub fn dpbusds_i32(src: &[i32], a: &[u8], b: &[i8], out: &mut [i32]) {
	assert_eq!(a.len(), src.len() * 4);
	assert_eq!(b.len(), a.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vnni::from_features(features) {
		return t.dpbusds_i32_slice(src, a, b, out);
	}
	if let Some(t) = AvxVnni::from_features(features) {
		return t.dpbusds_i32_slice_wide(src, a, b, out);
	}
	dpbusd_scalar(src, a, b, out, vnni_acc_saturating);
}

fn dpbusd_scalar(src: &[i32], a: &[u8], b: &[i8], out: &mut [i32], acc: fn(i32, i64) -> i32) {
	for (j, o) in out.iter_mut().enumerate() {
		let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
		*o = acc(src[j], sum);
	}
}

/// `out[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)`, wrapping
/// (`i16`x`i16`): same cascade shape as [`dpbusd_i32`].
///
/// # Panics
/// `a.len() != src.len() * 2 || b.len() != a.len() || out.len() != src.len()`.
#[inline]
pub fn dpwssd_i32(src: &[i32], a: &[i16], b: &[i16], out: &mut [i32]) {
	assert_eq!(a.len(), src.len() * 2);
	assert_eq!(b.len(), a.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vnni::from_features(features) {
		return t.dpwssd_i32_slice(src, a, b, out);
	}
	if let Some(t) = AvxVnni::from_features(features) {
		return t.dpwssd_i32_slice_wide(src, a, b, out);
	}
	dpwssd_scalar(src, a, b, out, vnni_acc_wrapping);
}

/// Saturating [`dpwssd_i32`].
///
/// # Panics
/// Same as [`dpwssd_i32`].
#[inline]
pub fn dpwssds_i32(src: &[i32], a: &[i16], b: &[i16], out: &mut [i32]) {
	assert_eq!(a.len(), src.len() * 2);
	assert_eq!(b.len(), a.len());
	assert_eq!(out.len(), src.len());
	let features = super::detect_features();
	if let Some(t) = Avx512Vnni::from_features(features) {
		return t.dpwssds_i32_slice(src, a, b, out);
	}
	if let Some(t) = AvxVnni::from_features(features) {
		return t.dpwssds_i32_slice_wide(src, a, b, out);
	}
	dpwssd_scalar(src, a, b, out, vnni_acc_saturating);
}

fn dpwssd_scalar(src: &[i32], a: &[i16], b: &[i16], out: &mut [i32], acc: fn(i32, i64) -> i32) {
	for (j, o) in out.iter_mut().enumerate() {
		let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
		*o = acc(src[j], sum);
	}
}

/// `out[j] = src[j] + sum_n(a[n][2j]*b[2n] + a[n][2j+1]*b[2n+1])`: Xeon-Phi-only
/// AVX512_4VNNIW (`VP4DPWSSD`), software-composed as 4 folded `Avx512Vnni`
/// `dpwssd_i32` calls, no `avx512_4vnniw` CPUID bit needed. Scalar fallback
/// below `Avx512Vnni` (no lower-tier HW form to cascade through).
///
/// # Panics
/// `out.len() != src.len()`, or any `a[n].len() != src.len() * 2`.
#[inline]
pub fn p4dpwssd_i32(src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
	assert_eq!(out.len(), src.len());
	for an in &a {
		assert_eq!(an.len(), src.len() * 2);
	}
	if let Some(t) = Avx512Vnni::from_features(super::detect_features()) {
		return t.p4dpwssd_i32_slice(src, a, b, out);
	}
	p4dpwssd_scalar(src, a, b, out, vnni_acc_wrapping);
}

/// Saturating [`p4dpwssd_i32`] (`VP4DPWSSDS`).
///
/// # Panics
/// Same as [`p4dpwssd_i32`].
#[inline]
pub fn p4dpwssds_i32(src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
	assert_eq!(out.len(), src.len());
	for an in &a {
		assert_eq!(an.len(), src.len() * 2);
	}
	if let Some(t) = Avx512Vnni::from_features(super::detect_features()) {
		return t.p4dpwssds_i32_slice(src, a, b, out);
	}
	p4dpwssd_scalar(src, a, b, out, vnni_acc_saturating);
}

fn p4dpwssd_scalar(src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32], acc: fn(i32, i64) -> i32) {
	for (j, o) in out.iter_mut().enumerate() {
		let mut v = src[j];
		for n in 0..4 {
			let sum: i64 = a[n][2 * j] as i64 * b[2 * n] as i64 + a[n][2 * j + 1] as i64 * b[2 * n + 1] as i64;
			v = acc(v, sum);
		}
		*o = v;
	}
}

// clamp(x, lo, hi) = min(max(x, lo), hi): pure composition, no tier check of
// its own: just calls `max_*` above, kept here (not `auto_down.rs`, which is
// a private module) since these are public API.

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_f32`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_f32(a: &[f32], lo: &[f32], hi: &[f32], out: &mut [f32]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_f32(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_f64`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_f64(a: &[f64], lo: &[f64], hi: &[f64], out: &mut [f64]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_f64(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_i32`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_i32(a: &[i32], lo: &[i32], hi: &[i32], out: &mut [i32]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_i32(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_u32`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_u32(a: &[u32], lo: &[u32], hi: &[u32], out: &mut [u32]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_u32(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_i64`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_i64(a: &[i64], lo: &[i64], hi: &[i64], out: &mut [i64]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_i64(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

/// `out[i] = min(max(a[i], lo[i]), hi[i])`. Composes [`max_u64`] then in-place min with `hi`.
///
/// # Panics
/// Length mismatch among `a`, `lo`, `hi`, `out`.
#[inline]
pub fn clamp_u64(a: &[u64], lo: &[u64], hi: &[u64], out: &mut [u64]) {
	assert_eq!(a.len(), lo.len());
	assert_eq!(a.len(), hi.len());
	assert_eq!(out.len(), a.len());
	max_u64(a, lo, out);
	for (o, &h) in out.iter_mut().zip(hi) {
		*o = (*o).min(h);
	}
}

#[cfg(test)]
#[path = "test/auto_up.rs"]
mod tests;
