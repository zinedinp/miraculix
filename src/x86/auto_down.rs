//! Lower half of raw-feature slice auto-dispatch: SSE-family/`Avx` bottom tier plus scalar fallback.
//! Private implementation; callers go through `x86::auto_up`.
//! Wider-bus lift can probe raw AVX and dispatch a lifted 128-bit chain when the bottom rung is SSE-family only.

/// Must be used in tail position inside a `pub fn` (expands to `return`).
/// `$lifted` is a closure `|proof: Avx| -> _`
#[cfg(feature = "wider-bus-lift")]
macro_rules! auto_down {
	($lifted:expr, $plain:expr) => {{
		if let Some(proof) = crate::x86::ops::avx::avx::Avx::from_features(super::detect_features()) {
			return ($lifted)(proof);
		}
		return $plain;
	}};
}
#[cfg(not(feature = "wider-bus-lift"))]
macro_rules! auto_down {
	($lifted:expr, $plain:expr) => {{
		return $plain;
	}};
}
use super::ops::{avx::avx::Avx, sse::sse::Sse, sse::sse2::Sse2, sse::sse41::Sse41, sse::sse42::Sse42, sse::ssse3::Ssse3};

/// Regular int binop bottom, x86_64 ABI-baseline (`Sse2::assume_baseline`, zero
/// probe cost) with a real `detect`+scalar fallback on other arches. Shared by
/// both `auto_up.rs`'s `Avx512f`+`Avx2` and `Avx512Bw`+`Avx2` up-halves: the
/// down side doesn't care which one led here. `lifted_bottom = ...,` arm for the
/// 18 i32/u32 ops the wider-bus lift covers.
macro_rules! auto_binop_down_baseline {
	($fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty, lifted_bottom = $lifted_bottom:ident,) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			#[cfg(target_arch = "x86_64")]
			{
				auto_down!(
					|proof| Sse2::assume_baseline().$lifted_bottom(proof, a, b, out),
					Sse2::assume_baseline().$slice_method(a, b, out)
				);
			}
			#[cfg(not(target_arch = "x86_64"))]
			{
				if let Some(t) = Sse2::from_features(super::detect_features()) {
					auto_down!(|proof| t.$lifted_bottom(proof, a, b, out), t.$slice_method(a, b, out));
				}
				$scalar_fn_name(a, b, out);
			}
		}

		#[cfg(not(target_arch = "x86_64"))]
		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
	($fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			#[cfg(target_arch = "x86_64")]
			{
				Sse2::assume_baseline().$slice_method(a, b, out);
			}
			#[cfg(not(target_arch = "x86_64"))]
			{
				if let Some(t) = Sse2::from_features(super::detect_features()) {
					return t.$slice_method(a, b, out);
				}
				$scalar_fn_name(a, b, out);
			}
		}

		#[cfg(not(target_arch = "x86_64"))]
		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
}

/// Regular int binop bottom, always a real probe (`Sse41`/`Sse42`, no ABI
/// baseline exists for SSE4.x): scalar fallback available on every arch, no
/// `#[cfg(target_arch)]` split needed. `$Bottom` (`Sse41`/`Sse42`) is explicit
/// since (unlike the baseline family) it varies per op, not always `Sse2`.
macro_rules! auto_binop_down_probed {
	($Bottom:ty; $fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty, lifted_bottom = $lifted_bottom:ident,) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			if let Some(t) = <$Bottom>::from_features(super::detect_features()) {
				auto_down!(|proof| t.$lifted_bottom(proof, a, b, out), t.$slice_method(a, b, out));
			}
			$scalar_fn_name(a, b, out)
		}

		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
	($Bottom:ty; $fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			if let Some(t) = <$Bottom>::from_features(super::detect_features()) {
				return t.$slice_method(a, b, out);
			}
			$scalar_fn_name(a, b, out)
		}

		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
}

/// f32/f64 bottom: `Avx` (real rung, no lift wired: `auto_up.rs`'s module doc
/// covers this) -> `Sse`/`Sse2` ABI-baseline -> scalar (non-x86_64: real probe).
macro_rules! auto_f32f64_down {
	($Bottom:ty; $fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			if let Some(t) = Avx::from_features(super::detect_features()) {
				return t.$slice_method(a, b, out);
			}
			#[cfg(target_arch = "x86_64")]
			{
				<$Bottom>::assume_baseline().$slice_method(a, b, out);
			}
			#[cfg(not(target_arch = "x86_64"))]
			{
				if let Some(t) = <$Bottom>::from_features(super::detect_features()) {
					return t.$slice_method(a, b, out);
				}
				$scalar_fn_name(a, b, out);
			}
		}

		#[cfg(not(target_arch = "x86_64"))]
		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
}

/// Const-imm shift bottom: `Sse2` ABI-baseline -> scalar (non-x86_64: real
/// probe). Shared by both the `Avx512f`- and `Avx512Bw`-topped up-halves.
macro_rules! auto_shift_imm_down {
	($fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty) => {
		pub(crate) fn $fn_name<const IMM: u32>(a: &[$Elem], out: &mut [$Elem]) {
			#[cfg(target_arch = "x86_64")]
			{
				Sse2::assume_baseline().$slice_method::<IMM>(a, out);
			}
			#[cfg(not(target_arch = "x86_64"))]
			{
				if let Some(t) = Sse2::from_features(super::detect_features()) {
					return t.$slice_method::<IMM>(a, out);
				}
				$scalar_fn_name::<IMM>(a, out);
			}
		}

		#[cfg(not(target_arch = "x86_64"))]
		fn $scalar_fn_name<const IMM: u32>(a: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, u32) -> $Elem = $scalar;
			let imm = IMM as u32;
			for (&x, o) in a.iter().zip(out.iter_mut()) {
				*o = op(x, imm);
			}
		}
	};
}

auto_f32f64_down!(Sse; add_f32, add_f32_slice, add_f32_scalar, |x, y| x + y, f32);
auto_f32f64_down!(Sse; sub_f32, sub_f32_slice, sub_f32_scalar, |x, y| x - y, f32);
auto_f32f64_down!(Sse; mul_f32, mul_f32_slice, mul_f32_scalar, |x, y| x * y, f32);
auto_f32f64_down!(Sse; div_f32, div_f32_slice, div_f32_scalar, |x, y| x / y, f32);
auto_f32f64_down!(Sse; min_f32, min_f32_slice, min_f32_scalar, |x, y| x.min(y), f32);
auto_f32f64_down!(Sse; max_f32, max_f32_slice, max_f32_scalar, |x, y| x.max(y), f32);
auto_f32f64_down!(Sse; and_f32, and_f32_slice, and_f32_scalar, |x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()), f32);
auto_f32f64_down!(Sse; or_f32, or_f32_slice, or_f32_scalar, |x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()), f32);
auto_f32f64_down!(Sse; xor_f32, xor_f32_slice, xor_f32_scalar, |x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()), f32);
auto_f32f64_down!(Sse; andnot_f32, andnot_f32_slice, andnot_f32_scalar,
	|x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()), f32);
auto_f32f64_down!(Sse; cmpeq_f32, cmpeq_f32_slice, cmpeq_f32_scalar,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32);
auto_f32f64_down!(Sse; cmplt_f32, cmplt_f32_slice, cmplt_f32_scalar,
	|x, y| if x < y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32);
auto_f32f64_down!(Sse; cmple_f32, cmple_f32_slice, cmple_f32_scalar,
	|x, y| if x <= y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32);
auto_f32f64_down!(Sse; cmpgt_f32, cmpgt_f32_slice, cmpgt_f32_scalar,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32);
auto_f32f64_down!(Sse; cmpge_f32, cmpge_f32_slice, cmpge_f32_scalar,
	|x, y| if x >= y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32);

auto_f32f64_down!(Sse2; add_f64, add_f64_slice, add_f64_scalar, |x, y| x + y, f64);
auto_f32f64_down!(Sse2; sub_f64, sub_f64_slice, sub_f64_scalar, |x, y| x - y, f64);
auto_f32f64_down!(Sse2; mul_f64, mul_f64_slice, mul_f64_scalar, |x, y| x * y, f64);
auto_f32f64_down!(Sse2; div_f64, div_f64_slice, div_f64_scalar, |x, y| x / y, f64);
auto_f32f64_down!(Sse2; min_f64, min_f64_slice, min_f64_scalar, |x, y| x.min(y), f64);
auto_f32f64_down!(Sse2; max_f64, max_f64_slice, max_f64_scalar, |x, y| x.max(y), f64);
auto_f32f64_down!(Sse2; and_f64, and_f64_slice, and_f64_scalar, |x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()), f64);
auto_f32f64_down!(Sse2; or_f64, or_f64_slice, or_f64_scalar, |x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()), f64);
auto_f32f64_down!(Sse2; xor_f64, xor_f64_slice, xor_f64_scalar, |x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()), f64);
auto_f32f64_down!(Sse2; andnot_f64, andnot_f64_slice, andnot_f64_scalar,
	|x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()), f64);
auto_f32f64_down!(Sse2; cmpeq_f64, cmpeq_f64_slice, cmpeq_f64_scalar,
	|x, y| if x == y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64);
auto_f32f64_down!(Sse2; cmplt_f64, cmplt_f64_slice, cmplt_f64_scalar,
	|x, y| if x < y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64);
auto_f32f64_down!(Sse2; cmple_f64, cmple_f64_slice, cmple_f64_scalar,
	|x, y| if x <= y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64);
auto_f32f64_down!(Sse2; cmpgt_f64, cmpgt_f64_slice, cmpgt_f64_scalar,
	|x, y| if x > y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64);
auto_f32f64_down!(Sse2; cmpge_f64, cmpge_f64_slice, cmpge_f64_scalar,
	|x, y| if x >= y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64);

auto_binop_down_baseline!(add_i32, add_i32_slice, add_i32_scalar, |x: i32, y: i32| x.wrapping_add(y), i32,
	lifted_bottom = add_i32_slice_lifted,);
auto_binop_down_baseline!(sub_i32, sub_i32_slice, sub_i32_scalar, |x: i32, y: i32| x.wrapping_sub(y), i32,
	lifted_bottom = sub_i32_slice_lifted,);
auto_binop_down_baseline!(div_i32, div_i32_slice, div_i32_scalar, |x: i32, y: i32| x / y, i32);
auto_binop_down_probed!(Sse41; mul_i32, mul_i32_slice, mul_i32_scalar, |x: i32, y: i32| x.wrapping_mul(y), i32,
	lifted_bottom = mul_i32_slice_lifted,);
auto_binop_down_baseline!(min_i32, min_i32_slice, min_i32_scalar, |x, y| x.min(y), i32);
auto_binop_down_baseline!(max_i32, max_i32_slice, max_i32_scalar, |x, y| x.max(y), i32);
auto_binop_down_baseline!(and_i32, and_i32_slice, and_i32_scalar, |x, y| x & y, i32, lifted_bottom = and_i32_slice_lifted,);
auto_binop_down_baseline!(or_i32, or_i32_slice, or_i32_scalar, |x, y| x | y, i32, lifted_bottom = or_i32_slice_lifted,);
auto_binop_down_baseline!(xor_i32, xor_i32_slice, xor_i32_scalar, |x, y| x ^ y, i32, lifted_bottom = xor_i32_slice_lifted,);
auto_binop_down_baseline!(andnot_i32, andnot_i32_slice, andnot_i32_scalar, |x, y| !x & y, i32,
	lifted_bottom = andnot_i32_slice_lifted,);
auto_binop_down_baseline!(cmpeq_i32, cmpeq_i32_slice, cmpeq_i32_scalar, |x, y| if x == y { -1 } else { 0 }, i32,
	lifted_bottom = cmpeq_i32_slice_lifted,);
auto_binop_down_baseline!(cmpgt_i32, cmpgt_i32_slice, cmpgt_i32_scalar, |x, y| if x > y { -1 } else { 0 }, i32,
	lifted_bottom = cmpgt_i32_slice_lifted,);
auto_binop_down_baseline!(cmplt_i32, cmplt_i32_slice, cmplt_i32_scalar, |x, y| if x < y { -1 } else { 0 }, i32);
auto_binop_down_baseline!(cmple_i32, cmple_i32_slice, cmple_i32_scalar, |x, y| if x <= y { -1 } else { 0 }, i32);
auto_binop_down_baseline!(cmpge_i32, cmpge_i32_slice, cmpge_i32_scalar, |x, y| if x >= y { -1 } else { 0 }, i32);

auto_binop_down_baseline!(add_u32, add_u32_slice, add_u32_scalar, |x: u32, y: u32| x.wrapping_add(y), u32,
	lifted_bottom = add_u32_slice_lifted,);
auto_binop_down_baseline!(sub_u32, sub_u32_slice, sub_u32_scalar, |x: u32, y: u32| x.wrapping_sub(y), u32,
	lifted_bottom = sub_u32_slice_lifted,);
auto_binop_down_baseline!(div_u32, div_u32_slice, div_u32_scalar, |x: u32, y: u32| x / y, u32);
auto_binop_down_probed!(Sse41; mul_u32, mul_u32_slice, mul_u32_scalar, |x: u32, y: u32| x.wrapping_mul(y), u32,
	lifted_bottom = mul_u32_slice_lifted,);
auto_binop_down_baseline!(min_u32, min_u32_slice, min_u32_scalar, |x, y| x.min(y), u32);
auto_binop_down_baseline!(max_u32, max_u32_slice, max_u32_scalar, |x, y| x.max(y), u32);
auto_binop_down_baseline!(and_u32, and_u32_slice, and_u32_scalar, |x, y| x & y, u32, lifted_bottom = and_u32_slice_lifted,);
auto_binop_down_baseline!(or_u32, or_u32_slice, or_u32_scalar, |x, y| x | y, u32, lifted_bottom = or_u32_slice_lifted,);
auto_binop_down_baseline!(xor_u32, xor_u32_slice, xor_u32_scalar, |x, y| x ^ y, u32, lifted_bottom = xor_u32_slice_lifted,);
auto_binop_down_baseline!(andnot_u32, andnot_u32_slice, andnot_u32_scalar, |x, y| !x & y, u32,
	lifted_bottom = andnot_u32_slice_lifted,);
auto_binop_down_baseline!(cmpeq_u32, cmpeq_u32_slice, cmpeq_u32_scalar, |x, y| if x == y { !0 } else { 0 }, u32,
	lifted_bottom = cmpeq_u32_slice_lifted,);
auto_binop_down_baseline!(cmpgt_u32, cmpgt_u32_slice, cmpgt_u32_scalar, |x, y| if x > y { !0 } else { 0 }, u32);
auto_binop_down_baseline!(cmplt_u32, cmplt_u32_slice, cmplt_u32_scalar, |x, y| if x < y { !0 } else { 0 }, u32);
auto_binop_down_baseline!(cmple_u32, cmple_u32_slice, cmple_u32_scalar, |x, y| if x <= y { !0 } else { 0 }, u32);
auto_binop_down_baseline!(cmpge_u32, cmpge_u32_slice, cmpge_u32_scalar, |x, y| if x >= y { !0 } else { 0 }, u32);

auto_binop_down_baseline!(add_i64, add_i64_slice, add_i64_scalar, |x: i64, y: i64| x.wrapping_add(y), i64);
auto_binop_down_baseline!(sub_i64, sub_i64_slice, sub_i64_scalar, |x: i64, y: i64| x.wrapping_sub(y), i64);
auto_binop_down_baseline!(add_u64, add_u64_slice, add_u64_scalar, |x: u64, y: u64| x.wrapping_add(y), u64);
auto_binop_down_baseline!(sub_u64, sub_u64_slice, sub_u64_scalar, |x: u64, y: u64| x.wrapping_sub(y), u64);
auto_binop_down_baseline!(and_i64, and_i64_slice, and_i64_scalar, |x, y| x & y, i64);
auto_binop_down_baseline!(or_i64, or_i64_slice, or_i64_scalar, |x, y| x | y, i64);
auto_binop_down_baseline!(xor_i64, xor_i64_slice, xor_i64_scalar, |x, y| x ^ y, i64);
auto_binop_down_baseline!(andnot_i64, andnot_i64_slice, andnot_i64_scalar, |x, y| !x & y, i64);
auto_binop_down_baseline!(and_u64, and_u64_slice, and_u64_scalar, |x, y| x & y, u64);
auto_binop_down_baseline!(or_u64, or_u64_slice, or_u64_scalar, |x, y| x | y, u64);
auto_binop_down_baseline!(xor_u64, xor_u64_slice, xor_u64_scalar, |x, y| x ^ y, u64);
auto_binop_down_baseline!(andnot_u64, andnot_u64_slice, andnot_u64_scalar, |x, y| !x & y, u64);

auto_binop_down_probed!(Sse41; cmpeq_i64, cmpeq_i64_slice, cmpeq_i64_scalar, |x, y| if x == y { -1 } else { 0 }, i64);
auto_binop_down_probed!(Sse41; cmpeq_u64, cmpeq_u64_slice, cmpeq_u64_scalar, |x, y| if x == y { !0 } else { 0 }, u64);
auto_binop_down_probed!(Sse42; cmpgt_i64, cmpgt_i64_slice, cmpgt_i64_scalar, |x, y| if x > y { -1 } else { 0 }, i64);
auto_binop_down_probed!(Sse42; cmpgt_u64, cmpgt_u64_slice, cmpgt_u64_scalar, |x, y| if x > y { !0 } else { 0 }, u64);
auto_binop_down_probed!(Sse42; cmplt_i64, cmplt_i64_slice, cmplt_i64_scalar, |x, y| if x < y { -1 } else { 0 }, i64);
auto_binop_down_probed!(Sse42; cmple_i64, cmple_i64_slice, cmple_i64_scalar, |x, y| if x <= y { -1 } else { 0 }, i64);
auto_binop_down_probed!(Sse42; cmpge_i64, cmpge_i64_slice, cmpge_i64_scalar, |x, y| if x >= y { -1 } else { 0 }, i64);
auto_binop_down_probed!(Sse42; cmplt_u64, cmplt_u64_slice, cmplt_u64_scalar, |x, y| if x < y { !0 } else { 0 }, u64);
auto_binop_down_probed!(Sse42; cmple_u64, cmple_u64_slice, cmple_u64_scalar, |x, y| if x <= y { !0 } else { 0 }, u64);
auto_binop_down_probed!(Sse42; cmpge_u64, cmpge_u64_slice, cmpge_u64_scalar, |x, y| if x >= y { !0 } else { 0 }, u64);

auto_shift_imm_down!(shl_i32, shl_i32_slice, shl_i32_scalar, |x: i32, imm| x.wrapping_shl(imm), i32);
auto_shift_imm_down!(shr_i32, shr_i32_slice, shr_i32_scalar, |x: i32, imm| ((x as u32).wrapping_shr(imm)) as i32, i32);
auto_shift_imm_down!(sra_i32, sra_i32_slice, sra_i32_scalar, |x: i32, imm| x.wrapping_shr(imm), i32);
auto_shift_imm_down!(shl_u32, shl_u32_slice, shl_u32_scalar, |x: u32, imm| x.wrapping_shl(imm), u32);
auto_shift_imm_down!(shr_u32, shr_u32_slice, shr_u32_scalar, |x: u32, imm| x.wrapping_shr(imm), u32);

/// Narrow ints (i8/u8/i16/u16) bottom, x86_64 ABI-baseline `Sse2`. Shared by
/// both `auto_up.rs`'s `avx512bw_baseline`- and `avx512bw_probed`-topped ops
/// where the bottom itself is baseline (add/sub/adds/subs/cmp*/mul/min/max on
/// the elem widths where SSE2 natively has the op).
macro_rules! auto_binop_down_bw_baseline {
	($fn_name:ident, $slice_method:ident, $scalar_fn_name:ident, $scalar:expr, $Elem:ty) => {
		pub(crate) fn $fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			#[cfg(target_arch = "x86_64")]
			{
				Sse2::assume_baseline().$slice_method(a, b, out);
			}
			#[cfg(not(target_arch = "x86_64"))]
			{
				if let Some(t) = Sse2::from_features(super::detect_features()) {
					return t.$slice_method(a, b, out);
				}
				$scalar_fn_name(a, b, out);
			}
		}

		#[cfg(not(target_arch = "x86_64"))]
		fn $scalar_fn_name(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
				*o = op(x, y);
			}
		}
	};
}

/// Narrow ints bottom, real `Sse41` probe (`i8`/`u16` min/max: no native SSE2
/// form, unlike `u8`/`i16`).
fn min_i8_bw_scalar(a: &[i8], b: &[i8], out: &mut [i8]) {
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.min(y);
	}
}
fn max_i8_bw_scalar(a: &[i8], b: &[i8], out: &mut [i8]) {
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.max(y);
	}
}
pub(crate) fn min_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.min_i8_slice(a, b, out);
	}
	min_i8_bw_scalar(a, b, out)
}
pub(crate) fn max_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.max_i8_slice(a, b, out);
	}
	max_i8_bw_scalar(a, b, out)
}
pub(crate) fn min_u16(a: &[u16], b: &[u16], out: &mut [u16]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.min_u16_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.min(y);
	}
}
pub(crate) fn max_u16(a: &[u16], b: &[u16], out: &mut [u16]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.max_u16_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.max(y);
	}
}

// i8: add/sub/adds/subs/cmpeq/bitwise bottom SSE2 (ABI-baseline); min/max above.
auto_binop_down_bw_baseline!(add_i8, add_i8_slice, add_i8_scalar, |x: i8, y: i8| x.wrapping_add(y), i8);
auto_binop_down_bw_baseline!(sub_i8, sub_i8_slice, sub_i8_scalar, |x: i8, y: i8| x.wrapping_sub(y), i8);
auto_binop_down_bw_baseline!(adds_i8, adds_i8_slice, adds_i8_scalar, |x: i8, y: i8| x.saturating_add(y), i8);
auto_binop_down_bw_baseline!(subs_i8, subs_i8_slice, subs_i8_scalar, |x: i8, y: i8| x.saturating_sub(y), i8);
auto_binop_down_bw_baseline!(cmpeq_i8, cmpeq_i8_slice, cmpeq_i8_scalar, |x, y| if x == y { -1 } else { 0 }, i8);
auto_binop_down_bw_baseline!(cmpgt_i8, cmpgt_i8_slice, cmpgt_i8_scalar, |x, y| if x > y { -1 } else { 0 }, i8);
auto_binop_down_bw_baseline!(cmplt_i8, cmplt_i8_slice, cmplt_i8_scalar, |x, y| if x < y { -1 } else { 0 }, i8);
auto_binop_down_bw_baseline!(cmple_i8, cmple_i8_slice, cmple_i8_scalar, |x, y| if x <= y { -1 } else { 0 }, i8);
auto_binop_down_bw_baseline!(cmpge_i8, cmpge_i8_slice, cmpge_i8_scalar, |x, y| if x >= y { -1 } else { 0 }, i8);
auto_binop_down_baseline!(and_i8, and_i8_slice, and_i8_scalar, |x, y| x & y, i8);
auto_binop_down_baseline!(or_i8, or_i8_slice, or_i8_scalar, |x, y| x | y, i8);
auto_binop_down_baseline!(xor_i8, xor_i8_slice, xor_i8_scalar, |x, y| x ^ y, i8);
auto_binop_down_baseline!(andnot_i8, andnot_i8_slice, andnot_i8_scalar, |x, y| !x & y, i8);

// u8: same shape, but min/max also bottom SSE2 (native there, unlike i8's pminsb).
auto_binop_down_bw_baseline!(add_u8, add_u8_slice, add_u8_scalar, |x: u8, y: u8| x.wrapping_add(y), u8);
auto_binop_down_bw_baseline!(sub_u8, sub_u8_slice, sub_u8_scalar, |x: u8, y: u8| x.wrapping_sub(y), u8);
auto_binop_down_bw_baseline!(adds_u8, adds_u8_slice, adds_u8_scalar, |x: u8, y: u8| x.saturating_add(y), u8);
auto_binop_down_bw_baseline!(subs_u8, subs_u8_slice, subs_u8_scalar, |x: u8, y: u8| x.saturating_sub(y), u8);
auto_binop_down_bw_baseline!(cmpeq_u8, cmpeq_u8_slice, cmpeq_u8_scalar, |x, y| if x == y { !0 } else { 0 }, u8);
auto_binop_down_bw_baseline!(cmpgt_u8, cmpgt_u8_slice, cmpgt_u8_scalar, |x, y| if x > y { !0 } else { 0 }, u8);
auto_binop_down_bw_baseline!(cmplt_u8, cmplt_u8_slice, cmplt_u8_scalar, |x, y| if x < y { !0 } else { 0 }, u8);
auto_binop_down_bw_baseline!(cmple_u8, cmple_u8_slice, cmple_u8_scalar, |x, y| if x <= y { !0 } else { 0 }, u8);
auto_binop_down_bw_baseline!(cmpge_u8, cmpge_u8_slice, cmpge_u8_scalar, |x, y| if x >= y { !0 } else { 0 }, u8);
auto_binop_down_baseline!(and_u8, and_u8_slice, and_u8_scalar, |x, y| x & y, u8);
auto_binop_down_baseline!(or_u8, or_u8_slice, or_u8_scalar, |x, y| x | y, u8);
auto_binop_down_baseline!(xor_u8, xor_u8_slice, xor_u8_scalar, |x, y| x ^ y, u8);
auto_binop_down_baseline!(andnot_u8, andnot_u8_slice, andnot_u8_scalar, |x, y| !x & y, u8);
auto_binop_down_bw_baseline!(min_u8, min_u8_slice, min_u8_scalar, |x, y| x.min(y), u8);
auto_binop_down_bw_baseline!(max_u8, max_u8_slice, max_u8_scalar, |x, y| x.max(y), u8);
auto_binop_down_bw_baseline!(avg_u8, avg_u8_slice, avg_u8_scalar, |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8, u8);

// i16: add/sub/adds/subs/cmpeq/bitwise/mul/min/max all bottom SSE2 (all native there).
auto_binop_down_bw_baseline!(add_i16, add_i16_slice, add_i16_scalar, |x: i16, y: i16| x.wrapping_add(y), i16);
auto_binop_down_bw_baseline!(sub_i16, sub_i16_slice, sub_i16_scalar, |x: i16, y: i16| x.wrapping_sub(y), i16);
auto_binop_down_bw_baseline!(adds_i16, adds_i16_slice, adds_i16_scalar, |x: i16, y: i16| x.saturating_add(y), i16);
auto_binop_down_bw_baseline!(subs_i16, subs_i16_slice, subs_i16_scalar, |x: i16, y: i16| x.saturating_sub(y), i16);
auto_binop_down_bw_baseline!(cmpeq_i16, cmpeq_i16_slice, cmpeq_i16_scalar, |x, y| if x == y { -1 } else { 0 }, i16);
auto_binop_down_bw_baseline!(cmpgt_i16, cmpgt_i16_slice, cmpgt_i16_scalar, |x, y| if x > y { -1 } else { 0 }, i16);
auto_binop_down_bw_baseline!(cmplt_i16, cmplt_i16_slice, cmplt_i16_scalar, |x, y| if x < y { -1 } else { 0 }, i16);
auto_binop_down_bw_baseline!(cmple_i16, cmple_i16_slice, cmple_i16_scalar, |x, y| if x <= y { -1 } else { 0 }, i16);
auto_binop_down_bw_baseline!(cmpge_i16, cmpge_i16_slice, cmpge_i16_scalar, |x, y| if x >= y { -1 } else { 0 }, i16);
auto_binop_down_baseline!(and_i16, and_i16_slice, and_i16_scalar, |x, y| x & y, i16);
auto_binop_down_baseline!(or_i16, or_i16_slice, or_i16_scalar, |x, y| x | y, i16);
auto_binop_down_baseline!(xor_i16, xor_i16_slice, xor_i16_scalar, |x, y| x ^ y, i16);
auto_binop_down_baseline!(andnot_i16, andnot_i16_slice, andnot_i16_scalar, |x, y| !x & y, i16);
auto_binop_down_bw_baseline!(mul_i16, mul_i16_slice, mul_i16_scalar, |x: i16, y: i16| x.wrapping_mul(y), i16);
auto_binop_down_bw_baseline!(min_i16, min_i16_slice, min_i16_scalar, |x, y| x.min(y), i16);
auto_binop_down_bw_baseline!(max_i16, max_i16_slice, max_i16_scalar, |x, y| x.max(y), i16);

// u16: same shape, but min/max above (Sse41 probe: pminuw not native SSE2, unlike i16's pminsw).
auto_binop_down_bw_baseline!(add_u16, add_u16_slice, add_u16_scalar, |x: u16, y: u16| x.wrapping_add(y), u16);
auto_binop_down_bw_baseline!(sub_u16, sub_u16_slice, sub_u16_scalar, |x: u16, y: u16| x.wrapping_sub(y), u16);
auto_binop_down_bw_baseline!(adds_u16, adds_u16_slice, adds_u16_scalar, |x: u16, y: u16| x.saturating_add(y), u16);
auto_binop_down_bw_baseline!(subs_u16, subs_u16_slice, subs_u16_scalar, |x: u16, y: u16| x.saturating_sub(y), u16);
auto_binop_down_bw_baseline!(cmpeq_u16, cmpeq_u16_slice, cmpeq_u16_scalar, |x, y| if x == y { !0 } else { 0 }, u16);
auto_binop_down_bw_baseline!(cmpgt_u16, cmpgt_u16_slice, cmpgt_u16_scalar, |x, y| if x > y { !0 } else { 0 }, u16);
auto_binop_down_bw_baseline!(cmplt_u16, cmplt_u16_slice, cmplt_u16_scalar, |x, y| if x < y { !0 } else { 0 }, u16);
auto_binop_down_bw_baseline!(cmple_u16, cmple_u16_slice, cmple_u16_scalar, |x, y| if x <= y { !0 } else { 0 }, u16);
auto_binop_down_bw_baseline!(cmpge_u16, cmpge_u16_slice, cmpge_u16_scalar, |x, y| if x >= y { !0 } else { 0 }, u16);
auto_binop_down_baseline!(and_u16, and_u16_slice, and_u16_scalar, |x, y| x & y, u16);
auto_binop_down_baseline!(or_u16, or_u16_slice, or_u16_scalar, |x, y| x | y, u16);
auto_binop_down_baseline!(xor_u16, xor_u16_slice, xor_u16_scalar, |x, y| x ^ y, u16);
auto_binop_down_baseline!(andnot_u16, andnot_u16_slice, andnot_u16_scalar, |x, y| !x & y, u16);
auto_binop_down_bw_baseline!(mul_u16, mul_u16_slice, mul_u16_scalar, |x: u16, y: u16| x.wrapping_mul(y), u16);
auto_binop_down_bw_baseline!(avg_u16, avg_u16_slice, avg_u16_scalar,
	|x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16, u16);

// Const-imm shift bottom, BW family (i8/u8 have no shift cascade at all: this
// is only reached for i16/u16, whose `Sse2` bottom is identical to the i32/u32
// family's, hence sharing `auto_shift_imm_down!` rather than a separate macro.
auto_shift_imm_down!(shl_i16, shl_i16_slice, shl_i16_scalar, |x: i16, imm| x.wrapping_shl(imm), i16);
auto_shift_imm_down!(shr_i16, shr_i16_slice, shr_i16_scalar, |x: i16, imm| ((x as u16).wrapping_shr(imm)) as i16, i16);
auto_shift_imm_down!(sra_i16, sra_i16_slice, sra_i16_scalar, |x: i16, imm| x.wrapping_shr(imm), i16);
auto_shift_imm_down!(shl_u16, shl_u16_slice, shl_u16_scalar, |x: u16, imm| x.wrapping_shl(imm), u16);
auto_shift_imm_down!(shr_u16, shr_u16_slice, shr_u16_scalar, |x: u16, imm| x.wrapping_shr(imm), u16);

pub(crate) fn select_i32(a: &[i32], b: &[i32], mask: &[i32], out: &mut [i32]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_i32_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_u32(a: &[u32], b: &[u32], mask: &[u32], out: &mut [u32]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_u32_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_f32(a: &[f32], b: &[f32], mask: &[f32], out: &mut [f32]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_f32_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m.is_sign_negative() { y } else { x };
	}
}
pub(crate) fn select_i64(a: &[i64], b: &[i64], mask: &[i64], out: &mut [i64]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_i64_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_u64(a: &[u64], b: &[u64], mask: &[u64], out: &mut [u64]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_u64_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_f64(a: &[f64], b: &[f64], mask: &[f64], out: &mut [f64]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_f64_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m.is_sign_negative() { y } else { x };
	}
}
pub(crate) fn select_i8(a: &[i8], b: &[i8], mask: &[i8], out: &mut [i8]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_i8_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_u8(a: &[u8], b: &[u8], mask: &[u8], out: &mut [u8]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_u8_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_i16(a: &[i16], b: &[i16], mask: &[i16], out: &mut [i16]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_i16_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}
pub(crate) fn select_u16(a: &[u16], b: &[u16], mask: &[u16], out: &mut [u16]) {
	if let Some(t) = Sse41::from_features(super::detect_features()) {
		return t.select_u16_slice(a, b, mask, out);
	}
	for (((&x, &y), &m), o) in a.iter().zip(b).zip(mask).zip(out.iter_mut()) {
		*o = if m != 0 { y } else { x };
	}
}

pub(crate) fn min_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	if let Some(t) = Sse42::from_features(super::detect_features()) {
		return t.min_i64_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.min(y);
	}
}
pub(crate) fn max_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	if let Some(t) = Sse42::from_features(super::detect_features()) {
		return t.max_i64_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.max(y);
	}
}
pub(crate) fn min_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	if let Some(t) = Sse42::from_features(super::detect_features()) {
		return t.min_u64_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.min(y);
	}
}
pub(crate) fn max_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	if let Some(t) = Sse42::from_features(super::detect_features()) {
		return t.max_u64_slice(a, b, out);
	}
	for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
		*o = x.max(y);
	}
}

pub(crate) fn mullo_i64(a: &[i64], b: &[i64], out: &mut [i64]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().mullo_i64_slice(a, b, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.mullo_i64_slice(a, b, out);
		}
		for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
			*o = x.wrapping_mul(y);
		}
	}
}
pub(crate) fn mullo_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().mullo_u64_slice(a, b, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.mullo_u64_slice(a, b, out);
		}
		for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
			*o = x.wrapping_mul(y);
		}
	}
}

pub(crate) fn abs_i32(a: &[i32], out: &mut [i32]) {
	if let Some(t) = Ssse3::from_features(super::detect_features()) {
		return t.abs_i32_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.wrapping_abs();
	}
}
pub(crate) fn abs_i64(a: &[i64], out: &mut [i64]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().abs_i64_slice(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.abs_i64_slice(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = x.wrapping_abs();
		}
	}
}
pub(crate) fn abs_i8(a: &[i8], out: &mut [i8]) {
	if let Some(t) = Ssse3::from_features(super::detect_features()) {
		return t.abs_i8_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.wrapping_abs();
	}
}
pub(crate) fn abs_i16(a: &[i16], out: &mut [i16]) {
	if let Some(t) = Ssse3::from_features(super::detect_features()) {
		return t.abs_i16_slice(a, out);
	}
	for (&x, o) in a.iter().zip(out.iter_mut()) {
		*o = x.wrapping_abs();
	}
}

pub(crate) fn mul_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().mul_i8_slice(a, b, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.mul_i8_slice(a, b, out);
		}
		for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
			*o = x.wrapping_mul(y);
		}
	}
}
pub(crate) fn mul_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().mul_u8_slice(a, b, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.mul_u8_slice(a, b, out);
		}
		for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
			*o = x.wrapping_mul(y);
		}
	}
}
pub(crate) fn shl_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().shl_i8_slice::<IMM>(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.shl_i8_slice::<IMM>(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = x.wrapping_shl(IMM);
		}
	}
}
pub(crate) fn shl_u8<const IMM: u32>(a: &[u8], out: &mut [u8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().shl_u8_slice::<IMM>(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.shl_u8_slice::<IMM>(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = x.wrapping_shl(IMM);
		}
	}
}
pub(crate) fn shr_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().shr_i8_slice::<IMM>(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.shr_i8_slice::<IMM>(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = ((x as u8).wrapping_shr(IMM)) as i8;
		}
	}
}
pub(crate) fn shr_u8<const IMM: u32>(a: &[u8], out: &mut [u8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().shr_u8_slice::<IMM>(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.shr_u8_slice::<IMM>(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = x.wrapping_shr(IMM);
		}
	}
}
pub(crate) fn sra_i8<const IMM: u32>(a: &[i8], out: &mut [i8]) {
	#[cfg(target_arch = "x86_64")]
	{
		Sse2::assume_baseline().sra_i8_slice::<IMM>(a, out);
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		if let Some(t) = Sse2::from_features(super::detect_features()) {
			return t.sra_i8_slice::<IMM>(a, out);
		}
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = x.wrapping_shr(IMM);
		}
	}
}

