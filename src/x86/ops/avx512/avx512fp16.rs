//! AVX-512FP16: native binary16 arithmetic (`"avx512fp16"` / `"avx512fp16,avx512vl"`).
//! Lanes are raw `u16` bit patterns via stable integer load/store + bitcast (no unstable `_ph` scalar I/O).
//! `sqrt`/`rsqrt`/`rcp_ph` fixed-width only; merge/zero-masked arith where present (`abs_ph` has no masked EVEX).

use core::arch::x86_64::{
	__m128h, __m256h, __m512h, _mm_abs_ph, _mm_add_ph, _mm_castph_si128, _mm_castsi128_ph, _mm_div_ph, _mm_fmadd_ph,
	_mm_fmsub_ph, _mm_fnmadd_ph, _mm_fnmsub_ph, _mm_loadu_si128, _mm_max_ph, _mm_min_ph, _mm_mul_ph, _mm_rcp_ph,
	_mm_rsqrt_ph, _mm_sqrt_ph, _mm_storeu_si128, _mm_sub_ph, _mm256_abs_ph, _mm256_add_ph, _mm256_castph_si256,
	_mm256_castsi256_ph, _mm256_div_ph, _mm256_fmadd_ph, _mm256_fmsub_ph, _mm256_fnmadd_ph, _mm256_fnmsub_ph,
	_mm256_loadu_si256, _mm256_max_ph, _mm256_min_ph, _mm256_mul_ph, _mm256_rcp_ph, _mm256_rsqrt_ph, _mm256_sqrt_ph,
	_mm256_storeu_si256, _mm256_sub_ph, _mm512_abs_ph, _mm512_add_ph, _mm512_castph_si512, _mm512_castsi512_ph,
	_mm512_div_ph, _mm512_fmadd_ph, _mm512_fmaddsub_ph, _mm512_fmsub_ph, _mm512_fmsubadd_ph, _mm512_fnmadd_ph,
	_mm512_fnmsub_ph, _mm512_loadu_si512, _mm512_max_ph, _mm512_min_ph, _mm512_mul_ph, _mm512_rcp_ph,
	_mm512_rsqrt_ph, _mm512_sqrt_ph, _mm512_storeu_si512, _mm512_sub_ph,
};

use super::super::super::{Feature, FeatureSet};
use super::super::avx::f16c::{f16_to_f32_scalar, f32_to_f16_scalar};
use super::super::macros::{
	simd_binop, simd_binop_fixed, simd_binop_imm_fixed, simd_binop_imm_masked, simd_binop_masked, simd_cvt,
	simd_cvt_imm, simd_cvt_narrow, simd_cvt_narrow_masked, simd_cvt_widen, simd_cvt_widen_masked, simd_ternop,
	simd_ternop_fixed, simd_ternop_imm, simd_ternop_imm_masked, simd_ternop_masked, simd_unop, simd_unop_imm,
	simd_unop_imm_masked, simd_unop_masked,
};

/// Proof token: AVX-512FP16 available (512-bit forms; LLVM's `"avx512fp16"`
/// feature string implies AVX-512F, no separate check needed). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Fp16(());

impl Avx512Fp16 {
	/// `None` if the CPU (or the compile-time target) lacks AVX-512FP16.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512fp16).then_some(Avx512Fp16(()))
	}
}

/// Proof token: AVX-512FP16 *and* AVX-512VL, both required for the 128/256-bit
/// forms (`"avx512fp16,avx512vl"`: VL isn't implied at this width). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Fp16Vl(());

impl Avx512Fp16Vl {
	/// `None` unless the CPU has both AVX-512FP16 and AVX-512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512fp16) && set.contains(Feature::Avx512vl)).then_some(Avx512Fp16Vl(()))
	}
}

/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn loadu_ph512(p: *const u16) -> __m512h {
	unsafe { _mm512_castsi512_ph(_mm512_loadu_si512(p.cast())) }
}
/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn storeu_ph512(p: *mut u16, v: __m512h) {
	unsafe { _mm512_storeu_si512(p.cast(), _mm512_castph_si512(v)) }
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn loadu_ph256(p: *const u16) -> __m256h {
	unsafe { _mm256_castsi256_ph(_mm256_loadu_si256(p.cast())) }
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn storeu_ph256(p: *mut u16, v: __m256h) {
	unsafe { _mm256_storeu_si256(p.cast(), _mm256_castph_si256(v)) }
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn loadu_ph128(p: *const u16) -> __m128h {
	unsafe { _mm_castsi128_ph(_mm_loadu_si128(p.cast())) }
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn storeu_ph128(p: *mut u16, v: __m128h) {
	unsafe { _mm_storeu_si128(p.cast(), _mm_castph_si128(v)) }
}

macro_rules! fp16_binop_512 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_binop_256 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_binop_128 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Fp16Vl, vis = pub, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! fp16_unop_512 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_unop! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_unop_256 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_unop! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_unop_128 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_unop! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! fp16_ternop_512 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Avx512Fp16, vis = pub(crate), target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_ternop_256 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Avx512Fp16Vl, vis = pub(crate), target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}
macro_rules! fp16_ternop_128 {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Avx512Fp16Vl, vis = pub, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

fn add_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) + f16_to_f32_scalar(y))
}
fn sub_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) - f16_to_f32_scalar(y))
}
fn mul_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y))
}
fn div_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) / f16_to_f32_scalar(y))
}
fn min_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x).min(f16_to_f32_scalar(y)))
}
fn max_scalar(x: u16, y: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x).max(f16_to_f32_scalar(y)))
}
fn abs_scalar(x: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x).abs())
}
fn fmadd_scalar(x: u16, y: u16, z: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y) + f16_to_f32_scalar(z))
}
fn fmsub_scalar(x: u16, y: u16, z: u16) -> u16 {
	f32_to_f16_scalar(f16_to_f32_scalar(x) * f16_to_f32_scalar(y) - f16_to_f32_scalar(z))
}
fn fnmadd_scalar(x: u16, y: u16, z: u16) -> u16 {
	f32_to_f16_scalar(-(f16_to_f32_scalar(x) * f16_to_f32_scalar(y)) + f16_to_f32_scalar(z))
}
fn fnmsub_scalar(x: u16, y: u16, z: u16) -> u16 {
	f32_to_f16_scalar(-(f16_to_f32_scalar(x) * f16_to_f32_scalar(y)) - f16_to_f32_scalar(z))
}
/// Even lane: subtract; odd lane: add (`data.js` `_mm512_fmaddsub_ph` pseudocode).
fn fmaddsub_scalar_at(j: usize, x: u16, y: u16, z: u16) -> u16 {
	let prod = f16_to_f32_scalar(x) * f16_to_f32_scalar(y);
	let c = f16_to_f32_scalar(z);
	f32_to_f16_scalar(if j & 1 == 0 { prod - c } else { prod + c })
}
/// Even lane: add; odd lane: subtract (`data.js` `_mm512_fmsubadd_ph` pseudocode).
fn fmsubadd_scalar_at(j: usize, x: u16, y: u16, z: u16) -> u16 {
	let prod = f16_to_f32_scalar(x) * f16_to_f32_scalar(y);
	let c = f16_to_f32_scalar(z);
	f32_to_f16_scalar(if j & 1 == 0 { prod + c } else { prod - c })
}

fp16_binop_512!(add_ph_u16x32, add_ph_u16x32_slice, add_ph512_intrinsic, _mm512_add_ph, add_scalar,
	"`a + b` per lane (`vaddph`, 512-bit).",
	"`out[i] = a[i] + b[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(add_ph_u16x16, add_ph_u16x16_slice, add_ph256_intrinsic, _mm256_add_ph, add_scalar,
	"`a + b` per lane (`vaddph`, 256-bit).",
	"`out[i] = a[i] + b[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(add_ph_u16x8, add_ph_u16x8_slice, add_ph128_intrinsic, _mm_add_ph, add_scalar,
	"`a + b` per lane (`vaddph`, 128-bit).",
	"`out[i] = a[i] + b[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_binop_512!(sub_ph_u16x32, sub_ph_u16x32_slice, sub_ph512_intrinsic, _mm512_sub_ph, sub_scalar,
	"`a - b` per lane (`vsubph`, 512-bit).",
	"`out[i] = a[i] - b[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(sub_ph_u16x16, sub_ph_u16x16_slice, sub_ph256_intrinsic, _mm256_sub_ph, sub_scalar,
	"`a - b` per lane (`vsubph`, 256-bit).",
	"`out[i] = a[i] - b[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(sub_ph_u16x8, sub_ph_u16x8_slice, sub_ph128_intrinsic, _mm_sub_ph, sub_scalar,
	"`a - b` per lane (`vsubph`, 128-bit).",
	"`out[i] = a[i] - b[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_binop_512!(mul_ph_u16x32, mul_ph_u16x32_slice, mul_ph512_intrinsic, _mm512_mul_ph, mul_scalar,
	"`a * b` per lane (`vmulph`, 512-bit).",
	"`out[i] = a[i] * b[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(mul_ph_u16x16, mul_ph_u16x16_slice, mul_ph256_intrinsic, _mm256_mul_ph, mul_scalar,
	"`a * b` per lane (`vmulph`, 256-bit).",
	"`out[i] = a[i] * b[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(mul_ph_u16x8, mul_ph_u16x8_slice, mul_ph128_intrinsic, _mm_mul_ph, mul_scalar,
	"`a * b` per lane (`vmulph`, 128-bit).",
	"`out[i] = a[i] * b[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_binop_512!(div_ph_u16x32, div_ph_u16x32_slice, div_ph512_intrinsic, _mm512_div_ph, div_scalar,
	"`a / b` per lane (`vdivph`, 512-bit).",
	"`out[i] = a[i] / b[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(div_ph_u16x16, div_ph_u16x16_slice, div_ph256_intrinsic, _mm256_div_ph, div_scalar,
	"`a / b` per lane (`vdivph`, 256-bit).",
	"`out[i] = a[i] / b[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(div_ph_u16x8, div_ph_u16x8_slice, div_ph128_intrinsic, _mm_div_ph, div_scalar,
	"`a / b` per lane (`vdivph`, 128-bit).",
	"`out[i] = a[i] / b[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_binop_512!(min_ph_u16x32, min_ph_u16x32_slice, min_ph512_intrinsic, _mm512_min_ph, min_scalar,
	"Per-lane min (`vminph`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(min_ph_u16x16, min_ph_u16x16_slice, min_ph256_intrinsic, _mm256_min_ph, min_scalar,
	"Per-lane min (`vminph`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(min_ph_u16x8, min_ph_u16x8_slice, min_ph128_intrinsic, _mm_min_ph, min_scalar,
	"Per-lane min (`vminph`, 128-bit). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_binop_512!(max_ph_u16x32, max_ph_u16x32_slice, max_ph512_intrinsic, _mm512_max_ph, max_scalar,
	"Per-lane max (`vmaxph`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_binop_256!(max_ph_u16x16, max_ph_u16x16_slice, max_ph256_intrinsic, _mm256_max_ph, max_scalar,
	"Per-lane max (`vmaxph`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_binop_128!(max_ph_u16x8, max_ph_u16x8_slice, max_ph128_intrinsic, _mm_max_ph, max_scalar,
	"Per-lane max (`vmaxph`, 128-bit). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_unop_512!(abs_ph_u16x32, abs_ph_u16x32_slice, abs_ph512_intrinsic, _mm512_abs_ph, abs_scalar,
	"Per-lane absolute value, sign-bit clear (`vandps`-encoded, 512-bit).",
	"`out[i] = |a[i]|` (FP16 bit patterns). 32-wide chunks, scalar remainder.");
fp16_unop_256!(abs_ph_u16x16, abs_ph_u16x16_slice, abs_ph256_intrinsic, _mm256_abs_ph, abs_scalar,
	"Per-lane absolute value, sign-bit clear (`vandps`-encoded, 256-bit).",
	"`out[i] = |a[i]|` (FP16 bit patterns). 16-wide chunks, scalar remainder.");
fp16_unop_128!(abs_ph_u16x8, abs_ph_u16x8_slice, abs_ph128_intrinsic, _mm_abs_ph, abs_scalar,
	"Per-lane absolute value, sign-bit clear (`vandps`-encoded, 128-bit).",
	"`out[i] = |a[i]|` (FP16 bit patterns). 8-wide chunks, scalar remainder.");

fp16_ternop_512!(fmadd_ph_u16x32, fmadd_ph_u16x32_slice, fmadd_ph512_intrinsic, _mm512_fmadd_ph, fmadd_scalar,
	"`a*b + c` per lane, HW fused (`vfmaddph`, 512-bit).",
	"`out[i] = a[i]*b[i] + c[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder not fused.");
fp16_ternop_256!(fmadd_ph_u16x16, fmadd_ph_u16x16_slice, fmadd_ph256_intrinsic, _mm256_fmadd_ph, fmadd_scalar,
	"`a*b + c` per lane, HW fused (`vfmaddph`, 256-bit).",
	"`out[i] = a[i]*b[i] + c[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder not fused.");
fp16_ternop_128!(fmadd_ph_u16x8, fmadd_ph_u16x8_slice, fmadd_ph128_intrinsic, _mm_fmadd_ph, fmadd_scalar,
	"`a*b + c` per lane, HW fused (`vfmaddph`, 128-bit).",
	"`out[i] = a[i]*b[i] + c[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder not fused.");

fp16_ternop_512!(fmsub_ph_u16x32, fmsub_ph_u16x32_slice, fmsub_ph512_intrinsic, _mm512_fmsub_ph, fmsub_scalar,
	"`a*b - c` per lane, HW fused (`vfmsubph`, 512-bit).",
	"`out[i] = a[i]*b[i] - c[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder not fused.");
fp16_ternop_256!(fmsub_ph_u16x16, fmsub_ph_u16x16_slice, fmsub_ph256_intrinsic, _mm256_fmsub_ph, fmsub_scalar,
	"`a*b - c` per lane, HW fused (`vfmsubph`, 256-bit).",
	"`out[i] = a[i]*b[i] - c[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder not fused.");
fp16_ternop_128!(fmsub_ph_u16x8, fmsub_ph_u16x8_slice, fmsub_ph128_intrinsic, _mm_fmsub_ph, fmsub_scalar,
	"`a*b - c` per lane, HW fused (`vfmsubph`, 128-bit).",
	"`out[i] = a[i]*b[i] - c[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder not fused.");

fp16_ternop_512!(fnmadd_ph_u16x32, fnmadd_ph_u16x32_slice, fnmadd_ph512_intrinsic, _mm512_fnmadd_ph, fnmadd_scalar,
	"`-(a*b) + c` per lane, HW fused (`vfnmaddph`, 512-bit).",
	"`out[i] = -(a[i]*b[i]) + c[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder not fused.");
fp16_ternop_256!(fnmadd_ph_u16x16, fnmadd_ph_u16x16_slice, fnmadd_ph256_intrinsic, _mm256_fnmadd_ph, fnmadd_scalar,
	"`-(a*b) + c` per lane, HW fused (`vfnmaddph`, 256-bit).",
	"`out[i] = -(a[i]*b[i]) + c[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder not fused.");
fp16_ternop_128!(fnmadd_ph_u16x8, fnmadd_ph_u16x8_slice, fnmadd_ph128_intrinsic, _mm_fnmadd_ph, fnmadd_scalar,
	"`-(a*b) + c` per lane, HW fused (`vfnmaddph`, 128-bit).",
	"`out[i] = -(a[i]*b[i]) + c[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder not fused.");

fp16_ternop_512!(fnmsub_ph_u16x32, fnmsub_ph_u16x32_slice, fnmsub_ph512_intrinsic, _mm512_fnmsub_ph, fnmsub_scalar,
	"`-(a*b) - c` per lane, HW fused (`vfnmsubph`, 512-bit).",
	"`out[i] = -(a[i]*b[i]) - c[i]` (FP16 bit patterns). 32-wide chunks, scalar remainder not fused.");
fp16_ternop_256!(fnmsub_ph_u16x16, fnmsub_ph_u16x16_slice, fnmsub_ph256_intrinsic, _mm256_fnmsub_ph, fnmsub_scalar,
	"`-(a*b) - c` per lane, HW fused (`vfnmsubph`, 256-bit).",
	"`out[i] = -(a[i]*b[i]) - c[i]` (FP16 bit patterns). 16-wide chunks, scalar remainder not fused.");
fp16_ternop_128!(fnmsub_ph_u16x8, fnmsub_ph_u16x8_slice, fnmsub_ph128_intrinsic, _mm_fnmsub_ph, fnmsub_scalar,
	"`-(a*b) - c` per lane, HW fused (`vfnmsubph`, 128-bit).",
	"`out[i] = -(a[i]*b[i]) - c[i]` (FP16 bit patterns). 8-wide chunks, scalar remainder not fused.");

/// `fmaddsub_ph`/`fmsubadd_ph` alternate add/sub per lane index, so they don't
/// fit [`simd_ternop`]'s uniform-per-lane `scalar` closure: hand-written here.
impl Avx512Fp16 {
	/// Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused (`vfmaddsubph`, 512-bit).
	#[inline]
	pub fn fmaddsub_ph_u16x32(self, a: [u16; 32], b: [u16; 32], c: [u16; 32]) -> [u16; 32] {
		unsafe { fmaddsub_ph512_intrinsic(&a, &b, &c) }
	}

	/// `out[j] = a[j]*b[j] -/+ c[j]` alternating by lane parity (see [`Avx512Fp16::fmaddsub_ph_u16x32`]).
	/// 32-wide chunks, scalar remainder not fused.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `c`, `out`.
	pub fn fmaddsub_ph_u16_slice(self, a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), c.len());
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let c_chunks = c.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let c_rem = c_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);

		for (((ac, bc), cc), oc) in a_chunks.zip(b_chunks).zip(c_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			let cv: [u16; 32] = cc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.fmaddsub_ph_u16x32(av, bv, cv));
		}
		let base = a.len() - a_rem.len();
		for (j, (((&x, &y), &z), o)) in a_rem.iter().zip(b_rem).zip(c_rem).zip(out_chunks.into_remainder()).enumerate() {
			*o = fmaddsub_scalar_at(base + j, x, y, z);
		}
	}

	/// Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused (`vfmsubaddph`, 512-bit).
	#[inline]
	pub fn fmsubadd_ph_u16x32(self, a: [u16; 32], b: [u16; 32], c: [u16; 32]) -> [u16; 32] {
		unsafe { fmsubadd_ph512_intrinsic(&a, &b, &c) }
	}

	/// `out[j] = a[j]*b[j] +/- c[j]` alternating by lane parity (see [`Avx512Fp16::fmsubadd_ph_u16x32`]).
	/// 32-wide chunks, scalar remainder not fused.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `c`, `out`.
	pub fn fmsubadd_ph_u16_slice(self, a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), c.len());
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let c_chunks = c.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let c_rem = c_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);

		for (((ac, bc), cc), oc) in a_chunks.zip(b_chunks).zip(c_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			let cv: [u16; 32] = cc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.fmsubadd_ph_u16x32(av, bv, cv));
		}
		let base = a.len() - a_rem.len();
		for (j, (((&x, &y), &z), o)) in a_rem.iter().zip(b_rem).zip(c_rem).zip(out_chunks.into_remainder()).enumerate() {
			*o = fmsubadd_scalar_at(base + j, x, y, z);
		}
	}
}

/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn fmaddsub_ph512_intrinsic(a: &[u16; 32], b: &[u16; 32], c: &[u16; 32]) -> [u16; 32] {
	unsafe {
		let va = loadu_ph512(a.as_ptr());
		let vb = loadu_ph512(b.as_ptr());
		let vc = loadu_ph512(c.as_ptr());
		let vr = _mm512_fmaddsub_ph(va, vb, vc);
		let mut out = [0u16; 32];
		storeu_ph512(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn fmsubadd_ph512_intrinsic(a: &[u16; 32], b: &[u16; 32], c: &[u16; 32]) -> [u16; 32] {
	unsafe {
		let va = loadu_ph512(a.as_ptr());
		let vb = loadu_ph512(b.as_ptr());
		let vc = loadu_ph512(c.as_ptr());
		let vr = _mm512_fmsubadd_ph(va, vb, vc);
		let mut out = [0u16; 32];
		storeu_ph512(out.as_mut_ptr(), vr);
		out
	}
}

// `cvtph_pd`/`cvtpd_ph`: fixed-width only, same reasoning as DQ's `broadcast`/
// `extract`/`insert` family. The `__m128h` side is always a full 8-lane
// carrier regardless of how many lanes are actually live (2/4/8), so a slice
// API would have to zero-pad every chunk into an 8-wide buffer for no benefit
// (not about scalar-reference availability here, purely the carrier-width
// mismatch). `cvtph_pd` is numerically exact (every f16 round-trips losslessly
// through f64); `cvtpd_ph` is lossy and rounding-mode-sensitive (hardware
// rounds f64->f16 directly: `as f32 as u16-bits` in Rust risks double
// rounding) -> neither gets an auto-cascade either way.
use core::arch::x86_64::{__m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i};
use core::arch::x86_64::{
	_mm256_mask_cmp_ph_mask, _mm256_mask_cvtpd_ph, _mm256_mask_cvtph_pd, _mm256_mask_getmant_ph, _mm256_mask_reduce_ph, _mm256_mask_roundscale_ph,
	_mm256_maskz_cvtpd_ph, _mm256_maskz_cvtph_pd, _mm256_maskz_getmant_ph, _mm256_maskz_reduce_ph, _mm256_maskz_roundscale_ph,
	_mm512_mask_add_round_ph, _mm512_mask_cmp_ph_mask,
	_mm512_mask_cvtpd_ph, _mm512_mask_cvtph_pd, _mm512_mask_div_round_ph, _mm512_mask_fmadd_round_ph,
	_mm512_mask_fmaddsub_round_ph, _mm512_mask_fmsub_round_ph, _mm512_mask_fmsubadd_round_ph,
	_mm512_mask_fnmadd_round_ph, _mm512_mask_fnmsub_round_ph, _mm512_mask_getmant_ph, _mm512_mask_mul_round_ph, _mm512_mask_reduce_ph,
	_mm512_mask_roundscale_ph, _mm512_mask_sqrt_round_ph,
	_mm512_mask_sub_round_ph, _mm512_maskz_add_round_ph, _mm512_maskz_cvtpd_ph,
	_mm512_maskz_cvtph_pd, _mm512_maskz_div_round_ph, _mm512_maskz_fmadd_round_ph, _mm512_maskz_fmaddsub_round_ph,
	_mm512_maskz_fmsub_round_ph, _mm512_maskz_fmsubadd_round_ph, _mm512_maskz_fnmadd_round_ph,
	_mm512_maskz_fnmsub_round_ph, _mm512_maskz_getmant_ph, _mm512_maskz_mul_round_ph, _mm512_maskz_reduce_ph,
	_mm512_maskz_roundscale_ph, _mm512_maskz_sqrt_round_ph,
	_mm512_maskz_sub_round_ph, _mm_mask_cmp_ph_mask, _mm_mask_cvtpd_ph, _mm_mask_cvtph_pd,
	_mm_mask_getmant_ph, _mm_mask_reduce_ph, _mm_mask_roundscale_ph,
	_mm_maskz_cvtpd_ph, _mm_maskz_cvtph_pd, _mm_maskz_getmant_ph, _mm_maskz_reduce_ph, _mm_maskz_roundscale_ph,
};
use core::arch::x86_64::{
	_mm256_cvtepi16_ph, _mm256_cvtepi32_ph, _mm256_cvtepi64_ph, _mm256_cvtepu16_ph,
	_mm256_cvtepu32_ph, _mm256_cvtepu64_ph, _mm256_cvtpd_ph, _mm256_cvtph_epi16,
	_mm256_cvtph_epi32, _mm256_cvtph_epi64, _mm256_cvtph_epu16, _mm256_cvtph_epu32,
	_mm256_cvtph_epu64, _mm256_cvtph_pd, _mm256_cvttph_epi16, _mm256_cvttph_epi32,
	_mm256_cvttph_epi64, _mm256_cvttph_epu16, _mm256_cvttph_epu32, _mm256_cvttph_epu64,
	_mm256_cvtxph_ps, _mm256_cvtxps_ph, _mm512_cvt_roundepi16_ph, _mm512_cvt_roundepi32_ph,
	_mm512_cvt_roundepi64_ph, _mm512_cvt_roundepu16_ph, _mm512_cvt_roundepu32_ph,
	_mm512_cvt_roundepu64_ph, _mm512_cvt_roundph_epi16, _mm512_cvt_roundph_epi32,
	_mm512_cvt_roundph_epi64, _mm512_cvt_roundph_epu16, _mm512_cvt_roundph_epu32,
	_mm512_cvt_roundph_epu64, _mm512_cvtepi16_ph, _mm512_cvtepi32_ph, _mm512_cvtepi64_ph,
	_mm512_cvtepu16_ph, _mm512_cvtepu32_ph, _mm512_cvtepu64_ph, _mm512_cvtpd_ph,
	_mm512_cvtph_epi16, _mm512_cvtph_epi32, _mm512_cvtph_epi64, _mm512_cvtph_epu16,
	_mm512_cvtph_epu32, _mm512_cvtph_epu64, _mm512_cvtph_pd, _mm512_cvttph_epi16,
	_mm512_cvttph_epi32, _mm512_cvttph_epi64, _mm512_cvttph_epu16, _mm512_cvttph_epu32,
	_mm512_cvttph_epu64, _mm512_cvtx_roundph_ps, _mm512_cvtx_roundps_ph, _mm512_cvtxph_ps,
	_mm512_cvtxps_ph, _mm_cvtepi16_ph, _mm_cvtepi32_ph, _mm_cvtepi64_ph, _mm_cvtepu16_ph,
	_mm_cvtepu32_ph, _mm_cvtepu64_ph, _mm_cvtpd_ph, _mm_cvtph_epi16, _mm_cvtph_epi32,
	_mm_cvtph_epi64, _mm_cvtph_epu16, _mm_cvtph_epu32, _mm_cvtph_epu64, _mm_cvtph_pd,
	_mm_cvttph_epi16, _mm_cvttph_epi32, _mm_cvttph_epi64, _mm_cvttph_epu16, _mm_cvttph_epu32,
	_mm_cvttph_epu64, _mm_cvtxph_ps, _mm_cvtxps_ph
};

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_f64x2, intrinsic_fn = ph_to_f64x2_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m128d, out_storeu = core::arch::x86_64::_mm_storeu_pd,
	intrinsic = _mm_cvtph_pd,
	fixed_doc = "FP16 -> `f64`, low 2 lanes of the `__m128h` carrier (`vcvtph2pd`, 128-bit). Exact, no rounding.",
}
simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_f64x4, intrinsic_fn = ph_to_f64x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m256d, out_storeu = core::arch::x86_64::_mm256_storeu_pd,
	intrinsic = _mm256_cvtph_pd,
	fixed_doc = "FP16 -> `f64`, low 4 lanes of the `__m128h` carrier (`vcvtph2pd`, 256-bit). Exact, no rounding.",
}
simd_cvt_widen! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_f64x8, intrinsic_fn = ph_to_f64x8_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m512d, out_storeu = core::arch::x86_64::_mm512_storeu_pd,
	intrinsic = _mm512_cvtph_pd,
	fixed_doc = "FP16 -> `f64`, all 8 lanes of the `__m128h` carrier (`vcvtph2pd`, 512-bit). Exact, no rounding.",
}

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = f64x2_to_ph, intrinsic_fn = f64x2_to_ph_intrinsic,
	in_width = 2, out_width = 8,
	in_elem = f64, in_vec = __m128d, in_loadu = core::arch::x86_64::_mm_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtpd_ph,
	fixed_doc = "`f64` -> FP16, into the low 2 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtpd2ph`, 128-bit). Fixed-width only: hardware rounds f64->f16 directly, not reproducible via `as f32 as u16-bits` in Rust.",
}
simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = f64x4_to_ph, intrinsic_fn = f64x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = f64, in_vec = __m256d, in_loadu = core::arch::x86_64::_mm256_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtpd_ph,
	fixed_doc = "`f64` -> FP16, into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtpd2ph`, 256-bit). Fixed-width only, see [`Avx512Fp16Vl::f64x2_to_ph`].",
}
simd_cvt_narrow! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = f64x8_to_ph, intrinsic_fn = f64x8_to_ph_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = core::arch::x86_64::_mm512_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm512_cvtpd_ph,
	fixed_doc = "`f64` -> FP16, all 8 lanes (`vcvtpd2ph`, 512-bit). Fixed-width only, see [`Avx512Fp16Vl::f64x2_to_ph`].",
}

// i16/u16 <-> FP16: equal lane count at every width (both are 16-bit-per-
// lane types): plain `simd_cvt` throughout, no widen/narrow needed.

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i16x8_to_ph, intrinsic_fn = i16x8_to_ph_intrinsic,
	width = 8,
	in_elem = i16, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepi16_ph,
	fixed_doc = "`i16` to FP16, round-to-nearest-even (`vcvtw2ph`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i16x16_to_ph, intrinsic_fn = i16x16_to_ph_intrinsic,
	width = 16,
	in_elem = i16, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm256_cvtepi16_ph,
	fixed_doc = "`i16` to FP16, round-to-nearest-even (`vcvtw2ph`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i16x32_to_ph, intrinsic_fn = i16x32_to_ph_intrinsic,
	width = 32,
	in_elem = i16, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m512h, out_storeu = storeu_ph512,
	intrinsic = _mm512_cvtepi16_ph,
	fixed_doc = "`i16` to FP16, round-to-nearest-even (`vcvtw2ph`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u16x8_to_ph, intrinsic_fn = u16x8_to_ph_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepu16_ph,
	fixed_doc = "`u16` to FP16, round-to-nearest-even (`vcvtuw2ph`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u16x16_to_ph, intrinsic_fn = u16x16_to_ph_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm256_cvtepu16_ph,
	fixed_doc = "`u16` to FP16, round-to-nearest-even (`vcvtuw2ph`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u16x32_to_ph, intrinsic_fn = u16x32_to_ph_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m512h, out_storeu = storeu_ph512,
	intrinsic = _mm512_cvtepu16_ph,
	fixed_doc = "`u16` to FP16, round-to-nearest-even (`vcvtuw2ph`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i16x8, intrinsic_fn = ph_to_i16x8_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i16, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epi16,
	fixed_doc = "FP16 to `i16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2w`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i16x8_trunc, intrinsic_fn = ph_to_i16x8_trunc_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i16, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epi16,
	fixed_doc = "FP16 to `i16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2w`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i16x16, intrinsic_fn = ph_to_i16x16_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = i16, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epi16,
	fixed_doc = "FP16 to `i16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2w`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i16x16_trunc, intrinsic_fn = ph_to_i16x16_trunc_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = i16, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epi16,
	fixed_doc = "FP16 to `i16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2w`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i16x32, intrinsic_fn = ph_to_i16x32_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = i16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epi16,
	fixed_doc = "FP16 to `i16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2w`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i16x32_trunc, intrinsic_fn = ph_to_i16x32_trunc_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = i16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epi16,
	fixed_doc = "FP16 to `i16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2w`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u16x8, intrinsic_fn = ph_to_u16x8_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u16, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epu16,
	fixed_doc = "FP16 to `u16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2uw`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u16x8_trunc, intrinsic_fn = ph_to_u16x8_trunc_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u16, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epu16,
	fixed_doc = "FP16 to `u16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2uw`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u16x16, intrinsic_fn = ph_to_u16x16_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = u16, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epu16,
	fixed_doc = "FP16 to `u16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2uw`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u16x16_trunc, intrinsic_fn = ph_to_u16x16_trunc_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = u16, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epu16,
	fixed_doc = "FP16 to `u16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2uw`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u16x32, intrinsic_fn = ph_to_u16x32_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = u16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epu16,
	fixed_doc = "FP16 to `u16`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel, *not* a saturating cast (`vcvtph2uw`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u16x32_trunc, intrinsic_fn = ph_to_u16x32_trunc_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = u16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epu16,
	fixed_doc = "FP16 to `u16`, truncating toward zero. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvttph2uw`, 512-bit).",
}
// i32/u32 <-> FP16: `__m128h` carrier is oversized relative to the real
// i32 lane count only at 128-bit (4 real lanes in an 8-lane carrier) -
// `simd_cvt_narrow`/`simd_cvt_widen` there, plain `simd_cvt` at 256/512
// where the carrier width and 32-bit lane count happen to coincide.

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i32x4_to_ph, intrinsic_fn = i32x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = i32, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepi32_ph,
	fixed_doc = "`i32` to FP16, round-to-nearest-even, into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtdq2ph`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i32x8_to_ph, intrinsic_fn = i32x8_to_ph_intrinsic,
	width = 8,
	in_elem = i32, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtepi32_ph,
	fixed_doc = "`i32` to FP16, round-to-nearest-even (`vcvtdq2ph`, 256-bit). See [`Avx512Fp16Vl::i32x4_to_ph`] for the 128-bit carrier-padding case.",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i32x16_to_ph, intrinsic_fn = i32x16_to_ph_intrinsic,
	width = 16,
	in_elem = i32, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvtepi32_ph,
	fixed_doc = "`i32` to FP16, round-to-nearest-even (`vcvtdq2ph`, 512-bit). See [`Avx512Fp16::i32x4_to_ph`] for the 128-bit carrier-padding case.",
}

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u32x4_to_ph, intrinsic_fn = u32x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = u32, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepu32_ph,
	fixed_doc = "`u32` to FP16, round-to-nearest-even, into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtudq2ph`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u32x8_to_ph, intrinsic_fn = u32x8_to_ph_intrinsic,
	width = 8,
	in_elem = u32, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtepu32_ph,
	fixed_doc = "`u32` to FP16, round-to-nearest-even (`vcvtudq2ph`, 256-bit). See [`Avx512Fp16Vl::i32x4_to_ph`] for the 128-bit carrier-padding case.",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u32x16_to_ph, intrinsic_fn = u32x16_to_ph_intrinsic,
	width = 16,
	in_elem = u32, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvtepu32_ph,
	fixed_doc = "`u32` to FP16, round-to-nearest-even (`vcvtudq2ph`, 512-bit). See [`Avx512Fp16::i32x4_to_ph`] for the 128-bit carrier-padding case.",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i32x4, intrinsic_fn = ph_to_i32x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i32, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epi32,
	fixed_doc = "FP16 to `i32`, round-to-nearest-even, low 4 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2dq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i32x4_trunc, intrinsic_fn = ph_to_i32x4_trunc_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i32, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epi32,
	fixed_doc = "FP16 to `i32`, truncating toward zero, low 4 lanes of the `__m128h` carrier (`vcvttph2dq`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i32x8, intrinsic_fn = ph_to_i32x8_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i32, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epi32,
	fixed_doc = "FP16 to `i32`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2dq`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i32x8_trunc, intrinsic_fn = ph_to_i32x8_trunc_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i32, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epi32,
	fixed_doc = "FP16 to `i32`, truncating toward zero (`vcvttph2dq`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i32x16, intrinsic_fn = ph_to_i32x16_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = i32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epi32,
	fixed_doc = "FP16 to `i32`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2dq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i32x16_trunc, intrinsic_fn = ph_to_i32x16_trunc_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = i32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epi32,
	fixed_doc = "FP16 to `i32`, truncating toward zero (`vcvttph2dq`, 512-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u32x4, intrinsic_fn = ph_to_u32x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u32, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epu32,
	fixed_doc = "FP16 to `u32`, round-to-nearest-even, low 4 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2udq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u32x4_trunc, intrinsic_fn = ph_to_u32x4_trunc_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u32, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epu32,
	fixed_doc = "FP16 to `u32`, truncating toward zero, low 4 lanes of the `__m128h` carrier (`vcvttph2udq`, 128-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u32x8, intrinsic_fn = ph_to_u32x8_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u32, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epu32,
	fixed_doc = "FP16 to `u32`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2udq`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u32x8_trunc, intrinsic_fn = ph_to_u32x8_trunc_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u32, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epu32,
	fixed_doc = "FP16 to `u32`, truncating toward zero (`vcvttph2udq`, 256-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u32x16, intrinsic_fn = ph_to_u32x16_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = u32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epu32,
	fixed_doc = "FP16 to `u32`, round-to-nearest-even. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2udq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u32x16_trunc, intrinsic_fn = ph_to_u32x16_trunc_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = u32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epu32,
	fixed_doc = "FP16 to `u32`, truncating toward zero (`vcvttph2udq`, 512-bit).",
}
// i64/u64 <-> FP16: `__m128h` carrier stays fixed at 8 lanes across ALL
// tiers (unlike i32/u32, where 512-bit's carrier grows to `__m256h`) -
// `simd_cvt_narrow`/`simd_cvt_widen` at every width, matching the
// `f64x2/4/8_to_ph`/`ph_to_f64x2/4/8` precedent above exactly.

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i64x2_to_ph, intrinsic_fn = i64x2_to_ph_intrinsic,
	in_width = 2, out_width = 8,
	in_elem = i64, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepi64_ph,
	fixed_doc = "`i64` to FP16, round-to-nearest-even, into the low 2 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtqq2ph`, 128-bit).",
}

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = i64x4_to_ph, intrinsic_fn = i64x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = i64, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtepi64_ph,
	fixed_doc = "`i64` to FP16, round-to-nearest-even, into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtqq2ph`, 256-bit).",
}

simd_cvt_narrow! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i64x8_to_ph, intrinsic_fn = i64x8_to_ph_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm512_cvtepi64_ph,
	fixed_doc = "`i64` to FP16, round-to-nearest-even, into the low 8 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtqq2ph`, 512-bit).",
}

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u64x2_to_ph, intrinsic_fn = u64x2_to_ph_intrinsic,
	in_width = 2, out_width = 8,
	in_elem = u64, in_vec = __m128i, in_loadu = core::arch::x86_64::_mm_loadu_si128,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtepu64_ph,
	fixed_doc = "`u64` to FP16, round-to-nearest-even, into the low 2 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtuqq2ph`, 128-bit).",
}

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = u64x4_to_ph, intrinsic_fn = u64x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = u64, in_vec = __m256i, in_loadu = core::arch::x86_64::_mm256_loadu_si256,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtepu64_ph,
	fixed_doc = "`u64` to FP16, round-to-nearest-even, into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtuqq2ph`, 256-bit).",
}

simd_cvt_narrow! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u64x8_to_ph, intrinsic_fn = u64x8_to_ph_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm512_cvtepu64_ph,
	fixed_doc = "`u64` to FP16, round-to-nearest-even, into the low 8 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed (`vcvtuqq2ph`, 512-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i64x2, intrinsic_fn = ph_to_i64x2_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epi64,
	fixed_doc = "FP16 to `i64`, round-to-nearest-even, low 2 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2qq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i64x2_trunc, intrinsic_fn = ph_to_i64x2_trunc_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epi64,
	fixed_doc = "FP16 to `i64`, truncating toward zero, low 2 lanes of the `__m128h` carrier (`vcvttph2qq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i64x4, intrinsic_fn = ph_to_i64x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epi64,
	fixed_doc = "FP16 to `i64`, round-to-nearest-even, low 4 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2qq`, 256-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_i64x4_trunc, intrinsic_fn = ph_to_i64x4_trunc_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epi64,
	fixed_doc = "FP16 to `i64`, truncating toward zero, low 4 lanes of the `__m128h` carrier (`vcvttph2qq`, 256-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i64x8, intrinsic_fn = ph_to_i64x8_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epi64,
	fixed_doc = "FP16 to `i64`, round-to-nearest-even, low 8 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2qq`, 512-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i64x8_trunc, intrinsic_fn = ph_to_i64x8_trunc_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epi64,
	fixed_doc = "FP16 to `i64`, truncating toward zero, low 8 lanes of the `__m128h` carrier (`vcvttph2qq`, 512-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u64x2, intrinsic_fn = ph_to_u64x2_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvtph_epu64,
	fixed_doc = "FP16 to `u64`, round-to-nearest-even, low 2 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2uqq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u64x2_trunc, intrinsic_fn = ph_to_u64x2_trunc_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m128i, out_storeu = core::arch::x86_64::_mm_storeu_si128,
	intrinsic = _mm_cvttph_epu64,
	fixed_doc = "FP16 to `u64`, truncating toward zero, low 2 lanes of the `__m128h` carrier (`vcvttph2uqq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u64x4, intrinsic_fn = ph_to_u64x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvtph_epu64,
	fixed_doc = "FP16 to `u64`, round-to-nearest-even, low 4 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2uqq`, 256-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_u64x4_trunc, intrinsic_fn = ph_to_u64x4_trunc_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m256i, out_storeu = core::arch::x86_64::_mm256_storeu_si256,
	intrinsic = _mm256_cvttph_epu64,
	fixed_doc = "FP16 to `u64`, truncating toward zero, low 4 lanes of the `__m128h` carrier (`vcvttph2uqq`, 256-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u64x8, intrinsic_fn = ph_to_u64x8_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvtph_epu64,
	fixed_doc = "FP16 to `u64`, round-to-nearest-even, low 8 lanes of the `__m128h` carrier. Out-of-range or NaN inputs produce the HW integer-indefinite sentinel (`vcvtph2uqq`, 512-bit).",
}

simd_cvt_widen! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u64x8_trunc, intrinsic_fn = ph_to_u64x8_trunc_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvttph_epu64,
	fixed_doc = "FP16 to `u64`, truncating toward zero, low 8 lanes of the `__m128h` carrier (`vcvttph2uqq`, 512-bit).",
}
// Embedded-rounding (`_round_`) int<->FP16 variants: SAE-only, 512-bit
// exclusive (matches the `_round_ph` family and DQ's round-cvt above).
// All have equal in/out lane counts at 512-bit, so `simd_cvt_imm` (not a
// widen/narrow-imm variant) fits every one of them.

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i16x32_to_ph_round, intrinsic_fn = i16x32_to_ph_round_intrinsic,
	width = 32,
	in_elem = i16, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m512h, out_storeu = storeu_ph512,
	intrinsic = _mm512_cvt_roundepi16_ph,
	fixed_doc = "`i16` to FP16 with explicit rounding control (`vcvtw2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u16x32_to_ph_round, intrinsic_fn = u16x32_to_ph_round_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m512h, out_storeu = storeu_ph512,
	intrinsic = _mm512_cvt_roundepu16_ph,
	fixed_doc = "`u16` to FP16 with explicit rounding control (`vcvtuw2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i32x16_to_ph_round, intrinsic_fn = i32x16_to_ph_round_intrinsic,
	width = 16,
	in_elem = i32, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvt_roundepi32_ph,
	fixed_doc = "`i32` to FP16 with explicit rounding control (`vcvtdq2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u32x16_to_ph_round, intrinsic_fn = u32x16_to_ph_round_intrinsic,
	width = 16,
	in_elem = u32, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvt_roundepu32_ph,
	fixed_doc = "`u32` to FP16 with explicit rounding control (`vcvtudq2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = i64x8_to_ph_round, intrinsic_fn = i64x8_to_ph_round_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm512_cvt_roundepi64_ph,
	fixed_doc = "`i64` to FP16 with explicit rounding control (`vcvtqq2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = u64x8_to_ph_round, intrinsic_fn = u64x8_to_ph_round_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = core::arch::x86_64::_mm512_loadu_si512,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm512_cvt_roundepu64_ph,
	fixed_doc = "`u64` to FP16 with explicit rounding control (`vcvtuqq2ph`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i16x32_round, intrinsic_fn = ph_to_i16x32_round_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = i16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epi16,
	fixed_doc = "FP16 to `i16` with explicit rounding control (`vcvtph2w`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u16x32_round, intrinsic_fn = ph_to_u16x32_round_intrinsic,
	width = 32,
	in_elem = u16, in_vec = __m512h, in_loadu = loadu_ph512,
	out_elem = u16, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epu16,
	fixed_doc = "FP16 to `u16` with explicit rounding control (`vcvtph2uw`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i32x16_round, intrinsic_fn = ph_to_i32x16_round_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = i32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epi32,
	fixed_doc = "FP16 to `i32` with explicit rounding control (`vcvtph2dq`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u32x16_round, intrinsic_fn = ph_to_u32x16_round_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = u32, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epu32,
	fixed_doc = "FP16 to `u32` with explicit rounding control (`vcvtph2udq`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_i64x8_round, intrinsic_fn = ph_to_i64x8_round_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = i64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epi64,
	fixed_doc = "FP16 to `i64` with explicit rounding control (`vcvtph2qq`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_u64x8_round, intrinsic_fn = ph_to_u64x8_round_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = u64, out_vec = __m512i, out_storeu = core::arch::x86_64::_mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundph_epu64,
	fixed_doc = "FP16 to `u64` with explicit rounding control (`vcvtph2uqq`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}
// `cvtxps_ph`/`cvtxph_ps`: reduced-precision-named ("x") sibling of F16C's
// `cvtps_ph`/`cvtph_ps` (`f16c.rs`): a different instruction
// (`vcvtps2phx`/`vcvtph2psx`, no immediate rounding-control operand,
// MXCSR/SAE-controlled instead), so distinct names to avoid colliding
// with F16C's `f32_to_f16x4`-style wrappers. Same carrier-mismatch shape
// as `f32/i32x4_to_ph` above: `simd_cvt_narrow`/`simd_cvt_widen` only at
// 128-bit, plain `simd_cvt` at 256/512.

simd_cvt_narrow! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = f32x4_to_ph_x, intrinsic_fn = f32x4_to_ph_x_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = f32, in_vec = __m128, in_loadu = core::arch::x86_64::_mm_loadu_ps,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm_cvtxps_ph,
	fixed_doc = "`f32` to FP16 (`vcvtps2phx`, 128-bit), into the low 4 lanes of an 8-lane `__m128h` carrier, upper lanes zeroed. Distinct instruction from F16C's `cvtps_ph` (no immediate rounding control, MXCSR/SAE-based instead) - see [`super::avx::f16c::F16c`].",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = f32x8_to_ph_x, intrinsic_fn = f32x8_to_ph_x_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = core::arch::x86_64::_mm256_loadu_ps,
	out_elem = u16, out_vec = __m128h, out_storeu = storeu_ph128,
	intrinsic = _mm256_cvtxps_ph,
	fixed_doc = "`f32` to FP16 (`vcvtps2phx`, 256-bit). See [`Avx512Fp16Vl::f32x4_to_ph_x`].",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = f32x16_to_ph_x, intrinsic_fn = f32x16_to_ph_x_intrinsic,
	width = 16,
	in_elem = f32, in_vec = __m512, in_loadu = core::arch::x86_64::_mm512_loadu_ps,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvtxps_ph,
	fixed_doc = "`f32` to FP16 (`vcvtps2phx`, 512-bit). See [`Avx512Fp16::f32x4_to_ph_x`].",
}

simd_cvt_widen! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_f32x4_x, intrinsic_fn = ph_to_f32x4_x_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f32, out_vec = __m128, out_storeu = core::arch::x86_64::_mm_storeu_ps,
	intrinsic = _mm_cvtxph_ps,
	fixed_doc = "FP16 to `f32` (`vcvtph2psx`, 128-bit), low 4 lanes of the `__m128h` carrier. Distinct instruction from F16C's `cvtph_ps` - see [`super::avx::f16c::F16c`].",
}

simd_cvt! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = ph_to_f32x8_x, intrinsic_fn = ph_to_f32x8_x_intrinsic,
	width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f32, out_vec = __m256, out_storeu = core::arch::x86_64::_mm256_storeu_ps,
	intrinsic = _mm256_cvtxph_ps,
	fixed_doc = "FP16 to `f32` (`vcvtph2psx`, 256-bit). See [`Avx512Fp16Vl::ph_to_f32x4_x`].",
}

simd_cvt! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_f32x16_x, intrinsic_fn = ph_to_f32x16_x_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = f32, out_vec = __m512, out_storeu = core::arch::x86_64::_mm512_storeu_ps,
	intrinsic = _mm512_cvtxph_ps,
	fixed_doc = "FP16 to `f32` (`vcvtph2psx`, 512-bit). See [`Avx512Fp16::ph_to_f32x4_x`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = f32x16_to_ph_x_round, intrinsic_fn = f32x16_to_ph_x_round_intrinsic,
	width = 16,
	in_elem = f32, in_vec = __m512, in_loadu = core::arch::x86_64::_mm512_loadu_ps,
	out_elem = u16, out_vec = __m256h, out_storeu = storeu_ph256,
	intrinsic = _mm512_cvtx_roundps_ph,
	fixed_doc = "`f32` to FP16 with explicit rounding control (`vcvtps2phx`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = ph_to_f32x16_x_round, intrinsic_fn = ph_to_f32x16_x_round_intrinsic,
	width = 16,
	in_elem = u16, in_vec = __m256h, in_loadu = loadu_ph256,
	out_elem = f32, out_vec = __m512, out_storeu = core::arch::x86_64::_mm512_storeu_ps,
	intrinsic = _mm512_cvtx_roundph_ps,
	fixed_doc = "FP16 to `f32` with explicit rounding control (`vcvtph2psx`, 512-bit). Same `IMM8` encoding as [`super::avx512dq::Avx512Dq::f64_to_i64x8_round`].",
}

// `cmp_ph_mask`: reuses the 5 predicates `avx512f.rs` already established for
// `cmp_ps_mask`/`cmp_pd_mask` (EQ_OQ/LT_OQ/LE_OQ/GT_OQ/GE_OQ), but returns the
// **raw mask** (`u8`/`u16`/`u32`) instead of that file's all-1s/all-0s lane
// vector convention: this is a genuine bit-mask result, not a lane mask (see
// `mask-newtype` design note; naming carries the distinction, not a wrapper
// type).
macro_rules! fp16_cmp_mask_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16 {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 32], b: [u16; 32]) -> u32 {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $intrinsic_fn(a: &[u16; 32], b: &[u16; 32]) -> u32 {
			unsafe {
				let va = loadu_ph512(a.as_ptr());
				let vb = loadu_ph512(b.as_ptr());
				core::arch::x86_64::_mm512_cmp_ph_mask::<{ $pred }>(va, vb) as u32
			}
		}
	};
}
macro_rules! fp16_cmp_mask_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 16], b: [u16; 16]) -> u16 {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 16], b: &[u16; 16]) -> u16 {
			unsafe {
				let va = loadu_ph256(a.as_ptr());
				let vb = loadu_ph256(b.as_ptr());
				core::arch::x86_64::_mm256_cmp_ph_mask::<{ $pred }>(va, vb)
			}
		}
	};
}
macro_rules! fp16_cmp_mask_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 8], b: [u16; 8]) -> u8 {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 8], b: &[u16; 8]) -> u8 {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vb = loadu_ph128(b.as_ptr());
				core::arch::x86_64::_mm_cmp_ph_mask::<{ $pred }>(va, vb)
			}
		}
	};
}

fp16_cmp_mask_512!(cmpeq_ph_mask_u16x32, cmpeq_ph512_mask_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"Lane equality bit-mask (`vcmpph` EQ_OQ, 512-bit). Bit `i` set iff `a[i]==b[i]`.");
fp16_cmp_mask_512!(cmplt_ph_mask_u16x32, cmplt_ph512_mask_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"Lane less-than bit-mask (`vcmpph` LT_OQ, 512-bit). Bit `i` set iff `a[i]<b[i]`.");
fp16_cmp_mask_512!(cmple_ph_mask_u16x32, cmple_ph512_mask_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"Lane less-or-equal bit-mask (`vcmpph` LE_OQ, 512-bit). Bit `i` set iff `a[i]<=b[i]`.");
fp16_cmp_mask_512!(cmpgt_ph_mask_u16x32, cmpgt_ph512_mask_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"Lane greater-than bit-mask (`vcmpph` GT_OQ, 512-bit). Bit `i` set iff `a[i]>b[i]`.");
fp16_cmp_mask_512!(cmpge_ph_mask_u16x32, cmpge_ph512_mask_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"Lane greater-or-equal bit-mask (`vcmpph` GE_OQ, 512-bit). Bit `i` set iff `a[i]>=b[i]`.");

fp16_cmp_mask_256!(cmpeq_ph_mask_u16x16, cmpeq_ph256_mask_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"Lane equality bit-mask (`vcmpph` EQ_OQ, 256-bit). Bit `i` set iff `a[i]==b[i]`.");
fp16_cmp_mask_256!(cmplt_ph_mask_u16x16, cmplt_ph256_mask_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"Lane less-than bit-mask (`vcmpph` LT_OQ, 256-bit). Bit `i` set iff `a[i]<b[i]`.");
fp16_cmp_mask_256!(cmple_ph_mask_u16x16, cmple_ph256_mask_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"Lane less-or-equal bit-mask (`vcmpph` LE_OQ, 256-bit). Bit `i` set iff `a[i]<=b[i]`.");
fp16_cmp_mask_256!(cmpgt_ph_mask_u16x16, cmpgt_ph256_mask_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"Lane greater-than bit-mask (`vcmpph` GT_OQ, 256-bit). Bit `i` set iff `a[i]>b[i]`.");
fp16_cmp_mask_256!(cmpge_ph_mask_u16x16, cmpge_ph256_mask_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"Lane greater-or-equal bit-mask (`vcmpph` GE_OQ, 256-bit). Bit `i` set iff `a[i]>=b[i]`.");

fp16_cmp_mask_128!(cmpeq_ph_mask_u16x8, cmpeq_ph128_mask_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"Lane equality bit-mask (`vcmpph` EQ_OQ, 128-bit). Bit `i` set iff `a[i]==b[i]`.");
fp16_cmp_mask_128!(cmplt_ph_mask_u16x8, cmplt_ph128_mask_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"Lane less-than bit-mask (`vcmpph` LT_OQ, 128-bit). Bit `i` set iff `a[i]<b[i]`.");
fp16_cmp_mask_128!(cmple_ph_mask_u16x8, cmple_ph128_mask_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"Lane less-or-equal bit-mask (`vcmpph` LE_OQ, 128-bit). Bit `i` set iff `a[i]<=b[i]`.");
fp16_cmp_mask_128!(cmpgt_ph_mask_u16x8, cmpgt_ph128_mask_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"Lane greater-than bit-mask (`vcmpph` GT_OQ, 128-bit). Bit `i` set iff `a[i]>b[i]`.");
fp16_cmp_mask_128!(cmpge_ph_mask_u16x8, cmpge_ph128_mask_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"Lane greater-or-equal bit-mask (`vcmpph` GE_OQ, 128-bit). Bit `i` set iff `a[i]>=b[i]`.");

// Packed-complex `_pch`/`_sch`: each `__m*h` register holds interleaved
// (re, im) pairs, `complex = v[2k] + i*v[2k+1]`. No scalar-Rust closure for
// complex multiply/FMA -> fixed-width only, via `simd_binop_fixed!`/
// `simd_ternop_fixed!` (same shape as `_sch`'s "lane0 pair computed, rest of
// `a` passed through": HW already handles the passthrough, so `_sch` reuses
// the exact same macros as the 128-bit `_pch` forms, just a different
// intrinsic). `mul_pch`/`fmul_pch` and `cmul_pch`/`fcmul_pch` are literal
// aliases in stdarch (`_mm_fmul_pch` body is just `_mm_mul_pch(a, b)`, same
// for the `_sch` pair): only the canonical (non-`f`-prefixed) name ships
// here, no point wrapping the same instruction twice under two names.
// `cmul`/`fcmul` implement the TODO's "conj" (complex-conjugate-of-b
// multiply); `fmadd_pch`/`fcmadd_pch` are real, distinct FMA ops (no alias).
macro_rules! fp16_binop_fixed_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_binop_fixed! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_binop_fixed_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_binop_fixed! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_binop_fixed_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_binop_fixed! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_ternop_fixed_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_ternop_fixed! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_ternop_fixed_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_ternop_fixed! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_ternop_fixed_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_ternop_fixed! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

fp16_binop_fixed_512!(mul_pch_u16x32, mul_pch512_intrinsic, core::arch::x86_64::_mm512_mul_pch,
	"Packed complex multiply (`vfmulcph`, 512-bit): `(a+bi)(c+di) = (ac-bd)+(ad+bc)i` per adjacent pair.");
fp16_binop_fixed_256!(mul_pch_u16x16, mul_pch256_intrinsic, core::arch::x86_64::_mm256_mul_pch,
	"Packed complex multiply (`vfmulcph`, 256-bit). See [`Avx512Fp16::mul_pch_u16x32`].");
fp16_binop_fixed_128!(mul_pch_u16x8, mul_pch128_intrinsic, core::arch::x86_64::_mm_mul_pch,
	"Packed complex multiply (`vfmulcph`, 128-bit). See [`Avx512Fp16::mul_pch_u16x32`].");

fp16_binop_fixed_512!(cmul_pch_u16x32, cmul_pch512_intrinsic, core::arch::x86_64::_mm512_cmul_pch,
	"Packed complex multiply by the conjugate of `b` (`vfcmulcph`, 512-bit): `(a+bi)*conj(c+di) = (ac+bd)+(bc-ad)i`.");
fp16_binop_fixed_256!(cmul_pch_u16x16, cmul_pch256_intrinsic, core::arch::x86_64::_mm256_cmul_pch,
	"Packed complex multiply by the conjugate of `b` (`vfcmulcph`, 256-bit). See [`Avx512Fp16::cmul_pch_u16x32`].");
fp16_binop_fixed_128!(cmul_pch_u16x8, cmul_pch128_intrinsic, core::arch::x86_64::_mm_cmul_pch,
	"Packed complex multiply by the conjugate of `b` (`vfcmulcph`, 128-bit). See [`Avx512Fp16::cmul_pch_u16x32`].");

fp16_ternop_fixed_512!(fmadd_pch_u16x32, fmadd_pch512_intrinsic, core::arch::x86_64::_mm512_fmadd_pch,
	"Packed complex FMA (`vfmaddcph`, 512-bit): `a*b + c` per adjacent complex pair.");
fp16_ternop_fixed_256!(fmadd_pch_u16x16, fmadd_pch256_intrinsic, core::arch::x86_64::_mm256_fmadd_pch,
	"Packed complex FMA (`vfmaddcph`, 256-bit). See [`Avx512Fp16::fmadd_pch_u16x32`].");
fp16_ternop_fixed_128!(fmadd_pch_u16x8, fmadd_pch128_intrinsic, core::arch::x86_64::_mm_fmadd_pch,
	"Packed complex FMA (`vfmaddcph`, 128-bit). See [`Avx512Fp16::fmadd_pch_u16x32`].");

fp16_ternop_fixed_512!(fcmadd_pch_u16x32, fcmadd_pch512_intrinsic, core::arch::x86_64::_mm512_fcmadd_pch,
	"Packed complex FMA, `b` conjugated (`vfcmaddcph`, 512-bit): `a*conj(b) + c` per adjacent complex pair.");
fp16_ternop_fixed_256!(fcmadd_pch_u16x16, fcmadd_pch256_intrinsic, core::arch::x86_64::_mm256_fcmadd_pch,
	"Packed complex FMA, `b` conjugated (`vfcmaddcph`, 256-bit). See [`Avx512Fp16::fcmadd_pch_u16x32`].");
fp16_ternop_fixed_128!(fcmadd_pch_u16x8, fcmadd_pch128_intrinsic, core::arch::x86_64::_mm_fcmadd_pch,
	"Packed complex FMA, `b` conjugated (`vfcmaddcph`, 128-bit). See [`Avx512Fp16::fcmadd_pch_u16x32`].");

// `fmaddsub_ph`/`fmsubadd_ph` 128/256-bit: completes batch 1, which only
// shipped 512-bit (hand-written there because the alternating-lane scalar
// didn't fit `simd_ternop!`'s uniform-per-lane closure). `simd_ternop_fixed!`
// has no scalar closure at all, so these widths fit it directly: no need to
// hand-write a second time.
fp16_ternop_fixed_256!(fmaddsub_ph_u16x16, fmaddsub_ph256_intrinsic, core::arch::x86_64::_mm256_fmaddsub_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused (`vfmaddsubph`, 256-bit). See [`Avx512Fp16::fmaddsub_ph_u16x32`].");
fp16_ternop_fixed_128!(fmaddsub_ph_u16x8, fmaddsub_ph128_intrinsic, core::arch::x86_64::_mm_fmaddsub_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused (`vfmaddsubph`, 128-bit). See [`Avx512Fp16::fmaddsub_ph_u16x32`].");
fp16_ternop_fixed_256!(fmsubadd_ph_u16x16, fmsubadd_ph256_intrinsic, core::arch::x86_64::_mm256_fmsubadd_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused (`vfmsubaddph`, 256-bit). See [`Avx512Fp16::fmsubadd_ph_u16x32`].");
fp16_ternop_fixed_128!(fmsubadd_ph_u16x8, fmsubadd_ph128_intrinsic, core::arch::x86_64::_mm_fmsubadd_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused (`vfmsubaddph`, 128-bit). See [`Avx512Fp16::fmsubadd_ph_u16x32`].");

// Scalar `_sch`: lower complex pair computed, upper 3 pairs passed through
// from `a` (HW-handled, same macro as the 128-bit `_pch` forms above).
fp16_binop_fixed_128!(mul_sch_u16x8, mul_sch_intrinsic, core::arch::x86_64::_mm_mul_sch,
	"Scalar complex multiply, lane 0 only (`vfmulcsh`). Lanes 2..8 (upper 3 pairs) passed through from `a`.");
fp16_binop_fixed_128!(cmul_sch_u16x8, cmul_sch_intrinsic, core::arch::x86_64::_mm_cmul_sch,
	"Scalar complex multiply by conj(b), lane 0 only (`vfcmulcsh`). Lanes 2..8 passed through from `a`.");
fp16_ternop_fixed_128!(fmadd_sch_u16x8, fmadd_sch_intrinsic, core::arch::x86_64::_mm_fmadd_sch,
	"Scalar complex FMA, lane 0 only (`vfmaddcsh`). Lanes 2..8 passed through from `a`.");
fp16_ternop_fixed_128!(fcmadd_sch_u16x8, fcmadd_sch_intrinsic, core::arch::x86_64::_mm_fcmadd_sch,
	"Scalar complex FMA, `b` conjugated, lane 0 only (`vfcmaddcsh`). Lanes 2..8 passed through from `a`.");

// Scalar `_sh`: only wraps intrinsics that exist without embedded rounding,
// `add_sh`/`sub_sh`/`mul_sh`/`div_sh` don't exist as plain scalar ops (SDM
// only defines them as `_round`-suffixed embedded-rounding forms). Every op
// here is `(a, b) -> c`: lane 0 = `op(a[0], b[0])` (or `op(b[0])` for the
// nominally-unary ones
use core::arch::x86_64::{
	_MM_MANTISSA_NORM_ENUM, _MM_MANTISSA_SIGN_ENUM, _mm_cmp_sh_mask, _mm_getmant_sh, _mm_mask_cmp_sh_mask,
	_mm_mask_getmant_sh, _mm_mask_reduce_sh,
	_mm_mask_roundscale_sh, _mm_maskz_getmant_sh, _mm_maskz_reduce_sh,
	_mm_maskz_roundscale_sh, _mm_reduce_sh,
	_mm_roundscale_sh,
};

fp16_binop_fixed_128!(rcp_sh_u16x8, rcp_sh_intrinsic, core::arch::x86_64::_mm_rcp_sh,
	"Scalar approximate reciprocal of `b[0]` (`vrcpsh`), lanes 1..8 passed through from `a`. Max relative error < 1.5*2^-12.");
fp16_binop_fixed_128!(rsqrt_sh_u16x8, rsqrt_sh_intrinsic, core::arch::x86_64::_mm_rsqrt_sh,
	"Scalar approximate reciprocal sqrt of `b[0]` (`vrsqrtsh`), lanes 1..8 passed through from `a`. Max relative error < 1.5*2^-12.");
fp16_binop_fixed_128!(sqrt_sh_u16x8, sqrt_sh_intrinsic, core::arch::x86_64::_mm_sqrt_sh,
	"Scalar correctly-rounded sqrt of `b[0]` (`vsqrtsh`), lanes 1..8 passed through from `a`.");
fp16_binop_fixed_128!(min_sh_u16x8, min_sh_intrinsic, core::arch::x86_64::_mm_min_sh,
	"Scalar min of `a[0]`/`b[0]` (`vminsh`), lanes 1..8 passed through from `a`. NaN: second-operand-on-NaN, not IEEE `f32::min`.");
fp16_binop_fixed_128!(max_sh_u16x8, max_sh_intrinsic, core::arch::x86_64::_mm_max_sh,
	"Scalar max of `a[0]`/`b[0]` (`vmaxsh`), lanes 1..8 passed through from `a`. NaN: second-operand-on-NaN, not IEEE `f32::max`.");
fp16_binop_fixed_128!(getexp_sh_u16x8, getexp_sh_intrinsic, core::arch::x86_64::_mm_getexp_sh,
	"Scalar unbiased exponent of `b[0]` as a float (`vgetexpsh`), lanes 1..8 passed through from `a`.");
fp16_binop_fixed_128!(scalef_sh_u16x8, scalef_sh_intrinsic, core::arch::x86_64::_mm_scalef_sh,
	"Scalar `a[0] * 2^floor(b[0])` (`vscalefsh`), lanes 1..8 passed through from `a`.");

simd_binop_imm_fixed! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = reduce_sh_u16x8, intrinsic_fn = reduce_sh_intrinsic,
	width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
	intrinsic = _mm_reduce_sh,
	fixed_doc = "Scalar argument-reduction of `b[0]` by `IMM8` (`vreducesh`), lanes 1..8 passed through from `a`.",
}
simd_binop_imm_fixed! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = roundscale_sh_u16x8, intrinsic_fn = roundscale_sh_intrinsic,
	width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
	intrinsic = _mm_roundscale_sh,
	fixed_doc = "Scalar round-and-scale of `b[0]` by `IMM8` (`vrndscalesh`), lanes 1..8 passed through from `a`.",
}

impl Avx512Fp16Vl {
	/// Scalar mantissa normalization of `b[0]` (`vgetmantsh`), lanes 1..8
	/// passed through from `a`. `NORM`/`SIGN`: `_MM_MANT_NORM_*`/`_MM_MANT_SIGN_*`.
	#[inline]
	pub fn getmant_sh_u16x8<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
		self, a: [u16; 8], b: [u16; 8],
	) -> [u16; 8] {
		unsafe { getmant_sh_intrinsic::<NORM, SIGN>(&a, &b) }
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn getmant_sh_intrinsic<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
	a: &[u16; 8], b: &[u16; 8],
) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vb = loadu_ph128(b.as_ptr());
		let vr = _mm_getmant_sh::<NORM, SIGN>(va, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

macro_rules! fp16_cmp_sh_mask {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 8], b: [u16; 8]) -> u8 {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 8], b: &[u16; 8]) -> u8 {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vb = loadu_ph128(b.as_ptr());
				_mm_cmp_sh_mask::<{ $pred }>(va, vb)
			}
		}
	};
}

fp16_cmp_sh_mask!(cmpeq_sh_mask_u16x8, cmpeq_sh_mask_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"Scalar equality bit-mask of `a[0]`/`b[0]` (`vcmpsh` EQ_OQ). Bit 0 set iff `a[0]==b[0]`, bits 1..8 clear.");
fp16_cmp_sh_mask!(cmplt_sh_mask_u16x8, cmplt_sh_mask_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"Scalar less-than bit-mask of `a[0]`/`b[0]` (`vcmpsh` LT_OQ). Bit 0 set iff `a[0]<b[0]`, bits 1..8 clear.");
fp16_cmp_sh_mask!(cmple_sh_mask_u16x8, cmple_sh_mask_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"Scalar less-or-equal bit-mask of `a[0]`/`b[0]` (`vcmpsh` LE_OQ). Bit 0 set iff `a[0]<=b[0]`, bits 1..8 clear.");
fp16_cmp_sh_mask!(cmpgt_sh_mask_u16x8, cmpgt_sh_mask_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"Scalar greater-than bit-mask of `a[0]`/`b[0]` (`vcmpsh` GT_OQ). Bit 0 set iff `a[0]>b[0]`, bits 1..8 clear.");
fp16_cmp_sh_mask!(cmpge_sh_mask_u16x8, cmpge_sh_mask_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"Scalar greater-or-equal bit-mask of `a[0]`/`b[0]` (`vcmpsh` GE_OQ). Bit 0 set iff `a[0]>=b[0]`, bits 1..8 clear.");

// `comi_sh`/`ucomi_sh`: scalar predicate compares returning an `i32` boolean
// (0/1), not a mask bit: a different result shape than `cmp_sh_mask` above,
// so no shared macro. `comi` raises `#I` on QNaN, `ucomi` does not (same
// distinction as SSE's `comiss`/`ucomiss`, which this crate has no existing
// wrapper for to reuse).
macro_rules! fp16_comi_sh {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 8], b: [u16; 8]) -> i32 {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 8], b: &[u16; 8]) -> i32 {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vb = loadu_ph128(b.as_ptr());
				$intrinsic(va, vb)
			}
		}
	};
}

fp16_comi_sh!(comieq_sh, comieq_sh_intrinsic, core::arch::x86_64::_mm_comieq_sh,
	"Scalar equality of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(comige_sh, comige_sh_intrinsic, core::arch::x86_64::_mm_comige_sh,
	"Scalar greater-or-equal of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(comigt_sh, comigt_sh_intrinsic, core::arch::x86_64::_mm_comigt_sh,
	"Scalar greater-than of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(comile_sh, comile_sh_intrinsic, core::arch::x86_64::_mm_comile_sh,
	"Scalar less-or-equal of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(comilt_sh, comilt_sh_intrinsic, core::arch::x86_64::_mm_comilt_sh,
	"Scalar less-than of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(comineq_sh, comineq_sh_intrinsic, core::arch::x86_64::_mm_comineq_sh,
	"Scalar inequality of `a[0]`/`b[0]`, `1`/`0` boolean, raises `#I` on QNaN (`vcomish`).");
fp16_comi_sh!(ucomieq_sh, ucomieq_sh_intrinsic, core::arch::x86_64::_mm_ucomieq_sh,
	"Scalar equality of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");
fp16_comi_sh!(ucomige_sh, ucomige_sh_intrinsic, core::arch::x86_64::_mm_ucomige_sh,
	"Scalar greater-or-equal of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");
fp16_comi_sh!(ucomigt_sh, ucomigt_sh_intrinsic, core::arch::x86_64::_mm_ucomigt_sh,
	"Scalar greater-than of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");
fp16_comi_sh!(ucomile_sh, ucomile_sh_intrinsic, core::arch::x86_64::_mm_ucomile_sh,
	"Scalar less-or-equal of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");
fp16_comi_sh!(ucomilt_sh, ucomilt_sh_intrinsic, core::arch::x86_64::_mm_ucomilt_sh,
	"Scalar less-than of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");
fp16_comi_sh!(ucomineq_sh, ucomineq_sh_intrinsic, core::arch::x86_64::_mm_ucomineq_sh,
	"Scalar inequality of `a[0]`/`b[0]`, `1`/`0` boolean, quiet on QNaN (`vucomish`).");

// Scalar cross-type bridges: int/f32/f64 <-> FP16, lane 0 only. Heterogeneous
// types (not all `u16x8`) don't fit any generic macro here, same reasoning
// as `getmant_sh` above. `cvti32_sh`/`cvtu32_sh`/`cvtss_sh`/`cvtsd_sh` write
// into an FP16 register (lanes 1..8 passed through from `a`, standard scalar
// shape); `cvtsh_i32`/`cvtsh_u32`/`cvtsh_ss`/`cvtsh_sd` read lane 0 back out.
use core::arch::x86_64::{
	_mm_cvtsd_sh, _mm_cvtsh_i32, _mm_cvtsh_sd, _mm_cvtsh_ss, _mm_cvtsh_u32, _mm_cvti32_sh, _mm_cvtss_sh, _mm_cvtu32_sh,
	_mm_loadu_pd, _mm_loadu_ps, _mm_storeu_pd, _mm_storeu_ps,
};

impl Avx512Fp16Vl {
	/// Lane 0 = signed `i32` `b` converted to FP16, lanes 1..8 passed through from `a` (`vcvtsi2sh`).
	#[inline]
	pub fn cvti32_sh(self, a: [u16; 8], b: i32) -> [u16; 8] {
		unsafe { cvti32_sh_intrinsic(&a, b) }
	}

	/// Lane 0 = unsigned `u32` `b` converted to FP16, lanes 1..8 passed through from `a` (`vcvtusi2sh`).
	#[inline]
	pub fn cvtu32_sh(self, a: [u16; 8], b: u32) -> [u16; 8] {
		unsafe { cvtu32_sh_intrinsic(&a, b) }
	}

	/// Lane 0 = `f32` `b[0]` converted to FP16, lanes 1..8 passed through from `a` (`vcvtss2sh`).
	#[inline]
	pub fn cvtss_sh(self, a: [u16; 8], b: [f32; 4]) -> [u16; 8] {
		unsafe { cvtss_sh_intrinsic(&a, &b) }
	}

	/// Lane 0 = `f64` `b[0]` converted to FP16, lanes 1..8 passed through from `a` (`vcvtsd2sh`).
	#[inline]
	pub fn cvtsd_sh(self, a: [u16; 8], b: [f64; 2]) -> [u16; 8] {
		unsafe { cvtsd_sh_intrinsic(&a, &b) }
	}

	/// FP16 `a[0]` converted to `i32`, round-to-nearest-even (`vcvtsh2si`).
	#[inline]
	pub fn cvtsh_i32(self, a: [u16; 8]) -> i32 {
		unsafe { cvtsh_i32_intrinsic(&a) }
	}

	/// FP16 `a[0]` converted to `u32`, round-to-nearest-even (`vcvtsh2usi`).
	#[inline]
	pub fn cvtsh_u32(self, a: [u16; 8]) -> u32 {
		unsafe { cvtsh_u32_intrinsic(&a) }
	}

	/// Lane 0 = FP16 `b[0]` converted to `f32`, lanes 1..4 passed through from `a` (`vcvtsh2ss`).
	#[inline]
	pub fn cvtsh_ss(self, a: [f32; 4], b: [u16; 8]) -> [f32; 4] {
		unsafe { cvtsh_ss_intrinsic(&a, &b) }
	}

	/// Lane 0 = FP16 `b[0]` converted to `f64`, lane 1 passed through from `a` (`vcvtsh2sd`).
	#[inline]
	pub fn cvtsh_sd(self, a: [f64; 2], b: [u16; 8]) -> [f64; 2] {
		unsafe { cvtsh_sd_intrinsic(&a, &b) }
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvti32_sh_intrinsic(a: &[u16; 8], b: i32) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vr = _mm_cvti32_sh(va, b);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtu32_sh_intrinsic(a: &[u16; 8], b: u32) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vr = _mm_cvtu32_sh(va, b);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtss_sh_intrinsic(a: &[u16; 8], b: &[f32; 4]) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vb = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_cvtss_sh(va, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtsd_sh_intrinsic(a: &[u16; 8], b: &[f64; 2]) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vb = _mm_loadu_pd(b.as_ptr());
		let vr = _mm_cvtsd_sh(va, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtsh_i32_intrinsic(a: &[u16; 8]) -> i32 {
	unsafe { _mm_cvtsh_i32(loadu_ph128(a.as_ptr())) }
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtsh_u32_intrinsic(a: &[u16; 8]) -> u32 {
	unsafe { _mm_cvtsh_u32(loadu_ph128(a.as_ptr())) }
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtsh_ss_intrinsic(a: &[f32; 4], b: &[u16; 8]) -> [f32; 4] {
	unsafe {
		let va = _mm_loadu_ps(a.as_ptr());
		let vb = loadu_ph128(b.as_ptr());
		let vr = _mm_cvtsh_ss(va, vb);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn cvtsh_sd_intrinsic(a: &[f64; 2], b: &[u16; 8]) -> [f64; 2] {
	unsafe {
		let va = _mm_loadu_pd(a.as_ptr());
		let vb = loadu_ph128(b.as_ptr());
		let vr = _mm_cvtsh_sd(va, vb);
		let mut out = [0.0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

// Round-control `_round_ph`: embedded rounding is EVEX/SAE-only -> 512-bit
// only (no `_mm128_`/`_mm256_..._round_ph` exist in stdarch). Raw
// `<const ROUNDING: i32>` passthrough, same pattern as
// `super::super::sse::sse41::Sse41::round_f32x4`. Fixed-width only: rounding-
// mode-sensitive, no Rust default-rounding scalar reference.
use core::arch::x86_64::{
	_mm512_add_round_ph, _mm512_div_round_ph, _mm512_fmadd_round_ph, _mm512_fmaddsub_round_ph, _mm512_fmsub_round_ph,
	_mm512_fmsubadd_round_ph, _mm512_fnmadd_round_ph, _mm512_fnmsub_round_ph, _mm512_mul_round_ph,
	_mm512_sqrt_round_ph, _mm512_sub_round_ph,
};

macro_rules! fp16_binop_imm_fixed_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_binop_imm_fixed! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}
macro_rules! fp16_ternop_imm_fixed_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_ternop_imm! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

fp16_binop_imm_fixed_512!(add_round_ph_u16x32, add_round_ph_intrinsic, _mm512_add_round_ph,
	"`a + b` per lane with explicit rounding control (`vaddph`, 512-bit). `IMM8`: `_MM_FROUND_CUR_DIRECTION`, or `_MM_FROUND_TO_*` bitwise-OR'd with `_MM_FROUND_NO_EXC` (required for embedded rounding - `static_assert_rounding!` in stdarch rejects any other combination).");
fp16_binop_imm_fixed_512!(sub_round_ph_u16x32, sub_round_ph_intrinsic, _mm512_sub_round_ph,
	"`a - b` per lane with explicit rounding control (`vsubph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_binop_imm_fixed_512!(mul_round_ph_u16x32, mul_round_ph_intrinsic, _mm512_mul_round_ph,
	"`a * b` per lane with explicit rounding control (`vmulph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_binop_imm_fixed_512!(div_round_ph_u16x32, div_round_ph_intrinsic, _mm512_div_round_ph,
	"`a / b` per lane with explicit rounding control (`vdivph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");

simd_unop_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = sqrt_round_ph_u16x32, intrinsic_fn = sqrt_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
	intrinsic = _mm512_sqrt_round_ph,
	fixed_doc = "Per-lane sqrt with explicit rounding control (`vsqrtph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].",
}

fp16_ternop_imm_fixed_512!(fmadd_round_ph_u16x32, fmadd_round_ph_intrinsic, _mm512_fmadd_round_ph,
	"`a*b + c` per lane, HW fused, explicit rounding control (`vfmaddph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_ternop_imm_fixed_512!(fmsub_round_ph_u16x32, fmsub_round_ph_intrinsic, _mm512_fmsub_round_ph,
	"`a*b - c` per lane, HW fused, explicit rounding control (`vfmsubph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_ternop_imm_fixed_512!(fnmadd_round_ph_u16x32, fnmadd_round_ph_intrinsic, _mm512_fnmadd_round_ph,
	"`-(a*b) + c` per lane, HW fused, explicit rounding control (`vfnmaddph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_ternop_imm_fixed_512!(fnmsub_round_ph_u16x32, fnmsub_round_ph_intrinsic, _mm512_fnmsub_round_ph,
	"`-(a*b) - c` per lane, HW fused, explicit rounding control (`vfnmsubph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_ternop_imm_fixed_512!(fmaddsub_round_ph_u16x32, fmaddsub_round_ph_intrinsic, _mm512_fmaddsub_round_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused, explicit rounding control (`vfmaddsubph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");
fp16_ternop_imm_fixed_512!(fmsubadd_round_ph_u16x32, fmsubadd_round_ph_intrinsic, _mm512_fmsubadd_round_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused, explicit rounding control (`vfmsubaddph`, 512-bit). Same `IMM8` operand as [`Avx512Fp16::add_round_ph_u16x32`].");

// `getexp`/`getmant`/`reduce`/`roundscale`/`scalef` (vector): IMM8-selected or
// libm-shaped (`scalef` = `x * 2^floor(y)`, no core-available closure under
// `no_std`, same reasoning as `sqrt_ph`) -> fixed-width only, all 3 widths.
// `getexp_ph` (no imm) reuses the `fp16_fixed_unop_*` macros defined below for
// `sqrt`/`rsqrt`/`rcp` (declared ahead of their first use here since macros
// are visited in file order); `reduce_ph`/`roundscale_ph` reuse the existing
// crate-wide `simd_unop_imm!`; `scalef_ph` reuses `fp16_binop_fixed_*` from
// the packed-complex section above; `getmant_ph` (2 const params) is
// hand-written, same shape as `getmant_sh`.
use core::arch::x86_64::{
	_mm_getmant_ph, _mm_reduce_ph, _mm_roundscale_ph, _mm256_getmant_ph, _mm256_reduce_ph, _mm256_roundscale_ph,
	_mm512_getmant_ph, _mm512_reduce_ph, _mm512_roundscale_ph,
};

simd_unop_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = reduce_ph_u16x32, intrinsic_fn = reduce_ph512_intrinsic,
	width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
	intrinsic = _mm512_reduce_ph,
	fixed_doc = "Per-lane argument-reduction by `IMM8` (`vreduceph`, 512-bit). Fixed-width only, see module doc.",
}
simd_unop_imm! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = reduce_ph_u16x16, intrinsic_fn = reduce_ph256_intrinsic,
	width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
	intrinsic = _mm256_reduce_ph,
	fixed_doc = "Per-lane argument-reduction by `IMM8` (`vreduceph`, 256-bit). Fixed-width only, see module doc.",
}
simd_unop_imm! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = reduce_ph_u16x8, intrinsic_fn = reduce_ph128_intrinsic,
	width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
	intrinsic = _mm_reduce_ph,
	fixed_doc = "Per-lane argument-reduction by `IMM8` (`vreduceph`, 128-bit). Fixed-width only, see module doc.",
}

simd_unop_imm! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	fixed_fn = roundscale_ph_u16x32, intrinsic_fn = roundscale_ph512_intrinsic,
	width = 32, elem = u16, vec = __m512h, loadu = loadu_ph512, storeu = storeu_ph512,
	intrinsic = _mm512_roundscale_ph,
	fixed_doc = "Per-lane round-and-scale by `IMM8` (`vrndscaleph`, 512-bit). Fixed-width only, see module doc.",
}
simd_unop_imm! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = roundscale_ph_u16x16, intrinsic_fn = roundscale_ph256_intrinsic,
	width = 16, elem = u16, vec = __m256h, loadu = loadu_ph256, storeu = storeu_ph256,
	intrinsic = _mm256_roundscale_ph,
	fixed_doc = "Per-lane round-and-scale by `IMM8` (`vrndscaleph`, 256-bit). Fixed-width only, see module doc.",
}
simd_unop_imm! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	fixed_fn = roundscale_ph_u16x8, intrinsic_fn = roundscale_ph128_intrinsic,
	width = 8, elem = u16, vec = __m128h, loadu = loadu_ph128, storeu = storeu_ph128,
	intrinsic = _mm_roundscale_ph,
	fixed_doc = "Per-lane round-and-scale by `IMM8` (`vrndscaleph`, 128-bit). Fixed-width only, see module doc.",
}

fp16_binop_fixed_512!(scalef_ph_u16x32, scalef_ph512_intrinsic, core::arch::x86_64::_mm512_scalef_ph,
	"Per-lane `a * 2^floor(b)` (`vscalefph`, 512-bit). Fixed-width only, see module doc.");
fp16_binop_fixed_256!(scalef_ph_u16x16, scalef_ph256_intrinsic, core::arch::x86_64::_mm256_scalef_ph,
	"Per-lane `a * 2^floor(b)` (`vscalefph`, 256-bit). Fixed-width only, see module doc.");
fp16_binop_fixed_128!(scalef_ph_u16x8, scalef_ph128_intrinsic, core::arch::x86_64::_mm_scalef_ph,
	"Per-lane `a * 2^floor(b)` (`vscalefph`, 128-bit). Fixed-width only, see module doc.");

macro_rules! fp16_getmant_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $doc:literal) => {
		impl Avx512Fp16 {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, a: [u16; 32],
			) -> [u16; 32] {
				unsafe { $intrinsic_fn::<NORM, SIGN>(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			a: &[u16; 32],
		) -> [u16; 32] {
			unsafe {
				let va = loadu_ph512(a.as_ptr());
				let vr = _mm512_getmant_ph::<NORM, SIGN>(va);
				let mut out = [0u16; 32];
				storeu_ph512(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_getmant_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, a: [u16; 16],
			) -> [u16; 16] {
				unsafe { $intrinsic_fn::<NORM, SIGN>(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			a: &[u16; 16],
		) -> [u16; 16] {
			unsafe {
				let va = loadu_ph256(a.as_ptr());
				let vr = _mm256_getmant_ph::<NORM, SIGN>(va);
				let mut out = [0u16; 16];
				storeu_ph256(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_getmant_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, a: [u16; 8],
			) -> [u16; 8] {
				unsafe { $intrinsic_fn::<NORM, SIGN>(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			a: &[u16; 8],
		) -> [u16; 8] {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vr = _mm_getmant_ph::<NORM, SIGN>(va);
				let mut out = [0u16; 8];
				storeu_ph128(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}

fp16_getmant_512!(getmant_ph_u16x32, getmant_ph512_intrinsic,
	"Per-lane mantissa normalization (`vgetmantph`, 512-bit). `NORM`/`SIGN`: `_MM_MANT_NORM_*`/`_MM_MANT_SIGN_*`. Fixed-width only, see module doc.");
fp16_getmant_256!(getmant_ph_u16x16, getmant_ph256_intrinsic,
	"Per-lane mantissa normalization (`vgetmantph`, 256-bit). See [`Avx512Fp16::getmant_ph_u16x32`]. Fixed-width only, see module doc.");
fp16_getmant_128!(getmant_ph_u16x8, getmant_ph128_intrinsic,
	"Per-lane mantissa normalization (`vgetmantph`, 128-bit). See [`Avx512Fp16::getmant_ph_u16x32`]. Fixed-width only, see module doc.");

// `permutexvar_ph`/`permutex2var_ph`: pure data shuffle, `idx` is a same-width
// *integer* vector (raw lane indices, only the low bits per lane meaningful);
// separate type from the FP16 data, loaded via plain `_mm*_loadu_si*`
// (no `ph` bitcast needed).
impl Avx512Fp16 {
	/// Permute `a`'s 32 lanes by `idx` (`vpermw`+`ph`, 512-bit): `out[i] = a[idx[i] & 0x1f]`.
	#[inline]
	pub fn permutexvar_ph_u16x32(self, idx: [u16; 32], a: [u16; 32]) -> [u16; 32] {
		unsafe { permutexvar_ph512_intrinsic(&idx, &a) }
	}

	/// Permute across `a`/`b`'s 32 lanes by `idx` (`vpermi2w`+`ph`, 512-bit):
	/// `out[i] = (idx[i]&0x20 == 0 ? a : b)[idx[i] & 0x1f]`.
	#[inline]
	pub fn permutex2var_ph_u16x32(self, a: [u16; 32], idx: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		unsafe { permutex2var_ph512_intrinsic(&a, &idx, &b) }
	}
}
impl Avx512Fp16Vl {
	/// Permute `a`'s 16 lanes by `idx` (`vpermw`+`ph`, 256-bit): `out[i] = a[idx[i] & 0xf]`.
	#[inline]
	pub fn permutexvar_ph_u16x16(self, idx: [u16; 16], a: [u16; 16]) -> [u16; 16] {
		unsafe { permutexvar_ph256_intrinsic(&idx, &a) }
	}

	/// Permute across `a`/`b`'s 16 lanes by `idx` (`vpermi2w`+`ph`, 256-bit):
	/// `out[i] = (idx[i]&0x10 == 0 ? a : b)[idx[i] & 0xf]`.
	#[inline]
	pub fn permutex2var_ph_u16x16(self, a: [u16; 16], idx: [u16; 16], b: [u16; 16]) -> [u16; 16] {
		unsafe { permutex2var_ph256_intrinsic(&a, &idx, &b) }
	}

	/// Permute `a`'s 8 lanes by `idx` (`vpermw`+`ph`, 128-bit): `out[i] = a[idx[i] & 0x7]`.
	#[inline]
	pub fn permutexvar_ph_u16x8(self, idx: [u16; 8], a: [u16; 8]) -> [u16; 8] {
		unsafe { permutexvar_ph128_intrinsic(&idx, &a) }
	}

	/// Permute across `a`/`b`'s 8 lanes by `idx` (`vpermi2w`+`ph`, 128-bit):
	/// `out[i] = (idx[i]&0x8 == 0 ? a : b)[idx[i] & 0x7]`.
	#[inline]
	pub fn permutex2var_ph_u16x8(self, a: [u16; 8], idx: [u16; 8], b: [u16; 8]) -> [u16; 8] {
		unsafe { permutex2var_ph128_intrinsic(&a, &idx, &b) }
	}
}

/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn permutexvar_ph512_intrinsic(idx: &[u16; 32], a: &[u16; 32]) -> [u16; 32] {
	unsafe {
		let vidx = core::arch::x86_64::_mm512_loadu_si512(idx.as_ptr().cast());
		let va = loadu_ph512(a.as_ptr());
		let vr = core::arch::x86_64::_mm512_permutexvar_ph(vidx, va);
		let mut out = [0u16; 32];
		storeu_ph512(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16 via the token.
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn permutex2var_ph512_intrinsic(a: &[u16; 32], idx: &[u16; 32], b: &[u16; 32]) -> [u16; 32] {
	unsafe {
		let va = loadu_ph512(a.as_ptr());
		let vidx = core::arch::x86_64::_mm512_loadu_si512(idx.as_ptr().cast());
		let vb = loadu_ph512(b.as_ptr());
		let vr = core::arch::x86_64::_mm512_permutex2var_ph(va, vidx, vb);
		let mut out = [0u16; 32];
		storeu_ph512(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn permutexvar_ph256_intrinsic(idx: &[u16; 16], a: &[u16; 16]) -> [u16; 16] {
	unsafe {
		let vidx = core::arch::x86_64::_mm256_loadu_si256(idx.as_ptr().cast());
		let va = loadu_ph256(a.as_ptr());
		let vr = core::arch::x86_64::_mm256_permutexvar_ph(vidx, va);
		let mut out = [0u16; 16];
		storeu_ph256(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn permutex2var_ph256_intrinsic(a: &[u16; 16], idx: &[u16; 16], b: &[u16; 16]) -> [u16; 16] {
	unsafe {
		let va = loadu_ph256(a.as_ptr());
		let vidx = core::arch::x86_64::_mm256_loadu_si256(idx.as_ptr().cast());
		let vb = loadu_ph256(b.as_ptr());
		let vr = core::arch::x86_64::_mm256_permutex2var_ph(va, vidx, vb);
		let mut out = [0u16; 16];
		storeu_ph256(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn permutexvar_ph128_intrinsic(idx: &[u16; 8], a: &[u16; 8]) -> [u16; 8] {
	unsafe {
		let vidx = _mm_loadu_si128(idx.as_ptr().cast());
		let va = loadu_ph128(a.as_ptr());
		let vr = core::arch::x86_64::_mm_permutexvar_ph(vidx, va);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn permutex2var_ph128_intrinsic(a: &[u16; 8], idx: &[u16; 8], b: &[u16; 8]) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vidx = _mm_loadu_si128(idx.as_ptr().cast());
		let vb = loadu_ph128(b.as_ptr());
		let vr = core::arch::x86_64::_mm_permutex2var_ph(va, vidx, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

// Fixed-width-only: `sqrt`/`rsqrt`/`rcp` need no libm for the HW op itself
// (real `vsqrtph`/`vrsqrtph`/`vrcpph`), but `simd_unop!`'s `_slice` wrapper
// always bundles a scalar-remainder closure, which *would* need libm. No
// `_slice`/`auto` form; callers needing arbitrary-length slices must chunk
// by hand (`chunks_exact(32)` etc.) and handle the remainder themselves.
macro_rules! fp16_fixed_unop_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		impl Avx512Fp16 {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 32]) -> [u16; 32] {
				unsafe { $intrinsic_fn(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $intrinsic_fn(a: &[u16; 32]) -> [u16; 32] {
			unsafe {
				let va = loadu_ph512(a.as_ptr());
				let vr = $intrinsic(va);
				let mut out = [0u16; 32];
				storeu_ph512(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_fixed_unop_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 16]) -> [u16; 16] {
				unsafe { $intrinsic_fn(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 16]) -> [u16; 16] {
			unsafe {
				let va = loadu_ph256(a.as_ptr());
				let vr = $intrinsic(va);
				let mut out = [0u16; 16];
				storeu_ph256(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_fixed_unop_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u16; 8]) -> [u16; 8] {
				unsafe { $intrinsic_fn(&a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(a: &[u16; 8]) -> [u16; 8] {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vr = $intrinsic(va);
				let mut out = [0u16; 8];
				storeu_ph128(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}

fp16_fixed_unop_512!(sqrt_ph_u16x32, sqrt_ph512_intrinsic, _mm512_sqrt_ph,
	"Correctly-rounded per-lane sqrt (`vsqrtph`, 512-bit). Fixed-width only, see module doc.");
fp16_fixed_unop_256!(sqrt_ph_u16x16, sqrt_ph256_intrinsic, _mm256_sqrt_ph,
	"Correctly-rounded per-lane sqrt (`vsqrtph`, 256-bit). Fixed-width only, see module doc.");
fp16_fixed_unop_128!(sqrt_ph_u16x8, sqrt_ph128_intrinsic, _mm_sqrt_ph,
	"Correctly-rounded per-lane sqrt (`vsqrtph`, 128-bit). Fixed-width only, see module doc.");

fp16_fixed_unop_512!(rsqrt_ph_u16x32, rsqrt_ph512_intrinsic, _mm512_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt (`vrsqrtph`, 512-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");
fp16_fixed_unop_256!(rsqrt_ph_u16x16, rsqrt_ph256_intrinsic, _mm256_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt (`vrsqrtph`, 256-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");
fp16_fixed_unop_128!(rsqrt_ph_u16x8, rsqrt_ph128_intrinsic, _mm_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt (`vrsqrtph`, 128-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");

fp16_fixed_unop_512!(rcp_ph_u16x32, rcp_ph512_intrinsic, _mm512_rcp_ph,
	"Approximate per-lane reciprocal (`vrcpph`, 512-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");
fp16_fixed_unop_256!(rcp_ph_u16x16, rcp_ph256_intrinsic, _mm256_rcp_ph,
	"Approximate per-lane reciprocal (`vrcpph`, 256-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");
fp16_fixed_unop_128!(rcp_ph_u16x8, rcp_ph128_intrinsic, _mm_rcp_ph,
	"Approximate per-lane reciprocal (`vrcpph`, 128-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.");

fp16_fixed_unop_512!(getexp_ph_u16x32, getexp_ph512_intrinsic, core::arch::x86_64::_mm512_getexp_ph,
	"Per-lane unbiased exponent as a float (`vgetexpph`, 512-bit). Fixed-width only, see module doc.");
fp16_fixed_unop_256!(getexp_ph_u16x16, getexp_ph256_intrinsic, core::arch::x86_64::_mm256_getexp_ph,
	"Per-lane unbiased exponent as a float (`vgetexpph`, 256-bit). Fixed-width only, see module doc.");
fp16_fixed_unop_128!(getexp_ph_u16x8, getexp_ph128_intrinsic, core::arch::x86_64::_mm_getexp_ph,
	"Per-lane unbiased exponent as a float (`vgetexpph`, 128-bit). Fixed-width only, see module doc.");

// Merge/zero-masked forms of the arithmetic families above. `abs_ph` has no
// masked sibling: the ISA implements it as a bitwise clear, not an EVEX
// arithmetic op, so there is nothing to mask.
macro_rules! fp16_binop_masked_512 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, mask = u32,
			loadu = loadu_ph512, storeu = storeu_ph512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_binop_masked_256 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, mask = u16,
			loadu = loadu_ph256, storeu = storeu_ph256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_binop_masked_128 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, mask = u8,
			loadu = loadu_ph128, storeu = storeu_ph128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! fp16_unop_masked_512 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_unop_masked! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, mask = u32,
			loadu = loadu_ph512, storeu = storeu_ph512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_unop_masked_256 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_unop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, mask = u16,
			loadu = loadu_ph256, storeu = storeu_ph256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_unop_masked_128 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_unop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, mask = u8,
			loadu = loadu_ph128, storeu = storeu_ph128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! fp16_ternop_masked_512 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512Fp16, target_feature = "avx512fp16",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = u16, vec = __m512h, mask = u32,
			loadu = loadu_ph512, storeu = storeu_ph512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_ternop_masked_256 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u16, vec = __m256h, mask = u16,
			loadu = loadu_ph256, storeu = storeu_ph256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}
macro_rules! fp16_ternop_masked_128 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u16, vec = __m128h, mask = u8,
			loadu = loadu_ph128, storeu = storeu_ph128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

fp16_binop_masked_512!(
	add_ph_u16x32_merge_masked, add_ph_u16x32_zero_masked,
	mask_add_ph512_intrinsic, maskz_add_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_add_ph, core::arch::x86_64::_mm512_maskz_add_ph,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddph`, 512-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	add_ph_u16x16_merge_masked, add_ph_u16x16_zero_masked,
	mask_add_ph256_intrinsic, maskz_add_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_add_ph, core::arch::x86_64::_mm256_maskz_add_ph,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddph`, 256-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	add_ph_u16x8_merge_masked, add_ph_u16x8_zero_masked,
	mask_add_ph128_intrinsic, maskz_add_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_add_ph, core::arch::x86_64::_mm_maskz_add_ph,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddph`, 128-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	sub_ph_u16x32_merge_masked, sub_ph_u16x32_zero_masked,
	mask_sub_ph512_intrinsic, maskz_sub_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_sub_ph, core::arch::x86_64::_mm512_maskz_sub_ph,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubph`, 512-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	sub_ph_u16x16_merge_masked, sub_ph_u16x16_zero_masked,
	mask_sub_ph256_intrinsic, maskz_sub_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_sub_ph, core::arch::x86_64::_mm256_maskz_sub_ph,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubph`, 256-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	sub_ph_u16x8_merge_masked, sub_ph_u16x8_zero_masked,
	mask_sub_ph128_intrinsic, maskz_sub_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_sub_ph, core::arch::x86_64::_mm_maskz_sub_ph,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubph`, 128-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	mul_ph_u16x32_merge_masked, mul_ph_u16x32_zero_masked,
	mask_mul_ph512_intrinsic, maskz_mul_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_mul_ph, core::arch::x86_64::_mm512_maskz_mul_ph,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulph`, 512-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	mul_ph_u16x16_merge_masked, mul_ph_u16x16_zero_masked,
	mask_mul_ph256_intrinsic, maskz_mul_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_mul_ph, core::arch::x86_64::_mm256_maskz_mul_ph,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulph`, 256-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	mul_ph_u16x8_merge_masked, mul_ph_u16x8_zero_masked,
	mask_mul_ph128_intrinsic, maskz_mul_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_mul_ph, core::arch::x86_64::_mm_maskz_mul_ph,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulph`, 128-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	div_ph_u16x32_merge_masked, div_ph_u16x32_zero_masked,
	mask_div_ph512_intrinsic, maskz_div_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_div_ph, core::arch::x86_64::_mm512_maskz_div_ph,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivph`, 512-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	div_ph_u16x16_merge_masked, div_ph_u16x16_zero_masked,
	mask_div_ph256_intrinsic, maskz_div_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_div_ph, core::arch::x86_64::_mm256_maskz_div_ph,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivph`, 256-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	div_ph_u16x8_merge_masked, div_ph_u16x8_zero_masked,
	mask_div_ph128_intrinsic, maskz_div_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_div_ph, core::arch::x86_64::_mm_maskz_div_ph,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivph`, 128-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	min_ph_u16x32_merge_masked, min_ph_u16x32_zero_masked,
	mask_min_ph512_intrinsic, maskz_min_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_min_ph, core::arch::x86_64::_mm512_maskz_min_ph,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminph`, 512-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	min_ph_u16x16_merge_masked, min_ph_u16x16_zero_masked,
	mask_min_ph256_intrinsic, maskz_min_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_min_ph, core::arch::x86_64::_mm256_maskz_min_ph,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminph`, 256-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	min_ph_u16x8_merge_masked, min_ph_u16x8_zero_masked,
	mask_min_ph128_intrinsic, maskz_min_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_min_ph, core::arch::x86_64::_mm_maskz_min_ph,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminph`, 128-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	max_ph_u16x32_merge_masked, max_ph_u16x32_zero_masked,
	mask_max_ph512_intrinsic, maskz_max_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_max_ph, core::arch::x86_64::_mm512_maskz_max_ph,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxph`, 512-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	max_ph_u16x16_merge_masked, max_ph_u16x16_zero_masked,
	mask_max_ph256_intrinsic, maskz_max_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_max_ph, core::arch::x86_64::_mm256_maskz_max_ph,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxph`, 256-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	max_ph_u16x8_merge_masked, max_ph_u16x8_zero_masked,
	mask_max_ph128_intrinsic, maskz_max_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_max_ph, core::arch::x86_64::_mm_maskz_max_ph,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxph`, 128-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxph`, 128-bit, zero-masked)."
);

fp16_unop_masked_512!(
	sqrt_ph_u16x32_merge_masked, sqrt_ph_u16x32_zero_masked,
	mask_sqrt_ph512_intrinsic, maskz_sqrt_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_sqrt_ph, core::arch::x86_64::_mm512_maskz_sqrt_ph,
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else copied from `src` (`vsqrtph`, 512-bit, merge-masked).",
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else zero (`vsqrtph`, 512-bit, zero-masked)."
);
fp16_unop_masked_256!(
	sqrt_ph_u16x16_merge_masked, sqrt_ph_u16x16_zero_masked,
	mask_sqrt_ph256_intrinsic, maskz_sqrt_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_sqrt_ph, core::arch::x86_64::_mm256_maskz_sqrt_ph,
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else copied from `src` (`vsqrtph`, 256-bit, merge-masked).",
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else zero (`vsqrtph`, 256-bit, zero-masked)."
);
fp16_unop_masked_128!(
	sqrt_ph_u16x8_merge_masked, sqrt_ph_u16x8_zero_masked,
	mask_sqrt_ph128_intrinsic, maskz_sqrt_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_sqrt_ph, core::arch::x86_64::_mm_maskz_sqrt_ph,
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else copied from `src` (`vsqrtph`, 128-bit, merge-masked).",
	"Correctly-rounded per-lane sqrt where `mask` bit is set, else zero (`vsqrtph`, 128-bit, zero-masked)."
);

fp16_unop_masked_512!(
	rsqrt_ph_u16x32_merge_masked, rsqrt_ph_u16x32_zero_masked,
	mask_rsqrt_ph512_intrinsic, maskz_rsqrt_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_rsqrt_ph, core::arch::x86_64::_mm512_maskz_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else copied from `src` (`vrsqrtph`, 512-bit, merge-masked).",
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else zero (`vrsqrtph`, 512-bit, zero-masked)."
);
fp16_unop_masked_256!(
	rsqrt_ph_u16x16_merge_masked, rsqrt_ph_u16x16_zero_masked,
	mask_rsqrt_ph256_intrinsic, maskz_rsqrt_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_rsqrt_ph, core::arch::x86_64::_mm256_maskz_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else copied from `src` (`vrsqrtph`, 256-bit, merge-masked).",
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else zero (`vrsqrtph`, 256-bit, zero-masked)."
);
fp16_unop_masked_128!(
	rsqrt_ph_u16x8_merge_masked, rsqrt_ph_u16x8_zero_masked,
	mask_rsqrt_ph128_intrinsic, maskz_rsqrt_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_rsqrt_ph, core::arch::x86_64::_mm_maskz_rsqrt_ph,
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else copied from `src` (`vrsqrtph`, 128-bit, merge-masked).",
	"Approximate per-lane reciprocal sqrt where `mask` bit is set, else zero (`vrsqrtph`, 128-bit, zero-masked)."
);

fp16_unop_masked_512!(
	rcp_ph_u16x32_merge_masked, rcp_ph_u16x32_zero_masked,
	mask_rcp_ph512_intrinsic, maskz_rcp_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_rcp_ph, core::arch::x86_64::_mm512_maskz_rcp_ph,
	"Approximate per-lane reciprocal where `mask` bit is set, else copied from `src` (`vrcpph`, 512-bit, merge-masked).",
	"Approximate per-lane reciprocal where `mask` bit is set, else zero (`vrcpph`, 512-bit, zero-masked)."
);
fp16_unop_masked_256!(
	rcp_ph_u16x16_merge_masked, rcp_ph_u16x16_zero_masked,
	mask_rcp_ph256_intrinsic, maskz_rcp_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_rcp_ph, core::arch::x86_64::_mm256_maskz_rcp_ph,
	"Approximate per-lane reciprocal where `mask` bit is set, else copied from `src` (`vrcpph`, 256-bit, merge-masked).",
	"Approximate per-lane reciprocal where `mask` bit is set, else zero (`vrcpph`, 256-bit, zero-masked)."
);
fp16_unop_masked_128!(
	rcp_ph_u16x8_merge_masked, rcp_ph_u16x8_zero_masked,
	mask_rcp_ph128_intrinsic, maskz_rcp_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_rcp_ph, core::arch::x86_64::_mm_maskz_rcp_ph,
	"Approximate per-lane reciprocal where `mask` bit is set, else copied from `src` (`vrcpph`, 128-bit, merge-masked).",
	"Approximate per-lane reciprocal where `mask` bit is set, else zero (`vrcpph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fmadd_ph_u16x32_merge_masked, fmadd_ph_u16x32_zero_masked,
	mask_fmadd_ph512_intrinsic, maskz_fmadd_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fmadd_ph, core::arch::x86_64::_mm512_maskz_fmadd_ph,
	"`a*b + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmaddph`, 512-bit, merge-masked).",
	"`a*b + c` per lane, HW fused where `mask` bit is set, else zero (`vfmaddph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fmadd_ph_u16x16_merge_masked, fmadd_ph_u16x16_zero_masked,
	mask_fmadd_ph256_intrinsic, maskz_fmadd_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fmadd_ph, core::arch::x86_64::_mm256_maskz_fmadd_ph,
	"`a*b + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmaddph`, 256-bit, merge-masked).",
	"`a*b + c` per lane, HW fused where `mask` bit is set, else zero (`vfmaddph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fmadd_ph_u16x8_merge_masked, fmadd_ph_u16x8_zero_masked,
	mask_fmadd_ph128_intrinsic, maskz_fmadd_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fmadd_ph, core::arch::x86_64::_mm_maskz_fmadd_ph,
	"`a*b + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmaddph`, 128-bit, merge-masked).",
	"`a*b + c` per lane, HW fused where `mask` bit is set, else zero (`vfmaddph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fmsub_ph_u16x32_merge_masked, fmsub_ph_u16x32_zero_masked,
	mask_fmsub_ph512_intrinsic, maskz_fmsub_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fmsub_ph, core::arch::x86_64::_mm512_maskz_fmsub_ph,
	"`a*b - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmsubph`, 512-bit, merge-masked).",
	"`a*b - c` per lane, HW fused where `mask` bit is set, else zero (`vfmsubph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fmsub_ph_u16x16_merge_masked, fmsub_ph_u16x16_zero_masked,
	mask_fmsub_ph256_intrinsic, maskz_fmsub_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fmsub_ph, core::arch::x86_64::_mm256_maskz_fmsub_ph,
	"`a*b - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmsubph`, 256-bit, merge-masked).",
	"`a*b - c` per lane, HW fused where `mask` bit is set, else zero (`vfmsubph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fmsub_ph_u16x8_merge_masked, fmsub_ph_u16x8_zero_masked,
	mask_fmsub_ph128_intrinsic, maskz_fmsub_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fmsub_ph, core::arch::x86_64::_mm_maskz_fmsub_ph,
	"`a*b - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfmsubph`, 128-bit, merge-masked).",
	"`a*b - c` per lane, HW fused where `mask` bit is set, else zero (`vfmsubph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fnmadd_ph_u16x32_merge_masked, fnmadd_ph_u16x32_zero_masked,
	mask_fnmadd_ph512_intrinsic, maskz_fnmadd_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fnmadd_ph, core::arch::x86_64::_mm512_maskz_fnmadd_ph,
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmaddph`, 512-bit, merge-masked).",
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else zero (`vfnmaddph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fnmadd_ph_u16x16_merge_masked, fnmadd_ph_u16x16_zero_masked,
	mask_fnmadd_ph256_intrinsic, maskz_fnmadd_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fnmadd_ph, core::arch::x86_64::_mm256_maskz_fnmadd_ph,
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmaddph`, 256-bit, merge-masked).",
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else zero (`vfnmaddph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fnmadd_ph_u16x8_merge_masked, fnmadd_ph_u16x8_zero_masked,
	mask_fnmadd_ph128_intrinsic, maskz_fnmadd_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fnmadd_ph, core::arch::x86_64::_mm_maskz_fnmadd_ph,
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmaddph`, 128-bit, merge-masked).",
	"`-(a*b) + c` per lane, HW fused where `mask` bit is set, else zero (`vfnmaddph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fnmsub_ph_u16x32_merge_masked, fnmsub_ph_u16x32_zero_masked,
	mask_fnmsub_ph512_intrinsic, maskz_fnmsub_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fnmsub_ph, core::arch::x86_64::_mm512_maskz_fnmsub_ph,
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmsubph`, 512-bit, merge-masked).",
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else zero (`vfnmsubph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fnmsub_ph_u16x16_merge_masked, fnmsub_ph_u16x16_zero_masked,
	mask_fnmsub_ph256_intrinsic, maskz_fnmsub_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fnmsub_ph, core::arch::x86_64::_mm256_maskz_fnmsub_ph,
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmsubph`, 256-bit, merge-masked).",
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else zero (`vfnmsubph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fnmsub_ph_u16x8_merge_masked, fnmsub_ph_u16x8_zero_masked,
	mask_fnmsub_ph128_intrinsic, maskz_fnmsub_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fnmsub_ph, core::arch::x86_64::_mm_maskz_fnmsub_ph,
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else copied from `a` (`vfnmsubph`, 128-bit, merge-masked).",
	"`-(a*b) - c` per lane, HW fused where `mask` bit is set, else zero (`vfnmsubph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fmaddsub_ph_u16x32_merge_masked, fmaddsub_ph_u16x32_zero_masked,
	mask_fmaddsub_ph512_intrinsic, maskz_fmaddsub_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fmaddsub_ph, core::arch::x86_64::_mm512_maskz_fmaddsub_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else copied from `a` (`vfmaddsubph`, 512-bit, merge-masked).",
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else zero (`vfmaddsubph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fmaddsub_ph_u16x16_merge_masked, fmaddsub_ph_u16x16_zero_masked,
	mask_fmaddsub_ph256_intrinsic, maskz_fmaddsub_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fmaddsub_ph, core::arch::x86_64::_mm256_maskz_fmaddsub_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else copied from `a` (`vfmaddsubph`, 256-bit, merge-masked).",
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else zero (`vfmaddsubph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fmaddsub_ph_u16x8_merge_masked, fmaddsub_ph_u16x8_zero_masked,
	mask_fmaddsub_ph128_intrinsic, maskz_fmaddsub_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fmaddsub_ph, core::arch::x86_64::_mm_maskz_fmaddsub_ph,
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else copied from `a` (`vfmaddsubph`, 128-bit, merge-masked).",
	"Even lanes: `a*b - c`; odd lanes: `a*b + c`, HW fused where `mask` bit is set, else zero (`vfmaddsubph`, 128-bit, zero-masked)."
);

fp16_ternop_masked_512!(
	fmsubadd_ph_u16x32_merge_masked, fmsubadd_ph_u16x32_zero_masked,
	mask_fmsubadd_ph512_intrinsic, maskz_fmsubadd_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_fmsubadd_ph, core::arch::x86_64::_mm512_maskz_fmsubadd_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else copied from `a` (`vfmsubaddph`, 512-bit, merge-masked).",
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else zero (`vfmsubaddph`, 512-bit, zero-masked)."
);
fp16_ternop_masked_256!(
	fmsubadd_ph_u16x16_merge_masked, fmsubadd_ph_u16x16_zero_masked,
	mask_fmsubadd_ph256_intrinsic, maskz_fmsubadd_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_fmsubadd_ph, core::arch::x86_64::_mm256_maskz_fmsubadd_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else copied from `a` (`vfmsubaddph`, 256-bit, merge-masked).",
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else zero (`vfmsubaddph`, 256-bit, zero-masked)."
);
fp16_ternop_masked_128!(
	fmsubadd_ph_u16x8_merge_masked, fmsubadd_ph_u16x8_zero_masked,
	mask_fmsubadd_ph128_intrinsic, maskz_fmsubadd_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_fmsubadd_ph, core::arch::x86_64::_mm_maskz_fmsubadd_ph,
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else copied from `a` (`vfmsubaddph`, 128-bit, merge-masked).",
	"Even lanes: `a*b + c`; odd lanes: `a*b - c`, HW fused where `mask` bit is set, else zero (`vfmsubaddph`, 128-bit, zero-masked)."
);

// Merge/zero-masked getexp_ph/scalef_ph (reuse the arithmetic-family local
// macros above), reduce_ph/roundscale_ph (simd_unop_imm_masked), and
// getmant_ph (hand-written, same NORM/SIGN const-generic shape as
// fp16_getmant_512/256/128 plus src/mask).
fp16_unop_masked_512!(
	getexp_ph_u16x32_merge_masked, getexp_ph_u16x32_zero_masked,
	mask_getexp_ph512_intrinsic, maskz_getexp_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_getexp_ph, core::arch::x86_64::_mm512_maskz_getexp_ph,
	"[`Avx512Fp16::getexp_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vgetexpph`, 512-bit, merge-masked).",
	"[`Avx512Fp16::getexp_ph_u16x32`] where `mask` bit is set, else zero (`vgetexpph`, 512-bit, zero-masked)."
);
fp16_unop_masked_256!(
	getexp_ph_u16x16_merge_masked, getexp_ph_u16x16_zero_masked,
	mask_getexp_ph256_intrinsic, maskz_getexp_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_getexp_ph, core::arch::x86_64::_mm256_maskz_getexp_ph,
	"[`Avx512Fp16Vl::getexp_ph_u16x16`] where `mask` bit is set, else copied from `src` (`vgetexpph`, 256-bit, merge-masked).",
	"[`Avx512Fp16Vl::getexp_ph_u16x16`] where `mask` bit is set, else zero (`vgetexpph`, 256-bit, zero-masked)."
);
fp16_unop_masked_128!(
	getexp_ph_u16x8_merge_masked, getexp_ph_u16x8_zero_masked,
	mask_getexp_ph128_intrinsic, maskz_getexp_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_getexp_ph, core::arch::x86_64::_mm_maskz_getexp_ph,
	"[`Avx512Fp16Vl::getexp_ph_u16x8`] where `mask` bit is set, else copied from `src` (`vgetexpph`, 128-bit, merge-masked).",
	"[`Avx512Fp16Vl::getexp_ph_u16x8`] where `mask` bit is set, else zero (`vgetexpph`, 128-bit, zero-masked)."
);

fp16_binop_masked_512!(
	scalef_ph_u16x32_merge_masked, scalef_ph_u16x32_zero_masked,
	mask_scalef_ph512_intrinsic, maskz_scalef_ph512_intrinsic,
	core::arch::x86_64::_mm512_mask_scalef_ph, core::arch::x86_64::_mm512_maskz_scalef_ph,
	"[`Avx512Fp16::scalef_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vscalefph`, 512-bit, merge-masked).",
	"[`Avx512Fp16::scalef_ph_u16x32`] where `mask` bit is set, else zero (`vscalefph`, 512-bit, zero-masked)."
);
fp16_binop_masked_256!(
	scalef_ph_u16x16_merge_masked, scalef_ph_u16x16_zero_masked,
	mask_scalef_ph256_intrinsic, maskz_scalef_ph256_intrinsic,
	core::arch::x86_64::_mm256_mask_scalef_ph, core::arch::x86_64::_mm256_maskz_scalef_ph,
	"[`Avx512Fp16Vl::scalef_ph_u16x16`] where `mask` bit is set, else copied from `src` (`vscalefph`, 256-bit, merge-masked).",
	"[`Avx512Fp16Vl::scalef_ph_u16x16`] where `mask` bit is set, else zero (`vscalefph`, 256-bit, zero-masked)."
);
fp16_binop_masked_128!(
	scalef_ph_u16x8_merge_masked, scalef_ph_u16x8_zero_masked,
	mask_scalef_ph128_intrinsic, maskz_scalef_ph128_intrinsic,
	core::arch::x86_64::_mm_mask_scalef_ph, core::arch::x86_64::_mm_maskz_scalef_ph,
	"[`Avx512Fp16Vl::scalef_ph_u16x8`] where `mask` bit is set, else copied from `src` (`vscalefph`, 128-bit, merge-masked).",
	"[`Avx512Fp16Vl::scalef_ph_u16x8`] where `mask` bit is set, else zero (`vscalefph`, 128-bit, zero-masked)."
);
simd_unop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = reduce_ph_u16x32_merge_masked, zero_fn = reduce_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_reduce_ph32_intrinsic, zero_intrinsic_fn = maskz_reduce_ph32_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_reduce_ph, zero_intrinsic = _mm512_maskz_reduce_ph,
	merge_doc = "[`Avx512Fp16::reduce_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vreduceph`, 512-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16::reduce_ph_u16x32`] where `mask` bit is set, else zero (`vreduceph`, 512-bit, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = reduce_ph_u16x16_merge_masked, zero_fn = reduce_ph_u16x16_zero_masked,
	merge_intrinsic_fn = mask_reduce_ph16_intrinsic, zero_intrinsic_fn = maskz_reduce_ph16_intrinsic,
	width = 16, elem = u16, vec = __m256h, mask = u16,
	loadu = loadu_ph256, storeu = storeu_ph256,
	merge_intrinsic = _mm256_mask_reduce_ph, zero_intrinsic = _mm256_maskz_reduce_ph,
	merge_doc = "[`Avx512Fp16Vl::reduce_ph_u16x16`] where `mask` bit is set, else copied from `src` (`vreduceph`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::reduce_ph_u16x16`] where `mask` bit is set, else zero (`vreduceph`, 256-bit, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = reduce_ph_u16x8_merge_masked, zero_fn = reduce_ph_u16x8_zero_masked,
	merge_intrinsic_fn = mask_reduce_ph8_intrinsic, zero_intrinsic_fn = maskz_reduce_ph8_intrinsic,
	width = 8, elem = u16, vec = __m128h, mask = u8,
	loadu = loadu_ph128, storeu = storeu_ph128,
	merge_intrinsic = _mm_mask_reduce_ph, zero_intrinsic = _mm_maskz_reduce_ph,
	merge_doc = "[`Avx512Fp16Vl::reduce_ph_u16x8`] where `mask` bit is set, else copied from `src` (`vreduceph`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::reduce_ph_u16x8`] where `mask` bit is set, else zero (`vreduceph`, 128-bit, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = roundscale_ph_u16x32_merge_masked, zero_fn = roundscale_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_roundscale_ph32_intrinsic, zero_intrinsic_fn = maskz_roundscale_ph32_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_roundscale_ph, zero_intrinsic = _mm512_maskz_roundscale_ph,
	merge_doc = "[`Avx512Fp16::roundscale_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vrndscaleph`, 512-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16::roundscale_ph_u16x32`] where `mask` bit is set, else zero (`vrndscaleph`, 512-bit, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = roundscale_ph_u16x16_merge_masked, zero_fn = roundscale_ph_u16x16_zero_masked,
	merge_intrinsic_fn = mask_roundscale_ph16_intrinsic, zero_intrinsic_fn = maskz_roundscale_ph16_intrinsic,
	width = 16, elem = u16, vec = __m256h, mask = u16,
	loadu = loadu_ph256, storeu = storeu_ph256,
	merge_intrinsic = _mm256_mask_roundscale_ph, zero_intrinsic = _mm256_maskz_roundscale_ph,
	merge_doc = "[`Avx512Fp16Vl::roundscale_ph_u16x16`] where `mask` bit is set, else copied from `src` (`vrndscaleph`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::roundscale_ph_u16x16`] where `mask` bit is set, else zero (`vrndscaleph`, 256-bit, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = roundscale_ph_u16x8_merge_masked, zero_fn = roundscale_ph_u16x8_zero_masked,
	merge_intrinsic_fn = mask_roundscale_ph8_intrinsic, zero_intrinsic_fn = maskz_roundscale_ph8_intrinsic,
	width = 8, elem = u16, vec = __m128h, mask = u8,
	loadu = loadu_ph128, storeu = storeu_ph128,
	merge_intrinsic = _mm_mask_roundscale_ph, zero_intrinsic = _mm_maskz_roundscale_ph,
	merge_doc = "[`Avx512Fp16Vl::roundscale_ph_u16x8`] where `mask` bit is set, else copied from `src` (`vrndscaleph`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::roundscale_ph_u16x8`] where `mask` bit is set, else zero (`vrndscaleph`, 128-bit, zero-masked).",
}

// getmant_ph masked: same NORM/SIGN const-generic shape as the unmasked
// fp16_getmant_512/256/128 macros above, plus src/mask: hand-written since
// no generic macro combines two const generics with merge/zero masking.
macro_rules! fp16_getmant_masked_512 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident) => {
		impl Avx512Fp16 {
			/// [`Avx512Fp16::getmant_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vgetmantph`, 512-bit, merge-masked).
			#[inline]
			pub fn $merge_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, src: [u16; 32], mask: u32, a: [u16; 32],
			) -> [u16; 32] {
				unsafe { $merge_intrinsic_fn::<NORM, SIGN>(&src, mask, &a) }
			}

			/// [`Avx512Fp16::getmant_ph_u16x32`] where `mask` bit is set, else zero (`vgetmantph`, 512-bit, zero-masked).
			#[inline]
			pub fn $zero_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, mask: u32, a: [u16; 32],
			) -> [u16; 32] {
				unsafe { $zero_intrinsic_fn::<NORM, SIGN>(mask, &a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $merge_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			src: &[u16; 32], mask: u32, a: &[u16; 32],
		) -> [u16; 32] {
			unsafe {
				let vsrc = loadu_ph512(src.as_ptr());
				let va = loadu_ph512(a.as_ptr());
				let vr = _mm512_mask_getmant_ph::<NORM, SIGN>(vsrc, mask, va);
				let mut out = [0u16; 32];
				storeu_ph512(out.as_mut_ptr(), vr);
				out
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $zero_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			mask: u32, a: &[u16; 32],
		) -> [u16; 32] {
			unsafe {
				let va = loadu_ph512(a.as_ptr());
				let vr = _mm512_maskz_getmant_ph::<NORM, SIGN>(mask, va);
				let mut out = [0u16; 32];
				storeu_ph512(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_getmant_masked_256 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident) => {
		impl Avx512Fp16Vl {
			/// [`Avx512Fp16Vl::getmant_ph_u16x16`] where `mask` bit is set, else copied from `src` (`vgetmantph`, 256-bit, merge-masked).
			#[inline]
			pub fn $merge_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, src: [u16; 16], mask: u16, a: [u16; 16],
			) -> [u16; 16] {
				unsafe { $merge_intrinsic_fn::<NORM, SIGN>(&src, mask, &a) }
			}

			/// [`Avx512Fp16Vl::getmant_ph_u16x16`] where `mask` bit is set, else zero (`vgetmantph`, 256-bit, zero-masked).
			#[inline]
			pub fn $zero_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, mask: u16, a: [u16; 16],
			) -> [u16; 16] {
				unsafe { $zero_intrinsic_fn::<NORM, SIGN>(mask, &a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $merge_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			src: &[u16; 16], mask: u16, a: &[u16; 16],
		) -> [u16; 16] {
			unsafe {
				let vsrc = loadu_ph256(src.as_ptr());
				let va = loadu_ph256(a.as_ptr());
				let vr = _mm256_mask_getmant_ph::<NORM, SIGN>(vsrc, mask, va);
				let mut out = [0u16; 16];
				storeu_ph256(out.as_mut_ptr(), vr);
				out
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $zero_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			mask: u16, a: &[u16; 16],
		) -> [u16; 16] {
			unsafe {
				let va = loadu_ph256(a.as_ptr());
				let vr = _mm256_maskz_getmant_ph::<NORM, SIGN>(mask, va);
				let mut out = [0u16; 16];
				storeu_ph256(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}
macro_rules! fp16_getmant_masked_128 {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident) => {
		impl Avx512Fp16Vl {
			/// [`Avx512Fp16Vl::getmant_ph_u16x8`] where `mask` bit is set, else copied from `src` (`vgetmantph`, 128-bit, merge-masked).
			#[inline]
			pub fn $merge_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, src: [u16; 8], mask: u8, a: [u16; 8],
			) -> [u16; 8] {
				unsafe { $merge_intrinsic_fn::<NORM, SIGN>(&src, mask, &a) }
			}

			/// [`Avx512Fp16Vl::getmant_ph_u16x8`] where `mask` bit is set, else zero (`vgetmantph`, 128-bit, zero-masked).
			#[inline]
			pub fn $zero_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
				self, mask: u8, a: [u16; 8],
			) -> [u16; 8] {
				unsafe { $zero_intrinsic_fn::<NORM, SIGN>(mask, &a) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $merge_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			src: &[u16; 8], mask: u8, a: &[u16; 8],
		) -> [u16; 8] {
			unsafe {
				let vsrc = loadu_ph128(src.as_ptr());
				let va = loadu_ph128(a.as_ptr());
				let vr = _mm_mask_getmant_ph::<NORM, SIGN>(vsrc, mask, va);
				let mut out = [0u16; 8];
				storeu_ph128(out.as_mut_ptr(), vr);
				out
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $zero_intrinsic_fn<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
			mask: u8, a: &[u16; 8],
		) -> [u16; 8] {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vr = _mm_maskz_getmant_ph::<NORM, SIGN>(mask, va);
				let mut out = [0u16; 8];
				storeu_ph128(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}

fp16_getmant_masked_512!(getmant_ph_u16x32_merge_masked, getmant_ph_u16x32_zero_masked, mask_getmant_ph512_intrinsic, maskz_getmant_ph512_intrinsic);
fp16_getmant_masked_256!(getmant_ph_u16x16_merge_masked, getmant_ph_u16x16_zero_masked, mask_getmant_ph256_intrinsic, maskz_getmant_ph256_intrinsic);
fp16_getmant_masked_128!(getmant_ph_u16x8_merge_masked, getmant_ph_u16x8_zero_masked, mask_getmant_ph128_intrinsic, maskz_getmant_ph128_intrinsic);

// Merge/zero-masked `cvtph_pd`/`cvtpd_ph`: new `simd_cvt_widen_masked!`/
// `simd_cvt_narrow_masked!` (added alongside the DQ f32<->i64/u64 work).
// Mask stays `u8` at all 3 widths (architectural minimum, confirmed via
// stdarch), unlike the unmasked forms' own width scaling.

simd_cvt_widen_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = ph_to_f64x2_merge_masked, zero_fn = ph_to_f64x2_zero_masked,
	merge_intrinsic_fn = mask_ph_to_f64x2_intrinsic, zero_intrinsic_fn = maskz_ph_to_f64x2_intrinsic,
	in_width = 8, out_width = 2,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m128d, out_loadu = core::arch::x86_64::_mm_loadu_pd, out_storeu = core::arch::x86_64::_mm_storeu_pd, mask = u8,
	merge_intrinsic = _mm_mask_cvtph_pd, zero_intrinsic = _mm_maskz_cvtph_pd,
	merge_doc = "[`Avx512Fp16Vl::ph_to_f64x2`] where `mask` bit is set, else copied from `src` (`vcvtph2pd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::ph_to_f64x2`] where `mask` bit is set, else zero (`vcvtph2pd`, 128-bit, zero-masked).",
}
simd_cvt_widen_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = ph_to_f64x4_merge_masked, zero_fn = ph_to_f64x4_zero_masked,
	merge_intrinsic_fn = mask_ph_to_f64x4_intrinsic, zero_intrinsic_fn = maskz_ph_to_f64x4_intrinsic,
	in_width = 8, out_width = 4,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m256d, out_loadu = core::arch::x86_64::_mm256_loadu_pd, out_storeu = core::arch::x86_64::_mm256_storeu_pd, mask = u8,
	merge_intrinsic = _mm256_mask_cvtph_pd, zero_intrinsic = _mm256_maskz_cvtph_pd,
	merge_doc = "[`Avx512Fp16Vl::ph_to_f64x4`] where `mask` bit is set, else copied from `src` (`vcvtph2pd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::ph_to_f64x4`] where `mask` bit is set, else zero (`vcvtph2pd`, 256-bit, zero-masked).",
}
simd_cvt_widen_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = ph_to_f64x8_merge_masked, zero_fn = ph_to_f64x8_zero_masked,
	merge_intrinsic_fn = mask_ph_to_f64x8_intrinsic, zero_intrinsic_fn = maskz_ph_to_f64x8_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = u16, in_vec = __m128h, in_loadu = loadu_ph128,
	out_elem = f64, out_vec = __m512d, out_loadu = core::arch::x86_64::_mm512_loadu_pd, out_storeu = core::arch::x86_64::_mm512_storeu_pd, mask = u8,
	merge_intrinsic = _mm512_mask_cvtph_pd, zero_intrinsic = _mm512_maskz_cvtph_pd,
	merge_doc = "[`Avx512Fp16::ph_to_f64x8`] where `mask` bit is set, else copied from `src` (`vcvtph2pd`, 512-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16::ph_to_f64x8`] where `mask` bit is set, else zero (`vcvtph2pd`, 512-bit, zero-masked).",
}

simd_cvt_narrow_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = f64x2_to_ph_merge_masked, zero_fn = f64x2_to_ph_zero_masked,
	merge_intrinsic_fn = mask_f64x2_to_ph_intrinsic, zero_intrinsic_fn = maskz_f64x2_to_ph_intrinsic,
	in_width = 2, out_width = 8,
	in_elem = f64, in_vec = __m128d, in_loadu = core::arch::x86_64::_mm_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_loadu = loadu_ph128, out_storeu = storeu_ph128, mask = u8,
	merge_intrinsic = _mm_mask_cvtpd_ph, zero_intrinsic = _mm_maskz_cvtpd_ph,
	merge_doc = "[`Avx512Fp16Vl::f64x2_to_ph`] where `mask` bit is set, else copied from `src` (`vcvtpd2ph`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::f64x2_to_ph`] where `mask` bit is set, else zero (`vcvtpd2ph`, 128-bit, zero-masked).",
}
simd_cvt_narrow_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = f64x4_to_ph_merge_masked, zero_fn = f64x4_to_ph_zero_masked,
	merge_intrinsic_fn = mask_f64x4_to_ph_intrinsic, zero_intrinsic_fn = maskz_f64x4_to_ph_intrinsic,
	in_width = 4, out_width = 8,
	in_elem = f64, in_vec = __m256d, in_loadu = core::arch::x86_64::_mm256_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_loadu = loadu_ph128, out_storeu = storeu_ph128, mask = u8,
	merge_intrinsic = _mm256_mask_cvtpd_ph, zero_intrinsic = _mm256_maskz_cvtpd_ph,
	merge_doc = "[`Avx512Fp16Vl::f64x4_to_ph`] where `mask` bit is set, else copied from `src` (`vcvtpd2ph`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::f64x4_to_ph`] where `mask` bit is set, else zero (`vcvtpd2ph`, 256-bit, zero-masked).",
}
simd_cvt_narrow_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = f64x8_to_ph_merge_masked, zero_fn = f64x8_to_ph_zero_masked,
	merge_intrinsic_fn = mask_f64x8_to_ph_intrinsic, zero_intrinsic_fn = maskz_f64x8_to_ph_intrinsic,
	in_width = 8, out_width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = core::arch::x86_64::_mm512_loadu_pd,
	out_elem = u16, out_vec = __m128h, out_loadu = loadu_ph128, out_storeu = storeu_ph128, mask = u8,
	merge_intrinsic = _mm512_mask_cvtpd_ph, zero_intrinsic = _mm512_maskz_cvtpd_ph,
	merge_doc = "[`Avx512Fp16::f64x8_to_ph`] where `mask` bit is set, else copied from `src` (`vcvtpd2ph`, 512-bit, merge-masked).",
	zero_doc = "[`Avx512Fp16::f64x8_to_ph`] where `mask` bit is set, else zero (`vcvtpd2ph`, 512-bit, zero-masked).",
}

// Merge/zero-masked `_round_ph` (add/sub/mul/div via `simd_binop_imm_masked`,
// sqrt via `simd_unop_imm_masked`, FMA family via `simd_ternop_imm_masked`).
// 512-bit only, matching the unmasked `_round_ph` family's own scope.

simd_binop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = add_round_ph_u16x32_merge_masked, zero_fn = add_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_add_round_ph_intrinsic, zero_intrinsic_fn = maskz_add_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_add_round_ph, zero_intrinsic = _mm512_maskz_add_round_ph,
	merge_doc = "[`Avx512Fp16::add_round_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vaddph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::add_round_ph_u16x32`] where `mask` bit is set, else zero (`vaddph`, zero-masked).",
}
simd_binop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = sub_round_ph_u16x32_merge_masked, zero_fn = sub_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_sub_round_ph_intrinsic, zero_intrinsic_fn = maskz_sub_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_sub_round_ph, zero_intrinsic = _mm512_maskz_sub_round_ph,
	merge_doc = "[`Avx512Fp16::sub_round_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vsubph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::sub_round_ph_u16x32`] where `mask` bit is set, else zero (`vsubph`, zero-masked).",
}
simd_binop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = mul_round_ph_u16x32_merge_masked, zero_fn = mul_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_mul_round_ph_intrinsic, zero_intrinsic_fn = maskz_mul_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_mul_round_ph, zero_intrinsic = _mm512_maskz_mul_round_ph,
	merge_doc = "[`Avx512Fp16::mul_round_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vmulph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::mul_round_ph_u16x32`] where `mask` bit is set, else zero (`vmulph`, zero-masked).",
}
simd_binop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = div_round_ph_u16x32_merge_masked, zero_fn = div_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_div_round_ph_intrinsic, zero_intrinsic_fn = maskz_div_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_div_round_ph, zero_intrinsic = _mm512_maskz_div_round_ph,
	merge_doc = "[`Avx512Fp16::div_round_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vdivph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::div_round_ph_u16x32`] where `mask` bit is set, else zero (`vdivph`, zero-masked).",
}
simd_unop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = sqrt_round_ph_u16x32_merge_masked, zero_fn = sqrt_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_sqrt_round_ph_intrinsic, zero_intrinsic_fn = maskz_sqrt_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_sqrt_round_ph, zero_intrinsic = _mm512_maskz_sqrt_round_ph,
	merge_doc = "[`Avx512Fp16::sqrt_round_ph_u16x32`] where `mask` bit is set, else copied from `src` (`vsqrtph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::sqrt_round_ph_u16x32`] where `mask` bit is set, else zero (`vsqrtph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fmadd_round_ph_u16x32_merge_masked, zero_fn = fmadd_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fmadd_round_ph_intrinsic, zero_intrinsic_fn = maskz_fmadd_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fmadd_round_ph, zero_intrinsic = _mm512_maskz_fmadd_round_ph,
	merge_doc = "[`Avx512Fp16::fmadd_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfmaddph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fmadd_round_ph_u16x32`] where `mask` bit is set, else zero (`vfmaddph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fmsub_round_ph_u16x32_merge_masked, zero_fn = fmsub_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fmsub_round_ph_intrinsic, zero_intrinsic_fn = maskz_fmsub_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fmsub_round_ph, zero_intrinsic = _mm512_maskz_fmsub_round_ph,
	merge_doc = "[`Avx512Fp16::fmsub_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfmsubph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fmsub_round_ph_u16x32`] where `mask` bit is set, else zero (`vfmsubph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fnmadd_round_ph_u16x32_merge_masked, zero_fn = fnmadd_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fnmadd_round_ph_intrinsic, zero_intrinsic_fn = maskz_fnmadd_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fnmadd_round_ph, zero_intrinsic = _mm512_maskz_fnmadd_round_ph,
	merge_doc = "[`Avx512Fp16::fnmadd_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfnmaddph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fnmadd_round_ph_u16x32`] where `mask` bit is set, else zero (`vfnmaddph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fnmsub_round_ph_u16x32_merge_masked, zero_fn = fnmsub_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fnmsub_round_ph_intrinsic, zero_intrinsic_fn = maskz_fnmsub_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fnmsub_round_ph, zero_intrinsic = _mm512_maskz_fnmsub_round_ph,
	merge_doc = "[`Avx512Fp16::fnmsub_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfnmsubph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fnmsub_round_ph_u16x32`] where `mask` bit is set, else zero (`vfnmsubph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fmaddsub_round_ph_u16x32_merge_masked, zero_fn = fmaddsub_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fmaddsub_round_ph_intrinsic, zero_intrinsic_fn = maskz_fmaddsub_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fmaddsub_round_ph, zero_intrinsic = _mm512_maskz_fmaddsub_round_ph,
	merge_doc = "[`Avx512Fp16::fmaddsub_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfmaddsubph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fmaddsub_round_ph_u16x32`] where `mask` bit is set, else zero (`vfmaddsubph`, zero-masked).",
}
simd_ternop_imm_masked! {
	token = Avx512Fp16, target_feature = "avx512fp16",
	merge_fn = fmsubadd_round_ph_u16x32_merge_masked, zero_fn = fmsubadd_round_ph_u16x32_zero_masked,
	merge_intrinsic_fn = mask_fmsubadd_round_ph_intrinsic, zero_intrinsic_fn = maskz_fmsubadd_round_ph_intrinsic,
	width = 32, elem = u16, vec = __m512h, mask = u32,
	loadu = loadu_ph512, storeu = storeu_ph512,
	merge_intrinsic = _mm512_mask_fmsubadd_round_ph, zero_intrinsic = _mm512_maskz_fmsubadd_round_ph,
	merge_doc = "[`Avx512Fp16::fmsubadd_round_ph_u16x32`] where `mask` bit is set, else copied from `a` (`vfmsubaddph`, merge-masked).",
	zero_doc = "[`Avx512Fp16::fmsubadd_round_ph_u16x32`] where `mask` bit is set, else zero (`vfmsubaddph`, zero-masked).",
}

// `cmp_ph_mask` k1-gated form: `_mm512_mask_cmp_ph_mask(k1, a, b)` already
// computes `cmp(a, b) & k1` in hardware (confirmed via stdarch): same
// "one gated method, no merge/zero pair" shape as DQ's `fpclass_gated`.
macro_rules! fp16_cmp_mask_gated_512 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16 {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, k1: u32, a: [u16; 32], b: [u16; 32]) -> u32 {
				unsafe { $intrinsic_fn(k1, &a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16 via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16")]
		unsafe fn $intrinsic_fn(k1: u32, a: &[u16; 32], b: &[u16; 32]) -> u32 {
			unsafe {
				let va = loadu_ph512(a.as_ptr());
				let vb = loadu_ph512(b.as_ptr());
				_mm512_mask_cmp_ph_mask::<{ $pred }>(k1, va, vb)
			}
		}
	};
}
macro_rules! fp16_cmp_mask_gated_256 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, k1: u16, a: [u16; 16], b: [u16; 16]) -> u16 {
				unsafe { $intrinsic_fn(k1, &a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(k1: u16, a: &[u16; 16], b: &[u16; 16]) -> u16 {
			unsafe {
				let va = loadu_ph256(a.as_ptr());
				let vb = loadu_ph256(b.as_ptr());
				_mm256_mask_cmp_ph_mask::<{ $pred }>(k1, va, vb)
			}
		}
	};
}
macro_rules! fp16_cmp_mask_gated_128 {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, k1: u8, a: [u16; 8], b: [u16; 8]) -> u8 {
				unsafe { $intrinsic_fn(k1, &a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(k1: u8, a: &[u16; 8], b: &[u16; 8]) -> u8 {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vb = loadu_ph128(b.as_ptr());
				_mm_mask_cmp_ph_mask::<{ $pred }>(k1, va, vb)
			}
		}
	};
}
fp16_cmp_mask_gated_512!(cmpeq_ph_mask_u16x32_gated, cmpeq_ph512_mask_gated_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"[`Avx512Fp16::cmpeq_ph_mask_u16x32`] ANDed with `k1` (`vcmpph`, 512-bit, mask-gated).");
fp16_cmp_mask_gated_512!(cmplt_ph_mask_u16x32_gated, cmplt_ph512_mask_gated_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"[`Avx512Fp16::cmplt_ph_mask_u16x32`] ANDed with `k1` (`vcmpph`, 512-bit, mask-gated).");
fp16_cmp_mask_gated_512!(cmple_ph_mask_u16x32_gated, cmple_ph512_mask_gated_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"[`Avx512Fp16::cmple_ph_mask_u16x32`] ANDed with `k1` (`vcmpph`, 512-bit, mask-gated).");
fp16_cmp_mask_gated_512!(cmpgt_ph_mask_u16x32_gated, cmpgt_ph512_mask_gated_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"[`Avx512Fp16::cmpgt_ph_mask_u16x32`] ANDed with `k1` (`vcmpph`, 512-bit, mask-gated).");
fp16_cmp_mask_gated_512!(cmpge_ph_mask_u16x32_gated, cmpge_ph512_mask_gated_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"[`Avx512Fp16::cmpge_ph_mask_u16x32`] ANDed with `k1` (`vcmpph`, 512-bit, mask-gated).");
fp16_cmp_mask_gated_256!(cmpeq_ph_mask_u16x16_gated, cmpeq_ph256_mask_gated_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"[`Avx512Fp16Vl::cmpeq_ph_mask_u16x16`] ANDed with `k1` (`vcmpph`, 256-bit, mask-gated).");
fp16_cmp_mask_gated_256!(cmplt_ph_mask_u16x16_gated, cmplt_ph256_mask_gated_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"[`Avx512Fp16Vl::cmplt_ph_mask_u16x16`] ANDed with `k1` (`vcmpph`, 256-bit, mask-gated).");
fp16_cmp_mask_gated_256!(cmple_ph_mask_u16x16_gated, cmple_ph256_mask_gated_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"[`Avx512Fp16Vl::cmple_ph_mask_u16x16`] ANDed with `k1` (`vcmpph`, 256-bit, mask-gated).");
fp16_cmp_mask_gated_256!(cmpgt_ph_mask_u16x16_gated, cmpgt_ph256_mask_gated_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"[`Avx512Fp16Vl::cmpgt_ph_mask_u16x16`] ANDed with `k1` (`vcmpph`, 256-bit, mask-gated).");
fp16_cmp_mask_gated_256!(cmpge_ph_mask_u16x16_gated, cmpge_ph256_mask_gated_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"[`Avx512Fp16Vl::cmpge_ph_mask_u16x16`] ANDed with `k1` (`vcmpph`, 256-bit, mask-gated).");
fp16_cmp_mask_gated_128!(cmpeq_ph_mask_u16x8_gated, cmpeq_ph128_mask_gated_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"[`Avx512Fp16Vl::cmpeq_ph_mask_u16x8`] ANDed with `k1` (`vcmpph`, 128-bit, mask-gated).");
fp16_cmp_mask_gated_128!(cmplt_ph_mask_u16x8_gated, cmplt_ph128_mask_gated_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"[`Avx512Fp16Vl::cmplt_ph_mask_u16x8`] ANDed with `k1` (`vcmpph`, 128-bit, mask-gated).");
fp16_cmp_mask_gated_128!(cmple_ph_mask_u16x8_gated, cmple_ph128_mask_gated_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"[`Avx512Fp16Vl::cmple_ph_mask_u16x8`] ANDed with `k1` (`vcmpph`, 128-bit, mask-gated).");
fp16_cmp_mask_gated_128!(cmpgt_ph_mask_u16x8_gated, cmpgt_ph128_mask_gated_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"[`Avx512Fp16Vl::cmpgt_ph_mask_u16x8`] ANDed with `k1` (`vcmpph`, 128-bit, mask-gated).");
fp16_cmp_mask_gated_128!(cmpge_ph_mask_u16x8_gated, cmpge_ph128_mask_gated_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"[`Avx512Fp16Vl::cmpge_ph_mask_u16x8`] ANDed with `k1` (`vcmpph`, 128-bit, mask-gated).");

// Merge/zero-masked scalar `_sh` ops. `rcp/rsqrt/sqrt/min/max/getexp/
// scalef_sh` are `(src, mask, a, b) -> [u16;8]`, the exact shape
// `fp16_binop_masked_128!` already generates: pure reuse. `reduce_sh`/
// `roundscale_sh` add a single `IMM8` const generic, fitting
// `simd_binop_imm_masked!` directly (both `a`/`b` are `u16x8`, same as any
// other binop-imm-masked caller). `getmant_sh` has two const generics
// (`NORM`/`SIGN`), which no generic macro expresses: hand-written, mirroring
// `getmant_sh_intrinsic`'s existing shape above plus `src`/`mask`. `cmp_sh_mask`
// gets a k1-gated form only (same reasoning as `cmp_ph_mask` above).

fp16_binop_masked_128!(
	rcp_sh_merge_masked, rcp_sh_zero_masked,
	mask_rcp_sh_intrinsic, maskz_rcp_sh_intrinsic,
	core::arch::x86_64::_mm_mask_rcp_sh, core::arch::x86_64::_mm_maskz_rcp_sh,
	"[`Avx512Fp16Vl::rcp_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vrcpsh`, merge-masked).",
	"[`Avx512Fp16Vl::rcp_sh_u16x8`] where `mask` bit is set, else zero (`vrcpsh`, zero-masked)."
);
fp16_binop_masked_128!(
	rsqrt_sh_merge_masked, rsqrt_sh_zero_masked,
	mask_rsqrt_sh_intrinsic, maskz_rsqrt_sh_intrinsic,
	core::arch::x86_64::_mm_mask_rsqrt_sh, core::arch::x86_64::_mm_maskz_rsqrt_sh,
	"[`Avx512Fp16Vl::rsqrt_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vrsqrtsh`, merge-masked).",
	"[`Avx512Fp16Vl::rsqrt_sh_u16x8`] where `mask` bit is set, else zero (`vrsqrtsh`, zero-masked)."
);
fp16_binop_masked_128!(
	sqrt_sh_merge_masked, sqrt_sh_zero_masked,
	mask_sqrt_sh_intrinsic, maskz_sqrt_sh_intrinsic,
	core::arch::x86_64::_mm_mask_sqrt_sh, core::arch::x86_64::_mm_maskz_sqrt_sh,
	"[`Avx512Fp16Vl::sqrt_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vsqrtsh`, merge-masked).",
	"[`Avx512Fp16Vl::sqrt_sh_u16x8`] where `mask` bit is set, else zero (`vsqrtsh`, zero-masked)."
);
fp16_binop_masked_128!(
	min_sh_merge_masked, min_sh_zero_masked,
	mask_min_sh_intrinsic, maskz_min_sh_intrinsic,
	core::arch::x86_64::_mm_mask_min_sh, core::arch::x86_64::_mm_maskz_min_sh,
	"[`Avx512Fp16Vl::min_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vminsh`, merge-masked).",
	"[`Avx512Fp16Vl::min_sh_u16x8`] where `mask` bit is set, else zero (`vminsh`, zero-masked)."
);
fp16_binop_masked_128!(
	max_sh_merge_masked, max_sh_zero_masked,
	mask_max_sh_intrinsic, maskz_max_sh_intrinsic,
	core::arch::x86_64::_mm_mask_max_sh, core::arch::x86_64::_mm_maskz_max_sh,
	"[`Avx512Fp16Vl::max_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vmaxsh`, merge-masked).",
	"[`Avx512Fp16Vl::max_sh_u16x8`] where `mask` bit is set, else zero (`vmaxsh`, zero-masked)."
);
fp16_binop_masked_128!(
	getexp_sh_merge_masked, getexp_sh_zero_masked,
	mask_getexp_sh_intrinsic, maskz_getexp_sh_intrinsic,
	core::arch::x86_64::_mm_mask_getexp_sh, core::arch::x86_64::_mm_maskz_getexp_sh,
	"[`Avx512Fp16Vl::getexp_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vgetexpsh`, merge-masked).",
	"[`Avx512Fp16Vl::getexp_sh_u16x8`] where `mask` bit is set, else zero (`vgetexpsh`, zero-masked)."
);
fp16_binop_masked_128!(
	scalef_sh_merge_masked, scalef_sh_zero_masked,
	mask_scalef_sh_intrinsic, maskz_scalef_sh_intrinsic,
	core::arch::x86_64::_mm_mask_scalef_sh, core::arch::x86_64::_mm_maskz_scalef_sh,
	"[`Avx512Fp16Vl::scalef_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vscalefsh`, merge-masked).",
	"[`Avx512Fp16Vl::scalef_sh_u16x8`] where `mask` bit is set, else zero (`vscalefsh`, zero-masked)."
);

simd_binop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = reduce_sh_u16x8_merge_masked, zero_fn = reduce_sh_u16x8_zero_masked,
	merge_intrinsic_fn = mask_reduce_sh_intrinsic, zero_intrinsic_fn = maskz_reduce_sh_intrinsic,
	width = 8, elem = u16, vec = __m128h, mask = u8,
	loadu = loadu_ph128, storeu = storeu_ph128,
	merge_intrinsic = _mm_mask_reduce_sh, zero_intrinsic = _mm_maskz_reduce_sh,
	merge_doc = "[`Avx512Fp16Vl::reduce_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vreducesh`, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::reduce_sh_u16x8`] where `mask` bit is set, else zero (`vreducesh`, zero-masked).",
}
simd_binop_imm_masked! {
	token = Avx512Fp16Vl, target_feature = "avx512fp16,avx512vl",
	merge_fn = roundscale_sh_u16x8_merge_masked, zero_fn = roundscale_sh_u16x8_zero_masked,
	merge_intrinsic_fn = mask_roundscale_sh_intrinsic, zero_intrinsic_fn = maskz_roundscale_sh_intrinsic,
	width = 8, elem = u16, vec = __m128h, mask = u8,
	loadu = loadu_ph128, storeu = storeu_ph128,
	merge_intrinsic = _mm_mask_roundscale_sh, zero_intrinsic = _mm_maskz_roundscale_sh,
	merge_doc = "[`Avx512Fp16Vl::roundscale_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vrndscalesh`, merge-masked).",
	zero_doc = "[`Avx512Fp16Vl::roundscale_sh_u16x8`] where `mask` bit is set, else zero (`vrndscalesh`, zero-masked).",
}

impl Avx512Fp16Vl {
	/// [`Avx512Fp16Vl::getmant_sh_u16x8`] where `mask` bit is set, else copied from `src` (`vgetmantsh`, merge-masked).
	#[inline]
	pub fn getmant_sh_merge_masked<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
		self, src: [u16; 8], mask: u8, a: [u16; 8], b: [u16; 8],
	) -> [u16; 8] {
		unsafe { mask_getmant_sh_intrinsic::<NORM, SIGN>(&src, mask, &a, &b) }
	}

	/// [`Avx512Fp16Vl::getmant_sh_u16x8`] where `mask` bit is set, else zero (`vgetmantsh`, zero-masked).
	#[inline]
	pub fn getmant_sh_zero_masked<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
		self, mask: u8, a: [u16; 8], b: [u16; 8],
	) -> [u16; 8] {
		unsafe { maskz_getmant_sh_intrinsic::<NORM, SIGN>(mask, &a, &b) }
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn mask_getmant_sh_intrinsic<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
	src: &[u16; 8], mask: u8, a: &[u16; 8], b: &[u16; 8],
) -> [u16; 8] {
	unsafe {
		let vsrc = loadu_ph128(src.as_ptr());
		let va = loadu_ph128(a.as_ptr());
		let vb = loadu_ph128(b.as_ptr());
		let vr = _mm_mask_getmant_sh::<NORM, SIGN>(vsrc, mask, va, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}
/// # Safety
/// Caller proved AVX-512FP16+AVX-512VL via the token.
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
unsafe fn maskz_getmant_sh_intrinsic<const NORM: _MM_MANTISSA_NORM_ENUM, const SIGN: _MM_MANTISSA_SIGN_ENUM>(
	mask: u8, a: &[u16; 8], b: &[u16; 8],
) -> [u16; 8] {
	unsafe {
		let va = loadu_ph128(a.as_ptr());
		let vb = loadu_ph128(b.as_ptr());
		let vr = _mm_maskz_getmant_sh::<NORM, SIGN>(mask, va, vb);
		let mut out = [0u16; 8];
		storeu_ph128(out.as_mut_ptr(), vr);
		out
	}
}

macro_rules! fp16_cmp_sh_mask_gated {
	($fixed_fn:ident, $intrinsic_fn:ident, $pred:path, $doc:literal) => {
		impl Avx512Fp16Vl {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, k1: u8, a: [u16; 8], b: [u16; 8]) -> u8 {
				unsafe { $intrinsic_fn(k1, &a, &b) }
			}
		}
		/// # Safety
		/// Caller proved AVX-512FP16+AVX-512VL via the token.
		#[inline]
		#[target_feature(enable = "avx512fp16,avx512vl")]
		unsafe fn $intrinsic_fn(k1: u8, a: &[u16; 8], b: &[u16; 8]) -> u8 {
			unsafe {
				let va = loadu_ph128(a.as_ptr());
				let vb = loadu_ph128(b.as_ptr());
				_mm_mask_cmp_sh_mask::<{ $pred }>(k1, va, vb)
			}
		}
	};
}

fp16_cmp_sh_mask_gated!(cmpeq_sh_mask_gated, cmpeq_sh_mask_gated_intrinsic, core::arch::x86_64::_CMP_EQ_OQ,
	"[`Avx512Fp16Vl::cmpeq_sh_mask_u16x8`] ANDed with `k1` (`vcmpsh`, mask-gated).");
fp16_cmp_sh_mask_gated!(cmplt_sh_mask_gated, cmplt_sh_mask_gated_intrinsic, core::arch::x86_64::_CMP_LT_OQ,
	"[`Avx512Fp16Vl::cmplt_sh_mask_u16x8`] ANDed with `k1` (`vcmpsh`, mask-gated).");
fp16_cmp_sh_mask_gated!(cmple_sh_mask_gated, cmple_sh_mask_gated_intrinsic, core::arch::x86_64::_CMP_LE_OQ,
	"[`Avx512Fp16Vl::cmple_sh_mask_u16x8`] ANDed with `k1` (`vcmpsh`, mask-gated).");
fp16_cmp_sh_mask_gated!(cmpgt_sh_mask_gated, cmpgt_sh_mask_gated_intrinsic, core::arch::x86_64::_CMP_GT_OQ,
	"[`Avx512Fp16Vl::cmpgt_sh_mask_u16x8`] ANDed with `k1` (`vcmpsh`, mask-gated).");
fp16_cmp_sh_mask_gated!(cmpge_sh_mask_gated, cmpge_sh_mask_gated_intrinsic, core::arch::x86_64::_CMP_GE_OQ,
	"[`Avx512Fp16Vl::cmpge_sh_mask_u16x8`] ANDed with `k1` (`vcmpsh`, mask-gated).");

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512fp16.rs"]
mod tests;
