//! AVX-512F base 512-bit ZMM rung (`avx512f`). Token: `Avx512f`.
//! Provides float FMA, wide integer arithmetic, masks, and compares.
//! `sqrt`/`rcp14`/`rsqrt14` are fixed-width only, see `simd_unop_fixed` doc.

use core::arch::x86_64::{
	__m128i, __m512, __m512d, __m512i, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _mm512_add_epi32,
	_mm512_add_epi64, _mm512_add_pd, _mm512_add_ps, _mm512_and_si512, _mm512_andnot_si512, _mm512_castpd_si512,
	_mm512_castps_si512, _mm512_castsi512_pd, _mm512_castsi512_ps, _mm512_cmpeq_epi32_mask, _mm512_cmpeq_epi64_mask,
	_mm512_cmp_pd_mask, _mm512_cmp_ps_mask, _mm512_cmpgt_epi32_mask, _mm512_cmpgt_epi64_mask, _mm512_cmpgt_epu32_mask,
	_mm512_cmpgt_epu64_mask, _mm512_div_pd, _mm512_div_ps, _mm512_fmadd_pd, _mm512_fmadd_ps, _mm512_fmsub_pd,
	_mm512_fmsub_ps, _mm512_fnmadd_pd, _mm512_fnmadd_ps, _mm512_fnmsub_pd, _mm512_fnmsub_ps, _mm512_loadu_pd,
	_mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_abs_epi32, _mm512_mask_abs_epi64, _mm512_mask_add_epi32,
	_mm512_mask_add_epi64, _mm512_mask_add_pd, _mm512_mask_add_ps, _mm512_mask_blend_epi32, _mm512_mask_blend_epi64,
	_mm512_mask_blend_pd, _mm512_mask_blend_ps, _mm512_mask_compress_epi32, _mm512_mask_compress_epi64,
	_mm512_mask_compress_pd, _mm512_mask_compress_ps, _mm512_mask_compressstoreu_epi32,
	_mm512_mask_compressstoreu_epi64, _mm512_mask_compressstoreu_pd, _mm512_mask_compressstoreu_ps,
	_mm512_mask_div_pd, _mm512_mask_div_ps,
	_mm512_mask_expand_epi32, _mm512_mask_expand_epi64, _mm512_mask_expand_pd, _mm512_mask_expand_ps,
	_mm512_mask_expandloadu_epi32, _mm512_mask_expandloadu_epi64, _mm512_mask_expandloadu_pd,
	_mm512_mask_expandloadu_ps, _mm512_maskz_expandloadu_epi32, _mm512_maskz_expandloadu_epi64,
	_mm512_maskz_expandloadu_pd, _mm512_maskz_expandloadu_ps,
	_mm512_mask_fmadd_pd, _mm512_mask_fmadd_ps, _mm512_mask_fmsub_pd, _mm512_mask_fmsub_ps, _mm512_mask_fnmadd_pd,
	_mm512_mask_fnmadd_ps, _mm512_mask_fnmsub_pd, _mm512_mask_fnmsub_ps, _mm512_mask_max_epi32, _mm512_mask_max_epi64,
	_mm512_mask_max_epu32, _mm512_mask_max_epu64, _mm512_mask_max_pd, _mm512_mask_max_ps, _mm512_mask_min_epi32,
	_mm512_mask_min_epi64, _mm512_mask_min_epu32, _mm512_mask_min_epu64, _mm512_mask_min_pd, _mm512_mask_min_ps,
	_mm512_mask_mul_pd, _mm512_mask_mul_ps, _mm512_mask_mullo_epi32, _mm512_mask_sub_epi32, _mm512_mask_sub_epi64,
	_mm512_mask_sub_pd, _mm512_mask_sub_ps, _mm512_maskz_abs_epi32, _mm512_maskz_abs_epi64, _mm512_maskz_add_epi32,
	_mm512_maskz_add_epi64, _mm512_maskz_add_pd, _mm512_maskz_add_ps, _mm512_maskz_compress_epi32,
	_mm512_maskz_compress_epi64, _mm512_maskz_compress_pd, _mm512_maskz_compress_ps, _mm512_maskz_div_pd,
	_mm512_maskz_div_ps, _mm512_maskz_expand_epi32, _mm512_maskz_expand_epi64, _mm512_maskz_expand_pd,
	_mm512_maskz_expand_ps, _mm512_maskz_fmadd_pd,
	_mm512_maskz_fmadd_ps, _mm512_maskz_fmsub_pd, _mm512_maskz_fmsub_ps, _mm512_maskz_fnmadd_pd, _mm512_maskz_fnmadd_ps,
	_mm512_maskz_fnmsub_pd, _mm512_maskz_fnmsub_ps, _mm512_maskz_max_epi32, _mm512_maskz_max_epi64, _mm512_maskz_max_epu32,
	_mm512_maskz_max_epu64, _mm512_maskz_max_pd, _mm512_maskz_max_ps, _mm512_maskz_min_epi32, _mm512_maskz_min_epi64,
	_mm512_maskz_min_epu32, _mm512_maskz_min_epu64, _mm512_maskz_min_pd, _mm512_maskz_min_ps, _mm512_maskz_mul_pd,
	_mm512_maskz_mul_ps, _mm512_maskz_mullo_epi32, _mm512_maskz_set1_epi32, _mm512_maskz_set1_epi64,
	_mm512_maskz_sub_epi32, _mm512_maskz_sub_epi64, _mm512_maskz_sub_pd, _mm512_maskz_sub_ps, _mm512_max_epi32,
	_mm512_max_epi64, _mm512_max_epu32, _mm512_max_epu64, _mm512_max_pd, _mm512_max_ps, _mm512_min_epi32,
	_mm512_min_epi64, _mm512_min_epu32, _mm512_min_epu64, _mm512_min_pd, _mm512_min_ps, _mm512_mul_pd, _mm512_mul_ps,
	_mm512_abs_epi32, _mm512_abs_epi64, _mm512_mullo_epi32, _mm512_or_si512, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_sll_epi32,
	_mm512_sllv_epi32, _mm512_sllv_epi64, _mm512_sra_epi32, _mm512_srav_epi32, _mm512_srav_epi64, _mm512_srl_epi32,
	_mm512_srlv_epi32, _mm512_srlv_epi64, _mm512_storeu_pd, _mm512_storeu_ps, _mm512_storeu_si512, _mm512_sub_epi32,
	_mm512_sub_epi64, _mm512_sub_pd, _mm512_sub_ps, _mm512_mask_ternarylogic_epi32, _mm512_mask_ternarylogic_epi64,
	_mm512_maskz_ternarylogic_epi32, _mm512_maskz_ternarylogic_epi64, _mm512_ternarylogic_epi32,
	_mm512_ternarylogic_epi64, _mm512_test_epi32_mask, _mm512_test_epi64_mask, _mm512_xor_si512, _mm512_rcp14_pd,
	_mm512_rcp14_ps, _mm512_rsqrt14_pd, _mm512_rsqrt14_ps, _mm512_sqrt_pd, _mm512_sqrt_ps,
	_mm512_extracti32x4_epi32, _mm512_inserti32x4, _mm_loadu_si128, _mm_storeu_si128,
	_mm512_shuffle_ps, _mm512_unpackhi_ps, _mm512_unpacklo_ps,
	_mm512_rol_epi32, _mm512_ror_epi32, _mm512_rol_epi64, _mm512_ror_epi64,
	_mm512_reduce_add_epi32, _mm512_reduce_add_epi64, _mm512_reduce_add_ps, _mm512_reduce_add_pd,
	_mm512_reduce_mul_epi32, _mm512_reduce_mul_epi64, _mm512_reduce_mul_ps, _mm512_reduce_mul_pd,
	_mm512_reduce_max_epi32, _mm512_reduce_max_epu32, _mm512_reduce_max_epi64, _mm512_reduce_max_epu64,
	_mm512_reduce_max_ps, _mm512_reduce_max_pd,
	_mm512_reduce_min_epi32, _mm512_reduce_min_epu32, _mm512_reduce_min_epi64, _mm512_reduce_min_epu64,
	_mm512_reduce_min_ps, _mm512_reduce_min_pd,
	_mm512_mul_epi32, _mm512_mul_epu32, _mm512_srli_epi64, _mm512_slli_epi64,
	_mm512_moveldup_ps, _mm512_movehdup_ps, _mm512_permute_ps, _mm512_permute_pd,
	_mm512_unpacklo_pd, _mm512_unpackhi_pd, _mm512_fmaddsub_ps, _mm512_fmaddsub_pd,
	_mm512_maskz_loadu_ps, _mm512_mask_storeu_ps, _mm512_maskz_loadu_pd, _mm512_mask_storeu_pd,
	_mm512_maskz_loadu_epi32, _mm512_mask_storeu_epi32, _mm512_maskz_loadu_epi64, _mm512_mask_storeu_epi64,
};


use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::avx::f16c::{f16_to_f32_scalar, f32_to_f16_scalar};
use super::super::macros::{
	scalar_only_binop, simd_binop, simd_binop_fixed, simd_binop_imm_fixed, simd_binop_masked, simd_compressstoreu,
	simd_expandloadu, simd_extract_imm, simd_insert_imm, simd_reduce, simd_shift_imm, simd_ternarylogic, simd_ternop,
	simd_ternop_masked, simd_unop, simd_unop_fixed, simd_unop_imm, simd_unop_masked,
};

/// Proof token: AVX-512F available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512f(());

impl Avx512f {
	/// `None` if the CPU (or the compile-time target) lacks AVX-512F.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From resolved tier (`V4` lists `Feature::Avx512f`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx512f).then_some(Avx512f(()))
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`): unlike
	/// [`Self::from_level`], not gated on the full AVX-512 V4 bundle.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512f).then_some(Avx512f(()))
	}
}

macro_rules! avx512_f32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = f32, vec = __m512, loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_f32_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = f32, vec = __m512, mask = u16,
			loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512_f64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = f64, vec = __m512d, loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_f64_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = f64, vec = __m512d, mask = u8,
			loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512_i32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_i32_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = i32, vec = __m512i, mask = u16,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512_u32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u32_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, mask = u16,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn and_ps_bitcast(a: __m512, b: __m512) -> __m512 {
	_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn or_ps_bitcast(a: __m512, b: __m512) -> __m512 {
	_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor_ps_bitcast(a: __m512, b: __m512) -> __m512 {
	_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn andnot_ps_bitcast(a: __m512, b: __m512) -> __m512 {
	_mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn and_pd_bitcast(a: __m512d, b: __m512d) -> __m512d {
	_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn or_pd_bitcast(a: __m512d, b: __m512d) -> __m512d {
	_mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor_pd_bitcast(a: __m512d, b: __m512d) -> __m512d {
	_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
}
/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn andnot_pd_bitcast(a: __m512d, b: __m512d) -> __m512d {
	_mm512_castsi512_pd(_mm512_andnot_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
}

avx512_f32_binop!(
	add_f32x16, add_f32_slice, addps, _mm512_add_ps, |x, y| x + y,
	"`a + b` per lane (`vaddps`, 512-bit).",
	"`out[i] = a[i] + b[i]`. 16-wide `add_f32x16` chunks, scalar remainder."
);
avx512_f32_binop!(
	sub_f32x16, sub_f32_slice, subps, _mm512_sub_ps, |x, y| x - y,
	"`a - b` per lane (`vsubps`, 512-bit).",
	"`out[i] = a[i] - b[i]`. 16-wide `sub_f32x16` chunks, scalar remainder."
);
avx512_f32_binop!(
	mul_f32x16, mul_f32_slice, mulps, _mm512_mul_ps, |x, y| x * y,
	"`a * b` per lane (`vmulps`, 512-bit).",
	"`out[i] = a[i] * b[i]`. 16-wide `mul_f32x16` chunks, scalar remainder."
);
avx512_f32_binop!(
	div_f32x16, div_f32_slice, divps, _mm512_div_ps, |x, y| x / y,
	"`a / b` per lane (`vdivps`, 512-bit).",
	"`out[i] = a[i] / b[i]`. 16-wide `div_f32x16` chunks, scalar remainder."
);
avx512_f32_binop!(
	min_f32x16, min_f32_slice, minps, _mm512_min_ps, |x, y| x.min(y),
	"Per-lane min (`vminps`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])`. 16-wide `min_f32x16` chunks, scalar remainder."
);
avx512_f32_binop!(
	max_f32x16, max_f32_slice, maxps, _mm512_max_ps, |x, y| x.max(y),
	"Per-lane max (`vmaxps`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])`. 16-wide `max_f32x16` chunks, scalar remainder."
);
avx512_f32_binop_masked!(
	add_f32x16_merge_masked, add_f32x16_zero_masked, mask_add_ps_intrinsic, maskz_add_ps_intrinsic,
	_mm512_mask_add_ps, _mm512_maskz_add_ps,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddps`, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddps`, zero-masked)."
);
avx512_f32_binop_masked!(
	sub_f32x16_merge_masked, sub_f32x16_zero_masked, mask_sub_ps_intrinsic, maskz_sub_ps_intrinsic,
	_mm512_mask_sub_ps, _mm512_maskz_sub_ps,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubps`, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubps`, zero-masked)."
);
avx512_f32_binop_masked!(
	mul_f32x16_merge_masked, mul_f32x16_zero_masked, mask_mul_ps_intrinsic, maskz_mul_ps_intrinsic,
	_mm512_mask_mul_ps, _mm512_maskz_mul_ps,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulps`, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulps`, zero-masked)."
);
avx512_f32_binop_masked!(
	div_f32x16_merge_masked, div_f32x16_zero_masked, mask_div_ps_intrinsic, maskz_div_ps_intrinsic,
	_mm512_mask_div_ps, _mm512_maskz_div_ps,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivps`, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivps`, zero-masked)."
);
avx512_f32_binop_masked!(
	min_f32x16_merge_masked, min_f32x16_zero_masked, mask_min_ps_intrinsic, maskz_min_ps_intrinsic,
	_mm512_mask_min_ps, _mm512_maskz_min_ps,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminps`, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminps`, zero-masked)."
);
avx512_f32_binop_masked!(
	max_f32x16_merge_masked, max_f32x16_zero_masked, mask_max_ps_intrinsic, maskz_max_ps_intrinsic,
	_mm512_mask_max_ps, _mm512_maskz_max_ps,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxps`, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxps`, zero-masked)."
);

// Structural (not elementwise) 512-bit float ops: no honest per-lane scalar
// reference (result lane depends on lane *position*, not just `a[i]`/`b[i]`),
// so fixed-width only, same reasoning as the DQ extract/insert family and
// the mirroring 256-bit `Avx` ops.
simd_binop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = unpacklo_f32x16, intrinsic_fn = unpacklo_ps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_unpacklo_ps,
	fixed_doc = "Interleaves the low half of each 128-bit lane of `a`/`b` (`vunpcklps`, 512-bit).",
}
simd_binop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = unpackhi_f32x16, intrinsic_fn = unpackhi_ps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_unpackhi_ps,
	fixed_doc = "Interleaves the high half of each 128-bit lane of `a`/`b` (`vunpckhps`, 512-bit).",
}
simd_binop_imm_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = shuffle_f32x16, intrinsic_fn = shuffle_ps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_shuffle_ps,
	fixed_doc = "Per-128-bit-lane 4-way shuffle of `a`/`b` by `IMM8` (`vshufps`, 512-bit).",
}

avx512_f32_binop!(
	and_f32x16, and_f32_slice, andps, and_ps_bitcast, |x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()),
	"`a & b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f32::from_bits(a[i].to_bits() & b[i].to_bits())`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	or_f32x16, or_f32_slice, orps, or_ps_bitcast, |x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()),
	"`a | b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f32::from_bits(a[i].to_bits() | b[i].to_bits())`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	xor_f32x16, xor_f32_slice, xorps, xor_ps_bitcast, |x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()),
	"`a ^ b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f32::from_bits(a[i].to_bits() ^ b[i].to_bits())`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	andnot_f32x16, andnot_f32_slice, andnps, andnot_ps_bitcast, |x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()),
	"`!a & b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f32::from_bits(!a[i].to_bits() & b[i].to_bits())`. 16-wide chunks, scalar remainder."
);

// Float compare: k-mask + maskz_set1_epi32/64 cast back to float (all-1s bits).
#[inline]
unsafe fn vcmpeq_ps512(a: __m512, b: __m512) -> __m512 {
	unsafe {
		let k = _mm512_cmp_ps_mask::<{ _CMP_EQ_OQ }>(a, b);
		_mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1))
	}
}
#[inline]
unsafe fn vcmplt_ps512(a: __m512, b: __m512) -> __m512 {
	unsafe {
		let k = _mm512_cmp_ps_mask::<{ _CMP_LT_OQ }>(a, b);
		_mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1))
	}
}
#[inline]
unsafe fn vcmple_ps512(a: __m512, b: __m512) -> __m512 {
	unsafe {
		let k = _mm512_cmp_ps_mask::<{ _CMP_LE_OQ }>(a, b);
		_mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1))
	}
}
#[inline]
unsafe fn vcmpgt_ps512(a: __m512, b: __m512) -> __m512 {
	unsafe {
		let k = _mm512_cmp_ps_mask::<{ _CMP_GT_OQ }>(a, b);
		_mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1))
	}
}
#[inline]
unsafe fn vcmpge_ps512(a: __m512, b: __m512) -> __m512 {
	unsafe {
		let k = _mm512_cmp_ps_mask::<{ _CMP_GE_OQ }>(a, b);
		_mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1))
	}
}
#[inline]
unsafe fn vcmpeq_pd512(a: __m512d, b: __m512d) -> __m512d {
	unsafe {
		let k = _mm512_cmp_pd_mask::<{ _CMP_EQ_OQ }>(a, b);
		_mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, -1))
	}
}
#[inline]
unsafe fn vcmplt_pd512(a: __m512d, b: __m512d) -> __m512d {
	unsafe {
		let k = _mm512_cmp_pd_mask::<{ _CMP_LT_OQ }>(a, b);
		_mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, -1))
	}
}
#[inline]
unsafe fn vcmple_pd512(a: __m512d, b: __m512d) -> __m512d {
	unsafe {
		let k = _mm512_cmp_pd_mask::<{ _CMP_LE_OQ }>(a, b);
		_mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, -1))
	}
}
#[inline]
unsafe fn vcmpgt_pd512(a: __m512d, b: __m512d) -> __m512d {
	unsafe {
		let k = _mm512_cmp_pd_mask::<{ _CMP_GT_OQ }>(a, b);
		_mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, -1))
	}
}
#[inline]
unsafe fn vcmpge_pd512(a: __m512d, b: __m512d) -> __m512d {
	unsafe {
		let k = _mm512_cmp_pd_mask::<{ _CMP_GE_OQ }>(a, b);
		_mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, -1))
	}
}

avx512_f32_binop!(
	cmpeq_f32x16, cmpeq_f32_slice, vcmpeqps, vcmpeq_ps512,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane equality mask (`vcmpps` EQ_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	cmplt_f32x16, cmplt_f32_slice, vcmpltps, vcmplt_ps512,
	|x, y| if x < y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-than mask (`vcmpps` LT_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	cmple_f32x16, cmple_f32_slice, vcmpleps, vcmple_ps512,
	|x, y| if x <= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-equal mask (`vcmpps` LE_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	cmpgt_f32x16, cmpgt_f32_slice, vcmpgtps, vcmpgt_ps512,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-than mask (`vcmpps` GT_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx512_f32_binop!(
	cmpge_f32x16, cmpge_f32_slice, vcmpgeps, vcmpge_ps512,
	|x, y| if x >= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-equal mask (`vcmpps` GE_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 16-wide chunks, scalar remainder."
);

avx512_f64_binop!(
	add_f64x8, add_f64_slice, addpd, _mm512_add_pd, |x, y| x + y,
	"`a + b` per lane (`vaddpd`, 512-bit).",
	"`out[i] = a[i] + b[i]`. 8-wide `add_f64x8` chunks, scalar remainder."
);
avx512_f64_binop!(
	sub_f64x8, sub_f64_slice, subpd, _mm512_sub_pd, |x, y| x - y,
	"`a - b` per lane (`vsubpd`, 512-bit).",
	"`out[i] = a[i] - b[i]`. 8-wide `sub_f64x8` chunks, scalar remainder."
);
avx512_f64_binop!(
	mul_f64x8, mul_f64_slice, mulpd, _mm512_mul_pd, |x, y| x * y,
	"`a * b` per lane (`vmulpd`, 512-bit).",
	"`out[i] = a[i] * b[i]`. 8-wide `mul_f64x8` chunks, scalar remainder."
);
avx512_f64_binop!(
	div_f64x8, div_f64_slice, divpd, _mm512_div_pd, |x, y| x / y,
	"`a / b` per lane (`vdivpd`, 512-bit).",
	"`out[i] = a[i] / b[i]`. 8-wide `div_f64x8` chunks, scalar remainder."
);
avx512_f64_binop!(
	min_f64x8, min_f64_slice, minpd, _mm512_min_pd, |x, y| x.min(y),
	"Per-lane min (`vminpd`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f64::min`.",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_f64x8` chunks, scalar remainder."
);
avx512_f64_binop!(
	max_f64x8, max_f64_slice, maxpd, _mm512_max_pd, |x, y| x.max(y),
	"Per-lane max (`vmaxpd`, 512-bit). NaN: second-operand-on-NaN, not IEEE `f64::max`.",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_f64x8` chunks, scalar remainder."
);
avx512_f64_binop_masked!(
	add_f64x8_merge_masked, add_f64x8_zero_masked, mask_add_pd_intrinsic, maskz_add_pd_intrinsic,
	_mm512_mask_add_pd, _mm512_maskz_add_pd,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddpd`, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddpd`, zero-masked)."
);
avx512_f64_binop_masked!(
	sub_f64x8_merge_masked, sub_f64x8_zero_masked, mask_sub_pd_intrinsic, maskz_sub_pd_intrinsic,
	_mm512_mask_sub_pd, _mm512_maskz_sub_pd,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubpd`, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubpd`, zero-masked)."
);
avx512_f64_binop_masked!(
	mul_f64x8_merge_masked, mul_f64x8_zero_masked, mask_mul_pd_intrinsic, maskz_mul_pd_intrinsic,
	_mm512_mask_mul_pd, _mm512_maskz_mul_pd,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulpd`, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulpd`, zero-masked)."
);
avx512_f64_binop_masked!(
	div_f64x8_merge_masked, div_f64x8_zero_masked, mask_div_pd_intrinsic, maskz_div_pd_intrinsic,
	_mm512_mask_div_pd, _mm512_maskz_div_pd,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivpd`, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivpd`, zero-masked)."
);
avx512_f64_binop_masked!(
	min_f64x8_merge_masked, min_f64x8_zero_masked, mask_min_pd_intrinsic, maskz_min_pd_intrinsic,
	_mm512_mask_min_pd, _mm512_maskz_min_pd,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminpd`, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminpd`, zero-masked)."
);
avx512_f64_binop_masked!(
	max_f64x8_merge_masked, max_f64x8_zero_masked, mask_max_pd_intrinsic, maskz_max_pd_intrinsic,
	_mm512_mask_max_pd, _mm512_maskz_max_pd,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxpd`, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxpd`, zero-masked)."
);

avx512_f64_binop!(
	and_f64x8, and_f64_slice, andpd, and_pd_bitcast, |x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()),
	"`a & b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f64::from_bits(a[i].to_bits() & b[i].to_bits())`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	or_f64x8, or_f64_slice, orpd, or_pd_bitcast, |x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()),
	"`a | b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f64::from_bits(a[i].to_bits() | b[i].to_bits())`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	xor_f64x8, xor_f64_slice, xorpd, xor_pd_bitcast, |x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()),
	"`a ^ b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f64::from_bits(a[i].to_bits() ^ b[i].to_bits())`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	andnot_f64x8, andnot_f64_slice, andnpd, andnot_pd_bitcast, |x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()),
	"`!a & b` per lane, bitwise via `si512` bitcast (no AVX-512DQ needed for this).",
	"`out[i] = f64::from_bits(!a[i].to_bits() & b[i].to_bits())`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	cmpeq_f64x8, cmpeq_f64_slice, vcmpeqpd, vcmpeq_pd512,
	|x, y| if x == y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane equality mask (`vcmppd` EQ_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	cmplt_f64x8, cmplt_f64_slice, vcmpltpd, vcmplt_pd512,
	|x, y| if x < y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-than mask (`vcmppd` LT_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	cmple_f64x8, cmple_f64_slice, vcmplepd, vcmple_pd512,
	|x, y| if x <= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-equal mask (`vcmppd` LE_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	cmpgt_f64x8, cmpgt_f64_slice, vcmpgtpd, vcmpgt_pd512,
	|x, y| if x > y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-than mask (`vcmppd` GT_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx512_f64_binop!(
	cmpge_f64x8, cmpge_f64_slice, vcmpgepd, vcmpge_pd512,
	|x, y| if x >= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-equal mask (`vcmppd` GE_OQ via k-mask, 512-bit).",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 8-wide chunks, scalar remainder."
);

// Fixed-width only (no `_slice`/`auto`), see `simd_unop_fixed` doc.
// `rcp14`/`rsqrt14` are a distinct, more precise (<=2^-14 relative error per
// SDM) mnemonic/instruction from SSE/AVX's `rcpps`/`rsqrtps` (<1.5*2^-12) -
// named accordingly, not `rcp_f32x16`/`rsqrt_f32x16`.
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = sqrt_f32x16, intrinsic_fn = sqrtps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_sqrt_ps,
	fixed_doc = "Correctly-rounded per-lane sqrt (`vsqrtps`, 512-bit). Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = sqrt_f64x8, intrinsic_fn = sqrtpd512,
	width = 8, elem = f64, vec = __m512d,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd, intrinsic = _mm512_sqrt_pd,
	fixed_doc = "Correctly-rounded per-lane sqrt (`vsqrtpd`, 512-bit). Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = rcp14_f32x16, intrinsic_fn = rcp14ps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_rcp14_ps,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14ps`, 512-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = rcp14_f64x8, intrinsic_fn = rcp14pd512,
	width = 8, elem = f64, vec = __m512d,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd, intrinsic = _mm512_rcp14_pd,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14pd`, 512-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = rsqrt14_f32x16, intrinsic_fn = rsqrt14ps512,
	width = 16, elem = f32, vec = __m512,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_rsqrt14_ps,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14ps`, 512-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = rsqrt14_f64x8, intrinsic_fn = rsqrt14pd512,
	width = 8, elem = f64, vec = __m512d,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd, intrinsic = _mm512_rsqrt14_pd,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14pd`, 512-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}

avx512_i32_binop!(
	add_i32x16, add_i32_slice, paddd, _mm512_add_epi32, |x: i32, y: i32| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddd`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide `add_i32x16` chunks, scalar remainder."
);
avx512_i32_binop!(
	sub_i32x16, sub_i32_slice, psubd, _mm512_sub_epi32, |x: i32, y: i32| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubd`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide `sub_i32x16` chunks, scalar remainder."
);
avx512_i32_binop!(
	mul_i32x16, mul_i32_slice, pmulld, _mm512_mullo_epi32, |x: i32, y: i32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`vpmulld`, 512-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 16-wide `mul_i32x16` chunks, scalar remainder."
);
avx512_i32_binop!(
	min_i32x16, min_i32_slice, pminsd, _mm512_min_epi32, |x, y| x.min(y),
	"Per-lane signed min (`vpminsd`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 16-wide `min_i32x16` chunks, scalar remainder."
);
avx512_i32_binop!(
	max_i32x16, max_i32_slice, pmaxsd, _mm512_max_epi32, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsd`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 16-wide `max_i32x16` chunks, scalar remainder."
);
avx512_i32_binop_masked!(
	add_i32x16_merge_masked, add_i32x16_zero_masked, mask_add_epi32_intrinsic, maskz_add_epi32_intrinsic,
	_mm512_mask_add_epi32, _mm512_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, zero-masked)."
);
avx512_i32_binop_masked!(
	sub_i32x16_merge_masked, sub_i32x16_zero_masked, mask_sub_epi32_intrinsic, maskz_sub_epi32_intrinsic,
	_mm512_mask_sub_epi32, _mm512_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, zero-masked)."
);
avx512_i32_binop_masked!(
	mul_i32x16_merge_masked, mul_i32x16_zero_masked, mask_mullo_epi32_intrinsic, maskz_mullo_epi32_intrinsic,
	_mm512_mask_mullo_epi32, _mm512_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, merge-masked).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, zero-masked)."
);
avx512_i32_binop_masked!(
	min_i32x16_merge_masked, min_i32x16_zero_masked, mask_min_epi32_intrinsic, maskz_min_epi32_intrinsic,
	_mm512_mask_min_epi32, _mm512_maskz_min_epi32,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsd`, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsd`, zero-masked)."
);
avx512_i32_binop_masked!(
	max_i32x16_merge_masked, max_i32x16_zero_masked, mask_max_epi32_intrinsic, maskz_max_epi32_intrinsic,
	_mm512_mask_max_epi32, _mm512_maskz_max_epi32,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsd`, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsd`, zero-masked)."
);

scalar_only_binop! {
	token = Avx512f,
	fixed_fn = div_i32x16, slice_fn = div_i32_slice,
	width = 16, elem = i32,
	scalar = |x: i32, y: i32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor or `i32::MIN / -1`, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

avx512_u32_binop!(
	add_u32x16, add_u32_slice, paddd_u, _mm512_add_epi32, |x: u32, y: u32| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddd`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide `add_u32x16` chunks, scalar remainder."
);
avx512_u32_binop!(
	sub_u32x16, sub_u32_slice, psubd_u, _mm512_sub_epi32, |x: u32, y: u32| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubd`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide `sub_u32x16` chunks, scalar remainder."
);
avx512_u32_binop!(
	mul_u32x16, mul_u32_slice, pmulld_u, _mm512_mullo_epi32, |x: u32, y: u32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`vpmulld`, 512-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 16-wide `mul_u32x16` chunks, scalar remainder."
);
avx512_u32_binop!(
	min_u32x16, min_u32_slice, pminud, _mm512_min_epu32, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminud`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 16-wide `min_u32x16` chunks, scalar remainder."
);
avx512_u32_binop!(
	max_u32x16, max_u32_slice, pmaxud, _mm512_max_epu32, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxud`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 16-wide `max_u32x16` chunks, scalar remainder."
);
avx512_u32_binop_masked!(
	add_u32x16_merge_masked, add_u32x16_zero_masked, mask_add_epu32_intrinsic, maskz_add_epu32_intrinsic,
	_mm512_mask_add_epi32, _mm512_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, zero-masked)."
);
avx512_u32_binop_masked!(
	sub_u32x16_merge_masked, sub_u32x16_zero_masked, mask_sub_epu32_intrinsic, maskz_sub_epu32_intrinsic,
	_mm512_mask_sub_epi32, _mm512_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, zero-masked)."
);
avx512_u32_binop_masked!(
	mul_u32x16_merge_masked, mul_u32x16_zero_masked, mask_mullo_epu32_intrinsic, maskz_mullo_epu32_intrinsic,
	_mm512_mask_mullo_epi32, _mm512_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, merge-masked).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, zero-masked)."
);
avx512_u32_binop_masked!(
	min_u32x16_merge_masked, min_u32x16_zero_masked, mask_min_epu32_intrinsic, maskz_min_epu32_intrinsic,
	_mm512_mask_min_epu32, _mm512_maskz_min_epu32,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminud`, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminud`, zero-masked)."
);
avx512_u32_binop_masked!(
	max_u32x16_merge_masked, max_u32x16_zero_masked, mask_max_epu32_intrinsic, maskz_max_epu32_intrinsic,
	_mm512_mask_max_epu32, _mm512_maskz_max_epu32,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxud`, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxud`, zero-masked)."
);

scalar_only_binop! {
	token = Avx512f,
	fixed_fn = div_u32x16, slice_fn = div_u32_slice,
	width = 16, elem = u32,
	scalar = |x: u32, y: u32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

// i64/u64: add/sub also at SSE2/AVX2; min/max 512-only here (no VL token).
macro_rules! avx512_i64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_i64_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = i64, vec = __m512i, mask = u8,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512_u64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u64_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u64, vec = __m512i, mask = u8,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512_i64_binop!(
	add_i64x8, add_i64_slice, paddq, _mm512_add_epi64, |x: i64, y: i64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddq`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide `add_i64x8` chunks, scalar remainder."
);
avx512_i64_binop!(
	sub_i64x8, sub_i64_slice, psubq, _mm512_sub_epi64, |x: i64, y: i64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubq`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide `sub_i64x8` chunks, scalar remainder."
);
avx512_i64_binop!(
	min_i64x8, min_i64_slice, pminsq, _mm512_min_epi64, |x, y| x.min(y),
	"Per-lane signed min (`vpminsq`, 512-bit). No pre-AVX-512 form.",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_i64x8` chunks, scalar remainder."
);
avx512_i64_binop!(
	max_i64x8, max_i64_slice, pmaxsq, _mm512_max_epi64, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsq`, 512-bit). No pre-AVX-512 form.",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_i64x8` chunks, scalar remainder."
);

avx512_i64_binop_masked!(
	add_i64x8_merge_masked, add_i64x8_zero_masked, mask_add_epi64_intrinsic, maskz_add_epi64_intrinsic,
	_mm512_mask_add_epi64, _mm512_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, zero-masked)."
);
avx512_i64_binop_masked!(
	sub_i64x8_merge_masked, sub_i64x8_zero_masked, mask_sub_epi64_intrinsic, maskz_sub_epi64_intrinsic,
	_mm512_mask_sub_epi64, _mm512_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, zero-masked)."
);
avx512_i64_binop_masked!(
	min_i64x8_merge_masked, min_i64x8_zero_masked, mask_min_epi64_intrinsic, maskz_min_epi64_intrinsic,
	_mm512_mask_min_epi64, _mm512_maskz_min_epi64,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsq`, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsq`, zero-masked)."
);
avx512_i64_binop_masked!(
	max_i64x8_merge_masked, max_i64x8_zero_masked, mask_max_epi64_intrinsic, maskz_max_epi64_intrinsic,
	_mm512_mask_max_epi64, _mm512_maskz_max_epi64,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsq`, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsq`, zero-masked)."
);

avx512_u64_binop!(
	add_u64x8, add_u64_slice, paddq_u, _mm512_add_epi64, |x: u64, y: u64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddq`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide `add_u64x8` chunks, scalar remainder."
);
avx512_u64_binop!(
	sub_u64x8, sub_u64_slice, psubq_u, _mm512_sub_epi64, |x: u64, y: u64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubq`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide `sub_u64x8` chunks, scalar remainder."
);
avx512_u64_binop!(
	min_u64x8, min_u64_slice, pminuq, _mm512_min_epu64, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminuq`, 512-bit). No pre-AVX-512 form.",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_u64x8` chunks, scalar remainder."
);
avx512_u64_binop!(
	max_u64x8, max_u64_slice, pmaxuq, _mm512_max_epu64, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxuq`, 512-bit). No pre-AVX-512 form.",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_u64x8` chunks, scalar remainder."
);
// i64/u64 bitwise: same si512 ops as i32 (view-only).
avx512_u64_binop_masked!(
	add_u64x8_merge_masked, add_u64x8_zero_masked, mask_add_epu64_intrinsic, maskz_add_epu64_intrinsic,
	_mm512_mask_add_epi64, _mm512_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, zero-masked)."
);
avx512_u64_binop_masked!(
	sub_u64x8_merge_masked, sub_u64x8_zero_masked, mask_sub_epu64_intrinsic, maskz_sub_epu64_intrinsic,
	_mm512_mask_sub_epi64, _mm512_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, zero-masked)."
);
avx512_u64_binop_masked!(
	min_u64x8_merge_masked, min_u64x8_zero_masked, mask_min_epu64_intrinsic, maskz_min_epu64_intrinsic,
	_mm512_mask_min_epu64, _mm512_maskz_min_epu64,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuq`, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuq`, zero-masked)."
);
avx512_u64_binop_masked!(
	max_u64x8_merge_masked, max_u64x8_zero_masked, mask_max_epu64_intrinsic, maskz_max_epu64_intrinsic,
	_mm512_mask_max_epu64, _mm512_maskz_max_epu64,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuq`, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuq`, zero-masked)."
);

avx512_i64_binop!(
	and_i64x8, and_i64_slice, vpandq, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandq`, 512-bit).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
avx512_i64_binop!(
	or_i64x8, or_i64_slice, vporq, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vporq`, 512-bit).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder."
);
avx512_i64_binop!(
	xor_i64x8, xor_i64_slice, vpxorq, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxorq`, 512-bit).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder."
);
avx512_i64_binop!(
	andnot_i64x8, andnot_i64_slice, vpandnq, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnq`, 512-bit).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	and_u64x8, and_u64_slice, vpandq_u, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandq`, 512-bit).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	or_u64x8, or_u64_slice, vporq_u, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vporq`, 512-bit).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	xor_u64x8, xor_u64_slice, vpxorq_u, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxorq`, 512-bit).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	andnot_u64x8, andnot_u64_slice, vpandnq_u, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnq`, 512-bit).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder."
);

simd_unop! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = abs_i32x16, slice_fn = abs_i32_slice, intrinsic_fn = pabsd,
	width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_abs_epi32, scalar = |x: i32| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsd`, 512-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 16-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = abs_i64x8, slice_fn = abs_i64_slice, intrinsic_fn = pabsq,
	width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_abs_epi64, scalar = |x: i64| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsq`, 512-bit; native despite no pre-AVX-512 form).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 8-wide chunks, scalar remainder.",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = abs_i32x16_merge_masked, zero_fn = abs_i32x16_zero_masked,
	merge_intrinsic_fn = mask_abs_epi32_intrinsic, zero_intrinsic_fn = maskz_abs_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_abs_epi32, zero_intrinsic = _mm512_maskz_abs_epi32,
	merge_doc = "Per-lane absolute value where `mask` bit is set, else copied from `src` (`vpabsd`, merge-masked).",
	zero_doc = "Per-lane absolute value where `mask` bit is set, else zero (`vpabsd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = abs_i64x8_merge_masked, zero_fn = abs_i64x8_zero_masked,
	merge_intrinsic_fn = mask_abs_epi64_intrinsic, zero_intrinsic_fn = maskz_abs_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_abs_epi64, zero_intrinsic = _mm512_maskz_abs_epi64,
	merge_doc = "Per-lane absolute value where `mask` bit is set, else copied from `src` (`vpabsq`, merge-masked).",
	zero_doc = "Per-lane absolute value where `mask` bit is set, else zero (`vpabsq`, zero-masked).",
}

// compress/expand exist only as masked ops: no unmasked base op in the ISA.
// `epi32`/`epi64` are reused bit-identically for `u32`/`u64` (no `epu32`/
// `epu64` compress/expand intrinsic exists: lane selection is bit-pattern
// only, doesn't care about signedness).

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_i32x16_merge_masked, zero_fn = compress_i32x16_zero_masked,
	merge_intrinsic_fn = mask_compress_epi32_intrinsic, zero_intrinsic_fn = maskz_compress_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi32, zero_intrinsic = _mm512_maskz_compress_epi32,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressd`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_u32x16_merge_masked, zero_fn = compress_u32x16_zero_masked,
	merge_intrinsic_fn = mask_compress_epu32_intrinsic, zero_intrinsic_fn = maskz_compress_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi32, zero_intrinsic = _mm512_maskz_compress_epi32,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressd`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_i64x8_merge_masked, zero_fn = compress_i64x8_zero_masked,
	merge_intrinsic_fn = mask_compress_epi64_intrinsic, zero_intrinsic_fn = maskz_compress_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi64, zero_intrinsic = _mm512_maskz_compress_epi64,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressq`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressq`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_u64x8_merge_masked, zero_fn = compress_u64x8_zero_masked,
	merge_intrinsic_fn = mask_compress_epu64_intrinsic, zero_intrinsic_fn = maskz_compress_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi64, zero_intrinsic = _mm512_maskz_compress_epi64,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressq`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressq`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_f32x16_merge_masked, zero_fn = compress_f32x16_zero_masked,
	merge_intrinsic_fn = mask_compress_ps_intrinsic, zero_intrinsic_fn = maskz_compress_ps_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_compress_ps, zero_intrinsic = _mm512_maskz_compress_ps,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vcompressps`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vcompressps`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = compress_f64x8_merge_masked, zero_fn = compress_f64x8_zero_masked,
	merge_intrinsic_fn = mask_compress_pd_intrinsic, zero_intrinsic_fn = maskz_compress_pd_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_compress_pd, zero_intrinsic = _mm512_maskz_compress_pd,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vcompresspd`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vcompresspd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_i32x16_merge_masked, zero_fn = expand_i32x16_zero_masked,
	merge_intrinsic_fn = mask_expand_epi32_intrinsic, zero_intrinsic_fn = maskz_expand_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi32, zero_intrinsic = _mm512_maskz_expand_epi32,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandd`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_u32x16_merge_masked, zero_fn = expand_u32x16_zero_masked,
	merge_intrinsic_fn = mask_expand_epu32_intrinsic, zero_intrinsic_fn = maskz_expand_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi32, zero_intrinsic = _mm512_maskz_expand_epi32,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandd`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_i64x8_merge_masked, zero_fn = expand_i64x8_zero_masked,
	merge_intrinsic_fn = mask_expand_epi64_intrinsic, zero_intrinsic_fn = maskz_expand_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi64, zero_intrinsic = _mm512_maskz_expand_epi64,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandq`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandq`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_u64x8_merge_masked, zero_fn = expand_u64x8_zero_masked,
	merge_intrinsic_fn = mask_expand_epu64_intrinsic, zero_intrinsic_fn = maskz_expand_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi64, zero_intrinsic = _mm512_maskz_expand_epi64,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandq`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandq`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_f32x16_merge_masked, zero_fn = expand_f32x16_zero_masked,
	merge_intrinsic_fn = mask_expand_ps_intrinsic, zero_intrinsic_fn = maskz_expand_ps_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_expand_ps, zero_intrinsic = _mm512_maskz_expand_ps,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vexpandps`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vexpandps`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expand_f64x8_merge_masked, zero_fn = expand_f64x8_zero_masked,
	merge_intrinsic_fn = mask_expand_pd_intrinsic, zero_intrinsic_fn = maskz_expand_pd_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_expand_pd, zero_intrinsic = _mm512_maskz_expand_pd,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vexpandpd`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vexpandpd`, zero-masked).",
}

// Memory forms of the same two ops: `compressstoreu` writes only the selected
// lanes (no merge/zero split: unselected lanes produce no store), while
// `expandloadu` reads only as many elements as the mask selects. Both are
// pointer-based in the ISA; the safe wrappers bound them with a popcount
// length assert.
simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_i32x16, intrinsic_fn = compressstoreu_i32x16_intrinsic,
	width = 16, elem = i32, ptr_elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi32,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressd`, store form).",
}

simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_u32x16, intrinsic_fn = compressstoreu_u32x16_intrinsic,
	width = 16, elem = u32, ptr_elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi32,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressd`, store form).",
}

simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_i64x8, intrinsic_fn = compressstoreu_i64x8_intrinsic,
	width = 8, elem = i64, ptr_elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi64,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressq`, store form).",
}

simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_u64x8, intrinsic_fn = compressstoreu_u64x8_intrinsic,
	width = 8, elem = u64, ptr_elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi64,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressq`, store form).",
}

simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_f32x16, intrinsic_fn = compressstoreu_f32x16_intrinsic,
	width = 16, elem = f32, ptr_elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, intrinsic = _mm512_mask_compressstoreu_ps,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vcompressps`, store form).",
}

simd_compressstoreu! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = compressstoreu_f64x8, intrinsic_fn = compressstoreu_f64x8_intrinsic,
	width = 8, elem = f64, ptr_elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, intrinsic = _mm512_mask_compressstoreu_pd,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vcompresspd`, store form).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_i32x16_merge_masked, zero_fn = expandloadu_i32x16_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_i32_intrinsic, zero_intrinsic_fn = maskz_expandloadu_i32_intrinsic,
	width = 16, elem = i32, ptr_elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi32, zero_intrinsic = _mm512_maskz_expandloadu_epi32,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandd`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandd`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_u32x16_merge_masked, zero_fn = expandloadu_u32x16_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_u32_intrinsic, zero_intrinsic_fn = maskz_expandloadu_u32_intrinsic,
	width = 16, elem = u32, ptr_elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi32, zero_intrinsic = _mm512_maskz_expandloadu_epi32,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandd`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandd`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_i64x8_merge_masked, zero_fn = expandloadu_i64x8_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_i64_intrinsic, zero_intrinsic_fn = maskz_expandloadu_i64_intrinsic,
	width = 8, elem = i64, ptr_elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi64, zero_intrinsic = _mm512_maskz_expandloadu_epi64,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandq`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandq`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_u64x8_merge_masked, zero_fn = expandloadu_u64x8_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_u64_intrinsic, zero_intrinsic_fn = maskz_expandloadu_u64_intrinsic,
	width = 8, elem = u64, ptr_elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi64, zero_intrinsic = _mm512_maskz_expandloadu_epi64,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandq`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandq`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_f32x16_merge_masked, zero_fn = expandloadu_f32x16_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_f32_intrinsic, zero_intrinsic_fn = maskz_expandloadu_f32_intrinsic,
	width = 16, elem = f32, ptr_elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_expandloadu_ps, zero_intrinsic = _mm512_maskz_expandloadu_ps,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vexpandps`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vexpandps`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512f, target_feature = "avx512f",
	merge_fn = expandloadu_f64x8_merge_masked, zero_fn = expandloadu_f64x8_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_f64_intrinsic, zero_intrinsic_fn = maskz_expandloadu_f64_intrinsic,
	width = 8, elem = f64, ptr_elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_expandloadu_pd, zero_intrinsic = _mm512_maskz_expandloadu_pd,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vexpandpd`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vexpandpd`, load form, zero-masked).",
}

// Bitwise ternary logic (`vpternlogd`/`vpternlogq`): a genuinely new capability,
// not a masked variant of an existing unmasked op: unlike compress/expand above,
// there is no non-AVX-512 hardware equivalent at all (not even an unmasked-only
// wrapper on Sse/Avx2), so both the unmasked and masked forms are new here.
// `epi32`/`epi64` reused bit-identically for `u32`/`u64` (purely bitwise, no
// `epu32`/`epu64` intrinsic exists).

simd_ternarylogic! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = ternarylogic_i32x16, merge_fn = ternarylogic_i32x16_merge_masked, zero_fn = ternarylogic_i32x16_zero_masked,
	intrinsic_fn = ternarylogic_epi32_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi32_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_ternarylogic_epi32, merge_intrinsic = _mm512_mask_ternarylogic_epi32, zero_intrinsic = _mm512_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 512-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = ternarylogic_u32x16, merge_fn = ternarylogic_u32x16_merge_masked, zero_fn = ternarylogic_u32x16_zero_masked,
	intrinsic_fn = ternarylogic_epu32_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu32_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_ternarylogic_epi32, merge_intrinsic = _mm512_mask_ternarylogic_epi32, zero_intrinsic = _mm512_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 512-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = ternarylogic_i64x8, merge_fn = ternarylogic_i64x8_merge_masked, zero_fn = ternarylogic_i64x8_zero_masked,
	intrinsic_fn = ternarylogic_epi64_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi64_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_ternarylogic_epi64, merge_intrinsic = _mm512_mask_ternarylogic_epi64, zero_intrinsic = _mm512_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 512-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = ternarylogic_u64x8, merge_fn = ternarylogic_u64x8_merge_masked, zero_fn = ternarylogic_u64x8_zero_masked,
	intrinsic_fn = ternarylogic_epu64_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu64_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_ternarylogic_epi64, merge_intrinsic = _mm512_mask_ternarylogic_epi64, zero_intrinsic = _mm512_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 512-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, zero-masked).",
}

avx512_i32_binop!(
	and_i32x16, and_i32_slice, vpandd, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
avx512_i32_binop!(
	or_i32x16, or_i32_slice, vpord, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
avx512_i32_binop!(
	xor_i32x16, xor_i32_slice, vpxord, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
avx512_i32_binop!(
	andnot_i32x16, andnot_i32_slice, vpandnd, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);

avx512_u32_binop!(
	and_u32x16, and_u32_slice, vpandd_u, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
avx512_u32_binop!(
	or_u32x16, or_u32_slice, vpord_u, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
avx512_u32_binop!(
	xor_u32x16, xor_u32_slice, vpxord_u, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
avx512_u32_binop!(
	andnot_u32x16, andnot_u32_slice, vpandnd_u, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);

// Narrow bitwise: same si512 ops as i32 (no BW needed); feeds auto_up cascade.
macro_rules! avx512_i8_bitop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 64, elem = i8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u8_bitop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_i16_bitop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u16_bitop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512_i8_bitop!(and_i8x64, and_i8_slice, vpandd_i8, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).", "`out[i] = a[i] & b[i]`. 64-wide chunks, scalar remainder.");
avx512_i8_bitop!(or_i8x64, or_i8_slice, vpord_i8, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).", "`out[i] = a[i] | b[i]`. 64-wide chunks, scalar remainder.");
avx512_i8_bitop!(xor_i8x64, xor_i8_slice, vpxord_i8, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).", "`out[i] = a[i] ^ b[i]`. 64-wide chunks, scalar remainder.");
avx512_i8_bitop!(andnot_i8x64, andnot_i8_slice, vpandnd_i8, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).", "`out[i] = !a[i] & b[i]`. 64-wide chunks, scalar remainder.");

avx512_u8_bitop!(and_u8x64, and_u8_slice, vpandd_u8, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).", "`out[i] = a[i] & b[i]`. 64-wide chunks, scalar remainder.");
avx512_u8_bitop!(or_u8x64, or_u8_slice, vpord_u8, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).", "`out[i] = a[i] | b[i]`. 64-wide chunks, scalar remainder.");
avx512_u8_bitop!(xor_u8x64, xor_u8_slice, vpxord_u8, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).", "`out[i] = a[i] ^ b[i]`. 64-wide chunks, scalar remainder.");
avx512_u8_bitop!(andnot_u8x64, andnot_u8_slice, vpandnd_u8, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).", "`out[i] = !a[i] & b[i]`. 64-wide chunks, scalar remainder.");

avx512_i16_bitop!(and_i16x32, and_i16_slice, vpandd_i16, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).", "`out[i] = a[i] & b[i]`. 32-wide chunks, scalar remainder.");
avx512_i16_bitop!(or_i16x32, or_i16_slice, vpord_i16, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).", "`out[i] = a[i] | b[i]`. 32-wide chunks, scalar remainder.");
avx512_i16_bitop!(xor_i16x32, xor_i16_slice, vpxord_i16, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).", "`out[i] = a[i] ^ b[i]`. 32-wide chunks, scalar remainder.");
avx512_i16_bitop!(andnot_i16x32, andnot_i16_slice, vpandnd_i16, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).", "`out[i] = !a[i] & b[i]`. 32-wide chunks, scalar remainder.");

avx512_u16_bitop!(and_u16x32, and_u16_slice, vpandd_u16, _mm512_and_si512, |x, y| x & y,
	"`a & b` per lane (`vpandd`, 512-bit).", "`out[i] = a[i] & b[i]`. 32-wide chunks, scalar remainder.");
avx512_u16_bitop!(or_u16x32, or_u16_slice, vpord_u16, _mm512_or_si512, |x, y| x | y,
	"`a | b` per lane (`vpord`, 512-bit).", "`out[i] = a[i] | b[i]`. 32-wide chunks, scalar remainder.");
avx512_u16_bitop!(xor_u16x32, xor_u16_slice, vpxord_u16, _mm512_xor_si512, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxord`, 512-bit).", "`out[i] = a[i] ^ b[i]`. 32-wide chunks, scalar remainder.");
avx512_u16_bitop!(andnot_u16x32, andnot_u16_slice, vpandnd_u16, _mm512_andnot_si512, |x, y| !x & y,
	"`!a & b` per lane (`vpandnd`, 512-bit).", "`out[i] = !a[i] & b[i]`. 32-wide chunks, scalar remainder.");

// AVX-512 compare returns a k-mask, not all-0/1 lanes; expand with maskz_set1.
impl Avx512f {
	/// Lane equality mask: all-1s if equal, else 0 (`vpcmpeqd` via k-mask).
	#[inline]
	pub fn cmpeq_i32x16(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
		unsafe { vpcmpeqd_i32(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [i32; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_i32x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { -1 } else { 0 };
		}
	}

	/// Lane equality mask as `u32` all-1s / 0.
	#[inline]
	pub fn cmpeq_u32x16(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
		let ai: [i32; 16] = core::array::from_fn(|i| a[i] as i32);
		let bi: [i32; 16] = core::array::from_fn(|i| b[i] as i32);
		let r = self.cmpeq_i32x16(ai, bi);
		core::array::from_fn(|i| r[i] as u32)
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_u32x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { !0 } else { 0 };
		}
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpeqd_i32(a: &[i32; 16], b: &[i32; 16]) -> [i32; 16] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpeq_epi32_mask(va, vb);
		let vr = _mm512_maskz_set1_epi32(k, -1);
		let mut out = [0i32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

// cmpgt: same k-mask expand as cmpeq; unsigned is native (no sign-bit flip).
impl Avx512f {
	/// Lane greater-than mask: all-1s if `a>b`, else 0 (`vpcmpgtd` via k-mask).
	#[inline]
	pub fn cmpgt_i32x16(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
		unsafe { vpcmpgtd_i32(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [i32; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_i32x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { -1 } else { 0 };
		}
	}

	/// Unsigned greater-than mask (`vpcmpgtud` via k-mask; native, no flip).
	#[inline]
	pub fn cmpgt_u32x16(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
		unsafe { vpcmpgtud_u32(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u32x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i32x16`].
	#[inline]
	pub fn cmplt_i32x16(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
		self.cmpgt_i32x16(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Avx512f::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i32x16`].
	#[inline]
	pub fn cmple_i32x16(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
		let gt = self.cmpgt_i32x16(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Avx512f::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i32x16`].
	#[inline]
	pub fn cmpge_i32x16(self, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
		let lt = self.cmplt_i32x16(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Avx512f::cmplt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmplt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u32x16`].
	#[inline]
	pub fn cmplt_u32x16(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
		self.cmpgt_u32x16(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmpgt_u32_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u32x16`].
	#[inline]
	pub fn cmple_u32x16(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
		let gt = self.cmpgt_u32x16(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmpgt_u32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u32x16`].
	#[inline]
	pub fn cmpge_u32x16(self, a: [u32; 16], b: [u32; 16]) -> [u32; 16] {
		let lt = self.cmplt_u32x16(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmplt_u32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpgtd_i32(a: &[i32; 16], b: &[i32; 16]) -> [i32; 16] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epi32_mask(va, vb);
		let vr = _mm512_maskz_set1_epi32(k, -1);
		let mut out = [0i32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpgtud_u32(a: &[u32; 16], b: &[u32; 16]) -> [u32; 16] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epu32_mask(va, vb);
		let vr = _mm512_maskz_set1_epi32(k, -1);
		let mut out = [0u32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

// select: k-mask via test_epi32_mask on all-0/1 lane mask (same public shape as blendv tiers).
impl Avx512f {
	/// Per-lane select (`vptestmd` + `vpblendmd`). `mask`: all-0/1 (e.g. cmpeq/cmpgt).
	#[inline]
	pub fn select_i32x16(self, a: [i32; 16], b: [i32; 16], mask: [i32; 16]) -> [i32; 16] {
		unsafe { vpblendmd_i32(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_i32_slice(self, a: &[i32], b: &[i32], mask: &[i32], out: &mut [i32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let mask_chunks = mask.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [i32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [i32; 16] = bc.try_into().expect("chunks_exact width");
			let mv: [i32; 16] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_i32x16(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select, `u32` view of [`Avx512f::select_i32x16`].
	#[inline]
	pub fn select_u32x16(self, a: [u32; 16], b: [u32; 16], mask: [u32; 16]) -> [u32; 16] {
		let ai: [i32; 16] = core::array::from_fn(|i| a[i] as i32);
		let bi: [i32; 16] = core::array::from_fn(|i| b[i] as i32);
		let mi: [i32; 16] = core::array::from_fn(|i| mask[i] as i32);
		let r = self.select_i32x16(ai, bi, mi);
		core::array::from_fn(|i| r[i] as u32)
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u32` view).
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_u32_slice(self, a: &[u32], b: &[u32], mask: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let mask_chunks = mask.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 16] = bc.try_into().expect("chunks_exact width");
			let mv: [u32; 16] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_u32x16(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select: `mask` sign bit picks `b`, else `a` (same as `Sse41::blend_f32x4`).
	#[inline]
	pub fn select_f32x16(self, a: [f32; 16], b: [f32; 16], mask: [f32; 16]) -> [f32; 16] {
		unsafe { vpblendmps_f32(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_f32_slice(self, a: &[f32], b: &[f32], mask: &[f32], out: &mut [f32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let mask_chunks = mask.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [f32; 16] = bc.try_into().expect("chunks_exact width");
			let mv: [f32; 16] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_f32x16(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m.is_sign_negative() { y } else { x };
		}
	}

	/// Per-lane select (`vptestmq` + `vpblendmq`). `mask`: all-0/1 (e.g. cmpeq/cmpgt i64).
	#[inline]
	pub fn select_i64x8(self, a: [i64; 8], b: [i64; 8], mask: [i64; 8]) -> [i64; 8] {
		unsafe { vpblendmq_i64(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_i64_slice(self, a: &[i64], b: &[i64], mask: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let mask_chunks = mask.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 8] = bc.try_into().expect("chunks_exact width");
			let mv: [i64; 8] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_i64x8(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select, `u64` view of [`Avx512f::select_i64x8`].
	#[inline]
	pub fn select_u64x8(self, a: [u64; 8], b: [u64; 8], mask: [u64; 8]) -> [u64; 8] {
		let ai: [i64; 8] = core::array::from_fn(|i| a[i] as i64);
		let bi: [i64; 8] = core::array::from_fn(|i| b[i] as i64);
		let mi: [i64; 8] = core::array::from_fn(|i| mask[i] as i64);
		let r = self.select_i64x8(ai, bi, mi);
		core::array::from_fn(|i| r[i] as u64)
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u64` view).
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_u64_slice(self, a: &[u64], b: &[u64], mask: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let mask_chunks = mask.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 8] = bc.try_into().expect("chunks_exact width");
			let mv: [u64; 8] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_u64x8(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select: `mask` sign bit picks `b`, else `a` (same domain as SSE4.1 `blendvpd`).
	#[inline]
	pub fn select_f64x8(self, a: [f64; 8], b: [f64; 8], mask: [f64; 8]) -> [f64; 8] {
		unsafe { vpblendmpd_f64(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_f64_slice(self, a: &[f64], b: &[f64], mask: &[f64], out: &mut [f64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let mask_chunks = mask.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [f64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [f64; 8] = bc.try_into().expect("chunks_exact width");
			let mv: [f64; 8] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_f64x8(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m.is_sign_negative() { y } else { x };
		}
	}
}

// i64/u64 compare family (k-mask expand, same shape as i32).
impl Avx512f {
	/// Lane equality mask: all-1s if equal, else 0 (`vpcmpeqq` via k-mask).
	#[inline]
	pub fn cmpeq_i64x8(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
		unsafe { vpcmpeqq_i64(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_i64x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { -1 } else { 0 };
		}
	}

	/// Lane equality mask as `u64` all-1s / 0.
	#[inline]
	pub fn cmpeq_u64x8(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		let ai: [i64; 8] = core::array::from_fn(|i| a[i] as i64);
		let bi: [i64; 8] = core::array::from_fn(|i| b[i] as i64);
		let r = self.cmpeq_i64x8(ai, bi);
		core::array::from_fn(|i| r[i] as u64)
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_u64x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { !0 } else { 0 };
		}
	}

	/// Lane greater-than mask: all-1s if `a>b`, else 0 (`vpcmpgtq` via k-mask).
	#[inline]
	pub fn cmpgt_i64x8(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
		unsafe { vpcmpgtq_i64(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0`. 8-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_i64x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { -1 } else { 0 };
		}
	}

	/// Unsigned greater-than mask (`vpcmpgtuq` via k-mask; native, no flip).
	#[inline]
	pub fn cmpgt_u64x8(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		unsafe { vpcmpgtuq_u64(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u64x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i64x8`].
	#[inline]
	pub fn cmplt_i64x8(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
		self.cmpgt_i64x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Avx512f::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i64x8`].
	#[inline]
	pub fn cmple_i64x8(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
		let gt = self.cmpgt_i64x8(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Avx512f::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i64x8`].
	#[inline]
	pub fn cmpge_i64x8(self, a: [i64; 8], b: [i64; 8]) -> [i64; 8] {
		let lt = self.cmplt_i64x8(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Avx512f::cmplt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmplt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u64x8`].
	#[inline]
	pub fn cmplt_u64x8(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		self.cmpgt_u64x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmpgt_u64_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u64x8`].
	#[inline]
	pub fn cmple_u64x8(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		let gt = self.cmpgt_u64x8(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmpgt_u64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u64x8`].
	#[inline]
	pub fn cmpge_u64x8(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		let lt = self.cmplt_u64x8(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmplt_u64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpeqq_i64(a: &[i64; 8], b: &[i64; 8]) -> [i64; 8] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpeq_epi64_mask(va, vb);
		let vr = _mm512_maskz_set1_epi64(k, -1);
		let mut out = [0i64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpgtq_i64(a: &[i64; 8], b: &[i64; 8]) -> [i64; 8] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epi64_mask(va, vb);
		let vr = _mm512_maskz_set1_epi64(k, -1);
		let mut out = [0i64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpcmpgtuq_u64(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epu64_mask(va, vb);
		let vr = _mm512_maskz_set1_epi64(k, -1);
		let mut out = [0u64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpblendmq_i64(a: &[i64; 8], b: &[i64; 8], mask: &[i64; 8]) -> [i64; 8] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let vm = _mm512_loadu_si512(mask.as_ptr().cast());
		let k = _mm512_test_epi64_mask(vm, vm);
		let vr = _mm512_mask_blend_epi64(k, va, vb);
		let mut out = [0i64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpblendmpd_f64(a: &[f64; 8], b: &[f64; 8], mask: &[f64; 8]) -> [f64; 8] {
	unsafe {
		let va = _mm512_loadu_pd(a.as_ptr());
		let vb = _mm512_loadu_pd(b.as_ptr());
		let vm_bits = _mm512_loadu_si512(mask.as_ptr().cast());
		let sign_bit = _mm512_set1_epi64(i64::MIN);
		let k = _mm512_test_epi64_mask(vm_bits, sign_bit);
		let vr = _mm512_mask_blend_pd(k, va, vb);
		let mut out = [0f64; 8];
		_mm512_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpblendmd_i32(a: &[i32; 16], b: &[i32; 16], mask: &[i32; 16]) -> [i32; 16] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let vm = _mm512_loadu_si512(mask.as_ptr().cast());
		let k = _mm512_test_epi32_mask(vm, vm);
		let vr = _mm512_mask_blend_epi32(k, va, vb);
		let mut out = [0i32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpblendmps_f32(a: &[f32; 16], b: &[f32; 16], mask: &[f32; 16]) -> [f32; 16] {
	unsafe {
		let va = _mm512_loadu_ps(a.as_ptr());
		let vb = _mm512_loadu_ps(b.as_ptr());
		// Sign bit only (AND i32::MIN + test); plain test_epi32 would be nonzero, not sign.
		// movepi32_mask would need AVX-512DQ.
		let vm_bits = _mm512_loadu_si512(mask.as_ptr().cast());
		let sign_bit = _mm512_set1_epi32(i32::MIN);
		let k = _mm512_test_epi32_mask(vm_bits, sign_bit);
		let vr = _mm512_mask_blend_ps(k, va, vb);
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

macro_rules! avx512_i32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512_i32_shift_imm!(
	shl_i32x16, shl_i32_slice, vpslld, _mm512_sll_epi32, |x: i32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpslld`, 512-bit).",
	"`out[i] = a[i] << IMM`. 16-wide chunks, scalar remainder."
);
avx512_i32_shift_imm!(
	shr_i32x16, shr_i32_slice, vpsrld, _mm512_srl_epi32, |x: i32, imm| ((x as u32).wrapping_shr(imm)) as i32,
	"`a >> IMM` logical per lane (`vpsrld`, 512-bit).",
	"`out[i] = a[i] logical >> IMM`. 16-wide chunks, scalar remainder."
);
avx512_i32_shift_imm!(
	sra_i32x16, sra_i32_slice, vpsrad, _mm512_sra_epi32, |x: i32, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`vpsrad`, 512-bit).",
	"`out[i] = a[i] arithmetic >> IMM`. 16-wide chunks, scalar remainder."
);
avx512_u32_shift_imm!(
	shl_u32x16, shl_u32_slice, vpslld_u, _mm512_sll_epi32, |x: u32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpslld`, 512-bit).",
	"`out[i] = a[i] << IMM`. 16-wide chunks, scalar remainder."
);
avx512_u32_shift_imm!(
	shr_u32x16, shr_u32_slice, vpsrld_u, _mm512_srl_epi32, |x: u32, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`vpsrld`, 512-bit).",
	"`out[i] = a[i] >> IMM`. 16-wide chunks, scalar remainder."
);

// Per-lane variable shift (count is a vector, not a broadcast IMM): plain `simd_binop!`
// fits directly. Same overflow rule as ops/avx/avx2.rs: count>=32 zeroes (sllv/srlv) or
// sign-fills (srav), not Rust's wrapping-count semantics.
macro_rules! avx512_i32_varshift {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_u32_varshift {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512_i32_varshift!(
	sllv_i32x16, sllv_i32_slice, vpsllvd, _mm512_sllv_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector not a broadcast IMM (`vpsllvd`, 512-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] << count[i]`. 16-wide chunks, scalar remainder."
);
avx512_i32_varshift!(
	srlv_i32x16, srlv_i32_slice, vpsrlvd, _mm512_srlv_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { ((x as u32).wrapping_shr(count as u32)) as i32 },
	"`a >> count` logical per lane, `count` a vector not a broadcast IMM (`vpsrlvd`, 512-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] logical >> count[i]`. 16-wide chunks, scalar remainder."
);
avx512_i32_varshift!(
	srav_i32x16, srav_i32_slice, vpsravd, _mm512_srav_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { x >> 31 } else { x.wrapping_shr(count as u32) },
	"`a >> count` arithmetic per lane, `count` a vector not a broadcast IMM (`vpsravd`, 512-bit).",
	"`out[i] = sign-fill if count[i]>=32 else a[i] arithmetic >> count[i]`. 16-wide chunks, scalar remainder."
);
avx512_u32_varshift!(
	sllv_u32x16, sllv_u32_slice, vpsllvd_u, _mm512_sllv_epi32,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shl(count) },
	"`a << count` per lane, `count` a vector not a broadcast IMM (`vpsllvd`, 512-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] << count[i]`. 16-wide chunks, scalar remainder."
);
avx512_u32_varshift!(
	srlv_u32x16, srlv_u32_slice, vpsrlvd_u, _mm512_srlv_epi32,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shr(count) },
	"`a >> count` per lane, `count` a vector not a broadcast IMM (`vpsrlvd`, 512-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] >> count[i]`. 16-wide chunks, scalar remainder."
);
// i64/u64 variable shift: sllv/srlv also on AVX2; srav_i64 is AVX-512F-only (no vpsravq @256).
avx512_i64_binop!(
	sllv_i64x8, sllv_i64_slice, vpsllvq, _mm512_sllv_epi64,
	|x: i64, count: i64| if (count as u64) >= 64 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector (`vpsllvq`, 512-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] << count[i]`. 8-wide chunks, scalar remainder."
);
avx512_i64_binop!(
	srlv_i64x8, srlv_i64_slice, vpsrlvq, _mm512_srlv_epi64,
	|x: i64, count: i64| if (count as u64) >= 64 { 0 } else { ((x as u64).wrapping_shr(count as u32)) as i64 },
	"`a >> count` logical per lane, `count` a vector (`vpsrlvq`, 512-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] logical >> count[i]`. 8-wide chunks, scalar remainder."
);
avx512_i64_binop!(
	srav_i64x8, srav_i64_slice, vpsravq, _mm512_srav_epi64,
	|x: i64, count: i64| if (count as u64) >= 64 { x >> 63 } else { x.wrapping_shr(count as u32) },
	"`a >> count` arithmetic per lane, `count` a vector (`vpsravq`, 512-bit only).",
	"`out[i] = sign-fill if count[i]>=64 else a[i] arithmetic >> count[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	sllv_u64x8, sllv_u64_slice, vpsllvq_u, _mm512_sllv_epi64,
	|x: u64, count: u64| if count >= 64 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector (`vpsllvq`, 512-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] << count[i]`. 8-wide chunks, scalar remainder."
);
avx512_u64_binop!(
	srlv_u64x8, srlv_u64_slice, vpsrlvq_u, _mm512_srlv_epi64,
	|x: u64, count: u64| if count >= 64 { 0 } else { x.wrapping_shr(count as u32) },
	"`a >> count` per lane, `count` a vector (`vpsrlvq`, 512-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] >> count[i]`. 8-wide chunks, scalar remainder."
);

// Bit rotate within each lane (`vprold`/`vprorq`, true IMM8-encoded form -
// distinct from the register-count shifts above). Bit pattern is
// sign-agnostic, so i32/u32 (and i64/u64) share one intrinsic, same as the
// shift family. Fixed-width only, matching `simd_unop_imm`'s other users
// (RANGE/REDUCE): the low 5 (epi32) / 6 (epi64) bits of IMM8 are all
// hardware reads, no scalar-Rust remainder to keep in sync.
macro_rules! avx512_i32_rotate_imm {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_unop_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_u32_rotate_imm {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_unop_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_i64_rotate_imm {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_unop_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_u64_rotate_imm {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $fixed_doc:literal) => {
		simd_unop_imm! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

avx512_i32_rotate_imm!(
	rol_i32x16, rol_i32x16_intrinsic, _mm512_rol_epi32,
	"Rotate each lane's bits left by `IMM8` (mod 32) (`vprold`, 512-bit)."
);
avx512_i32_rotate_imm!(
	ror_i32x16, ror_i32x16_intrinsic, _mm512_ror_epi32,
	"Rotate each lane's bits right by `IMM8` (mod 32) (`vprord`, 512-bit)."
);
avx512_u32_rotate_imm!(
	rol_u32x16, rol_u32x16_intrinsic, _mm512_rol_epi32,
	"Rotate each lane's bits left by `IMM8` (mod 32) (`vprold`, 512-bit)."
);
avx512_u32_rotate_imm!(
	ror_u32x16, ror_u32x16_intrinsic, _mm512_ror_epi32,
	"Rotate each lane's bits right by `IMM8` (mod 32) (`vprord`, 512-bit)."
);
avx512_i64_rotate_imm!(
	rol_i64x8, rol_i64x8_intrinsic, _mm512_rol_epi64,
	"Rotate each lane's bits left by `IMM8` (mod 64) (`vprolq`, 512-bit)."
);
avx512_i64_rotate_imm!(
	ror_i64x8, ror_i64x8_intrinsic, _mm512_ror_epi64,
	"Rotate each lane's bits right by `IMM8` (mod 64) (`vprorq`, 512-bit)."
);
avx512_u64_rotate_imm!(
	rol_u64x8, rol_u64x8_intrinsic, _mm512_rol_epi64,
	"Rotate each lane's bits left by `IMM8` (mod 64) (`vprolq`, 512-bit)."
);
avx512_u64_rotate_imm!(
	ror_u64x8, ror_u64x8_intrinsic, _mm512_ror_epi64,
	"Rotate each lane's bits right by `IMM8` (mod 64) (`vprorq`, 512-bit)."
);

// Cross-lane horizontal reduce (whole register -> one scalar). No single HW
// instruction backs these (compiler-synthesized shuffle+op chain, still
// gated on the feature like everything else here). `u32`/`u64` add/mul
// share the signed intrinsic (wrapping add/mul has an identical bit
// pattern either way); max/min use the real, distinct `epu*` intrinsics.
macro_rules! avx512_reduce_i32 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $out:ty, $fixed_doc:literal) => {
		simd_reduce! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = $elem, out = $out, vec = __m512i, loadu = _mm512_loadu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_reduce_i64 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $out:ty, $fixed_doc:literal) => {
		simd_reduce! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = $elem, out = $out, vec = __m512i, loadu = _mm512_loadu_si512,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_reduce_f32 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_reduce! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = f32, out = f32, vec = __m512, loadu = _mm512_loadu_ps,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

macro_rules! avx512_reduce_f64 {
	($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $fixed_doc:literal) => {
		simd_reduce! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = f64, out = f64, vec = __m512d, loadu = _mm512_loadu_pd,
			intrinsic = $intrinsic,
			fixed_doc = $fixed_doc,
		}
	};
}

avx512_reduce_i32!(reduce_add_i32x16, reduce_add_i32x16_intrinsic, _mm512_reduce_add_epi32, i32, i32, "Sum of all 16 lanes (wrapping) (`vpaddd`-chain, 512-bit).");
avx512_reduce_i32!(reduce_add_u32x16, reduce_add_u32x16_intrinsic, _mm512_reduce_add_epi32, u32, u32, "Sum of all 16 lanes (wrapping) (`vpaddd`-chain, 512-bit).");
avx512_reduce_i64!(reduce_add_i64x8, reduce_add_i64x8_intrinsic, _mm512_reduce_add_epi64, i64, i64, "Sum of all 8 lanes (wrapping) (`vpaddq`-chain, 512-bit).");
avx512_reduce_i64!(reduce_add_u64x8, reduce_add_u64x8_intrinsic, _mm512_reduce_add_epi64, u64, u64, "Sum of all 8 lanes (wrapping) (`vpaddq`-chain, 512-bit).");
avx512_reduce_f32!(reduce_add_f32x16, reduce_add_f32x16_intrinsic, _mm512_reduce_add_ps, "Sum of all 16 lanes (`vaddps`-chain, 512-bit, order unspecified).");
avx512_reduce_f64!(reduce_add_f64x8, reduce_add_f64x8_intrinsic, _mm512_reduce_add_pd, "Sum of all 8 lanes (`vaddpd`-chain, 512-bit, order unspecified).");

avx512_reduce_i32!(reduce_mul_i32x16, reduce_mul_i32x16_intrinsic, _mm512_reduce_mul_epi32, i32, i32, "Product of all 16 lanes (wrapping) (`vpmulld`-chain, 512-bit).");
avx512_reduce_i32!(reduce_mul_u32x16, reduce_mul_u32x16_intrinsic, _mm512_reduce_mul_epi32, u32, u32, "Product of all 16 lanes (wrapping) (`vpmulld`-chain, 512-bit).");
avx512_reduce_i64!(reduce_mul_i64x8, reduce_mul_i64x8_intrinsic, _mm512_reduce_mul_epi64, i64, i64, "Product of all 8 lanes (wrapping) (`vpmullq`-chain, 512-bit).");
avx512_reduce_i64!(reduce_mul_u64x8, reduce_mul_u64x8_intrinsic, _mm512_reduce_mul_epi64, u64, u64, "Product of all 8 lanes (wrapping) (`vpmullq`-chain, 512-bit).");
avx512_reduce_f32!(reduce_mul_f32x16, reduce_mul_f32x16_intrinsic, _mm512_reduce_mul_ps, "Product of all 16 lanes (`vmulps`-chain, 512-bit, order unspecified).");
avx512_reduce_f64!(reduce_mul_f64x8, reduce_mul_f64x8_intrinsic, _mm512_reduce_mul_pd, "Product of all 8 lanes (`vmulpd`-chain, 512-bit, order unspecified).");

avx512_reduce_i32!(reduce_max_i32x16, reduce_max_i32x16_intrinsic, _mm512_reduce_max_epi32, i32, i32, "Largest of all 16 lanes, signed (`vpmaxsd`-chain, 512-bit).");
avx512_reduce_i32!(reduce_max_u32x16, reduce_max_u32x16_intrinsic, _mm512_reduce_max_epu32, u32, u32, "Largest of all 16 lanes, unsigned (`vpmaxud`-chain, 512-bit).");
avx512_reduce_i64!(reduce_max_i64x8, reduce_max_i64x8_intrinsic, _mm512_reduce_max_epi64, i64, i64, "Largest of all 8 lanes, signed (`vpmaxsq`-chain, 512-bit).");
avx512_reduce_i64!(reduce_max_u64x8, reduce_max_u64x8_intrinsic, _mm512_reduce_max_epu64, u64, u64, "Largest of all 8 lanes, unsigned (`vpmaxuq`-chain, 512-bit).");
avx512_reduce_f32!(reduce_max_f32x16, reduce_max_f32x16_intrinsic, _mm512_reduce_max_ps, "Largest of all 16 lanes (`vmaxps`-chain, 512-bit, SIMD NaN rules).");
avx512_reduce_f64!(reduce_max_f64x8, reduce_max_f64x8_intrinsic, _mm512_reduce_max_pd, "Largest of all 8 lanes (`vmaxpd`-chain, 512-bit, SIMD NaN rules).");

avx512_reduce_i32!(reduce_min_i32x16, reduce_min_i32x16_intrinsic, _mm512_reduce_min_epi32, i32, i32, "Smallest of all 16 lanes, signed (`vpminsd`-chain, 512-bit).");
avx512_reduce_i32!(reduce_min_u32x16, reduce_min_u32x16_intrinsic, _mm512_reduce_min_epu32, u32, u32, "Smallest of all 16 lanes, unsigned (`vpminud`-chain, 512-bit).");
avx512_reduce_i64!(reduce_min_i64x8, reduce_min_i64x8_intrinsic, _mm512_reduce_min_epi64, i64, i64, "Smallest of all 8 lanes, signed (`vpminsq`-chain, 512-bit).");
avx512_reduce_i64!(reduce_min_u64x8, reduce_min_u64x8_intrinsic, _mm512_reduce_min_epu64, u64, u64, "Smallest of all 8 lanes, unsigned (`vpminuq`-chain, 512-bit).");
avx512_reduce_f32!(reduce_min_f32x16, reduce_min_f32x16_intrinsic, _mm512_reduce_min_ps, "Smallest of all 16 lanes (`vminps`-chain, 512-bit, SIMD NaN rules).");
avx512_reduce_f64!(reduce_min_f64x8, reduce_min_f64x8_intrinsic, _mm512_reduce_min_pd, "Smallest of all 8 lanes (`vminpd`-chain, 512-bit, SIMD NaN rules).");

impl Avx512f {
	/// Widening multiply: full 64-bit product of each of the 16 lanes, split
	/// into low/high 32-bit halves: same 16-lane count on both outputs,
	/// unlike [`Avx512f::mul_i32x16`] which narrows back down to 32 bits.
	/// `pmuludq` only reads each 64-bit slot's low half, so the even lanes
	/// come from one pass and the odd lanes need a second pass on the
	/// shifted-down input, then both get re-interleaved with a mask blend
	/// (`vpmuludq`+`vpsrlq`+`vpsllq`+blend, 512-bit).
	#[inline]
	pub fn widening_mul_u32x16(self, a: [u32; 16], b: [u32; 16]) -> ([u32; 16], [u32; 16]) {
		unsafe { widening_mul_u32x16_composed(&a, &b) }
	}

	/// Signed sibling of [`Avx512f::widening_mul_u32x16`] (`vpmuldq` for the
	/// even-lane pass instead of `vpmuludq`; the odd-lane pass and
	/// re-interleave are identical since the low 32 bits of a product don't
	/// depend on signedness).
	#[inline]
	pub fn widening_mul_i32x16(self, a: [i32; 16], b: [i32; 16]) -> ([i32; 16], [i32; 16]) {
		unsafe { widening_mul_i32x16_composed(&a, &b) }
	}
}

const WIDENING_MUL_ODD_LANES: u16 = 0b1010_1010_1010_1010;

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn widening_mul_u32x16_composed(a: &[u32; 16], b: &[u32; 16]) -> ([u32; 16], [u32; 16]) {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let ab_evens = _mm512_mul_epu32(va, vb);
		let ab_odds = _mm512_mul_epu32(_mm512_srli_epi64::<32>(va), _mm512_srli_epi64::<32>(vb));
		let lo = _mm512_mask_blend_epi32(WIDENING_MUL_ODD_LANES, ab_evens, _mm512_slli_epi64::<32>(ab_odds));
		let hi = _mm512_mask_blend_epi32(WIDENING_MUL_ODD_LANES, _mm512_srli_epi64::<32>(ab_evens), ab_odds);
		let mut lo_out = [0u32; 16];
		let mut hi_out = [0u32; 16];
		_mm512_storeu_si512(lo_out.as_mut_ptr().cast(), lo);
		_mm512_storeu_si512(hi_out.as_mut_ptr().cast(), hi);
		(lo_out, hi_out)
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn widening_mul_i32x16_composed(a: &[i32; 16], b: &[i32; 16]) -> ([i32; 16], [i32; 16]) {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let ab_evens = _mm512_mul_epi32(va, vb);
		let ab_odds = _mm512_mul_epi32(_mm512_srli_epi64::<32>(va), _mm512_srli_epi64::<32>(vb));
		let lo = _mm512_mask_blend_epi32(WIDENING_MUL_ODD_LANES, ab_evens, _mm512_slli_epi64::<32>(ab_odds));
		let hi = _mm512_mask_blend_epi32(WIDENING_MUL_ODD_LANES, _mm512_srli_epi64::<32>(ab_evens), ab_odds);
		let mut lo_out = [0i32; 16];
		let mut hi_out = [0i32; 16];
		_mm512_storeu_si512(lo_out.as_mut_ptr().cast(), lo);
		_mm512_storeu_si512(hi_out.as_mut_ptr().cast(), hi);
		(lo_out, hi_out)
	}
}

// Complex f32/f64, interleaved `[re0, im0, re1, im1, ...]` layout, 2 pairs
// per 128-bit lane. Same AP-15 network as `Sse3`/`Avx`'s complex ops (see
// `sse3.rs` module doc): `permute_ps`/`permute_pd` apply the swap-immediate
// per 128-bit lane, so the 2-complex-per-lane shape carries over unchanged.
// No plain (non-fused) `addsub_ps`/`addsub_pd` exists at 512-bit: AVX-512F
// only has the fused `fmaddsub`/`fmsubadd` forms, so `mul`/`conj_mul` use
// `fmaddsub` directly instead of a separate mul+addsub pass (one fewer op
// than the 128/256-bit versions). Sign-flip reuses the file's existing
// `xor_ps_bitcast`/`xor_pd_bitcast` (no native `_mm512_xor_ps`/`_mm512_xor_pd`
// at this width: those are AVX-512DQ-gated, so plain AVX-512F goes through
// `xor_si512` + bitcast).
const COMPLEX_SWAP_PAIRS_F32: i32 = 0b10_11_00_01;
const COMPLEX_SWAP_PAIRS_F64: i32 = 0b0101_0101;
const COMPLEX_CONJ_SIGN_F32X16: [f32; 16] = [
	0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
];
const COMPLEX_CONJ_SIGN_F64X8: [f64; 8] = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
/// `mul_c*_intrinsic(conj=true)` negates `movehdup`/`unpackhi`'s broadcast of
/// `a.im` (present in *both* lanes of a pair), so it needs an all-lanes
/// negation, not the alternating `COMPLEX_CONJ_SIGN_*` pattern.
const COMPLEX_NEGATE_ALL_F32X16: [f32; 16] = [-0.0; 16];
const COMPLEX_NEGATE_ALL_F64X8: [f64; 8] = [-0.0; 8];

impl Avx512f {
	/// Negate the imaginary lane of each complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c32x16(self, a: [f32; 16]) -> [f32; 16] {
		unsafe { conj_c32x16_intrinsic(&a) }
	}

	/// Complex multiply per pair: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c32x16(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
		unsafe { mul_c32x16_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b` per pair, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c32x16(self, a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
		unsafe { mul_c32x16_intrinsic(&a, &b, true) }
	}

	/// `|a|^2` per pair, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c32x16(self, a: [f32; 16]) -> [f32; 16] {
		unsafe { abs2_c32x16_intrinsic(&a) }
	}

	/// Negate the imaginary lane of each complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c64x8(self, a: [f64; 8]) -> [f64; 8] {
		unsafe { conj_c64x8_intrinsic(&a) }
	}

	/// Complex multiply per pair: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c64x8(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
		unsafe { mul_c64x8_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b` per pair, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c64x8(self, a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
		unsafe { mul_c64x8_intrinsic(&a, &b, true) }
	}

	/// `|a|^2` per pair, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c64x8(self, a: [f64; 8]) -> [f64; 8] {
		unsafe { abs2_c64x8_intrinsic(&a) }
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn conj_c32x16_intrinsic(a: &[f32; 16]) -> [f32; 16] {
	unsafe {
		let va = _mm512_loadu_ps(a.as_ptr());
		let sign = _mm512_loadu_ps(COMPLEX_CONJ_SIGN_F32X16.as_ptr());
		let vr = xor_ps_bitcast(va, sign);
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `conj` selects the negated-`b` conjugate-multiply variant instead of a separate pre-pass.
///
/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_c32x16_intrinsic(a: &[f32; 16], b: &[f32; 16], conj: bool) -> [f32; 16] {
	unsafe {
		let ab = _mm512_loadu_ps(a.as_ptr());
		let xy = _mm512_loadu_ps(b.as_ptr());
		let yx = _mm512_permute_ps::<COMPLEX_SWAP_PAIRS_F32>(xy);
		let aa = _mm512_moveldup_ps(ab);
		let mut bb = _mm512_movehdup_ps(ab);
		if conj {
			let sign = _mm512_loadu_ps(COMPLEX_NEGATE_ALL_F32X16.as_ptr());
			bb = xor_ps_bitcast(bb, sign);
		}
		let vr = _mm512_fmaddsub_ps(aa, xy, _mm512_mul_ps(bb, yx));
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn abs2_c32x16_intrinsic(a: &[f32; 16]) -> [f32; 16] {
	unsafe {
		let va = _mm512_loadu_ps(a.as_ptr());
		let sqr = _mm512_mul_ps(va, va);
		let sqr_rev = _mm512_permute_ps::<COMPLEX_SWAP_PAIRS_F32>(sqr);
		let vr = _mm512_add_ps(sqr, sqr_rev);
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn conj_c64x8_intrinsic(a: &[f64; 8]) -> [f64; 8] {
	unsafe {
		let va = _mm512_loadu_pd(a.as_ptr());
		let sign = _mm512_loadu_pd(COMPLEX_CONJ_SIGN_F64X8.as_ptr());
		let vr = xor_pd_bitcast(va, sign);
		let mut out = [0f64; 8];
		_mm512_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_c64x8_intrinsic(a: &[f64; 8], b: &[f64; 8], conj: bool) -> [f64; 8] {
	unsafe {
		let ab = _mm512_loadu_pd(a.as_ptr());
		let xy = _mm512_loadu_pd(b.as_ptr());
		let yx = _mm512_permute_pd::<COMPLEX_SWAP_PAIRS_F64>(xy);
		let aa = _mm512_unpacklo_pd(ab, ab);
		let mut bb = _mm512_unpackhi_pd(ab, ab);
		if conj {
			let sign = _mm512_loadu_pd(COMPLEX_NEGATE_ALL_F64X8.as_ptr());
			bb = xor_pd_bitcast(bb, sign);
		}
		let vr = _mm512_fmaddsub_pd(aa, xy, _mm512_mul_pd(bb, yx));
		let mut out = [0f64; 8];
		_mm512_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn abs2_c64x8_intrinsic(a: &[f64; 8]) -> [f64; 8] {
	unsafe {
		let va = _mm512_loadu_pd(a.as_ptr());
		let sqr = _mm512_mul_pd(va, va);
		let sqr_rev = _mm512_permute_pd::<COMPLEX_SWAP_PAIRS_F64>(sqr);
		let vr = _mm512_add_pd(sqr, sqr_rev);
		let mut out = [0f64; 8];
		_mm512_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

macro_rules! avx512_f32_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = f32, vec = __m512, loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_f64_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Avx512f, target_feature = "avx512f",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = f64, vec = __m512d, loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512_f32_ternop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = f32, vec = __m512, mask = u16,
			loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512_f64_ternop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512f, target_feature = "avx512f",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = f64, vec = __m512d, mask = u8,
			loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512_f32_ternop!(
	fmadd_f32x16, fmadd_f32_slice, vfmaddps512, _mm512_fmadd_ps, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddps`, 512-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 16-wide chunks, scalar remainder."
);
avx512_f64_ternop!(
	fmadd_f64x8, fmadd_f64_slice, vfmaddpd512, _mm512_fmadd_pd, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddpd`, 512-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
avx512_f32_ternop_masked!(
	fmadd_f32x16_merge_masked, fmadd_f32x16_zero_masked, mask_fmadd_ps_intrinsic, maskz_fmadd_ps_intrinsic,
	_mm512_mask_fmadd_ps, _mm512_maskz_fmadd_ps,
	"`a * b + c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfmaddps`, merge-masked - `a` is both an input and the merge fallback, matching hardware's 3-operand FMA encoding).",
	"`a * b + c` per lane, fused, where `mask` bit is set, else zero (`vfmaddps`, zero-masked)."
);
avx512_f64_ternop_masked!(
	fmadd_f64x8_merge_masked, fmadd_f64x8_zero_masked, mask_fmadd_pd_intrinsic, maskz_fmadd_pd_intrinsic,
	_mm512_mask_fmadd_pd, _mm512_maskz_fmadd_pd,
	"`a * b + c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfmaddpd`, merge-masked).",
	"`a * b + c` per lane, fused, where `mask` bit is set, else zero (`vfmaddpd`, zero-masked)."
);

avx512_f32_ternop!(
	fmsub_f32x16, fmsub_f32_slice, vfmsubps512, _mm512_fmsub_ps, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubps`, 512-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 16-wide chunks, scalar remainder."
);
avx512_f64_ternop!(
	fmsub_f64x8, fmsub_f64_slice, vfmsubpd512, _mm512_fmsub_pd, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubpd`, 512-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
avx512_f32_ternop_masked!(
	fmsub_f32x16_merge_masked, fmsub_f32x16_zero_masked, mask_fmsub_ps_intrinsic, maskz_fmsub_ps_intrinsic,
	_mm512_mask_fmsub_ps, _mm512_maskz_fmsub_ps,
	"`a * b - c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfmsubps`, merge-masked).",
	"`a * b - c` per lane, fused, where `mask` bit is set, else zero (`vfmsubps`, zero-masked)."
);
avx512_f64_ternop_masked!(
	fmsub_f64x8_merge_masked, fmsub_f64x8_zero_masked, mask_fmsub_pd_intrinsic, maskz_fmsub_pd_intrinsic,
	_mm512_mask_fmsub_pd, _mm512_maskz_fmsub_pd,
	"`a * b - c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfmsubpd`, merge-masked).",
	"`a * b - c` per lane, fused, where `mask` bit is set, else zero (`vfmsubpd`, zero-masked)."
);
avx512_f32_ternop!(
	fnmadd_f32x16, fnmadd_f32_slice, vfnmaddps512, _mm512_fnmadd_ps, |a: f32, b: f32, c: f32| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddps`, 512-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 16-wide chunks, scalar remainder."
);
avx512_f64_ternop!(
	fnmadd_f64x8, fnmadd_f64_slice, vfnmaddpd512, _mm512_fnmadd_pd, |a: f64, b: f64, c: f64| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddpd`, 512-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
avx512_f32_ternop_masked!(
	fnmadd_f32x16_merge_masked, fnmadd_f32x16_zero_masked, mask_fnmadd_ps_intrinsic, maskz_fnmadd_ps_intrinsic,
	_mm512_mask_fnmadd_ps, _mm512_maskz_fnmadd_ps,
	"`-(a * b) + c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfnmaddps`, merge-masked).",
	"`-(a * b) + c` per lane, fused, where `mask` bit is set, else zero (`vfnmaddps`, zero-masked)."
);
avx512_f64_ternop_masked!(
	fnmadd_f64x8_merge_masked, fnmadd_f64x8_zero_masked, mask_fnmadd_pd_intrinsic, maskz_fnmadd_pd_intrinsic,
	_mm512_mask_fnmadd_pd, _mm512_maskz_fnmadd_pd,
	"`-(a * b) + c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfnmaddpd`, merge-masked).",
	"`-(a * b) + c` per lane, fused, where `mask` bit is set, else zero (`vfnmaddpd`, zero-masked)."
);
avx512_f32_ternop!(
	fnmsub_f32x16, fnmsub_f32_slice, vfnmsubps512, _mm512_fnmsub_ps, |a: f32, b: f32, c: f32| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubps`, 512-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 16-wide chunks, scalar remainder."
);
avx512_f64_ternop!(
	fnmsub_f64x8, fnmsub_f64_slice, vfnmsubpd512, _mm512_fnmsub_pd, |a: f64, b: f64, c: f64| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubpd`, 512-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
avx512_f32_ternop_masked!(
	fnmsub_f32x16_merge_masked, fnmsub_f32x16_zero_masked, mask_fnmsub_ps_intrinsic, maskz_fnmsub_ps_intrinsic,
	_mm512_mask_fnmsub_ps, _mm512_maskz_fnmsub_ps,
	"`-(a * b) - c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfnmsubps`, merge-masked).",
	"`-(a * b) - c` per lane, fused, where `mask` bit is set, else zero (`vfnmsubps`, zero-masked)."
);
avx512_f64_ternop_masked!(
	fnmsub_f64x8_merge_masked, fnmsub_f64x8_zero_masked, mask_fnmsub_pd_intrinsic, maskz_fnmsub_pd_intrinsic,
	_mm512_mask_fnmsub_pd, _mm512_maskz_fnmsub_pd,
	"`-(a * b) - c` per lane, fused, where `mask` bit is set, else copied from `a` (`vfnmsubpd`, merge-masked).",
	"`-(a * b) - c` per lane, fused, where `mask` bit is set, else zero (`vfnmsubpd`, zero-masked)."
);

impl Avx512f {
	/// `a + b[0]*c[0] + b[1]*c[1] + b[2]*c[2] + b[3]*c[3]`, 4 fused adds folded
	/// in sequence (`VP4FMADDPS` semantics, software-composed from `vfmaddps`).
	#[inline]
	pub fn p4fmadd_f32x16(self, a: [f32; 16], b: [[f32; 16]; 4], c: [f32; 4]) -> [f32; 16] {
		let mut acc = a;
		for n in 0..4 {
			acc = self.fmadd_f32x16(b[n], [c[n]; 16], acc);
		}
		acc
	}

	/// `a - b[0]*c[0] - b[1]*c[1] - b[2]*c[2] - b[3]*c[3]`, 4 fused subtracts
	/// folded in sequence (`VP4FNMADDPS` semantics, software-composed).
	#[inline]
	pub fn p4fnmadd_f32x16(self, a: [f32; 16], b: [[f32; 16]; 4], c: [f32; 4]) -> [f32; 16] {
		let mut acc = a;
		for n in 0..4 {
			acc = self.fnmadd_f32x16(b[n], [c[n]; 16], acc);
		}
		acc
	}

	/// [`Avx512f::p4fmadd_f32x16`] over slices: `out[i] = a[i] + sum_n(b[n][i] * c[n])`.
	/// 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a`/`out`/every `b[n]` length mismatch.
	pub fn p4fmadd_f32_slice(self, a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
		assert_eq!(out.len(), a.len());
		for bn in &b {
			assert_eq!(bn.len(), a.len());
		}

		let mut a_chunks = a.chunks_exact(16);
		let mut b_chunks: [_; 4] = core::array::from_fn(|n| b[n].chunks_exact(16));
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in (&mut a_chunks).zip(&mut out_chunks) {
			let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [[f32; 16]; 4] = core::array::from_fn(|n| {
				b_chunks[n].next().expect("chunks_exact len").try_into().expect("chunks_exact width")
			});
			oc.copy_from_slice(&self.p4fmadd_f32x16(av, bv, c));
		}

		let a_rem = a_chunks.remainder();
		let b_rem: [&[f32]; 4] = core::array::from_fn(|n| b_chunks[n].remainder());
		for (i, (&av, o)) in a_rem.iter().zip(out_chunks.into_remainder()).enumerate() {
			let mut acc = av;
			for n in 0..4 {
				acc += b_rem[n][i] * c[n];
			}
			*o = acc;
		}
	}

	/// [`Avx512f::p4fnmadd_f32x16`] over slices: `out[i] = a[i] - sum_n(b[n][i] * c[n])`.
	/// 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a`/`out`/every `b[n]` length mismatch.
	pub fn p4fnmadd_f32_slice(self, a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
		assert_eq!(out.len(), a.len());
		for bn in &b {
			assert_eq!(bn.len(), a.len());
		}

		let mut a_chunks = a.chunks_exact(16);
		let mut b_chunks: [_; 4] = core::array::from_fn(|n| b[n].chunks_exact(16));
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in (&mut a_chunks).zip(&mut out_chunks) {
			let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [[f32; 16]; 4] = core::array::from_fn(|n| {
				b_chunks[n].next().expect("chunks_exact len").try_into().expect("chunks_exact width")
			});
			oc.copy_from_slice(&self.p4fnmadd_f32x16(av, bv, c));
		}

		let a_rem = a_chunks.remainder();
		let b_rem: [&[f32]; 4] = core::array::from_fn(|n| b_chunks[n].remainder());
		for (i, (&av, o)) in a_rem.iter().zip(out_chunks.into_remainder()).enumerate() {
			let mut acc = av;
			for n in 0..4 {
				acc -= b_rem[n][i] * c[n];
			}
			*o = acc;
		}
	}
}

/// `cvtph_ps`/`cvtps_ph` 512-bit rung. The 128/256-bit forms are F16C
/// (`avx/f16c.rs`, `F16c` token); this 512-bit pair is a separate,
/// surprising gap: it's gated by plain `"avx512f"` (stable since 1.89,
/// predates AVX-512FP16 entirely) and represents the FP16 side as `__m256i`/
/// `__m512i`, not `__m128h`: unrelated to [`super::avx512fp16`]. Hand-written
/// (cross-type, same shape as `f16c.rs`), full fixed+slice reusing that
/// file's `f16_to_f32_scalar`/`f32_to_f16_scalar`.
impl Avx512f {
	/// 16 half-float bit patterns to `f32` (`vcvtph2ps`, 512-bit).
	#[inline]
	pub fn f16_to_f32x16(self, a: [u16; 16]) -> [f32; 16] {
		unsafe { cvtph2ps_x16(&a) }
	}

	/// 16 `f32` to half-float bits (`vcvtps2ph`, 512-bit). `ROUNDING`: same
	/// `_MM_FROUND_TO_*`/`_MM_FROUND_CUR_DIRECTION` operand as
	/// [`super::super::avx::f16c::F16c::f32_to_f16x8`].
	#[inline]
	pub fn f32_to_f16x16<const ROUNDING: i32>(self, a: [f32; 16]) -> [u16; 16] {
		unsafe { cvtps2ph_x16::<ROUNDING>(&a) }
	}

	/// `out[i] = f16_to_f32(a[i])`. 16-wide chunks, software scalar rem.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn f16_to_f32_slice(self, a: &[u16], out: &mut [f32]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u16; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.f16_to_f32x16(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f16_to_f32_scalar(x);
		}
	}

	/// `out[i] = f32_to_f16(a[i])`. Vector chunks use `ROUNDING`; scalar rem
	/// always RNE (ignores `ROUNDING`).
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn f32_to_f16_slice<const ROUNDING: i32>(self, a: &[f32], out: &mut [u16]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.f32_to_f16x16::<ROUNDING>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f32_to_f16_scalar(x);
		}
	}
}

/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cvtph2ps_x16(a: &[u16; 16]) -> [f32; 16] {
	unsafe {
		let va: core::arch::x86_64::__m256i = core::arch::x86_64::_mm256_loadu_si256(a.as_ptr().cast());
		let vr = core::arch::x86_64::_mm512_cvtph_ps(va);
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via the token.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn cvtps2ph_x16<const ROUNDING: i32>(a: &[f32; 16]) -> [u16; 16] {
	unsafe {
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vr: core::arch::x86_64::__m256i = core::arch::x86_64::_mm512_cvtps_ph::<ROUNDING>(va);
		let mut out = [0u16; 16];
		core::arch::x86_64::_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

simd_extract_imm! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = extract_u8x16_from_x64, intrinsic_fn = extract_u8x16_from_x64_intrinsic,
	wide_width = 64, narrow_width = 16, elem = u8, wide_vec = __m512i, narrow_vec = __m128i,
	wide_loadu = _mm512_loadu_si512, storeu = _mm_storeu_si128, intrinsic = _mm512_extracti32x4_epi32,
	fixed_doc = "Extracts the `IMM2 & 3`-selected 16-byte quarter of `a` (`vextracti32x4`, 512-bit source).",
}

simd_insert_imm! {
	token = Avx512f, target_feature = "avx512f",
	fixed_fn = insert_u8x16_into_x64, intrinsic_fn = insert_u8x16_into_x64_intrinsic,
	wide_width = 64, narrow_width = 16, elem = u8, wide_vec = __m512i, narrow_vec = __m128i,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_inserti32x4,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 3`-selected 16-byte quarter with `b` (`vinserti32x4`, 512-bit).",
}

// Partial (ragged-tail) load/store: read/write fewer than the full lane
// count without a scalar remainder loop. AVX-512's masked load/store are
// fault-suppressed on masked-off lanes: unlike masked *arithmetic*, they
// genuinely never touch memory past the mask, so reading/writing straight
// from/to a short `&[T]`/`&mut [T]` (no full-width backing buffer required)
// is sound. This is the primitive [`Avx512f::mask_between_mask16`]'s doc
// already pointed at ("the shape a masked partial/tail load or store
// needs").
//
// Mask-width-16 element types (f32/i32/u32) reuse `self.mask_between_mask16`
// directly (same token). Mask-width-8 types (f64/i64/u64) do *not* reuse
// `Avx512Dq::mask_between_mask8`: the load/store instructions below are
// plain AVX-512F (only k-register ALU ops on 8-bit masks need DQ), so
// `low_mask8` duplicates that one-line formula instead of pulling in a
// second token for no hardware reason.
//
// i16/i8 (BW-gated, mask32/mask64) are intentionally not covered: nothing
// built so far uses them, and adding two more element types here would be
// exactly the kind of unrequested breadth this primitive doesn't need yet.
fn low_mask8(n: u32) -> u8 {
	if n >= 8 { u8::MAX } else { (1u8 << n) - 1 }
}

impl Avx512f {
	/// Loads `slice.len().min(16)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_f32x16(self, slice: &[f32]) -> [f32; 16] {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_load_f32x16_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(16)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_f32x16(self, v: [f32; 16], slice: &mut [f32]) {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_store_f32x16_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}

	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_f64x8(self, slice: &[f64]) -> [f64; 8] {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_load_f64x8_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_f64x8(self, v: [f64; 8], slice: &mut [f64]) {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_store_f64x8_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}

	/// Loads `slice.len().min(16)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_i32x16(self, slice: &[i32]) -> [i32; 16] {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_load_i32x16_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(16)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_i32x16(self, v: [i32; 16], slice: &mut [i32]) {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_store_i32x16_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}

	/// Loads `slice.len().min(16)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_u32x16(self, slice: &[u32]) -> [u32; 16] {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_load_u32x16_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(16)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_u32x16(self, v: [u32; 16], slice: &mut [u32]) {
		let mask = self.mask_between_mask16(slice.len().min(16) as u32);
		unsafe { partial_store_u32x16_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}

	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_i64x8(self, slice: &[i64]) -> [i64; 8] {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_load_i64x8_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_i64x8(self, v: [i64; 8], slice: &mut [i64]) {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_store_i64x8_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}

	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_u64x8(self, slice: &[u64]) -> [u64; 8] {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_load_u64x8_intrinsic(slice.as_ptr(), mask) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_u64x8(self, v: [u64; 8], slice: &mut [u64]) {
		let mask = low_mask8(slice.len().min(8) as u32);
		unsafe { partial_store_u64x8_intrinsic(slice.as_mut_ptr(), mask, &v) }
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed
/// `ptr`'s valid element count (masked-off lanes are hardware-fault-suppressed,
/// so this is the one load in the file allowed a short source buffer).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_f32x16_intrinsic(ptr: *const f32, mask: u16) -> [f32; 16] {
	unsafe {
		let v = _mm512_maskz_loadu_ps(mask, ptr);
		let mut out = [0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_f32x16_intrinsic(ptr: *mut f32, mask: u16, v: &[f32; 16]) {
	unsafe {
		let vv = _mm512_loadu_ps(v.as_ptr());
		_mm512_mask_storeu_ps(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_f64x8_intrinsic(ptr: *const f64, mask: u8) -> [f64; 8] {
	unsafe {
		let v = _mm512_maskz_loadu_pd(mask, ptr);
		let mut out = [0f64; 8];
		_mm512_storeu_pd(out.as_mut_ptr(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_f64x8_intrinsic(ptr: *mut f64, mask: u8, v: &[f64; 8]) {
	unsafe {
		let vv = _mm512_loadu_pd(v.as_ptr());
		_mm512_mask_storeu_pd(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_i32x16_intrinsic(ptr: *const i32, mask: u16) -> [i32; 16] {
	unsafe {
		let v = _mm512_maskz_loadu_epi32(mask, ptr);
		let mut out = [0i32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_i32x16_intrinsic(ptr: *mut i32, mask: u16, v: &[i32; 16]) {
	unsafe {
		let vv = _mm512_loadu_si512(v.as_ptr().cast());
		_mm512_mask_storeu_epi32(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_u32x16_intrinsic(ptr: *const u32, mask: u16) -> [u32; 16] {
	unsafe {
		let v = _mm512_maskz_loadu_epi32(mask, ptr.cast());
		let mut out = [0u32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_u32x16_intrinsic(ptr: *mut u32, mask: u16, v: &[u32; 16]) {
	unsafe {
		let vv = _mm512_loadu_si512(v.as_ptr().cast());
		_mm512_mask_storeu_epi32(ptr.cast(), mask, vv);
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_i64x8_intrinsic(ptr: *const i64, mask: u8) -> [i64; 8] {
	unsafe {
		let v = _mm512_maskz_loadu_epi64(mask, ptr);
		let mut out = [0i64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_i64x8_intrinsic(ptr: *mut i64, mask: u8, v: &[i64; 8]) {
	unsafe {
		let vv = _mm512_loadu_si512(v.as_ptr().cast());
		_mm512_mask_storeu_epi64(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_load_u64x8_intrinsic(ptr: *const u64, mask: u8) -> [u64; 8] {
	unsafe {
		let v = _mm512_maskz_loadu_epi64(mask, ptr.cast());
		let mut out = [0u64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`]. `mask`'s set bits must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn partial_store_u64x8_intrinsic(ptr: *mut u64, mask: u8, v: &[u64; 8]) {
	unsafe {
		let vv = _mm512_loadu_si512(v.as_ptr().cast());
		_mm512_mask_storeu_epi64(ptr.cast(), mask, vv);
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512f.rs"]
mod tests;
