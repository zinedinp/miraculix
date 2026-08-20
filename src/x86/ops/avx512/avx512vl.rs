//! AVX512VL: composed 128/256-bit rungs for eight host extensions.
//! Each requires its own bit plus `"avx512vl"`. Not wired into `auto`, because the 512-bit token already wins whenever the base AVX-512F bit is present.
//! This module is for `ops::` callers needing a fixed 128/256-bit width.
//! `Avx512FVl`'s `rcp14`/`rsqrt14` are fixed-width only

use core::arch::x86_64::{
	__m128, __m128bh, __m128d, __m128i, __m256, __m256bh, __m256d, __m256i, _mm_bitshuffle_epi64_mask, _mm_dpbusd_epi32,
	_mm_dpbusds_epi32, _mm_dpwssd_epi32, _mm_dpwssds_epi32, _mm_loadu_pd, _mm_loadu_ps, _mm_loadu_si128,
	_mm_madd52hi_epu64, _mm_madd52lo_epu64, _mm_mask_madd52hi_epu64, _mm_mask_madd52lo_epu64,
	_mm_maskz_madd52hi_epu64, _mm_maskz_madd52lo_epu64, _mm_mask_add_epi32, _mm_mask_add_epi64, _mm_mask_add_pd,
	_mm_mask_add_ps, _mm_mask_div_pd, _mm_mask_div_ps, _mm_mask_max_epi32, _mm_mask_max_epi64, _mm_mask_max_epu32,
	_mm_mask_max_epu64, _mm_mask_max_pd, _mm_mask_max_ps, _mm_mask_min_epi32, _mm_mask_min_epi64,
	_mm_mask_min_epu32, _mm_mask_min_epu64, _mm_mask_min_pd, _mm_mask_min_ps, _mm_mask_mul_pd, _mm_mask_mul_ps,
	_mm_mask_mullo_epi32, _mm_mask_sub_epi32, _mm_mask_sub_epi64, _mm_mask_sub_pd, _mm_mask_sub_ps,
	_mm_maskz_add_epi32, _mm_maskz_add_epi64, _mm_maskz_add_pd, _mm_maskz_add_ps, _mm_maskz_div_pd,
	_mm_maskz_div_ps, _mm_maskz_max_epi32, _mm_maskz_max_epi64, _mm_maskz_max_epu32, _mm_maskz_max_epu64,
	_mm_maskz_max_pd, _mm_maskz_max_ps, _mm_maskz_min_epi32, _mm_maskz_min_epi64, _mm_maskz_min_epu32,
	_mm_maskz_min_epu64, _mm_maskz_min_pd, _mm_maskz_min_ps, _mm_maskz_mul_pd, _mm_maskz_mul_ps,
	_mm_maskz_mullo_epi32, _mm_maskz_sub_epi32, _mm_maskz_sub_epi64, _mm_maskz_sub_pd, _mm_maskz_sub_ps,
	_mm_multishift_epi64_epi8, _mm_mullo_epi64, _mm_permutex2var_epi8, _mm_permutexvar_epi8, _mm_popcnt_epi16,
	_mm_popcnt_epi32, _mm_popcnt_epi64, _mm_popcnt_epi8, _mm_shldi_epi16, _mm_shldi_epi32, _mm_shldi_epi64,
	_mm_shldv_epi16, _mm_shldv_epi32, _mm_shldv_epi64, _mm_shrdi_epi16, _mm_shrdi_epi32, _mm_shrdi_epi64,
	_mm_shrdv_epi16, _mm_shrdv_epi32, _mm_shrdv_epi64, _mm_storeu_pd, _mm_storeu_ps, _mm_storeu_si128,
	_mm256_bitshuffle_epi64_mask, _mm256_dpbusd_epi32, _mm256_dpbusds_epi32, _mm256_dpwssd_epi32, _mm256_shldi_epi16,
	_mm256_shldi_epi32, _mm256_shldi_epi64, _mm256_shrdi_epi16, _mm256_shrdi_epi32, _mm256_shrdi_epi64,
	_mm256_dpwssds_epi32, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd52hi_epu64,
	_mm256_madd52lo_epu64, _mm256_mask_madd52hi_epu64, _mm256_mask_madd52lo_epu64, _mm256_maskz_madd52hi_epu64,
	_mm256_maskz_madd52lo_epu64, _mm256_mask_add_epi32, _mm256_mask_add_epi64, _mm256_mask_add_pd, _mm256_mask_add_ps,
	_mm256_mask_div_pd, _mm256_mask_div_ps, _mm256_mask_max_epi32, _mm256_mask_max_epi64, _mm256_mask_max_epu32,
	_mm256_mask_max_epu64, _mm256_mask_max_pd, _mm256_mask_max_ps, _mm256_mask_min_epi32, _mm256_mask_min_epi64,
	_mm256_mask_min_epu32, _mm256_mask_min_epu64, _mm256_mask_min_pd, _mm256_mask_min_ps, _mm256_mask_mul_pd,
	_mm256_mask_mul_ps, _mm256_mask_mullo_epi32, _mm256_mask_sub_epi32, _mm256_mask_sub_epi64, _mm256_mask_sub_pd,
	_mm256_mask_sub_ps, _mm256_maskz_add_epi32, _mm256_maskz_add_epi64, _mm256_maskz_add_pd, _mm256_maskz_add_ps,
	_mm256_maskz_div_pd, _mm256_maskz_div_ps, _mm256_maskz_max_epi32, _mm256_maskz_max_epi64,
	_mm256_maskz_max_epu32, _mm256_maskz_max_epu64, _mm256_maskz_max_pd, _mm256_maskz_max_ps,
	_mm256_maskz_min_epi32, _mm256_maskz_min_epi64, _mm256_maskz_min_epu32, _mm256_maskz_min_epu64,
	_mm256_maskz_min_pd, _mm256_maskz_min_ps, _mm256_maskz_mul_pd, _mm256_maskz_mul_ps, _mm256_maskz_mullo_epi32,
	_mm256_maskz_sub_epi32, _mm256_maskz_sub_epi64, _mm256_maskz_sub_pd, _mm256_maskz_sub_ps,
	_mm256_multishift_epi64_epi8, _mm256_mullo_epi64, _mm256_permutex2var_epi8, _mm256_permutexvar_epi8,
	_mm256_popcnt_epi16, _mm256_popcnt_epi32, _mm256_popcnt_epi64, _mm256_popcnt_epi8, _mm256_shldv_epi16,
	_mm256_shldv_epi32, _mm256_shldv_epi64, _mm256_shrdv_epi16, _mm256_shrdv_epi32, _mm256_shrdv_epi64,
	_mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm_mask_ternarylogic_epi32, _mm_mask_ternarylogic_epi64,
	_mm_maskz_ternarylogic_epi32, _mm_maskz_ternarylogic_epi64, _mm_ternarylogic_epi32, _mm_ternarylogic_epi64,
	_mm256_mask_ternarylogic_epi32, _mm256_mask_ternarylogic_epi64, _mm256_maskz_ternarylogic_epi32,
	_mm256_maskz_ternarylogic_epi64, _mm256_ternarylogic_epi32, _mm256_ternarylogic_epi64, _mm_cvtne2ps_pbh,
	_mm_cvtneps_pbh, _mm_dpbf16_ps, _mm256_cvtne2ps_pbh, _mm256_cvtneps_pbh, _mm256_dpbf16_ps, _mm_cvtepi64_pd,
	_mm_cvtepu64_pd, _mm_cvtpd_epi64, _mm_cvtpd_epu64, _mm_cvttpd_epi64, _mm_cvttpd_epu64, _mm_fpclass_pd_mask,
	_mm_fpclass_ps_mask, _mm_range_pd, _mm_range_ps, _mm_reduce_pd, _mm_reduce_ps, _mm256_broadcast_f32x2,
	_mm256_broadcast_f64x2, _mm256_broadcast_i32x2, _mm256_broadcast_i64x2, _mm256_cvtepi64_pd, _mm256_cvtepu64_pd,
	_mm256_cvtpd_epi64, _mm256_cvtpd_epu64, _mm256_cvttpd_epi64, _mm256_cvttpd_epu64, _mm256_extractf64x2_pd,
	_mm256_extracti64x2_epi64, _mm256_fpclass_pd_mask, _mm256_fpclass_ps_mask, _mm256_insertf64x2,
	_mm256_inserti64x2, _mm256_range_pd, _mm256_range_ps, _mm256_reduce_pd, _mm256_reduce_ps,
	// Merge/zero-masked forms for DqVl/VnniVl/Bf16Vl/VbmiVl.
	_mm_mask_mullo_epi64, _mm_maskz_mullo_epi64, _mm256_mask_mullo_epi64, _mm256_maskz_mullo_epi64,
	_mm_mask_cvtepi64_pd, _mm_maskz_cvtepi64_pd, _mm256_mask_cvtepi64_pd, _mm256_maskz_cvtepi64_pd,
	_mm_mask_cvtepu64_pd, _mm_maskz_cvtepu64_pd, _mm256_mask_cvtepu64_pd, _mm256_maskz_cvtepu64_pd,
	_mm_mask_cvtpd_epi64, _mm_maskz_cvtpd_epi64, _mm256_mask_cvtpd_epi64, _mm256_maskz_cvtpd_epi64,
	_mm_mask_cvttpd_epi64, _mm_maskz_cvttpd_epi64, _mm256_mask_cvttpd_epi64, _mm256_maskz_cvttpd_epi64,
	_mm_mask_cvtpd_epu64, _mm_maskz_cvtpd_epu64, _mm256_mask_cvtpd_epu64, _mm256_maskz_cvtpd_epu64,
	_mm_mask_cvttpd_epu64, _mm_maskz_cvttpd_epu64, _mm256_mask_cvttpd_epu64, _mm256_maskz_cvttpd_epu64,
	_mm_mask_range_pd, _mm_maskz_range_pd, _mm256_mask_range_pd, _mm256_maskz_range_pd,
	_mm_mask_range_ps, _mm_maskz_range_ps, _mm256_mask_range_ps, _mm256_maskz_range_ps,
	_mm_mask_reduce_pd, _mm_maskz_reduce_pd, _mm256_mask_reduce_pd, _mm256_maskz_reduce_pd,
	_mm_mask_reduce_ps, _mm_maskz_reduce_ps, _mm256_mask_reduce_ps, _mm256_maskz_reduce_ps,
	_mm_mask_fpclass_pd_mask, _mm256_mask_fpclass_pd_mask, _mm_mask_fpclass_ps_mask, _mm256_mask_fpclass_ps_mask,
	_mm256_mask_broadcast_f32x2, _mm256_maskz_broadcast_f32x2, _mm256_mask_broadcast_i32x2,
	_mm256_maskz_broadcast_i32x2, _mm256_mask_broadcast_f64x2, _mm256_maskz_broadcast_f64x2,
	_mm256_mask_broadcast_i64x2, _mm256_maskz_broadcast_i64x2, _mm256_mask_extractf64x2_pd,
	_mm256_maskz_extractf64x2_pd, _mm256_mask_extracti64x2_epi64, _mm256_maskz_extracti64x2_epi64,
	_mm256_mask_insertf64x2, _mm256_maskz_insertf64x2, _mm256_mask_inserti64x2, _mm256_maskz_inserti64x2,
	_mm_mask_dpbusd_epi32, _mm_maskz_dpbusd_epi32, _mm256_mask_dpbusd_epi32, _mm256_maskz_dpbusd_epi32,
	_mm_mask_dpbusds_epi32, _mm_maskz_dpbusds_epi32, _mm256_mask_dpbusds_epi32, _mm256_maskz_dpbusds_epi32,
	_mm_mask_dpwssd_epi32, _mm_maskz_dpwssd_epi32, _mm256_mask_dpwssd_epi32, _mm256_maskz_dpwssd_epi32,
	_mm_mask_dpwssds_epi32, _mm_maskz_dpwssds_epi32, _mm256_mask_dpwssds_epi32, _mm256_maskz_dpwssds_epi32,
	_mm_mask_permutexvar_epi8, _mm_maskz_permutexvar_epi8, _mm256_mask_permutexvar_epi8,
	_mm256_maskz_permutexvar_epi8, _mm_mask_permutex2var_epi8, _mm_maskz_permutex2var_epi8,
	_mm256_mask_permutex2var_epi8, _mm256_maskz_permutex2var_epi8, _mm_mask_multishift_epi64_epi8,
	_mm_maskz_multishift_epi64_epi8, _mm256_mask_multishift_epi64_epi8, _mm256_maskz_multishift_epi64_epi8,
	_mm_mask_dpbf16_ps, _mm_maskz_dpbf16_ps, _mm256_mask_dpbf16_ps, _mm256_maskz_dpbf16_ps,
	_mm_mask_cvtneps_pbh, _mm_maskz_cvtneps_pbh, _mm256_mask_cvtneps_pbh, _mm256_maskz_cvtneps_pbh,
	_mm_mask_cvtne2ps_pbh, _mm_maskz_cvtne2ps_pbh, _mm256_mask_cvtne2ps_pbh, _mm256_maskz_cvtne2ps_pbh,
	_mm_cvtpbh_ps, _mm256_cvtpbh_ps, _mm_mask_cvtpbh_ps, _mm_maskz_cvtpbh_ps, _mm256_mask_cvtpbh_ps,
	_mm256_maskz_cvtpbh_ps, _mm_rcp14_pd, _mm_rcp14_ps, _mm_rsqrt14_pd, _mm_rsqrt14_ps, _mm256_rcp14_pd,
	_mm256_rcp14_ps, _mm256_rsqrt14_pd, _mm256_rsqrt14_ps,
	// shldi/shrdi/shldv/shrdv merge/zero-masked, closes the VBMI2-rest deferral.
	_mm_mask_shldi_epi16, _mm_maskz_shldi_epi16, _mm256_mask_shldi_epi16, _mm256_maskz_shldi_epi16,
	_mm_mask_shldi_epi32, _mm_maskz_shldi_epi32, _mm256_mask_shldi_epi32, _mm256_maskz_shldi_epi32,
	_mm_mask_shldi_epi64, _mm_maskz_shldi_epi64, _mm256_mask_shldi_epi64, _mm256_maskz_shldi_epi64,
	_mm_mask_shrdi_epi16, _mm_maskz_shrdi_epi16, _mm256_mask_shrdi_epi16, _mm256_maskz_shrdi_epi16,
	_mm_mask_shrdi_epi32, _mm_maskz_shrdi_epi32, _mm256_mask_shrdi_epi32, _mm256_maskz_shrdi_epi32,
	_mm_mask_shrdi_epi64, _mm_maskz_shrdi_epi64, _mm256_mask_shrdi_epi64, _mm256_maskz_shrdi_epi64,
	_mm_mask_shldv_epi16, _mm_maskz_shldv_epi16, _mm256_mask_shldv_epi16, _mm256_maskz_shldv_epi16,
	_mm_mask_shldv_epi32, _mm_maskz_shldv_epi32, _mm256_mask_shldv_epi32, _mm256_maskz_shldv_epi32,
	_mm_mask_shldv_epi64, _mm_maskz_shldv_epi64, _mm256_mask_shldv_epi64, _mm256_maskz_shldv_epi64,
	_mm_mask_shrdv_epi16, _mm_maskz_shrdv_epi16, _mm256_mask_shrdv_epi16, _mm256_maskz_shrdv_epi16,
	_mm_mask_shrdv_epi32, _mm_maskz_shrdv_epi32, _mm256_mask_shrdv_epi32, _mm256_maskz_shrdv_epi32,
	_mm_mask_shrdv_epi64, _mm_maskz_shrdv_epi64, _mm256_mask_shrdv_epi64, _mm256_maskz_shrdv_epi64,
	// f32 (ps) <-> i64/u64 at 128/256-bit.
	_mm_cvtps_epi64, _mm_cvttps_epi64, _mm_cvtps_epu64, _mm_cvttps_epu64, _mm_cvtepi64_ps, _mm_cvtepu64_ps,
	_mm256_cvtps_epi64, _mm256_cvttps_epi64, _mm256_cvtps_epu64, _mm256_cvttps_epu64, _mm256_cvtepi64_ps,
	_mm256_cvtepu64_ps, _mm_mask_cvtps_epi64, _mm_maskz_cvtps_epi64, _mm_mask_cvttps_epi64, _mm_maskz_cvttps_epi64,
	_mm_mask_cvtps_epu64, _mm_maskz_cvtps_epu64, _mm_mask_cvttps_epu64, _mm_maskz_cvttps_epu64,
	_mm_mask_cvtepi64_ps, _mm_maskz_cvtepi64_ps, _mm_mask_cvtepu64_ps, _mm_maskz_cvtepu64_ps,
	_mm256_mask_cvtps_epi64, _mm256_maskz_cvtps_epi64, _mm256_mask_cvttps_epi64, _mm256_maskz_cvttps_epi64,
	_mm256_mask_cvtps_epu64, _mm256_maskz_cvtps_epu64, _mm256_mask_cvttps_epu64, _mm256_maskz_cvttps_epu64,
	_mm256_mask_cvtepi64_ps, _mm256_maskz_cvtepi64_ps, _mm256_mask_cvtepu64_ps, _mm256_maskz_cvtepu64_ps,
};

use super::super::super::{Feature, FeatureSet};
use super::super::avx::avx_vnni::{vnni_acc_saturating, vnni_acc_wrapping};
use super::super::macros::{
	simd_binop, simd_binop_imm, simd_binop_imm_fixed, simd_binop_imm_masked, simd_binop_masked, simd_broadcast,
	simd_broadcast_masked, simd_cvt, simd_cvt_masked, simd_cvt_narrow, simd_cvt_narrow_masked, simd_cvt_widen,
	simd_cvt_widen_masked, simd_extract_imm, simd_extract_imm_masked, simd_insert_imm, simd_insert_imm_masked,
	simd_ternarylogic, simd_ternop, simd_ternop_masked, simd_unop, simd_unop_fixed, simd_unop_imm,
	simd_unop_imm_mask, simd_unop_imm_mask_gated, simd_unop_imm_masked, simd_vnni_dot, simd_vnni_dot_masked,
};
use super::avx512ifma::{madd52hi_scalar, madd52lo_scalar};
use super::avx512vbmi2::{
	shldi_i16_scalar, shldi_i32_scalar, shldi_i64_scalar, shldi_u16_scalar, shldi_u32_scalar, shldi_u64_scalar,
	shldv_i16_scalar, shldv_i32_scalar, shldv_i64_scalar, shldv_u16_scalar, shldv_u32_scalar, shldv_u64_scalar,
	shrdi_i16_scalar, shrdi_i32_scalar, shrdi_i64_scalar, shrdi_u16_scalar, shrdi_u32_scalar, shrdi_u64_scalar,
	shrdv_i16_scalar, shrdv_i32_scalar, shrdv_i64_scalar, shrdv_u16_scalar, shrdv_u32_scalar, shrdv_u64_scalar,
};

/// Proof token: AVX512BITALG *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512BitalgVl(());

impl Avx512BitalgVl {
	/// `None` unless the CPU has both AVX512BITALG and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512bitalg) && set.contains(Feature::Avx512vl)).then_some(Avx512BitalgVl(()))
	}
}

simd_unop! {
	token = Avx512BitalgVl, target_feature = "avx512bitalg,avx512vl",
	fixed_fn = popcnt_u8x16, slice_fn = popcnt_u8_slice, intrinsic_fn = popcnt_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_popcnt_epi8, scalar = |x: u8| x.count_ones() as u8,
	fixed_doc = "Per-lane population count (`vpopcntb`, 128-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 16-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512BitalgVl, target_feature = "avx512bitalg,avx512vl",
	fixed_fn = popcnt_u8x32, slice_fn = popcnt_u8_slice_wide, intrinsic_fn = popcnt_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_popcnt_epi8, scalar = |x: u8| x.count_ones() as u8,
	fixed_doc = "Per-lane population count (`vpopcntb`, 256-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 32-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512BitalgVl, target_feature = "avx512bitalg,avx512vl",
	fixed_fn = popcnt_u16x8, slice_fn = popcnt_u16_slice, intrinsic_fn = popcnt_u16x8_intrinsic,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_popcnt_epi16, scalar = |x: u16| x.count_ones() as u16,
	fixed_doc = "Per-lane population count (`vpopcntw`, 128-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 8-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512BitalgVl, target_feature = "avx512bitalg,avx512vl",
	fixed_fn = popcnt_u16x16, slice_fn = popcnt_u16_slice_wide, intrinsic_fn = popcnt_u16x16_intrinsic,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_popcnt_epi16, scalar = |x: u16| x.count_ones() as u16,
	fixed_doc = "Per-lane population count (`vpopcntw`, 256-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 16-wide chunks, scalar remainder.",
}

impl Avx512BitalgVl {
	/// 256-bit `vpshufbitqmb`: 4 qword lanes -> 32 mask bits.
	#[inline]
	pub fn bitshuffle_mask_u64x4(self, b: [u64; 4], c: [u64; 4]) -> u32 {
		unsafe { bitshuffle_qmb_256(&b, &c) }
	}

	/// 128-bit `vpshufbitqmb`: 2 qword lanes -> 16 mask bits.
	#[inline]
	pub fn bitshuffle_mask_u64x2(self, b: [u64; 2], c: [u64; 2]) -> u16 {
		unsafe { bitshuffle_qmb_128(&b, &c) }
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
unsafe fn bitshuffle_qmb_256(b: &[u64; 4], c: &[u64; 4]) -> u32 {
	unsafe {
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vc: __m256i = _mm256_loadu_si256(c.as_ptr().cast());
		_mm256_bitshuffle_epi64_mask(vb, vc)
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512bitalg,avx512vl")]
unsafe fn bitshuffle_qmb_128(b: &[u64; 2], c: &[u64; 2]) -> u16 {
	unsafe {
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vc: __m128i = _mm_loadu_si128(c.as_ptr().cast());
		_mm_bitshuffle_epi64_mask(vb, vc)
	}
}

/// Proof token: AVX-512DQ *and* AVX-512VL, for the 128/256-bit companions of
/// [`super::avx512dq::Avx512Dq`]: `mullo_epi64`, range/reduce/cvt f64<->i64/
/// u64, broadcast/extract/insert, and `fpclass`. Every family beyond
/// `mullo_epi64` is fixed-width-only (no `_slice`/`auto`): see
/// `ops/macros/cvt.rs`'s `simd_cvt` doc for why. Merge/zero-masked forms exist
/// for every family except `fpclass`, which gets one mask-gated form instead
/// (same reasoning as [`super::avx512dq::Avx512Dq`]'s doc).
#[derive(Debug, Clone, Copy)]
pub struct Avx512DqVl(());

impl Avx512DqVl {
	/// `None` unless the CPU has both AVX-512DQ and AVX-512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512dq) && set.contains(Feature::Avx512vl)).then_some(Avx512DqVl(()))
	}
}

simd_binop! {
	token = Avx512DqVl, vis = pub, target_feature = "avx512dq,avx512vl",
	fixed_fn = mullo_i64x2, slice_fn = mullo_i64_slice, intrinsic_fn = mullo_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_mullo_epi64, scalar = |x: i64, y: i64| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, low 64 bits (`vpmullq`, 128-bit).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 2-wide chunks, scalar remainder.",
}

simd_binop! {
	token = Avx512DqVl, vis = pub, target_feature = "avx512dq,avx512vl",
	fixed_fn = mullo_i64x4, slice_fn = mullo_i64_slice_wide, intrinsic_fn = mullo_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_mullo_epi64, scalar = |x: i64, y: i64| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, low 64 bits (`vpmullq`, 256-bit).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 4-wide chunks, scalar remainder.",
}

simd_binop! {
	token = Avx512DqVl, vis = pub, target_feature = "avx512dq,avx512vl",
	fixed_fn = mullo_u64x2, slice_fn = mullo_u64_slice, intrinsic_fn = mullo_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_mullo_epi64, scalar = |x: u64, y: u64| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, low 64 bits (`vpmullq`, 128-bit).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 2-wide chunks, scalar remainder.",
}

simd_binop! {
	token = Avx512DqVl, vis = pub, target_feature = "avx512dq,avx512vl",
	fixed_fn = mullo_u64x4, slice_fn = mullo_u64_slice_wide, intrinsic_fn = mullo_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_mullo_epi64, scalar = |x: u64, y: u64| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, low 64 bits (`vpmullq`, 256-bit).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 4-wide chunks, scalar remainder.",
}

simd_binop_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = mullo_i64x2_merge_masked, zero_fn = mullo_i64x2_zero_masked,
	merge_intrinsic_fn = mask_mullo_i64x2_intrinsic, zero_intrinsic_fn = maskz_mullo_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_mullo_epi64, zero_intrinsic = _mm_maskz_mullo_epi64,
	merge_doc = "[`Avx512DqVl::mullo_i64x2`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512DqVl::mullo_i64x2`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = mullo_i64x4_merge_masked, zero_fn = mullo_i64x4_zero_masked,
	merge_intrinsic_fn = mask_mullo_i64x4_intrinsic, zero_intrinsic_fn = maskz_mullo_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_mullo_epi64, zero_intrinsic = _mm256_maskz_mullo_epi64,
	merge_doc = "[`Avx512DqVl::mullo_i64x4`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512DqVl::mullo_i64x4`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = mullo_u64x2_merge_masked, zero_fn = mullo_u64x2_zero_masked,
	merge_intrinsic_fn = mask_mullo_u64x2_intrinsic, zero_intrinsic_fn = maskz_mullo_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_mullo_epi64, zero_intrinsic = _mm_maskz_mullo_epi64,
	merge_doc = "[`Avx512DqVl::mullo_u64x2`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512DqVl::mullo_u64x2`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = mullo_u64x4_merge_masked, zero_fn = mullo_u64x4_zero_masked,
	merge_intrinsic_fn = mask_mullo_u64x4_intrinsic, zero_intrinsic_fn = maskz_mullo_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_mullo_epi64, zero_intrinsic = _mm256_maskz_mullo_epi64,
	merge_doc = "[`Avx512DqVl::mullo_u64x4`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512DqVl::mullo_u64x4`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = i64_to_f64x2, intrinsic_fn = i64_to_f64x2_intrinsic,
	width = 2,
	in_elem = i64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f64, out_vec = __m128d, out_storeu = _mm_storeu_pd,
	intrinsic = _mm_cvtepi64_pd,
	fixed_doc = "Signed `i64` to `f64`, round-to-nearest-even (`vcvtqq2pd`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = i64_to_f64x4, intrinsic_fn = i64_to_f64x4_intrinsic,
	width = 4,
	in_elem = i64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f64, out_vec = __m256d, out_storeu = _mm256_storeu_pd,
	intrinsic = _mm256_cvtepi64_pd,
	fixed_doc = "Signed `i64` to `f64`, round-to-nearest-even (`vcvtqq2pd`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = u64_to_f64x2, intrinsic_fn = u64_to_f64x2_intrinsic,
	width = 2,
	in_elem = u64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f64, out_vec = __m128d, out_storeu = _mm_storeu_pd,
	intrinsic = _mm_cvtepu64_pd,
	fixed_doc = "Unsigned `u64` to `f64`, round-to-nearest-even (`vcvtuqq2pd`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = u64_to_f64x4, intrinsic_fn = u64_to_f64x4_intrinsic,
	width = 4,
	in_elem = u64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f64, out_vec = __m256d, out_storeu = _mm256_storeu_pd,
	intrinsic = _mm256_cvtepu64_pd,
	fixed_doc = "Unsigned `u64` to `f64`, round-to-nearest-even (`vcvtuqq2pd`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_i64x2, intrinsic_fn = f64_to_i64x2_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = i64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvtpd_epi64,
	fixed_doc = "`f64` to `i64`, round-to-nearest-even. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2qq`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_i64x4, intrinsic_fn = f64_to_i64x4_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = i64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvtpd_epi64,
	fixed_doc = "`f64` to `i64`, round-to-nearest-even. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2qq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_i64x2_trunc, intrinsic_fn = f64_to_i64x2_trunc_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = i64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvttpd_epi64,
	fixed_doc = "`f64` to `i64`, truncating toward zero. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2qq`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_i64x4_trunc, intrinsic_fn = f64_to_i64x4_trunc_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = i64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvttpd_epi64,
	fixed_doc = "`f64` to `i64`, truncating toward zero. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2qq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_u64x2, intrinsic_fn = f64_to_u64x2_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = u64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvtpd_epu64,
	fixed_doc = "`f64` to `u64`, round-to-nearest-even. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2uqq`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_u64x4, intrinsic_fn = f64_to_u64x4_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = u64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvtpd_epu64,
	fixed_doc = "`f64` to `u64`, round-to-nearest-even. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2uqq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_u64x2_trunc, intrinsic_fn = f64_to_u64x2_trunc_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = u64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvttpd_epu64,
	fixed_doc = "`f64` to `u64`, truncating toward zero. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2uqq`, 128-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f64_to_u64x4_trunc, intrinsic_fn = f64_to_u64x4_trunc_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = u64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvttpd_epu64,
	fixed_doc = "`f64` to `u64`, truncating toward zero. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2uqq`, 256-bit).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = i64_to_f64x2_merge_masked, zero_fn = i64_to_f64x2_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f64x2_intrinsic, zero_intrinsic_fn = maskz_i64_to_f64x2_intrinsic,
	width = 2,
	in_elem = i64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f64, out_vec = __m128d, out_loadu = _mm_loadu_pd, out_storeu = _mm_storeu_pd, mask = u8,
	merge_intrinsic = _mm_mask_cvtepi64_pd, zero_intrinsic = _mm_maskz_cvtepi64_pd,
	merge_doc = "[`Avx512DqVl::i64_to_f64x2`] where `mask` bit is set, else copied from `src` (`vcvtqq2pd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::i64_to_f64x2`] where `mask` bit is set, else zero (`vcvtqq2pd`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = i64_to_f64x4_merge_masked, zero_fn = i64_to_f64x4_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f64x4_intrinsic, zero_intrinsic_fn = maskz_i64_to_f64x4_intrinsic,
	width = 4,
	in_elem = i64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f64, out_vec = __m256d, out_loadu = _mm256_loadu_pd, out_storeu = _mm256_storeu_pd, mask = u8,
	merge_intrinsic = _mm256_mask_cvtepi64_pd, zero_intrinsic = _mm256_maskz_cvtepi64_pd,
	merge_doc = "[`Avx512DqVl::i64_to_f64x4`] where `mask` bit is set, else copied from `src` (`vcvtqq2pd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::i64_to_f64x4`] where `mask` bit is set, else zero (`vcvtqq2pd`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = u64_to_f64x2_merge_masked, zero_fn = u64_to_f64x2_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f64x2_intrinsic, zero_intrinsic_fn = maskz_u64_to_f64x2_intrinsic,
	width = 2,
	in_elem = u64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f64, out_vec = __m128d, out_loadu = _mm_loadu_pd, out_storeu = _mm_storeu_pd, mask = u8,
	merge_intrinsic = _mm_mask_cvtepu64_pd, zero_intrinsic = _mm_maskz_cvtepu64_pd,
	merge_doc = "[`Avx512DqVl::u64_to_f64x2`] where `mask` bit is set, else copied from `src` (`vcvtuqq2pd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::u64_to_f64x2`] where `mask` bit is set, else zero (`vcvtuqq2pd`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = u64_to_f64x4_merge_masked, zero_fn = u64_to_f64x4_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f64x4_intrinsic, zero_intrinsic_fn = maskz_u64_to_f64x4_intrinsic,
	width = 4,
	in_elem = u64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f64, out_vec = __m256d, out_loadu = _mm256_loadu_pd, out_storeu = _mm256_storeu_pd, mask = u8,
	merge_intrinsic = _mm256_mask_cvtepu64_pd, zero_intrinsic = _mm256_maskz_cvtepu64_pd,
	merge_doc = "[`Avx512DqVl::u64_to_f64x4`] where `mask` bit is set, else copied from `src` (`vcvtuqq2pd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::u64_to_f64x4`] where `mask` bit is set, else zero (`vcvtuqq2pd`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_i64x2_merge_masked, zero_fn = f64_to_i64x2_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x2_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x2_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = i64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvtpd_epi64, zero_intrinsic = _mm_maskz_cvtpd_epi64,
	merge_doc = "[`Avx512DqVl::f64_to_i64x2`] where `mask` bit is set, else copied from `src` (`vcvtpd2qq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_i64x2`] where `mask` bit is set, else zero (`vcvtpd2qq`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_i64x4_merge_masked, zero_fn = f64_to_i64x4_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x4_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x4_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = i64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvtpd_epi64, zero_intrinsic = _mm256_maskz_cvtpd_epi64,
	merge_doc = "[`Avx512DqVl::f64_to_i64x4`] where `mask` bit is set, else copied from `src` (`vcvtpd2qq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_i64x4`] where `mask` bit is set, else zero (`vcvtpd2qq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_i64x2_trunc_merge_masked, zero_fn = f64_to_i64x2_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x2_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x2_trunc_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = i64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvttpd_epi64, zero_intrinsic = _mm_maskz_cvttpd_epi64,
	merge_doc = "[`Avx512DqVl::f64_to_i64x2_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2qq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_i64x2_trunc`] where `mask` bit is set, else zero (`vcvttpd2qq`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_i64x4_trunc_merge_masked, zero_fn = f64_to_i64x4_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x4_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x4_trunc_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = i64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvttpd_epi64, zero_intrinsic = _mm256_maskz_cvttpd_epi64,
	merge_doc = "[`Avx512DqVl::f64_to_i64x4_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2qq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_i64x4_trunc`] where `mask` bit is set, else zero (`vcvttpd2qq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_u64x2_merge_masked, zero_fn = f64_to_u64x2_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x2_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x2_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = u64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvtpd_epu64, zero_intrinsic = _mm_maskz_cvtpd_epu64,
	merge_doc = "[`Avx512DqVl::f64_to_u64x2`] where `mask` bit is set, else copied from `src` (`vcvtpd2uqq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_u64x2`] where `mask` bit is set, else zero (`vcvtpd2uqq`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_u64x4_merge_masked, zero_fn = f64_to_u64x4_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x4_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x4_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = u64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvtpd_epu64, zero_intrinsic = _mm256_maskz_cvtpd_epu64,
	merge_doc = "[`Avx512DqVl::f64_to_u64x4`] where `mask` bit is set, else copied from `src` (`vcvtpd2uqq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_u64x4`] where `mask` bit is set, else zero (`vcvtpd2uqq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_u64x2_trunc_merge_masked, zero_fn = f64_to_u64x2_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x2_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x2_trunc_intrinsic,
	width = 2,
	in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
	out_elem = u64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvttpd_epu64, zero_intrinsic = _mm_maskz_cvttpd_epu64,
	merge_doc = "[`Avx512DqVl::f64_to_u64x2_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2uqq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_u64x2_trunc`] where `mask` bit is set, else zero (`vcvttpd2uqq`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f64_to_u64x4_trunc_merge_masked, zero_fn = f64_to_u64x4_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x4_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x4_trunc_intrinsic,
	width = 4,
	in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
	out_elem = u64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvttpd_epu64, zero_intrinsic = _mm256_maskz_cvttpd_epu64,
	merge_doc = "[`Avx512DqVl::f64_to_u64x4_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2uqq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f64_to_u64x4_trunc`] where `mask` bit is set, else zero (`vcvttpd2uqq`, 256-bit, zero-masked).",
}

// f32 (ps) <-> i64/u64. 256-bit has equal in/out lane counts, same shape as
// the f64 forms above. 128-bit reads/writes a full 4-lane `f32` register
// even though only 2 lanes convert (`vcvtps2qq`/`vcvtqq2ps` at 128-bit are
// genuinely lane-count-mismatched) -> `simd_cvt_widen`/`simd_cvt_narrow`,
// same shape as FP16's `cvtph_pd`/`cvtpd_ph`.

simd_cvt_widen! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_i64x2, intrinsic_fn = f32_to_i64x2_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvtps_epi64,
	fixed_doc = "`f32` to `i64`, round-to-nearest-even. Reads a full 4-lane `f32` register, only the low 2 lanes convert (`vcvtps2qq`, 128-bit).",
}

simd_cvt_widen! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_i64x2_trunc, intrinsic_fn = f32_to_i64x2_trunc_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvttps_epi64,
	fixed_doc = "`f32` to `i64`, truncating toward zero, low 2 lanes only (`vcvttps2qq`, 128-bit). See [`Avx512DqVl::f32_to_i64x2`].",
}

simd_cvt_widen! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_u64x2, intrinsic_fn = f32_to_u64x2_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvtps_epu64,
	fixed_doc = "`f32` to `u64`, round-to-nearest-even, low 2 lanes only (`vcvtps2uqq`, 128-bit). See [`Avx512DqVl::f32_to_i64x2`].",
}

simd_cvt_widen! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_u64x2_trunc, intrinsic_fn = f32_to_u64x2_trunc_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m128i, out_storeu = _mm_storeu_si128,
	intrinsic = _mm_cvttps_epu64,
	fixed_doc = "`f32` to `u64`, truncating toward zero, low 2 lanes only (`vcvttps2uqq`, 128-bit). See [`Avx512DqVl::f32_to_i64x2`].",
}

simd_cvt_narrow! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = i64_to_f32x2, intrinsic_fn = i64_to_f32x2_intrinsic,
	in_width = 2, out_width = 4,
	in_elem = i64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
	intrinsic = _mm_cvtepi64_ps,
	fixed_doc = "Signed `i64` to `f32`, round-to-nearest-even, into the low 2 lanes of a 4-lane `f32` register, upper 2 lanes zeroed (`vcvtqq2ps`, 128-bit).",
}

simd_cvt_narrow! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = u64_to_f32x2, intrinsic_fn = u64_to_f32x2_intrinsic,
	in_width = 2, out_width = 4,
	in_elem = u64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
	intrinsic = _mm_cvtepu64_ps,
	fixed_doc = "Unsigned `u64` to `f32`, round-to-nearest-even, low 2 lanes only, upper 2 zeroed (`vcvtuqq2ps`, 128-bit). See [`Avx512DqVl::i64_to_f32x2`].",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_i64x4, intrinsic_fn = f32_to_i64x4_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvtps_epi64,
	fixed_doc = "`f32` to `i64`, round-to-nearest-even (`vcvtps2qq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_i64x4_trunc, intrinsic_fn = f32_to_i64x4_trunc_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvttps_epi64,
	fixed_doc = "`f32` to `i64`, truncating toward zero (`vcvttps2qq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_u64x4, intrinsic_fn = f32_to_u64x4_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvtps_epu64,
	fixed_doc = "`f32` to `u64`, round-to-nearest-even (`vcvtps2uqq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = f32_to_u64x4_trunc, intrinsic_fn = f32_to_u64x4_trunc_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
	intrinsic = _mm256_cvttps_epu64,
	fixed_doc = "`f32` to `u64`, truncating toward zero (`vcvttps2uqq`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = i64_to_f32x4, intrinsic_fn = i64_to_f32x4_intrinsic,
	width = 4,
	in_elem = i64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
	intrinsic = _mm256_cvtepi64_ps,
	fixed_doc = "Signed `i64` to `f32`, round-to-nearest-even (`vcvtqq2ps`, 256-bit).",
}

simd_cvt! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = u64_to_f32x4, intrinsic_fn = u64_to_f32x4_intrinsic,
	width = 4,
	in_elem = u64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
	intrinsic = _mm256_cvtepu64_ps,
	fixed_doc = "Unsigned `u64` to `f32`, round-to-nearest-even (`vcvtuqq2ps`, 256-bit).",
}

simd_cvt_widen_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_i64x2_merge_masked, zero_fn = f32_to_i64x2_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x2_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x2_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvtps_epi64, zero_intrinsic = _mm_maskz_cvtps_epi64,
	merge_doc = "[`Avx512DqVl::f32_to_i64x2`] where `mask` bit is set, else copied from `src` (`vcvtps2qq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_i64x2`] where `mask` bit is set, else zero (`vcvtps2qq`, 128-bit, zero-masked).",
}

simd_cvt_widen_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_i64x2_trunc_merge_masked, zero_fn = f32_to_i64x2_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x2_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x2_trunc_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvttps_epi64, zero_intrinsic = _mm_maskz_cvttps_epi64,
	merge_doc = "[`Avx512DqVl::f32_to_i64x2_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2qq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_i64x2_trunc`] where `mask` bit is set, else zero (`vcvttps2qq`, 128-bit, zero-masked).",
}

simd_cvt_widen_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_u64x2_merge_masked, zero_fn = f32_to_u64x2_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x2_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x2_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvtps_epu64, zero_intrinsic = _mm_maskz_cvtps_epu64,
	merge_doc = "[`Avx512DqVl::f32_to_u64x2`] where `mask` bit is set, else copied from `src` (`vcvtps2uqq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_u64x2`] where `mask` bit is set, else zero (`vcvtps2uqq`, 128-bit, zero-masked).",
}

simd_cvt_widen_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_u64x2_trunc_merge_masked, zero_fn = f32_to_u64x2_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x2_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x2_trunc_intrinsic,
	in_width = 4, out_width = 2,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m128i, out_loadu = _mm_loadu_si128, out_storeu = _mm_storeu_si128, mask = u8,
	merge_intrinsic = _mm_mask_cvttps_epu64, zero_intrinsic = _mm_maskz_cvttps_epu64,
	merge_doc = "[`Avx512DqVl::f32_to_u64x2_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2uqq`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_u64x2_trunc`] where `mask` bit is set, else zero (`vcvttps2uqq`, 128-bit, zero-masked).",
}

simd_cvt_narrow_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = i64_to_f32x2_merge_masked, zero_fn = i64_to_f32x2_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f32x2_intrinsic, zero_intrinsic_fn = maskz_i64_to_f32x2_intrinsic,
	in_width = 2, out_width = 4,
	in_elem = i64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f32, out_vec = __m128, out_loadu = _mm_loadu_ps, out_storeu = _mm_storeu_ps, mask = u8,
	merge_intrinsic = _mm_mask_cvtepi64_ps, zero_intrinsic = _mm_maskz_cvtepi64_ps,
	merge_doc = "[`Avx512DqVl::i64_to_f32x2`], lanes 0-1 where `mask` bit is set else copied from `src`; lanes 2-3 always zero, not `src`-controlled (`vcvtqq2ps`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::i64_to_f32x2`], lanes 0-1 where `mask` bit is set else zero; lanes 2-3 always zero (`vcvtqq2ps`, 128-bit, zero-masked).",
}

simd_cvt_narrow_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = u64_to_f32x2_merge_masked, zero_fn = u64_to_f32x2_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f32x2_intrinsic, zero_intrinsic_fn = maskz_u64_to_f32x2_intrinsic,
	in_width = 2, out_width = 4,
	in_elem = u64, in_vec = __m128i, in_loadu = _mm_loadu_si128,
	out_elem = f32, out_vec = __m128, out_loadu = _mm_loadu_ps, out_storeu = _mm_storeu_ps, mask = u8,
	merge_intrinsic = _mm_mask_cvtepu64_ps, zero_intrinsic = _mm_maskz_cvtepu64_ps,
	merge_doc = "[`Avx512DqVl::u64_to_f32x2`], lanes 0-1 where `mask` bit is set else copied from `src`; lanes 2-3 always zero, not `src`-controlled (`vcvtuqq2ps`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::u64_to_f32x2`], lanes 0-1 where `mask` bit is set else zero; lanes 2-3 always zero (`vcvtuqq2ps`, 128-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_i64x4_merge_masked, zero_fn = f32_to_i64x4_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x4_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x4_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvtps_epi64, zero_intrinsic = _mm256_maskz_cvtps_epi64,
	merge_doc = "[`Avx512DqVl::f32_to_i64x4`] where `mask` bit is set, else copied from `src` (`vcvtps2qq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_i64x4`] where `mask` bit is set, else zero (`vcvtps2qq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_i64x4_trunc_merge_masked, zero_fn = f32_to_i64x4_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x4_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x4_trunc_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = i64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvttps_epi64, zero_intrinsic = _mm256_maskz_cvttps_epi64,
	merge_doc = "[`Avx512DqVl::f32_to_i64x4_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2qq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_i64x4_trunc`] where `mask` bit is set, else zero (`vcvttps2qq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_u64x4_merge_masked, zero_fn = f32_to_u64x4_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x4_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x4_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvtps_epu64, zero_intrinsic = _mm256_maskz_cvtps_epu64,
	merge_doc = "[`Avx512DqVl::f32_to_u64x4`] where `mask` bit is set, else copied from `src` (`vcvtps2uqq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_u64x4`] where `mask` bit is set, else zero (`vcvtps2uqq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = f32_to_u64x4_trunc_merge_masked, zero_fn = f32_to_u64x4_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x4_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x4_trunc_intrinsic,
	width = 4,
	in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
	out_elem = u64, out_vec = __m256i, out_loadu = _mm256_loadu_si256, out_storeu = _mm256_storeu_si256, mask = u8,
	merge_intrinsic = _mm256_mask_cvttps_epu64, zero_intrinsic = _mm256_maskz_cvttps_epu64,
	merge_doc = "[`Avx512DqVl::f32_to_u64x4_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2uqq`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::f32_to_u64x4_trunc`] where `mask` bit is set, else zero (`vcvttps2uqq`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = i64_to_f32x4_merge_masked, zero_fn = i64_to_f32x4_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f32x4_intrinsic, zero_intrinsic_fn = maskz_i64_to_f32x4_intrinsic,
	width = 4,
	in_elem = i64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f32, out_vec = __m128, out_loadu = _mm_loadu_ps, out_storeu = _mm_storeu_ps, mask = u8,
	merge_intrinsic = _mm256_mask_cvtepi64_ps, zero_intrinsic = _mm256_maskz_cvtepi64_ps,
	merge_doc = "[`Avx512DqVl::i64_to_f32x4`] where `mask` bit is set, else copied from `src` (`vcvtqq2ps`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::i64_to_f32x4`] where `mask` bit is set, else zero (`vcvtqq2ps`, 256-bit, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = u64_to_f32x4_merge_masked, zero_fn = u64_to_f32x4_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f32x4_intrinsic, zero_intrinsic_fn = maskz_u64_to_f32x4_intrinsic,
	width = 4,
	in_elem = u64, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
	out_elem = f32, out_vec = __m128, out_loadu = _mm_loadu_ps, out_storeu = _mm_storeu_ps, mask = u8,
	merge_intrinsic = _mm256_mask_cvtepu64_ps, zero_intrinsic = _mm256_maskz_cvtepu64_ps,
	merge_doc = "[`Avx512DqVl::u64_to_f32x4`] where `mask` bit is set, else copied from `src` (`vcvtuqq2ps`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::u64_to_f32x4`] where `mask` bit is set, else zero (`vcvtuqq2ps`, 256-bit, zero-masked).",
}

simd_binop_imm_fixed! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = range_f64x2, intrinsic_fn = range_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	intrinsic = _mm_range_pd,
	fixed_doc = "`a`/`b` combined per lane - same `IMM8` encoding as [`super::avx512dq::Avx512Dq::range_f64x8`] (`vrangepd`, 128-bit).",
}

simd_binop_imm_fixed! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = range_f64x4, intrinsic_fn = range_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	intrinsic = _mm256_range_pd,
	fixed_doc = "`a`/`b` combined per lane - same `IMM8` encoding as [`super::avx512dq::Avx512Dq::range_f64x8`] (`vrangepd`, 256-bit).",
}

simd_binop_imm_fixed! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = range_f32x4, intrinsic_fn = range_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_range_ps,
	fixed_doc = "`a`/`b` combined per lane - same `IMM8` encoding as [`super::avx512dq::Avx512Dq::range_f64x8`] (`vrangeps`, 128-bit).",
}

simd_binop_imm_fixed! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = range_f32x8, intrinsic_fn = range_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	intrinsic = _mm256_range_ps,
	fixed_doc = "`a`/`b` combined per lane - same `IMM8` encoding as [`super::avx512dq::Avx512Dq::range_f64x8`] (`vrangeps`, 256-bit).",
}

simd_unop_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = reduce_f64x2, intrinsic_fn = reduce_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	intrinsic = _mm_reduce_pd,
	fixed_doc = "Argument reduction, same `IMM8` encoding as [`super::avx512dq::Avx512Dq::reduce_f64x8`] (`vreducepd`, 128-bit).",
}

simd_unop_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = reduce_f64x4, intrinsic_fn = reduce_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	intrinsic = _mm256_reduce_pd,
	fixed_doc = "Argument reduction, same `IMM8` encoding as [`super::avx512dq::Avx512Dq::reduce_f64x8`] (`vreducepd`, 256-bit).",
}

simd_unop_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = reduce_f32x4, intrinsic_fn = reduce_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_reduce_ps,
	fixed_doc = "Argument reduction, same `IMM8` encoding as [`super::avx512dq::Avx512Dq::reduce_f64x8`] (`vreduceps`, 128-bit).",
}

simd_unop_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = reduce_f32x8, intrinsic_fn = reduce_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	intrinsic = _mm256_reduce_ps,
	fixed_doc = "Argument reduction, same `IMM8` encoding as [`super::avx512dq::Avx512Dq::reduce_f64x8`] (`vreduceps`, 256-bit).",
}

simd_unop_imm_mask! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f64x2, intrinsic_fn = fpclass_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, intrinsic = _mm_fpclass_pd_mask,
	fixed_doc = "Per-lane category test, same `IMM8` bit encoding as [`super::avx512dq::Avx512Dq::fpclass_f64x8`] (`vfpclasspd`, 128-bit).",
}

simd_unop_imm_mask! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f64x4, intrinsic_fn = fpclass_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, mask = u8,
	loadu = _mm256_loadu_pd, intrinsic = _mm256_fpclass_pd_mask,
	fixed_doc = "Per-lane category test, same `IMM8` bit encoding as [`super::avx512dq::Avx512Dq::fpclass_f64x8`] (`vfpclasspd`, 256-bit).",
}

simd_unop_imm_mask! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f32x4, intrinsic_fn = fpclass_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, intrinsic = _mm_fpclass_ps_mask,
	fixed_doc = "Per-lane category test, same `IMM8` bit encoding as [`super::avx512dq::Avx512Dq::fpclass_f64x8`] (`vfpclassps`, 128-bit).",
}

simd_unop_imm_mask! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f32x8, intrinsic_fn = fpclass_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, mask = u8,
	loadu = _mm256_loadu_ps, intrinsic = _mm256_fpclass_ps_mask,
	fixed_doc = "Per-lane category test, same `IMM8` bit encoding as [`super::avx512dq::Avx512Dq::fpclass_f64x8`] (`vfpclassps`, 256-bit).",
}

simd_broadcast! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = broadcast_f32x2_to_x8, intrinsic_fn = broadcast_f32x2_to_x8_intrinsic,
	narrow_width = 4, wide_width = 8, elem = f32, narrow_vec = __m128, wide_vec = __m256,
	narrow_loadu = _mm_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_broadcast_f32x2,
	fixed_doc = "Broadcasts `a`'s lower 2 `f32` lanes across all 8 output lanes; `a`'s upper 2 lanes (of its 4-lane `__m128` load) are ignored (`vbroadcastf32x2`, 256-bit).",
}

simd_broadcast! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = broadcast_i32x2_to_x8, intrinsic_fn = broadcast_i32x2_to_x8_intrinsic,
	narrow_width = 4, wide_width = 8, elem = i32, narrow_vec = __m128i, wide_vec = __m256i,
	narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256, intrinsic = _mm256_broadcast_i32x2,
	fixed_doc = "Broadcasts `a`'s lower 2 `i32` lanes across all 8 output lanes; `a`'s upper 2 lanes are ignored (`vbroadcasti32x2`, 256-bit).",
}

simd_broadcast! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = broadcast_f64x2_to_x4, intrinsic_fn = broadcast_f64x2_to_x4_intrinsic,
	narrow_width = 2, wide_width = 4, elem = f64, narrow_vec = __m128d, wide_vec = __m256d,
	narrow_loadu = _mm_loadu_pd, storeu = _mm256_storeu_pd, intrinsic = _mm256_broadcast_f64x2,
	fixed_doc = "Broadcasts `a`'s 2 `f64` lanes across all 4 output lanes (`vbroadcastf64x2`, 256-bit).",
}

simd_broadcast! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = broadcast_i64x2_to_x4, intrinsic_fn = broadcast_i64x2_to_x4_intrinsic,
	narrow_width = 2, wide_width = 4, elem = i64, narrow_vec = __m128i, wide_vec = __m256i,
	narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256, intrinsic = _mm256_broadcast_i64x2,
	fixed_doc = "Broadcasts `a`'s 2 `i64` lanes across all 4 output lanes (`vbroadcasti64x2`, 256-bit).",
}

simd_extract_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = extract_f64x2_from_x4, intrinsic_fn = extract_f64x2_from_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = f64, wide_vec = __m256d, narrow_vec = __m128d,
	wide_loadu = _mm256_loadu_pd, storeu = _mm_storeu_pd, intrinsic = _mm256_extractf64x2_pd,
	fixed_doc = "Extracts the `IMM8 & 1`-selected 2-lane half of `a` (`vextractf64x2`, 256-bit source).",
}

simd_extract_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = extract_i64x2_from_x4, intrinsic_fn = extract_i64x2_from_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = i64, wide_vec = __m256i, narrow_vec = __m128i,
	wide_loadu = _mm256_loadu_si256, storeu = _mm_storeu_si128, intrinsic = _mm256_extracti64x2_epi64,
	fixed_doc = "Extracts the `IMM8 & 1`-selected 2-lane half of `a` (`vextracti64x2`, 256-bit source).",
}

simd_insert_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = insert_f64x2_into_x4, intrinsic_fn = insert_f64x2_into_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = f64, wide_vec = __m256d, narrow_vec = __m128d,
	wide_loadu = _mm256_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm256_storeu_pd,
	intrinsic = _mm256_insertf64x2,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 1`-selected 2-lane half with `b` (`vinsertf64x2`, 256-bit).",
}

simd_insert_imm! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = insert_i64x2_into_x4, intrinsic_fn = insert_i64x2_into_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = i64, wide_vec = __m256i, narrow_vec = __m128i,
	wide_loadu = _mm256_loadu_si256, narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_inserti64x2,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 1`-selected 2-lane half with `b` (`vinserti64x2`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = range_f64x2_merge_masked, zero_fn = range_f64x2_zero_masked,
	merge_intrinsic_fn = mask_range_f64x2_intrinsic, zero_intrinsic_fn = maskz_range_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	merge_intrinsic = _mm_mask_range_pd, zero_intrinsic = _mm_maskz_range_pd,
	merge_doc = "[`Avx512DqVl::range_f64x2`] where `mask` bit is set, else copied from `src` (`vrangepd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::range_f64x2`] where `mask` bit is set, else zero (`vrangepd`, 128-bit, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = range_f64x4_merge_masked, zero_fn = range_f64x4_zero_masked,
	merge_intrinsic_fn = mask_range_f64x4_intrinsic, zero_intrinsic_fn = maskz_range_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, mask = u8,
	loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	merge_intrinsic = _mm256_mask_range_pd, zero_intrinsic = _mm256_maskz_range_pd,
	merge_doc = "[`Avx512DqVl::range_f64x4`] where `mask` bit is set, else copied from `src` (`vrangepd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::range_f64x4`] where `mask` bit is set, else zero (`vrangepd`, 256-bit, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = range_f32x4_merge_masked, zero_fn = range_f32x4_zero_masked,
	merge_intrinsic_fn = mask_range_f32x4_intrinsic, zero_intrinsic_fn = maskz_range_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	merge_intrinsic = _mm_mask_range_ps, zero_intrinsic = _mm_maskz_range_ps,
	merge_doc = "[`Avx512DqVl::range_f32x4`] where `mask` bit is set, else copied from `src` (`vrangeps`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::range_f32x4`] where `mask` bit is set, else zero (`vrangeps`, 128-bit, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = range_f32x8_merge_masked, zero_fn = range_f32x8_zero_masked,
	merge_intrinsic_fn = mask_range_f32x8_intrinsic, zero_intrinsic_fn = maskz_range_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, mask = u8,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	merge_intrinsic = _mm256_mask_range_ps, zero_intrinsic = _mm256_maskz_range_ps,
	merge_doc = "[`Avx512DqVl::range_f32x8`] where `mask` bit is set, else copied from `src` (`vrangeps`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::range_f32x8`] where `mask` bit is set, else zero (`vrangeps`, 256-bit, zero-masked).",
}

simd_unop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = reduce_f64x2_merge_masked, zero_fn = reduce_f64x2_zero_masked,
	merge_intrinsic_fn = mask_reduce_f64x2_intrinsic, zero_intrinsic_fn = maskz_reduce_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	merge_intrinsic = _mm_mask_reduce_pd, zero_intrinsic = _mm_maskz_reduce_pd,
	merge_doc = "[`Avx512DqVl::reduce_f64x2`] where `mask` bit is set, else copied from `src` (`vreducepd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::reduce_f64x2`] where `mask` bit is set, else zero (`vreducepd`, 128-bit, zero-masked).",
}

simd_unop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = reduce_f64x4_merge_masked, zero_fn = reduce_f64x4_zero_masked,
	merge_intrinsic_fn = mask_reduce_f64x4_intrinsic, zero_intrinsic_fn = maskz_reduce_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, mask = u8,
	loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	merge_intrinsic = _mm256_mask_reduce_pd, zero_intrinsic = _mm256_maskz_reduce_pd,
	merge_doc = "[`Avx512DqVl::reduce_f64x4`] where `mask` bit is set, else copied from `src` (`vreducepd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::reduce_f64x4`] where `mask` bit is set, else zero (`vreducepd`, 256-bit, zero-masked).",
}

simd_unop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = reduce_f32x4_merge_masked, zero_fn = reduce_f32x4_zero_masked,
	merge_intrinsic_fn = mask_reduce_f32x4_intrinsic, zero_intrinsic_fn = maskz_reduce_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	merge_intrinsic = _mm_mask_reduce_ps, zero_intrinsic = _mm_maskz_reduce_ps,
	merge_doc = "[`Avx512DqVl::reduce_f32x4`] where `mask` bit is set, else copied from `src` (`vreduceps`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::reduce_f32x4`] where `mask` bit is set, else zero (`vreduceps`, 128-bit, zero-masked).",
}

simd_unop_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = reduce_f32x8_merge_masked, zero_fn = reduce_f32x8_zero_masked,
	merge_intrinsic_fn = mask_reduce_f32x8_intrinsic, zero_intrinsic_fn = maskz_reduce_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, mask = u8,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	merge_intrinsic = _mm256_mask_reduce_ps, zero_intrinsic = _mm256_maskz_reduce_ps,
	merge_doc = "[`Avx512DqVl::reduce_f32x8`] where `mask` bit is set, else copied from `src` (`vreduceps`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512DqVl::reduce_f32x8`] where `mask` bit is set, else zero (`vreduceps`, 256-bit, zero-masked).",
}

simd_unop_imm_mask_gated! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f64x2_gated, intrinsic_fn = mask_fpclass_f64x2_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, intrinsic = _mm_mask_fpclass_pd_mask,
	fixed_doc = "[`Avx512DqVl::fpclass_f64x2`] ANDed with `k1` (`vfpclasspd`, 128-bit, mask-gated): `fpclass_f64x2::<IMM8>(a) & k1`.",
}

simd_unop_imm_mask_gated! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f64x4_gated, intrinsic_fn = mask_fpclass_f64x4_intrinsic,
	width = 4, elem = f64, vec = __m256d, mask = u8,
	loadu = _mm256_loadu_pd, intrinsic = _mm256_mask_fpclass_pd_mask,
	fixed_doc = "[`Avx512DqVl::fpclass_f64x4`] ANDed with `k1` (`vfpclasspd`, 256-bit, mask-gated): `fpclass_f64x4::<IMM8>(a) & k1`.",
}

simd_unop_imm_mask_gated! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f32x4_gated, intrinsic_fn = mask_fpclass_f32x4_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, intrinsic = _mm_mask_fpclass_ps_mask,
	fixed_doc = "[`Avx512DqVl::fpclass_f32x4`] ANDed with `k1` (`vfpclassps`, 128-bit, mask-gated): `fpclass_f32x4::<IMM8>(a) & k1`.",
}

simd_unop_imm_mask_gated! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	fixed_fn = fpclass_f32x8_gated, intrinsic_fn = mask_fpclass_f32x8_intrinsic,
	width = 8, elem = f32, vec = __m256, mask = u8,
	loadu = _mm256_loadu_ps, intrinsic = _mm256_mask_fpclass_ps_mask,
	fixed_doc = "[`Avx512DqVl::fpclass_f32x8`] ANDed with `k1` (`vfpclassps`, 256-bit, mask-gated): `fpclass_f32x8::<IMM8>(a) & k1`.",
}

simd_broadcast_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = broadcast_f32x2_to_x8_merge_masked, zero_fn = broadcast_f32x2_to_x8_zero_masked,
	merge_intrinsic_fn = mask_broadcast_f32x2_to_x8_intrinsic, zero_intrinsic_fn = maskz_broadcast_f32x2_to_x8_intrinsic,
	narrow_width = 4, wide_width = 8, elem = f32, narrow_vec = __m128, wide_vec = __m256, mask = u8,
	narrow_loadu = _mm_loadu_ps, wide_loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	merge_intrinsic = _mm256_mask_broadcast_f32x2, zero_intrinsic = _mm256_maskz_broadcast_f32x2,
	merge_doc = "[`Avx512DqVl::broadcast_f32x2_to_x8`] where `mask` bit is set, else copied from `src` (`vbroadcastf32x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::broadcast_f32x2_to_x8`] where `mask` bit is set, else zero (`vbroadcastf32x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = broadcast_i32x2_to_x8_merge_masked, zero_fn = broadcast_i32x2_to_x8_zero_masked,
	merge_intrinsic_fn = mask_broadcast_i32x2_to_x8_intrinsic, zero_intrinsic_fn = maskz_broadcast_i32x2_to_x8_intrinsic,
	narrow_width = 4, wide_width = 8, elem = i32, narrow_vec = __m128i, wide_vec = __m256i, mask = u8,
	narrow_loadu = _mm_loadu_si128, wide_loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_broadcast_i32x2, zero_intrinsic = _mm256_maskz_broadcast_i32x2,
	merge_doc = "[`Avx512DqVl::broadcast_i32x2_to_x8`] where `mask` bit is set, else copied from `src` (`vbroadcasti32x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::broadcast_i32x2_to_x8`] where `mask` bit is set, else zero (`vbroadcasti32x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = broadcast_f64x2_to_x4_merge_masked, zero_fn = broadcast_f64x2_to_x4_zero_masked,
	merge_intrinsic_fn = mask_broadcast_f64x2_to_x4_intrinsic, zero_intrinsic_fn = maskz_broadcast_f64x2_to_x4_intrinsic,
	narrow_width = 2, wide_width = 4, elem = f64, narrow_vec = __m128d, wide_vec = __m256d, mask = u8,
	narrow_loadu = _mm_loadu_pd, wide_loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	merge_intrinsic = _mm256_mask_broadcast_f64x2, zero_intrinsic = _mm256_maskz_broadcast_f64x2,
	merge_doc = "[`Avx512DqVl::broadcast_f64x2_to_x4`] where `mask` bit is set, else copied from `src` (`vbroadcastf64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::broadcast_f64x2_to_x4`] where `mask` bit is set, else zero (`vbroadcastf64x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = broadcast_i64x2_to_x4_merge_masked, zero_fn = broadcast_i64x2_to_x4_zero_masked,
	merge_intrinsic_fn = mask_broadcast_i64x2_to_x4_intrinsic, zero_intrinsic_fn = maskz_broadcast_i64x2_to_x4_intrinsic,
	narrow_width = 2, wide_width = 4, elem = i64, narrow_vec = __m128i, wide_vec = __m256i, mask = u8,
	narrow_loadu = _mm_loadu_si128, wide_loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_broadcast_i64x2, zero_intrinsic = _mm256_maskz_broadcast_i64x2,
	merge_doc = "[`Avx512DqVl::broadcast_i64x2_to_x4`] where `mask` bit is set, else copied from `src` (`vbroadcasti64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::broadcast_i64x2_to_x4`] where `mask` bit is set, else zero (`vbroadcasti64x2`, zero-masked).",
}

simd_extract_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = extract_f64x2_from_x4_merge_masked, zero_fn = extract_f64x2_from_x4_zero_masked,
	merge_intrinsic_fn = mask_extract_f64x2_from_x4_intrinsic, zero_intrinsic_fn = maskz_extract_f64x2_from_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = f64, wide_vec = __m256d, narrow_vec = __m128d, mask = u8,
	wide_loadu = _mm256_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	merge_intrinsic = _mm256_mask_extractf64x2_pd, zero_intrinsic = _mm256_maskz_extractf64x2_pd,
	merge_doc = "[`Avx512DqVl::extract_f64x2_from_x4`] where `mask` bit is set, else copied from `src` (`vextractf64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::extract_f64x2_from_x4`] where `mask` bit is set, else zero (`vextractf64x2`, zero-masked).",
}

simd_extract_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = extract_i64x2_from_x4_merge_masked, zero_fn = extract_i64x2_from_x4_zero_masked,
	merge_intrinsic_fn = mask_extract_i64x2_from_x4_intrinsic, zero_intrinsic_fn = maskz_extract_i64x2_from_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = i64, wide_vec = __m256i, narrow_vec = __m128i, mask = u8,
	wide_loadu = _mm256_loadu_si256, narrow_loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm256_mask_extracti64x2_epi64, zero_intrinsic = _mm256_maskz_extracti64x2_epi64,
	merge_doc = "[`Avx512DqVl::extract_i64x2_from_x4`] where `mask` bit is set, else copied from `src` (`vextracti64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::extract_i64x2_from_x4`] where `mask` bit is set, else zero (`vextracti64x2`, zero-masked).",
}

simd_insert_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = insert_f64x2_into_x4_merge_masked, zero_fn = insert_f64x2_into_x4_zero_masked,
	merge_intrinsic_fn = mask_insert_f64x2_into_x4_intrinsic, zero_intrinsic_fn = maskz_insert_f64x2_into_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = f64, wide_vec = __m256d, narrow_vec = __m128d, mask = u8,
	wide_loadu = _mm256_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm256_storeu_pd,
	merge_intrinsic = _mm256_mask_insertf64x2, zero_intrinsic = _mm256_maskz_insertf64x2,
	merge_doc = "[`Avx512DqVl::insert_f64x2_into_x4`] where `mask` bit is set, else copied from `src` (`vinsertf64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::insert_f64x2_into_x4`] where `mask` bit is set, else zero (`vinsertf64x2`, zero-masked).",
}

simd_insert_imm_masked! {
	token = Avx512DqVl, target_feature = "avx512dq,avx512vl",
	merge_fn = insert_i64x2_into_x4_merge_masked, zero_fn = insert_i64x2_into_x4_zero_masked,
	merge_intrinsic_fn = mask_insert_i64x2_into_x4_intrinsic, zero_intrinsic_fn = maskz_insert_i64x2_into_x4_intrinsic,
	wide_width = 4, narrow_width = 2, elem = i64, wide_vec = __m256i, narrow_vec = __m128i, mask = u8,
	wide_loadu = _mm256_loadu_si256, narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_inserti64x2, zero_intrinsic = _mm256_maskz_inserti64x2,
	merge_doc = "[`Avx512DqVl::insert_i64x2_into_x4`] where `mask` bit is set, else copied from `src` (`vinserti64x2`, merge-masked).",
	zero_doc = "[`Avx512DqVl::insert_i64x2_into_x4`] where `mask` bit is set, else zero (`vinserti64x2`, zero-masked).",
}

/// Proof token: AVX-512F *and* AVX-512VL: merge/zero-masked arithmetic at
/// 128/256-bit. Unlike every other token in this file, plain (unmasked)
/// `f32`/`f64`/`i32`/`u32`/`i64`/`u64` add/sub/mul/div/min/max at these
/// widths already exist on `Sse`/`Sse41`/`Avx`/`Avx2` via AVX2/SSE: no
/// AVX-512 needed there. This token exists solely because *masking* is an
/// AVX-512-only concept: there is no k-register on AVX2, so the merge/zero
/// forms have no pre-AVX-512 equivalent to live on instead (same shape as
/// [`super::avx512f`]'s 512-bit masked family: see
/// `simd_binop_masked` doc). All ops confirmed `__mmask8` regardless of
/// lane count (stdarch: even 8-lane `f32x8` uses `__mmask8`, not a wider
/// mask type: AVX-512 mask registers are always at least 8 bits).
///
/// One exception to the "masking-only" rule: `ternarylogic` (`vpternlogd`/
/// `vpternlogq`) has no pre-AVX-512 equivalent at all, unmasked or masked,
/// unlike the arithmetic families above: this token carries its unmasked
/// form too, not just merge/zero.
#[derive(Debug, Clone, Copy)]
pub struct Avx512FVl(());

impl Avx512FVl {
	/// `None` unless the CPU has both AVX-512F and AVX-512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512f) && set.contains(Feature::Avx512vl)).then_some(Avx512FVl(()))
	}
}

macro_rules! avx512f_vl_f32x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = f32, vec = __m128, mask = u8,
			loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_f32x8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = f32, vec = __m256, mask = u8,
			loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_f64x2_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 2, elem = f64, vec = __m128d, mask = u8,
			loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_f64x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = f64, vec = __m256d, mask = u8,
			loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_i32x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = i32, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_i32x8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = i32, vec = __m256i, mask = u8,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_u32x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = u32, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_u32x8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u32, vec = __m256i, mask = u8,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_i64x2_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 2, elem = i64, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_i64x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = i64, vec = __m256i, mask = u8,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_u64x2_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 2, elem = u64, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512f_vl_u64x4_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512FVl, target_feature = "avx512f,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 4, elem = u64, vec = __m256i, mask = u8,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512f_vl_f32x4_binop_masked!(
	add_f32x4_merge_masked, add_f32x4_zero_masked, mask_add_ps128_intrinsic, maskz_add_ps128_intrinsic,
	_mm_mask_add_ps, _mm_maskz_add_ps,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddps`, 128-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddps`, 128-bit, zero-masked)."
);
avx512f_vl_f32x4_binop_masked!(
	sub_f32x4_merge_masked, sub_f32x4_zero_masked, mask_sub_ps128_intrinsic, maskz_sub_ps128_intrinsic,
	_mm_mask_sub_ps, _mm_maskz_sub_ps,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubps`, 128-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubps`, 128-bit, zero-masked)."
);
avx512f_vl_f32x4_binop_masked!(
	mul_f32x4_merge_masked, mul_f32x4_zero_masked, mask_mul_ps128_intrinsic, maskz_mul_ps128_intrinsic,
	_mm_mask_mul_ps, _mm_maskz_mul_ps,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulps`, 128-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulps`, 128-bit, zero-masked)."
);
avx512f_vl_f32x4_binop_masked!(
	div_f32x4_merge_masked, div_f32x4_zero_masked, mask_div_ps128_intrinsic, maskz_div_ps128_intrinsic,
	_mm_mask_div_ps, _mm_maskz_div_ps,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivps`, 128-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivps`, 128-bit, zero-masked)."
);
avx512f_vl_f32x4_binop_masked!(
	min_f32x4_merge_masked, min_f32x4_zero_masked, mask_min_ps128_intrinsic, maskz_min_ps128_intrinsic,
	_mm_mask_min_ps, _mm_maskz_min_ps,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminps`, 128-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminps`, 128-bit, zero-masked)."
);
avx512f_vl_f32x4_binop_masked!(
	max_f32x4_merge_masked, max_f32x4_zero_masked, mask_max_ps128_intrinsic, maskz_max_ps128_intrinsic,
	_mm_mask_max_ps, _mm_maskz_max_ps,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxps`, 128-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxps`, 128-bit, zero-masked)."
);

avx512f_vl_f32x8_binop_masked!(
	add_f32x8_merge_masked, add_f32x8_zero_masked, mask_add_ps256_intrinsic, maskz_add_ps256_intrinsic,
	_mm256_mask_add_ps, _mm256_maskz_add_ps,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddps`, 256-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddps`, 256-bit, zero-masked)."
);
avx512f_vl_f32x8_binop_masked!(
	sub_f32x8_merge_masked, sub_f32x8_zero_masked, mask_sub_ps256_intrinsic, maskz_sub_ps256_intrinsic,
	_mm256_mask_sub_ps, _mm256_maskz_sub_ps,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubps`, 256-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubps`, 256-bit, zero-masked)."
);
avx512f_vl_f32x8_binop_masked!(
	mul_f32x8_merge_masked, mul_f32x8_zero_masked, mask_mul_ps256_intrinsic, maskz_mul_ps256_intrinsic,
	_mm256_mask_mul_ps, _mm256_maskz_mul_ps,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulps`, 256-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulps`, 256-bit, zero-masked)."
);
avx512f_vl_f32x8_binop_masked!(
	div_f32x8_merge_masked, div_f32x8_zero_masked, mask_div_ps256_intrinsic, maskz_div_ps256_intrinsic,
	_mm256_mask_div_ps, _mm256_maskz_div_ps,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivps`, 256-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivps`, 256-bit, zero-masked)."
);
avx512f_vl_f32x8_binop_masked!(
	min_f32x8_merge_masked, min_f32x8_zero_masked, mask_min_ps256_intrinsic, maskz_min_ps256_intrinsic,
	_mm256_mask_min_ps, _mm256_maskz_min_ps,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminps`, 256-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminps`, 256-bit, zero-masked)."
);
avx512f_vl_f32x8_binop_masked!(
	max_f32x8_merge_masked, max_f32x8_zero_masked, mask_max_ps256_intrinsic, maskz_max_ps256_intrinsic,
	_mm256_mask_max_ps, _mm256_maskz_max_ps,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxps`, 256-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxps`, 256-bit, zero-masked)."
);

avx512f_vl_f64x2_binop_masked!(
	add_f64x2_merge_masked, add_f64x2_zero_masked, mask_add_pd128_intrinsic, maskz_add_pd128_intrinsic,
	_mm_mask_add_pd, _mm_maskz_add_pd,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddpd`, 128-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddpd`, 128-bit, zero-masked)."
);
avx512f_vl_f64x2_binop_masked!(
	sub_f64x2_merge_masked, sub_f64x2_zero_masked, mask_sub_pd128_intrinsic, maskz_sub_pd128_intrinsic,
	_mm_mask_sub_pd, _mm_maskz_sub_pd,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubpd`, 128-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubpd`, 128-bit, zero-masked)."
);
avx512f_vl_f64x2_binop_masked!(
	mul_f64x2_merge_masked, mul_f64x2_zero_masked, mask_mul_pd128_intrinsic, maskz_mul_pd128_intrinsic,
	_mm_mask_mul_pd, _mm_maskz_mul_pd,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulpd`, 128-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulpd`, 128-bit, zero-masked)."
);
avx512f_vl_f64x2_binop_masked!(
	div_f64x2_merge_masked, div_f64x2_zero_masked, mask_div_pd128_intrinsic, maskz_div_pd128_intrinsic,
	_mm_mask_div_pd, _mm_maskz_div_pd,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivpd`, 128-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivpd`, 128-bit, zero-masked)."
);
avx512f_vl_f64x2_binop_masked!(
	min_f64x2_merge_masked, min_f64x2_zero_masked, mask_min_pd128_intrinsic, maskz_min_pd128_intrinsic,
	_mm_mask_min_pd, _mm_maskz_min_pd,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminpd`, 128-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminpd`, 128-bit, zero-masked)."
);
avx512f_vl_f64x2_binop_masked!(
	max_f64x2_merge_masked, max_f64x2_zero_masked, mask_max_pd128_intrinsic, maskz_max_pd128_intrinsic,
	_mm_mask_max_pd, _mm_maskz_max_pd,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxpd`, 128-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxpd`, 128-bit, zero-masked)."
);

avx512f_vl_f64x4_binop_masked!(
	add_f64x4_merge_masked, add_f64x4_zero_masked, mask_add_pd256_intrinsic, maskz_add_pd256_intrinsic,
	_mm256_mask_add_pd, _mm256_maskz_add_pd,
	"`a + b` per lane where `mask` bit is set, else copied from `src` (`vaddpd`, 256-bit, merge-masked).",
	"`a + b` per lane where `mask` bit is set, else zero (`vaddpd`, 256-bit, zero-masked)."
);
avx512f_vl_f64x4_binop_masked!(
	sub_f64x4_merge_masked, sub_f64x4_zero_masked, mask_sub_pd256_intrinsic, maskz_sub_pd256_intrinsic,
	_mm256_mask_sub_pd, _mm256_maskz_sub_pd,
	"`a - b` per lane where `mask` bit is set, else copied from `src` (`vsubpd`, 256-bit, merge-masked).",
	"`a - b` per lane where `mask` bit is set, else zero (`vsubpd`, 256-bit, zero-masked)."
);
avx512f_vl_f64x4_binop_masked!(
	mul_f64x4_merge_masked, mul_f64x4_zero_masked, mask_mul_pd256_intrinsic, maskz_mul_pd256_intrinsic,
	_mm256_mask_mul_pd, _mm256_maskz_mul_pd,
	"`a * b` per lane where `mask` bit is set, else copied from `src` (`vmulpd`, 256-bit, merge-masked).",
	"`a * b` per lane where `mask` bit is set, else zero (`vmulpd`, 256-bit, zero-masked)."
);
avx512f_vl_f64x4_binop_masked!(
	div_f64x4_merge_masked, div_f64x4_zero_masked, mask_div_pd256_intrinsic, maskz_div_pd256_intrinsic,
	_mm256_mask_div_pd, _mm256_maskz_div_pd,
	"`a / b` per lane where `mask` bit is set, else copied from `src` (`vdivpd`, 256-bit, merge-masked).",
	"`a / b` per lane where `mask` bit is set, else zero (`vdivpd`, 256-bit, zero-masked)."
);
avx512f_vl_f64x4_binop_masked!(
	min_f64x4_merge_masked, min_f64x4_zero_masked, mask_min_pd256_intrinsic, maskz_min_pd256_intrinsic,
	_mm256_mask_min_pd, _mm256_maskz_min_pd,
	"Per-lane min where `mask` bit is set, else copied from `src` (`vminpd`, 256-bit, merge-masked).",
	"Per-lane min where `mask` bit is set, else zero (`vminpd`, 256-bit, zero-masked)."
);
avx512f_vl_f64x4_binop_masked!(
	max_f64x4_merge_masked, max_f64x4_zero_masked, mask_max_pd256_intrinsic, maskz_max_pd256_intrinsic,
	_mm256_mask_max_pd, _mm256_maskz_max_pd,
	"Per-lane max where `mask` bit is set, else copied from `src` (`vmaxpd`, 256-bit, merge-masked).",
	"Per-lane max where `mask` bit is set, else zero (`vmaxpd`, 256-bit, zero-masked)."
);

simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rcp14_f32x4, intrinsic_fn = rcp14ps128,
	width = 4, elem = f32, vec = __m128,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps, intrinsic = _mm_rcp14_ps,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14ps`, 128-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rcp14_f32x8, intrinsic_fn = rcp14ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_rcp14_ps,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14ps`, 256-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rsqrt14_f32x4, intrinsic_fn = rsqrt14ps128,
	width = 4, elem = f32, vec = __m128,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps, intrinsic = _mm_rsqrt14_ps,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14ps`, 128-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rsqrt14_f32x8, intrinsic_fn = rsqrt14ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_rsqrt14_ps,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14ps`, 256-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rcp14_f64x2, intrinsic_fn = rcp14pd128,
	width = 2, elem = f64, vec = __m128d,
	loadu = _mm_loadu_pd, storeu = _mm_storeu_pd, intrinsic = _mm_rcp14_pd,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14pd`, 128-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rcp14_f64x4, intrinsic_fn = rcp14pd256,
	width = 4, elem = f64, vec = __m256d,
	loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd, intrinsic = _mm256_rcp14_pd,
	fixed_doc = "Approximate per-lane reciprocal (`vrcp14pd`, 256-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rsqrt14_f64x2, intrinsic_fn = rsqrt14pd128,
	width = 2, elem = f64, vec = __m128d,
	loadu = _mm_loadu_pd, storeu = _mm_storeu_pd, intrinsic = _mm_rsqrt14_pd,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14pd`, 128-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = rsqrt14_f64x4, intrinsic_fn = rsqrt14pd256,
	width = 4, elem = f64, vec = __m256d,
	loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd, intrinsic = _mm256_rsqrt14_pd,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrt14pd`, 256-bit), max relative error <= 2^-14. Fixed-width only, see module doc.",
}

avx512f_vl_i32x4_binop_masked!(
	add_i32x4_merge_masked, add_i32x4_zero_masked, mask_add_epi32_128_intrinsic, maskz_add_epi32_128_intrinsic,
	_mm_mask_add_epi32, _mm_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, 128-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, 128-bit, zero-masked)."
);
avx512f_vl_i32x4_binop_masked!(
	sub_i32x4_merge_masked, sub_i32x4_zero_masked, mask_sub_epi32_128_intrinsic, maskz_sub_epi32_128_intrinsic,
	_mm_mask_sub_epi32, _mm_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, 128-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, 128-bit, zero-masked)."
);
avx512f_vl_i32x4_binop_masked!(
	mul_i32x4_merge_masked, mul_i32x4_zero_masked, mask_mullo_epi32_128_intrinsic, maskz_mullo_epi32_128_intrinsic,
	_mm_mask_mullo_epi32, _mm_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, 128-bit, merge-masked).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, 128-bit, zero-masked)."
);
avx512f_vl_i32x4_binop_masked!(
	min_i32x4_merge_masked, min_i32x4_zero_masked, mask_min_epi32_128_intrinsic, maskz_min_epi32_128_intrinsic,
	_mm_mask_min_epi32, _mm_maskz_min_epi32,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsd`, 128-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsd`, 128-bit, zero-masked)."
);
avx512f_vl_i32x4_binop_masked!(
	max_i32x4_merge_masked, max_i32x4_zero_masked, mask_max_epi32_128_intrinsic, maskz_max_epi32_128_intrinsic,
	_mm_mask_max_epi32, _mm_maskz_max_epi32,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsd`, 128-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsd`, 128-bit, zero-masked)."
);

avx512f_vl_i32x8_binop_masked!(
	add_i32x8_merge_masked, add_i32x8_zero_masked, mask_add_epi32_256_intrinsic, maskz_add_epi32_256_intrinsic,
	_mm256_mask_add_epi32, _mm256_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, 256-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, 256-bit, zero-masked)."
);
avx512f_vl_i32x8_binop_masked!(
	sub_i32x8_merge_masked, sub_i32x8_zero_masked, mask_sub_epi32_256_intrinsic, maskz_sub_epi32_256_intrinsic,
	_mm256_mask_sub_epi32, _mm256_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, 256-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, 256-bit, zero-masked)."
);
avx512f_vl_i32x8_binop_masked!(
	mul_i32x8_merge_masked, mul_i32x8_zero_masked, mask_mullo_epi32_256_intrinsic, maskz_mullo_epi32_256_intrinsic,
	_mm256_mask_mullo_epi32, _mm256_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, 256-bit, merge-masked).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, 256-bit, zero-masked)."
);
avx512f_vl_i32x8_binop_masked!(
	min_i32x8_merge_masked, min_i32x8_zero_masked, mask_min_epi32_256_intrinsic, maskz_min_epi32_256_intrinsic,
	_mm256_mask_min_epi32, _mm256_maskz_min_epi32,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsd`, 256-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsd`, 256-bit, zero-masked)."
);
avx512f_vl_i32x8_binop_masked!(
	max_i32x8_merge_masked, max_i32x8_zero_masked, mask_max_epi32_256_intrinsic, maskz_max_epi32_256_intrinsic,
	_mm256_mask_max_epi32, _mm256_maskz_max_epi32,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsd`, 256-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsd`, 256-bit, zero-masked)."
);

avx512f_vl_u32x4_binop_masked!(
	add_u32x4_merge_masked, add_u32x4_zero_masked, mask_add_epu32_128_intrinsic, maskz_add_epu32_128_intrinsic,
	_mm_mask_add_epi32, _mm_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, 128-bit, merge-masked; bit-identical to the signed form, no `epu32` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, 128-bit, zero-masked)."
);
avx512f_vl_u32x4_binop_masked!(
	sub_u32x4_merge_masked, sub_u32x4_zero_masked, mask_sub_epu32_128_intrinsic, maskz_sub_epu32_128_intrinsic,
	_mm_mask_sub_epi32, _mm_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, 128-bit, merge-masked; bit-identical to the signed form, no `epu32` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, 128-bit, zero-masked)."
);
avx512f_vl_u32x4_binop_masked!(
	mul_u32x4_merge_masked, mul_u32x4_zero_masked, mask_mullo_epu32_128_intrinsic, maskz_mullo_epu32_128_intrinsic,
	_mm_mask_mullo_epi32, _mm_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, 128-bit, merge-masked; bit-identical to the signed form).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, 128-bit, zero-masked)."
);
avx512f_vl_u32x4_binop_masked!(
	min_u32x4_merge_masked, min_u32x4_zero_masked, mask_min_epu32_128_intrinsic, maskz_min_epu32_128_intrinsic,
	_mm_mask_min_epu32, _mm_maskz_min_epu32,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminud`, 128-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminud`, 128-bit, zero-masked)."
);
avx512f_vl_u32x4_binop_masked!(
	max_u32x4_merge_masked, max_u32x4_zero_masked, mask_max_epu32_128_intrinsic, maskz_max_epu32_128_intrinsic,
	_mm_mask_max_epu32, _mm_maskz_max_epu32,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxud`, 128-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxud`, 128-bit, zero-masked)."
);

avx512f_vl_u32x8_binop_masked!(
	add_u32x8_merge_masked, add_u32x8_zero_masked, mask_add_epu32_256_intrinsic, maskz_add_epu32_256_intrinsic,
	_mm256_mask_add_epi32, _mm256_maskz_add_epi32,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddd`, 256-bit, merge-masked; bit-identical to the signed form, no `epu32` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddd`, 256-bit, zero-masked)."
);
avx512f_vl_u32x8_binop_masked!(
	sub_u32x8_merge_masked, sub_u32x8_zero_masked, mask_sub_epu32_256_intrinsic, maskz_sub_epu32_256_intrinsic,
	_mm256_mask_sub_epi32, _mm256_maskz_sub_epi32,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubd`, 256-bit, merge-masked; bit-identical to the signed form, no `epu32` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubd`, 256-bit, zero-masked)."
);
avx512f_vl_u32x8_binop_masked!(
	mul_u32x8_merge_masked, mul_u32x8_zero_masked, mask_mullo_epu32_256_intrinsic, maskz_mullo_epu32_256_intrinsic,
	_mm256_mask_mullo_epi32, _mm256_maskz_mullo_epi32,
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else copied from `src` (`vpmulld`, 256-bit, merge-masked; bit-identical to the signed form).",
	"`a * b` per lane, low 32 bits, where `mask` bit is set, else zero (`vpmulld`, 256-bit, zero-masked)."
);
avx512f_vl_u32x8_binop_masked!(
	min_u32x8_merge_masked, min_u32x8_zero_masked, mask_min_epu32_256_intrinsic, maskz_min_epu32_256_intrinsic,
	_mm256_mask_min_epu32, _mm256_maskz_min_epu32,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminud`, 256-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminud`, 256-bit, zero-masked)."
);
avx512f_vl_u32x8_binop_masked!(
	max_u32x8_merge_masked, max_u32x8_zero_masked, mask_max_epu32_256_intrinsic, maskz_max_epu32_256_intrinsic,
	_mm256_mask_max_epu32, _mm256_maskz_max_epu32,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxud`, 256-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxud`, 256-bit, zero-masked)."
);

avx512f_vl_i64x2_binop_masked!(
	add_i64x2_merge_masked, add_i64x2_zero_masked, mask_add_epi64_128_intrinsic, maskz_add_epi64_128_intrinsic,
	_mm_mask_add_epi64, _mm_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, 128-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, 128-bit, zero-masked)."
);
avx512f_vl_i64x2_binop_masked!(
	sub_i64x2_merge_masked, sub_i64x2_zero_masked, mask_sub_epi64_128_intrinsic, maskz_sub_epi64_128_intrinsic,
	_mm_mask_sub_epi64, _mm_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, 128-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, 128-bit, zero-masked)."
);
avx512f_vl_i64x2_binop_masked!(
	min_i64x2_merge_masked, min_i64x2_zero_masked, mask_min_epi64_128_intrinsic, maskz_min_epi64_128_intrinsic,
	_mm_mask_min_epi64, _mm_maskz_min_epi64,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsq`, 128-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsq`, 128-bit, zero-masked)."
);
avx512f_vl_i64x2_binop_masked!(
	max_i64x2_merge_masked, max_i64x2_zero_masked, mask_max_epi64_128_intrinsic, maskz_max_epi64_128_intrinsic,
	_mm_mask_max_epi64, _mm_maskz_max_epi64,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsq`, 128-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsq`, 128-bit, zero-masked)."
);

avx512f_vl_i64x4_binop_masked!(
	add_i64x4_merge_masked, add_i64x4_zero_masked, mask_add_epi64_256_intrinsic, maskz_add_epi64_256_intrinsic,
	_mm256_mask_add_epi64, _mm256_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, 256-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, 256-bit, zero-masked)."
);
avx512f_vl_i64x4_binop_masked!(
	sub_i64x4_merge_masked, sub_i64x4_zero_masked, mask_sub_epi64_256_intrinsic, maskz_sub_epi64_256_intrinsic,
	_mm256_mask_sub_epi64, _mm256_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, 256-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, 256-bit, zero-masked)."
);
avx512f_vl_i64x4_binop_masked!(
	min_i64x4_merge_masked, min_i64x4_zero_masked, mask_min_epi64_256_intrinsic, maskz_min_epi64_256_intrinsic,
	_mm256_mask_min_epi64, _mm256_maskz_min_epi64,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsq`, 256-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsq`, 256-bit, zero-masked)."
);
avx512f_vl_i64x4_binop_masked!(
	max_i64x4_merge_masked, max_i64x4_zero_masked, mask_max_epi64_256_intrinsic, maskz_max_epi64_256_intrinsic,
	_mm256_mask_max_epi64, _mm256_maskz_max_epi64,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsq`, 256-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsq`, 256-bit, zero-masked)."
);

avx512f_vl_u64x2_binop_masked!(
	add_u64x2_merge_masked, add_u64x2_zero_masked, mask_add_epu64_128_intrinsic, maskz_add_epu64_128_intrinsic,
	_mm_mask_add_epi64, _mm_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, 128-bit, merge-masked; bit-identical to the signed form, no `epu64` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, 128-bit, zero-masked)."
);
avx512f_vl_u64x2_binop_masked!(
	sub_u64x2_merge_masked, sub_u64x2_zero_masked, mask_sub_epu64_128_intrinsic, maskz_sub_epu64_128_intrinsic,
	_mm_mask_sub_epi64, _mm_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, 128-bit, merge-masked; bit-identical to the signed form, no `epu64` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, 128-bit, zero-masked)."
);
avx512f_vl_u64x2_binop_masked!(
	min_u64x2_merge_masked, min_u64x2_zero_masked, mask_min_epu64_128_intrinsic, maskz_min_epu64_128_intrinsic,
	_mm_mask_min_epu64, _mm_maskz_min_epu64,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuq`, 128-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuq`, 128-bit, zero-masked)."
);
avx512f_vl_u64x2_binop_masked!(
	max_u64x2_merge_masked, max_u64x2_zero_masked, mask_max_epu64_128_intrinsic, maskz_max_epu64_128_intrinsic,
	_mm_mask_max_epu64, _mm_maskz_max_epu64,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuq`, 128-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuq`, 128-bit, zero-masked)."
);

avx512f_vl_u64x4_binop_masked!(
	add_u64x4_merge_masked, add_u64x4_zero_masked, mask_add_epu64_256_intrinsic, maskz_add_epu64_256_intrinsic,
	_mm256_mask_add_epi64, _mm256_maskz_add_epi64,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddq`, 256-bit, merge-masked; bit-identical to the signed form, no `epu64` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddq`, 256-bit, zero-masked)."
);
avx512f_vl_u64x4_binop_masked!(
	sub_u64x4_merge_masked, sub_u64x4_zero_masked, mask_sub_epu64_256_intrinsic, maskz_sub_epu64_256_intrinsic,
	_mm256_mask_sub_epi64, _mm256_maskz_sub_epi64,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubq`, 256-bit, merge-masked; bit-identical to the signed form, no `epu64` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubq`, 256-bit, zero-masked)."
);
avx512f_vl_u64x4_binop_masked!(
	min_u64x4_merge_masked, min_u64x4_zero_masked, mask_min_epu64_256_intrinsic, maskz_min_epu64_256_intrinsic,
	_mm256_mask_min_epu64, _mm256_maskz_min_epu64,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuq`, 256-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuq`, 256-bit, zero-masked)."
);
avx512f_vl_u64x4_binop_masked!(
	max_u64x4_merge_masked, max_u64x4_zero_masked, mask_max_epu64_256_intrinsic, maskz_max_epu64_256_intrinsic,
	_mm256_mask_max_epu64, _mm256_maskz_max_epu64,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuq`, 256-bit, merge-masked; no pre-AVX-512 form).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuq`, 256-bit, zero-masked)."
);

// Bitwise ternary logic (`vpternlogd`/`vpternlogq`) at 128/256-bit: same
// "genuinely new capability" reasoning as `avx512f.rs`'s 512-bit block:
// no pre-AVX-512 equivalent exists at any width, unmasked or masked.
// `epu32`/`epu64` reuse the signed `epi32`/`epi64` intrinsic bit-identically.

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_i32x4, merge_fn = ternarylogic_i32x4_merge_masked, zero_fn = ternarylogic_i32x4_zero_masked,
	intrinsic_fn = ternarylogic_epi32_128_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi32_128_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi32_128_intrinsic,
	width = 4, elem = i32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_ternarylogic_epi32, merge_intrinsic = _mm_mask_ternarylogic_epi32, zero_intrinsic = _mm_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 128-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, 128-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, 128-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_u32x4, merge_fn = ternarylogic_u32x4_merge_masked, zero_fn = ternarylogic_u32x4_zero_masked,
	intrinsic_fn = ternarylogic_epu32_128_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu32_128_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu32_128_intrinsic,
	width = 4, elem = u32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_ternarylogic_epi32, merge_intrinsic = _mm_mask_ternarylogic_epi32, zero_intrinsic = _mm_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 128-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, 128-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, 128-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_i32x8, merge_fn = ternarylogic_i32x8_merge_masked, zero_fn = ternarylogic_i32x8_zero_masked,
	intrinsic_fn = ternarylogic_epi32_256_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi32_256_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi32_256_intrinsic,
	width = 8, elem = i32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_ternarylogic_epi32, merge_intrinsic = _mm256_mask_ternarylogic_epi32, zero_intrinsic = _mm256_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 256-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, 256-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, 256-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_u32x8, merge_fn = ternarylogic_u32x8_merge_masked, zero_fn = ternarylogic_u32x8_zero_masked,
	intrinsic_fn = ternarylogic_epu32_256_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu32_256_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu32_256_intrinsic,
	width = 8, elem = u32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_ternarylogic_epi32, merge_intrinsic = _mm256_mask_ternarylogic_epi32, zero_intrinsic = _mm256_maskz_ternarylogic_epi32,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogd`, 256-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogd`, 256-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogd`, 256-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_i64x2, merge_fn = ternarylogic_i64x2_merge_masked, zero_fn = ternarylogic_i64x2_zero_masked,
	intrinsic_fn = ternarylogic_epi64_128_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi64_128_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi64_128_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_ternarylogic_epi64, merge_intrinsic = _mm_mask_ternarylogic_epi64, zero_intrinsic = _mm_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 128-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, 128-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, 128-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_u64x2, merge_fn = ternarylogic_u64x2_merge_masked, zero_fn = ternarylogic_u64x2_zero_masked,
	intrinsic_fn = ternarylogic_epu64_128_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu64_128_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu64_128_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_ternarylogic_epi64, merge_intrinsic = _mm_mask_ternarylogic_epi64, zero_intrinsic = _mm_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 128-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, 128-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, 128-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_i64x4, merge_fn = ternarylogic_i64x4_merge_masked, zero_fn = ternarylogic_i64x4_zero_masked,
	intrinsic_fn = ternarylogic_epi64_256_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epi64_256_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epi64_256_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_ternarylogic_epi64, merge_intrinsic = _mm256_mask_ternarylogic_epi64, zero_intrinsic = _mm256_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 256-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, 256-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, 256-bit, zero-masked).",
}

simd_ternarylogic! {
	token = Avx512FVl, target_feature = "avx512f,avx512vl",
	fixed_fn = ternarylogic_u64x4, merge_fn = ternarylogic_u64x4_merge_masked, zero_fn = ternarylogic_u64x4_zero_masked,
	intrinsic_fn = ternarylogic_epu64_256_intrinsic, merge_intrinsic_fn = mask_ternarylogic_epu64_256_intrinsic,
	zero_intrinsic_fn = maskz_ternarylogic_epu64_256_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_ternarylogic_epi64, merge_intrinsic = _mm256_mask_ternarylogic_epi64, zero_intrinsic = _mm256_maskz_ternarylogic_epi64,
	fixed_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)` (`vpternlogq`, 256-bit).",
	merge_doc = "Per-bit 3-input truth table `IMM8` over `(src, a, b)`, else copied from `src` where `mask` bit is unset (`vpternlogq`, 256-bit, merge-masked).",
	zero_doc = "Per-bit 3-input truth table `IMM8` over `(a, b, c)`, else zero where `mask` bit is unset (`vpternlogq`, 256-bit, zero-masked).",
}

// AVX512BW-family masked-arithmetic intrinsics for `Avx512BwVl`, kept in a
// dedicated `use` block (rather than merged into the file-top one above)
// since this is a self-contained ~50-instantiation family added in one
// batch: easier to audit against the token it serves.
use core::arch::x86_64::{
	_mm_mask_add_epi16, _mm_mask_add_epi8, _mm_mask_adds_epi16, _mm_mask_adds_epi8, _mm_mask_adds_epu16,
	_mm_mask_adds_epu8, _mm_mask_avg_epu16, _mm_mask_avg_epu8, _mm_mask_max_epi16, _mm_mask_max_epi8,
	_mm_mask_max_epu16, _mm_mask_max_epu8,
	_mm_mask_min_epi16, _mm_mask_min_epi8, _mm_mask_min_epu16, _mm_mask_min_epu8, _mm_mask_mullo_epi16,
	_mm_mask_sub_epi16, _mm_mask_sub_epi8, _mm_mask_subs_epi16, _mm_mask_subs_epi8, _mm_mask_subs_epu16,
	_mm_mask_subs_epu8, _mm_maskz_add_epi16, _mm_maskz_add_epi8, _mm_maskz_adds_epi16, _mm_maskz_adds_epi8,
	_mm_maskz_adds_epu16, _mm_maskz_adds_epu8, _mm_maskz_avg_epu16, _mm_maskz_avg_epu8, _mm_maskz_max_epi16,
	_mm_maskz_max_epi8, _mm_maskz_max_epu16,
	_mm_maskz_max_epu8, _mm_maskz_min_epi16, _mm_maskz_min_epi8, _mm_maskz_min_epu16, _mm_maskz_min_epu8,
	_mm_maskz_mullo_epi16, _mm_maskz_sub_epi16, _mm_maskz_sub_epi8, _mm_maskz_subs_epi16, _mm_maskz_subs_epi8,
	_mm_maskz_subs_epu16, _mm_maskz_subs_epu8, _mm256_mask_add_epi16, _mm256_mask_add_epi8, _mm256_mask_adds_epi16,
	_mm256_mask_adds_epi8, _mm256_mask_adds_epu16, _mm256_mask_adds_epu8, _mm256_mask_avg_epu16,
	_mm256_mask_avg_epu8, _mm256_mask_max_epi16,
	_mm256_mask_max_epi8, _mm256_mask_max_epu16, _mm256_mask_max_epu8, _mm256_mask_min_epi16, _mm256_mask_min_epi8,
	_mm256_mask_min_epu16, _mm256_mask_min_epu8, _mm256_mask_mullo_epi16, _mm256_mask_sub_epi16,
	_mm256_mask_sub_epi8, _mm256_mask_subs_epi16, _mm256_mask_subs_epi8, _mm256_mask_subs_epu16,
	_mm256_mask_subs_epu8, _mm256_maskz_add_epi16, _mm256_maskz_add_epi8, _mm256_maskz_adds_epi16,
	_mm256_maskz_adds_epi8, _mm256_maskz_adds_epu16, _mm256_maskz_adds_epu8, _mm256_maskz_avg_epu16,
	_mm256_maskz_avg_epu8, _mm256_maskz_max_epi16,
	_mm256_maskz_max_epi8, _mm256_maskz_max_epu16, _mm256_maskz_max_epu8, _mm256_maskz_min_epi16,
	_mm256_maskz_min_epi8, _mm256_maskz_min_epu16, _mm256_maskz_min_epu8, _mm256_maskz_mullo_epi16,
	_mm256_maskz_sub_epi16, _mm256_maskz_sub_epi8, _mm256_maskz_subs_epi16, _mm256_maskz_subs_epi8,
	_mm256_maskz_subs_epu16, _mm256_maskz_subs_epu8,
};

/// Proof token: AVX512BW *and* AVX512VL, for the 128/256-bit forms. No
/// unmasked-op sibling in this file: unlike every other `*Vl` token here,
/// plain 128/256-bit `i8`/`u8`/`i16`/`u16` add/sub/etc. already live on
/// `Sse2`/`Avx2` (no AVX-512 needed for the unmasked op): only the k-mask
/// merge/zero forms are AVX-512-exclusive, same reasoning as [`Avx512FVl`].
/// `epu8`/`epu16` add/sub/mullo reuse the signed `epi8`/`epi16` intrinsic
/// bit-identically (confirmed via stdarch: no separate unsigned add/sub/
/// mullo intrinsic exists at any width here), matching `avx512bw.rs`'s
/// 512-bit masked family.
#[derive(Debug, Clone, Copy)]
pub struct Avx512BwVl(());

impl Avx512BwVl {
	/// `None` unless the CPU has both AVX-512BW and AVX-512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512bw) && set.contains(Feature::Avx512vl)).then_some(Avx512BwVl(()))
	}
}

macro_rules! avx512bw_vl_i8x16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = i8, vec = __m128i, mask = u16,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_i8x32_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = i8, vec = __m256i, mask = u32,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_u8x16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u8, vec = __m128i, mask = u16,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_u8x32_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = u8, vec = __m256i, mask = u32,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_i16x8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = i16, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_i16x16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = i16, vec = __m256i, mask = u16,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_u16x8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 8, elem = u16, vec = __m128i, mask = u8,
			loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_vl_u16x16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512BwVl, target_feature = "avx512bw,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 16, elem = u16, vec = __m256i, mask = u16,
			loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512bw_vl_i8x16_binop_masked!(
	add_i8x16_merge_masked, add_i8x16_zero_masked, mask_add_epi8_128_intrinsic, maskz_add_epi8_128_intrinsic,
	_mm_mask_add_epi8, _mm_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, 128-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, 128-bit, zero-masked)."
);
avx512bw_vl_i8x16_binop_masked!(
	sub_i8x16_merge_masked, sub_i8x16_zero_masked, mask_sub_epi8_128_intrinsic, maskz_sub_epi8_128_intrinsic,
	_mm_mask_sub_epi8, _mm_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, 128-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, 128-bit, zero-masked)."
);
avx512bw_vl_i8x16_binop_masked!(
	adds_i8x16_merge_masked, adds_i8x16_zero_masked, mask_adds_epi8_128_intrinsic, maskz_adds_epi8_128_intrinsic,
	_mm_mask_adds_epi8, _mm_maskz_adds_epi8,
	"`a + b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpaddsb`, 128-bit, merge-masked).",
	"`a + b` per lane, signed saturating, where `mask` bit is set, else zero (`vpaddsb`, 128-bit, zero-masked)."
);
avx512bw_vl_i8x16_binop_masked!(
	subs_i8x16_merge_masked, subs_i8x16_zero_masked, mask_subs_epi8_128_intrinsic, maskz_subs_epi8_128_intrinsic,
	_mm_mask_subs_epi8, _mm_maskz_subs_epi8,
	"`a - b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpsubsb`, 128-bit, merge-masked).",
	"`a - b` per lane, signed saturating, where `mask` bit is set, else zero (`vpsubsb`, 128-bit, zero-masked)."
);
avx512bw_vl_i8x16_binop_masked!(
	min_i8x16_merge_masked, min_i8x16_zero_masked, mask_min_epi8_128_intrinsic, maskz_min_epi8_128_intrinsic,
	_mm_mask_min_epi8, _mm_maskz_min_epi8,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsb`, 128-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsb`, 128-bit, zero-masked)."
);
avx512bw_vl_i8x16_binop_masked!(
	max_i8x16_merge_masked, max_i8x16_zero_masked, mask_max_epi8_128_intrinsic, maskz_max_epi8_128_intrinsic,
	_mm_mask_max_epi8, _mm_maskz_max_epi8,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsb`, 128-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsb`, 128-bit, zero-masked)."
);

avx512bw_vl_i8x32_binop_masked!(
	add_i8x32_merge_masked, add_i8x32_zero_masked, mask_add_epi8_256_intrinsic, maskz_add_epi8_256_intrinsic,
	_mm256_mask_add_epi8, _mm256_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, 256-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, 256-bit, zero-masked)."
);
avx512bw_vl_i8x32_binop_masked!(
	sub_i8x32_merge_masked, sub_i8x32_zero_masked, mask_sub_epi8_256_intrinsic, maskz_sub_epi8_256_intrinsic,
	_mm256_mask_sub_epi8, _mm256_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, 256-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, 256-bit, zero-masked)."
);
avx512bw_vl_i8x32_binop_masked!(
	adds_i8x32_merge_masked, adds_i8x32_zero_masked, mask_adds_epi8_256_intrinsic, maskz_adds_epi8_256_intrinsic,
	_mm256_mask_adds_epi8, _mm256_maskz_adds_epi8,
	"`a + b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpaddsb`, 256-bit, merge-masked).",
	"`a + b` per lane, signed saturating, where `mask` bit is set, else zero (`vpaddsb`, 256-bit, zero-masked)."
);
avx512bw_vl_i8x32_binop_masked!(
	subs_i8x32_merge_masked, subs_i8x32_zero_masked, mask_subs_epi8_256_intrinsic, maskz_subs_epi8_256_intrinsic,
	_mm256_mask_subs_epi8, _mm256_maskz_subs_epi8,
	"`a - b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpsubsb`, 256-bit, merge-masked).",
	"`a - b` per lane, signed saturating, where `mask` bit is set, else zero (`vpsubsb`, 256-bit, zero-masked)."
);
avx512bw_vl_i8x32_binop_masked!(
	min_i8x32_merge_masked, min_i8x32_zero_masked, mask_min_epi8_256_intrinsic, maskz_min_epi8_256_intrinsic,
	_mm256_mask_min_epi8, _mm256_maskz_min_epi8,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsb`, 256-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsb`, 256-bit, zero-masked)."
);
avx512bw_vl_i8x32_binop_masked!(
	max_i8x32_merge_masked, max_i8x32_zero_masked, mask_max_epi8_256_intrinsic, maskz_max_epi8_256_intrinsic,
	_mm256_mask_max_epi8, _mm256_maskz_max_epi8,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsb`, 256-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsb`, 256-bit, zero-masked)."
);

avx512bw_vl_u8x16_binop_masked!(
	add_u8x16_merge_masked, add_u8x16_zero_masked, mask_add_epu8_128_intrinsic, maskz_add_epu8_128_intrinsic,
	_mm_mask_add_epi8, _mm_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, 128-bit, merge-masked; bit-identical to the signed form, no `epu8` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	sub_u8x16_merge_masked, sub_u8x16_zero_masked, mask_sub_epu8_128_intrinsic, maskz_sub_epu8_128_intrinsic,
	_mm_mask_sub_epi8, _mm_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, 128-bit, merge-masked; bit-identical to the signed form, no `epu8` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	adds_u8x16_merge_masked, adds_u8x16_zero_masked, mask_adds_epu8_128_intrinsic, maskz_adds_epu8_128_intrinsic,
	_mm_mask_adds_epu8, _mm_maskz_adds_epu8,
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpaddusb`, 128-bit, merge-masked).",
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpaddusb`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	subs_u8x16_merge_masked, subs_u8x16_zero_masked, mask_subs_epu8_128_intrinsic, maskz_subs_epu8_128_intrinsic,
	_mm_mask_subs_epu8, _mm_maskz_subs_epu8,
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpsubusb`, 128-bit, merge-masked).",
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpsubusb`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	min_u8x16_merge_masked, min_u8x16_zero_masked, mask_min_epu8_128_intrinsic, maskz_min_epu8_128_intrinsic,
	_mm_mask_min_epu8, _mm_maskz_min_epu8,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminub`, 128-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminub`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	max_u8x16_merge_masked, max_u8x16_zero_masked, mask_max_epu8_128_intrinsic, maskz_max_epu8_128_intrinsic,
	_mm_mask_max_epu8, _mm_maskz_max_epu8,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxub`, 128-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxub`, 128-bit, zero-masked)."
);
avx512bw_vl_u8x16_binop_masked!(
	avg_u8x16_merge_masked, avg_u8x16_zero_masked, mask_avg_epu8_128_intrinsic, maskz_avg_epu8_128_intrinsic,
	_mm_mask_avg_epu8, _mm_maskz_avg_epu8,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgb`, 128-bit, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgb`, 128-bit, zero-masked)."
);

avx512bw_vl_u8x32_binop_masked!(
	add_u8x32_merge_masked, add_u8x32_zero_masked, mask_add_epu8_256_intrinsic, maskz_add_epu8_256_intrinsic,
	_mm256_mask_add_epi8, _mm256_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, 256-bit, merge-masked; bit-identical to the signed form, no `epu8` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	sub_u8x32_merge_masked, sub_u8x32_zero_masked, mask_sub_epu8_256_intrinsic, maskz_sub_epu8_256_intrinsic,
	_mm256_mask_sub_epi8, _mm256_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, 256-bit, merge-masked; bit-identical to the signed form, no `epu8` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	adds_u8x32_merge_masked, adds_u8x32_zero_masked, mask_adds_epu8_256_intrinsic, maskz_adds_epu8_256_intrinsic,
	_mm256_mask_adds_epu8, _mm256_maskz_adds_epu8,
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpaddusb`, 256-bit, merge-masked).",
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpaddusb`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	subs_u8x32_merge_masked, subs_u8x32_zero_masked, mask_subs_epu8_256_intrinsic, maskz_subs_epu8_256_intrinsic,
	_mm256_mask_subs_epu8, _mm256_maskz_subs_epu8,
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpsubusb`, 256-bit, merge-masked).",
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpsubusb`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	min_u8x32_merge_masked, min_u8x32_zero_masked, mask_min_epu8_256_intrinsic, maskz_min_epu8_256_intrinsic,
	_mm256_mask_min_epu8, _mm256_maskz_min_epu8,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminub`, 256-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminub`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	max_u8x32_merge_masked, max_u8x32_zero_masked, mask_max_epu8_256_intrinsic, maskz_max_epu8_256_intrinsic,
	_mm256_mask_max_epu8, _mm256_maskz_max_epu8,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxub`, 256-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxub`, 256-bit, zero-masked)."
);
avx512bw_vl_u8x32_binop_masked!(
	avg_u8x32_merge_masked, avg_u8x32_zero_masked, mask_avg_epu8_256_intrinsic, maskz_avg_epu8_256_intrinsic,
	_mm256_mask_avg_epu8, _mm256_maskz_avg_epu8,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgb`, 256-bit, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgb`, 256-bit, zero-masked)."
);

avx512bw_vl_i16x8_binop_masked!(
	add_i16x8_merge_masked, add_i16x8_zero_masked, mask_add_epi16_128_intrinsic, maskz_add_epi16_128_intrinsic,
	_mm_mask_add_epi16, _mm_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, 128-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	sub_i16x8_merge_masked, sub_i16x8_zero_masked, mask_sub_epi16_128_intrinsic, maskz_sub_epi16_128_intrinsic,
	_mm_mask_sub_epi16, _mm_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, 128-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	adds_i16x8_merge_masked, adds_i16x8_zero_masked, mask_adds_epi16_128_intrinsic, maskz_adds_epi16_128_intrinsic,
	_mm_mask_adds_epi16, _mm_maskz_adds_epi16,
	"`a + b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpaddsw`, 128-bit, merge-masked).",
	"`a + b` per lane, signed saturating, where `mask` bit is set, else zero (`vpaddsw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	subs_i16x8_merge_masked, subs_i16x8_zero_masked, mask_subs_epi16_128_intrinsic, maskz_subs_epi16_128_intrinsic,
	_mm_mask_subs_epi16, _mm_maskz_subs_epi16,
	"`a - b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpsubsw`, 128-bit, merge-masked).",
	"`a - b` per lane, signed saturating, where `mask` bit is set, else zero (`vpsubsw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	mul_i16x8_merge_masked, mul_i16x8_zero_masked, mask_mullo_epi16_128_intrinsic, maskz_mullo_epi16_128_intrinsic,
	_mm_mask_mullo_epi16, _mm_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, 128-bit, merge-masked).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	min_i16x8_merge_masked, min_i16x8_zero_masked, mask_min_epi16_128_intrinsic, maskz_min_epi16_128_intrinsic,
	_mm_mask_min_epi16, _mm_maskz_min_epi16,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsw`, 128-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsw`, 128-bit, zero-masked)."
);
avx512bw_vl_i16x8_binop_masked!(
	max_i16x8_merge_masked, max_i16x8_zero_masked, mask_max_epi16_128_intrinsic, maskz_max_epi16_128_intrinsic,
	_mm_mask_max_epi16, _mm_maskz_max_epi16,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsw`, 128-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsw`, 128-bit, zero-masked)."
);

avx512bw_vl_i16x16_binop_masked!(
	add_i16x16_merge_masked, add_i16x16_zero_masked, mask_add_epi16_256_intrinsic, maskz_add_epi16_256_intrinsic,
	_mm256_mask_add_epi16, _mm256_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, 256-bit, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	sub_i16x16_merge_masked, sub_i16x16_zero_masked, mask_sub_epi16_256_intrinsic, maskz_sub_epi16_256_intrinsic,
	_mm256_mask_sub_epi16, _mm256_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, 256-bit, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	adds_i16x16_merge_masked, adds_i16x16_zero_masked, mask_adds_epi16_256_intrinsic, maskz_adds_epi16_256_intrinsic,
	_mm256_mask_adds_epi16, _mm256_maskz_adds_epi16,
	"`a + b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpaddsw`, 256-bit, merge-masked).",
	"`a + b` per lane, signed saturating, where `mask` bit is set, else zero (`vpaddsw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	subs_i16x16_merge_masked, subs_i16x16_zero_masked, mask_subs_epi16_256_intrinsic, maskz_subs_epi16_256_intrinsic,
	_mm256_mask_subs_epi16, _mm256_maskz_subs_epi16,
	"`a - b` per lane, signed saturating, where `mask` bit is set, else copied from `src` (`vpsubsw`, 256-bit, merge-masked).",
	"`a - b` per lane, signed saturating, where `mask` bit is set, else zero (`vpsubsw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	mul_i16x16_merge_masked, mul_i16x16_zero_masked, mask_mullo_epi16_256_intrinsic, maskz_mullo_epi16_256_intrinsic,
	_mm256_mask_mullo_epi16, _mm256_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, 256-bit, merge-masked).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	min_i16x16_merge_masked, min_i16x16_zero_masked, mask_min_epi16_256_intrinsic, maskz_min_epi16_256_intrinsic,
	_mm256_mask_min_epi16, _mm256_maskz_min_epi16,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsw`, 256-bit, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsw`, 256-bit, zero-masked)."
);
avx512bw_vl_i16x16_binop_masked!(
	max_i16x16_merge_masked, max_i16x16_zero_masked, mask_max_epi16_256_intrinsic, maskz_max_epi16_256_intrinsic,
	_mm256_mask_max_epi16, _mm256_maskz_max_epi16,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsw`, 256-bit, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsw`, 256-bit, zero-masked)."
);

avx512bw_vl_u16x8_binop_masked!(
	add_u16x8_merge_masked, add_u16x8_zero_masked, mask_add_epu16_128_intrinsic, maskz_add_epu16_128_intrinsic,
	_mm_mask_add_epi16, _mm_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, 128-bit, merge-masked; bit-identical to the signed form, no `epu16` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	sub_u16x8_merge_masked, sub_u16x8_zero_masked, mask_sub_epu16_128_intrinsic, maskz_sub_epu16_128_intrinsic,
	_mm_mask_sub_epi16, _mm_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, 128-bit, merge-masked; bit-identical to the signed form, no `epu16` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	adds_u16x8_merge_masked, adds_u16x8_zero_masked, mask_adds_epu16_128_intrinsic, maskz_adds_epu16_128_intrinsic,
	_mm_mask_adds_epu16, _mm_maskz_adds_epu16,
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpaddusw`, 128-bit, merge-masked).",
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpaddusw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	subs_u16x8_merge_masked, subs_u16x8_zero_masked, mask_subs_epu16_128_intrinsic, maskz_subs_epu16_128_intrinsic,
	_mm_mask_subs_epu16, _mm_maskz_subs_epu16,
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpsubusw`, 128-bit, merge-masked).",
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpsubusw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	mul_u16x8_merge_masked, mul_u16x8_zero_masked, mask_mullo_epu16_128_intrinsic, maskz_mullo_epu16_128_intrinsic,
	_mm_mask_mullo_epi16, _mm_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, 128-bit, merge-masked; bit-identical to the signed form).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	min_u16x8_merge_masked, min_u16x8_zero_masked, mask_min_epu16_128_intrinsic, maskz_min_epu16_128_intrinsic,
	_mm_mask_min_epu16, _mm_maskz_min_epu16,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuw`, 128-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	max_u16x8_merge_masked, max_u16x8_zero_masked, mask_max_epu16_128_intrinsic, maskz_max_epu16_128_intrinsic,
	_mm_mask_max_epu16, _mm_maskz_max_epu16,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuw`, 128-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuw`, 128-bit, zero-masked)."
);
avx512bw_vl_u16x8_binop_masked!(
	avg_u16x8_merge_masked, avg_u16x8_zero_masked, mask_avg_epu16_128_intrinsic, maskz_avg_epu16_128_intrinsic,
	_mm_mask_avg_epu16, _mm_maskz_avg_epu16,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgw`, 128-bit, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgw`, 128-bit, zero-masked)."
);

avx512bw_vl_u16x16_binop_masked!(
	add_u16x16_merge_masked, add_u16x16_zero_masked, mask_add_epu16_256_intrinsic, maskz_add_epu16_256_intrinsic,
	_mm256_mask_add_epi16, _mm256_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, 256-bit, merge-masked; bit-identical to the signed form, no `epu16` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	sub_u16x16_merge_masked, sub_u16x16_zero_masked, mask_sub_epu16_256_intrinsic, maskz_sub_epu16_256_intrinsic,
	_mm256_mask_sub_epi16, _mm256_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, 256-bit, merge-masked; bit-identical to the signed form, no `epu16` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	adds_u16x16_merge_masked, adds_u16x16_zero_masked, mask_adds_epu16_256_intrinsic, maskz_adds_epu16_256_intrinsic,
	_mm256_mask_adds_epu16, _mm256_maskz_adds_epu16,
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpaddusw`, 256-bit, merge-masked).",
	"`a + b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpaddusw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	subs_u16x16_merge_masked, subs_u16x16_zero_masked, mask_subs_epu16_256_intrinsic, maskz_subs_epu16_256_intrinsic,
	_mm256_mask_subs_epu16, _mm256_maskz_subs_epu16,
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else copied from `src` (`vpsubusw`, 256-bit, merge-masked).",
	"`a - b` per lane, unsigned saturating, where `mask` bit is set, else zero (`vpsubusw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	mul_u16x16_merge_masked, mul_u16x16_zero_masked, mask_mullo_epu16_256_intrinsic, maskz_mullo_epu16_256_intrinsic,
	_mm256_mask_mullo_epi16, _mm256_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, 256-bit, merge-masked; bit-identical to the signed form).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	min_u16x16_merge_masked, min_u16x16_zero_masked, mask_min_epu16_256_intrinsic, maskz_min_epu16_256_intrinsic,
	_mm256_mask_min_epu16, _mm256_maskz_min_epu16,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuw`, 256-bit, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	max_u16x16_merge_masked, max_u16x16_zero_masked, mask_max_epu16_256_intrinsic, maskz_max_epu16_256_intrinsic,
	_mm256_mask_max_epu16, _mm256_maskz_max_epu16,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuw`, 256-bit, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuw`, 256-bit, zero-masked)."
);
avx512bw_vl_u16x16_binop_masked!(
	avg_u16x16_merge_masked, avg_u16x16_zero_masked, mask_avg_epu16_256_intrinsic, maskz_avg_epu16_256_intrinsic,
	_mm256_mask_avg_epu16, _mm256_maskz_avg_epu16,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgw`, 256-bit, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgw`, 256-bit, zero-masked)."
);

/// Proof token: AVX512IFMA *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512IfmaVl(());

impl Avx512IfmaVl {
	/// `None` unless the CPU has both AVX512IFMA and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512ifma) && set.contains(Feature::Avx512vl)).then_some(Avx512IfmaVl(()))
	}
}

// Intrinsic order is (src, a, b) = Intel (a, b, c): acc first, then factors.
// That matches `simd_ternop`'s (a, b, c) -> `$intrinsic(va, vb, vc)`.

simd_ternop! {
	token = Avx512IfmaVl, vis = pub, target_feature = "avx512ifma,avx512vl",
	fixed_fn = madd52lo_u64x2, slice_fn = madd52lo_u64_slice, intrinsic_fn = madd52lo_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_madd52lo_epu64, scalar = madd52lo_scalar,
	fixed_doc = "`src + low52(a * b)` per lane (`vpmadd52luq`, 128-bit).",
	slice_doc = "`out[i] = src[i] + low52(a[i] * b[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512IfmaVl, vis = pub, target_feature = "avx512ifma,avx512vl",
	fixed_fn = madd52lo_u64x4, slice_fn = madd52lo_u64_slice_wide, intrinsic_fn = madd52lo_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_madd52lo_epu64, scalar = madd52lo_scalar,
	fixed_doc = "`src + low52(a * b)` per lane (`vpmadd52luq`, 256-bit).",
	slice_doc = "`out[i] = src[i] + low52(a[i] * b[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512IfmaVl, vis = pub, target_feature = "avx512ifma,avx512vl",
	fixed_fn = madd52hi_u64x2, slice_fn = madd52hi_u64_slice, intrinsic_fn = madd52hi_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_madd52hi_epu64, scalar = madd52hi_scalar,
	fixed_doc = "`src + high52(a * b)` per lane (`vpmadd52huq`, 128-bit).",
	slice_doc = "`out[i] = src[i] + high52(a[i] * b[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512IfmaVl, vis = pub, target_feature = "avx512ifma,avx512vl",
	fixed_fn = madd52hi_u64x4, slice_fn = madd52hi_u64_slice_wide, intrinsic_fn = madd52hi_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_madd52hi_epu64, scalar = madd52hi_scalar,
	fixed_doc = "`src + high52(a * b)` per lane (`vpmadd52huq`, 256-bit).",
	slice_doc = "`out[i] = src[i] + high52(a[i] * b[i])`. 4-wide chunks, software scalar rem.",
}

// The mask stays `u8` at both widths: `__mmask8` is the AVX-512 architectural
// minimum, so only the low 2 (128-bit) or 4 (256-bit) bits are live.
macro_rules! avx512ifma_vl_ternop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident,
	 $width:literal, $Vec:ty, $loadu:path, $storeu:path, $merge_intrinsic:path, $zero_intrinsic:path,
	 $merge_doc:literal, $zero_doc:literal) => {
		simd_ternop_masked! {
			token = Avx512IfmaVl, target_feature = "avx512ifma,avx512vl",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = $width, elem = u64, vec = $Vec, mask = u8,
			loadu = $loadu, storeu = $storeu,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512ifma_vl_ternop_masked!(
	madd52lo_u64x2_merge_masked, madd52lo_u64x2_zero_masked,
	mask_madd52lo_epu64_128_intrinsic, maskz_madd52lo_epu64_128_intrinsic,
	2, __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_mask_madd52lo_epu64, _mm_maskz_madd52lo_epu64,
	"`src + low52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52luq`, 128-bit, merge-masked).",
	"`src + low52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52luq`, 128-bit, zero-masked)."
);
avx512ifma_vl_ternop_masked!(
	madd52lo_u64x4_merge_masked, madd52lo_u64x4_zero_masked,
	mask_madd52lo_epu64_256_intrinsic, maskz_madd52lo_epu64_256_intrinsic,
	4, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_mask_madd52lo_epu64, _mm256_maskz_madd52lo_epu64,
	"`src + low52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52luq`, 256-bit, merge-masked).",
	"`src + low52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52luq`, 256-bit, zero-masked)."
);
avx512ifma_vl_ternop_masked!(
	madd52hi_u64x2_merge_masked, madd52hi_u64x2_zero_masked,
	mask_madd52hi_epu64_128_intrinsic, maskz_madd52hi_epu64_128_intrinsic,
	2, __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_mask_madd52hi_epu64, _mm_maskz_madd52hi_epu64,
	"`src + high52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52huq`, 128-bit, merge-masked).",
	"`src + high52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52huq`, 128-bit, zero-masked)."
);
avx512ifma_vl_ternop_masked!(
	madd52hi_u64x4_merge_masked, madd52hi_u64x4_zero_masked,
	mask_madd52hi_epu64_256_intrinsic, maskz_madd52hi_epu64_256_intrinsic,
	4, __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_mask_madd52hi_epu64, _mm256_maskz_madd52hi_epu64,
	"`src + high52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52huq`, 256-bit, merge-masked).",
	"`src + high52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52huq`, 256-bit, zero-masked)."
);

/// Proof token: AVX512VBMI *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512VbmiVl(());

impl Avx512VbmiVl {
	/// `None` unless the CPU has both AVX512VBMI and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512vbmi) && set.contains(Feature::Avx512vl)).then_some(Avx512VbmiVl(()))
	}

	/// 256-bit `out[i] = a[idx[i] & 31]` (`vpermb`).
	#[inline]
	pub fn permutexvar_u8x32(self, idx: [u8; 32], a: [u8; 32]) -> [u8; 32] {
		unsafe { permutexvar_256(&idx, &a) }
	}

	/// 128-bit `out[i] = a[idx[i] & 15]` (`vpermb`).
	#[inline]
	pub fn permutexvar_u8x16(self, idx: [u8; 16], a: [u8; 16]) -> [u8; 16] {
		unsafe { permutexvar_128(&idx, &a) }
	}

	/// 256-bit 2-source byte permute (`vpermi2b`): index bit 5 (`0x20`) selects `a`/`b`.
	#[inline]
	pub fn permutex2var_u8x32(self, a: [u8; 32], idx: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		unsafe { permutex2var_256(&a, &idx, &b) }
	}

	/// 128-bit 2-source byte permute (`vpermi2b`): index bit 4 (`0x10`) selects `a`/`b`.
	#[inline]
	pub fn permutex2var_u8x16(self, a: [u8; 16], idx: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		unsafe { permutex2var_128(&a, &idx, &b) }
	}

	/// 256-bit `vpmultishiftqb`, same per-qword semantics as
	/// [`super::avx512vbmi::Avx512Vbmi::multishift_u8x64`], 4 qwords.
	#[inline]
	pub fn multishift_u8x32(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		unsafe { multishift_256(&a, &b) }
	}

	/// 128-bit `vpmultishiftqb`, 2 qwords.
	#[inline]
	pub fn multishift_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		unsafe { multishift_128(&a, &b) }
	}
}

// Merge/zero-masked forms: same macro-reuse reasoning as
// `super::avx512vbmi`'s 512-bit masked forms: `permutexvar`/`multishift` fit
// `simd_binop_masked!`, `permutex2var`'s merge form reuses `a` as the
// fallback so it fits `simd_ternop_masked!`.
simd_binop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = permutexvar_u8x32_merge_masked, zero_fn = permutexvar_u8x32_zero_masked,
	merge_intrinsic_fn = mask_permutexvar_u8x32_intrinsic, zero_intrinsic_fn = maskz_permutexvar_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i, mask = u32,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_permutexvar_epi8, zero_intrinsic = _mm256_maskz_permutexvar_epi8,
	merge_doc = "[`Avx512VbmiVl::permutexvar_u8x32`] where `mask` bit is set, else copied from `src` (`vpermb`, merge-masked).",
	zero_doc = "[`Avx512VbmiVl::permutexvar_u8x32`] where `mask` bit is set, else zero (`vpermb`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = permutexvar_u8x16_merge_masked, zero_fn = permutexvar_u8x16_zero_masked,
	merge_intrinsic_fn = mask_permutexvar_u8x16_intrinsic, zero_intrinsic_fn = maskz_permutexvar_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i, mask = u16,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_permutexvar_epi8, zero_intrinsic = _mm_maskz_permutexvar_epi8,
	merge_doc = "[`Avx512VbmiVl::permutexvar_u8x16`] where `mask` bit is set, else copied from `src` (`vpermb`, merge-masked).",
	zero_doc = "[`Avx512VbmiVl::permutexvar_u8x16`] where `mask` bit is set, else zero (`vpermb`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = permutex2var_u8x32_merge_masked, zero_fn = permutex2var_u8x32_zero_masked,
	merge_intrinsic_fn = mask_permutex2var_u8x32_intrinsic, zero_intrinsic_fn = maskz_permutex2var_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i, mask = u32,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_permutex2var_epi8, zero_intrinsic = _mm256_maskz_permutex2var_epi8,
	merge_doc = "[`Avx512VbmiVl::permutex2var_u8x32`] where `mask` bit is set, else copied from `a` (`vpermi2b`, merge-masked). `a` doubles as both a permute input and the merge fallback - the encoding has no separate `src`.",
	zero_doc = "[`Avx512VbmiVl::permutex2var_u8x32`] where `mask` bit is set, else zero (`vpermi2b`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = permutex2var_u8x16_merge_masked, zero_fn = permutex2var_u8x16_zero_masked,
	merge_intrinsic_fn = mask_permutex2var_u8x16_intrinsic, zero_intrinsic_fn = maskz_permutex2var_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i, mask = u16,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_permutex2var_epi8, zero_intrinsic = _mm_maskz_permutex2var_epi8,
	merge_doc = "[`Avx512VbmiVl::permutex2var_u8x16`] where `mask` bit is set, else copied from `a` (`vpermi2b`, merge-masked).",
	zero_doc = "[`Avx512VbmiVl::permutex2var_u8x16`] where `mask` bit is set, else zero (`vpermi2b`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = multishift_u8x32_merge_masked, zero_fn = multishift_u8x32_zero_masked,
	merge_intrinsic_fn = mask_multishift_u8x32_intrinsic, zero_intrinsic_fn = maskz_multishift_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i, mask = u32,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_multishift_epi64_epi8, zero_intrinsic = _mm256_maskz_multishift_epi64_epi8,
	merge_doc = "[`Avx512VbmiVl::multishift_u8x32`] where `mask` bit is set, else copied from `src` (`vpmultishiftqb`, merge-masked).",
	zero_doc = "[`Avx512VbmiVl::multishift_u8x32`] where `mask` bit is set, else zero (`vpmultishiftqb`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512VbmiVl, target_feature = "avx512vbmi,avx512vl",
	merge_fn = multishift_u8x16_merge_masked, zero_fn = multishift_u8x16_zero_masked,
	merge_intrinsic_fn = mask_multishift_u8x16_intrinsic, zero_intrinsic_fn = maskz_multishift_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i, mask = u16,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_multishift_epi64_epi8, zero_intrinsic = _mm_maskz_multishift_epi64_epi8,
	merge_doc = "[`Avx512VbmiVl::multishift_u8x16`] where `mask` bit is set, else copied from `src` (`vpmultishiftqb`, merge-masked).",
	zero_doc = "[`Avx512VbmiVl::multishift_u8x16`] where `mask` bit is set, else zero (`vpmultishiftqb`, zero-masked).",
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn permutexvar_256(idx: &[u8; 32], a: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let vidx: __m256i = _mm256_loadu_si256(idx.as_ptr().cast());
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vr = _mm256_permutexvar_epi8(vidx, va);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn permutexvar_128(idx: &[u8; 16], a: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vidx: __m128i = _mm_loadu_si128(idx.as_ptr().cast());
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vr = _mm_permutexvar_epi8(vidx, va);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn permutex2var_256(a: &[u8; 32], idx: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vidx: __m256i = _mm256_loadu_si256(idx.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vr = _mm256_permutex2var_epi8(va, vidx, vb);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn permutex2var_128(a: &[u8; 16], idx: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vidx: __m128i = _mm_loadu_si128(idx.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_permutex2var_epi8(va, vidx, vb);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn multishift_256(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vr = _mm256_multishift_epi64_epi8(va, vb);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
unsafe fn multishift_128(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_multishift_epi64_epi8(va, vb);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Proof token: AVX512VBMI2 *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Vbmi2Vl(());

impl Avx512Vbmi2Vl {
	/// `None` unless the CPU has both AVX512VBMI2 and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512vbmi2) && set.contains(Feature::Avx512vl)).then_some(Avx512Vbmi2Vl(()))
	}
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i16x16, slice_fn = shldv_i16_slice_wide, intrinsic_fn = shldv_i16x16_intrinsic,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi16, scalar = shldv_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u16x16, slice_fn = shldv_u16_slice_wide, intrinsic_fn = shldv_u16x16_intrinsic,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi16, scalar = shldv_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i16x8, slice_fn = shldv_i16_slice, intrinsic_fn = shldv_i16x8_intrinsic,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi16, scalar = shldv_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u16x8, slice_fn = shldv_u16_slice, intrinsic_fn = shldv_u16x8_intrinsic,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi16, scalar = shldv_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i16x16, slice_fn = shrdv_i16_slice_wide, intrinsic_fn = shrdv_i16x16_intrinsic,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi16, scalar = shrdv_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u16x16, slice_fn = shrdv_u16_slice_wide, intrinsic_fn = shrdv_u16x16_intrinsic,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi16, scalar = shrdv_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i16x8, slice_fn = shrdv_i16_slice, intrinsic_fn = shrdv_i16x8_intrinsic,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi16, scalar = shrdv_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u16x8, slice_fn = shrdv_u16_slice, intrinsic_fn = shrdv_u16x8_intrinsic,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi16, scalar = shrdv_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i32x8, slice_fn = shldv_i32_slice_wide, intrinsic_fn = shldv_i32x8_intrinsic,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi32, scalar = shldv_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u32x8, slice_fn = shldv_u32_slice_wide, intrinsic_fn = shldv_u32x8_intrinsic,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi32, scalar = shldv_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i32x4, slice_fn = shldv_i32_slice, intrinsic_fn = shldv_i32x4_intrinsic,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi32, scalar = shldv_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u32x4, slice_fn = shldv_u32_slice, intrinsic_fn = shldv_u32x4_intrinsic,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi32, scalar = shldv_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i32x8, slice_fn = shrdv_i32_slice_wide, intrinsic_fn = shrdv_i32x8_intrinsic,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi32, scalar = shrdv_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u32x8, slice_fn = shrdv_u32_slice_wide, intrinsic_fn = shrdv_u32x8_intrinsic,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi32, scalar = shrdv_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i32x4, slice_fn = shrdv_i32_slice, intrinsic_fn = shrdv_i32x4_intrinsic,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi32, scalar = shrdv_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u32x4, slice_fn = shrdv_u32_slice, intrinsic_fn = shrdv_u32x4_intrinsic,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi32, scalar = shrdv_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i64x4, slice_fn = shldv_i64_slice_wide, intrinsic_fn = shldv_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi64, scalar = shldv_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u64x4, slice_fn = shldv_u64_slice_wide, intrinsic_fn = shldv_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldv_epi64, scalar = shldv_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 256-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_i64x2, slice_fn = shldv_i64_slice, intrinsic_fn = shldv_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi64, scalar = shldv_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldv_u64x2, slice_fn = shldv_u64_slice, intrinsic_fn = shldv_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldv_epi64, scalar = shldv_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 128-bit).",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i64x4, slice_fn = shrdv_i64_slice_wide, intrinsic_fn = shrdv_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi64, scalar = shrdv_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u64x4, slice_fn = shrdv_u64_slice_wide, intrinsic_fn = shrdv_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdv_epi64, scalar = shrdv_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 256-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_i64x2, slice_fn = shrdv_i64_slice, intrinsic_fn = shrdv_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi64, scalar = shrdv_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdv_u64x2, slice_fn = shrdv_u64_slice, intrinsic_fn = shrdv_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdv_epi64, scalar = shrdv_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 128-bit).",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 2-wide chunks, software scalar rem.",
}

// shldv/shrdv merge/zero-masked at 128/256-bit: same gap as shldi/shrdi
// above, `a` is the merge fallback (no separate `src`), same shape FMA uses.

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i16x16_merge_masked, zero_fn = shldv_i16x16_zero_masked,
	merge_intrinsic_fn = shldv_i16x16_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i16x16_zero_masked_intrinsic,
	width = 16, elem = i16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi16, zero_intrinsic = _mm256_maskz_shldv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i16x16`] (`vpshldvw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i16x16`] (`vpshldvw`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u16x16_merge_masked, zero_fn = shldv_u16x16_zero_masked,
	merge_intrinsic_fn = shldv_u16x16_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u16x16_zero_masked_intrinsic,
	width = 16, elem = u16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi16, zero_intrinsic = _mm256_maskz_shldv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u16x16`] (`vpshldvw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u16x16`] (`vpshldvw`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i16x8_merge_masked, zero_fn = shldv_i16x8_zero_masked,
	merge_intrinsic_fn = shldv_i16x8_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i16x8_zero_masked_intrinsic,
	width = 8, elem = i16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi16, zero_intrinsic = _mm_maskz_shldv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i16x8`] (`vpshldvw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i16x8`] (`vpshldvw`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u16x8_merge_masked, zero_fn = shldv_u16x8_zero_masked,
	merge_intrinsic_fn = shldv_u16x8_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u16x8_zero_masked_intrinsic,
	width = 8, elem = u16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi16, zero_intrinsic = _mm_maskz_shldv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u16x8`] (`vpshldvw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u16x8`] (`vpshldvw`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i16x16_merge_masked, zero_fn = shrdv_i16x16_zero_masked,
	merge_intrinsic_fn = shrdv_i16x16_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i16x16_zero_masked_intrinsic,
	width = 16, elem = i16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi16, zero_intrinsic = _mm256_maskz_shrdv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i16x16`] (`vpshrdvw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i16x16`] (`vpshrdvw`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u16x16_merge_masked, zero_fn = shrdv_u16x16_zero_masked,
	merge_intrinsic_fn = shrdv_u16x16_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u16x16_zero_masked_intrinsic,
	width = 16, elem = u16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi16, zero_intrinsic = _mm256_maskz_shrdv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u16x16`] (`vpshrdvw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u16x16`] (`vpshrdvw`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i16x8_merge_masked, zero_fn = shrdv_i16x8_zero_masked,
	merge_intrinsic_fn = shrdv_i16x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i16x8_zero_masked_intrinsic,
	width = 8, elem = i16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi16, zero_intrinsic = _mm_maskz_shrdv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i16x8`] (`vpshrdvw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i16x8`] (`vpshrdvw`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u16x8_merge_masked, zero_fn = shrdv_u16x8_zero_masked,
	merge_intrinsic_fn = shrdv_u16x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u16x8_zero_masked_intrinsic,
	width = 8, elem = u16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi16, zero_intrinsic = _mm_maskz_shrdv_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u16x8`] (`vpshrdvw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u16x8`] (`vpshrdvw`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i32x8_merge_masked, zero_fn = shldv_i32x8_zero_masked,
	merge_intrinsic_fn = shldv_i32x8_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i32x8_zero_masked_intrinsic,
	width = 8, elem = i32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi32, zero_intrinsic = _mm256_maskz_shldv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i32x8`] (`vpshldvd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i32x8`] (`vpshldvd`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u32x8_merge_masked, zero_fn = shldv_u32x8_zero_masked,
	merge_intrinsic_fn = shldv_u32x8_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u32x8_zero_masked_intrinsic,
	width = 8, elem = u32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi32, zero_intrinsic = _mm256_maskz_shldv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u32x8`] (`vpshldvd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u32x8`] (`vpshldvd`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i32x4_merge_masked, zero_fn = shldv_i32x4_zero_masked,
	merge_intrinsic_fn = shldv_i32x4_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i32x4_zero_masked_intrinsic,
	width = 4, elem = i32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi32, zero_intrinsic = _mm_maskz_shldv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i32x4`] (`vpshldvd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i32x4`] (`vpshldvd`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u32x4_merge_masked, zero_fn = shldv_u32x4_zero_masked,
	merge_intrinsic_fn = shldv_u32x4_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u32x4_zero_masked_intrinsic,
	width = 4, elem = u32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi32, zero_intrinsic = _mm_maskz_shldv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u32x4`] (`vpshldvd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u32x4`] (`vpshldvd`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i32x8_merge_masked, zero_fn = shrdv_i32x8_zero_masked,
	merge_intrinsic_fn = shrdv_i32x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i32x8_zero_masked_intrinsic,
	width = 8, elem = i32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi32, zero_intrinsic = _mm256_maskz_shrdv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i32x8`] (`vpshrdvd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i32x8`] (`vpshrdvd`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u32x8_merge_masked, zero_fn = shrdv_u32x8_zero_masked,
	merge_intrinsic_fn = shrdv_u32x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u32x8_zero_masked_intrinsic,
	width = 8, elem = u32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi32, zero_intrinsic = _mm256_maskz_shrdv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u32x8`] (`vpshrdvd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u32x8`] (`vpshrdvd`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i32x4_merge_masked, zero_fn = shrdv_i32x4_zero_masked,
	merge_intrinsic_fn = shrdv_i32x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i32x4_zero_masked_intrinsic,
	width = 4, elem = i32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi32, zero_intrinsic = _mm_maskz_shrdv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i32x4`] (`vpshrdvd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i32x4`] (`vpshrdvd`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u32x4_merge_masked, zero_fn = shrdv_u32x4_zero_masked,
	merge_intrinsic_fn = shrdv_u32x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u32x4_zero_masked_intrinsic,
	width = 4, elem = u32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi32, zero_intrinsic = _mm_maskz_shrdv_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u32x4`] (`vpshrdvd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u32x4`] (`vpshrdvd`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i64x4_merge_masked, zero_fn = shldv_i64x4_zero_masked,
	merge_intrinsic_fn = shldv_i64x4_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i64x4_zero_masked_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi64, zero_intrinsic = _mm256_maskz_shldv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i64x4`] (`vpshldvq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i64x4`] (`vpshldvq`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u64x4_merge_masked, zero_fn = shldv_u64x4_zero_masked,
	merge_intrinsic_fn = shldv_u64x4_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u64x4_zero_masked_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldv_epi64, zero_intrinsic = _mm256_maskz_shldv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u64x4`] (`vpshldvq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u64x4`] (`vpshldvq`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_i64x2_merge_masked, zero_fn = shldv_i64x2_zero_masked,
	merge_intrinsic_fn = shldv_i64x2_merge_masked_intrinsic, zero_intrinsic_fn = shldv_i64x2_zero_masked_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi64, zero_intrinsic = _mm_maskz_shldv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_i64x2`] (`vpshldvq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_i64x2`] (`vpshldvq`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldv_u64x2_merge_masked, zero_fn = shldv_u64x2_zero_masked,
	merge_intrinsic_fn = shldv_u64x2_merge_masked_intrinsic, zero_intrinsic_fn = shldv_u64x2_zero_masked_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldv_epi64, zero_intrinsic = _mm_maskz_shldv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldv_u64x2`] (`vpshldvq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldv_u64x2`] (`vpshldvq`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i64x4_merge_masked, zero_fn = shrdv_i64x4_zero_masked,
	merge_intrinsic_fn = shrdv_i64x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i64x4_zero_masked_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi64, zero_intrinsic = _mm256_maskz_shrdv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i64x4`] (`vpshrdvq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i64x4`] (`vpshrdvq`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u64x4_merge_masked, zero_fn = shrdv_u64x4_zero_masked,
	merge_intrinsic_fn = shrdv_u64x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u64x4_zero_masked_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdv_epi64, zero_intrinsic = _mm256_maskz_shrdv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u64x4`] (`vpshrdvq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u64x4`] (`vpshrdvq`, 256-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_i64x2_merge_masked, zero_fn = shrdv_i64x2_zero_masked,
	merge_intrinsic_fn = shrdv_i64x2_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_i64x2_zero_masked_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi64, zero_intrinsic = _mm_maskz_shrdv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_i64x2`] (`vpshrdvq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_i64x2`] (`vpshrdvq`, 128-bit).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdv_u64x2_merge_masked, zero_fn = shrdv_u64x2_zero_masked,
	merge_intrinsic_fn = shrdv_u64x2_merge_masked_intrinsic, zero_intrinsic_fn = shrdv_u64x2_zero_masked_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdv_epi64, zero_intrinsic = _mm_maskz_shrdv_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdv_u64x2`] (`vpshrdvq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdv_u64x2`] (`vpshrdvq`, 128-bit).",
}

// shldi/shrdi: immediate funnel shift. Masked forms follow the unmasked
// block below (see avx512vbmi2.rs module docs for the src/mask shape).

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i16x16, slice_fn = shldi_i16_slice_wide, intrinsic_fn = shldi_i16x16_intrinsic,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi16, scalar = shldi_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u16x16, slice_fn = shldi_u16_slice_wide, intrinsic_fn = shldi_u16x16_intrinsic,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi16, scalar = shldi_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i16x8, slice_fn = shldi_i16_slice, intrinsic_fn = shldi_i16x8_intrinsic,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi16, scalar = shldi_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u16x8, slice_fn = shldi_u16_slice, intrinsic_fn = shldi_u16x8_intrinsic,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi16, scalar = shldi_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i16x16, slice_fn = shrdi_i16_slice_wide, intrinsic_fn = shrdi_i16x16_intrinsic,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi16, scalar = shrdi_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u16x16, slice_fn = shrdi_u16_slice_wide, intrinsic_fn = shrdi_u16x16_intrinsic,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi16, scalar = shrdi_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i16x8, slice_fn = shrdi_i16_slice, intrinsic_fn = shrdi_i16x8_intrinsic,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi16, scalar = shrdi_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u16x8, slice_fn = shrdi_u16_slice, intrinsic_fn = shrdi_u16x8_intrinsic,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi16, scalar = shrdi_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i32x8, slice_fn = shldi_i32_slice_wide, intrinsic_fn = shldi_i32x8_intrinsic,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi32, scalar = shldi_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u32x8, slice_fn = shldi_u32_slice_wide, intrinsic_fn = shldi_u32x8_intrinsic,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi32, scalar = shldi_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i32x4, slice_fn = shldi_i32_slice, intrinsic_fn = shldi_i32x4_intrinsic,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi32, scalar = shldi_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u32x4, slice_fn = shldi_u32_slice, intrinsic_fn = shldi_u32x4_intrinsic,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi32, scalar = shldi_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i32x8, slice_fn = shrdi_i32_slice_wide, intrinsic_fn = shrdi_i32x8_intrinsic,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi32, scalar = shrdi_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u32x8, slice_fn = shrdi_u32_slice_wide, intrinsic_fn = shrdi_u32x8_intrinsic,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi32, scalar = shrdi_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i32x4, slice_fn = shrdi_i32_slice, intrinsic_fn = shrdi_i32x4_intrinsic,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi32, scalar = shrdi_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u32x4, slice_fn = shrdi_u32_slice, intrinsic_fn = shrdi_u32x4_intrinsic,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi32, scalar = shrdi_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i64x4, slice_fn = shldi_i64_slice_wide, intrinsic_fn = shldi_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi64, scalar = shldi_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u64x4, slice_fn = shldi_u64_slice_wide, intrinsic_fn = shldi_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shldi_epi64, scalar = shldi_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 256-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_i64x2, slice_fn = shldi_i64_slice, intrinsic_fn = shldi_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi64, scalar = shldi_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 2-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shldi_u64x2, slice_fn = shldi_u64_slice, intrinsic_fn = shldi_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shldi_epi64, scalar = shldi_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 128-bit).",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 2-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i64x4, slice_fn = shrdi_i64_slice_wide, intrinsic_fn = shrdi_i64x4_intrinsic,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi64, scalar = shrdi_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u64x4, slice_fn = shrdi_u64_slice_wide, intrinsic_fn = shrdi_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_shrdi_epi64, scalar = shrdi_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 256-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 4-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_i64x2, slice_fn = shrdi_i64_slice, intrinsic_fn = shrdi_i64x2_intrinsic,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi64, scalar = shrdi_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 2-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2Vl, vis = pub, target_feature = "avx512vbmi2,avx512vl",
	fixed_fn = shrdi_u64x2, slice_fn = shrdi_u64_slice, intrinsic_fn = shrdi_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shrdi_epi64, scalar = shrdi_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 128-bit).",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 2-wide chunks, software scalar rem.",
}

// shldi/shrdi merge/zero-masked at 128/256-bit: closes the VBMI2-rest
// deferral, `Avx512Vbmi2` (512-bit) already had these.

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i16x16_merge_masked, zero_fn = shldi_i16x16_zero_masked,
	merge_intrinsic_fn = shldi_i16x16_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i16x16_zero_masked_intrinsic,
	width = 16, elem = i16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi16, zero_intrinsic = _mm256_maskz_shldi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i16x16`] (`vpshldw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i16x16`] (`vpshldw`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u16x16_merge_masked, zero_fn = shldi_u16x16_zero_masked,
	merge_intrinsic_fn = shldi_u16x16_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u16x16_zero_masked_intrinsic,
	width = 16, elem = u16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi16, zero_intrinsic = _mm256_maskz_shldi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u16x16`] (`vpshldw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u16x16`] (`vpshldw`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i16x8_merge_masked, zero_fn = shldi_i16x8_zero_masked,
	merge_intrinsic_fn = shldi_i16x8_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i16x8_zero_masked_intrinsic,
	width = 8, elem = i16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi16, zero_intrinsic = _mm_maskz_shldi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i16x8`] (`vpshldw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i16x8`] (`vpshldw`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u16x8_merge_masked, zero_fn = shldi_u16x8_zero_masked,
	merge_intrinsic_fn = shldi_u16x8_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u16x8_zero_masked_intrinsic,
	width = 8, elem = u16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi16, zero_intrinsic = _mm_maskz_shldi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u16x8`] (`vpshldw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u16x8`] (`vpshldw`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i16x16_merge_masked, zero_fn = shrdi_i16x16_zero_masked,
	merge_intrinsic_fn = shrdi_i16x16_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i16x16_zero_masked_intrinsic,
	width = 16, elem = i16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi16, zero_intrinsic = _mm256_maskz_shrdi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i16x16`] (`vpshrdw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i16x16`] (`vpshrdw`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u16x16_merge_masked, zero_fn = shrdi_u16x16_zero_masked,
	merge_intrinsic_fn = shrdi_u16x16_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u16x16_zero_masked_intrinsic,
	width = 16, elem = u16, vec = __m256i, mask = u16,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi16, zero_intrinsic = _mm256_maskz_shrdi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u16x16`] (`vpshrdw`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u16x16`] (`vpshrdw`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i16x8_merge_masked, zero_fn = shrdi_i16x8_zero_masked,
	merge_intrinsic_fn = shrdi_i16x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i16x8_zero_masked_intrinsic,
	width = 8, elem = i16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi16, zero_intrinsic = _mm_maskz_shrdi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i16x8`] (`vpshrdw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i16x8`] (`vpshrdw`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u16x8_merge_masked, zero_fn = shrdi_u16x8_zero_masked,
	merge_intrinsic_fn = shrdi_u16x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u16x8_zero_masked_intrinsic,
	width = 8, elem = u16, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi16, zero_intrinsic = _mm_maskz_shrdi_epi16,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u16x8`] (`vpshrdw`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u16x8`] (`vpshrdw`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i32x8_merge_masked, zero_fn = shldi_i32x8_zero_masked,
	merge_intrinsic_fn = shldi_i32x8_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i32x8_zero_masked_intrinsic,
	width = 8, elem = i32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi32, zero_intrinsic = _mm256_maskz_shldi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i32x8`] (`vpshldd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i32x8`] (`vpshldd`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u32x8_merge_masked, zero_fn = shldi_u32x8_zero_masked,
	merge_intrinsic_fn = shldi_u32x8_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u32x8_zero_masked_intrinsic,
	width = 8, elem = u32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi32, zero_intrinsic = _mm256_maskz_shldi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u32x8`] (`vpshldd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u32x8`] (`vpshldd`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i32x4_merge_masked, zero_fn = shldi_i32x4_zero_masked,
	merge_intrinsic_fn = shldi_i32x4_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i32x4_zero_masked_intrinsic,
	width = 4, elem = i32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi32, zero_intrinsic = _mm_maskz_shldi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i32x4`] (`vpshldd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i32x4`] (`vpshldd`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u32x4_merge_masked, zero_fn = shldi_u32x4_zero_masked,
	merge_intrinsic_fn = shldi_u32x4_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u32x4_zero_masked_intrinsic,
	width = 4, elem = u32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi32, zero_intrinsic = _mm_maskz_shldi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u32x4`] (`vpshldd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u32x4`] (`vpshldd`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i32x8_merge_masked, zero_fn = shrdi_i32x8_zero_masked,
	merge_intrinsic_fn = shrdi_i32x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i32x8_zero_masked_intrinsic,
	width = 8, elem = i32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi32, zero_intrinsic = _mm256_maskz_shrdi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i32x8`] (`vpshrdd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i32x8`] (`vpshrdd`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u32x8_merge_masked, zero_fn = shrdi_u32x8_zero_masked,
	merge_intrinsic_fn = shrdi_u32x8_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u32x8_zero_masked_intrinsic,
	width = 8, elem = u32, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi32, zero_intrinsic = _mm256_maskz_shrdi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u32x8`] (`vpshrdd`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u32x8`] (`vpshrdd`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i32x4_merge_masked, zero_fn = shrdi_i32x4_zero_masked,
	merge_intrinsic_fn = shrdi_i32x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i32x4_zero_masked_intrinsic,
	width = 4, elem = i32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi32, zero_intrinsic = _mm_maskz_shrdi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i32x4`] (`vpshrdd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i32x4`] (`vpshrdd`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u32x4_merge_masked, zero_fn = shrdi_u32x4_zero_masked,
	merge_intrinsic_fn = shrdi_u32x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u32x4_zero_masked_intrinsic,
	width = 4, elem = u32, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi32, zero_intrinsic = _mm_maskz_shrdi_epi32,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u32x4`] (`vpshrdd`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u32x4`] (`vpshrdd`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i64x4_merge_masked, zero_fn = shldi_i64x4_zero_masked,
	merge_intrinsic_fn = shldi_i64x4_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i64x4_zero_masked_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi64, zero_intrinsic = _mm256_maskz_shldi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i64x4`] (`vpshldq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i64x4`] (`vpshldq`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u64x4_merge_masked, zero_fn = shldi_u64x4_zero_masked,
	merge_intrinsic_fn = shldi_u64x4_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u64x4_zero_masked_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shldi_epi64, zero_intrinsic = _mm256_maskz_shldi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u64x4`] (`vpshldq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u64x4`] (`vpshldq`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_i64x2_merge_masked, zero_fn = shldi_i64x2_zero_masked,
	merge_intrinsic_fn = shldi_i64x2_merge_masked_intrinsic, zero_intrinsic_fn = shldi_i64x2_zero_masked_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi64, zero_intrinsic = _mm_maskz_shldi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_i64x2`] (`vpshldq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_i64x2`] (`vpshldq`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shldi_u64x2_merge_masked, zero_fn = shldi_u64x2_zero_masked,
	merge_intrinsic_fn = shldi_u64x2_merge_masked_intrinsic, zero_intrinsic_fn = shldi_u64x2_zero_masked_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shldi_epi64, zero_intrinsic = _mm_maskz_shldi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shldi_u64x2`] (`vpshldq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shldi_u64x2`] (`vpshldq`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i64x4_merge_masked, zero_fn = shrdi_i64x4_zero_masked,
	merge_intrinsic_fn = shrdi_i64x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i64x4_zero_masked_intrinsic,
	width = 4, elem = i64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi64, zero_intrinsic = _mm256_maskz_shrdi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i64x4`] (`vpshrdq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i64x4`] (`vpshrdq`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u64x4_merge_masked, zero_fn = shrdi_u64x4_zero_masked,
	merge_intrinsic_fn = shrdi_u64x4_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u64x4_zero_masked_intrinsic,
	width = 4, elem = u64, vec = __m256i, mask = u8,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_shrdi_epi64, zero_intrinsic = _mm256_maskz_shrdi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u64x4`] (`vpshrdq`, 256-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u64x4`] (`vpshrdq`, 256-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_i64x2_merge_masked, zero_fn = shrdi_i64x2_zero_masked,
	merge_intrinsic_fn = shrdi_i64x2_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_i64x2_zero_masked_intrinsic,
	width = 2, elem = i64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi64, zero_intrinsic = _mm_maskz_shrdi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_i64x2`] (`vpshrdq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_i64x2`] (`vpshrdq`, 128-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2Vl, target_feature = "avx512vbmi2,avx512vl",
	merge_fn = shrdi_u64x2_merge_masked, zero_fn = shrdi_u64x2_zero_masked,
	merge_intrinsic_fn = shrdi_u64x2_merge_masked_intrinsic, zero_intrinsic_fn = shrdi_u64x2_zero_masked_intrinsic,
	width = 2, elem = u64, vec = __m128i, mask = u8,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_shrdi_epi64, zero_intrinsic = _mm_maskz_shrdi_epi64,
	merge_doc = "Merge-masked [`Avx512Vbmi2Vl::shrdi_u64x2`] (`vpshrdq`, 128-bit).",
	zero_doc = "Zero-masked [`Avx512Vbmi2Vl::shrdi_u64x2`] (`vpshrdq`, 128-bit).",
}

/// Proof token: AVX512VNNI *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512VnniVl(());

impl Avx512VnniVl {
	/// `None` unless the CPU has both AVX512VNNI and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512vnni) && set.contains(Feature::Avx512vl)).then_some(Avx512VnniVl(()))
	}
}

// Plain EVEX names only (not `_avx` / INT8 / INT16). Intel order: (src, a, b).

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpbusd_i32x4, slice_fn = dpbusd_i32_slice, intrinsic_fn = dpbusd_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbusd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbusd`, 128-bit, `u8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpbusd_i32x8, slice_fn = dpbusd_i32_slice_wide, intrinsic_fn = dpbusd_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbusd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbusd`, 256-bit, `u8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpbusds_i32x4, slice_fn = dpbusds_i32_slice, intrinsic_fn = dpbusds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbusds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512VnniVl::dpbusd_i32x4`] (`vpdpbusds`, 128-bit).",
	slice_doc = "Saturating [`Avx512VnniVl::dpbusd_i32_slice`]. 4-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpbusds_i32x8, slice_fn = dpbusds_i32_slice_wide, intrinsic_fn = dpbusds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbusds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512VnniVl::dpbusd_i32x8`] (`vpdpbusds`, 256-bit).",
	slice_doc = "Saturating [`Avx512VnniVl::dpbusd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpwssd_i32x4, slice_fn = dpwssd_i32_slice, intrinsic_fn = dpwssd_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwssd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwssd`, 128-bit, `i16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpwssd_i32x8, slice_fn = dpwssd_i32_slice_wide, intrinsic_fn = dpwssd_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwssd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwssd`, 256-bit, `i16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpwssds_i32x4, slice_fn = dpwssds_i32_slice, intrinsic_fn = dpwssds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwssds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512VnniVl::dpwssd_i32x4`] (`vpdpwssds`, 128-bit).",
	slice_doc = "Saturating [`Avx512VnniVl::dpwssd_i32_slice`]. 4-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	fixed_fn = dpwssds_i32x8, slice_fn = dpwssds_i32_slice_wide, intrinsic_fn = dpwssds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwssds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512VnniVl::dpwssd_i32x8`] (`vpdpwssds`, 256-bit).",
	slice_doc = "Saturating [`Avx512VnniVl::dpwssd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpbusd_i32x4_merge_masked, zero_fn = dpbusd_i32x4_zero_masked,
	merge_intrinsic_fn = mask_dpbusd_i32x4_intrinsic, zero_intrinsic_fn = maskz_dpbusd_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, mask = u8, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_dpbusd_epi32, zero_intrinsic = _mm_maskz_dpbusd_epi32,
	merge_doc = "[`Avx512VnniVl::dpbusd_i32x4`] where `mask` bit is set, else copied from `src` (`vpdpbusd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpbusd_i32x4`] where `mask` bit is set, else zero (`vpdpbusd`, 128-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpbusd_i32x8_merge_masked, zero_fn = dpbusd_i32x8_zero_masked,
	merge_intrinsic_fn = mask_dpbusd_i32x8_intrinsic, zero_intrinsic_fn = maskz_dpbusd_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, mask = u8, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_dpbusd_epi32, zero_intrinsic = _mm256_maskz_dpbusd_epi32,
	merge_doc = "[`Avx512VnniVl::dpbusd_i32x8`] where `mask` bit is set, else copied from `src` (`vpdpbusd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpbusd_i32x8`] where `mask` bit is set, else zero (`vpdpbusd`, 256-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpbusds_i32x4_merge_masked, zero_fn = dpbusds_i32x4_zero_masked,
	merge_intrinsic_fn = mask_dpbusds_i32x4_intrinsic, zero_intrinsic_fn = maskz_dpbusds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, mask = u8, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_dpbusds_epi32, zero_intrinsic = _mm_maskz_dpbusds_epi32,
	merge_doc = "[`Avx512VnniVl::dpbusds_i32x4`] where `mask` bit is set, else copied from `src` (`vpdpbusds`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpbusds_i32x4`] where `mask` bit is set, else zero (`vpdpbusds`, 128-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpbusds_i32x8_merge_masked, zero_fn = dpbusds_i32x8_zero_masked,
	merge_intrinsic_fn = mask_dpbusds_i32x8_intrinsic, zero_intrinsic_fn = maskz_dpbusds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, mask = u8, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_dpbusds_epi32, zero_intrinsic = _mm256_maskz_dpbusds_epi32,
	merge_doc = "[`Avx512VnniVl::dpbusds_i32x8`] where `mask` bit is set, else copied from `src` (`vpdpbusds`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpbusds_i32x8`] where `mask` bit is set, else zero (`vpdpbusds`, 256-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpwssd_i32x4_merge_masked, zero_fn = dpwssd_i32x4_zero_masked,
	merge_intrinsic_fn = mask_dpwssd_i32x4_intrinsic, zero_intrinsic_fn = maskz_dpwssd_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, mask = u8, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_dpwssd_epi32, zero_intrinsic = _mm_maskz_dpwssd_epi32,
	merge_doc = "[`Avx512VnniVl::dpwssd_i32x4`] where `mask` bit is set, else copied from `src` (`vpdpwssd`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpwssd_i32x4`] where `mask` bit is set, else zero (`vpdpwssd`, 128-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpwssd_i32x8_merge_masked, zero_fn = dpwssd_i32x8_zero_masked,
	merge_intrinsic_fn = mask_dpwssd_i32x8_intrinsic, zero_intrinsic_fn = maskz_dpwssd_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, mask = u8, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_dpwssd_epi32, zero_intrinsic = _mm256_maskz_dpwssd_epi32,
	merge_doc = "[`Avx512VnniVl::dpwssd_i32x8`] where `mask` bit is set, else copied from `src` (`vpdpwssd`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpwssd_i32x8`] where `mask` bit is set, else zero (`vpdpwssd`, 256-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpwssds_i32x4_merge_masked, zero_fn = dpwssds_i32x4_zero_masked,
	merge_intrinsic_fn = mask_dpwssds_i32x4_intrinsic, zero_intrinsic_fn = maskz_dpwssds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, mask = u8, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm_mask_dpwssds_epi32, zero_intrinsic = _mm_maskz_dpwssds_epi32,
	merge_doc = "[`Avx512VnniVl::dpwssds_i32x4`] where `mask` bit is set, else copied from `src` (`vpdpwssds`, 128-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpwssds_i32x4`] where `mask` bit is set, else zero (`vpdpwssds`, 128-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

simd_vnni_dot_masked! {
	token = Avx512VnniVl, target_feature = "avx512vnni,avx512vl",
	merge_fn = dpwssds_i32x8_merge_masked, zero_fn = dpwssds_i32x8_zero_masked,
	merge_intrinsic_fn = mask_dpwssds_i32x8_intrinsic, zero_intrinsic_fn = maskz_dpwssds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, mask = u8, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm256_mask_dpwssds_epi32, zero_intrinsic = _mm256_maskz_dpwssds_epi32,
	merge_doc = "[`Avx512VnniVl::dpwssds_i32x8`] where `mask` bit is set, else copied from `src` (`vpdpwssds`, 256-bit, merge-masked).",
	zero_doc = "[`Avx512VnniVl::dpwssds_i32x8`] where `mask` bit is set, else zero (`vpdpwssds`, 256-bit, zero-masked). `src` is still a real input here, not just a merge fallback - see `simd_vnni_dot_masked`'s doc.",
}

/// Proof token: AVX512VPOPCNTDQ *and* AVX512VL, for the 128/256-bit forms.
#[derive(Debug, Clone, Copy)]
pub struct Avx512VpopcntdqVl(());

impl Avx512VpopcntdqVl {
	/// `None` unless the CPU has both AVX512VPOPCNTDQ and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512vpopcntdq) && set.contains(Feature::Avx512vl)).then_some(Avx512VpopcntdqVl(()))
	}
}

simd_unop! {
	token = Avx512VpopcntdqVl, target_feature = "avx512vpopcntdq,avx512vl",
	fixed_fn = popcnt_u32x4, slice_fn = popcnt_u32_slice, intrinsic_fn = popcnt_u32x4_intrinsic,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_popcnt_epi32, scalar = u32::count_ones,
	fixed_doc = "Per-lane population count (`vpopcntd`, 128-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 4-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512VpopcntdqVl, target_feature = "avx512vpopcntdq,avx512vl",
	fixed_fn = popcnt_u32x8, slice_fn = popcnt_u32_slice_wide, intrinsic_fn = popcnt_u32x8_intrinsic,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_popcnt_epi32, scalar = u32::count_ones,
	fixed_doc = "Per-lane population count (`vpopcntd`, 256-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 8-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512VpopcntdqVl, target_feature = "avx512vpopcntdq,avx512vl",
	fixed_fn = popcnt_u64x2, slice_fn = popcnt_u64_slice, intrinsic_fn = popcnt_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_popcnt_epi64, scalar = |x: u64| x.count_ones() as u64,
	fixed_doc = "Per-lane population count (`vpopcntq`, 128-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 2-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512VpopcntdqVl, target_feature = "avx512vpopcntdq,avx512vl",
	fixed_fn = popcnt_u64x4, slice_fn = popcnt_u64_slice_wide, intrinsic_fn = popcnt_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_popcnt_epi64, scalar = |x: u64| x.count_ones() as u64,
	fixed_doc = "Per-lane population count (`vpopcntq`, 256-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 4-wide chunks, scalar remainder.",
}

/// Proof token: AVX512BF16 *and* AVX512VL, for the 128/256-bit forms. BF16
/// dot-product accumulate and `f32`<->BF16 conversions, same op family as
/// [`super::avx512bf16::Avx512Bf16`]'s 512-bit forms; BF16 carried as raw
/// `u16` bits, no crate newtype (same choice as the 512-bit token). No
/// slice/`auto` wrapper: fixed-width only, matching [`Avx512VbmiVl`]'s
/// hand-written ops above.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Bf16Vl(());

impl Avx512Bf16Vl {
	/// `None` unless the CPU has both AVX512BF16 and AVX512VL.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512bf16) && set.contains(Feature::Avx512vl)).then_some(Avx512Bf16Vl(()))
	}

	/// `dst[j] = src[j] + f32(a[2j+1])*f32(b[2j+1]) + f32(a[2j])*f32(b[2j])`
	/// (`vdpbf16ps`, 256-bit). `a`/`b` are 16 BF16 bit patterns (8 pairs);
	/// `src`/result are 8 `f32`.
	#[inline]
	pub fn dpbf16_ps_f32x8(self, src: [f32; 8], a: [u16; 16], b: [u16; 16]) -> [f32; 8] {
		unsafe { dpbf16_ps_f32x8_intrinsic(&src, &a, &b) }
	}

	/// `dst[j] = src[j] + f32(a[2j+1])*f32(b[2j+1]) + f32(a[2j])*f32(b[2j])`
	/// (`vdpbf16ps`, 128-bit). `a`/`b` are 8 BF16 bit patterns (4 pairs);
	/// `src`/result are 4 `f32`.
	#[inline]
	pub fn dpbf16_ps_f32x4(self, src: [f32; 4], a: [u16; 8], b: [u16; 8]) -> [f32; 4] {
		unsafe { dpbf16_ps_f32x4_intrinsic(&src, &a, &b) }
	}

	/// 8 `f32` -> 8 BF16 bit patterns, round-to-nearest-even (`vcvtneps2bf16`,
	/// 256-bit source, 128-bit BF16 result).
	#[inline]
	pub fn cvtneps_pbh_u16x8(self, a: [f32; 8]) -> [u16; 8] {
		unsafe { cvtneps_pbh_u16x8_intrinsic(&a) }
	}

	/// 4 `f32` -> 4 BF16 bit patterns, round-to-nearest-even (`vcvtneps2bf16`,
	/// 128-bit). Hardware always produces a full 128-bit (8-lane) result -
	/// the instruction has no analog for 4 missing upper `f32` inputs: so
	/// only the low 4 lanes are meaningful; the other 4 are dropped here,
	/// same pattern as [`super::f16c::F16c`]'s 128-bit `f32_to_f16x4`.
	#[inline]
	pub fn cvtneps_pbh_u16x4(self, a: [f32; 4]) -> [u16; 4] {
		unsafe { cvtneps_pbh_u16x4_intrinsic(&a) }
	}

	/// Two 8-`f32` vectors -> 16 BF16 bit patterns, round-to-nearest-even;
	/// `b`'s lanes land in the low half, `a`'s in the high half
	/// (`vcvtne2ps2bf16`, 256-bit).
	#[inline]
	pub fn cvtne2ps_pbh_u16x16(self, a: [f32; 8], b: [f32; 8]) -> [u16; 16] {
		unsafe { cvtne2ps_pbh_u16x16_intrinsic(&a, &b) }
	}

	/// Two 4-`f32` vectors -> 8 BF16 bit patterns, round-to-nearest-even;
	/// `b`'s lanes land in the low half, `a`'s in the high half
	/// (`vcvtne2ps2bf16`, 128-bit).
	#[inline]
	pub fn cvtne2ps_pbh_u16x8(self, a: [f32; 4], b: [f32; 4]) -> [u16; 8] {
		unsafe { cvtne2ps_pbh_u16x8_intrinsic(&a, &b) }
	}

	/// 8 BF16 bit patterns -> 8 `f32`, exact: same op family as
	/// [`super::avx512bf16::Avx512Bf16::cvtpbh_ps_f32x16`] (`vcvtpbh2ps`,
	/// 128-bit BF16 source, 256-bit result).
	#[inline]
	pub fn cvtpbh_ps_f32x8(self, a: [u16; 8]) -> [f32; 8] {
		unsafe { cvtpbh_ps_f32x8_intrinsic(&a) }
	}

	/// 4 BF16 bit patterns -> 4 `f32`, exact (`vcvtpbh2ps`, 128-bit). Only the
	/// low 4 lanes of the 8-lane BF16 register the hardware reads are
	/// meaningful: same padding shape as [`Avx512Bf16Vl::cvtneps_pbh_u16x4`],
	/// mirrored (there the *output* is padded and truncated; here it's the
	/// *input*).
	#[inline]
	pub fn cvtpbh_ps_f32x4(self, a: [u16; 4]) -> [f32; 4] {
		unsafe { cvtpbh_ps_f32x4_intrinsic(&a) }
	}

	/// [`Avx512Bf16Vl::dpbf16_ps_f32x8`] where `mask` bit is set, else copied
	/// from `src` (`vdpbf16ps`, 256-bit, merge-masked). `src` is a real input
	/// here, not just a merge fallback: same reasoning as
	/// [`super::avx512vnni::Avx512Vnni::dpbusd_i32x16`].
	#[inline]
	pub fn dpbf16_ps_f32x8_merge_masked(self, src: [f32; 8], mask: u8, a: [u16; 16], b: [u16; 16]) -> [f32; 8] {
		unsafe { mask_dpbf16_ps_f32x8_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::dpbf16_ps_f32x8`] where `mask` bit is set, else zero
	/// (`vdpbf16ps`, 256-bit, zero-masked).
	#[inline]
	pub fn dpbf16_ps_f32x8_zero_masked(self, mask: u8, src: [f32; 8], a: [u16; 16], b: [u16; 16]) -> [f32; 8] {
		unsafe { maskz_dpbf16_ps_f32x8_intrinsic(mask, &src, &a, &b) }
	}

	/// [`Avx512Bf16Vl::dpbf16_ps_f32x4`] where `mask` bit is set, else copied
	/// from `src` (`vdpbf16ps`, 128-bit, merge-masked).
	#[inline]
	pub fn dpbf16_ps_f32x4_merge_masked(self, src: [f32; 4], mask: u8, a: [u16; 8], b: [u16; 8]) -> [f32; 4] {
		unsafe { mask_dpbf16_ps_f32x4_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::dpbf16_ps_f32x4`] where `mask` bit is set, else zero
	/// (`vdpbf16ps`, 128-bit, zero-masked).
	#[inline]
	pub fn dpbf16_ps_f32x4_zero_masked(self, mask: u8, src: [f32; 4], a: [u16; 8], b: [u16; 8]) -> [f32; 4] {
		unsafe { maskz_dpbf16_ps_f32x4_intrinsic(mask, &src, &a, &b) }
	}

	/// [`Avx512Bf16Vl::cvtneps_pbh_u16x8`] where `mask` bit is set, else
	/// copied from `src` (`vcvtneps2bf16`, 256-bit source, merge-masked).
	#[inline]
	pub fn cvtneps_pbh_u16x8_merge_masked(self, src: [u16; 8], mask: u8, a: [f32; 8]) -> [u16; 8] {
		unsafe { mask_cvtneps_pbh_u16x8_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtneps_pbh_u16x8`] where `mask` bit is set, else zero
	/// (`vcvtneps2bf16`, 256-bit source, zero-masked).
	#[inline]
	pub fn cvtneps_pbh_u16x8_zero_masked(self, mask: u8, a: [f32; 8]) -> [u16; 8] {
		unsafe { maskz_cvtneps_pbh_u16x8_intrinsic(mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtneps_pbh_u16x4`] where `mask` bit is set, else
	/// copied from `src` (`vcvtneps2bf16`, 128-bit, merge-masked). Same
	/// 4-of-8-lane truncation as the unmasked op; only `mask`'s low 4 bits
	/// are meaningful.
	#[inline]
	pub fn cvtneps_pbh_u16x4_merge_masked(self, src: [u16; 4], mask: u8, a: [f32; 4]) -> [u16; 4] {
		unsafe { mask_cvtneps_pbh_u16x4_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtneps_pbh_u16x4`] where `mask` bit is set, else zero
	/// (`vcvtneps2bf16`, 128-bit, zero-masked).
	#[inline]
	pub fn cvtneps_pbh_u16x4_zero_masked(self, mask: u8, a: [f32; 4]) -> [u16; 4] {
		unsafe { maskz_cvtneps_pbh_u16x4_intrinsic(mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtne2ps_pbh_u16x16`] where `mask` bit is set, else
	/// copied from `src` (`vcvtne2ps2bf16`, 256-bit, merge-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x16_merge_masked(self, src: [u16; 16], mask: u16, a: [f32; 8], b: [f32; 8]) -> [u16; 16] {
		unsafe { mask_cvtne2ps_pbh_u16x16_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::cvtne2ps_pbh_u16x16`] where `mask` bit is set, else
	/// zero (`vcvtne2ps2bf16`, 256-bit, zero-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x16_zero_masked(self, mask: u16, a: [f32; 8], b: [f32; 8]) -> [u16; 16] {
		unsafe { maskz_cvtne2ps_pbh_u16x16_intrinsic(mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::cvtne2ps_pbh_u16x8`] where `mask` bit is set, else
	/// copied from `src` (`vcvtne2ps2bf16`, 128-bit, merge-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x8_merge_masked(self, src: [u16; 8], mask: u8, a: [f32; 4], b: [f32; 4]) -> [u16; 8] {
		unsafe { mask_cvtne2ps_pbh_u16x8_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::cvtne2ps_pbh_u16x8`] where `mask` bit is set, else zero
	/// (`vcvtne2ps2bf16`, 128-bit, zero-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x8_zero_masked(self, mask: u8, a: [f32; 4], b: [f32; 4]) -> [u16; 8] {
		unsafe { maskz_cvtne2ps_pbh_u16x8_intrinsic(mask, &a, &b) }
	}

	/// [`Avx512Bf16Vl::cvtpbh_ps_f32x8`] where `mask` bit is set, else copied
	/// from `src` (`vcvtpbh2ps`, 128-bit BF16 source, merge-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x8_merge_masked(self, src: [f32; 8], mask: u8, a: [u16; 8]) -> [f32; 8] {
		unsafe { mask_cvtpbh_ps_f32x8_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtpbh_ps_f32x8`] where `mask` bit is set, else zero
	/// (`vcvtpbh2ps`, 128-bit BF16 source, zero-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x8_zero_masked(self, mask: u8, a: [u16; 8]) -> [f32; 8] {
		unsafe { maskz_cvtpbh_ps_f32x8_intrinsic(mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtpbh_ps_f32x4`] where `mask` bit is set, else copied
	/// from `src` (`vcvtpbh2ps`, 128-bit, merge-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x4_merge_masked(self, src: [f32; 4], mask: u8, a: [u16; 4]) -> [f32; 4] {
		unsafe { mask_cvtpbh_ps_f32x4_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16Vl::cvtpbh_ps_f32x4`] where `mask` bit is set, else zero
	/// (`vcvtpbh2ps`, 128-bit, zero-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x4_zero_masked(self, mask: u8, a: [u16; 4]) -> [f32; 4] {
		unsafe { maskz_cvtpbh_ps_f32x4_intrinsic(mask, &a) }
	}
}

#[inline]
unsafe fn load_bf16x16(a: &[u16; 16]) -> __m256bh {
	unsafe {
		let v: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		core::mem::transmute::<__m256i, __m256bh>(v)
	}
}

#[inline]
unsafe fn store_bf16x16(v: __m256bh) -> [u16; 16] {
	unsafe { core::mem::transmute::<__m256bh, [u16; 16]>(v) }
}

#[inline]
unsafe fn load_bf16x8(a: &[u16; 8]) -> __m128bh {
	unsafe {
		let v: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		core::mem::transmute::<__m128i, __m128bh>(v)
	}
}

#[inline]
unsafe fn store_bf16x8(v: __m128bh) -> [u16; 8] {
	unsafe { core::mem::transmute::<__m128bh, [u16; 8]>(v) }
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn dpbf16_ps_f32x8_intrinsic(src: &[f32; 8], a: &[u16; 16], b: &[u16; 16]) -> [f32; 8] {
	unsafe {
		let vsrc: __m256 = _mm256_loadu_ps(src.as_ptr());
		let va = load_bf16x16(a);
		let vb = load_bf16x16(b);
		let vr = _mm256_dpbf16_ps(vsrc, va, vb);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn dpbf16_ps_f32x4_intrinsic(src: &[f32; 4], a: &[u16; 8], b: &[u16; 8]) -> [f32; 4] {
	unsafe {
		let vsrc: __m128 = _mm_loadu_ps(src.as_ptr());
		let va = load_bf16x8(a);
		let vb = load_bf16x8(b);
		let vr = _mm_dpbf16_ps(vsrc, va, vb);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtneps_pbh_u16x8_intrinsic(a: &[f32; 8]) -> [u16; 8] {
	unsafe {
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vr: __m128bh = _mm256_cvtneps_pbh(va);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtneps_pbh_u16x4_intrinsic(a: &[f32; 4]) -> [u16; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vr: __m128bh = _mm_cvtneps_pbh(va);
		let padded = store_bf16x8(vr);
		let mut out = [0u16; 4];
		out.copy_from_slice(&padded[..4]);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtne2ps_pbh_u16x16_intrinsic(a: &[f32; 8], b: &[f32; 8]) -> [u16; 16] {
	unsafe {
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vb: __m256 = _mm256_loadu_ps(b.as_ptr());
		let vr = _mm256_cvtne2ps_pbh(va, vb);
		store_bf16x16(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtne2ps_pbh_u16x8_intrinsic(a: &[f32; 4], b: &[f32; 4]) -> [u16; 8] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_cvtne2ps_pbh(va, vb);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtpbh_ps_f32x8_intrinsic(a: &[u16; 8]) -> [f32; 8] {
	unsafe {
		let va = load_bf16x8(a);
		let vr = _mm256_cvtpbh_ps(va);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn cvtpbh_ps_f32x4_intrinsic(a: &[u16; 4]) -> [f32; 4] {
	unsafe {
		let mut padded = [0u16; 8];
		padded[..4].copy_from_slice(a);
		let va = load_bf16x8(&padded);
		let vr = _mm_cvtpbh_ps(va);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

// Merge/zero-masked forms below, one pair per op above. `dpbf16_ps`'s zero
// form still takes `src` as a real input (not just a merge fallback): same
// reasoning as AVX512VNNI's `dpbusd`.

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_dpbf16_ps_f32x8_intrinsic(src: &[f32; 8], mask: u8, a: &[u16; 16], b: &[u16; 16]) -> [f32; 8] {
	unsafe {
		let vsrc: __m256 = _mm256_loadu_ps(src.as_ptr());
		let va = load_bf16x16(a);
		let vb = load_bf16x16(b);
		let vr = _mm256_mask_dpbf16_ps(vsrc, mask, va, vb);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_dpbf16_ps_f32x8_intrinsic(mask: u8, src: &[f32; 8], a: &[u16; 16], b: &[u16; 16]) -> [f32; 8] {
	unsafe {
		let vsrc: __m256 = _mm256_loadu_ps(src.as_ptr());
		let va = load_bf16x16(a);
		let vb = load_bf16x16(b);
		let vr = _mm256_maskz_dpbf16_ps(mask, vsrc, va, vb);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_dpbf16_ps_f32x4_intrinsic(src: &[f32; 4], mask: u8, a: &[u16; 8], b: &[u16; 8]) -> [f32; 4] {
	unsafe {
		let vsrc: __m128 = _mm_loadu_ps(src.as_ptr());
		let va = load_bf16x8(a);
		let vb = load_bf16x8(b);
		let vr = _mm_mask_dpbf16_ps(vsrc, mask, va, vb);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_dpbf16_ps_f32x4_intrinsic(mask: u8, src: &[f32; 4], a: &[u16; 8], b: &[u16; 8]) -> [f32; 4] {
	unsafe {
		let vsrc: __m128 = _mm_loadu_ps(src.as_ptr());
		let va = load_bf16x8(a);
		let vb = load_bf16x8(b);
		let vr = _mm_maskz_dpbf16_ps(mask, vsrc, va, vb);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtneps_pbh_u16x8_intrinsic(src: &[u16; 8], mask: u8, a: &[f32; 8]) -> [u16; 8] {
	unsafe {
		let vsrc = load_bf16x8(src);
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vr = _mm256_mask_cvtneps_pbh(vsrc, mask, va);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtneps_pbh_u16x8_intrinsic(mask: u8, a: &[f32; 8]) -> [u16; 8] {
	unsafe {
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vr = _mm256_maskz_cvtneps_pbh(mask, va);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`]. `src`/the
/// result's upper 4 lanes are padding: the instruction always produces a
/// full 8-lane register, only the low 4 are meaningful, same as the unmasked
/// `cvtneps_pbh_u16x4_intrinsic`.
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtneps_pbh_u16x4_intrinsic(src: &[u16; 4], mask: u8, a: &[f32; 4]) -> [u16; 4] {
	unsafe {
		let mut padded_src = [0u16; 8];
		padded_src[..4].copy_from_slice(src);
		let vsrc = load_bf16x8(&padded_src);
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vr = _mm_mask_cvtneps_pbh(vsrc, mask, va);
		let padded_out = store_bf16x8(vr);
		let mut out = [0u16; 4];
		out.copy_from_slice(&padded_out[..4]);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtneps_pbh_u16x4_intrinsic(mask: u8, a: &[f32; 4]) -> [u16; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vr = _mm_maskz_cvtneps_pbh(mask, va);
		let padded_out = store_bf16x8(vr);
		let mut out = [0u16; 4];
		out.copy_from_slice(&padded_out[..4]);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtne2ps_pbh_u16x16_intrinsic(src: &[u16; 16], mask: u16, a: &[f32; 8], b: &[f32; 8]) -> [u16; 16] {
	unsafe {
		let vsrc = load_bf16x16(src);
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vb: __m256 = _mm256_loadu_ps(b.as_ptr());
		let vr = _mm256_mask_cvtne2ps_pbh(vsrc, mask, va, vb);
		store_bf16x16(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtne2ps_pbh_u16x16_intrinsic(mask: u16, a: &[f32; 8], b: &[f32; 8]) -> [u16; 16] {
	unsafe {
		let va: __m256 = _mm256_loadu_ps(a.as_ptr());
		let vb: __m256 = _mm256_loadu_ps(b.as_ptr());
		let vr = _mm256_maskz_cvtne2ps_pbh(mask, va, vb);
		store_bf16x16(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtne2ps_pbh_u16x8_intrinsic(src: &[u16; 8], mask: u8, a: &[f32; 4], b: &[f32; 4]) -> [u16; 8] {
	unsafe {
		let vsrc = load_bf16x8(src);
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_mask_cvtne2ps_pbh(vsrc, mask, va, vb);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtne2ps_pbh_u16x8_intrinsic(mask: u8, a: &[f32; 4], b: &[f32; 4]) -> [u16; 8] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_maskz_cvtne2ps_pbh(mask, va, vb);
		store_bf16x8(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtpbh_ps_f32x8_intrinsic(src: &[f32; 8], mask: u8, a: &[u16; 8]) -> [f32; 8] {
	unsafe {
		let vsrc: __m256 = _mm256_loadu_ps(src.as_ptr());
		let va = load_bf16x8(a);
		let vr = _mm256_mask_cvtpbh_ps(vsrc, mask, va);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtpbh_ps_f32x8_intrinsic(mask: u8, a: &[u16; 8]) -> [f32; 8] {
	unsafe {
		let va = load_bf16x8(a);
		let vr = _mm256_maskz_cvtpbh_ps(mask, va);
		let mut out = [0.0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn mask_cvtpbh_ps_f32x4_intrinsic(src: &[f32; 4], mask: u8, a: &[u16; 4]) -> [f32; 4] {
	unsafe {
		let vsrc: __m128 = _mm_loadu_ps(src.as_ptr());
		let mut padded = [0u16; 8];
		padded[..4].copy_from_slice(a);
		let va = load_bf16x8(&padded);
		let vr = _mm_mask_cvtpbh_ps(vsrc, mask, va);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX512VL via [`Avx512Bf16Vl`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512vl")]
unsafe fn maskz_cvtpbh_ps_f32x4_intrinsic(mask: u8, a: &[u16; 4]) -> [f32; 4] {
	unsafe {
		let mut padded = [0u16; 8];
		padded[..4].copy_from_slice(a);
		let va = load_bf16x8(&padded);
		let vr = _mm_maskz_cvtpbh_ps(mask, va);
		let mut out = [0.0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512vl.rs"]
mod tests;
