//! AVX512DQ: 512-bit DQ extension with `mullo_epi64`, f64/f32<->i64/u64 cvt, broadcast/extract, and `fpclass`.
//! Token: `Avx512Dq`. `mullo_epi64` may fall back to narrower software forms; other ops are fixed-width only.
//! 128/256-bit companions live in `avx512vl`. `fpclass` has a k1-gated form
//! instead of merge/zero (see comment above `fpclass_f64x8_gated`).

use core::arch::x86_64::{
	__m128, __m128d, __m128i, __m256, __m256i, __m512, __m512d, __m512i, _mm_loadu_pd, _mm_loadu_ps,
	_mm_fpclass_sd_mask, _mm_fpclass_ss_mask, _mm_mask_fpclass_sd_mask, _mm_mask_fpclass_ss_mask,
	_mm_loadu_si128, _mm_storeu_pd, _mm_storeu_si128, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_storeu_ps,
	_mm256_storeu_si256, _mm512_broadcast_f32x2, _mm512_broadcast_f64x2, _mm512_broadcast_i32x2,
	_mm512_broadcast_i64x2, _mm512_cvtepi64_pd, _mm512_cvtepu64_pd, _mm512_cvtpd_epi64, _mm512_cvtpd_epu64,
	_mm512_cvttpd_epi64, _mm512_cvttpd_epu64, _mm512_extractf32x8_ps, _mm512_extractf64x2_pd,
	_mm512_extracti32x8_epi32, _mm512_extracti64x2_epi64, _mm512_fpclass_pd_mask, _mm512_fpclass_ps_mask,
	_mm512_insertf32x8, _mm512_insertf64x2, _mm512_inserti32x8, _mm512_inserti64x2, _mm512_loadu_pd,
	_mm512_loadu_ps, _mm512_loadu_si512, _mm512_mask_broadcast_f32x2, _mm512_mask_broadcast_f64x2,
	_mm512_mask_broadcast_i32x2, _mm512_mask_broadcast_i64x2, _mm512_mask_cvtepi64_pd, _mm512_mask_cvtepu64_pd,
	_mm512_mask_cvtpd_epi64, _mm512_mask_cvtpd_epu64, _mm512_mask_cvttpd_epi64, _mm512_mask_cvttpd_epu64,
	_mm512_mask_extractf32x8_ps, _mm512_mask_extractf64x2_pd, _mm512_mask_extracti32x8_epi32,
	_mm512_mask_extracti64x2_epi64, _mm512_mask_fpclass_pd_mask, _mm512_mask_fpclass_ps_mask,
	_mm512_mask_insertf32x8, _mm512_mask_insertf64x2, _mm512_mask_inserti32x8, _mm512_mask_inserti64x2,
	_mm512_mask_mullo_epi64, _mm512_mask_range_pd, _mm512_mask_range_ps, _mm512_mask_reduce_pd,
	_mm512_mask_reduce_ps, _mm512_maskz_broadcast_f32x2, _mm512_maskz_broadcast_f64x2, _mm512_maskz_broadcast_i32x2,
	_mm512_maskz_broadcast_i64x2, _mm512_maskz_cvtepi64_pd, _mm512_maskz_cvtepu64_pd, _mm512_maskz_cvtpd_epi64,
	_mm512_maskz_cvtpd_epu64, _mm512_maskz_cvttpd_epi64, _mm512_maskz_cvttpd_epu64, _mm512_maskz_extractf32x8_ps,
	_mm512_maskz_extractf64x2_pd, _mm512_maskz_extracti32x8_epi32, _mm512_maskz_extracti64x2_epi64,
	_mm512_maskz_insertf32x8, _mm512_maskz_insertf64x2, _mm512_maskz_inserti32x8, _mm512_maskz_inserti64x2,
	_mm512_maskz_mullo_epi64, _mm512_maskz_range_pd, _mm512_maskz_range_ps, _mm512_maskz_reduce_pd,
	_mm512_maskz_reduce_ps, _mm512_mullo_epi64, _mm512_range_pd, _mm512_range_ps, _mm512_reduce_pd,
	_mm512_reduce_ps, _mm512_storeu_pd, _mm512_storeu_ps, _mm512_storeu_si512,
	// f32 (ps) <-> i64/u64.
	_mm512_cvtps_epi64, _mm512_cvttps_epi64, _mm512_cvtps_epu64, _mm512_cvttps_epu64, _mm512_cvtepi64_ps,
	_mm512_cvtepu64_ps, _mm512_mask_cvtps_epi64, _mm512_maskz_cvtps_epi64, _mm512_mask_cvttps_epi64,
	_mm512_maskz_cvttps_epi64, _mm512_mask_cvtps_epu64, _mm512_maskz_cvtps_epu64, _mm512_mask_cvttps_epu64,
	_mm512_maskz_cvttps_epu64, _mm512_mask_cvtepi64_ps, _mm512_maskz_cvtepi64_ps, _mm512_mask_cvtepu64_ps,
	_mm512_maskz_cvtepu64_ps,
	// Embedded-rounding (`_round_`) cvt, 512-bit only.
	_mm512_cvt_roundpd_epi64, _mm512_cvt_roundpd_epu64, _mm512_cvt_roundepi64_pd, _mm512_cvt_roundepu64_pd,
	_mm512_cvt_roundps_epi64, _mm512_cvt_roundps_epu64, _mm512_cvt_roundepi64_ps, _mm512_cvt_roundepu64_ps,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::macros::{
	simd_binop, simd_binop_imm_fixed, simd_binop_imm_masked, simd_binop_masked, simd_broadcast,
	simd_broadcast_masked, simd_cvt, simd_cvt_masked, simd_extract_imm, simd_extract_imm_masked, simd_insert_imm,
	simd_cvt_imm, simd_insert_imm_masked, simd_unop_imm, simd_unop_imm_mask, simd_unop_imm_mask_gated,
	simd_unop_imm_masked,
};

/// Proof token: AVX-512DQ available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Dq(());

impl Avx512Dq {
	/// `None` if the CPU (or the compile-time target) lacks AVX-512DQ.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From resolved tier (`V4` lists `Feature::Avx512dq`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx512dq).then_some(Avx512Dq(()))
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512dq).then_some(Avx512Dq(()))
	}
}

macro_rules! avx512dq_i64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Dq, vis = pub, target_feature = "avx512dq",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512dq_u64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Dq, vis = pub, target_feature = "avx512dq",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512dq_i64_binop!(
	mullo_i64x8, mullo_i64_slice, vpmullq, _mm512_mullo_epi64, |x: i64, y: i64| x.wrapping_mul(y),
	"`a * b` per lane, low 64 bits (`vpmullq`, 512-bit). No native single-instruction form below AVX-512DQ; `auto_up::mullo_i64`/`u64` fall back to a composed schoolbook cascade instead of scalar.",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide chunks, scalar remainder."
);
avx512dq_u64_binop!(
	mullo_u64x8, mullo_u64_slice, vpmullq_u, _mm512_mullo_epi64, |x: u64, y: u64| x.wrapping_mul(y),
	"`a * b` per lane, low 64 bits (`vpmullq`, 512-bit). No native single-instruction form below AVX-512DQ; `auto_up::mullo_i64`/`u64` fall back to a composed schoolbook cascade instead of scalar.",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide chunks, scalar remainder."
);

simd_binop_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = mullo_i64x8_merge_masked, zero_fn = mullo_i64x8_zero_masked,
	merge_intrinsic_fn = mask_mullo_i64x8_intrinsic, zero_intrinsic_fn = maskz_mullo_i64x8_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_mullo_epi64, zero_intrinsic = _mm512_maskz_mullo_epi64,
	merge_doc = "[`Avx512Dq::mullo_i64x8`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512Dq::mullo_i64x8`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = mullo_u64x8_merge_masked, zero_fn = mullo_u64x8_zero_masked,
	merge_intrinsic_fn = mask_mullo_u64x8_intrinsic, zero_intrinsic_fn = maskz_mullo_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_mullo_epi64, zero_intrinsic = _mm512_maskz_mullo_epi64,
	merge_doc = "[`Avx512Dq::mullo_u64x8`] where `mask` bit is set, else copied from `src` (`vpmullq`, merge-masked).",
	zero_doc = "[`Avx512Dq::mullo_u64x8`] where `mask` bit is set, else zero (`vpmullq`, zero-masked).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = i64_to_f64x8, intrinsic_fn = i64_to_f64x8_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_storeu = _mm512_storeu_pd,
	intrinsic = _mm512_cvtepi64_pd,
	fixed_doc = "Signed `i64` to `f64`, round-to-nearest-even (`vcvtqq2pd`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = u64_to_f64x8, intrinsic_fn = u64_to_f64x8_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_storeu = _mm512_storeu_pd,
	intrinsic = _mm512_cvtepu64_pd,
	fixed_doc = "Unsigned `u64` to `f64`, round-to-nearest-even (`vcvtuqq2pd`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_i64x8, intrinsic_fn = f64_to_i64x8_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvtpd_epi64,
	fixed_doc = "`f64` to `i64`, round-to-nearest-even. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2qq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_i64x8_trunc, intrinsic_fn = f64_to_i64x8_trunc_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvttpd_epi64,
	fixed_doc = "`f64` to `i64`, truncating toward zero. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2qq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_u64x8, intrinsic_fn = f64_to_u64x8_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvtpd_epu64,
	fixed_doc = "`f64` to `u64`, round-to-nearest-even. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtpd2uqq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_u64x8_trunc, intrinsic_fn = f64_to_u64x8_trunc_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvttpd_epu64,
	fixed_doc = "`f64` to `u64`, truncating toward zero. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttpd2uqq`, 512-bit).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = i64_to_f64x8_merge_masked, zero_fn = i64_to_f64x8_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f64x8_intrinsic, zero_intrinsic_fn = maskz_i64_to_f64x8_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_loadu = _mm512_loadu_pd, out_storeu = _mm512_storeu_pd, mask = u8,
	merge_intrinsic = _mm512_mask_cvtepi64_pd, zero_intrinsic = _mm512_maskz_cvtepi64_pd,
	merge_doc = "[`Avx512Dq::i64_to_f64x8`] where `mask` bit is set, else copied from `src` (`vcvtqq2pd`, merge-masked).",
	zero_doc = "[`Avx512Dq::i64_to_f64x8`] where `mask` bit is set, else zero (`vcvtqq2pd`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = u64_to_f64x8_merge_masked, zero_fn = u64_to_f64x8_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f64x8_intrinsic, zero_intrinsic_fn = maskz_u64_to_f64x8_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_loadu = _mm512_loadu_pd, out_storeu = _mm512_storeu_pd, mask = u8,
	merge_intrinsic = _mm512_mask_cvtepu64_pd, zero_intrinsic = _mm512_maskz_cvtepu64_pd,
	merge_doc = "[`Avx512Dq::u64_to_f64x8`] where `mask` bit is set, else copied from `src` (`vcvtuqq2pd`, merge-masked).",
	zero_doc = "[`Avx512Dq::u64_to_f64x8`] where `mask` bit is set, else zero (`vcvtuqq2pd`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f64_to_i64x8_merge_masked, zero_fn = f64_to_i64x8_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x8_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x8_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = i64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvtpd_epi64, zero_intrinsic = _mm512_maskz_cvtpd_epi64,
	merge_doc = "[`Avx512Dq::f64_to_i64x8`] where `mask` bit is set, else copied from `src` (`vcvtpd2qq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f64_to_i64x8`] where `mask` bit is set, else zero (`vcvtpd2qq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f64_to_i64x8_trunc_merge_masked, zero_fn = f64_to_i64x8_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_i64x8_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_i64x8_trunc_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = i64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvttpd_epi64, zero_intrinsic = _mm512_maskz_cvttpd_epi64,
	merge_doc = "[`Avx512Dq::f64_to_i64x8_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2qq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f64_to_i64x8_trunc`] where `mask` bit is set, else zero (`vcvttpd2qq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f64_to_u64x8_merge_masked, zero_fn = f64_to_u64x8_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x8_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x8_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = u64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvtpd_epu64, zero_intrinsic = _mm512_maskz_cvtpd_epu64,
	merge_doc = "[`Avx512Dq::f64_to_u64x8`] where `mask` bit is set, else copied from `src` (`vcvtpd2uqq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f64_to_u64x8`] where `mask` bit is set, else zero (`vcvtpd2uqq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f64_to_u64x8_trunc_merge_masked, zero_fn = f64_to_u64x8_trunc_zero_masked,
	merge_intrinsic_fn = mask_f64_to_u64x8_trunc_intrinsic, zero_intrinsic_fn = maskz_f64_to_u64x8_trunc_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = u64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvttpd_epu64, zero_intrinsic = _mm512_maskz_cvttpd_epu64,
	merge_doc = "[`Avx512Dq::f64_to_u64x8_trunc`] where `mask` bit is set, else copied from `src` (`vcvttpd2uqq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f64_to_u64x8_trunc`] where `mask` bit is set, else zero (`vcvttpd2uqq`, zero-masked).",
}

// f32 (ps) <-> i64/u64: same shape as the f64 (pd) forms above, 512-bit only
// here (128/256-bit companions, where the carrier register and the real
// lane count diverge, live in `avx512vl`).

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_i64x8, intrinsic_fn = f32_to_i64x8_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvtps_epi64,
	fixed_doc = "`f32` to `i64`, round-to-nearest-even. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtps2qq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_i64x8_trunc, intrinsic_fn = f32_to_i64x8_trunc_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvttps_epi64,
	fixed_doc = "`f32` to `i64`, truncating toward zero. Out-of-range or NaN inputs produce `i64::MIN` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttps2qq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_u64x8, intrinsic_fn = f32_to_u64x8_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvtps_epu64,
	fixed_doc = "`f32` to `u64`, round-to-nearest-even. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvtps2uqq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_u64x8_trunc, intrinsic_fn = f32_to_u64x8_trunc_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvttps_epu64,
	fixed_doc = "`f32` to `u64`, truncating toward zero. Out-of-range or NaN inputs produce `u64::MAX` (the HW \"integer indefinite\" value), *not* a saturating cast (`vcvttps2uqq`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = i64_to_f32x8, intrinsic_fn = i64_to_f32x8_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_storeu = _mm256_storeu_ps,
	intrinsic = _mm512_cvtepi64_ps,
	fixed_doc = "Signed `i64` to `f32`, round-to-nearest-even (`vcvtqq2ps`, 512-bit).",
}

simd_cvt! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = u64_to_f32x8, intrinsic_fn = u64_to_f32x8_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_storeu = _mm256_storeu_ps,
	intrinsic = _mm512_cvtepu64_ps,
	fixed_doc = "Unsigned `u64` to `f32`, round-to-nearest-even (`vcvtuqq2ps`, 512-bit).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f32_to_i64x8_merge_masked, zero_fn = f32_to_i64x8_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x8_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x8_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = i64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvtps_epi64, zero_intrinsic = _mm512_maskz_cvtps_epi64,
	merge_doc = "[`Avx512Dq::f32_to_i64x8`] where `mask` bit is set, else copied from `src` (`vcvtps2qq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f32_to_i64x8`] where `mask` bit is set, else zero (`vcvtps2qq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f32_to_i64x8_trunc_merge_masked, zero_fn = f32_to_i64x8_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_i64x8_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_i64x8_trunc_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = i64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvttps_epi64, zero_intrinsic = _mm512_maskz_cvttps_epi64,
	merge_doc = "[`Avx512Dq::f32_to_i64x8_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2qq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f32_to_i64x8_trunc`] where `mask` bit is set, else zero (`vcvttps2qq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f32_to_u64x8_merge_masked, zero_fn = f32_to_u64x8_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x8_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x8_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = u64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvtps_epu64, zero_intrinsic = _mm512_maskz_cvtps_epu64,
	merge_doc = "[`Avx512Dq::f32_to_u64x8`] where `mask` bit is set, else copied from `src` (`vcvtps2uqq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f32_to_u64x8`] where `mask` bit is set, else zero (`vcvtps2uqq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = f32_to_u64x8_trunc_merge_masked, zero_fn = f32_to_u64x8_trunc_zero_masked,
	merge_intrinsic_fn = mask_f32_to_u64x8_trunc_intrinsic, zero_intrinsic_fn = maskz_f32_to_u64x8_trunc_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = u64, out_vec = __m512i, out_loadu = _mm512_loadu_si512, out_storeu = _mm512_storeu_si512, mask = u8,
	merge_intrinsic = _mm512_mask_cvttps_epu64, zero_intrinsic = _mm512_maskz_cvttps_epu64,
	merge_doc = "[`Avx512Dq::f32_to_u64x8_trunc`] where `mask` bit is set, else copied from `src` (`vcvttps2uqq`, merge-masked).",
	zero_doc = "[`Avx512Dq::f32_to_u64x8_trunc`] where `mask` bit is set, else zero (`vcvttps2uqq`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = i64_to_f32x8_merge_masked, zero_fn = i64_to_f32x8_zero_masked,
	merge_intrinsic_fn = mask_i64_to_f32x8_intrinsic, zero_intrinsic_fn = maskz_i64_to_f32x8_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_loadu = _mm256_loadu_ps, out_storeu = _mm256_storeu_ps, mask = u8,
	merge_intrinsic = _mm512_mask_cvtepi64_ps, zero_intrinsic = _mm512_maskz_cvtepi64_ps,
	merge_doc = "[`Avx512Dq::i64_to_f32x8`] where `mask` bit is set, else copied from `src` (`vcvtqq2ps`, merge-masked).",
	zero_doc = "[`Avx512Dq::i64_to_f32x8`] where `mask` bit is set, else zero (`vcvtqq2ps`, zero-masked).",
}

simd_cvt_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = u64_to_f32x8_merge_masked, zero_fn = u64_to_f32x8_zero_masked,
	merge_intrinsic_fn = mask_u64_to_f32x8_intrinsic, zero_intrinsic_fn = maskz_u64_to_f32x8_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_loadu = _mm256_loadu_ps, out_storeu = _mm256_storeu_ps, mask = u8,
	merge_intrinsic = _mm512_mask_cvtepu64_ps, zero_intrinsic = _mm512_maskz_cvtepu64_ps,
	merge_doc = "[`Avx512Dq::u64_to_f32x8`] where `mask` bit is set, else copied from `src` (`vcvtuqq2ps`, merge-masked).",
	zero_doc = "[`Avx512Dq::u64_to_f32x8`] where `mask` bit is set, else zero (`vcvtuqq2ps`, zero-masked).",
}

// Embedded-rounding (`_round_`) cvt: SAE-only, so 512-bit exclusive, no
// 128/256-bit companions (the ISA has none). `IMM8` mirrors the crate's
// AVX512FP16 `_round_ph` convention (stdarch itself calls the parameter
// `ROUNDING`): `_MM_FROUND_TO_*` bitwise-OR'd with `_MM_FROUND_NO_EXC`, or
// `_MM_FROUND_CUR_DIRECTION`: stdarch's `static_assert_rounding!` rejects
// any other combination at compile time.

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_i64x8_round, intrinsic_fn = f64_to_i64x8_round_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundpd_epi64,
	fixed_doc = "`f64` to `i64` with explicit rounding control (`vcvtpd2qq`, 512-bit). See module docs for the `IMM8` encoding.",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f64_to_u64x8_round, intrinsic_fn = f64_to_u64x8_round_intrinsic,
	width = 8,
	in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundpd_epu64,
	fixed_doc = "`f64` to `u64` with explicit rounding control (`vcvtpd2uqq`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = i64_to_f64x8_round, intrinsic_fn = i64_to_f64x8_round_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_storeu = _mm512_storeu_pd,
	intrinsic = _mm512_cvt_roundepi64_pd,
	fixed_doc = "Signed `i64` to `f64` with explicit rounding control (`vcvtqq2pd`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = u64_to_f64x8_round, intrinsic_fn = u64_to_f64x8_round_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f64, out_vec = __m512d, out_storeu = _mm512_storeu_pd,
	intrinsic = _mm512_cvt_roundepu64_pd,
	fixed_doc = "Unsigned `u64` to `f64` with explicit rounding control (`vcvtuqq2pd`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_i64x8_round, intrinsic_fn = f32_to_i64x8_round_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = i64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundps_epi64,
	fixed_doc = "`f32` to `i64` with explicit rounding control (`vcvtps2qq`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = f32_to_u64x8_round, intrinsic_fn = f32_to_u64x8_round_intrinsic,
	width = 8,
	in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
	out_elem = u64, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
	intrinsic = _mm512_cvt_roundps_epu64,
	fixed_doc = "`f32` to `u64` with explicit rounding control (`vcvtps2uqq`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = i64_to_f32x8_round, intrinsic_fn = i64_to_f32x8_round_intrinsic,
	width = 8,
	in_elem = i64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_storeu = _mm256_storeu_ps,
	intrinsic = _mm512_cvt_roundepi64_ps,
	fixed_doc = "Signed `i64` to `f32` with explicit rounding control (`vcvtqq2ps`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_cvt_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = u64_to_f32x8_round, intrinsic_fn = u64_to_f32x8_round_intrinsic,
	width = 8,
	in_elem = u64, in_vec = __m512i, in_loadu = _mm512_loadu_si512,
	out_elem = f32, out_vec = __m256, out_storeu = _mm256_storeu_ps,
	intrinsic = _mm512_cvt_roundepu64_ps,
	fixed_doc = "Unsigned `u64` to `f32` with explicit rounding control (`vcvtuqq2ps`, 512-bit). Same `IMM8` as [`Avx512Dq::f64_to_i64x8_round`].",
}

simd_binop_imm_fixed! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = range_f64x8, intrinsic_fn = range_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	intrinsic = _mm512_range_pd,
	fixed_doc = "`a`/`b` combined per lane, selected by `IMM8`: bits[1:0] pick min/max/abs-min/abs-max, bits[3:2] pick the result's sign (from `a`/from the compare/cleared/set) (`vrangepd`, 512-bit). See the Intel SDM's `RANGE` pseudocode for the full 16-entry `IMM8` table.",
}

simd_binop_imm_fixed! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = range_f32x16, intrinsic_fn = range_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	intrinsic = _mm512_range_ps,
	fixed_doc = "`a`/`b` combined per lane, selected by `IMM8` - same encoding as [`Avx512Dq::range_f64x8`] (`vrangeps`, 512-bit).",
}

simd_binop_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = range_f64x8_merge_masked, zero_fn = range_f64x8_zero_masked,
	merge_intrinsic_fn = mask_range_f64x8_intrinsic, zero_intrinsic_fn = maskz_range_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_range_pd, zero_intrinsic = _mm512_maskz_range_pd,
	merge_doc = "[`Avx512Dq::range_f64x8`] where `mask` bit is set, else copied from `src` (`vrangepd`, merge-masked).",
	zero_doc = "[`Avx512Dq::range_f64x8`] where `mask` bit is set, else zero (`vrangepd`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = range_f32x16_merge_masked, zero_fn = range_f32x16_zero_masked,
	merge_intrinsic_fn = mask_range_f32x16_intrinsic, zero_intrinsic_fn = maskz_range_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_range_ps, zero_intrinsic = _mm512_maskz_range_ps,
	merge_doc = "[`Avx512Dq::range_f32x16`] where `mask` bit is set, else copied from `src` (`vrangeps`, merge-masked).",
	zero_doc = "[`Avx512Dq::range_f32x16`] where `mask` bit is set, else zero (`vrangeps`, zero-masked).",
}

simd_unop_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = reduce_f64x8, intrinsic_fn = reduce_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	intrinsic = _mm512_reduce_pd,
	fixed_doc = "`a - 2^-M * round(2^M * a, mode)` per lane: `IMM8` bits[7:4] give `M` (fraction bits kept), bits[3:0] give the rounding mode (same encoding as `roundpd`'s immediate) (`vreducepd`, 512-bit).",
}

simd_unop_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = reduce_f32x16, intrinsic_fn = reduce_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	intrinsic = _mm512_reduce_ps,
	fixed_doc = "`a - 2^-M * round(2^M * a, mode)` per lane - same `IMM8` encoding as [`Avx512Dq::reduce_f64x8`] (`vreduceps`, 512-bit).",
}

simd_unop_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = reduce_f64x8_merge_masked, zero_fn = reduce_f64x8_zero_masked,
	merge_intrinsic_fn = mask_reduce_f64x8_intrinsic, zero_intrinsic_fn = maskz_reduce_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_reduce_pd, zero_intrinsic = _mm512_maskz_reduce_pd,
	merge_doc = "[`Avx512Dq::reduce_f64x8`] where `mask` bit is set, else copied from `src` (`vreducepd`, merge-masked).",
	zero_doc = "[`Avx512Dq::reduce_f64x8`] where `mask` bit is set, else zero (`vreducepd`, zero-masked).",
}

simd_unop_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = reduce_f32x16_merge_masked, zero_fn = reduce_f32x16_zero_masked,
	merge_intrinsic_fn = mask_reduce_f32x16_intrinsic, zero_intrinsic_fn = maskz_reduce_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_reduce_ps, zero_intrinsic = _mm512_maskz_reduce_ps,
	merge_doc = "[`Avx512Dq::reduce_f32x16`] where `mask` bit is set, else copied from `src` (`vreduceps`, merge-masked).",
	zero_doc = "[`Avx512Dq::reduce_f32x16`] where `mask` bit is set, else zero (`vreduceps`, zero-masked).",
}

simd_unop_imm_mask! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_f64x8, intrinsic_fn = fpclass_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, intrinsic = _mm512_fpclass_pd_mask,
	fixed_doc = "Per-lane category test selected by `IMM8` bit mask - bit0 QNaN, bit1 +0, bit2 -0, bit3 +Inf, bit4 -Inf, bit5 denormal, bit6 negative finite, bit7 SNaN - one result bit per lane (`vfpclasspd`, 512-bit).",
}

simd_unop_imm_mask! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_f32x16, intrinsic_fn = fpclass_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, intrinsic = _mm512_fpclass_ps_mask,
	fixed_doc = "Per-lane category test, same `IMM8` bit encoding as [`Avx512Dq::fpclass_f64x8`] (`vfpclassps`, 512-bit).",
}

// `fpclass` gets one gated form, not a merge/zero pair: the output is already
// a mask, so there's nothing for a separate "zero" variant to do that plain
// `&` doesn't already do, and stdarch has no `_maskz_fpclass_*` intrinsic -
// same reasoning as `compressstoreu` having no `maskz` sibling in the
// previous mask-register batch.
simd_unop_imm_mask_gated! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_f64x8_gated, intrinsic_fn = mask_fpclass_f64x8_intrinsic,
	width = 8, elem = f64, vec = __m512d, mask = u8,
	loadu = _mm512_loadu_pd, intrinsic = _mm512_mask_fpclass_pd_mask,
	fixed_doc = "[`Avx512Dq::fpclass_f64x8`] ANDed with `k1` (`vfpclasspd`, mask-gated): `fpclass_f64x8::<IMM8>(a) & k1`.",
}

simd_unop_imm_mask_gated! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_f32x16_gated, intrinsic_fn = mask_fpclass_f32x16_intrinsic,
	width = 16, elem = f32, vec = __m512, mask = u16,
	loadu = _mm512_loadu_ps, intrinsic = _mm512_mask_fpclass_ps_mask,
	fixed_doc = "[`Avx512Dq::fpclass_f32x16`] ANDed with `k1` (`vfpclassps`, mask-gated): `fpclass_f32x16::<IMM8>(a) & k1`.",
}

// Scalar (lane-0-only) fpclass: same IMM8 encoding as the packed forms
// above, xmm carrier, DQ alone suffices (no VL needed: SDM only lists
// AVX512DQ for VFPCLASSSD/SS).

simd_unop_imm_mask! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_sd, intrinsic_fn = fpclass_sd_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, intrinsic = _mm_fpclass_sd_mask,
	fixed_doc = "Lane-0 category test, same `IMM8` bits as [`Avx512Dq::fpclass_f64x8`] (`vfpclasssd`, scalar).",
}

simd_unop_imm_mask! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_ss, intrinsic_fn = fpclass_ss_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, intrinsic = _mm_fpclass_ss_mask,
	fixed_doc = "Lane-0 category test, same `IMM8` bits as [`Avx512Dq::fpclass_f64x8`] (`vfpclassss`, scalar).",
}

simd_unop_imm_mask_gated! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_sd_gated, intrinsic_fn = mask_fpclass_sd_intrinsic,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, intrinsic = _mm_mask_fpclass_sd_mask,
	fixed_doc = "[`Avx512Dq::fpclass_sd`] ANDed with `k1` (`vfpclasssd`, mask-gated).",
}

simd_unop_imm_mask_gated! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = fpclass_ss_gated, intrinsic_fn = mask_fpclass_ss_intrinsic,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, intrinsic = _mm_mask_fpclass_ss_mask,
	fixed_doc = "[`Avx512Dq::fpclass_ss`] ANDed with `k1` (`vfpclassss`, mask-gated).",
}

simd_broadcast! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = broadcast_f32x2_to_x16, intrinsic_fn = broadcast_f32x2_to_x16_intrinsic,
	narrow_width = 4, wide_width = 16, elem = f32, narrow_vec = __m128, wide_vec = __m512,
	narrow_loadu = _mm_loadu_ps, storeu = _mm512_storeu_ps, intrinsic = _mm512_broadcast_f32x2,
	fixed_doc = "Broadcasts `a`'s lower 2 `f32` lanes across all 16 output lanes; `a`'s upper 2 lanes (of its 4-lane `__m128` load) are ignored (`vbroadcastf32x2`, 512-bit).",
}

simd_broadcast! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = broadcast_i32x2_to_x16, intrinsic_fn = broadcast_i32x2_to_x16_intrinsic,
	narrow_width = 4, wide_width = 16, elem = i32, narrow_vec = __m128i, wide_vec = __m512i,
	narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512, intrinsic = _mm512_broadcast_i32x2,
	fixed_doc = "Broadcasts `a`'s lower 2 `i32` lanes across all 16 output lanes; `a`'s upper 2 lanes are ignored (`vbroadcasti32x2`, 512-bit).",
}

simd_broadcast! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = broadcast_f64x2_to_x8, intrinsic_fn = broadcast_f64x2_to_x8_intrinsic,
	narrow_width = 2, wide_width = 8, elem = f64, narrow_vec = __m128d, wide_vec = __m512d,
	narrow_loadu = _mm_loadu_pd, storeu = _mm512_storeu_pd, intrinsic = _mm512_broadcast_f64x2,
	fixed_doc = "Broadcasts `a`'s 2 `f64` lanes across all 8 output lanes (`vbroadcastf64x2`, 512-bit).",
}

simd_broadcast! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = broadcast_i64x2_to_x8, intrinsic_fn = broadcast_i64x2_to_x8_intrinsic,
	narrow_width = 2, wide_width = 8, elem = i64, narrow_vec = __m128i, wide_vec = __m512i,
	narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512, intrinsic = _mm512_broadcast_i64x2,
	fixed_doc = "Broadcasts `a`'s 2 `i64` lanes across all 8 output lanes (`vbroadcasti64x2`, 512-bit).",
}

simd_broadcast_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = broadcast_f32x2_to_x16_merge_masked, zero_fn = broadcast_f32x2_to_x16_zero_masked,
	merge_intrinsic_fn = mask_broadcast_f32x2_to_x16_intrinsic, zero_intrinsic_fn = maskz_broadcast_f32x2_to_x16_intrinsic,
	narrow_width = 4, wide_width = 16, elem = f32, narrow_vec = __m128, wide_vec = __m512, mask = u16,
	narrow_loadu = _mm_loadu_ps, wide_loadu = _mm512_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_broadcast_f32x2, zero_intrinsic = _mm512_maskz_broadcast_f32x2,
	merge_doc = "[`Avx512Dq::broadcast_f32x2_to_x16`] where `mask` bit is set, else copied from `src` (`vbroadcastf32x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::broadcast_f32x2_to_x16`] where `mask` bit is set, else zero (`vbroadcastf32x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = broadcast_i32x2_to_x16_merge_masked, zero_fn = broadcast_i32x2_to_x16_zero_masked,
	merge_intrinsic_fn = mask_broadcast_i32x2_to_x16_intrinsic, zero_intrinsic_fn = maskz_broadcast_i32x2_to_x16_intrinsic,
	narrow_width = 4, wide_width = 16, elem = i32, narrow_vec = __m128i, wide_vec = __m512i, mask = u16,
	narrow_loadu = _mm_loadu_si128, wide_loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_broadcast_i32x2, zero_intrinsic = _mm512_maskz_broadcast_i32x2,
	merge_doc = "[`Avx512Dq::broadcast_i32x2_to_x16`] where `mask` bit is set, else copied from `src` (`vbroadcasti32x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::broadcast_i32x2_to_x16`] where `mask` bit is set, else zero (`vbroadcasti32x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = broadcast_f64x2_to_x8_merge_masked, zero_fn = broadcast_f64x2_to_x8_zero_masked,
	merge_intrinsic_fn = mask_broadcast_f64x2_to_x8_intrinsic, zero_intrinsic_fn = maskz_broadcast_f64x2_to_x8_intrinsic,
	narrow_width = 2, wide_width = 8, elem = f64, narrow_vec = __m128d, wide_vec = __m512d, mask = u8,
	narrow_loadu = _mm_loadu_pd, wide_loadu = _mm512_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_broadcast_f64x2, zero_intrinsic = _mm512_maskz_broadcast_f64x2,
	merge_doc = "[`Avx512Dq::broadcast_f64x2_to_x8`] where `mask` bit is set, else copied from `src` (`vbroadcastf64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::broadcast_f64x2_to_x8`] where `mask` bit is set, else zero (`vbroadcastf64x2`, zero-masked).",
}

simd_broadcast_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = broadcast_i64x2_to_x8_merge_masked, zero_fn = broadcast_i64x2_to_x8_zero_masked,
	merge_intrinsic_fn = mask_broadcast_i64x2_to_x8_intrinsic, zero_intrinsic_fn = maskz_broadcast_i64x2_to_x8_intrinsic,
	narrow_width = 2, wide_width = 8, elem = i64, narrow_vec = __m128i, wide_vec = __m512i, mask = u8,
	narrow_loadu = _mm_loadu_si128, wide_loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_broadcast_i64x2, zero_intrinsic = _mm512_maskz_broadcast_i64x2,
	merge_doc = "[`Avx512Dq::broadcast_i64x2_to_x8`] where `mask` bit is set, else copied from `src` (`vbroadcasti64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::broadcast_i64x2_to_x8`] where `mask` bit is set, else zero (`vbroadcasti64x2`, zero-masked).",
}

simd_extract_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = extract_f32x8_from_x16, intrinsic_fn = extract_f32x8_from_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = f32, wide_vec = __m512, narrow_vec = __m256,
	wide_loadu = _mm512_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm512_extractf32x8_ps,
	fixed_doc = "Extracts the `IMM8 & 1`-selected 8-lane half of `a` (`vextractf32x8`, 512-bit source).",
}

simd_extract_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = extract_i32x8_from_x16, intrinsic_fn = extract_i32x8_from_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = i32, wide_vec = __m512i, narrow_vec = __m256i,
	wide_loadu = _mm512_loadu_si512, storeu = _mm256_storeu_si256, intrinsic = _mm512_extracti32x8_epi32,
	fixed_doc = "Extracts the `IMM8 & 1`-selected 8-lane half of `a` (`vextracti32x8`, 512-bit source).",
}

simd_extract_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = extract_f64x2_from_x8, intrinsic_fn = extract_f64x2_from_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = f64, wide_vec = __m512d, narrow_vec = __m128d,
	wide_loadu = _mm512_loadu_pd, storeu = _mm_storeu_pd, intrinsic = _mm512_extractf64x2_pd,
	fixed_doc = "Extracts the `IMM8 & 3`-selected 2-lane quarter of `a` (`vextractf64x2`, 512-bit source).",
}

simd_extract_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = extract_i64x2_from_x8, intrinsic_fn = extract_i64x2_from_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = i64, wide_vec = __m512i, narrow_vec = __m128i,
	wide_loadu = _mm512_loadu_si512, storeu = _mm_storeu_si128, intrinsic = _mm512_extracti64x2_epi64,
	fixed_doc = "Extracts the `IMM8 & 3`-selected 2-lane quarter of `a` (`vextracti64x2`, 512-bit source).",
}

simd_extract_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = extract_f32x8_from_x16_merge_masked, zero_fn = extract_f32x8_from_x16_zero_masked,
	merge_intrinsic_fn = mask_extract_f32x8_from_x16_intrinsic, zero_intrinsic_fn = maskz_extract_f32x8_from_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = f32, wide_vec = __m512, narrow_vec = __m256, mask = u8,
	wide_loadu = _mm512_loadu_ps, narrow_loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	merge_intrinsic = _mm512_mask_extractf32x8_ps, zero_intrinsic = _mm512_maskz_extractf32x8_ps,
	merge_doc = "[`Avx512Dq::extract_f32x8_from_x16`] where `mask` bit is set, else copied from `src` (`vextractf32x8`, merge-masked).",
	zero_doc = "[`Avx512Dq::extract_f32x8_from_x16`] where `mask` bit is set, else zero (`vextractf32x8`, zero-masked).",
}

simd_extract_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = extract_i32x8_from_x16_merge_masked, zero_fn = extract_i32x8_from_x16_zero_masked,
	merge_intrinsic_fn = mask_extract_i32x8_from_x16_intrinsic, zero_intrinsic_fn = maskz_extract_i32x8_from_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = i32, wide_vec = __m512i, narrow_vec = __m256i, mask = u8,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	merge_intrinsic = _mm512_mask_extracti32x8_epi32, zero_intrinsic = _mm512_maskz_extracti32x8_epi32,
	merge_doc = "[`Avx512Dq::extract_i32x8_from_x16`] where `mask` bit is set, else copied from `src` (`vextracti32x8`, merge-masked).",
	zero_doc = "[`Avx512Dq::extract_i32x8_from_x16`] where `mask` bit is set, else zero (`vextracti32x8`, zero-masked).",
}

simd_extract_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = extract_f64x2_from_x8_merge_masked, zero_fn = extract_f64x2_from_x8_zero_masked,
	merge_intrinsic_fn = mask_extract_f64x2_from_x8_intrinsic, zero_intrinsic_fn = maskz_extract_f64x2_from_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = f64, wide_vec = __m512d, narrow_vec = __m128d, mask = u8,
	wide_loadu = _mm512_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	merge_intrinsic = _mm512_mask_extractf64x2_pd, zero_intrinsic = _mm512_maskz_extractf64x2_pd,
	merge_doc = "[`Avx512Dq::extract_f64x2_from_x8`] where `mask` bit is set, else copied from `src` (`vextractf64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::extract_f64x2_from_x8`] where `mask` bit is set, else zero (`vextractf64x2`, zero-masked).",
}

simd_extract_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = extract_i64x2_from_x8_merge_masked, zero_fn = extract_i64x2_from_x8_zero_masked,
	merge_intrinsic_fn = mask_extract_i64x2_from_x8_intrinsic, zero_intrinsic_fn = maskz_extract_i64x2_from_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = i64, wide_vec = __m512i, narrow_vec = __m128i, mask = u8,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	merge_intrinsic = _mm512_mask_extracti64x2_epi64, zero_intrinsic = _mm512_maskz_extracti64x2_epi64,
	merge_doc = "[`Avx512Dq::extract_i64x2_from_x8`] where `mask` bit is set, else copied from `src` (`vextracti64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::extract_i64x2_from_x8`] where `mask` bit is set, else zero (`vextracti64x2`, zero-masked).",
}

simd_insert_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = insert_f32x8_into_x16, intrinsic_fn = insert_f32x8_into_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = f32, wide_vec = __m512, narrow_vec = __m256,
	wide_loadu = _mm512_loadu_ps, narrow_loadu = _mm256_loadu_ps, storeu = _mm512_storeu_ps,
	intrinsic = _mm512_insertf32x8,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 1`-selected 8-lane half with `b` (`vinsertf32x8`, 512-bit).",
}

simd_insert_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = insert_i32x8_into_x16, intrinsic_fn = insert_i32x8_into_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = i32, wide_vec = __m512i, narrow_vec = __m256i,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm256_loadu_si256, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_inserti32x8,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 1`-selected 8-lane half with `b` (`vinserti32x8`, 512-bit).",
}

simd_insert_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = insert_f64x2_into_x8, intrinsic_fn = insert_f64x2_into_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = f64, wide_vec = __m512d, narrow_vec = __m128d,
	wide_loadu = _mm512_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm512_storeu_pd,
	intrinsic = _mm512_insertf64x2,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 3`-selected 2-lane quarter with `b` (`vinsertf64x2`, 512-bit).",
}

simd_insert_imm! {
	token = Avx512Dq, target_feature = "avx512dq",
	fixed_fn = insert_i64x2_into_x8, intrinsic_fn = insert_i64x2_into_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = i64, wide_vec = __m512i, narrow_vec = __m128i,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_inserti64x2,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 3`-selected 2-lane quarter with `b` (`vinserti64x2`, 512-bit).",
}

simd_insert_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = insert_f32x8_into_x16_merge_masked, zero_fn = insert_f32x8_into_x16_zero_masked,
	merge_intrinsic_fn = mask_insert_f32x8_into_x16_intrinsic, zero_intrinsic_fn = maskz_insert_f32x8_into_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = f32, wide_vec = __m512, narrow_vec = __m256, mask = u16,
	wide_loadu = _mm512_loadu_ps, narrow_loadu = _mm256_loadu_ps, storeu = _mm512_storeu_ps,
	merge_intrinsic = _mm512_mask_insertf32x8, zero_intrinsic = _mm512_maskz_insertf32x8,
	merge_doc = "[`Avx512Dq::insert_f32x8_into_x16`] where `mask` bit is set, else copied from `src` (`vinsertf32x8`, merge-masked).",
	zero_doc = "[`Avx512Dq::insert_f32x8_into_x16`] where `mask` bit is set, else zero (`vinsertf32x8`, zero-masked).",
}

simd_insert_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = insert_i32x8_into_x16_merge_masked, zero_fn = insert_i32x8_into_x16_zero_masked,
	merge_intrinsic_fn = mask_insert_i32x8_into_x16_intrinsic, zero_intrinsic_fn = maskz_insert_i32x8_into_x16_intrinsic,
	wide_width = 16, narrow_width = 8, elem = i32, wide_vec = __m512i, narrow_vec = __m256i, mask = u16,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm256_loadu_si256, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_inserti32x8, zero_intrinsic = _mm512_maskz_inserti32x8,
	merge_doc = "[`Avx512Dq::insert_i32x8_into_x16`] where `mask` bit is set, else copied from `src` (`vinserti32x8`, merge-masked).",
	zero_doc = "[`Avx512Dq::insert_i32x8_into_x16`] where `mask` bit is set, else zero (`vinserti32x8`, zero-masked).",
}

simd_insert_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = insert_f64x2_into_x8_merge_masked, zero_fn = insert_f64x2_into_x8_zero_masked,
	merge_intrinsic_fn = mask_insert_f64x2_into_x8_intrinsic, zero_intrinsic_fn = maskz_insert_f64x2_into_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = f64, wide_vec = __m512d, narrow_vec = __m128d, mask = u8,
	wide_loadu = _mm512_loadu_pd, narrow_loadu = _mm_loadu_pd, storeu = _mm512_storeu_pd,
	merge_intrinsic = _mm512_mask_insertf64x2, zero_intrinsic = _mm512_maskz_insertf64x2,
	merge_doc = "[`Avx512Dq::insert_f64x2_into_x8`] where `mask` bit is set, else copied from `src` (`vinsertf64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::insert_f64x2_into_x8`] where `mask` bit is set, else zero (`vinsertf64x2`, zero-masked).",
}

simd_insert_imm_masked! {
	token = Avx512Dq, target_feature = "avx512dq",
	merge_fn = insert_i64x2_into_x8_merge_masked, zero_fn = insert_i64x2_into_x8_zero_masked,
	merge_intrinsic_fn = mask_insert_i64x2_into_x8_intrinsic, zero_intrinsic_fn = maskz_insert_i64x2_into_x8_intrinsic,
	wide_width = 8, narrow_width = 2, elem = i64, wide_vec = __m512i, narrow_vec = __m128i, mask = u8,
	wide_loadu = _mm512_loadu_si512, narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_inserti64x2, zero_intrinsic = _mm512_maskz_inserti64x2,
	merge_doc = "[`Avx512Dq::insert_i64x2_into_x8`] where `mask` bit is set, else copied from `src` (`vinserti64x2`, merge-masked).",
	zero_doc = "[`Avx512Dq::insert_i64x2_into_x8`] where `mask` bit is set, else zero (`vinserti64x2`, zero-masked).",
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512dq.rs"]
mod tests;
