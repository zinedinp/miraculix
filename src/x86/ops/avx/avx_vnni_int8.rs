//! AVX-VNNI-INT8: all-int8 dot-product-accumulate into `i32` (`avxvnniint8`).
//! Token: [`AvxVnniInt8::detect`]. Same shape as `super::avx_vnni` (group = 4).

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_dpbssd_epi32, _mm256_dpbssds_epi32, _mm256_dpbsud_epi32, _mm256_dpbsuds_epi32,
	_mm256_dpbuud_epi32, _mm256_dpbuuds_epi32, _mm256_loadu_si256, _mm256_storeu_si256, _mm_dpbssd_epi32,
	_mm_dpbssds_epi32, _mm_dpbsud_epi32, _mm_dpbsuds_epi32, _mm_dpbuud_epi32, _mm_dpbuuds_epi32, _mm_loadu_si128,
	_mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_vnni_dot;
use super::avx_vnni::{vnni_acc_saturating, vnni_acc_wrapping};

/// Proof token: AVX-VNNI-INT8 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct AvxVnniInt8(());

impl AvxVnniInt8 {
	/// `None` if the CPU (or the compile-time target) lacks AVX-VNNI-INT8.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::AvxVnniInt8).then_some(AvxVnniInt8(()))
	}
}

simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbssd_i32x4, slice_fn = dpbssd_i32_slice, intrinsic_fn = dpbssd_i32x4_intrinsic,
	width = 4, group = 4, a_elem = i8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbssd_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbssd`, 128-bit, `i8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbssd_i32x8, slice_fn = dpbssd_i32_slice_wide, intrinsic_fn = dpbssd_i32x8_intrinsic,
	width = 8, group = 4, a_elem = i8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbssd_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbssd`, 256-bit, `i8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbssds_i32x4, slice_fn = dpbssds_i32_slice, intrinsic_fn = dpbssds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = i8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbssds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbssd_i32x4`] (`vpdpbssds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbssd_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbssds_i32x8, slice_fn = dpbssds_i32_slice_wide, intrinsic_fn = dpbssds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = i8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbssds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbssd_i32x8`] (`vpdpbssds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbssd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbsud_i32x4, slice_fn = dpbsud_i32_slice, intrinsic_fn = dpbsud_i32x4_intrinsic,
	width = 4, group = 4, a_elem = i8, b_elem = u8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbsud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbsud`, 128-bit, `i8`x`u8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbsud_i32x8, slice_fn = dpbsud_i32_slice_wide, intrinsic_fn = dpbsud_i32x8_intrinsic,
	width = 8, group = 4, a_elem = i8, b_elem = u8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbsud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbsud`, 256-bit, `i8`x`u8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbsuds_i32x4, slice_fn = dpbsuds_i32_slice, intrinsic_fn = dpbsuds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = i8, b_elem = u8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbsuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbsud_i32x4`] (`vpdpbsuds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbsud_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbsuds_i32x8, slice_fn = dpbsuds_i32_slice_wide, intrinsic_fn = dpbsuds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = i8, b_elem = u8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbsuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbsud_i32x8`] (`vpdpbsuds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbsud_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbuud_i32x4, slice_fn = dpbuud_i32_slice, intrinsic_fn = dpbuud_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = u8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbuud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbuud`, 128-bit, `u8`x`u8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbuud_i32x8, slice_fn = dpbuud_i32_slice_wide, intrinsic_fn = dpbuud_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = u8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbuud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbuud`, 256-bit, `u8`x`u8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbuuds_i32x4, slice_fn = dpbuuds_i32_slice, intrinsic_fn = dpbuuds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = u8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbuuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbuud_i32x4`] (`vpdpbuuds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbuud_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt8, target_feature = "avxvnniint8",
	fixed_fn = dpbuuds_i32x8, slice_fn = dpbuuds_i32_slice_wide, intrinsic_fn = dpbuuds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = u8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbuuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt8::dpbuud_i32x8`] (`vpdpbuuds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt8::dpbuud_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx_vnni_int8.rs"]
mod tests;
