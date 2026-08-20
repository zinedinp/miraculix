//! AVX-VNNI-INT16: all-int16 dot-product-accumulate into `i32`
//! (`avxvnniint16`). Token: [`AvxVnniInt16::detect`]. Same shape as
//! `super::avx_vnni` (group = 2).

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_dpwsud_epi32, _mm256_dpwsuds_epi32, _mm256_dpwuud_epi32, _mm256_dpwuuds_epi32,
	_mm256_dpwusd_epi32, _mm256_dpwusds_epi32, _mm256_loadu_si256, _mm256_storeu_si256, _mm_dpwsud_epi32,
	_mm_dpwsuds_epi32, _mm_dpwuud_epi32, _mm_dpwuuds_epi32, _mm_dpwusd_epi32, _mm_dpwusds_epi32, _mm_loadu_si128,
	_mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_vnni_dot;
use super::avx_vnni::{vnni_acc_saturating, vnni_acc_wrapping};

/// Proof token: AVX-VNNI-INT16 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct AvxVnniInt16(());

impl AvxVnniInt16 {
	/// `None` if the CPU (or the compile-time target) lacks AVX-VNNI-INT16.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::AvxVnniInt16).then_some(AvxVnniInt16(()))
	}
}

simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwsud_i32x4, slice_fn = dpwsud_i32_slice, intrinsic_fn = dpwsud_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = u16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwsud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwsud`, 128-bit, `i16`x`u16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwsud_i32x8, slice_fn = dpwsud_i32_slice_wide, intrinsic_fn = dpwsud_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = u16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwsud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwsud`, 256-bit, `i16`x`u16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwsuds_i32x4, slice_fn = dpwsuds_i32_slice, intrinsic_fn = dpwsuds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = u16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwsuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwsud_i32x4`] (`vpdpwsuds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwsud_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwsuds_i32x8, slice_fn = dpwsuds_i32_slice_wide, intrinsic_fn = dpwsuds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = u16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwsuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwsud_i32x8`] (`vpdpwsuds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwsud_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwusd_i32x4, slice_fn = dpwusd_i32_slice, intrinsic_fn = dpwusd_i32x4_intrinsic,
	width = 4, group = 2, a_elem = u16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwusd_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwusd`, 128-bit, `u16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwusd_i32x8, slice_fn = dpwusd_i32_slice_wide, intrinsic_fn = dpwusd_i32x8_intrinsic,
	width = 8, group = 2, a_elem = u16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwusd_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwusd`, 256-bit, `u16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwusds_i32x4, slice_fn = dpwusds_i32_slice, intrinsic_fn = dpwusds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = u16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwusds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwusd_i32x4`] (`vpdpwusds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwusd_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwusds_i32x8, slice_fn = dpwusds_i32_slice_wide, intrinsic_fn = dpwusds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = u16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwusds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwusd_i32x8`] (`vpdpwusds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwusd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwuud_i32x4, slice_fn = dpwuud_i32_slice, intrinsic_fn = dpwuud_i32x4_intrinsic,
	width = 4, group = 2, a_elem = u16, b_elem = u16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwuud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwuud`, 128-bit, `u16`x`u16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwuud_i32x8, slice_fn = dpwuud_i32_slice_wide, intrinsic_fn = dpwuud_i32x8_intrinsic,
	width = 8, group = 2, a_elem = u16, b_elem = u16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwuud_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwuud`, 256-bit, `u16`x`u16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwuuds_i32x4, slice_fn = dpwuuds_i32_slice, intrinsic_fn = dpwuuds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = u16, b_elem = u16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwuuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwuud_i32x4`] (`vpdpwuuds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwuud_i32_slice`]. 4-wide chunks, software scalar rem.",
}
simd_vnni_dot! {
	token = AvxVnniInt16, target_feature = "avxvnniint16",
	fixed_fn = dpwuuds_i32x8, slice_fn = dpwuuds_i32_slice_wide, intrinsic_fn = dpwuuds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = u16, b_elem = u16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwuuds_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnniInt16::dpwuud_i32x8`] (`vpdpwuuds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnniInt16::dpwuud_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx_vnni_int16.rs"]
mod tests;
