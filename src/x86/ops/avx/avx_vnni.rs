//! AVX-VNNI: VEX narrow dot-product-accumulate into `i32` (`avxvnni`).
//! Token: [`AvxVnni::detect`]. Hand-written cross-type implementations that cascade to scalar when higher rungs are absent.

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_dpbusd_avx_epi32, _mm256_dpbusds_avx_epi32, _mm256_dpwssd_avx_epi32,
	_mm256_dpwssds_avx_epi32, _mm256_loadu_si256, _mm256_storeu_si256, _mm_dpbusd_avx_epi32, _mm_dpbusds_avx_epi32,
	_mm_dpwssd_avx_epi32, _mm_dpwssds_avx_epi32, _mm_loadu_si128, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_vnni_dot;

/// Proof token: AVX-VNNI available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct AvxVnni(());

impl AvxVnni {
	/// `None` if the CPU (or the compile-time target) lacks AVX-VNNI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::AvxVnni).then_some(AvxVnni(()))
	}
}

/// Scalar rem: `src + sum(products)` with 32-bit wrap (non-`s` VNNI).
/// Product sum is `i64` so extreme `i16`/`u16` pairs cannot overflow mid-sum.
#[inline]
pub(crate) fn vnni_acc_wrapping(src: i32, product_sum: i64) -> i32 {
	(src as i64).wrapping_add(product_sum) as i32
}

/// Scalar rem: signed sat of `src + sum(products)` (`s`-suffixed VNNI).
#[inline]
pub(crate) fn vnni_acc_saturating(src: i32, product_sum: i64) -> i32 {
	let t = (src as i64).saturating_add(product_sum);
	t.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpbusd_i32x4, slice_fn = dpbusd_i32_slice, intrinsic_fn = dpbusd_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbusd_avx_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbusd`, 128-bit, `u8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpbusd_i32x8, slice_fn = dpbusd_i32_slice_wide, intrinsic_fn = dpbusd_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbusd_avx_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbusd`, 256-bit, `u8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpbusds_i32x4, slice_fn = dpbusds_i32_slice, intrinsic_fn = dpbusds_i32x4_intrinsic,
	width = 4, group = 4, a_elem = u8, b_elem = i8,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpbusds_avx_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnni::dpbusd_i32x4`] (`vpdpbusds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnni::dpbusd_i32_slice`]. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpbusds_i32x8, slice_fn = dpbusds_i32_slice_wide, intrinsic_fn = dpbusds_i32x8_intrinsic,
	width = 8, group = 4, a_elem = u8, b_elem = i8,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpbusds_avx_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnni::dpbusd_i32x8`] (`vpdpbusds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnni::dpbusd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpwssd_i32x4, slice_fn = dpwssd_i32_slice, intrinsic_fn = dpwssd_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwssd_avx_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwssd`, 128-bit, `i16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpwssd_i32x8, slice_fn = dpwssd_i32_slice_wide, intrinsic_fn = dpwssd_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwssd_avx_epi32,
	acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwssd`, 256-bit, `i16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 8-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpwssds_i32x4, slice_fn = dpwssds_i32_slice, intrinsic_fn = dpwssds_i32x4_intrinsic,
	width = 4, group = 2, a_elem = i16, b_elem = i16,
	vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_dpwssds_avx_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnni::dpwssd_i32x4`] (`vpdpwssds`, 128-bit).",
	slice_doc = "Saturating [`AvxVnni::dpwssd_i32_slice`]. 4-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = AvxVnni, target_feature = "avxvnni",
	fixed_fn = dpwssds_i32x8, slice_fn = dpwssds_i32_slice_wide, intrinsic_fn = dpwssds_i32x8_intrinsic,
	width = 8, group = 2, a_elem = i16, b_elem = i16,
	vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_dpwssds_avx_epi32,
	acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`AvxVnni::dpwssd_i32x8`] (`vpdpwssds`, 256-bit).",
	slice_doc = "Saturating [`AvxVnni::dpwssd_i32_slice_wide`]. 8-wide chunks, software scalar rem.",
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx_vnni.rs"]
mod tests;
