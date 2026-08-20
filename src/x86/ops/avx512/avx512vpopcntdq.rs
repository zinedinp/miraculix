//! AVX512VPOPCNTDQ: 512-bit per-lane popcnt on `u32`/`u64` (`avx512vpopcntdq`).
//! Straight `simd_unop` plus masked forms. Not in V4 and no `from_level`.
//! Top rung for `auto_up::popcnt_u32`/`popcnt_u64`. 128/256-bit VL forms live in `super::avx512vl`.

use core::arch::x86_64::{
	__m512i, _mm512_loadu_si512, _mm512_mask_popcnt_epi32, _mm512_mask_popcnt_epi64, _mm512_maskz_popcnt_epi32,
	_mm512_maskz_popcnt_epi64, _mm512_popcnt_epi32, _mm512_popcnt_epi64, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{simd_unop, simd_unop_masked};

/// Proof token: AVX512VPOPCNTDQ, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Vpopcntdq(());

impl Avx512Vpopcntdq {
	/// `None` if the CPU (or the compile-time target) lacks AVX512VPOPCNTDQ.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512vpopcntdq).then_some(Avx512Vpopcntdq(()))
	}
}

simd_unop! {
	token = Avx512Vpopcntdq, target_feature = "avx512vpopcntdq",
	fixed_fn = popcnt_u32x16, slice_fn = popcnt_u32_slice, intrinsic_fn = popcnt_u32x16_intrinsic,
	width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_popcnt_epi32, scalar = u32::count_ones,
	fixed_doc = "Per-lane population count (`vpopcntd`, 512-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 16-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512Vpopcntdq, target_feature = "avx512vpopcntdq",
	fixed_fn = popcnt_u64x8, slice_fn = popcnt_u64_slice, intrinsic_fn = popcnt_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_popcnt_epi64, scalar = |x: u64| x.count_ones() as u64,
	fixed_doc = "Per-lane population count (`vpopcntq`, 512-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 8-wide chunks, scalar remainder.",
}

simd_unop_masked! {
	token = Avx512Vpopcntdq, target_feature = "avx512vpopcntdq",
	merge_fn = popcnt_u32x16_merge_masked, zero_fn = popcnt_u32x16_zero_masked,
	merge_intrinsic_fn = mask_popcnt_epi32_intrinsic, zero_intrinsic_fn = maskz_popcnt_epi32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_popcnt_epi32, zero_intrinsic = _mm512_maskz_popcnt_epi32,
	merge_doc = "Per-lane population count where `mask` bit is set, else copied from `src` (`vpopcntd`, merge-masked).",
	zero_doc = "Per-lane population count where `mask` bit is set, else zero (`vpopcntd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vpopcntdq, target_feature = "avx512vpopcntdq",
	merge_fn = popcnt_u64x8_merge_masked, zero_fn = popcnt_u64x8_zero_masked,
	merge_intrinsic_fn = mask_popcnt_epi64_intrinsic, zero_intrinsic_fn = maskz_popcnt_epi64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_popcnt_epi64, zero_intrinsic = _mm512_maskz_popcnt_epi64,
	merge_doc = "Per-lane population count where `mask` bit is set, else copied from `src` (`vpopcntq`, merge-masked).",
	zero_doc = "Per-lane population count where `mask` bit is set, else zero (`vpopcntq`, zero-masked).",
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512vpopcntdq.rs"]
mod tests;
