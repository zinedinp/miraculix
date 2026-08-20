//! SM4 (~2024 Sierra Forest/Granite Rapids): GB/T 32907-2016, 4-rounds key expand + encrypt (`"sm4"`).
//! Token: [`Sm4::detect`] (no auto). Hand-written; 8 calls = full schedule/encrypt; 256-bit = 2x128 lanes.

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_loadu_si256, _mm256_sm4key4_epi32, _mm256_sm4rnds4_epi32, _mm256_storeu_si256,
	_mm_loadu_si128, _mm_sm4key4_epi32, _mm_sm4rnds4_epi32, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SM4 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sm4(());

impl Sm4 {
	/// `None` if the CPU (or the compile-time target) lacks SM4.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sm4).then_some(Sm4(()))
	}
}

/// Slide-4 F: `dst[i]=f(...dst[i-1]...,rk[i])`, `f=x0^l(sbox(x1^x2^x3^rk))`.
macro_rules! sm4_slide4 {
	(
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, vec = $Vec:ty, loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl Sm4 {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u32; $width], round_key: [u32; $width]) -> [u32; $width] {
				unsafe { $intrinsic_fn(&a, &round_key) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "sm4,avx")]
		unsafe fn $intrinsic_fn(a: &[u32; $width], round_key: &[u32; $width]) -> [u32; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(round_key.as_ptr().cast());
				let vr = $intrinsic(va, vb);
				let mut out = [0u32; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

sm4_slide4! {
	fixed_fn = sm4key4_u32x4, intrinsic_fn = sm4key4_u32x4_intrinsic,
	width = 4, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128, intrinsic = _mm_sm4key4_epi32,
	fixed_doc = "4 rounds of SM4 key expansion (`vsm4key4`, 128-bit).",
}
sm4_slide4! {
	fixed_fn = sm4key4_u32x8, intrinsic_fn = sm4key4_u32x8_intrinsic,
	width = 8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_sm4key4_epi32,
	fixed_doc = "4 rounds of SM4 key expansion, 2 independent 128-bit lanes (`vsm4key4`, 256-bit).",
}
sm4_slide4! {
	fixed_fn = sm4rnds4_u32x4, intrinsic_fn = sm4rnds4_u32x4_intrinsic,
	width = 4, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128, intrinsic = _mm_sm4rnds4_epi32,
	fixed_doc = "4 rounds of SM4 encryption (`vsm4rnds4`, 128-bit).",
}
sm4_slide4! {
	fixed_fn = sm4rnds4_u32x8, intrinsic_fn = sm4rnds4_u32x8_intrinsic,
	width = 8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_sm4rnds4_epi32,
	fixed_doc = "4 rounds of SM4 encryption, 2 independent 128-bit lanes (`vsm4rnds4`, 256-bit).",
}

#[cfg(test)]
#[path = "../../test/ops/avx/sm4.rs"]
mod tests;
