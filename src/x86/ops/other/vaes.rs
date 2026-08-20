//! VAES (Ice Lake, 2019): AES-NI widened to 256/512-bit, N independent 128-bit AESENC lanes.
//! Hand-written (no elementwise scalar). Tokens: [`Vaes`] (`"vaes"`), [`Vaes512`] (`"vaes,avx512f"`).
//! No 128-bit form (use [`super::aes::Aes`]).

use core::arch::x86_64::{
	__m256i, __m512i, _mm256_aesdec_epi128, _mm256_aesdeclast_epi128, _mm256_aesenc_epi128, _mm256_aesenclast_epi128,
	_mm256_loadu_si256, _mm256_storeu_si256, _mm512_aesdec_epi128, _mm512_aesdeclast_epi128, _mm512_aesenc_epi128,
	_mm512_aesenclast_epi128, _mm512_loadu_si512, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: VAES available (256-bit ops). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Vaes(());

impl Vaes {
	/// `None` if the CPU (or the compile-time target) lacks VAES.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Vaes).then_some(Vaes(()))
	}
}

/// Proof token: VAES *and* AVX-512F, both required for the 512-bit forms
/// (see `gfni.rs`'s [`super::gfni::Gfni512`] for why). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Vaes512(());

impl Vaes512 {
	/// `None` unless the CPU has both VAES and AVX-512F.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Vaes) && set.contains(Feature::Avx512f)).then_some(Vaes512(()))
	}
}

macro_rules! vaes_op {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, vec = $Vec:ty, loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [u8; $width], round_key: [u8; $width]) -> [u8; $width] {
				unsafe { $intrinsic_fn(&a, &round_key) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[u8; $width], round_key: &[u8; $width]) -> [u8; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vk: $Vec = $loadu(round_key.as_ptr().cast());
				let vr = $intrinsic(va, vk);
				let mut out = [0u8; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

vaes_op! {
	token = Vaes, target_feature = "vaes",
	fixed_fn = aesenc_u8x32, intrinsic_fn = aesenc_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_aesenc_epi128,
	fixed_doc = "Two independent AES encrypt rounds, one per 128-bit lane (`vaesenc`, 256-bit).",
}
vaes_op! {
	token = Vaes, target_feature = "vaes",
	fixed_fn = aesenclast_u8x32, intrinsic_fn = aesenclast_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_aesenclast_epi128,
	fixed_doc = "Two independent final AES encrypt rounds, one per 128-bit lane (`vaesenclast`, 256-bit).",
}
vaes_op! {
	token = Vaes, target_feature = "vaes",
	fixed_fn = aesdec_u8x32, intrinsic_fn = aesdec_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_aesdec_epi128,
	fixed_doc = "Two independent AES decrypt rounds, one per 128-bit lane (`vaesdec`, 256-bit).",
}
vaes_op! {
	token = Vaes, target_feature = "vaes",
	fixed_fn = aesdeclast_u8x32, intrinsic_fn = aesdeclast_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_aesdeclast_epi128,
	fixed_doc = "Two independent final AES decrypt rounds, one per 128-bit lane (`vaesdeclast`, 256-bit).",
}
vaes_op! {
	token = Vaes512, target_feature = "vaes,avx512f",
	fixed_fn = aesenc_u8x64, intrinsic_fn = aesenc_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_aesenc_epi128,
	fixed_doc = "Four independent AES encrypt rounds, one per 128-bit lane (`vaesenc`, 512-bit).",
}
vaes_op! {
	token = Vaes512, target_feature = "vaes,avx512f",
	fixed_fn = aesenclast_u8x64, intrinsic_fn = aesenclast_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_aesenclast_epi128,
	fixed_doc = "Four independent final AES encrypt rounds, one per 128-bit lane (`vaesenclast`, 512-bit).",
}
vaes_op! {
	token = Vaes512, target_feature = "vaes,avx512f",
	fixed_fn = aesdec_u8x64, intrinsic_fn = aesdec_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_aesdec_epi128,
	fixed_doc = "Four independent AES decrypt rounds, one per 128-bit lane (`vaesdec`, 512-bit).",
}
vaes_op! {
	token = Vaes512, target_feature = "vaes,avx512f",
	fixed_fn = aesdeclast_u8x64, intrinsic_fn = aesdeclast_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_aesdeclast_epi128,
	fixed_doc = "Four independent final AES decrypt rounds, one per 128-bit lane (`vaesdeclast`, 512-bit).",
}

#[cfg(test)]
#[path = "../../test/ops/other/vaes.rs"]
mod tests;
