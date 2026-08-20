//! VPCLMULQDQ (Ice Lake, 2019): PCLMULQDQ widened to 256/512-bit, N independent 128-bit lanes.
//! Hand-written (const-generic imm). Tokens: [`Vpclmulqdq`], [`Vpclmulqdq512`] (needs `avx512f`).
//! No 128-bit form (use [`super::pclmulqdq::Pclmulqdq`]).

use core::arch::x86_64::{
	__m256i, __m512i, _mm256_clmulepi64_epi128, _mm256_loadu_si256, _mm256_storeu_si256, _mm512_clmulepi64_epi128,
	_mm512_loadu_si512, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: VPCLMULQDQ available (256-bit ops). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Vpclmulqdq(());

impl Vpclmulqdq {
	/// `None` if the CPU (or the compile-time target) lacks VPCLMULQDQ.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Vpclmulqdq).then_some(Vpclmulqdq(()))
	}

	/// Two independent carry-less multiplies, one per 128-bit lane (same
	/// half-selection as [`super::pclmulqdq::Pclmulqdq::clmul`]) (`vpclmulqdq`,
	/// 256-bit).
	#[inline]
	pub fn clmul_u64x4<const IMM8: i32>(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		unsafe { clmul_u64x4_intrinsic::<IMM8>(&a, &b) }
	}
}

/// Proof token: VPCLMULQDQ *and* AVX-512F, both required for the 512-bit
/// form (see `gfni.rs`'s [`super::gfni::Gfni512`] for why). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Vpclmulqdq512(());

impl Vpclmulqdq512 {
	/// `None` unless the CPU has both VPCLMULQDQ and AVX-512F.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Vpclmulqdq) && set.contains(Feature::Avx512f)).then_some(Vpclmulqdq512(()))
	}

	/// Four independent carry-less multiplies, one per 128-bit lane
	/// (`vpclmulqdq`, 512-bit).
	#[inline]
	pub fn clmul_u64x8<const IMM8: i32>(self, a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
		unsafe { clmul_u64x8_intrinsic::<IMM8>(&a, &b) }
	}
}

/// # Safety
/// Caller proved VPCLMULQDQ via [`Vpclmulqdq`].
#[inline]
#[target_feature(enable = "vpclmulqdq")]
unsafe fn clmul_u64x4_intrinsic<const IMM8: i32>(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vr = _mm256_clmulepi64_epi128::<IMM8>(va, vb);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved VPCLMULQDQ + AVX-512F via [`Vpclmulqdq512`].
#[inline]
#[target_feature(enable = "vpclmulqdq,avx512f")]
unsafe fn clmul_u64x8_intrinsic<const IMM8: i32>(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let vr = _mm512_clmulepi64_epi128::<IMM8>(va, vb);
		let mut out = [0u64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/other/vpclmulqdq.rs"]
mod tests;
