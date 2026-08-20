//! SHA512 (~2024 Sierra Forest/Granite Rapids): msg schedule + double-round (`"sha512"`).
//! Distinct from `Feature::Sha` (SHA-1/256 NI). Token: [`Sha512::detect`] (no auto).
//! Hand-written whole-block; between msg1/msg2 callers add aligned `W[t-7..t-4]`.

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_loadu_si256, _mm256_sha512msg1_epi64, _mm256_sha512msg2_epi64,
	_mm256_sha512rnds2_epi64, _mm256_storeu_si256, _mm_loadu_si128,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SHA512 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sha512(());

impl Sha512 {
	/// `None` if the CPU (or the compile-time target) lacks SHA512.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sha512).then_some(Sha512(()))
	}
}

impl Sha512 {
	/// Msg schedule step1: `dst[i] = a[i] + s0(w4[i])`, `w4=[a[1],a[2],a[3],b[0]]`,
	/// `s0=ROR1^ROR8^(x>>7)` (`vsha512msg1`). Then add `W[t-7..t-4]` before msg2.
	#[inline]
	pub fn sha512msg1(self, a: [u64; 4], b: [u64; 2]) -> [u64; 4] {
		unsafe { sha512msg1_intrinsic(&a, &b) }
	}

	/// Msg schedule step2: `s1=ROR19^ROR61^(x>>6)`; chained `a[i]+s1(...)` into 4 words
	/// (`vsha512msg2`).
	#[inline]
	pub fn sha512msg2(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		unsafe { sha512msg2_intrinsic(&a, &b) }
	}

	/// Two SHA-512 rounds. `cdgh`/`abef` = `[H,G,D,C]`/`[F,E,B,A]`; `wk=[W+K,W'+K']`.
	/// Returns new `abef` (next call's `cdgh`) (`vsha512rnds2`).
	#[inline]
	pub fn sha512rnds2(self, cdgh: [u64; 4], abef: [u64; 4], wk: [u64; 2]) -> [u64; 4] {
		unsafe { sha512rnds2_intrinsic(&cdgh, &abef, &wk) }
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sha512,avx")]
unsafe fn sha512msg1_intrinsic(a: &[u64; 4], b: &[u64; 2]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm256_sha512msg1_epi64(va, vb);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sha512,avx")]
unsafe fn sha512msg2_intrinsic(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vr = _mm256_sha512msg2_epi64(va, vb);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sha512,avx")]
unsafe fn sha512rnds2_intrinsic(cdgh: &[u64; 4], abef: &[u64; 4], wk: &[u64; 2]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(cdgh.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(abef.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(wk.as_ptr().cast());
		let vr = _mm256_sha512rnds2_epi64(va, vb, vk);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx/sha512.rs"]
mod tests;
