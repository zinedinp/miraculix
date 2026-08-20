//! SM3 (~2024 Sierra Forest/Granite Rapids): GB/T 32905-2016 hash, msg schedule + double-round (`"sm3"`).
//! Token: [`Sm3::detect`] (no auto). Hand-written whole-block; msg2 takes msg1 output directly.

use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sm3msg1_epi32, _mm_sm3msg2_epi32, _mm_sm3rnds2_epi32, _mm_storeu_si128};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SM3 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sm3(());

impl Sm3 {
	/// `None` if the CPU (or the compile-time target) lacks SM3.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sm3).then_some(Sm3(()))
	}
}

impl Sm3 {
	/// Msg schedule step1 (`vsm3msg1`): `a=W[t-9..]`, `b=[W[t-3..],_]`, `c=W[t-16..]`.
	/// Lanes 0..2: `P1(c^a^ROL(b,15))`; lane 3: `P1(c^a)` (msg2 patches ROL term).
	/// `P1(x)=x^ROL15^ROL23`.
	#[inline]
	pub fn sm3msg1(self, a: [u32; 4], b: [u32; 4], c: [u32; 4]) -> [u32; 4] {
		unsafe { sm3msg1_intrinsic(&a, &b, &c) }
	}

	/// Msg schedule step2 (`vsm3msg2`): `a=msg1 out`, `b=W[t-13..]`, `c=W[t-6..]`.
	/// Lanes 0..2: `ROL(b,7)^c^a`; lane 3 also `^P1(ROL(dst[0],15))`.
	#[inline]
	pub fn sm3msg2(self, a: [u32; 4], b: [u32; 4], c: [u32; 4]) -> [u32; 4] {
		unsafe { sm3msg2_intrinsic(&a, &b, &c) }
	}

	/// Two SM3 rounds (`vsm3rnds2`). `cdgh`/`abef` = `[H,G,D,C]`/`[F,E,B,A]`
	/// (same ping-pong as SHA512). `wp=[W[j],W[j+1],W'[j],W'[j+1]]`,
	/// `W'=W^W[+4]`. `IMM8 & 0x3e == j` (even, `0..=62`).
	#[inline]
	pub fn sm3rnds2<const IMM8: i32>(self, cdgh: [u32; 4], abef: [u32; 4], wp: [u32; 4]) -> [u32; 4] {
		unsafe { sm3rnds2_intrinsic::<IMM8>(&cdgh, &abef, &wp) }
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sm3,avx")]
unsafe fn sm3msg1_intrinsic(a: &[u32; 4], b: &[u32; 4], c: &[u32; 4]) -> [u32; 4] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vc: __m128i = _mm_loadu_si128(c.as_ptr().cast());
		let vr = _mm_sm3msg1_epi32(va, vb, vc);
		let mut out = [0u32; 4];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sm3,avx")]
unsafe fn sm3msg2_intrinsic(a: &[u32; 4], b: &[u32; 4], c: &[u32; 4]) -> [u32; 4] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vc: __m128i = _mm_loadu_si128(c.as_ptr().cast());
		let vr = _mm_sm3msg2_epi32(va, vb, vc);
		let mut out = [0u32; 4];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sm3,avx")]
unsafe fn sm3rnds2_intrinsic<const IMM8: i32>(cdgh: &[u32; 4], abef: &[u32; 4], wp: &[u32; 4]) -> [u32; 4] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(cdgh.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(abef.as_ptr().cast());
		let vc: __m128i = _mm_loadu_si128(wp.as_ptr().cast());
		let vr = _mm_sm3rnds2_epi32::<IMM8>(va, vb, vc);
		let mut out = [0u32; 4];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx/sm3.rs"]
mod tests;
