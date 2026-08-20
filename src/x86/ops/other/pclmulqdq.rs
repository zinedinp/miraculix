//! PCLMULQDQ (Westmere, 2010): carry-less 64x64->128 (`pclmulqdq`).
//! Stable `core::arch`. Token: [`Pclmulqdq::detect`]. Hand-written;
//! `const IMM8` selects operand halves. Scalar XOR-shift ref in tests.

use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_loadu_si128, _mm_storeu_si128};

use super::super::super::{Feature, FeatureSet};

/// Proof token: PCLMULQDQ available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Pclmulqdq(());

impl Pclmulqdq {
	/// `None` if the CPU (or the compile-time target) lacks PCLMULQDQ.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Pclmulqdq).then_some(Pclmulqdq(()))
	}

	/// Carry-less multiply of one 64-bit half of `a` by one 64-bit half of
	/// `b`, no carry propagation, 128-bit product as `[low, high]`
	/// (`pclmulqdq`). `IMM8` bit 0 selects `a`'s half (0 = low, 1 = high);
	/// bit 4 selects `b`'s half the same way. The usual callers pass
	/// `0x00`/`0x01`/`0x10`/`0x11`.
	#[inline]
	pub fn clmul<const IMM8: i32>(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		unsafe { pclmulqdq::<IMM8>(&a, &b) }
	}
}

/// # Safety
/// Caller proved PCLMULQDQ via [`Pclmulqdq`].
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn pclmulqdq<const IMM8: i32>(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_clmulepi64_si128::<IMM8>(va, vb);
		let mut out = [0u64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/other/pclmulqdq.rs"]
mod tests;
