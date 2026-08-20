//! SHA-1/SHA-256-NI (`"sha"`): msg schedule + round compress. Distinct from `Feature::Sha512`.
//! Token: [`Sha::detect`] (no auto/AVX dependency). Hand-written whole-block; Guide semantics.
//! Array index 0 = bits `[31:0]`; verified against `hashlib` before transcription.

use core::arch::x86_64::{
	__m128i, _mm_loadu_si128, _mm_sha1msg1_epu32, _mm_sha1msg2_epu32, _mm_sha1nexte_epu32, _mm_sha1rnds4_epu32,
	_mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SHA-1/SHA-256-NI available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sha(());

impl Sha {
	/// `None` if the CPU (or the compile-time target) lacks SHA-NI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sha).then_some(Sha(()))
	}

	/// SHA-1 msg schedule step 1: 4 words of `W[i-16..i-13] XOR W[i-8..i-5]`
	/// (no rotate yet; see [`Sha::sha1msg2`]) (`sha1msg1`).
	#[inline]
	pub fn sha1msg1(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha1msg1_intrinsic(&a, &b) }
	}

	/// SHA-1 msg schedule step 2: XORs in `W[i-3..i]` and rotates left by 1,
	/// completing `W[i] = ROTL1(W[i-3]^W[i-8]^W[i-14]^W[i-16])` for 4 words
	/// (`sha1msg2`).
	#[inline]
	pub fn sha1msg2(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha1msg2_intrinsic(&a, &b) }
	}

	/// Folds `e + ROTL30(a_prev)` into the next message-word group's first
	/// slot, the value [`Sha::sha1rnds4`] needs but has no direct `e` input
	/// for (`sha1nexte`).
	#[inline]
	pub fn sha1nexte(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha1nexte_intrinsic(&a, &b) }
	}

	/// 4 SHA-1 rounds. `FUNC` selects the round function/constant: `0`=`Ch`
	/// (rounds 0-19), `1`=`Parity` (20-39), `2`=`Maj` (40-59), `3`=`Parity`
	/// (60-79) (`sha1rnds4`).
	#[inline]
	pub fn sha1rnds4<const FUNC: i32>(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha1rnds4_intrinsic::<FUNC>(&a, &b) }
	}

	/// SHA-256 msg schedule step 1: `dst[i] = a[i] + sigma0(w4[i])`, the
	/// message-schedule analogue of [`Sha::sha1msg1`] (`sha256msg1`).
	#[inline]
	pub fn sha256msg1(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha256msg1_intrinsic(&a, &b) }
	}

	/// SHA-256 msg schedule step 2: chained `+sigma1(...)` into 4 new words,
	/// the analogue of [`Sha::sha1msg2`] (`sha256msg2`).
	#[inline]
	pub fn sha256msg2(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		unsafe { sha256msg2_intrinsic(&a, &b) }
	}

	/// 2 SHA-256 rounds. `a`/`b` = `[H,G,D,C]`/`[F,E,B,A]`; `k`'s low 2
	/// lanes hold `W[0]+K[0]`, `W[1]+K[1]` (top 2 unused). Returns new
	/// `[F,E,B,A]` (next call's `[H,G,D,C]`) (`sha256rnds2`).
	#[inline]
	pub fn sha256rnds2(self, a: [u32; 4], b: [u32; 4], k: [u32; 4]) -> [u32; 4] {
		unsafe { sha256rnds2_intrinsic(&a, &b, &k) }
	}
}

macro_rules! sha_op {
	($intrinsic_fn:ident, $intrinsic:path) => {
		/// # Safety
		/// Caller proved SHA-NI via [`Sha`].
		#[inline]
		#[target_feature(enable = "sha")]
		unsafe fn $intrinsic_fn(a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
			unsafe {
				let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
				let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
				let vr = $intrinsic(va, vb);
				let mut out = [0u32; 4];
				_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

sha_op!(sha1msg1_intrinsic, _mm_sha1msg1_epu32);
sha_op!(sha1msg2_intrinsic, _mm_sha1msg2_epu32);
sha_op!(sha1nexte_intrinsic, _mm_sha1nexte_epu32);
sha_op!(sha256msg1_intrinsic, _mm_sha256msg1_epu32);
sha_op!(sha256msg2_intrinsic, _mm_sha256msg2_epu32);

/// # Safety
/// Caller proved SHA-NI via [`Sha`].
#[inline]
#[target_feature(enable = "sha")]
unsafe fn sha1rnds4_intrinsic<const FUNC: i32>(a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_sha1rnds4_epu32::<FUNC>(va, vb);
		let mut out = [0u32; 4];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved SHA-NI via [`Sha`].
#[inline]
#[target_feature(enable = "sha")]
unsafe fn sha256rnds2_intrinsic(a: &[u32; 4], b: &[u32; 4], k: &[u32; 4]) -> [u32; 4] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(k.as_ptr().cast());
		let vr = _mm_sha256rnds2_epu32(va, vb, vk);
		let mut out = [0u32; 4];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/other/sha.rs"]
mod tests;
