//! AES-NI (Westmere, 2010): six whole-block ops on opaque `[u8; 16]`.
//! Stable `core::arch`. Token: [`Aes::detect`]. Hand-written (not per-lane).
//! Tests: AES-128 schedule + encrypt/decrypt vs FIPS-197 Appendix B.

use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_storeu_si128};
use core::arch::x86_64::{
	_mm_aesdec_si128, _mm_aesdeclast_si128, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_aesimc_si128,
	_mm_aeskeygenassist_si128,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: AES-NI available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Aes(());

impl Aes {
	/// `None` if the CPU (or the compile-time target) lacks AES-NI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Aes).then_some(Aes(()))
	}

	/// One AES encrypt round: `ShiftRows -> SubBytes -> MixColumns -> XOR
	/// round_key` (`aesenc`).
	#[inline]
	pub fn aesenc(self, state: [u8; 16], round_key: [u8; 16]) -> [u8; 16] {
		unsafe { aesenc_raw(&state, &round_key) }
	}

	/// Final AES encrypt round: `ShiftRows -> SubBytes -> XOR round_key`
	/// (no `MixColumns`) (`aesenclast`).
	#[inline]
	pub fn aesenclast(self, state: [u8; 16], round_key: [u8; 16]) -> [u8; 16] {
		unsafe { aesenclast_raw(&state, &round_key) }
	}

	/// One AES decrypt round: `InvShiftRows -> InvSubBytes -> InvMixColumns
	/// -> XOR round_key` (`aesdec`).
	#[inline]
	pub fn aesdec(self, state: [u8; 16], round_key: [u8; 16]) -> [u8; 16] {
		unsafe { aesdec_raw(&state, &round_key) }
	}

	/// Final AES decrypt round: `InvShiftRows -> InvSubBytes -> XOR
	/// round_key` (no `InvMixColumns`) (`aesdeclast`).
	#[inline]
	pub fn aesdeclast(self, state: [u8; 16], round_key: [u8; 16]) -> [u8; 16] {
		unsafe { aesdeclast_raw(&state, &round_key) }
	}

	/// `InvMixColumns(round_key)` (`aesimc`): converts an encrypt-direction
	/// round key into the form [`Aes::aesdec`] expects (the standard
	/// AES-NI "equivalent inverse cipher" construction).
	#[inline]
	pub fn aesimc(self, round_key: [u8; 16]) -> [u8; 16] {
		unsafe { aesimc_raw(&round_key) }
	}

	/// Key-schedule helper (`aeskeygenassist`): `SubWord(RotWord(key[12..16]))`
	/// broadcast into lanes 1 and 3, XORed with `RCON` in lanes 1 and 3 (lanes
	/// 0 and 2 hold the un-rotated `SubWord`). `RCON` is the raw round
	/// constant byte.
	#[inline]
	pub fn aeskeygenassist<const RCON: i32>(self, key: [u8; 16]) -> [u8; 16] {
		unsafe { aeskeygenassist_raw::<RCON>(&key) }
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aesenc_raw(state: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vs: __m128i = _mm_loadu_si128(state.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(round_key.as_ptr().cast());
		let vr = _mm_aesenc_si128(vs, vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aesenclast_raw(state: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vs: __m128i = _mm_loadu_si128(state.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(round_key.as_ptr().cast());
		let vr = _mm_aesenclast_si128(vs, vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aesdec_raw(state: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vs: __m128i = _mm_loadu_si128(state.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(round_key.as_ptr().cast());
		let vr = _mm_aesdec_si128(vs, vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aesdeclast_raw(state: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vs: __m128i = _mm_loadu_si128(state.as_ptr().cast());
		let vk: __m128i = _mm_loadu_si128(round_key.as_ptr().cast());
		let vr = _mm_aesdeclast_si128(vs, vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aesimc_raw(round_key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vk: __m128i = _mm_loadu_si128(round_key.as_ptr().cast());
		let vr = _mm_aesimc_si128(vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AES-NI via [`Aes`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn aeskeygenassist_raw<const RCON: i32>(key: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let vk: __m128i = _mm_loadu_si128(key.as_ptr().cast());
		let vr = _mm_aeskeygenassist_si128::<RCON>(vk);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/other/aes.rs"]
mod tests;
