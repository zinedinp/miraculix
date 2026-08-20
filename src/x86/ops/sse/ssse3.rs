//! SSSE3: `pshufb` byte LUT/permute, `pabs*`, and `palignr` byte-window align.
//! Fixed-width only. Token: [`Ssse3::detect`].

use core::arch::x86_64::{
	__m128i, _mm_abs_epi16, _mm_abs_epi32, _mm_abs_epi8, _mm_alignr_epi8, _mm_loadu_si128, _mm_shuffle_epi8,
	_mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_unop;

/// Proof token: SSSE3 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Ssse3(());

impl Ssse3 {
	/// `None` if the CPU (or the compile-time target) lacks SSSE3.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Ssse3).then_some(Ssse3(()))
	}

	/// Per-byte LUT (`pshufb`): `out[i]=a[idx&0x0F]`, or 0 if `idx` high bit set.
	#[inline]
	pub fn shuffle_i8x16(self, a: [i8; 16], indices: [i8; 16]) -> [i8; 16] {
		unsafe { pshufb(&a, &indices) }
	}

	/// `palignr`: concatenate `[b, a]` into a 32-byte window (`b` low, `a`
	/// high), shift right by `IMM8` bytes, keep the low 16. `IMM8 >= 32`
	/// yields all zero; `16 <= IMM8 < 32` reads only from `a` shifted by
	/// `IMM8 - 16`, zero-filled from the top.
	#[inline]
	pub fn alignr_u8x16<const IMM8: i32>(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		unsafe { alignr::<IMM8>(&a, &b) }
	}
}

/// `pshufb` via unaligned `movdqu`.
///
/// # Safety
/// Caller proved SSSE3 via [`Ssse3`].
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn pshufb(a: &[i8; 16], indices: &[i8; 16]) -> [i8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vi: __m128i = _mm_loadu_si128(indices.as_ptr().cast());
		let vr = _mm_shuffle_epi8(va, vi);
		let mut out = [0i8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSSE3 via [`Ssse3`].
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn alignr<const IMM8: i32>(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_alignr_epi8::<IMM8>(va, vb);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Software reference: `out[i] = window[imm+i]` where `window = [b, a]`
/// concatenated (32 bytes, `b` first), or 0 past the end. Shared with
/// `super::super::avx::avx2`'s per-lane 256-bit test.
#[cfg(test)]
pub(crate) fn alignr_scalar(a: &[u8], b: &[u8], imm: i32) -> Vec<u8> {
	debug_assert_eq!(a.len(), b.len());
	let n = a.len();
	let window: Vec<u8> = b.iter().chain(a.iter()).copied().collect();
	(0..n)
		.map(|i| {
			let pos = imm as i64 + i as i64;
			if pos < 0 || pos as usize >= window.len() { 0 } else { window[pos as usize] }
		})
		.collect()
}

simd_unop! {
	token = Ssse3, target_feature = "ssse3",
	fixed_fn = abs_i8x16, slice_fn = abs_i8_slice, intrinsic_fn = pabsb,
	width = 16, elem = i8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_abs_epi8, scalar = |x: i8| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`pabsb`).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 16-wide chunks, scalar remainder.",
}
simd_unop! {
	token = Ssse3, target_feature = "ssse3",
	fixed_fn = abs_i16x8, slice_fn = abs_i16_slice, intrinsic_fn = pabsw,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_abs_epi16, scalar = |x: i16| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`pabsw`).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 8-wide chunks, scalar remainder.",
}
simd_unop! {
	token = Ssse3, target_feature = "ssse3",
	fixed_fn = abs_i32x4, slice_fn = abs_i32_slice, intrinsic_fn = pabsd,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_abs_epi32, scalar = |x: i32| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`pabsd`).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 4-wide chunks, scalar remainder.",
}

#[cfg(test)]
#[path = "../../test/ops/sse/ssse3.rs"]
mod tests;
