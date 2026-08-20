//! AVX512BITALG: per-lane population counts and 512-bit bitshuffle.
//! Token: `Avx512Bitalg`.

use core::arch::x86_64::{
	__m512i, _mm512_bitshuffle_epi64_mask, _mm512_loadu_si512, _mm512_mask_popcnt_epi16, _mm512_mask_popcnt_epi8,
	_mm512_maskz_popcnt_epi16, _mm512_maskz_popcnt_epi8, _mm512_popcnt_epi16, _mm512_popcnt_epi8,
	_mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{simd_unop, simd_unop_masked};

/// Proof token: AVX512BITALG, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Bitalg(());

impl Avx512Bitalg {
	/// `None` if the CPU (or the compile-time target) lacks AVX512BITALG.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512bitalg).then_some(Avx512Bitalg(()))
	}
}

simd_unop! {
	token = Avx512Bitalg, target_feature = "avx512bitalg",
	fixed_fn = popcnt_u8x64, slice_fn = popcnt_u8_slice, intrinsic_fn = popcnt_u8x64_intrinsic,
	width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_popcnt_epi8, scalar = |x: u8| x.count_ones() as u8,
	fixed_doc = "Per-lane population count (`vpopcntb`, 512-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 64-wide chunks, scalar remainder.",
}

simd_unop! {
	token = Avx512Bitalg, target_feature = "avx512bitalg",
	fixed_fn = popcnt_u16x32, slice_fn = popcnt_u16_slice, intrinsic_fn = popcnt_u16x32_intrinsic,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_popcnt_epi16, scalar = |x: u16| x.count_ones() as u16,
	fixed_doc = "Per-lane population count (`vpopcntw`, 512-bit).",
	slice_doc = "`out[i] = a[i].count_ones()`. 32-wide chunks, scalar remainder.",
}

simd_unop_masked! {
	token = Avx512Bitalg, target_feature = "avx512bitalg",
	merge_fn = popcnt_u8x64_merge_masked, zero_fn = popcnt_u8x64_zero_masked,
	merge_intrinsic_fn = mask_popcnt_epi8_intrinsic, zero_intrinsic_fn = maskz_popcnt_epi8_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_popcnt_epi8, zero_intrinsic = _mm512_maskz_popcnt_epi8,
	merge_doc = "Per-lane population count where `mask` bit is set, else copied from `src` (`vpopcntb`, merge-masked).",
	zero_doc = "Per-lane population count where `mask` bit is set, else zero (`vpopcntb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Bitalg, target_feature = "avx512bitalg",
	merge_fn = popcnt_u16x32_merge_masked, zero_fn = popcnt_u16x32_zero_masked,
	merge_intrinsic_fn = mask_popcnt_epi16_intrinsic, zero_intrinsic_fn = maskz_popcnt_epi16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_popcnt_epi16, zero_intrinsic = _mm512_maskz_popcnt_epi16,
	merge_doc = "Per-lane population count where `mask` bit is set, else copied from `src` (`vpopcntw`, merge-masked).",
	zero_doc = "Per-lane population count where `mask` bit is set, else zero (`vpopcntw`, zero-masked).",
}

impl Avx512Bitalg {
	/// `vpshufbitqmb` (512-bit): for qword lane `i` and control byte `j`,
	/// take bit `(c[i].byte[j] & 0x3F)` of `b[i]` into mask bit `i*8+j`.
	/// Eight lanes x eight bytes -> 64 mask bits.
	#[inline]
	pub fn bitshuffle_mask_u64x8(self, b: [u64; 8], c: [u64; 8]) -> u64 {
		unsafe { bitshuffle_qmb_512(&b, &c) }
	}
}

/// Software stand-in for bitshuffle: bit `i*8+j` = bit `(c[i].byte[j] & 0x3F)` of `b[i]`.
/// Shared with `Avx512BitalgVl`'s tests in `super::avx512vl`.
#[cfg(test)]
pub(crate) fn bitshuffle_scalar(b: &[u64], c: &[u64]) -> u64 {
	debug_assert_eq!(b.len(), c.len());
	let mut out = 0u64;
	for (i, (&bl, &cl)) in b.iter().zip(c).enumerate() {
		for j in 0..8 {
			let byte = (cl >> (j * 8)) as u8;
			let m = byte & 0x3F;
			let bit = (bl >> m) & 1;
			out |= bit << (i * 8 + j);
		}
	}
	out
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512bitalg")]
unsafe fn bitshuffle_qmb_512(b: &[u64; 8], c: &[u64; 8]) -> u64 {
	unsafe {
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let vc: __m512i = _mm512_loadu_si512(c.as_ptr().cast());
		_mm512_bitshuffle_epi64_mask(vb, vc)
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512bitalg.rs"]
mod tests;
