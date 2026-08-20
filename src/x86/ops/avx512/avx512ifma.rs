//! AVX512IFMA: 52-bit integer FMA on `u64` lanes (`"avx512ifma"`).
//! Token: [`Avx512Ifma`]. Same math as VEX `avxifma` but a different CPUID bit.
//! This file uses EVEX names; the VEX-suffixed forms are in `avx_ifma.rs`.
//! 128/256-bit `Avx512IfmaVl` lives in `super::avx512vl`.

use core::arch::x86_64::{
	__m512i, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_madd52hi_epu64,
	_mm512_mask_madd52lo_epu64, _mm512_maskz_madd52hi_epu64, _mm512_maskz_madd52lo_epu64, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{simd_ternop, simd_ternop_masked};

/// Proof token: AVX512IFMA, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Ifma(());

impl Avx512Ifma {
	/// `None` if the CPU (or the compile-time target) lacks AVX512IFMA.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512ifma).then_some(Avx512Ifma(()))
	}
}

/// Low 52 bits of each *operand* only; the accumulator is a full `u64`.
pub(crate) const MASK52: u64 = (1 << 52) - 1;

/// `src + low52((a & MASK52) * (b & MASK52))`, wrapping. Shared with the
/// 128/256-bit `Avx512IfmaVl` forms in `super::avx512vl`.
pub(crate) fn madd52lo_scalar(src: u64, a: u64, b: u64) -> u64 {
	let product = (a & MASK52) as u128 * (b & MASK52) as u128;
	src.wrapping_add((product & MASK52 as u128) as u64)
}

/// `src + high52((a & MASK52) * (b & MASK52))`, wrapping. Shared with `super::avx512vl`.
pub(crate) fn madd52hi_scalar(src: u64, a: u64, b: u64) -> u64 {
	let product = (a & MASK52) as u128 * (b & MASK52) as u128;
	src.wrapping_add(((product >> 52) & MASK52 as u128) as u64)
}

// Intrinsic order is (src, a, b) = Intel (a, b, c): acc first, then factors.
// That matches `simd_ternop`'s (a, b, c) -> `$intrinsic(va, vb, vc)`.

simd_ternop! {
	token = Avx512Ifma, vis = pub, target_feature = "avx512ifma",
	fixed_fn = madd52lo_u64x8, slice_fn = madd52lo_u64_slice, intrinsic_fn = madd52lo_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_madd52lo_epu64, scalar = madd52lo_scalar,
	fixed_doc = "`src + low52(a * b)` per lane (`vpmadd52luq`, 512-bit).",
	slice_doc = "`out[i] = src[i] + low52(a[i] * b[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Ifma, vis = pub, target_feature = "avx512ifma",
	fixed_fn = madd52hi_u64x8, slice_fn = madd52hi_u64_slice, intrinsic_fn = madd52hi_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_madd52hi_epu64, scalar = madd52hi_scalar,
	fixed_doc = "`src + high52(a * b)` per lane (`vpmadd52huq`, 512-bit).",
	slice_doc = "`out[i] = src[i] + high52(a[i] * b[i])`. 8-wide chunks, software scalar rem.",
}

// `simd_ternop_masked`'s "first operand doubles as the merge fallback" lands on
// the accumulator here: `_mm512_mask_madd52lo_epu64(src, k, a, b)` keeps the
// untouched accumulator lanes, which is what an unmasked lane should hold.
simd_ternop_masked! {
	token = Avx512Ifma, target_feature = "avx512ifma",
	merge_fn = madd52lo_u64x8_merge_masked, zero_fn = madd52lo_u64x8_zero_masked,
	merge_intrinsic_fn = mask_madd52lo_epu64_intrinsic, zero_intrinsic_fn = maskz_madd52lo_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_madd52lo_epu64, zero_intrinsic = _mm512_maskz_madd52lo_epu64,
	merge_doc = "`src + low52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52luq`, merge-masked).",
	zero_doc = "`src + low52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52luq`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Ifma, target_feature = "avx512ifma",
	merge_fn = madd52hi_u64x8_merge_masked, zero_fn = madd52hi_u64x8_zero_masked,
	merge_intrinsic_fn = mask_madd52hi_epu64_intrinsic, zero_intrinsic_fn = maskz_madd52hi_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_madd52hi_epu64, zero_intrinsic = _mm512_maskz_madd52hi_epu64,
	merge_doc = "`src + high52(a * b)` per lane where `mask` bit is set, else the accumulator lane `src` unchanged (`vpmadd52huq`, merge-masked).",
	zero_doc = "`src + high52(a * b)` per lane where `mask` bit is set, else zero (`vpmadd52huq`, zero-masked).",
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512ifma.rs"]
mod tests;
