//! AVX-IFMA: VEX 52-bit integer FMA on `u64` lanes (`avxifma`).
//! Token: [`AvxIfma::detect`]. Same math as EVEX `avx512ifma` but a distinct CPUID bit; falls back to scalar if absent.

use core::arch::x86_64::{
	__m128i, __m256i, _mm256_loadu_si256, _mm256_madd52hi_avx_epu64, _mm256_madd52lo_avx_epu64, _mm256_storeu_si256,
	_mm_loadu_si128, _mm_madd52hi_avx_epu64, _mm_madd52lo_avx_epu64, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_ternop;

/// Proof token: AVX-IFMA available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct AvxIfma(());

impl AvxIfma {
	/// `None` if the CPU (or the compile-time target) lacks AVX-IFMA.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::AvxIfma).then_some(AvxIfma(()))
	}
}

/// Operand mask only (acc unmasked). Matches Guide `ZeroExtend64` of low 52.
const MASK52: u64 = (1 << 52) - 1;

fn madd52lo_scalar(src: u64, a: u64, b: u64) -> u64 {
	let product = (a & MASK52) as u128 * (b & MASK52) as u128;
	src.wrapping_add((product & MASK52 as u128) as u64)
}

fn madd52hi_scalar(src: u64, a: u64, b: u64) -> u64 {
	let product = (a & MASK52) as u128 * (b & MASK52) as u128;
	src.wrapping_add(((product >> 52) & MASK52 as u128) as u64)
}

simd_ternop! {
	token = AvxIfma, vis = pub, target_feature = "avxifma",
	fixed_fn = madd52lo_u64x2, slice_fn = madd52lo_u64_slice, intrinsic_fn = madd52lo_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_madd52lo_avx_epu64, scalar = madd52lo_scalar,
	fixed_doc = "`src + low52(a * b)` per lane (`vpmadd52luq`, 128-bit).",
	slice_doc = "`out[i] = src[i] + low52(a[i] * b[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = AvxIfma, vis = pub, target_feature = "avxifma",
	fixed_fn = madd52lo_u64x4, slice_fn = madd52lo_u64_slice_wide, intrinsic_fn = madd52lo_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_madd52lo_avx_epu64, scalar = madd52lo_scalar,
	fixed_doc = "`src + low52(a * b)` per lane (`vpmadd52luq`, 256-bit).",
	slice_doc = "`out[i] = src[i] + low52(a[i] * b[i])`. 4-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = AvxIfma, vis = pub, target_feature = "avxifma",
	fixed_fn = madd52hi_u64x2, slice_fn = madd52hi_u64_slice, intrinsic_fn = madd52hi_u64x2_intrinsic,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_madd52hi_avx_epu64, scalar = madd52hi_scalar,
	fixed_doc = "`src + high52(a * b)` per lane (`vpmadd52huq`, 128-bit).",
	slice_doc = "`out[i] = src[i] + high52(a[i] * b[i])`. 2-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = AvxIfma, vis = pub, target_feature = "avxifma",
	fixed_fn = madd52hi_u64x4, slice_fn = madd52hi_u64_slice_wide, intrinsic_fn = madd52hi_u64x4_intrinsic,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_madd52hi_avx_epu64, scalar = madd52hi_scalar,
	fixed_doc = "`src + high52(a * b)` per lane (`vpmadd52huq`, 256-bit).",
	slice_doc = "`out[i] = src[i] + high52(a[i] * b[i])`. 4-wide chunks, software scalar rem.",
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx_ifma.rs"]
mod tests;
