//! AVX-512CD (2016/17): 512-bit only, `"avx512cd"` (no VL). MSRV 1.89.
//! Token: [`Avx512cd::detect`]. Ops: `conflict_*` (hand), `leading_zeros_*` (`simd_unop`+masked), `broadcast_mask_*`.
//! u32/u64 only. No auto.

use core::arch::x86_64::{
	__m512i, _mm512_broadcastmb_epi64, _mm512_broadcastmw_epi32, _mm512_conflict_epi32, _mm512_conflict_epi64,
	_mm512_loadu_si512, _mm512_lzcnt_epi32, _mm512_lzcnt_epi64, _mm512_mask_lzcnt_epi32, _mm512_mask_lzcnt_epi64,
	_mm512_maskz_lzcnt_epi32, _mm512_maskz_lzcnt_epi64, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::macros::{simd_unop, simd_unop_masked};

/// Proof token: AVX-512CD available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512cd(());

impl Avx512cd {
	/// `None` if the CPU (or the compile-time target) lacks AVX-512CD.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512cd).then_some(Avx512cd(()))
	}

	/// From resolved tier (`V4` lists `Feature::Avx512cd`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx512cd).then_some(Avx512cd(()))
	}
}

macro_rules! avx512cd_u32_unop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_unop! {
			token = Avx512cd, target_feature = "avx512cd",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512cd_u64_unop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_unop! {
			token = Avx512cd, target_feature = "avx512cd",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512cd_u32_unop!(
	leading_zeros_u32x16, leading_zeros_u32_slice, plzcntd, _mm512_lzcnt_epi32, u32::leading_zeros,
	"Per-lane leading-zero-bit count (`vplzcntd`, 512-bit).",
	"`out[i] = a[i].leading_zeros()`. 16-wide `leading_zeros_u32x16` chunks, scalar remainder."
);
avx512cd_u64_unop!(
	leading_zeros_u64x8, leading_zeros_u64_slice, plzcntq, _mm512_lzcnt_epi64, |x: u64| x.leading_zeros() as u64,
	"Per-lane leading-zero-bit count (`vplzcntq`, 512-bit).",
	"`out[i] = a[i].leading_zeros()`. 8-wide `leading_zeros_u64x8` chunks, scalar remainder."
);

simd_unop_masked! {
	token = Avx512cd, target_feature = "avx512cd",
	merge_fn = leading_zeros_u32x16_merge_masked, zero_fn = leading_zeros_u32x16_zero_masked,
	merge_intrinsic_fn = mask_lzcnt_epi32_intrinsic, zero_intrinsic_fn = maskz_lzcnt_epi32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_lzcnt_epi32, zero_intrinsic = _mm512_maskz_lzcnt_epi32,
	merge_doc = "Per-lane leading-zero-bit count where `mask` bit is set, else copied from `src` (`vplzcntd`, merge-masked).",
	zero_doc = "Per-lane leading-zero-bit count where `mask` bit is set, else zero (`vplzcntd`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512cd, target_feature = "avx512cd",
	merge_fn = leading_zeros_u64x8_merge_masked, zero_fn = leading_zeros_u64x8_zero_masked,
	merge_intrinsic_fn = mask_lzcnt_epi64_intrinsic, zero_intrinsic_fn = maskz_lzcnt_epi64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_lzcnt_epi64, zero_intrinsic = _mm512_maskz_lzcnt_epi64,
	merge_doc = "Per-lane leading-zero-bit count where `mask` bit is set, else copied from `src` (`vplzcntq`, merge-masked).",
	zero_doc = "Per-lane leading-zero-bit count where `mask` bit is set, else zero (`vplzcntq`, zero-masked).",
}

impl Avx512cd {
	/// For lane `i`, a bitmask of earlier lanes `j < i` (within this
	/// 16-lane call) with `a[j] == a[i]` (`vpconflictd`, 512-bit).
	#[inline]
	pub fn conflict_u32x16(self, a: [u32; 16]) -> [u32; 16] {
		unsafe { pconflictd(&a) }
	}

	/// `out[i]` is a bitmask of earlier lanes (within the 16-wide window)
	/// equal to `a[i]`. Chunked by 16; the remainder is handled independently.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn conflict_u32_slice(self, a: &[u32], out: &mut [u32]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u32; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.conflict_u32x16(av));
		}
		conflict_scalar_window_u32(a_rem, out_chunks.into_remainder());
	}

	/// For lane `i`, a bitmask of earlier lanes `j < i` (within this
	/// 8-lane call) with `a[j] == a[i]` (`vpconflictq`, 512-bit).
	#[inline]
	pub fn conflict_u64x8(self, a: [u64; 8]) -> [u64; 8] {
		unsafe { pconflictq(&a) }
	}

	/// `out[i]` is a bitmask of earlier lanes (within the 8-wide window)
	/// equal to `a[i]`. Same shape as `conflict_u32_slice`.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn conflict_u64_slice(self, a: &[u64], out: &mut [u64]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u64; 8] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.conflict_u64x8(av));
		}
		conflict_scalar_window_u64(a_rem, out_chunks.into_remainder());
	}

	/// Every lane becomes `mask` zero-extended to `u32` (`vpbroadcastmw2d`,
	/// 512-bit): all 16 lanes are identical copies of `mask`, not a
	/// per-bit selector. No slice wrapper: `mask` is a single scalar, not
	/// a same-length buffer.
	#[inline]
	pub fn broadcast_mask_u32x16(self, mask: u16) -> [u32; 16] {
		unsafe { pbroadcastmw2d(mask) }
	}

	/// Every lane becomes `mask` zero-extended to `u64` (`vpbroadcastmb2q`,
	/// 512-bit): all 8 lanes are identical copies of `mask`, not a
	/// per-bit selector. No slice wrapper: `mask` is a single scalar, not
	/// a same-length buffer.
	#[inline]
	pub fn broadcast_mask_u64x8(self, mask: u8) -> [u64; 8] {
		unsafe { pbroadcastmb2q(mask) }
	}
}

/// Plain-Rust completion of `conflict_u32x16`'s window-relative semantic
/// to a window shorter than the hardware width (the scalar remainder
/// path): for each index `i` in `window`, a bitmask of earlier indices
/// `j < i` in the *same* `window` with `window[j] == window[i]`.
fn conflict_scalar_window_u32(window: &[u32], out: &mut [u32]) {
	for i in 0..window.len() {
		let mut mask = 0u32;
		for j in 0..i {
			if window[j] == window[i] {
				mask |= 1 << j;
			}
		}
		out[i] = mask;
	}
}

/// Same as [`conflict_scalar_window_u32`], for `conflict_u64x8`'s
/// remainder.
fn conflict_scalar_window_u64(window: &[u64], out: &mut [u64]) {
	for i in 0..window.len() {
		let mut mask = 0u64;
		for j in 0..i {
			if window[j] == window[i] {
				mask |= 1 << j;
			}
		}
		out[i] = mask;
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512cd")]
unsafe fn pconflictd(a: &[u32; 16]) -> [u32; 16] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vr = _mm512_conflict_epi32(va);
		let mut out = [0u32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512cd")]
unsafe fn pconflictq(a: &[u64; 8]) -> [u64; 8] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vr = _mm512_conflict_epi64(va);
		let mut out = [0u64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512cd")]
unsafe fn pbroadcastmw2d(mask: u16) -> [u32; 16] {
	unsafe {
		let vr = _mm512_broadcastmw_epi32(mask);
		let mut out = [0u32; 16];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avx512cd")]
unsafe fn pbroadcastmb2q(mask: u8) -> [u64; 8] {
	unsafe {
		let vr = _mm512_broadcastmb_epi64(mask);
		let mut out = [0u64; 8];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512cd.rs"]
mod tests;
