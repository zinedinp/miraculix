//! AVX512VBMI (Cannon Lake, 2018): byte permute/bit-shuffle ops, hand-written (index selects lanes).
//! 512-bit token [`Avx512Vbmi`] (`"avx512vbmi"`): `permutexvar`, `permutex2var`, `multishift`.
//! Tier-unique, no `auto`. 128/256-bit: [`super::avx512vl`] (`Avx512VbmiVl`).

use core::arch::x86_64::{
	__m512i, _mm512_loadu_si512, _mm512_mask_multishift_epi64_epi8, _mm512_mask_permutex2var_epi8,
	_mm512_mask_permutexvar_epi8, _mm512_maskz_multishift_epi64_epi8, _mm512_maskz_permutex2var_epi8,
	_mm512_maskz_permutexvar_epi8, _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
	_mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{simd_binop_masked, simd_ternop_masked};

/// Proof token: AVX512VBMI, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Vbmi(());

impl Avx512Vbmi {
	/// `None` if the CPU (or the compile-time target) lacks AVX512VBMI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512vbmi).then_some(Avx512Vbmi(()))
	}

	/// `out[i] = a[idx[i] & 63]` (`vpermb`): 1-source byte permute, index wraps mod 64.
	#[inline]
	pub fn permutexvar_u8x64(self, idx: [u8; 64], a: [u8; 64]) -> [u8; 64] {
		unsafe { permutexvar(&idx, &a) }
	}

	/// `off = idx[i] & 63; out[i] = if idx[i] & 64 != 0 { b[off] } else { a[off] }`
	/// (`vpermi2b`): 2-source byte permute, index bit 6 selects `a`/`b`.
	#[inline]
	pub fn permutex2var_u8x64(self, a: [u8; 64], idx: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		unsafe { permutex2var(&a, &idx, &b) }
	}

	/// `vpmultishiftqb`: per 64-bit lane of `b`, byte `j` of the result is the
	/// 8-bit window of that lane starting at bit `a[byte j] & 63` (wrapping
	/// within the 64-bit lane, LSB-first). Control comes from `a`, data from `b`.
	#[inline]
	pub fn multishift_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		unsafe { multishift(&a, &b) }
	}
}

// Merge/zero-masked forms. None of the three unmasked ops fit `simd_*op!`
// (no honest scalar closure: see module doc), but the raw masked wrappers
// never needed one to begin with, so they fit the existing generic macros
// once checked against stdarch: `permutexvar`/`multishift` are binop-shaped
// (two same-elem, same-width vector inputs), and `permutex2var`'s merge form
// reuses its first operand as the fallback exactly like `ternarylogic`/IFMA's
// `madd52lo`: `_mm512_mask_permutex2var_epi8(a, k, idx, b)`.
simd_binop_masked! {
	token = Avx512Vbmi, target_feature = "avx512vbmi",
	merge_fn = permutexvar_u8x64_merge_masked, zero_fn = permutexvar_u8x64_zero_masked,
	merge_intrinsic_fn = mask_permutexvar_u8x64_intrinsic, zero_intrinsic_fn = maskz_permutexvar_u8x64_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_permutexvar_epi8, zero_intrinsic = _mm512_maskz_permutexvar_epi8,
	merge_doc = "[`Avx512Vbmi::permutexvar_u8x64`] where `mask` bit is set, else copied from `src` (`vpermb`, merge-masked).",
	zero_doc = "[`Avx512Vbmi::permutexvar_u8x64`] where `mask` bit is set, else zero (`vpermb`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi, target_feature = "avx512vbmi",
	merge_fn = permutex2var_u8x64_merge_masked, zero_fn = permutex2var_u8x64_zero_masked,
	merge_intrinsic_fn = mask_permutex2var_u8x64_intrinsic, zero_intrinsic_fn = maskz_permutex2var_u8x64_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_permutex2var_epi8, zero_intrinsic = _mm512_maskz_permutex2var_epi8,
	merge_doc = "[`Avx512Vbmi::permutex2var_u8x64`] where `mask` bit is set, else copied from `a` (`vpermi2b`, merge-masked). `a` doubles as both a permute input and the merge fallback - the encoding has no separate `src`.",
	zero_doc = "[`Avx512Vbmi::permutex2var_u8x64`] where `mask` bit is set, else zero (`vpermi2b`, zero-masked).",
}

simd_binop_masked! {
	token = Avx512Vbmi, target_feature = "avx512vbmi",
	merge_fn = multishift_u8x64_merge_masked, zero_fn = multishift_u8x64_zero_masked,
	merge_intrinsic_fn = mask_multishift_u8x64_intrinsic, zero_intrinsic_fn = maskz_multishift_u8x64_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_multishift_epi64_epi8, zero_intrinsic = _mm512_maskz_multishift_epi64_epi8,
	merge_doc = "[`Avx512Vbmi::multishift_u8x64`] where `mask` bit is set, else copied from `src` (`vpmultishiftqb`, merge-masked).",
	zero_doc = "[`Avx512Vbmi::multishift_u8x64`] where `mask` bit is set, else zero (`vpmultishiftqb`, zero-masked).",
}

/// # Safety
/// Caller proved AVX512VBMI via [`Avx512Vbmi`].
#[inline]
#[target_feature(enable = "avx512vbmi")]
unsafe fn permutexvar(idx: &[u8; 64], a: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let vidx: __m512i = _mm512_loadu_si512(idx.as_ptr().cast());
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vr = _mm512_permutexvar_epi8(vidx, va);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512VBMI via [`Avx512Vbmi`].
#[inline]
#[target_feature(enable = "avx512vbmi")]
unsafe fn permutex2var(a: &[u8; 64], idx: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vidx: __m512i = _mm512_loadu_si512(idx.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let vr = _mm512_permutex2var_epi8(va, vidx, vb);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512VBMI via [`Avx512Vbmi`].
#[inline]
#[target_feature(enable = "avx512vbmi")]
unsafe fn multishift(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let vr = _mm512_multishift_epi64_epi8(va, vb);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Software reference: `out[i] = a[idx[i] & (len-1)]`. `len` is the qword-group
/// size (64/32/16), one power-of-2 lane count per instantiation width. Shared
/// with `Avx512VbmiVl`'s tests in `super::avx512vl`.
#[cfg(test)]
pub(crate) fn permutexvar_scalar(idx: &[u8], a: &[u8]) -> Vec<u8> {
	let mask = (a.len() - 1) as u8;
	idx.iter().map(|&i| a[(i & mask) as usize]).collect()
}

/// Software reference for `permutex2var`: `len` doubles as the source-select bit
/// (a power of 2 equal to the lane count). Shared with `super::avx512vl`.
#[cfg(test)]
pub(crate) fn permutex2var_scalar(a: &[u8], idx: &[u8], b: &[u8]) -> Vec<u8> {
	let len = a.len();
	let mask = (len - 1) as u8;
	let select_bit = len as u8;
	idx.iter()
		.map(|&i| {
			let off = (i & mask) as usize;
			if i & select_bit != 0 { b[off] } else { a[off] }
		})
		.collect()
}

/// Software reference for `multishift`: 8-byte qword groups, bit-level window
/// extraction. Shared with `super::avx512vl`.
#[cfg(test)]
pub(crate) fn multishift_scalar(a: &[u8], b: &[u8]) -> Vec<u8> {
	debug_assert_eq!(a.len(), b.len());
	debug_assert_eq!(a.len() % 8, 0);
	let bit = |bytes: &[u8], pos: usize| (bytes[pos / 8] >> (pos % 8)) & 1;
	let mut out = vec![0u8; a.len()];
	for q in 0..(a.len() / 8) {
		let a_q = &a[q * 8..q * 8 + 8];
		let b_q = &b[q * 8..q * 8 + 8];
		for j in 0..8 {
			let ctrl = (a_q[j] & 63) as usize;
			let mut byte = 0u8;
			for l in 0..8 {
				byte |= bit(b_q, (ctrl + l) & 63) << l;
			}
			out[q * 8 + j] = byte;
		}
	}
	out
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512vbmi.rs"]
mod tests;
