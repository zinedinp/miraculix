//! SSE4.2: CRC32-C and improved integer compare/min/max support. Token: [`Sse42::detect`].
//! Provides `crc32_u8/u16/u32/u64` and native 64-bit compare ops.

use core::arch::x86_64::{
	__m128i, _mm_blendv_epi8, _mm_cmpgt_epi64, _mm_crc32_u16, _mm_crc32_u32, _mm_crc32_u64, _mm_crc32_u8,
	_mm_loadu_si128, _mm_set1_epi64x, _mm_storeu_si128, _mm_xor_si128,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SSE4.2 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sse42(());

impl Sse42 {
	/// `None` if the CPU (or the compile-time target) lacks SSE4.2.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse42).then_some(Sse42(()))
	}

	/// Accumulates `byte` into the running CRC32-C `crc` (`crc32b`).
	#[inline]
	pub fn crc32_u8(self, crc: u32, byte: u8) -> u32 {
		unsafe { crc32b(crc, byte) }
	}

	/// Accumulates `v`'s 2 little-endian bytes into the running CRC32-C `crc` (`crc32w`).
	#[inline]
	pub fn crc32_u16(self, crc: u32, v: u16) -> u32 {
		unsafe { crc32w(crc, v) }
	}

	/// Accumulates `v`'s 4 little-endian bytes into the running CRC32-C `crc` (`crc32d`).
	#[inline]
	pub fn crc32_u32(self, crc: u32, v: u32) -> u32 {
		unsafe { crc32d(crc, v) }
	}

	/// Accumulates `v`'s 8 little-endian bytes into the running CRC32-C `crc` (`crc32q`).
	#[inline]
	pub fn crc32_u64(self, crc: u64, v: u64) -> u64 {
		unsafe { crc32q(crc, v) }
	}

	/// Lane greater-than mask, `i64` (all-1s if `a>b`, else 0; `pcmpgtq`, native only from SSE4.2).
	#[inline]
	pub fn cmpgt_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		unsafe { pcmpgtq(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub fn cmpgt_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_i64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { -1 } else { 0 };
		}
	}

	/// Lane greater-than mask, `u64`: sign-bit flip + [`cmpgt_i64x2`](Self::cmpgt_i64x2).
	#[inline]
	pub fn cmpgt_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		let ai: [i64; 2] = core::array::from_fn(|i| (a[i] ^ 0x8000_0000_0000_0000) as i64);
		let bi: [i64; 2] = core::array::from_fn(|i| (b[i] ^ 0x8000_0000_0000_0000) as i64);
		let r = self.cmpgt_i64x2(ai, bi);
		core::array::from_fn(|i| r[i] as u64)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub fn cmpgt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i64x2`].
	#[inline]
	pub fn cmplt_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		self.cmpgt_i64x2(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Sse42::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i64x2`].
	#[inline]
	pub fn cmple_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		let gt = self.cmpgt_i64x2(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Sse42::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i64x2`].
	#[inline]
	pub fn cmpge_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		let lt = self.cmplt_i64x2(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Sse42::cmplt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmplt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u64x2`].
	#[inline]
	pub fn cmplt_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		self.cmpgt_u64x2(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmpgt_u64_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u64x2`].
	#[inline]
	pub fn cmple_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		let gt = self.cmpgt_u64x2(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmpgt_u64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u64x2`].
	#[inline]
	pub fn cmpge_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		let lt = self.cmplt_u64x2(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmplt_u64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Per-lane signed min (`pcmpgtq` + `pblendvb`; no native `pminsq` below AVX-512F).
	#[inline]
	pub fn min_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		unsafe { min_i64x2_composed(&a, &b) }
	}

	/// `out[i] = min(a[i], b[i])`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_i64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane signed max (`pcmpgtq` + `pblendvb`; no native `pmaxsq` below AVX-512F).
	#[inline]
	pub fn max_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		unsafe { max_i64x2_composed(&a, &b) }
	}

	/// `out[i] = max(a[i], b[i])`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_i64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}

	/// Per-lane unsigned min: sign-bit-flip compare + [`min_i64x2`](Self::min_i64x2)'s blend, on the
	/// original unflipped values.
	#[inline]
	pub fn min_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		unsafe { min_u64x2_composed(&a, &b) }
	}

	/// `out[i] = min(a[i], b[i])` (`u64`). 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_u64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane unsigned max: sign-bit-flip compare + [`max_i64x2`](Self::max_i64x2)'s blend, on the
	/// original unflipped values.
	#[inline]
	pub fn max_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		unsafe { max_u64x2_composed(&a, &b) }
	}

	/// `out[i] = max(a[i], b[i])` (`u64`). 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let b_chunks = b.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 2] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 2] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_u64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}
}

/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32b(crc: u32, byte: u8) -> u32 {
	_mm_crc32_u8(crc, byte)
}

/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32w(crc: u32, v: u16) -> u32 {
	_mm_crc32_u16(crc, v)
}

/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32d(crc: u32, v: u32) -> u32 {
	_mm_crc32_u32(crc, v)
}

/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32q(crc: u64, v: u64) -> u64 {
	_mm_crc32_u64(crc, v)
}

/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn pcmpgtq(a: &[i64; 2], b: &[i64; 2]) -> [i64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_cmpgt_epi64(va, vb);
		let mut out = [0i64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// `min(a,b) = blendv(a, b, cmpgt(a,b))`: pick `b` (the smaller value) where `a>b`, else keep `a`.
///
/// # Safety
/// Caller proved SSE4.2 via [`Sse42`] (implies SSE4.1's `pblendvb`).
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn min_i64x2_composed(a: &[i64; 2], b: &[i64; 2]) -> [i64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let gt = _mm_cmpgt_epi64(va, vb);
		let vr = _mm_blendv_epi8(va, vb, gt);
		let mut out = [0i64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// `max(a,b) = blendv(a, b, cmpgt(b,a))`: pick `b` (the larger value) where `b>a`, else keep `a`.
///
/// # Safety
/// Caller proved SSE4.2 via [`Sse42`] (implies SSE4.1's `pblendvb`).
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn max_i64x2_composed(a: &[i64; 2], b: &[i64; 2]) -> [i64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let gt = _mm_cmpgt_epi64(vb, va);
		let vr = _mm_blendv_epi8(va, vb, gt);
		let mut out = [0i64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Unsigned min implemented by flipping sign bits before a signed compare,
/// then blending the original operands on that mask.
///
/// # Safety
/// Caller proved SSE4.2 via [`Sse42`].
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn min_u64x2_composed(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let sign = _mm_set1_epi64x(i64::MIN);
		let af = _mm_xor_si128(va, sign);
		let bf = _mm_xor_si128(vb, sign);
		let gt = _mm_cmpgt_epi64(af, bf);
		let vr = _mm_blendv_epi8(va, vb, gt);
		let mut out = [0u64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Unsigned max: mirror of [`min_u64x2_composed`] (operand-swapped compare).
///
/// # Safety
/// Caller proved SSE4.2 via [`Sse42`] (implies SSE4.1's `pblendvb`).
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn max_u64x2_composed(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let sign = _mm_set1_epi64x(i64::MIN);
		let af = _mm_xor_si128(va, sign);
		let bf = _mm_xor_si128(vb, sign);
		let gt = _mm_cmpgt_epi64(bf, af);
		let vr = _mm_blendv_epi8(va, vb, gt);
		let mut out = [0u64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/sse/sse42.rs"]
mod tests;
