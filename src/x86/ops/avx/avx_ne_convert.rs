//! AVX-NE-CONVERT: no-exception BF16/FP16 converts (`avxneconvert`). Token:
//! [`AvxNeConvert::detect`]. Provides deinterleave helpers and conversion
//! intrinsics for BF16/FP16 to `f32`.

use core::arch::x86_64::{
	__m128, __m128bh, __m128h, __m256, __m256bh, __m256h, _mm256_cvtneebf16_ps, _mm256_cvtneeph_ps,
	_mm256_cvtneobf16_ps, _mm256_cvtneoph_ps, _mm256_cvtneps_avx_pbh, _mm256_loadu_ps, _mm256_storeu_ps,
	_mm_cvtneebf16_ps, _mm_cvtneeph_ps, _mm_cvtneobf16_ps, _mm_cvtneoph_ps, _mm_cvtneps_avx_pbh, _mm_loadu_ps,
	_mm_storeu_ps,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: AVX-NE-CONVERT available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct AvxNeConvert(());

impl AvxNeConvert {
	/// `None` if the CPU (or the compile-time target) lacks AVX-NE-CONVERT.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::AvxNeConvert).then_some(AvxNeConvert(()))
	}
}

/// Deinterleave: `a` is `2*width` half bits, `dst[j] = to_f32(a[2*j+offset])`.
macro_rules! ne_convert_deinterleave_unop {
	(
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, offset = $offset:literal,
		half_vec = $HalfVec:ty, vec = $Vec:ty, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl AvxNeConvert {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: &[u16; $width * 2]) -> [f32; $width] {
				unsafe { $intrinsic_fn(a) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != out.len() * 2`.
			pub fn $slice_fn(self, a: &[u16], out: &mut [f32]) {
				assert_eq!(a.len(), out.len() * 2);

				let a_chunks = a.chunks_exact($width * 2);
				let a_rem = a_chunks.remainder();
				let mut out_chunks = out.chunks_exact_mut($width);

				for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
					let av: [u16; $width * 2] = ac.try_into().expect("chunks_exact width");
					oc.copy_from_slice(&self.$fixed_fn(&av));
				}
				let scalar: fn(u16) -> f32 = $scalar;
				for (rc, o) in a_rem.chunks_exact(2).zip(out_chunks.into_remainder()) {
					*o = scalar(rc[$offset]);
				}
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "avxneconvert")]
		unsafe fn $intrinsic_fn(a: &[u16; $width * 2]) -> [f32; $width] {
			unsafe {
				let vr: $Vec = $intrinsic(a.as_ptr().cast::<$HalfVec>());
				let mut out = [0f32; $width];
				$storeu(out.as_mut_ptr(), vr);
				out
			}
		}
	};
}

ne_convert_deinterleave_unop! {
	fixed_fn = cvtneebf16_ps_x4, slice_fn = cvtneebf16_ps_slice, intrinsic_fn = cvtneebf16_ps_x4_intrinsic,
	width = 4, offset = 0,
	half_vec = __m128bh, vec = __m128, storeu = _mm_storeu_ps, intrinsic = _mm_cvtneebf16_ps,
	scalar = bf16_to_f32_scalar,
	fixed_doc = "Even-indexed `bf16` bits (`a[0], a[2], a[4], a[6]`) -> `f32` (`vcvtneebf162ps`, 128-bit).",
	slice_doc = "`out[j] = bf16_to_f32(a[2*j])`. 8-wide `a` / 4-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneebf16_ps_x8, slice_fn = cvtneebf16_ps_slice_wide, intrinsic_fn = cvtneebf16_ps_x8_intrinsic,
	width = 8, offset = 0,
	half_vec = __m256bh, vec = __m256, storeu = _mm256_storeu_ps, intrinsic = _mm256_cvtneebf16_ps,
	scalar = bf16_to_f32_scalar,
	fixed_doc = "Even-indexed `bf16` bits -> `f32` (`vcvtneebf162ps`, 256-bit).",
	slice_doc = "`out[j] = bf16_to_f32(a[2*j])`. 16-wide `a` / 8-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneobf16_ps_x4, slice_fn = cvtneobf16_ps_slice, intrinsic_fn = cvtneobf16_ps_x4_intrinsic,
	width = 4, offset = 1,
	half_vec = __m128bh, vec = __m128, storeu = _mm_storeu_ps, intrinsic = _mm_cvtneobf16_ps,
	scalar = bf16_to_f32_scalar,
	fixed_doc = "Odd-indexed `bf16` bits (`a[1], a[3], a[5], a[7]`) -> `f32` (`vcvtneobf162ps`, 128-bit).",
	slice_doc = "`out[j] = bf16_to_f32(a[2*j+1])`. 8-wide `a` / 4-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneobf16_ps_x8, slice_fn = cvtneobf16_ps_slice_wide, intrinsic_fn = cvtneobf16_ps_x8_intrinsic,
	width = 8, offset = 1,
	half_vec = __m256bh, vec = __m256, storeu = _mm256_storeu_ps, intrinsic = _mm256_cvtneobf16_ps,
	scalar = bf16_to_f32_scalar,
	fixed_doc = "Odd-indexed `bf16` bits -> `f32` (`vcvtneobf162ps`, 256-bit).",
	slice_doc = "`out[j] = bf16_to_f32(a[2*j+1])`. 16-wide `a` / 8-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneeph_ps_x4, slice_fn = cvtneeph_ps_slice, intrinsic_fn = cvtneeph_ps_x4_intrinsic,
	width = 4, offset = 0,
	half_vec = __m128h, vec = __m128, storeu = _mm_storeu_ps, intrinsic = _mm_cvtneeph_ps,
	scalar = f16_to_f32_scalar,
	fixed_doc = "Even-indexed `f16` bits -> `f32` (`vcvtneeph2ps`, 128-bit).",
	slice_doc = "`out[j] = f16_to_f32(a[2*j])`. 8-wide `a` / 4-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneeph_ps_x8, slice_fn = cvtneeph_ps_slice_wide, intrinsic_fn = cvtneeph_ps_x8_intrinsic,
	width = 8, offset = 0,
	half_vec = __m256h, vec = __m256, storeu = _mm256_storeu_ps, intrinsic = _mm256_cvtneeph_ps,
	scalar = f16_to_f32_scalar,
	fixed_doc = "Even-indexed `f16` bits -> `f32` (`vcvtneeph2ps`, 256-bit).",
	slice_doc = "`out[j] = f16_to_f32(a[2*j])`. 16-wide `a` / 8-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneoph_ps_x4, slice_fn = cvtneoph_ps_slice, intrinsic_fn = cvtneoph_ps_x4_intrinsic,
	width = 4, offset = 1,
	half_vec = __m128h, vec = __m128, storeu = _mm_storeu_ps, intrinsic = _mm_cvtneoph_ps,
	scalar = f16_to_f32_scalar,
	fixed_doc = "Odd-indexed `f16` bits -> `f32` (`vcvtneoph2ps`, 128-bit).",
	slice_doc = "`out[j] = f16_to_f32(a[2*j+1])`. 8-wide `a` / 4-wide `out` chunks, software scalar rem.",
}
ne_convert_deinterleave_unop! {
	fixed_fn = cvtneoph_ps_x8, slice_fn = cvtneoph_ps_slice_wide, intrinsic_fn = cvtneoph_ps_x8_intrinsic,
	width = 8, offset = 1,
	half_vec = __m256h, vec = __m256, storeu = _mm256_storeu_ps, intrinsic = _mm256_cvtneoph_ps,
	scalar = f16_to_f32_scalar,
	fixed_doc = "Odd-indexed `f16` bits -> `f32` (`vcvtneoph2ps`, 256-bit).",
	slice_doc = "`out[j] = f16_to_f32(a[2*j+1])`. 16-wide `a` / 8-wide `out` chunks, software scalar rem.",
}

impl AvxNeConvert {
	/// `f32` -> `bf16` bits, RNE (`vcvtneps2bf16`, 128-bit in). Low 4 of 8-lane
	/// `__m128bh` result (upper 4 are padding; not exposed).
	#[inline]
	pub fn cvtneps_avx_pbh_x4(self, a: [f32; 4]) -> [u16; 4] {
		unsafe { cvtneps_avx_pbh_x4_intrinsic(&a) }
	}

	/// `f32` -> `bf16` bits, RNE (`vcvtneps2bf16`, 256-bit in, all 8 lanes).
	#[inline]
	pub fn cvtneps_avx_pbh_x8(self, a: [f32; 8]) -> [u16; 8] {
		unsafe { cvtneps_avx_pbh_x8_intrinsic(&a) }
	}

	/// `out[i] = f32_to_bf16(a[i])`. 8-wide chunks, software scalar rem.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn cvtneps_avx_pbh_slice(self, a: &[f32], out: &mut [u16]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [f32; 8] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cvtneps_avx_pbh_x8(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f32_to_bf16_scalar(x);
		}
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avxneconvert")]
unsafe fn cvtneps_avx_pbh_x4_intrinsic(a: &[f32; 4]) -> [u16; 4] {
	unsafe {
		let va = _mm_loadu_ps(a.as_ptr());
		let vr: __m128bh = _mm_cvtneps_avx_pbh(va);
		let mut padded = [0u16; 8];
		core::ptr::copy_nonoverlapping((&raw const vr).cast::<u16>(), padded.as_mut_ptr(), 8);
		let mut out = [0u16; 4];
		out.copy_from_slice(&padded[..4]);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "avxneconvert")]
unsafe fn cvtneps_avx_pbh_x8_intrinsic(a: &[f32; 8]) -> [u16; 8] {
	unsafe {
		let va = _mm256_loadu_ps(a.as_ptr());
		let vr: __m128bh = _mm256_cvtneps_avx_pbh(va);
		let mut out = [0u16; 8];
		core::ptr::copy_nonoverlapping((&raw const vr).cast::<u16>(), out.as_mut_ptr(), 8);
		out
	}
}

/// binary16 -> binary32, exact. Duplicated from `f16c` (file is self-contained).
fn f16_to_f32_scalar(h: u16) -> f32 {
	let sign = ((h >> 15) & 1) as u32;
	let exp = ((h >> 10) & 0x1f) as u32;
	let mant = (h & 0x3ff) as u32;

	let (f_exp, f_mant) = if exp == 0 {
		if mant == 0 {
			(0u32, 0u32)
		} else {
			let mut mant = mant;
			let mut e = 0i32;
			while mant & 0x400 == 0 {
				mant <<= 1;
				e -= 1;
			}
			mant &= 0x3ff;
			(((127 - 15 + 1 + e) as u32), mant << 13)
		}
	} else if exp == 0x1f {
		(0xff, mant << 13)
	} else {
		(exp + 127 - 15, mant << 13)
	};

	f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
}

/// bf16 -> f32, exact (zero-extend low 16; no subnormal renormalize).
fn bf16_to_f32_scalar(h: u16) -> f32 {
	f32::from_bits((h as u32) << 16)
}

/// f32 -> bf16 bits, RNE (bias `0x7fff` + sticky LSB; ties to even).
fn f32_to_bf16_scalar(f: f32) -> u16 {
	let bits = f.to_bits();
	if f.is_nan() {
		// Keep NaN; force top mant bit so truncate can't yield non-NaN zero.
		return ((bits >> 16) as u16) | 0x0040;
	}
	let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
	(bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx_ne_convert.rs"]
mod tests;
