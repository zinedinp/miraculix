//! AVX512BF16: BF16 dot-product accumulate and `f32`<->BF16 conversions.
//! 512-bit forms require `avx512bf16` + `avx512f`. 128/256-bit VL forms are not built here.
//! Merge/zero-masked versions exist for every op.

use core::arch::x86_64::{
	__m256bh, __m256i, __m512, __m512bh, __m512i, _mm256_loadu_si256, _mm512_cvtne2ps_pbh,
	_mm512_cvtneps_pbh, _mm512_cvtpbh_ps, _mm512_dpbf16_ps, _mm512_loadu_ps, _mm512_loadu_si512,
	_mm512_mask_cvtne2ps_pbh, _mm512_mask_cvtneps_pbh, _mm512_mask_cvtpbh_ps, _mm512_mask_dpbf16_ps,
	_mm512_maskz_cvtne2ps_pbh, _mm512_maskz_cvtneps_pbh, _mm512_maskz_cvtpbh_ps, _mm512_maskz_dpbf16_ps,
	_mm512_storeu_ps,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: AVX512BF16 *and* AVX-512F, both required for the 512-bit
/// forms (see module doc). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Bf16(());

impl Avx512Bf16 {
	/// `None` unless the CPU has both AVX512BF16 and AVX-512F.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Avx512bf16) && set.contains(Feature::Avx512f)).then_some(Avx512Bf16(()))
	}

	/// `dst[j] = src[j] + f32(a[2j+1])*f32(b[2j+1]) + f32(a[2j])*f32(b[2j])`,
	/// each product/accumulate a real `f32` op (`vdpbf16ps`, 512-bit). `a`/
	/// `b` are 32 BF16 bit patterns (16 pairs); `src`/result are 16 `f32`.
	#[inline]
	pub fn dpbf16_ps_f32x16(self, src: [f32; 16], a: [u16; 32], b: [u16; 32]) -> [f32; 16] {
		unsafe { dpbf16_ps_f32x16_intrinsic(&src, &a, &b) }
	}

	/// 16 `f32` -> 16 BF16 bit patterns, round-to-nearest-even
	/// (`vcvtneps2bf16`, 512-bit source, 256-bit BF16 result).
	#[inline]
	pub fn cvtneps_pbh_u16x16(self, a: [f32; 16]) -> [u16; 16] {
		unsafe { cvtneps_pbh_u16x16_intrinsic(&a) }
	}

	/// Two 16-`f32` vectors -> 32 BF16 bit patterns, round-to-nearest-even;
	/// `b`'s lanes land in the low half, `a`'s in the high half
	/// (`vcvtne2ps2bf16`, 512-bit).
	#[inline]
	pub fn cvtne2ps_pbh_u16x32(self, a: [f32; 16], b: [f32; 16]) -> [u16; 32] {
		unsafe { cvtne2ps_pbh_u16x32_intrinsic(&a, &b) }
	}

	/// 16 BF16 bit patterns -> 16 `f32`, exact (BF16 is `f32`'s top 16 bits,
	/// zero-extended: same value [`bf16_to_f32_scalar`] computes in software)
	/// (`vcvtpbh2ps`, 256-bit BF16 source, 512-bit result).
	#[inline]
	pub fn cvtpbh_ps_f32x16(self, a: [u16; 16]) -> [f32; 16] {
		unsafe { cvtpbh_ps_f32x16_intrinsic(&a) }
	}

	/// [`Avx512Bf16::dpbf16_ps_f32x16`] where `mask` bit is set, else copied
	/// from `src` (`vdpbf16ps`, merge-masked). `src` is a real input here, not
	/// just a merge fallback: the dot-product-accumulate is computed for
	/// every lane before masking zeroes the unselected ones, same as
	/// AVX512VNNI's `dpbusd`.
	#[inline]
	pub fn dpbf16_ps_f32x16_merge_masked(self, src: [f32; 16], mask: u16, a: [u16; 32], b: [u16; 32]) -> [f32; 16] {
		unsafe { mask_dpbf16_ps_f32x16_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16::dpbf16_ps_f32x16`] where `mask` bit is set, else zero
	/// (`vdpbf16ps`, zero-masked).
	#[inline]
	pub fn dpbf16_ps_f32x16_zero_masked(self, mask: u16, src: [f32; 16], a: [u16; 32], b: [u16; 32]) -> [f32; 16] {
		unsafe { maskz_dpbf16_ps_f32x16_intrinsic(mask, &src, &a, &b) }
	}

	/// [`Avx512Bf16::cvtneps_pbh_u16x16`] where `mask` bit is set, else copied
	/// from `src` (`vcvtneps2bf16`, merge-masked).
	#[inline]
	pub fn cvtneps_pbh_u16x16_merge_masked(self, src: [u16; 16], mask: u16, a: [f32; 16]) -> [u16; 16] {
		unsafe { mask_cvtneps_pbh_u16x16_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16::cvtneps_pbh_u16x16`] where `mask` bit is set, else zero
	/// (`vcvtneps2bf16`, zero-masked).
	#[inline]
	pub fn cvtneps_pbh_u16x16_zero_masked(self, mask: u16, a: [f32; 16]) -> [u16; 16] {
		unsafe { maskz_cvtneps_pbh_u16x16_intrinsic(mask, &a) }
	}

	/// [`Avx512Bf16::cvtne2ps_pbh_u16x32`] where `mask` bit is set, else
	/// copied from `src` (`vcvtne2ps2bf16`, merge-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x32_merge_masked(self, src: [u16; 32], mask: u32, a: [f32; 16], b: [f32; 16]) -> [u16; 32] {
		unsafe { mask_cvtne2ps_pbh_u16x32_intrinsic(&src, mask, &a, &b) }
	}

	/// [`Avx512Bf16::cvtne2ps_pbh_u16x32`] where `mask` bit is set, else zero
	/// (`vcvtne2ps2bf16`, zero-masked).
	#[inline]
	pub fn cvtne2ps_pbh_u16x32_zero_masked(self, mask: u32, a: [f32; 16], b: [f32; 16]) -> [u16; 32] {
		unsafe { maskz_cvtne2ps_pbh_u16x32_intrinsic(mask, &a, &b) }
	}

	/// [`Avx512Bf16::cvtpbh_ps_f32x16`] where `mask` bit is set, else copied
	/// from `src` (`vcvtpbh2ps`, merge-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x16_merge_masked(self, src: [f32; 16], mask: u16, a: [u16; 16]) -> [f32; 16] {
		unsafe { mask_cvtpbh_ps_f32x16_intrinsic(&src, mask, &a) }
	}

	/// [`Avx512Bf16::cvtpbh_ps_f32x16`] where `mask` bit is set, else zero
	/// (`vcvtpbh2ps`, zero-masked).
	#[inline]
	pub fn cvtpbh_ps_f32x16_zero_masked(self, mask: u16, a: [u16; 16]) -> [f32; 16] {
		unsafe { maskz_cvtpbh_ps_f32x16_intrinsic(mask, &a) }
	}

	/// `out[j] = src[j] + bf16(a[2j+1])*bf16(b[2j+1]) + bf16(a[2j])*bf16(b[2j])`.
	/// 16-wide `src`/`out` chunks (32-wide `a`/`b`), software scalar rem.
	///
	/// # Panics
	/// `a.len() != b.len() || a.len() != 2*src.len() || out.len() != src.len()`.
	pub fn dpbf16_ps_f32_slice(self, src: &[f32], a: &[u16], b: &[u16], out: &mut [f32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), 2 * src.len());
		assert_eq!(out.len(), src.len());

		let src_chunks = src.chunks_exact(16);
		let src_rem = src_chunks.remainder();
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_chunks = b.chunks_exact(32);
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);

		for (((sc, ac), bc), oc) in src_chunks.zip(a_chunks).zip(b_chunks).zip(out_chunks.by_ref()) {
			let sv: [f32; 16] = sc.try_into().expect("chunks_exact width");
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.dpbf16_ps_f32x16(sv, av, bv));
		}
		for (j, o) in out_chunks.into_remainder().iter_mut().enumerate() {
			let mut acc = src_rem[j];
			acc += bf16_to_f32_scalar(a_rem[2 * j + 1]) * bf16_to_f32_scalar(b_rem[2 * j + 1]);
			acc += bf16_to_f32_scalar(a_rem[2 * j]) * bf16_to_f32_scalar(b_rem[2 * j]);
			*o = acc;
		}
	}

	/// `out[i] = f32_to_bf16(a[i])`, RNE. 16-wide chunks, software scalar rem.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn cvtneps_pbh_u16_slice(self, a: &[f32], out: &mut [u16]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cvtneps_pbh_u16x16(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f32_to_bf16_scalar(x);
		}
	}

	/// `out[j] = f32_to_bf16(b[j])` for `j < a.len()`, else `f32_to_bf16(a[j-a.len()])`.
	/// Low half is `b`, high half is `a`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != 2*a.len()`.
	pub fn cvtne2ps_pbh_u16_slice(self, a: &[f32], b: &[f32], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), 2 * a.len());

		let n = a.len();
		let (out_lo, out_hi) = out.split_at_mut(n);
		let full_chunks = n / 16;

		for i in 0..full_chunks {
			let av: [f32; 16] = a[i * 16..i * 16 + 16].try_into().expect("chunk width");
			let bv: [f32; 16] = b[i * 16..i * 16 + 16].try_into().expect("chunk width");
			let r = self.cvtne2ps_pbh_u16x32(av, bv);
			out_lo[i * 16..i * 16 + 16].copy_from_slice(&r[0..16]);
			out_hi[i * 16..i * 16 + 16].copy_from_slice(&r[16..32]);
		}
		for i in full_chunks * 16..n {
			out_lo[i] = f32_to_bf16_scalar(b[i]);
			out_hi[i] = f32_to_bf16_scalar(a[i]);
		}
	}
}

/// IEEE 754 binary32 -> bfloat16, RNE (matches HW default rounding mode,
/// standard bias-then-truncate formula). NaN forces a nonzero mantissa bit
/// to stay a NaN after truncation.
pub(crate) fn f32_to_bf16_scalar(x: f32) -> u16 {
	if x.is_nan() {
		return ((x.to_bits() >> 16) as u16) | 0x0040;
	}
	let bits = x.to_bits();
	let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
	((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// bfloat16 -> `f32`, exact (bfloat16 is `f32`'s top 16 bits; zero-extend).
pub(crate) fn bf16_to_f32_scalar(bits: u16) -> f32 {
	f32::from_bits((bits as u32) << 16)
}

#[inline]
unsafe fn load_bf16x32(a: &[u16; 32]) -> __m512bh {
	unsafe {
		let v: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		core::mem::transmute::<__m512i, __m512bh>(v)
	}
}

#[inline]
unsafe fn store_bf16x32(v: __m512bh) -> [u16; 32] {
	unsafe { core::mem::transmute::<__m512bh, [u16; 32]>(v) }
}

#[inline]
unsafe fn load_bf16x16(a: &[u16; 16]) -> __m256bh {
	unsafe {
		let v: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		core::mem::transmute::<__m256i, __m256bh>(v)
	}
}

#[inline]
unsafe fn store_bf16x16(v: __m256bh) -> [u16; 16] {
	unsafe { core::mem::transmute::<__m256bh, [u16; 16]>(v) }
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn dpbf16_ps_f32x16_intrinsic(src: &[f32; 16], a: &[u16; 32], b: &[u16; 32]) -> [f32; 16] {
	unsafe {
		let vsrc: __m512 = _mm512_loadu_ps(src.as_ptr());
		let va = load_bf16x32(a);
		let vb = load_bf16x32(b);
		let vr = _mm512_dpbf16_ps(vsrc, va, vb);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn cvtneps_pbh_u16x16_intrinsic(a: &[f32; 16]) -> [u16; 16] {
	unsafe {
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vr: __m256bh = _mm512_cvtneps_pbh(va);
		core::mem::transmute::<__m256bh, [u16; 16]>(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn cvtne2ps_pbh_u16x32_intrinsic(a: &[f32; 16], b: &[f32; 16]) -> [u16; 32] {
	unsafe {
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vb: __m512 = _mm512_loadu_ps(b.as_ptr());
		let vr = _mm512_cvtne2ps_pbh(va, vb);
		store_bf16x32(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn cvtpbh_ps_f32x16_intrinsic(a: &[u16; 16]) -> [f32; 16] {
	unsafe {
		let va = load_bf16x16(a);
		let vr = _mm512_cvtpbh_ps(va);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

// Merge/zero-masked forms below, one pair per op above. `dpbf16_ps`'s zero
// form still takes `src` as a real input (not just a merge fallback): same
// reasoning as AVX512VNNI's `dpbusd`: the accumulate happens for every lane
// before masking zeroes the unselected ones.

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn mask_dpbf16_ps_f32x16_intrinsic(src: &[f32; 16], mask: u16, a: &[u16; 32], b: &[u16; 32]) -> [f32; 16] {
	unsafe {
		let vsrc: __m512 = _mm512_loadu_ps(src.as_ptr());
		let va = load_bf16x32(a);
		let vb = load_bf16x32(b);
		let vr = _mm512_mask_dpbf16_ps(vsrc, mask, va, vb);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn maskz_dpbf16_ps_f32x16_intrinsic(mask: u16, src: &[f32; 16], a: &[u16; 32], b: &[u16; 32]) -> [f32; 16] {
	unsafe {
		let vsrc: __m512 = _mm512_loadu_ps(src.as_ptr());
		let va = load_bf16x32(a);
		let vb = load_bf16x32(b);
		let vr = _mm512_maskz_dpbf16_ps(mask, vsrc, va, vb);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn mask_cvtneps_pbh_u16x16_intrinsic(src: &[u16; 16], mask: u16, a: &[f32; 16]) -> [u16; 16] {
	unsafe {
		let vsrc = load_bf16x16(src);
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vr = _mm512_mask_cvtneps_pbh(vsrc, mask, va);
		store_bf16x16(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn maskz_cvtneps_pbh_u16x16_intrinsic(mask: u16, a: &[f32; 16]) -> [u16; 16] {
	unsafe {
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vr = _mm512_maskz_cvtneps_pbh(mask, va);
		store_bf16x16(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn mask_cvtne2ps_pbh_u16x32_intrinsic(src: &[u16; 32], mask: u32, a: &[f32; 16], b: &[f32; 16]) -> [u16; 32] {
	unsafe {
		let vsrc = load_bf16x32(src);
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vb: __m512 = _mm512_loadu_ps(b.as_ptr());
		let vr = _mm512_mask_cvtne2ps_pbh(vsrc, mask, va, vb);
		store_bf16x32(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn maskz_cvtne2ps_pbh_u16x32_intrinsic(mask: u32, a: &[f32; 16], b: &[f32; 16]) -> [u16; 32] {
	unsafe {
		let va: __m512 = _mm512_loadu_ps(a.as_ptr());
		let vb: __m512 = _mm512_loadu_ps(b.as_ptr());
		let vr = _mm512_maskz_cvtne2ps_pbh(mask, va, vb);
		store_bf16x32(vr)
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn mask_cvtpbh_ps_f32x16_intrinsic(src: &[f32; 16], mask: u16, a: &[u16; 16]) -> [f32; 16] {
	unsafe {
		let vsrc: __m512 = _mm512_loadu_ps(src.as_ptr());
		let va = load_bf16x16(a);
		let vr = _mm512_mask_cvtpbh_ps(vsrc, mask, va);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX512BF16 + AVX-512F via [`Avx512Bf16`].
#[inline]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn maskz_cvtpbh_ps_f32x16_intrinsic(mask: u16, a: &[u16; 16]) -> [f32; 16] {
	unsafe {
		let va = load_bf16x16(a);
		let vr = _mm512_maskz_cvtpbh_ps(mask, va);
		let mut out = [0.0f32; 16];
		_mm512_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512bf16.rs"]
mod tests;
