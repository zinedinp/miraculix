//! F16C (2012): half-precision bit conversions `u16`<->`f32`.
//! Token: [`F16c::detect`] / [`F16c::from_level`]. Hand-written cross-type support.
//! Scalar fallback uses rounded-to-nearest-even, and `auto_up` cascades to the scalar routine directly.

use core::arch::x86_64::{
	__m128i, _mm256_cvtph_ps, _mm256_cvtps_ph, _mm_cvtph_ps, _mm_cvtps_ph, _mm_loadu_si128, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};

/// Proof token: F16C available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct F16c(());

impl F16c {
	/// `None` if the CPU (or the compile-time target) lacks F16C.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::F16c).then_some(F16c(()))
	}

	/// From resolved tier (`V3`/`V4` list `Feature::F16c`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::F16c).then_some(F16c(()))
	}
}

impl F16c {
	/// 4 half-float bit patterns to `f32` (`vcvtph2ps`, 128-bit).
	#[inline]
	pub fn f16_to_f32x4(self, a: [u16; 4]) -> [f32; 4] {
		unsafe { cvtph2ps_x4(&a) }
	}

	/// 8 half-float bit patterns to `f32` (`vcvtph2ps`, 256-bit).
	#[inline]
	pub fn f16_to_f32x8(self, a: [u16; 8]) -> [f32; 8] {
		unsafe { cvtph2ps_x8(&a) }
	}

	/// 4 `f32` to half-float bits (`vcvtps2ph`, 128-bit). `ROUNDING`:
	/// `_MM_FROUND_TO_*` or `_MM_FROUND_CUR_DIRECTION` (no `_MM_FROUND_NO_EXC`).
	#[inline]
	pub fn f32_to_f16x4<const ROUNDING: i32>(self, a: [f32; 4]) -> [u16; 4] {
		unsafe { cvtps2ph_x4::<ROUNDING>(&a) }
	}

	/// 8 `f32` to half-float bits (`vcvtps2ph`, 256-bit). Same `ROUNDING`
	/// as [`Self::f32_to_f16x4`].
	#[inline]
	pub fn f32_to_f16x8<const ROUNDING: i32>(self, a: [f32; 8]) -> [u16; 8] {
		unsafe { cvtps2ph_x8::<ROUNDING>(&a) }
	}

	/// `out[i] = f16_to_f32(a[i])`. 8-wide chunks, software scalar rem.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn f16_to_f32_slice(self, a: &[u16], out: &mut [f32]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u16; 8] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.f16_to_f32x8(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f16_to_f32_scalar(x);
		}
	}

	/// `out[i] = f32_to_f16(a[i])`. Vector chunks use `ROUNDING`; scalar
	/// rem always RNE (ignores `ROUNDING`).
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn f32_to_f16_slice<const ROUNDING: i32>(self, a: &[f32], out: &mut [u16]) {
		assert_eq!(out.len(), a.len());

		let a_chunks = a.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);

		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [f32; 8] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.f32_to_f16x8::<ROUNDING>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = f32_to_f16_scalar(x);
		}
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn cvtph2ps_x4(a: &[u16; 4]) -> [f32; 4] {
	unsafe {
		let mut padded = [0u16; 8];
		padded[..4].copy_from_slice(a);
		let va: __m128i = _mm_loadu_si128(padded.as_ptr().cast());
		let vr = _mm_cvtph_ps(va);
		let mut out = [0f32; 4];
		core::arch::x86_64::_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn cvtph2ps_x8(a: &[u16; 8]) -> [f32; 8] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vr = _mm256_cvtph_ps(va);
		let mut out = [0f32; 8];
		core::arch::x86_64::_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn cvtps2ph_x4<const ROUNDING: i32>(a: &[f32; 4]) -> [u16; 4] {
	unsafe {
		let va = core::arch::x86_64::_mm_loadu_ps(a.as_ptr());
		let vr: __m128i = _mm_cvtps_ph::<ROUNDING>(va);
		let mut padded = [0u16; 8];
		_mm_storeu_si128(padded.as_mut_ptr().cast(), vr);
		let mut out = [0u16; 4];
		out.copy_from_slice(&padded[..4]);
		out
	}
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn cvtps2ph_x8<const ROUNDING: i32>(a: &[f32; 8]) -> [u16; 8] {
	unsafe {
		let va = core::arch::x86_64::_mm256_loadu_ps(a.as_ptr());
		let vr: __m128i = _mm256_cvtps_ph::<ROUNDING>(va);
		let mut out = [0u16; 8];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// IEEE 754 binary16 -> binary32 (exact; every half is a float).
pub(crate) fn f16_to_f32_scalar(h: u16) -> f32 {
	let sign = ((h >> 15) & 1) as u32;
	let exp = ((h >> 10) & 0x1f) as u32;
	let mant = (h & 0x3ff) as u32;

	let (f_exp, f_mant) = if exp == 0 {
		if mant == 0 {
			(0u32, 0u32)
		} else {
			// Subnormal half: renormalize mantissa, adjust exp.
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
		// `exp + 127 - 15`: `exp - 15` as u32 underflows for exp=1.
		(exp + 127 - 15, mant << 13)
	};

	f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
}

/// IEEE 754 binary32 -> binary16, RNE (matches default HW mode). Overflow
/// -> +-inf; deep underflow -> signed zero/subnormal.
pub(crate) fn f32_to_f16_scalar(f: f32) -> u16 {
	let bits = f.to_bits();
	let sign = ((bits >> 31) & 1) as u16;
	let exp = ((bits >> 23) & 0xff) as i32;
	let mant = bits & 0x7f_ffff;

	if exp == 0xff {
		// Inf/NaN: nonzero mant top bit preserves NaN-ness.
		let h_mant = if mant == 0 { 0 } else { 0x200 };
		return (sign << 15) | 0x7c00 | h_mant;
	}

	let unbiased = exp - 127;
	let h_exp = unbiased + 15;

	if h_exp >= 0x1f {
		return (sign << 15) | 0x7c00;
	}

	if h_exp <= 0 {
		if h_exp < -10 {
			return sign << 15;
		}
		// Subnormal half: shift implicit-1 mant, RNE on discarded bits.
		let full_mant = mant | 0x80_0000;
		let shift = 14 - h_exp;
		let half_mant = round_shift_right_even(full_mant, shift as u32);
		return (sign << 15) | half_mant as u16;
	}

	// Normal: 23-bit mant -> 10 bits, RNE on the 13 discarded.
	let half_mant = round_shift_right_even(mant, 13);
	if half_mant & 0x400 != 0 {
		// Rounded into implicit bit: bump exp.
		return (sign << 15) | (((h_exp + 1) as u16) << 10);
	}
	(sign << 15) | ((h_exp as u16) << 10) | (half_mant as u16)
}

/// `value >> shift`, nearest, ties to even.
fn round_shift_right_even(value: u32, shift: u32) -> u32 {
	if shift == 0 {
		return value;
	}
	if shift >= 32 {
		return 0;
	}
	let half = 1u32 << (shift - 1);
	let mask = (1u32 << shift) - 1;
	let truncated = value >> shift;
	let remainder = value & mask;
	if remainder > half || (remainder == half && (truncated & 1) != 0) {
		truncated + 1
	} else {
		truncated
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx/f16c.rs"]
mod tests;
