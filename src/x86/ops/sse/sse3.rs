//! SSE3 (2004): `addsub` (even sub, odd add) and horizontal `hadd`/`hsub`
//! at f32x4/f64x2. Stable `core::arch`. Token: [`Sse3::detect`]. Hand-written
//! fixed-width only (lane-index / pair shape; no natural slice rem).

use core::arch::x86_64::{
	__m128, __m128d, _mm_add_pd, _mm_add_ps, _mm_addsub_pd, _mm_addsub_ps, _mm_hadd_pd, _mm_hadd_ps, _mm_hsub_pd,
	_mm_hsub_ps, _mm_loadu_pd, _mm_loadu_ps, _mm_mul_pd, _mm_mul_ps, _mm_moveldup_ps, _mm_movehdup_ps, _mm_shuffle_pd,
	_mm_shuffle_ps, _mm_storeu_pd, _mm_storeu_ps, _mm_unpackhi_pd, _mm_unpacklo_pd, _mm_xor_pd, _mm_xor_ps,
};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SSE3 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sse3(());

impl Sse3 {
	/// `None` if the CPU (or the compile-time target) lacks SSE3.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse3).then_some(Sse3(()))
	}

	/// Even lanes sub, odd lanes add (`addsubps`).
	#[inline]
	pub fn addsub_f32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { addsubps(&a, &b) }
	}

	/// Even lanes sub, odd lanes add, double precision (`addsubpd`).
	#[inline]
	pub fn addsub_f64x2(self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
		unsafe { addsubpd(&a, &b) }
	}

	/// Horizontal add within each input (`haddps`):
	/// `[a[0]+a[1], a[2]+a[3], b[0]+b[1], b[2]+b[3]]`.
	#[inline]
	pub fn hadd_f32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { haddps(&a, &b) }
	}

	/// Horizontal sub within each input (`hsubps`):
	/// `[a[0]-a[1], a[2]-a[3], b[0]-b[1], b[2]-b[3]]`.
	#[inline]
	pub fn hsub_f32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { hsubps(&a, &b) }
	}

	/// Horizontal add within each input, double precision (`haddpd`):
	/// `[a[0]+a[1], b[0]+b[1]]`.
	#[inline]
	pub fn hadd_f64x2(self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
		unsafe { haddpd(&a, &b) }
	}

	/// Horizontal sub within each input, double precision (`hsubpd`):
	/// `[a[0]-a[1], b[0]-b[1]]`.
	#[inline]
	pub fn hsub_f64x2(self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
		unsafe { hsubpd(&a, &b) }
	}
}

// Complex f32/f64, interleaved `[re0, im0, re1, im1, ...]` layout (matches
// `num_complex::Complex`'s memory shape and pulp's `c32s`/`c64s`). All four
// ops share one token even though `abs2`/`conj` alone would compile under
// plain `Sse`: keeping the whole complex API under `Sse3` means "if you can
// do one of these you can do all of them", and SSE3 is universal on any CPU
// worth gating for anyway.
//
// `mul`/`conj_mul` use the classic SSE3 complex-multiply network (Intel
// AP-15 "Complex Multiplication"): broadcast `a`'s re/im halves via
// `moveldup`/`movehdup`, multiply against `b` and `b` with re/im swapped,
// then `addsub` (even lane sub, odd lane add) recombines them. `conj_mul`
// negates the `movehdup` broadcast first (equivalent to conjugating `a`
// before multiplying, folded into one pass instead of two).
const COMPLEX_SWAP_PAIRS_F32: i32 = 0b10_11_00_01;
const COMPLEX_CONJ_SIGN_F32X4: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
const COMPLEX_CONJ_SIGN_F64X2: [f64; 2] = [0.0, -0.0];
/// `mul_c32x4_intrinsic(conj=true)` negates `movehdup`'s broadcast of `a.im`
/// (present in *both* lanes of a pair), so it needs an all-lanes negation,
/// not the alternating [`COMPLEX_CONJ_SIGN_F32X4`] pattern.
const COMPLEX_NEGATE_ALL_F32X4: [f32; 4] = [-0.0; 4];
const COMPLEX_NEGATE_ALL_F64X2: [f64; 2] = [-0.0; 2];

impl Sse3 {
	/// Negate the imaginary lane of each complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c32x4(self, a: [f32; 4]) -> [f32; 4] {
		unsafe { conj_c32x4_intrinsic(&a) }
	}

	/// Complex multiply per pair: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { mul_c32x4_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b` per pair, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { mul_c32x4_intrinsic(&a, &b, true) }
	}

	/// `|a|^2` per pair, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c32x4(self, a: [f32; 4]) -> [f32; 4] {
		unsafe { abs2_c32x4_intrinsic(&a) }
	}

	/// Negate the imaginary lane of the complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c64x2(self, a: [f64; 2]) -> [f64; 2] {
		unsafe { conj_c64x2_intrinsic(&a) }
	}

	/// Complex multiply: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c64x2(self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
		unsafe { mul_c64x2_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b`, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c64x2(self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
		unsafe { mul_c64x2_intrinsic(&a, &b, true) }
	}

	/// `|a|^2`, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c64x2(self, a: [f64; 2]) -> [f64; 2] {
		unsafe { abs2_c64x2_intrinsic(&a) }
	}
}

/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn conj_c32x4_intrinsic(a: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va = _mm_loadu_ps(a.as_ptr());
		let sign = _mm_loadu_ps(COMPLEX_CONJ_SIGN_F32X4.as_ptr());
		let vr = _mm_xor_ps(va, sign);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `conj` selects the negated-`b` conjugate-multiply variant instead of a
/// separate pre-pass.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn mul_c32x4_intrinsic(a: &[f32; 4], b: &[f32; 4], conj: bool) -> [f32; 4] {
	unsafe {
		let ab = _mm_loadu_ps(a.as_ptr());
		let xy = _mm_loadu_ps(b.as_ptr());
		let yx = _mm_shuffle_ps::<COMPLEX_SWAP_PAIRS_F32>(xy, xy);
		let aa = _mm_moveldup_ps(ab);
		let mut bb = _mm_movehdup_ps(ab);
		if conj {
			let sign = _mm_loadu_ps(COMPLEX_NEGATE_ALL_F32X4.as_ptr());
			bb = _mm_xor_ps(bb, sign);
		}
		let t1 = _mm_mul_ps(aa, xy);
		let t2 = _mm_mul_ps(bb, yx);
		let vr = _mm_addsub_ps(t1, t2);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn abs2_c32x4_intrinsic(a: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va = _mm_loadu_ps(a.as_ptr());
		let sqr = _mm_mul_ps(va, va);
		let sqr_rev = _mm_shuffle_ps::<COMPLEX_SWAP_PAIRS_F32>(sqr, sqr);
		let vr = _mm_add_ps(sqr, sqr_rev);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn conj_c64x2_intrinsic(a: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va = _mm_loadu_pd(a.as_ptr());
		let sign = _mm_loadu_pd(COMPLEX_CONJ_SIGN_F64X2.as_ptr());
		let vr = _mm_xor_pd(va, sign);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn mul_c64x2_intrinsic(a: &[f64; 2], b: &[f64; 2], conj: bool) -> [f64; 2] {
	unsafe {
		let ab = _mm_loadu_pd(a.as_ptr());
		let xy = _mm_loadu_pd(b.as_ptr());
		let yx = _mm_shuffle_pd::<1>(xy, xy);
		let aa = _mm_unpacklo_pd(ab, ab);
		let mut bb = _mm_unpackhi_pd(ab, ab);
		if conj {
			let sign = _mm_loadu_pd(COMPLEX_NEGATE_ALL_F64X2.as_ptr());
			bb = _mm_xor_pd(bb, sign);
		}
		let t1 = _mm_mul_pd(aa, xy);
		let t2 = _mm_mul_pd(bb, yx);
		let vr = _mm_addsub_pd(t1, t2);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn abs2_c64x2_intrinsic(a: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va = _mm_loadu_pd(a.as_ptr());
		let sqr = _mm_mul_pd(va, va);
		let sqr_rev = _mm_shuffle_pd::<1>(sqr, sqr);
		let vr = _mm_add_pd(sqr, sqr_rev);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// `addsubps` via unaligned `movups`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn addsubps(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_addsub_ps(va, vb);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `addsubpd` via unaligned `movupd`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn addsubpd(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va: __m128d = _mm_loadu_pd(a.as_ptr());
		let vb: __m128d = _mm_loadu_pd(b.as_ptr());
		let vr = _mm_addsub_pd(va, vb);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// `haddps` via unaligned `movups`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn haddps(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_hadd_ps(va, vb);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `hsubps` via unaligned `movups`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn hsubps(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vr = _mm_hsub_ps(va, vb);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `haddpd` via unaligned `movupd`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn haddpd(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va: __m128d = _mm_loadu_pd(a.as_ptr());
		let vb: __m128d = _mm_loadu_pd(b.as_ptr());
		let vr = _mm_hadd_pd(va, vb);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// `hsubpd` via unaligned `movupd`.
///
/// # Safety
/// Caller proved SSE3 via [`Sse3`].
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn hsubpd(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va: __m128d = _mm_loadu_pd(a.as_ptr());
		let vb: __m128d = _mm_loadu_pd(b.as_ptr());
		let vr = _mm_hsub_pd(va, vb);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

#[cfg(test)]
#[path = "../../test/ops/sse/sse3.rs"]
mod tests;
