//! SSE4.1: `"sse4.1"`. Token: [`Sse41::detect`].
//! Provides select/blend, i64 cmpeq, i32/u32 mul, native min/max, and fixed-width rounding.

use core::arch::x86_64::{
	__m128, __m128d, __m128i, _mm_blend_epi16, _mm_blendv_epi8, _mm_blendv_pd, _mm_blendv_ps, _mm_cmpeq_epi64,
	_mm_loadu_pd, _mm_loadu_ps, _mm_loadu_si128, _mm_max_epi32, _mm_max_epi8, _mm_max_epu16, _mm_max_epu32,
	_mm_min_epi32, _mm_min_epi8, _mm_min_epu16, _mm_min_epu32, _mm_mullo_epi32, _mm_round_pd, _mm_round_ps,
	_mm_storeu_pd, _mm_storeu_ps, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
#[cfg(feature = "wider-bus-lift")]
use super::super::avx::avx::Avx;
use super::super::macros::{simd_binop, simd_binop_imm_fixed, simd_binop_lifted, simd_ternop};

// Every `Sse41`-token lift below uses `lift_target_feature = "sse4.1,avx"`
// and `lift_proof = Avx` (VEX-encoded SSE4.1, still 128-bit). Same literal
// constraint as `sse2.rs`'s equivalent comment.

/// Proof token: SSE4.1 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sse41(());

impl Sse41 {
	/// `None` if the CPU (or the compile-time target) lacks SSE4.1.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse41).then_some(Sse41(()))
	}

	/// Per-lane select (`blendvps`): `mask` sign bit picks `b`, else `a`.
	#[inline]
	pub fn blend_f32x4(self, a: [f32; 4], b: [f32; 4], mask: [f32; 4]) -> [f32; 4] {
		unsafe { blendvps(&a, &b, &mask) }
	}

	/// Per-lane round to the mode `ROUNDING` selects (`roundps`). `ROUNDING`
	/// is the raw hardware operand: `_MM_FROUND_TO_NEAREST_INT`/
	/// `_MM_FROUND_TO_NEG_INF`/`_MM_FROUND_TO_POS_INF`/`_MM_FROUND_TO_ZERO`/
	/// `_MM_FROUND_CUR_DIRECTION`, optionally `| _MM_FROUND_NO_EXC`.
	#[inline]
	pub fn round_f32x4<const ROUNDING: i32>(self, a: [f32; 4]) -> [f32; 4] {
		unsafe { roundps::<ROUNDING>(&a) }
	}

	/// Per-lane round to the mode `ROUNDING` selects, double precision
	/// (`roundpd`). Same `ROUNDING` operand as [`Sse41::round_f32x4`].
	#[inline]
	pub fn round_f64x2<const ROUNDING: i32>(self, a: [f64; 2]) -> [f64; 2] {
		unsafe { roundpd::<ROUNDING>(&a) }
	}
}

simd_binop_imm_fixed! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = blend_i16x8, intrinsic_fn = blend_epi16_fixed,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blend_epi16,
	fixed_doc = "Per-lane select by compile-time `IMM8` bitmask: lane `i` from `b` if bit `i` set, else `a` (`pblendw`).",
}

/// `blendvps` via unaligned `movups`.
///
/// # Safety
/// Caller proved SSE4.1 via [`Sse41`].
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn blendvps(a: &[f32; 4], b: &[f32; 4], mask: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vb: __m128 = _mm_loadu_ps(b.as_ptr());
		let vm: __m128 = _mm_loadu_ps(mask.as_ptr());
		let vr = _mm_blendv_ps(va, vb, vm);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `roundps` via unaligned `movups`.
///
/// # Safety
/// Caller proved SSE4.1 via [`Sse41`].
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn roundps<const ROUNDING: i32>(a: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va: __m128 = _mm_loadu_ps(a.as_ptr());
		let vr = _mm_round_ps::<ROUNDING>(va);
		let mut out = [0f32; 4];
		_mm_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `roundpd` via unaligned `movupd`.
///
/// # Safety
/// Caller proved SSE4.1 via [`Sse41`].
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn roundpd<const ROUNDING: i32>(a: &[f64; 2]) -> [f64; 2] {
	unsafe {
		let va: __m128d = _mm_loadu_pd(a.as_ptr());
		let vr = _mm_round_pd::<ROUNDING>(va);
		let mut out = [0f64; 2];
		_mm_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

// select: blendv; mask must be all-0/all-1 lanes (e.g. cmpeq/cmpgt).
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_i32x4, slice_fn = select_i32_slice, intrinsic_fn = pblendvb_i32,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: i32| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_u32x4, slice_fn = select_u32_slice, intrinsic_fn = pblendvb_u32,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: u32| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_f32x4, slice_fn = select_f32_slice, intrinsic_fn = blendvps_select,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_blendv_ps, scalar = |a: f32, b: f32, m: f32| if m.is_sign_negative() { b } else { a },
	fixed_doc = "Per-lane select (`blendvps`): mask sign bit picks `b` (same as [`Sse41::blend_f32x4`]).",
	slice_doc = "`out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}

// select narrow: same pblendvb. blendv tests each byte's sign bit, not whole-lane
// nonzero; only agrees with `!= 0` for all-0/all-1 masks (see avx512f select doc).
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_i8x16, slice_fn = select_i8_slice, intrinsic_fn = pblendvb_i8,
	width = 16, elem = i8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: i8| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_u8x16, slice_fn = select_u8_slice, intrinsic_fn = pblendvb_u8,
	width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: u8| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_i16x8, slice_fn = select_i16_slice, intrinsic_fn = pblendvb_i16,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: i16| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_u16x8, slice_fn = select_u16_slice, intrinsic_fn = pblendvb_u16,
	width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: u16| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_i64x2, slice_fn = select_i64_slice, intrinsic_fn = pblendvb_i64,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: i64| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 2-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_u64x2, slice_fn = select_u64_slice, intrinsic_fn = pblendvb_u64,
	width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_blendv_epi8, scalar = |a, b, m: u64| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`pblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 2-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Sse41, target_feature = "sse4.1",
	fixed_fn = select_f64x2, slice_fn = select_f64_slice, intrinsic_fn = blendvpd_select,
	width = 2, elem = f64, vec = __m128d, loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
	intrinsic = _mm_blendv_pd, scalar = |a: f64, b: f64, m: f64| if m.is_sign_negative() { b } else { a },
	fixed_doc = "Per-lane select (`blendvpd`): mask sign bit picks `b`.",
	slice_doc = "`out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 2-wide chunks, scalar remainder.",
}

impl Sse41 {
	/// Lane equality mask, `i64` (all-1s if equal, else 0; `pcmpeqq`, native from SSE4.1).
	#[inline]
	pub fn cmpeq_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		unsafe { pcmpeqq(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
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
			oc.copy_from_slice(&self.cmpeq_i64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { -1 } else { 0 };
		}
	}

	/// Lane equality mask as `u64` all-1s / 0 (`pcmpeqq` view).
	#[inline]
	pub fn cmpeq_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		let ai: [i64; 2] = core::array::from_fn(|i| a[i] as i64);
		let bi: [i64; 2] = core::array::from_fn(|i| b[i] as i64);
		let r = self.cmpeq_i64x2(ai, bi);
		core::array::from_fn(|i| r[i] as u64)
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
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
			oc.copy_from_slice(&self.cmpeq_u64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { !0 } else { 0 };
		}
	}
}

/// # Safety
/// Caller proved SSE4.1 via [`Sse41`].
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn pcmpeqq(a: &[i64; 2], b: &[i64; 2]) -> [i64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_cmpeq_epi64(va, vb);
		let mut out = [0i64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

macro_rules! sse41_i32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		sse41_i32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Sse41, lift_target_feature = "sse4.1,avx",
			lifted_fn = $lifted_fn, lift_proof = Avx,
			width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse41, target_feature = "sse4.1",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse41_u32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		sse41_u32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Sse41, lift_target_feature = "sse4.1,avx",
			lifted_fn = $lifted_fn, lift_proof = Avx,
			width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse41, target_feature = "sse4.1",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse41_i32_binop!(
	mul_i32x4, mul_i32_slice, pmulld, _mm_mullo_epi32, |x: i32, y: i32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`pmulld`).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 4-wide `mul_i32x4` chunks, scalar remainder.",
	lifted_fn = mul_i32_slice_lifted,
);
// Native min/max: auto bottoms on SSE2-composed forms (no Sse41 probe), so slices
// are not cascade-private: public token API for callers that already hold Sse41.
simd_binop! {
	token = Sse41, vis = pub, target_feature = "sse4.1",
	fixed_fn = min_i32x4, slice_fn = min_i32_slice, intrinsic_fn = pminsd,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_min_epi32, scalar = |x, y| x.min(y),
	fixed_doc = "Per-lane signed min (`pminsd`).",
	slice_doc = "`out[i] = min(a[i], b[i])`. 4-wide `min_i32x4` chunks, scalar remainder.",
}
simd_binop! {
	token = Sse41, vis = pub, target_feature = "sse4.1",
	fixed_fn = max_i32x4, slice_fn = max_i32_slice, intrinsic_fn = pmaxsd,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_max_epi32, scalar = |x, y| x.max(y),
	fixed_doc = "Per-lane signed max (`pmaxsd`).",
	slice_doc = "`out[i] = max(a[i], b[i])`. 4-wide `max_i32x4` chunks, scalar remainder.",
}

sse41_u32_binop!(
	mul_u32x4, mul_u32_slice, pmulld_u, _mm_mullo_epi32, |x: u32, y: u32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`pmulld`).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 4-wide `mul_u32x4` chunks, scalar remainder.",
	lifted_fn = mul_u32_slice_lifted,
);
simd_binop! {
	token = Sse41, vis = pub, target_feature = "sse4.1",
	fixed_fn = min_u32x4, slice_fn = min_u32_slice, intrinsic_fn = pminud,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_min_epu32, scalar = |x, y| x.min(y),
	fixed_doc = "Per-lane unsigned min (`pminud`).",
	slice_doc = "`out[i] = min(a[i], b[i])`. 4-wide `min_u32x4` chunks, scalar remainder.",
}
simd_binop! {
	token = Sse41, vis = pub, target_feature = "sse4.1",
	fixed_fn = max_u32x4, slice_fn = max_u32_slice, intrinsic_fn = pmaxud,
	width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_max_epu32, scalar = |x, y| x.max(y),
	fixed_doc = "Per-lane unsigned max (`pmaxud`).",
	slice_doc = "`out[i] = max(a[i], b[i])`. 4-wide `max_u32x4` chunks, scalar remainder.",
}

macro_rules! sse41_i8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse41, target_feature = "sse4.1",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse41_u16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse41, target_feature = "sse4.1",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse41_i8_binop!(
	min_i8x16, min_i8_slice, pminsb, _mm_min_epi8, |x, y| x.min(y),
	"Per-lane signed min (`pminsb`). SSE2 only has this for u8/i16; see module doc.",
	"`out[i] = min(a[i], b[i])`. 16-wide `min_i8x16` chunks, scalar remainder."
);
sse41_i8_binop!(
	max_i8x16, max_i8_slice, pmaxsb, _mm_max_epi8, |x, y| x.max(y),
	"Per-lane signed max (`pmaxsb`). SSE2 only has this for u8/i16; see module doc.",
	"`out[i] = max(a[i], b[i])`. 16-wide `max_i8x16` chunks, scalar remainder."
);
sse41_u16_binop!(
	min_u16x8, min_u16_slice, pminuw, _mm_min_epu16, |x, y| x.min(y),
	"Per-lane unsigned min (`pminuw`). SSE2 only has this for u8/i16; see module doc.",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_u16x8` chunks, scalar remainder."
);
sse41_u16_binop!(
	max_u16x8, max_u16_slice, pmaxuw, _mm_max_epu16, |x, y| x.max(y),
	"Per-lane unsigned max (`pmaxuw`). SSE2 only has this for u8/i16; see module doc.",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_u16x8` chunks, scalar remainder."
);

#[cfg(test)]
#[path = "../../test/ops/sse/sse41.rs"]
mod tests;
