//! AVX-512BW: 512-bit byte/word ops over i8..u16. Token: `Avx512Bw`.
//! Provides wide integer arithmetic, compares, blends, and masked ops.

use core::arch::x86_64::{
	__m128i, __m512i, _mm512_abs_epi16, _mm512_abs_epi8, _mm512_add_epi16, _mm512_add_epi8, _mm512_adds_epi16,
	_mm512_broadcastb_epi8, _mm512_bslli_epi128, _mm_loadu_si128,
	_mm512_adds_epi8, _mm512_adds_epu16, _mm512_adds_epu8, _mm512_avg_epu16, _mm512_avg_epu8, _mm512_cmpeq_epi16_mask,
	_mm512_cmpeq_epi8_mask, _mm512_cmpgt_epi16_mask, _mm512_cmpgt_epi8_mask, _mm512_cmpgt_epu16_mask,
	_mm512_cmpgt_epu8_mask, _mm512_loadu_si512, _mm512_mask_abs_epi16, _mm512_mask_abs_epi8, _mm512_mask_add_epi16,
	_mm512_mask_add_epi8, _mm512_mask_adds_epi16, _mm512_mask_adds_epi8, _mm512_mask_adds_epu16,
	_mm512_mask_adds_epu8, _mm512_mask_avg_epu16, _mm512_mask_avg_epu8, _mm512_mask_blend_epi16,
	_mm512_mask_blend_epi8, _mm512_mask_max_epi16,
	_mm512_mask_max_epi8, _mm512_mask_max_epu16, _mm512_mask_max_epu8, _mm512_mask_min_epi16, _mm512_mask_min_epi8,
	_mm512_mask_min_epu16, _mm512_mask_min_epu8, _mm512_mask_mullo_epi16, _mm512_mask_sub_epi16, _mm512_mask_sub_epi8,
	_mm512_mask_subs_epi16, _mm512_mask_subs_epi8, _mm512_mask_subs_epu16, _mm512_mask_subs_epu8,
	_mm512_maskz_abs_epi16, _mm512_maskz_abs_epi8, _mm512_maskz_add_epi16, _mm512_maskz_add_epi8,
	_mm512_maskz_adds_epi16, _mm512_maskz_adds_epi8, _mm512_maskz_adds_epu16, _mm512_maskz_adds_epu8,
	_mm512_maskz_avg_epu16, _mm512_maskz_avg_epu8,
	_mm512_maskz_max_epi16, _mm512_maskz_max_epi8, _mm512_maskz_max_epu16, _mm512_maskz_max_epu8,
	_mm512_maskz_min_epi16, _mm512_maskz_min_epi8, _mm512_maskz_min_epu16, _mm512_maskz_min_epu8,
	_mm512_maskz_mullo_epi16, _mm512_maskz_set1_epi16, _mm512_maskz_set1_epi8, _mm512_maskz_sub_epi16,
	_mm512_maskz_sub_epi8, _mm512_maskz_subs_epi16, _mm512_maskz_subs_epi8, _mm512_maskz_subs_epu16,
	_mm512_maskz_subs_epu8, _mm512_max_epi16, _mm512_max_epi8, _mm512_max_epu16,
	_mm512_max_epu8, _mm512_min_epi16, _mm512_min_epi8, _mm512_min_epu16, _mm512_min_epu8, _mm512_mullo_epi16,
	_mm512_sll_epi16, _mm512_sra_epi16, _mm512_srl_epi16, _mm512_storeu_si512, _mm512_sub_epi16, _mm512_sub_epi8,
	_mm512_subs_epi16, _mm512_subs_epi8, _mm512_subs_epu16, _mm512_subs_epu8, _mm512_test_epi16_mask,
	_mm512_test_epi8_mask,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::macros::{simd_binop, simd_binop_masked, simd_shift_imm, simd_unop, simd_unop_masked};

/// Proof token: AVX-512BW available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Bw(());

impl Avx512Bw {
	/// `None` if the CPU (or the compile-time target) lacks AVX-512BW.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From resolved tier (`V4` lists `Feature::Avx512bw`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx512bw).then_some(Avx512Bw(()))
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512bw).then_some(Avx512Bw(()))
	}

	/// `vpslldq`: per-128-bit-lane byte shift left by `IMM8`, zero-filled
	/// from each lane's bottom: same lane-locked shape as AVX2's
	/// [`super::super::avx::avx2::Avx2::slli_u8x32`], just 4 lanes instead
	/// of 2. Not a per-lane numeric shift ([`simd_shift_imm`] doesn't fit,
	/// same reasoning as [`super::super::sse::sse2::Sse2::slli_u8x16`]).
	#[inline]
	pub fn bslli_u8x64<const IMM8: i32>(self, a: [u8; 64]) -> [u8; 64] {
		unsafe { bslli_epi128::<IMM8>(&a) }
	}

	/// `vpbroadcastb`: replicate `byte` across all 64 lanes.
	#[inline]
	pub fn broadcast_u8x64(self, byte: u8) -> [u8; 64] {
		unsafe { broadcastb(byte) }
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn bslli_epi128<const IMM8: i32>(a: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vr = _mm512_bslli_epi128::<IMM8>(va);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// `vpbroadcastb` takes its source from a 128-bit register (only byte 0
/// matters), same reasoning as AVX2's `broadcastb` impl fn.
///
/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn broadcastb(byte: u8) -> [u8; 64] {
	unsafe {
		let mut src = [0u8; 16];
		src[0] = byte;
		let va: __m128i = _mm_loadu_si128(src.as_ptr().cast());
		let vr = _mm512_broadcastb_epi8(va);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

macro_rules! avx512bw_i8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 64, elem = i8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512bw_u8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512bw_i16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512bw_u16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512bw_i8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Bw, target_feature = "avx512bw",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 64, elem = i8, vec = __m512i, mask = u64,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_u8_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Bw, target_feature = "avx512bw",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 64, elem = u8, vec = __m512i, mask = u64,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_i16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Bw, target_feature = "avx512bw",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = i16, vec = __m512i, mask = u32,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

macro_rules! avx512bw_u16_binop_masked {
	($merge_fn:ident, $zero_fn:ident, $merge_intrinsic_fn:ident, $zero_intrinsic_fn:ident, $merge_intrinsic:path, $zero_intrinsic:path, $merge_doc:literal, $zero_doc:literal) => {
		simd_binop_masked! {
			token = Avx512Bw, target_feature = "avx512bw",
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			width = 32, elem = u16, vec = __m512i, mask = u32,
			loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

avx512bw_i8_binop!(
	add_i8x64, add_i8_slice, paddb, _mm512_add_epi8, |x: i8, y: i8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddb`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_i8_binop!(
	sub_i8x64, sub_i8_slice, psubb, _mm512_sub_epi8, |x: i8, y: i8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubb`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_i8_binop!(
	adds_i8x64, adds_i8_slice, paddsb, _mm512_adds_epi8, |x: i8, y: i8| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddsb`, 512-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_i8_binop!(
	subs_i8x64, subs_i8_slice, psubsb, _mm512_subs_epi8, |x: i8, y: i8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubsb`, 512-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_i8_binop!(
	min_i8x64, min_i8_slice, pminsb, _mm512_min_epi8, |x, y| x.min(y),
	"Per-lane signed min (`vpminsb`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_i8_binop!(
	max_i8x64, max_i8_slice, pmaxsb, _mm512_max_epi8, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsb`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 64-wide chunks, scalar remainder."
);

avx512bw_i8_binop_masked!(
	add_i8x64_merge_masked, add_i8x64_zero_masked, mask_add_epi8_intrinsic, maskz_add_epi8_intrinsic,
	_mm512_mask_add_epi8, _mm512_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, zero-masked)."
);
avx512bw_i8_binop_masked!(
	sub_i8x64_merge_masked, sub_i8x64_zero_masked, mask_sub_epi8_intrinsic, maskz_sub_epi8_intrinsic,
	_mm512_mask_sub_epi8, _mm512_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, zero-masked)."
);
avx512bw_i8_binop_masked!(
	adds_i8x64_merge_masked, adds_i8x64_zero_masked, mask_adds_epi8_intrinsic, maskz_adds_epi8_intrinsic,
	_mm512_mask_adds_epi8, _mm512_maskz_adds_epi8,
	"`a + b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpaddsb`, merge-masked).",
	"`a + b` per lane, saturating, where `mask` bit is set, else zero (`vpaddsb`, zero-masked)."
);
avx512bw_i8_binop_masked!(
	subs_i8x64_merge_masked, subs_i8x64_zero_masked, mask_subs_epi8_intrinsic, maskz_subs_epi8_intrinsic,
	_mm512_mask_subs_epi8, _mm512_maskz_subs_epi8,
	"`a - b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpsubsb`, merge-masked).",
	"`a - b` per lane, saturating, where `mask` bit is set, else zero (`vpsubsb`, zero-masked)."
);
avx512bw_i8_binop_masked!(
	min_i8x64_merge_masked, min_i8x64_zero_masked, mask_min_epi8_intrinsic, maskz_min_epi8_intrinsic,
	_mm512_mask_min_epi8, _mm512_maskz_min_epi8,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsb`, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsb`, zero-masked)."
);
avx512bw_i8_binop_masked!(
	max_i8x64_merge_masked, max_i8x64_zero_masked, mask_max_epi8_intrinsic, maskz_max_epi8_intrinsic,
	_mm512_mask_max_epi8, _mm512_maskz_max_epi8,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsb`, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsb`, zero-masked)."
);

avx512bw_u8_binop!(
	add_u8x64, add_u8_slice, paddb_u, _mm512_add_epi8, |x: u8, y: u8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddb`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_u8_binop!(
	sub_u8x64, sub_u8_slice, psubb_u, _mm512_sub_epi8, |x: u8, y: u8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubb`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_u8_binop!(
	adds_u8x64, adds_u8_slice, paddusb, _mm512_adds_epu8, |x: u8, y: u8| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddusb`, 512-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_u8_binop!(
	subs_u8x64, subs_u8_slice, psubusb, _mm512_subs_epu8, |x: u8, y: u8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubusb`, 512-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_u8_binop!(
	min_u8x64, min_u8_slice, pminub, _mm512_min_epu8, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminub`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 64-wide chunks, scalar remainder."
);
avx512bw_u8_binop!(
	max_u8x64, max_u8_slice, pmaxub, _mm512_max_epu8, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxub`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 64-wide chunks, scalar remainder."
);

avx512bw_u8_binop_masked!(
	add_u8x64_merge_masked, add_u8x64_zero_masked, mask_add_epi8_u_intrinsic, maskz_add_epi8_u_intrinsic,
	_mm512_mask_add_epi8, _mm512_maskz_add_epi8,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddb`, merge-masked; bit-identical to the signed form, no `epu8` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddb`, zero-masked)."
);
avx512bw_u8_binop_masked!(
	sub_u8x64_merge_masked, sub_u8x64_zero_masked, mask_sub_epi8_u_intrinsic, maskz_sub_epi8_u_intrinsic,
	_mm512_mask_sub_epi8, _mm512_maskz_sub_epi8,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubb`, merge-masked; bit-identical to the signed form, no `epu8` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubb`, zero-masked)."
);
avx512bw_u8_binop_masked!(
	adds_u8x64_merge_masked, adds_u8x64_zero_masked, mask_adds_epu8_intrinsic, maskz_adds_epu8_intrinsic,
	_mm512_mask_adds_epu8, _mm512_maskz_adds_epu8,
	"`a + b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpaddusb`, merge-masked).",
	"`a + b` per lane, saturating, where `mask` bit is set, else zero (`vpaddusb`, zero-masked)."
);
avx512bw_u8_binop_masked!(
	subs_u8x64_merge_masked, subs_u8x64_zero_masked, mask_subs_epu8_intrinsic, maskz_subs_epu8_intrinsic,
	_mm512_mask_subs_epu8, _mm512_maskz_subs_epu8,
	"`a - b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpsubusb`, merge-masked).",
	"`a - b` per lane, saturating, where `mask` bit is set, else zero (`vpsubusb`, zero-masked)."
);
avx512bw_u8_binop_masked!(
	min_u8x64_merge_masked, min_u8x64_zero_masked, mask_min_epu8_intrinsic, maskz_min_epu8_intrinsic,
	_mm512_mask_min_epu8, _mm512_maskz_min_epu8,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminub`, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminub`, zero-masked)."
);
avx512bw_u8_binop_masked!(
	max_u8x64_merge_masked, max_u8x64_zero_masked, mask_max_epu8_intrinsic, maskz_max_epu8_intrinsic,
	_mm512_mask_max_epu8, _mm512_maskz_max_epu8,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxub`, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxub`, zero-masked)."
);

avx512bw_i16_binop!(
	add_i16x32, add_i16_slice, paddw, _mm512_add_epi16, |x: i16, y: i16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddw`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	sub_i16x32, sub_i16_slice, psubw, _mm512_sub_epi16, |x: i16, y: i16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubw`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	adds_i16x32, adds_i16_slice, paddsw, _mm512_adds_epi16, |x: i16, y: i16| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddsw`, 512-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	subs_i16x32, subs_i16_slice, psubsw, _mm512_subs_epi16, |x: i16, y: i16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubsw`, 512-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	mul_i16x32, mul_i16_slice, pmullw, _mm512_mullo_epi16, |x: i16, y: i16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`vpmullw`, 512-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	min_i16x32, min_i16_slice, pminsw, _mm512_min_epi16, |x, y| x.min(y),
	"Per-lane signed min (`vpminsw`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_binop!(
	max_i16x32, max_i16_slice, pmaxsw, _mm512_max_epi16, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsw`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 32-wide chunks, scalar remainder."
);

avx512bw_i16_binop_masked!(
	add_i16x32_merge_masked, add_i16x32_zero_masked, mask_add_epi16_intrinsic, maskz_add_epi16_intrinsic,
	_mm512_mask_add_epi16, _mm512_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, merge-masked).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	sub_i16x32_merge_masked, sub_i16x32_zero_masked, mask_sub_epi16_intrinsic, maskz_sub_epi16_intrinsic,
	_mm512_mask_sub_epi16, _mm512_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, merge-masked).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	adds_i16x32_merge_masked, adds_i16x32_zero_masked, mask_adds_epi16_intrinsic, maskz_adds_epi16_intrinsic,
	_mm512_mask_adds_epi16, _mm512_maskz_adds_epi16,
	"`a + b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpaddsw`, merge-masked).",
	"`a + b` per lane, saturating, where `mask` bit is set, else zero (`vpaddsw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	subs_i16x32_merge_masked, subs_i16x32_zero_masked, mask_subs_epi16_intrinsic, maskz_subs_epi16_intrinsic,
	_mm512_mask_subs_epi16, _mm512_maskz_subs_epi16,
	"`a - b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpsubsw`, merge-masked).",
	"`a - b` per lane, saturating, where `mask` bit is set, else zero (`vpsubsw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	mul_i16x32_merge_masked, mul_i16x32_zero_masked, mask_mullo_epi16_intrinsic, maskz_mullo_epi16_intrinsic,
	_mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, merge-masked).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	min_i16x32_merge_masked, min_i16x32_zero_masked, mask_min_epi16_intrinsic, maskz_min_epi16_intrinsic,
	_mm512_mask_min_epi16, _mm512_maskz_min_epi16,
	"Per-lane signed min where `mask` bit is set, else copied from `src` (`vpminsw`, merge-masked).",
	"Per-lane signed min where `mask` bit is set, else zero (`vpminsw`, zero-masked)."
);
avx512bw_i16_binop_masked!(
	max_i16x32_merge_masked, max_i16x32_zero_masked, mask_max_epi16_intrinsic, maskz_max_epi16_intrinsic,
	_mm512_mask_max_epi16, _mm512_maskz_max_epi16,
	"Per-lane signed max where `mask` bit is set, else copied from `src` (`vpmaxsw`, merge-masked).",
	"Per-lane signed max where `mask` bit is set, else zero (`vpmaxsw`, zero-masked)."
);

avx512bw_u16_binop!(
	add_u16x32, add_u16_slice, paddw_u, _mm512_add_epi16, |x: u16, y: u16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddw`, 512-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	sub_u16x32, sub_u16_slice, psubw_u, _mm512_sub_epi16, |x: u16, y: u16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubw`, 512-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	adds_u16x32, adds_u16_slice, paddusw, _mm512_adds_epu16, |x: u16, y: u16| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddusw`, 512-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	subs_u16x32, subs_u16_slice, psubusw, _mm512_subs_epu16, |x: u16, y: u16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubusw`, 512-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	mul_u16x32, mul_u16_slice, pmullw_u, _mm512_mullo_epi16, |x: u16, y: u16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`vpmullw`, 512-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	min_u16x32, min_u16_slice, pminuw, _mm512_min_epu16, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminuw`, 512-bit).",
	"`out[i] = min(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_binop!(
	max_u16x32, max_u16_slice, pmaxuw, _mm512_max_epu16, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxuw`, 512-bit).",
	"`out[i] = max(a[i], b[i])`. 32-wide chunks, scalar remainder."
);

avx512bw_u16_binop_masked!(
	add_u16x32_merge_masked, add_u16x32_zero_masked, mask_add_epi16_u_intrinsic, maskz_add_epi16_u_intrinsic,
	_mm512_mask_add_epi16, _mm512_maskz_add_epi16,
	"`a + b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpaddw`, merge-masked; bit-identical to the signed form, no `epu16` add exists).",
	"`a + b` per lane, wrapping, where `mask` bit is set, else zero (`vpaddw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	sub_u16x32_merge_masked, sub_u16x32_zero_masked, mask_sub_epi16_u_intrinsic, maskz_sub_epi16_u_intrinsic,
	_mm512_mask_sub_epi16, _mm512_maskz_sub_epi16,
	"`a - b` per lane, wrapping, where `mask` bit is set, else copied from `src` (`vpsubw`, merge-masked; bit-identical to the signed form, no `epu16` sub exists).",
	"`a - b` per lane, wrapping, where `mask` bit is set, else zero (`vpsubw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	adds_u16x32_merge_masked, adds_u16x32_zero_masked, mask_adds_epu16_intrinsic, maskz_adds_epu16_intrinsic,
	_mm512_mask_adds_epu16, _mm512_maskz_adds_epu16,
	"`a + b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpaddusw`, merge-masked).",
	"`a + b` per lane, saturating, where `mask` bit is set, else zero (`vpaddusw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	subs_u16x32_merge_masked, subs_u16x32_zero_masked, mask_subs_epu16_intrinsic, maskz_subs_epu16_intrinsic,
	_mm512_mask_subs_epu16, _mm512_maskz_subs_epu16,
	"`a - b` per lane, saturating, where `mask` bit is set, else copied from `src` (`vpsubusw`, merge-masked).",
	"`a - b` per lane, saturating, where `mask` bit is set, else zero (`vpsubusw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	mul_u16x32_merge_masked, mul_u16x32_zero_masked, mask_mullo_epi16_u_intrinsic, maskz_mullo_epi16_u_intrinsic,
	_mm512_mask_mullo_epi16, _mm512_maskz_mullo_epi16,
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else copied from `src` (`vpmullw`, merge-masked; bit-identical to the signed form).",
	"`a * b` per lane, low 16 bits, where `mask` bit is set, else zero (`vpmullw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	min_u16x32_merge_masked, min_u16x32_zero_masked, mask_min_epu16_intrinsic, maskz_min_epu16_intrinsic,
	_mm512_mask_min_epu16, _mm512_maskz_min_epu16,
	"Per-lane unsigned min where `mask` bit is set, else copied from `src` (`vpminuw`, merge-masked).",
	"Per-lane unsigned min where `mask` bit is set, else zero (`vpminuw`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	max_u16x32_merge_masked, max_u16x32_zero_masked, mask_max_epu16_intrinsic, maskz_max_epu16_intrinsic,
	_mm512_mask_max_epu16, _mm512_maskz_max_epu16,
	"Per-lane unsigned max where `mask` bit is set, else copied from `src` (`vpmaxuw`, merge-masked).",
	"Per-lane unsigned max where `mask` bit is set, else zero (`vpmaxuw`, zero-masked)."
);

// cmpeq: k-mask + maskz_set1 (same shape as avx512f i32 cmpeq).
impl Avx512Bw {
	/// Lane equality mask: all-1s if equal, else 0 (`vpcmpeqb` via k-mask).
	#[inline]
	pub fn cmpeq_i8x64(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
		unsafe { vpcmpeqb_i8(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0`. 64-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [i8; 64] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_i8x64(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { -1 } else { 0 };
		}
	}

	/// Lane equality mask as `u8` all-1s / 0.
	#[inline]
	pub fn cmpeq_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		let ai: [i8; 64] = core::array::from_fn(|i| a[i] as i8);
		let bi: [i8; 64] = core::array::from_fn(|i| b[i] as i8);
		let r = self.cmpeq_i8x64(ai, bi);
		core::array::from_fn(|i| r[i] as u8)
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [u8; 64] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_u8x64(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { !0 } else { 0 };
		}
	}

	/// Lane equality mask: all-1s if equal, else 0 (`vpcmpeqw` via k-mask).
	#[inline]
	pub fn cmpeq_i16x32(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
		unsafe { vpcmpeqw_i16(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0`. 32-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [i16; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_i16x32(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { -1 } else { 0 };
		}
	}

	/// Lane equality mask as `u16` all-1s / 0.
	#[inline]
	pub fn cmpeq_u16x32(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		let ai: [i16; 32] = core::array::from_fn(|i| a[i] as i16);
		let bi: [i16; 32] = core::array::from_fn(|i| b[i] as i16);
		let r = self.cmpeq_i16x32(ai, bi);
		core::array::from_fn(|i| r[i] as u16)
	}

	/// `out[i] = all-1s if a[i]==b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpeq_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpeq_u16x32(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x == y { !0 } else { 0 };
		}
	}

	/// Lane greater-than mask: all-1s if `a>b`, else 0 (`vpcmpgtb` via k-mask).
	#[inline]
	pub fn cmpgt_i8x64(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
		unsafe { vpcmpgtb_i8(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0`. 64-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [i8; 64] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_i8x64(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { -1 } else { 0 };
		}
	}

	/// Unsigned greater-than mask via native `vpcmpgtub` k-mask.
	#[inline]
	pub fn cmpgt_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		unsafe { vpcmpgtub_u8(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [u8; 64] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u8x64(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane greater-than mask: all-1s if `a>b`, else 0 (`vpcmpgtw` via k-mask).
	#[inline]
	pub fn cmpgt_i16x32(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
		unsafe { vpcmpgtw_i16(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0`. 32-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [i16; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_i16x32(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { -1 } else { 0 };
		}
	}

	/// Unsigned greater-than mask via native `vpcmpgtuw` k-mask.
	#[inline]
	pub fn cmpgt_u16x32(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		unsafe { vpcmpgtuw_u16(&a, &b) }
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u16x32(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	#[inline]
	pub fn cmplt_i8x64(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
		self.cmpgt_i8x64(b, a)
	}
	pub(crate) fn cmplt_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_i8x64(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
		core::array::from_fn(|i| !self.cmpgt_i8x64(a, b)[i])
	}
	pub(crate) fn cmple_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_i8x64(self, a: [i8; 64], b: [i8; 64]) -> [i8; 64] {
		core::array::from_fn(|i| !self.cmplt_i8x64(a, b)[i])
	}
	pub(crate) fn cmpge_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmplt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		self.cmpgt_u8x64(b, a)
	}
	pub(crate) fn cmplt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		core::array::from_fn(|i| !self.cmpgt_u8x64(a, b)[i])
	}
	pub(crate) fn cmple_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		core::array::from_fn(|i| !self.cmplt_u8x64(a, b)[i])
	}
	pub(crate) fn cmpge_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmplt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_i16x32(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
		self.cmpgt_i16x32(b, a)
	}
	pub(crate) fn cmplt_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_i16x32(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
		core::array::from_fn(|i| !self.cmpgt_i16x32(a, b)[i])
	}
	pub(crate) fn cmple_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_i16x32(self, a: [i16; 32], b: [i16; 32]) -> [i16; 32] {
		core::array::from_fn(|i| !self.cmplt_i16x32(a, b)[i])
	}
	pub(crate) fn cmpge_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmplt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_u16x32(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		self.cmpgt_u16x32(b, a)
	}
	pub(crate) fn cmplt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_u16x32(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		core::array::from_fn(|i| !self.cmpgt_u16x32(a, b)[i])
	}
	pub(crate) fn cmple_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_u16x32(self, a: [u16; 32], b: [u16; 32]) -> [u16; 32] {
		core::array::from_fn(|i| !self.cmplt_u16x32(a, b)[i])
	}
	pub(crate) fn cmpge_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmplt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Per-lane select (`vptestmb` + `vpblendmb`). `mask`: all-0/1 (e.g. cmpeq).
	#[inline]
	pub fn select_i8x64(self, a: [i8; 64], b: [i8; 64], mask: [i8; 64]) -> [i8; 64] {
		unsafe { vpblendmb_i8(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 64-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_i8_slice(self, a: &[i8], b: &[i8], mask: &[i8], out: &mut [i8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let mask_chunks = mask.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [i8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [i8; 64] = bc.try_into().expect("chunks_exact width");
			let mv: [i8; 64] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_i8x64(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select, `u8` view of [`Avx512Bw::select_i8x64`].
	#[inline]
	pub fn select_u8x64(self, a: [u8; 64], b: [u8; 64], mask: [u8; 64]) -> [u8; 64] {
		let ai: [i8; 64] = core::array::from_fn(|i| a[i] as i8);
		let bi: [i8; 64] = core::array::from_fn(|i| b[i] as i8);
		let mi: [i8; 64] = core::array::from_fn(|i| mask[i] as i8);
		let r = self.select_i8x64(ai, bi, mi);
		core::array::from_fn(|i| r[i] as u8)
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u8` view).
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_u8_slice(self, a: &[u8], b: &[u8], mask: &[u8], out: &mut [u8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(64);
		let b_chunks = b.chunks_exact(64);
		let mask_chunks = mask.chunks_exact(64);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(64);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [u8; 64] = ac.try_into().expect("chunks_exact width");
			let bv: [u8; 64] = bc.try_into().expect("chunks_exact width");
			let mv: [u8; 64] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_u8x64(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select (`vptestmw` + `vpblendmw`). `mask`: all-0/1 (e.g. cmpeq).
	#[inline]
	pub fn select_i16x32(self, a: [i16; 32], b: [i16; 32], mask: [i16; 32]) -> [i16; 32] {
		unsafe { vpblendmw_i16(&a, &b, &mask) }
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 32-wide chunks, scalar remainder.
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_i16_slice(self, a: &[i16], b: &[i16], mask: &[i16], out: &mut [i16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let mask_chunks = mask.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [i16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [i16; 32] = bc.try_into().expect("chunks_exact width");
			let mv: [i16; 32] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_i16x32(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}

	/// Per-lane select, `u16` view of [`Avx512Bw::select_i16x32`].
	#[inline]
	pub fn select_u16x32(self, a: [u16; 32], b: [u16; 32], mask: [u16; 32]) -> [u16; 32] {
		let ai: [i16; 32] = core::array::from_fn(|i| a[i] as i16);
		let bi: [i16; 32] = core::array::from_fn(|i| b[i] as i16);
		let mi: [i16; 32] = core::array::from_fn(|i| mask[i] as i16);
		let r = self.select_i16x32(ai, bi, mi);
		core::array::from_fn(|i| r[i] as u16)
	}

	/// `out[i] = if mask[i] != 0 { b[i] } else { a[i] }` (`u16` view).
	///
	/// # Panics
	/// Length mismatch among `a`, `b`, `mask`, `out`.
	pub(crate) fn select_u16_slice(self, a: &[u16], b: &[u16], mask: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(a.len(), mask.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let mask_chunks = mask.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mask_rem = mask_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (((ac, bc), mc), oc) in a_chunks.zip(b_chunks).zip(mask_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 32] = bc.try_into().expect("chunks_exact width");
			let mv: [u16; 32] = mc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.select_u16x32(av, bv, mv));
		}
		for (((&x, &y), &m), o) in a_rem.iter().zip(b_rem).zip(mask_rem).zip(out_chunks.into_remainder()) {
			*o = if m != 0 { y } else { x };
		}
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpblendmb_i8(a: &[i8; 64], b: &[i8; 64], mask: &[i8; 64]) -> [i8; 64] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let vm = _mm512_loadu_si512(mask.as_ptr().cast());
		let k = _mm512_test_epi8_mask(vm, vm);
		let vr = _mm512_mask_blend_epi8(k, va, vb);
		let mut out = [0i8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpblendmw_i16(a: &[i16; 32], b: &[i16; 32], mask: &[i16; 32]) -> [i16; 32] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let vm = _mm512_loadu_si512(mask.as_ptr().cast());
		let k = _mm512_test_epi16_mask(vm, vm);
		let vr = _mm512_mask_blend_epi16(k, va, vb);
		let mut out = [0i16; 32];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpeqb_i8(a: &[i8; 64], b: &[i8; 64]) -> [i8; 64] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpeq_epi8_mask(va, vb);
		let vr = _mm512_maskz_set1_epi8(k, -1);
		let mut out = [0i8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpeqw_i16(a: &[i16; 32], b: &[i16; 32]) -> [i16; 32] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpeq_epi16_mask(va, vb);
		let vr = _mm512_maskz_set1_epi16(k, -1);
		let mut out = [0i16; 32];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpgtb_i8(a: &[i8; 64], b: &[i8; 64]) -> [i8; 64] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epi8_mask(va, vb);
		let vr = _mm512_maskz_set1_epi8(k, -1);
		let mut out = [0i8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpgtub_u8(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epu8_mask(va, vb);
		let vr = _mm512_maskz_set1_epi8(k, -1);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpgtw_i16(a: &[i16; 32], b: &[i16; 32]) -> [i16; 32] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epi16_mask(va, vb);
		let vr = _mm512_maskz_set1_epi16(k, -1);
		let mut out = [0i16; 32];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn vpcmpgtuw_u16(a: &[u16; 32], b: &[u16; 32]) -> [u16; 32] {
	unsafe {
		let va = _mm512_loadu_si512(a.as_ptr().cast());
		let vb = _mm512_loadu_si512(b.as_ptr().cast());
		let k = _mm512_cmpgt_epu16_mask(va, vb);
		let vr = _mm512_maskz_set1_epi16(k, -1);
		let mut out = [0u16; 32];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

macro_rules! avx512bw_i16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx512bw_u16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx512Bw, target_feature = "avx512bw",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx512bw_i16_shift_imm!(
	shl_i16x32, shl_i16_slice, vpsllw, _mm512_sll_epi16, |x: i16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpsllw`, 512-bit).",
	"`out[i] = a[i] << IMM`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_shift_imm!(
	shr_i16x32, shr_i16_slice, vpsrlw, _mm512_srl_epi16, |x: i16, imm| ((x as u16).wrapping_shr(imm)) as i16,
	"`a >> IMM` logical per lane (`vpsrlw`, 512-bit).",
	"`out[i] = a[i] logical >> IMM`. 32-wide chunks, scalar remainder."
);
avx512bw_i16_shift_imm!(
	sra_i16x32, sra_i16_slice, vpsraw, _mm512_sra_epi16, |x: i16, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`vpsraw`, 512-bit).",
	"`out[i] = a[i] arithmetic >> IMM`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_shift_imm!(
	shl_u16x32, shl_u16_slice, vpsllw_u, _mm512_sll_epi16, |x: u16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpsllw`, 512-bit).",
	"`out[i] = a[i] << IMM`. 32-wide chunks, scalar remainder."
);
avx512bw_u16_shift_imm!(
	shr_u16x32, shr_u16_slice, vpsrlw_u, _mm512_srl_epi16, |x: u16, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`vpsrlw`, 512-bit).",
	"`out[i] = a[i] >> IMM`. 32-wide chunks, scalar remainder."
);

// abs/avg: also SSSE3(abs)/SSE2(avg) 128-bit, AVX2 256-bit: full cascade in auto_up.
simd_unop! {
	token = Avx512Bw, target_feature = "avx512bw",
	fixed_fn = abs_i8x64, slice_fn = abs_i8_slice, intrinsic_fn = abs_i8x64_intrinsic,
	width = 64, elem = i8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_abs_epi8, scalar = |x: i8| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsb`, 512-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 64-wide chunks, scalar remainder.",
}
simd_unop! {
	token = Avx512Bw, target_feature = "avx512bw",
	fixed_fn = abs_i16x32, slice_fn = abs_i16_slice, intrinsic_fn = abs_i16x32_intrinsic,
	width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_abs_epi16, scalar = |x: i16| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsw`, 512-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 32-wide chunks, scalar remainder.",
}

simd_unop_masked! {
	token = Avx512Bw, target_feature = "avx512bw",
	merge_fn = abs_i8x64_merge_masked, zero_fn = abs_i8x64_zero_masked,
	merge_intrinsic_fn = mask_abs_epi8_intrinsic, zero_intrinsic_fn = maskz_abs_epi8_intrinsic,
	width = 64, elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_abs_epi8, zero_intrinsic = _mm512_maskz_abs_epi8,
	merge_doc = "Per-lane absolute value where `mask` bit is set, else copied from `src` (`vpabsb`, merge-masked).",
	zero_doc = "Per-lane absolute value where `mask` bit is set, else zero (`vpabsb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Bw, target_feature = "avx512bw",
	merge_fn = abs_i16x32_merge_masked, zero_fn = abs_i16x32_zero_masked,
	merge_intrinsic_fn = mask_abs_epi16_intrinsic, zero_intrinsic_fn = maskz_abs_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_abs_epi16, zero_intrinsic = _mm512_maskz_abs_epi16,
	merge_doc = "Per-lane absolute value where `mask` bit is set, else copied from `src` (`vpabsw`, merge-masked).",
	zero_doc = "Per-lane absolute value where `mask` bit is set, else zero (`vpabsw`, zero-masked).",
}

simd_binop! {
	token = Avx512Bw, target_feature = "avx512bw",
	fixed_fn = avg_u8x64, slice_fn = avg_u8_slice, intrinsic_fn = pavgb,
	width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_avg_epu8, scalar = |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8,
	fixed_doc = "Per-lane rounded unsigned average, `(a+b+1)/2` (`vpavgb`, 512-bit). No signed form exists in the ISA.",
	slice_doc = "`out[i] = (a[i] as u16 + b[i] as u16 + 1) / 2`. 64-wide chunks, scalar remainder.",
}
simd_binop! {
	token = Avx512Bw, target_feature = "avx512bw",
	fixed_fn = avg_u16x32, slice_fn = avg_u16_slice, intrinsic_fn = pavgw,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_avg_epu16, scalar = |x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16,
	fixed_doc = "Per-lane rounded unsigned average, `(a+b+1)/2` (`vpavgw`, 512-bit). No signed form exists in the ISA.",
	slice_doc = "`out[i] = (a[i] as u32 + b[i] as u32 + 1) / 2`. 32-wide chunks, scalar remainder.",
}

avx512bw_u8_binop_masked!(
	avg_u8x64_merge_masked, avg_u8x64_zero_masked, mask_avg_epu8_intrinsic, maskz_avg_epu8_intrinsic,
	_mm512_mask_avg_epu8, _mm512_maskz_avg_epu8,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgb`, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgb`, zero-masked)."
);
avx512bw_u16_binop_masked!(
	avg_u16x32_merge_masked, avg_u16x32_zero_masked, mask_avg_epu16_intrinsic, maskz_avg_epu16_intrinsic,
	_mm512_mask_avg_epu16, _mm512_maskz_avg_epu16,
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else copied from `src` (`vpavgw`, merge-masked).",
	"Per-lane rounded unsigned average, `(a+b+1)/2`, where `mask` bit is set, else zero (`vpavgw`, zero-masked)."
);

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512bw.rs"]
mod tests;
