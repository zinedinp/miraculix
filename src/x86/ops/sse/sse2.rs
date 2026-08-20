//! SSE2: XMM i32/u32/f64 and i8/u8/i16/u16. x86-64 ABI baseline. Token: [`Sse2`].
//! Wide int add/sub/bit/cmp/shift, f64/bitwise; i64/u64 mullo schoolbook below AVX-512DQ; sat/avg on narrow ints.
//! Composed `mul_i8/u8` and byte `shl/shr/sra` (no native 8-bit mul/shift). `sqrt_f64x2` fixed-width only.

use core::arch::x86_64::{
	__m128d, __m128i, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_add_epi8, _mm_add_pd, _mm_adds_epi16,
	_mm_adds_epi8, _mm_adds_epu16, _mm_adds_epu8, _mm_and_pd, _mm_and_si128, _mm_andnot_pd, _mm_andnot_si128,
	_mm_avg_epu16, _mm_avg_epu8, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpeq_epi8, _mm_cmpeq_pd, _mm_cmpge_pd,
	_mm_cmpgt_epi16, _mm_cmpgt_epi32, _mm_cmpgt_epi8, _mm_cmpgt_pd, _mm_cmple_pd, _mm_cmplt_pd, _mm_cvtsi32_si128,
	_mm_div_pd, _mm_loadu_pd, _mm_loadu_si128, _mm_max_epi16, _mm_max_epu8, _mm_max_pd, _mm_min_epi16, _mm_min_epu8,
	_mm_min_pd, _mm_movemask_epi8, _mm_movemask_pd, _mm_mul_epu32, _mm_mul_pd, _mm_mullo_epi16, _mm_or_pd,
	_mm_or_si128, _mm_packs_epi16, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi8, _mm_setzero_si128,
	_mm_shuffle_epi32, _mm_shufflehi_epi16, _mm_shufflelo_epi16, _mm_sll_epi16, _mm_sll_epi32, _mm_slli_epi64,
	_mm_slli_si128, _mm_sqrt_pd, _mm_sra_epi16, _mm_sra_epi32, _mm_srai_epi16, _mm_srai_epi32, _mm_srl_epi16,
	_mm_srl_epi32, _mm_srli_epi64, _mm_srli_si128, _mm_storeu_pd, _mm_storeu_si128, _mm_sub_epi16, _mm_sub_epi32,
	_mm_sub_epi64, _mm_sub_epi8, _mm_sub_pd, _mm_subs_epi16, _mm_subs_epi8, _mm_subs_epu16, _mm_subs_epu8,
	_mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpackhi_epi8, _mm_unpacklo_epi16,
	_mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8, _mm_xor_pd, _mm_xor_si128,
};

use super::super::super::{Feature, FeatureSet};
#[cfg(feature = "wider-bus-lift")]
use super::super::avx::avx::Avx;
use super::super::macros::{
	scalar_only_binop, simd_binop, simd_binop_fixed, simd_binop_lifted, simd_movemask, simd_shift_imm,
	simd_unop_fixed, simd_unop_imm,
};

// Every `Sse2`-token lift below uses `lift_target_feature = "sse2,avx"` and
// `lift_proof = Avx`:
// still issues 128-bit `sse2` instructions (two independent chains per
// iteration), `avx` gets them VEX-encoded (non-destructive 3-operand form,
// 16 registers instead of 8): benefits AVX-but-not-AVX2 hosts (Sandy/Ivy
// Bridge class). `#[target_feature(enable = ...)]` requires a literal, so
// this can't be a `const`.

/// Proof that SSE2 is available. Zero-sized, `Copy`.
///
/// Obtain via [`Sse2::detect`] or [`Sse2::from_features`], then call methods
/// on the token. On x86_64, SSE2 is always part of the ABI.
#[derive(Debug, Clone, Copy)]
pub struct Sse2(());

impl Sse2 {
	/// Probe once: `Some(token)` if SSE2 is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from a set you already have (e.g. [`crate::x86::detect_features`]).
	///
	/// Returns `None` if `Feature::Sse2` is missing.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse2).then_some(Sse2(()))
	}

	/// x86_64 ABI always has SSE2; skip CPUID. 32-bit x86 must use [`detect`].
	#[cfg(target_arch = "x86_64")]
	pub(crate) fn assume_baseline() -> Self {
		Sse2(())
	}

	/// `pslldq`: whole-register byte shift left by `IMM8`, zero-filled from
	/// the bottom (`IMM8 >= 16` yields all zero). Not a per-lane numeric
	/// shift ([`simd_shift_imm`] doesn't fit: that macro's scalar-remainder
	/// story assumes elementwise integer semantics; this is a single
	/// whole-vector byte move with no scalar equivalent).
	#[inline]
	pub fn slli_u8x16<const IMM8: i32>(self, a: [u8; 16]) -> [u8; 16] {
		unsafe { slli_si128::<IMM8>(&a) }
	}

	/// `psrldq`: whole-register byte shift right by `IMM8`, zero-filled from
	/// the top (`IMM8 >= 16` yields all zero). See [`Self::slli_u8x16`]'s doc
	/// for why this isn't [`simd_shift_imm`](super::super::macros).
	#[inline]
	pub fn srli_u8x16<const IMM8: i32>(self, a: [u8; 16]) -> [u8; 16] {
		unsafe { srli_si128::<IMM8>(&a) }
	}

	/// Signed saturating narrow (`packssdw`): `a`'s 4 lanes to the low 4
	/// `i16` lanes, `b`'s 4 lanes to the high 4. No width-changing-binop
	/// macro fits (`simd_binop` assumes matching in/out element types).
	#[inline]
	pub fn pack_i32x4_to_i16x8(self, a: [i32; 4], b: [i32; 4]) -> [i16; 8] {
		unsafe { packs_epi32(&a, &b) }
	}
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn packs_epi32(a: &[i32; 4], b: &[i32; 4]) -> [i16; 8] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let vr = _mm_packs_epi32(va, vb);
		let mut out = [0i16; 8];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn slli_si128<const IMM8: i32>(a: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vr = _mm_slli_si128::<IMM8>(va);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn srli_si128<const IMM8: i32>(a: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vr = _mm_srli_si128::<IMM8>(va);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpacklo_i16x8, intrinsic_fn = unpacklo_epi16_fixed,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpacklo_epi16,
	fixed_doc = "Interleaves the low 4 lanes of `a`/`b` (`punpcklwd`).",
}
simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpackhi_i16x8, intrinsic_fn = unpackhi_epi16_fixed,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpackhi_epi16,
	fixed_doc = "Interleaves the high 4 lanes of `a`/`b` (`punpckhwd`).",
}
simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpacklo_i32x4, intrinsic_fn = unpacklo_epi32_fixed,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpacklo_epi32,
	fixed_doc = "Interleaves the low 2 lanes of `a`/`b` (`punpckldq`).",
}
simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpackhi_i32x4, intrinsic_fn = unpackhi_epi32_fixed,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpackhi_epi32,
	fixed_doc = "Interleaves the high 2 lanes of `a`/`b` (`punpckhdq`).",
}
simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpacklo_i64x2, intrinsic_fn = unpacklo_epi64_fixed,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpacklo_epi64,
	fixed_doc = "Interleaves the low lane of `a`/`b` (`punpcklqdq`): `[a[0], b[0]]`.",
}
simd_binop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = unpackhi_i64x2, intrinsic_fn = unpackhi_epi64_fixed,
	width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_unpackhi_epi64,
	fixed_doc = "Interleaves the high lane of `a`/`b` (`punpckhqdq`): `[a[1], b[1]]`.",
}

simd_unop_imm! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = shufflelo_i16x8, intrinsic_fn = shufflelo_epi16_fixed,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shufflelo_epi16,
	fixed_doc = "4-way shuffle of `a`'s low 4 lanes by `IMM8`; high 4 lanes pass through (`pshuflw`).",
}
simd_unop_imm! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = shufflehi_i16x8, intrinsic_fn = shufflehi_epi16_fixed,
	width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shufflehi_epi16,
	fixed_doc = "4-way shuffle of `a`'s high 4 lanes by `IMM8`; low 4 lanes pass through (`pshufhw`).",
}
simd_unop_imm! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = shuffle_i32x4, intrinsic_fn = shuffle_epi32_fixed,
	width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = _mm_shuffle_epi32,
	fixed_doc = "4-way shuffle of `a`'s lanes by `IMM8` (`pshufd`).",
}

macro_rules! sse2_f64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 2, elem = f64, vec = __m128d, loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_i32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		sse2_i32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Sse2, lift_target_feature = "sse2,avx",
			lifted_fn = $lifted_fn, lift_proof = Avx,
			width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		sse2_u32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Sse2, lift_target_feature = "sse2,avx",
			lifted_fn = $lifted_fn, lift_proof = Avx,
			width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse2_f64_binop!(
	add_f64x2, add_f64_slice, addpd, _mm_add_pd, |x, y| x + y,
	"`a + b` per lane, double precision (`addpd`).",
	"`out[i] = a[i] + b[i]`. 2-wide `add_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	sub_f64x2, sub_f64_slice, subpd, _mm_sub_pd, |x, y| x - y,
	"`a - b` per lane, double precision (`subpd`).",
	"`out[i] = a[i] - b[i]`. 2-wide `sub_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	mul_f64x2, mul_f64_slice, mulpd, _mm_mul_pd, |x, y| x * y,
	"`a * b` per lane, double precision (`mulpd`).",
	"`out[i] = a[i] * b[i]`. 2-wide `mul_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	div_f64x2, div_f64_slice, divpd, _mm_div_pd, |x, y| x / y,
	"`a / b` per lane, double precision (`divpd`).",
	"`out[i] = a[i] / b[i]`. 2-wide `div_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	min_f64x2, min_f64_slice, minpd, _mm_min_pd, |x, y| x.min(y),
	"Per-lane min, double precision (`minpd`). NaN: second-operand-on-NaN, not IEEE `f64::min`.",
	"`out[i] = min(a[i], b[i])`. 2-wide `min_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	max_f64x2, max_f64_slice, maxpd, _mm_max_pd, |x, y| x.max(y),
	"Per-lane max, double precision (`maxpd`). NaN: second-operand-on-NaN, not IEEE `f64::max`.",
	"`out[i] = max(a[i], b[i])`. 2-wide `max_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	and_f64x2, and_f64_slice, andpd, _mm_and_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()),
	"Bitwise AND per lane, double precision (`andpd`).",
	"`out[i] = a[i] & b[i]` bitwise. 2-wide `and_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	or_f64x2, or_f64_slice, orpd, _mm_or_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()),
	"Bitwise OR per lane, double precision (`orpd`).",
	"`out[i] = a[i] | b[i]` bitwise. 2-wide `or_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	xor_f64x2, xor_f64_slice, xorpd, _mm_xor_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()),
	"Bitwise XOR per lane, double precision (`xorpd`).",
	"`out[i] = a[i] ^ b[i]` bitwise. 2-wide `xor_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	andnot_f64x2, andnot_f64_slice, andnpd, _mm_andnot_pd, |x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()),
	"Bitwise `!a & b` per lane, double precision (`andnpd`).",
	"`out[i] = !a[i] & b[i]` bitwise. 2-wide `andnot_f64x2` chunks, scalar remainder."
);
sse2_f64_binop!(
	cmpeq_f64x2, cmpeq_f64_slice, cmpeqpd, _mm_cmpeq_pd,
	|x, y| if x == y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane equality mask (`cmpeqpd`): all-1s bits if equal, else 0. NaN never equals.",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 2-wide chunks, scalar remainder."
);
sse2_f64_binop!(
	cmplt_f64x2, cmplt_f64_slice, cmpltpd, _mm_cmplt_pd,
	|x, y| if x < y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-than mask (`cmpltpd`): all-1s bits if `a<b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 2-wide chunks, scalar remainder."
);
sse2_f64_binop!(
	cmple_f64x2, cmple_f64_slice, cmplepd, _mm_cmple_pd,
	|x, y| if x <= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-equal mask (`cmplepd`): all-1s bits if `a<=b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 2-wide chunks, scalar remainder."
);
sse2_f64_binop!(
	cmpgt_f64x2, cmpgt_f64_slice, cmpgtpd, _mm_cmpgt_pd,
	|x, y| if x > y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-than mask (`cmpgtpd`): all-1s bits if `a>b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 2-wide chunks, scalar remainder."
);
sse2_f64_binop!(
	cmpge_f64x2, cmpge_f64_slice, cmpgepd, _mm_cmpge_pd,
	|x, y| if x >= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-equal mask (`cmpgepd`): all-1s bits if `a>=b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 2-wide chunks, scalar remainder."
);

// Fixed-width only (no `_slice`/`auto`): the HW op needs no libm, but a
// `_slice` remainder closure would need `f64::sqrt`, unavailable under
// `no_std` without an external libm dependency. No `rcp`/`rsqrt` here: the
// ISA never defined packed-double reciprocal/rsqrt approximations below
// AVX-512 (`rcpps`/`rsqrtps` only ever exist for `ps`, not `pd`).
simd_unop_fixed! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = sqrt_f64x2, intrinsic_fn = sqrtpd,
	width = 2, elem = f64, vec = __m128d,
	loadu = _mm_loadu_pd, storeu = _mm_storeu_pd, intrinsic = _mm_sqrt_pd,
	fixed_doc = "Correctly-rounded per-lane sqrt (`sqrtpd`). Fixed-width only, see module doc.",
}

sse2_i32_binop!(
	add_i32x4, add_i32_slice, paddd, _mm_add_epi32, |x: i32, y: i32| x.wrapping_add(y),
	"`a + b` per lane, 32-bit integers, wrapping (`paddd`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 4-wide `add_i32x4` chunks, scalar remainder.",
	lifted_fn = add_i32_slice_lifted,
);
sse2_i32_binop!(
	sub_i32x4, sub_i32_slice, psubd, _mm_sub_epi32, |x: i32, y: i32| x.wrapping_sub(y),
	"`a - b` per lane, 32-bit integers, wrapping (`psubd`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 4-wide `sub_i32x4` chunks, scalar remainder.",
	lifted_fn = sub_i32_slice_lifted,
);
scalar_only_binop! {
	token = Sse2,
	fixed_fn = div_i32x4, slice_fn = div_i32_slice,
	width = 4, elem = i32,
	scalar = |x: i32, y: i32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor or `i32::MIN / -1`, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

sse2_u32_binop!(
	add_u32x4, add_u32_slice, paddd_u, _mm_add_epi32, |x: u32, y: u32| x.wrapping_add(y),
	"`a + b` per lane, 32-bit integers, wrapping (`paddd`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 4-wide `add_u32x4` chunks, scalar remainder.",
	lifted_fn = add_u32_slice_lifted,
);
sse2_u32_binop!(
	sub_u32x4, sub_u32_slice, psubd_u, _mm_sub_epi32, |x: u32, y: u32| x.wrapping_sub(y),
	"`a - b` per lane, 32-bit integers, wrapping (`psubd`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 4-wide `sub_u32x4` chunks, scalar remainder.",
	lifted_fn = sub_u32_slice_lifted,
);
scalar_only_binop! {
	token = Sse2,
	fixed_fn = div_u32x4, slice_fn = div_u32_slice,
	width = 4, elem = u32,
	scalar = |x: u32, y: u32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

sse2_i32_binop!(
	and_i32x4, and_i32_slice, pand, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = and_i32_slice_lifted,
);
sse2_i32_binop!(
	or_i32x4, or_i32_slice, por, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = or_i32_slice_lifted,
);
sse2_i32_binop!(
	xor_i32x4, xor_i32_slice, pxor, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = xor_i32_slice_lifted,
);
sse2_i32_binop!(
	andnot_i32x4, andnot_i32_slice, pandn, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = andnot_i32_slice_lifted,
);
sse2_i32_binop!(
	cmpeq_i32x4, cmpeq_i32_slice, pcmpeqd, _mm_cmpeq_epi32,
	|x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`pcmpeqd`): all-1s if equal, else 0. Not a bool vector.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder.",
	lifted_fn = cmpeq_i32_slice_lifted,
);

sse2_u32_binop!(
	and_u32x4, and_u32_slice, pand_u, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = and_u32_slice_lifted,
);
sse2_u32_binop!(
	or_u32x4, or_u32_slice, por_u, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = or_u32_slice_lifted,
);
sse2_u32_binop!(
	xor_u32x4, xor_u32_slice, pxor_u, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = xor_u32_slice_lifted,
);
sse2_u32_binop!(
	andnot_u32x4, andnot_u32_slice, pandn_u, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 4-wide chunks, scalar remainder.",
	lifted_fn = andnot_u32_slice_lifted,
);
sse2_u32_binop!(
	cmpeq_u32x4, cmpeq_u32_slice, pcmpeqd_u, _mm_cmpeq_epi32,
	|x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`pcmpeqd`): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder.",
	lifted_fn = cmpeq_u32_slice_lifted,
);

// i64/u64 add/sub: native SSE2 (min/max stay 512-only; need VL below that).
macro_rules! sse2_i64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 2, elem = i64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 2, elem = u64, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse2_i64_binop!(
	add_i64x2, add_i64_slice, paddq, _mm_add_epi64, |x: i64, y: i64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddq`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 2-wide `add_i64x2` chunks, scalar remainder."
);
sse2_i64_binop!(
	sub_i64x2, sub_i64_slice, psubq, _mm_sub_epi64, |x: i64, y: i64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubq`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 2-wide `sub_i64x2` chunks, scalar remainder."
);
sse2_u64_binop!(
	add_u64x2, add_u64_slice, paddq_u, _mm_add_epi64, |x: u64, y: u64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddq`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 2-wide `add_u64x2` chunks, scalar remainder."
);
sse2_u64_binop!(
	sub_u64x2, sub_u64_slice, psubq_u, _mm_sub_epi64, |x: u64, y: u64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubq`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 2-wide `sub_u64x2` chunks, scalar remainder."
);
// i64/u64 bitwise: same si128 ops as i32 (view-only).
sse2_i64_binop!(
	and_i64x2, and_i64_slice, pand_i64, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 2-wide chunks, scalar remainder."
);
sse2_i64_binop!(
	or_i64x2, or_i64_slice, por_i64, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 2-wide chunks, scalar remainder."
);
sse2_i64_binop!(
	xor_i64x2, xor_i64_slice, pxor_i64, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 2-wide chunks, scalar remainder."
);
sse2_i64_binop!(
	andnot_i64x2, andnot_i64_slice, pandn_i64, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 2-wide chunks, scalar remainder."
);
sse2_u64_binop!(
	and_u64x2, and_u64_slice, pand_u64, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 2-wide chunks, scalar remainder."
);
sse2_u64_binop!(
	or_u64x2, or_u64_slice, por_u64, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 2-wide chunks, scalar remainder."
);
sse2_u64_binop!(
	xor_u64x2, xor_u64_slice, pxor_u64, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 2-wide chunks, scalar remainder."
);
sse2_u64_binop!(
	andnot_u64x2, andnot_u64_slice, pandn_u64, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 2-wide chunks, scalar remainder."
);

// i64/u64 mullo: schoolbook via pmuludq halves (no native 64x64->64 multiply below
// AVX-512DQ). i64 abs: branchless sign-broadcast (shuffle+srai; no native 64-bit
// arithmetic shift below AVX-512, and no cmpgt/blendv needed either).
impl Sse2 {
	/// Per-lane low-64-bit multiply, wrapping: schoolbook decomposition into 32-bit
	/// half-lane products (`pmuludq`+shifts+adds; no native 64x64->64 multiply exists
	/// below AVX-512DQ).
	#[inline]
	pub fn mullo_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		unsafe { mullo_u64x2_composed(&a, &b) }
	}

	/// `out[i] = a[i].wrapping_mul(b[i])`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn mullo_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
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
			oc.copy_from_slice(&self.mullo_u64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.wrapping_mul(y);
		}
	}

	/// Signed view of [`mullo_u64x2`](Self::mullo_u64x2): wrapping low-64 multiply is
	/// bit-identical for signed and unsigned operands.
	#[inline]
	pub fn mullo_i64x2(self, a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
		let au: [u64; 2] = core::array::from_fn(|i| a[i] as u64);
		let bu: [u64; 2] = core::array::from_fn(|i| b[i] as u64);
		let r = self.mullo_u64x2(au, bu);
		core::array::from_fn(|i| r[i] as i64)
	}

	/// `out[i] = a[i].wrapping_mul(b[i])`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn mullo_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
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
			oc.copy_from_slice(&self.mullo_i64x2(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.wrapping_mul(y);
		}
	}

	/// Per-lane absolute value, wrapping (`i64::MIN` stays `i64::MIN`): branchless
	/// sign-broadcast mask (`shuffle`+`srai`) + `(a XOR mask) - mask`. No native 64-bit
	/// arithmetic shift exists below AVX-512, so the mask is built from two 32-bit halves.
	#[inline]
	pub fn abs_i64x2(self, a: [i64; 2]) -> [i64; 2] {
		unsafe { abs_i64x2_composed(&a) }
	}

	/// `out[i] = a[i].wrapping_abs()`. 2-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub(crate) fn abs_i64_slice(self, a: &[i64], out: &mut [i64]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(2);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(2);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i64; 2] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.abs_i64x2(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_abs();
		}
	}
}

/// Schoolbook 64x64->64 multiply (`a*b mod 2^64`) via 32-bit half-lane products
/// (`pmuludq`). Cross-term handling uses `psllq` to drop overflowing bits.
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn mullo_u64x2_composed(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let vb: __m128i = _mm_loadu_si128(b.as_ptr().cast());
		let ac = _mm_mul_epu32(va, vb);
		let a_hi = _mm_srli_epi64::<32>(va);
		let b_hi = _mm_srli_epi64::<32>(vb);
		let cross = _mm_add_epi64(_mm_mul_epu32(a_hi, vb), _mm_mul_epu32(va, b_hi));
		let vr = _mm_add_epi64(ac, _mm_slli_epi64::<32>(cross));
		let mut out = [0u64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Branchless 64-bit abs via sign-bit broadcast and `srai`-derived mask.
/// Implements `i64::wrapping_abs` without branches.
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn abs_i64x2_composed(a: &[i64; 2]) -> [i64; 2] {
	unsafe {
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		// 0xF5 = _MM_SHUFFLE(3,3,1,1): duplicate each lane's high dword into both dwords.
		let hi_dup = _mm_shuffle_epi32::<0xF5>(va);
		let mask = _mm_srai_epi32::<31>(hi_dup);
		let vr = _mm_sub_epi64(_mm_xor_si128(va, mask), mask);
		let mut out = [0i64; 2];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

sse2_i32_binop!(
	cmpgt_i32x4, cmpgt_i32_slice, pcmpgtd, _mm_cmpgt_epi32,
	|x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`pcmpgtd`): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 4-wide chunks, scalar remainder.",
	lifted_fn = cmpgt_i32_slice_lifted,
);

impl Sse2 {
	/// Unsigned greater-than mask (all-1s if `a>b`). Sign-bit flip + [`cmpgt_i32x4`].
	#[inline]
	pub fn cmpgt_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		let ai: [i32; 4] = core::array::from_fn(|i| (a[i] ^ 0x8000_0000) as i32);
		let bi: [i32; 4] = core::array::from_fn(|i| (b[i] ^ 0x8000_0000) as i32);
		let r = self.cmpgt_i32x4(ai, bi);
		core::array::from_fn(|i| r[i] as u32)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u32x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i32x4`].
	#[inline]
	pub fn cmplt_i32x4(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
		self.cmpgt_i32x4(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Sse2::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i32x4`].
	#[inline]
	pub fn cmple_i32x4(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
		let gt = self.cmpgt_i32x4(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Sse2::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i32x4`].
	#[inline]
	pub fn cmpge_i32x4(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
		let lt = self.cmplt_i32x4(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Sse2::cmplt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmplt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u32x4`].
	#[inline]
	pub fn cmplt_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		self.cmpgt_u32x4(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmpgt_u32_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u32x4`].
	#[inline]
	pub fn cmple_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		let gt = self.cmpgt_u32x4(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmpgt_u32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u32x4`].
	#[inline]
	pub fn cmpge_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		let lt = self.cmplt_u32x4(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmplt_u32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	// min/max via cmpgt + bitops (no blendv / no pminsd below SSE4.1):
	//   min(a,b) = (!gt & a) | (gt & b)   where gt = cmpgt(a,b)
	//   max(a,b) = (gt & a) | (!gt & b)

	/// Per-lane signed min: composed `cmpgt`+and/or/andnot (no native `pminsd` at SSE2).
	#[inline]
	pub fn min_i32x4(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
		let gt = self.cmpgt_i32x4(a, b);
		self.or_i32x4(self.andnot_i32x4(gt, a), self.and_i32x4(gt, b))
	}

	/// `out[i] = min(a[i], b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i32; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [i32; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_i32x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane signed max: composed `cmpgt`+and/or/andnot (no native `pmaxsd` at SSE2).
	#[inline]
	pub fn max_i32x4(self, a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
		let gt = self.cmpgt_i32x4(a, b);
		self.or_i32x4(self.and_i32x4(gt, a), self.andnot_i32x4(gt, b))
	}

	/// `out[i] = max(a[i], b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i32; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [i32; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_i32x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}

	/// Per-lane unsigned min: [`cmpgt_u32x4`] + same bitops as [`min_i32x4`].
	#[inline]
	pub fn min_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		let gt = self.cmpgt_u32x4(a, b);
		self.or_u32x4(self.andnot_u32x4(gt, a), self.and_u32x4(gt, b))
	}

	/// `out[i] = min(a[i], b[i])` (`u32`). 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_u32x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane unsigned max: [`cmpgt_u32x4`] + same bitops as [`max_i32x4`].
	#[inline]
	pub fn max_u32x4(self, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		let gt = self.cmpgt_u32x4(a, b);
		self.or_u32x4(self.and_u32x4(gt, a), self.andnot_u32x4(gt, b))
	}

	/// `out[i] = max(a[i], b[i])` (`u32`). 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_u32x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}
}

macro_rules! sse2_i32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = i32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = u32, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse2_i32_shift_imm!(
	shl_i32x4, shl_i32_slice, pslld, _mm_sll_epi32, |x: i32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`psll`, uniform count).",
	"`out[i] = a[i] << IMM`. 4-wide chunks, scalar remainder."
);
sse2_i32_shift_imm!(
	shr_i32x4, shr_i32_slice, psrld, _mm_srl_epi32, |x: i32, imm| ((x as u32).wrapping_shr(imm)) as i32,
	"`a >> IMM` logical per lane (`psrl`, uniform count).",
	"`out[i] = a[i] logical >> IMM`. 4-wide chunks, scalar remainder."
);
sse2_i32_shift_imm!(
	sra_i32x4, sra_i32_slice, psrad, _mm_sra_epi32, |x: i32, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`psra`, uniform count).",
	"`out[i] = a[i] arithmetic >> IMM`. 4-wide chunks, scalar remainder."
);
sse2_u32_shift_imm!(
	shl_u32x4, shl_u32_slice, pslld_u, _mm_sll_epi32, |x: u32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`psll`, uniform count).",
	"`out[i] = a[i] << IMM`. 4-wide chunks, scalar remainder."
);
sse2_u32_shift_imm!(
	shr_u32x4, shr_u32_slice, psrld_u, _mm_srl_epi32, |x: u32, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`psrl`, uniform count).",
	"`out[i] = a[i] >> IMM`. 4-wide chunks, scalar remainder."
);

macro_rules! sse2_i8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_i16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse2_i8_binop!(
	add_i8x16, add_i8_slice, paddb, _mm_add_epi8, |x: i8, y: i8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddb`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	sub_i8x16, sub_i8_slice, psubb, _mm_sub_epi8, |x: i8, y: i8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubb`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	adds_i8x16, adds_i8_slice, paddsb, _mm_adds_epi8, |x: i8, y: i8| x.saturating_add(y),
	"`a + b` per lane, saturating (`paddsb`).",
	"`out[i] = a[i].saturating_add(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	subs_i8x16, subs_i8_slice, psubsb, _mm_subs_epi8, |x: i8, y: i8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`psubsb`).",
	"`out[i] = a[i].saturating_sub(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	cmpeq_i8x16, cmpeq_i8_slice, pcmpeqb, _mm_cmpeq_epi8, |x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`pcmpeqb`): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	cmpgt_i8x16, cmpgt_i8_slice, pcmpgtb, _mm_cmpgt_epi8, |x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`pcmpgtb`): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	and_i8x16, and_i8_slice, pand_i8, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	or_i8x16, or_i8_slice, por_i8, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	xor_i8x16, xor_i8_slice, pxor_i8, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
sse2_i8_binop!(
	andnot_i8x16, andnot_i8_slice, pandn_i8, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
// mul_i8/u8, shl/shr/sra_i8/u8: no native 8-bit SIMD multiply or
// byte-granularity shift exists on x86 at any tier, but composed forms beat
// a scalar loop: see `mullo_epi8x16_composed`/`shl_u8x16_composed`/
// `shr_u8x16_composed`/`sra_i8x16_composed` below. `vis = pub` kept for API
// compatibility with the prior scalar-only methods; additionally cascaded
// via `auto_up::mul_i8`/`shl_i8`/etc. (`x86/auto_up.rs`/`auto_down.rs`).
simd_binop! {
	token = Sse2, vis = pub, target_feature = "sse2",
	fixed_fn = mul_i8x16, slice_fn = mul_i8_slice, intrinsic_fn = mul_i8x16_intrinsic,
	width = 16, elem = i8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = mullo_epi8x16_composed, scalar = |x: i8, y: i8| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, wrapping: composed via zero-extend+`pmullw`+`packuswb` (no native 8-bit SIMD multiply on x86 at any tier).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 16-wide chunks, scalar remainder.",
}

impl Sse2 {
	/// Per-lane logical left shift by `IMM` (masked to `IMM & 7`, matching
	/// [`u8::wrapping_shl`]), wrapping: composed via widening to 16-bit lanes
	/// (`psllw`, register-count form) + a byte-repeated mask that clears the
	/// bits shifted in from each byte's neighbor (no native byte-granularity
	/// shift exists on x86 SIMD at any tier).
	#[inline]
	pub fn shl_u8x16<const IMM: u32>(self, a: [u8; 16]) -> [u8; 16] {
		unsafe { shl_u8x16_composed::<IMM>(&a) }
	}

	/// `out[i] = a[i].wrapping_shl(IMM)`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shl_u8_slice<const IMM: u32>(self, a: &[u8], out: &mut [u8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u8; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shl_u8x16::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shl(IMM);
		}
	}

	/// Signed view of [`shl_u8x16`](Self::shl_u8x16): left shift doesn't depend
	/// on operand signedness.
	#[inline]
	pub fn shl_i8x16<const IMM: u32>(self, a: [i8; 16]) -> [i8; 16] {
		let au: [u8; 16] = core::array::from_fn(|i| a[i] as u8);
		let r = self.shl_u8x16::<IMM>(au);
		core::array::from_fn(|i| r[i] as i8)
	}

	/// `out[i] = a[i].wrapping_shl(IMM)`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shl_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shl_i8x16::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shl(IMM);
		}
	}

	/// Per-lane logical right shift by `IMM` (masked to `IMM & 7`, matching
	/// [`u8::wrapping_shr`]): composed via widening to 16-bit lanes (`psrlw`,
	/// register-count form) + a byte-repeated mask that clears the bits
	/// shifted in from each byte's neighbor. No native byte-granularity shift
	/// exists on x86 SIMD at any tier.
	#[inline]
	pub fn shr_u8x16<const IMM: u32>(self, a: [u8; 16]) -> [u8; 16] {
		unsafe { shr_u8x16_composed::<IMM>(&a) }
	}

	/// `out[i] = a[i].wrapping_shr(IMM)`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shr_u8_slice<const IMM: u32>(self, a: &[u8], out: &mut [u8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u8; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shr_u8x16::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shr(IMM);
		}
	}

	/// Logical (unsigned) view of [`shr_u8x16`](Self::shr_u8x16): matches the
	/// prior scalar-only `shr_i8x16`'s semantics (`(a as u8) >> IMM as i8`).
	#[inline]
	pub fn shr_i8x16<const IMM: u32>(self, a: [i8; 16]) -> [i8; 16] {
		let au: [u8; 16] = core::array::from_fn(|i| a[i] as u8);
		let r = self.shr_u8x16::<IMM>(au);
		core::array::from_fn(|i| r[i] as i8)
	}

	/// `out[i] = ((a[i] as u8).wrapping_shr(IMM)) as i8`. 16-wide chunks,
	/// scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shr_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shr_i8x16::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = ((x as u8).wrapping_shr(IMM)) as i8;
		}
	}

	/// Per-lane arithmetic right shift by `IMM` (masked to `IMM & 7`, matching
	/// [`i8::wrapping_shr`]), wrapping: composed via per-byte sign extension
	/// (`punpcklbw`/`punpckhbw` with self + `psraw` by the literal 8) then
	/// arithmetic-shifting the sign-extended words and packing back
	/// (`packsswb`; never saturates, the shifted value always fits in `i8`).
	/// No native byte-granularity shift exists on x86 SIMD at any tier.
	#[inline]
	pub fn sra_i8x16<const IMM: u32>(self, a: [i8; 16]) -> [i8; 16] {
		unsafe { sra_i8x16_composed::<IMM>(&a) }
	}

	/// `out[i] = a[i].wrapping_shr(IMM)`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn sra_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 16] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.sra_i8x16::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shr(IMM);
		}
	}
}

/// Composed 8-bit wrapping multiply (no native form on x86 at any tier):
/// zero-extend each half to 16-bit lanes (`punpcklbw`/`punpckhbw`), multiply
/// (`pmullw`), mask each product's low byte, then pack back down
/// (`packuswb`; masked values are always `0..=255` so this never saturates).
/// Bit-identical for `i8`/`u8` (wrapping multiply's low byte doesn't depend on
/// operand signedness), so both `mul_i8x16` and `mul_u8x16` share this.
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn mullo_epi8x16_composed(a: __m128i, b: __m128i) -> __m128i {
	// No `unsafe {}` needed: every op here is register-to-register (no raw
	// pointer deref), safe to call directly given the matching `#[target_feature]`.
	let zero = _mm_setzero_si128();
	let a_lo = _mm_unpacklo_epi8(a, zero);
	let a_hi = _mm_unpackhi_epi8(a, zero);
	let b_lo = _mm_unpacklo_epi8(b, zero);
	let b_hi = _mm_unpackhi_epi8(b, zero);
	let p_lo = _mm_mullo_epi16(a_lo, b_lo);
	let p_hi = _mm_mullo_epi16(a_hi, b_hi);
	let mask = _mm_set1_epi16(0x00FF);
	_mm_packus_epi16(_mm_and_si128(p_lo, mask), _mm_and_si128(p_hi, mask))
}

/// Composed per-byte logical left shift: widen to `psllw` (leaks bits across
/// each 16-bit lane's byte boundary), then mask with a byte-repeated
/// `0xFF<<(IMM&7)` pattern that clears exactly the leaked bits regardless of
/// which neighboring byte they came from.
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn shl_u8x16_composed<const IMM: u32>(a: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let shift = IMM & 7;
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let count = _mm_cvtsi32_si128(shift as i32);
		let wide = _mm_sll_epi16(va, count);
		let mask_byte = ((0xFFu32 << shift) & 0xFF) as u8;
		let mask = _mm_set1_epi8(mask_byte as i8);
		let vr = _mm_and_si128(wide, mask);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Composed per-byte logical right shift: mirror of
/// [`shl_u8x16_composed`] via `psrlw` + a `0xFF>>(IMM&7)` byte mask.
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn shr_u8x16_composed<const IMM: u32>(a: &[u8; 16]) -> [u8; 16] {
	unsafe {
		let shift = IMM & 7;
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let count = _mm_cvtsi32_si128(shift as i32);
		let wide = _mm_srl_epi16(va, count);
		let mask_byte = (0xFFu32 >> shift) as u8;
		let mask = _mm_set1_epi8(mask_byte as i8);
		let vr = _mm_and_si128(wide, mask);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Composed per-byte arithmetic right shift: sign-extend each byte to a full
/// 16-bit lane (`punpcklbw`/`punpckhbw` with self, then `psraw` by the
/// literal 8: a compile-time constant, not derived from `IMM`, so it can use
/// the true immediate form), arithmetic-shift by `IMM&7` (register-count
/// `psraw`, since `IMM` is a generic parameter and can't be threaded into
/// another intrinsic's own const-generic slot on stable Rust: same wall
/// `alignr_u8x32_full` hit), then pack back down (`packsswb`; the
/// sign-extended-then-shifted value always fits in `i8`, so this never
/// saturates).
///
/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sra_i8x16_composed<const IMM: u32>(a: &[i8; 16]) -> [i8; 16] {
	unsafe {
		let shift = IMM & 7;
		let va: __m128i = _mm_loadu_si128(a.as_ptr().cast());
		let lo_ext = _mm_srai_epi16::<8>(_mm_unpacklo_epi8(va, va));
		let hi_ext = _mm_srai_epi16::<8>(_mm_unpackhi_epi8(va, va));
		let count = _mm_cvtsi32_si128(shift as i32);
		let lo_shifted = _mm_sra_epi16(lo_ext, count);
		let hi_shifted = _mm_sra_epi16(hi_ext, count);
		let vr = _mm_packs_epi16(lo_shifted, hi_shifted);
		let mut out = [0i8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), vr);
		out
	}
}

sse2_u8_binop!(
	add_u8x16, add_u8_slice, paddb_u, _mm_add_epi8, |x: u8, y: u8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddb`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	sub_u8x16, sub_u8_slice, psubb_u, _mm_sub_epi8, |x: u8, y: u8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubb`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	adds_u8x16, adds_u8_slice, paddusb, _mm_adds_epu8, |x: u8, y: u8| x.saturating_add(y),
	"`a + b` per lane, saturating (`paddusb`).",
	"`out[i] = a[i].saturating_add(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	subs_u8x16, subs_u8_slice, psubusb, _mm_subs_epu8, |x: u8, y: u8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`psubusb`).",
	"`out[i] = a[i].saturating_sub(b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	cmpeq_u8x16, cmpeq_u8_slice, pcmpeqb_u, _mm_cmpeq_epi8, |x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`pcmpeqb`): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	and_u8x16, and_u8_slice, pand_u8, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	or_u8x16, or_u8_slice, por_u8, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	xor_u8x16, xor_u8_slice, pxor_u8, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	andnot_u8x16, andnot_u8_slice, pandn_u8, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	min_u8x16, min_u8_slice, pminub, _mm_min_epu8, |x, y| x.min(y),
	"Per-lane unsigned min (`pminub`).",
	"`out[i] = min(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	max_u8x16, max_u8_slice, pmaxub, _mm_max_epu8, |x, y| x.max(y),
	"Per-lane unsigned max (`pmaxub`).",
	"`out[i] = max(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
sse2_u8_binop!(
	avg_u8x16, avg_u8_slice, pavgb, _mm_avg_epu8, |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8,
	"Per-lane rounded unsigned average, `(a+b+1)/2` (`pavgb`). No signed form exists in the ISA.",
	"`out[i] = (a[i] as u16 + b[i] as u16 + 1) / 2`. 16-wide chunks, scalar remainder."
);
// Composed (see `mullo_epi8x16_composed`'s doc above); `vis = pub` for the
// same API-compatibility + also-`auto`-cascaded reason as `mul_i8x16`.
// `shl_u8x16`/`shr_u8x16` themselves are defined in the `impl Sse2` block
// above (shared by both `i8` and `u8`: shift doesn't depend on signedness).
simd_binop! {
	token = Sse2, vis = pub, target_feature = "sse2",
	fixed_fn = mul_u8x16, slice_fn = mul_u8_slice, intrinsic_fn = mul_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
	intrinsic = mullo_epi8x16_composed, scalar = |x: u8, y: u8| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, wrapping: composed via zero-extend+`pmullw`+`packuswb` (no native 8-bit SIMD multiply on x86 at any tier).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 16-wide chunks, scalar remainder.",
}

sse2_i16_binop!(
	add_i16x8, add_i16_slice, paddw, _mm_add_epi16, |x: i16, y: i16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddw`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	sub_i16x8, sub_i16_slice, psubw, _mm_sub_epi16, |x: i16, y: i16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubw`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	adds_i16x8, adds_i16_slice, paddsw, _mm_adds_epi16, |x: i16, y: i16| x.saturating_add(y),
	"`a + b` per lane, saturating (`paddsw`).",
	"`out[i] = a[i].saturating_add(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	subs_i16x8, subs_i16_slice, psubsw, _mm_subs_epi16, |x: i16, y: i16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`psubsw`).",
	"`out[i] = a[i].saturating_sub(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	cmpeq_i16x8, cmpeq_i16_slice, pcmpeqw, _mm_cmpeq_epi16, |x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`pcmpeqw`): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	cmpgt_i16x8, cmpgt_i16_slice, pcmpgtw, _mm_cmpgt_epi16, |x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`pcmpgtw`): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	and_i16x8, and_i16_slice, pand_i16, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	or_i16x8, or_i16_slice, por_i16, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	xor_i16x8, xor_i16_slice, pxor_i16, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	andnot_i16x8, andnot_i16_slice, pandn_i16, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	min_i16x8, min_i16_slice, pminsw, _mm_min_epi16, |x, y| x.min(y),
	"Per-lane signed min (`pminsw`).",
	"`out[i] = min(a[i], b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	max_i16x8, max_i16_slice, pmaxsw, _mm_max_epi16, |x, y| x.max(y),
	"Per-lane signed max (`pmaxsw`).",
	"`out[i] = max(a[i], b[i])`. 8-wide chunks, scalar remainder."
);
sse2_i16_binop!(
	mul_i16x8, mul_i16_slice, pmullw, _mm_mullo_epi16, |x: i16, y: i16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`pmullw`).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide chunks, scalar remainder."
);

sse2_u16_binop!(
	add_u16x8, add_u16_slice, paddw_u, _mm_add_epi16, |x: u16, y: u16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`paddw`).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	sub_u16x8, sub_u16_slice, psubw_u, _mm_sub_epi16, |x: u16, y: u16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`psubw`).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	adds_u16x8, adds_u16_slice, paddusw, _mm_adds_epu16, |x: u16, y: u16| x.saturating_add(y),
	"`a + b` per lane, saturating (`paddusw`).",
	"`out[i] = a[i].saturating_add(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	subs_u16x8, subs_u16_slice, psubusw, _mm_subs_epu16, |x: u16, y: u16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`psubusw`).",
	"`out[i] = a[i].saturating_sub(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	cmpeq_u16x8, cmpeq_u16_slice, pcmpeqw_u, _mm_cmpeq_epi16, |x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`pcmpeqw`): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	and_u16x8, and_u16_slice, pand_u16, _mm_and_si128, |x, y| x & y,
	"`a & b` per lane (`pand`).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	or_u16x8, or_u16_slice, por_u16, _mm_or_si128, |x, y| x | y,
	"`a | b` per lane (`por`).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	xor_u16x8, xor_u16_slice, pxor_u16, _mm_xor_si128, |x, y| x ^ y,
	"`a ^ b` per lane (`pxor`).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	andnot_u16x8, andnot_u16_slice, pandn_u16, _mm_andnot_si128, |x, y| !x & y,
	"`!a & b` per lane (`pandn`).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	mul_u16x8, mul_u16_slice, pmullw_u, _mm_mullo_epi16, |x: u16, y: u16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`pmullw`).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide chunks, scalar remainder."
);
sse2_u16_binop!(
	avg_u16x8, avg_u16_slice, pavgw, _mm_avg_epu16, |x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16,
	"Per-lane rounded unsigned average, `(a+b+1)/2` (`pavgw`). No signed form exists in the ISA.",
	"`out[i] = (a[i] as u32 + b[i] as u32 + 1) / 2`. 8-wide chunks, scalar remainder."
);

// Narrow ordering: signed native `pcmpgtb`/`pcmpgtw`; unsigned sign-bit flip; lt/le/ge = swap/NOT.
impl Sse2 {
	/// Unsigned greater-than mask (all-1s if `a>b`). Sign-bit flip + [`cmpgt_i8x16`].
	#[inline]
	pub fn cmpgt_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		let ai: [i8; 16] = core::array::from_fn(|i| (a[i] ^ 0x80) as i8);
		let bi: [i8; 16] = core::array::from_fn(|i| (b[i] ^ 0x80) as i8);
		let r = self.cmpgt_i8x16(ai, bi);
		core::array::from_fn(|i| r[i] as u8)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u8; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [u8; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u8x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Unsigned greater-than mask (all-1s if `a>b`). Sign-bit flip + [`cmpgt_i16x8`].
	#[inline]
	pub fn cmpgt_u16x8(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
		let ai: [i16; 8] = core::array::from_fn(|i| (a[i] ^ 0x8000) as i16);
		let bi: [i16; 8] = core::array::from_fn(|i| (b[i] ^ 0x8000) as i16);
		let r = self.cmpgt_i16x8(ai, bi);
		core::array::from_fn(|i| r[i] as u16)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u16x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i8x16`].
	#[inline]
	pub fn cmplt_i8x16(self, a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
		self.cmpgt_i8x16(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Sse2::cmpgt_i8_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i8x16`].
	#[inline]
	pub fn cmple_i8x16(self, a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
		let gt = self.cmpgt_i8x16(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Sse2::cmpgt_i8_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i8x16`].
	#[inline]
	pub fn cmpge_i8x16(self, a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
		let lt = self.cmplt_i8x16(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Sse2::cmplt_i8_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmplt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u8x16`].
	#[inline]
	pub fn cmplt_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		self.cmpgt_u8x16(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u8x16`].
	#[inline]
	pub fn cmple_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		let gt = self.cmpgt_u8x16(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u8x16`].
	#[inline]
	pub fn cmpge_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		let lt = self.cmplt_u8x16(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmplt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i16x8`].
	#[inline]
	pub fn cmplt_i16x8(self, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
		self.cmpgt_i16x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Sse2::cmpgt_i16_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i16x8`].
	#[inline]
	pub fn cmple_i16x8(self, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
		let gt = self.cmpgt_i16x8(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Sse2::cmpgt_i16_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i16x8`].
	#[inline]
	pub fn cmpge_i16x8(self, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
		let lt = self.cmplt_i16x8(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Sse2::cmplt_i16_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmplt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u16x8`].
	#[inline]
	pub fn cmplt_u16x8(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
		self.cmpgt_u16x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u16x8`].
	#[inline]
	pub fn cmple_u16x8(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
		let gt = self.cmpgt_u16x8(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u16x8`].
	#[inline]
	pub fn cmpge_u16x8(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
		let lt = self.cmplt_u16x8(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmplt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
}

macro_rules! sse2_i16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! sse2_u16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Sse2, target_feature = "sse2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse2_i16_shift_imm!(
	shl_i16x8, shl_i16_slice, psllw, _mm_sll_epi16, |x: i16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`psll`, uniform count).",
	"`out[i] = a[i] << IMM`. 8-wide chunks, scalar remainder."
);
sse2_i16_shift_imm!(
	shr_i16x8, shr_i16_slice, psrlw, _mm_srl_epi16, |x: i16, imm| ((x as u16).wrapping_shr(imm)) as i16,
	"`a >> IMM` logical per lane (`psrl`, uniform count).",
	"`out[i] = a[i] logical >> IMM`. 8-wide chunks, scalar remainder."
);
sse2_i16_shift_imm!(
	sra_i16x8, sra_i16_slice, psraw, _mm_sra_epi16, |x: i16, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`psra`, uniform count).",
	"`out[i] = a[i] arithmetic >> IMM`. 8-wide chunks, scalar remainder."
);
sse2_u16_shift_imm!(
	shl_u16x8, shl_u16_slice, psllw_u, _mm_sll_epi16, |x: u16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`psll`, uniform count).",
	"`out[i] = a[i] << IMM`. 8-wide chunks, scalar remainder."
);
sse2_u16_shift_imm!(
	shr_u16x8, shr_u16_slice, psrlw_u, _mm_srl_epi16, |x: u16, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`psrl`, uniform count).",
	"`out[i] = a[i] >> IMM`. 8-wide chunks, scalar remainder."
);

simd_movemask! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = movemask_i8x16, intrinsic_fn = movemask_epi8,
	width = 16, elem = i8, vec = __m128i, mask = u16,
	loadu = _mm_loadu_si128, intrinsic = _mm_movemask_epi8,
	doc = "Sign-bit mask, one bit per lane (`pmovmskb`).",
}
simd_movemask! {
	token = Sse2, target_feature = "sse2",
	fixed_fn = movemask_f64x2, intrinsic_fn = movemask_pd,
	width = 2, elem = f64, vec = __m128d, mask = u8,
	loadu = _mm_loadu_pd, intrinsic = _mm_movemask_pd,
	doc = "Sign-bit mask, one bit per lane (`movmskpd`). Low 2 bits meaningful, rest 0.",
}

#[cfg(test)]
#[path = "../../test/ops/sse/sse2.rs"]
mod tests;
