//! AVX2: 256-bit YMM integer ops (`"avx2"`). Token: [`Avx2`]. Floats are in [`super::avx::Avx`].
//! Supports wide integer add/sub/min/max/cmpeq/cmpgt/lt/le/ge/bit, saturating arithmetic, mul/shift, and i64/u64 add/sub/mullo/abs.
//! 128-bit lane min/max uses `cmpgt`+blend; 64x64->64 mullo is schoolbook below AVX-512DQ.
//! `mul_i8/u8` and byte-granular `shl/shr/sra_i8/u8` are composed (no native 8-bit multiply/shift on x86 at any tier).

use core::arch::x86_64::{
	__m128i, __m256, __m256d, __m256i, _mm_alignr_epi8, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_setzero_si128,
	_mm_storeu_si128, _mm256_abs_epi16,
	_mm256_abs_epi32, _mm256_abs_epi8, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8,
	_mm256_adds_epi16, _mm256_adds_epi8, _mm256_adds_epu16, _mm256_adds_epu8, _mm256_alignr_epi8, _mm256_and_si256,
	_mm256_andnot_si256, _mm256_avg_epu16, _mm256_avg_epu8, _mm256_blendv_epi8, _mm256_blendv_pd, _mm256_blendv_ps,
	_mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi64, _mm256_cmpeq_epi8,
	_mm256_broadcastb_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi64, _mm256_cmpgt_epi8,
	_mm256_extracti128_si256, _mm256_inserti128_si256,
	_mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_epi32, _mm256_maskload_epi64,
	_mm256_maskstore_epi32, _mm256_maskstore_epi64, _mm256_max_epi16, _mm256_max_epi32, _mm256_max_epi8,
	_mm256_max_epu16, _mm256_max_epu32, _mm256_max_epu8, _mm256_min_epi16, _mm256_min_epi32, _mm256_min_epi8,
	_mm256_min_epu16, _mm256_min_epu32, _mm256_min_epu8, _mm256_movemask_epi8, _mm256_mul_epi32, _mm256_mul_epu32,
	_mm256_blend_epi32, _mm256_mullo_epi16,
	_mm256_mullo_epi32, _mm256_or_si256, _mm256_packs_epi16, _mm256_packus_epi16, _mm256_set1_epi16, _mm256_set1_epi64x,
	_mm256_set1_epi8, _mm256_set_m128i, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_sll_epi16, _mm256_sll_epi32,
	_mm256_slli_epi64, _mm256_slli_si256, _mm256_sllv_epi32, _mm256_sllv_epi64, _mm256_sra_epi16, _mm256_sra_epi32, _mm256_srai_epi16,
	_mm256_srai_epi32, _mm256_srav_epi32, _mm256_srl_epi16, _mm256_srl_epi32, _mm256_srli_epi64, _mm256_srlv_epi32,
	_mm256_srlv_epi64, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
	_mm256_sub_epi64, _mm256_sub_epi8, _mm256_subs_epi16, _mm256_subs_epi8, _mm256_subs_epu16, _mm256_subs_epu8,
	_mm256_unpackhi_epi8, _mm256_unpacklo_epi8, _mm256_xor_si256,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
#[cfg(feature = "wider-bus-lift")]
use super::super::avx512::avx512vl::Avx512FVl;
use super::super::macros::{
	scalar_only_binop, simd_binop, simd_binop_lifted, simd_extract_imm, simd_insert_imm, simd_movemask, simd_shift_imm,
	simd_ternop, simd_unop,
};

// Every `Avx2`-token lift below uses `lift_target_feature = "avx2,avx512f,avx512vl"`
// and `lift_proof = Avx512FVl`:
// still issues 256-bit `avx2` instructions (two independent chains per
// iteration), `avx512f,avx512vl` only widens the register file/encoding
// LLVM can pick from. `#[target_feature(enable = ...)]` requires a literal,
// so this can't be a `const`.

/// Proof token: AVX2 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx2(());

impl Avx2 {
	/// `None` if the CPU (or the compile-time target) lacks AVX2.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From resolved tier (`V3`/`V4` list `Feature::Avx2`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx2).then_some(Avx2(()))
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx2).then_some(Avx2(()))
	}

	/// `vpalignr`: two independent `palignr`s, one per 128-bit lane (not a
	/// full 256-bit concatenation: lane 1 only ever sees `a`/`b`'s high
	/// 128 bits, lane 0 only the low 128 bits). Per lane: concatenate
	/// `[b_lane, a_lane]` into a 32-byte window, shift right by `IMM8`
	/// bytes, keep the low 16. Same hand-written shape as
	/// [`super::super::sse::ssse3::Ssse3::alignr_u8x16`] (no `simd_binop_imm`
	/// fit; see that method's doc for why).
	#[inline]
	pub fn alignr_u8x32<const IMM8: i32>(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		unsafe { alignr::<IMM8>(&a, &b) }
	}

	/// True 32-byte-window `alignr`, forced across the lane boundary that
	/// [`Self::alignr_u8x32`] is locked to (there is no single instruction
	/// for this: `vpalignr` is lane-locked at every width on real x86, see
	/// that method's doc). Composed from 4 [`Ssse3::alignr_u8x16`] calls (2
	/// used, 2 optimized away at compile time once `IMM8` picks which
	/// 16-byte halves matter) over the conceptual zero-padded array
	/// `[b_lo, b_hi, a_lo, a_hi, 0, 0, ...]`: `out[i] = window[IMM8+i]`
	/// where `window = concat(b, a)` (64 bytes, `b` first), 0 past the end.
	/// Same shape as `alignr_u8x16`/`alignr_u8x32`, just a 64-byte window
	/// instead of 32/16.
	#[inline]
	pub fn alignr_u8x32_full<const IMM8: i32>(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		unsafe { alignr_full::<IMM8>(&a, &b) }
	}

	/// `vpslldq`: per-128-bit-lane byte shift left by `IMM8`, zero-filled
	/// from each lane's bottom (`IMM8 >= 16` zeroes a lane entirely): lane
	/// 1 never sees lane 0's bytes, same lane-locked shape as
	/// [`Self::alignr_u8x32`]. Not a per-lane numeric shift ([`simd_shift_imm`]
	/// doesn't fit, same reasoning as [`super::super::sse::sse2::Sse2::slli_u8x16`]).
	#[inline]
	pub fn slli_u8x32<const IMM8: i32>(self, a: [u8; 32]) -> [u8; 32] {
		unsafe { slli_si256::<IMM8>(&a) }
	}

	/// `vpbroadcastb`: replicate `byte` across all 32 lanes.
	#[inline]
	pub fn broadcast_u8x32(self, byte: u8) -> [u8; 32] {
		unsafe { broadcastb(byte) }
	}
}

simd_extract_imm! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = extract_u8x16_from_x32, intrinsic_fn = extract_u8x16_from_x32_intrinsic,
	wide_width = 32, narrow_width = 16, elem = u8, wide_vec = __m256i, narrow_vec = __m128i,
	wide_loadu = _mm256_loadu_si256, storeu = _mm_storeu_si128, intrinsic = _mm256_extracti128_si256,
	fixed_doc = "Extracts the `IMM8 & 1`-selected 16-byte half of `a` (`vextracti128`, 256-bit source).",
}

simd_insert_imm! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = insert_u8x16_into_x32, intrinsic_fn = insert_u8x16_into_x32_intrinsic,
	wide_width = 32, narrow_width = 16, elem = u8, wide_vec = __m256i, narrow_vec = __m128i,
	wide_loadu = _mm256_loadu_si256, narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_inserti128_si256,
	fixed_doc = "Copies `a`, then overwrites its `IMM8 & 1`-selected 16-byte half with `b` (`vinserti128`, 256-bit).",
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn alignr<const IMM8: i32>(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let vr = _mm256_alignr_epi8::<IMM8>(va, vb);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Composed from 2 [`_mm_alignr_epi8`] calls (SSSE3: guaranteed present on
/// every AVX2 CPU, no separate token needed) over the conceptual
/// zero-padded halves array `[b_lo, b_hi, a_lo, a_hi, 0, 0, ...]`, indexed
/// by `IMM8`'s 16-byte segment (`k = IMM8.div_euclid(16)`) and the residual
/// shift within that segment (`r = IMM8.rem_euclid(16)`):
/// `out_lo128 = alignr(halves[k+1], halves[k], r)`,
/// `out_hi128 = alignr(halves[k+2], halves[k+1], r)`. This is the same
/// "shift a concatenated window right by `r`, keep 16 bytes" identity
/// `_mm_alignr_epi8` itself implements, just walked one 16-byte segment at
/// a time instead of pinned to segments 0/1. `k` and `r` are per-instantiation
/// constants (both derived from the const `IMM8`) but not expressible as a
/// *derived* const-generic argument on stable Rust, so `r`'s use as
/// `_mm_alignr_epi8`'s immediate goes through a 16-arm literal match
/// (folds to the one live arm at monomorphization time): the `pick`
/// selection is plain runtime `match`, similarly foldable since `k` is
/// per-instantiation constant.
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`] (which architecturally implies SSSE3 -
/// no x86-64 CPU has AVX2 without it).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn alignr_full<const IMM8: i32>(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let b_lo = _mm256_castsi256_si128(vb);
		let b_hi = _mm256_extracti128_si256::<1>(vb);
		let a_lo = _mm256_castsi256_si128(va);
		let a_hi = _mm256_extracti128_si256::<1>(va);
		let z = _mm_setzero_si128();

		let k = IMM8.div_euclid(16);
		let r = IMM8.rem_euclid(16);

		let pick = |idx: i32| -> __m128i {
			match idx {
				0 => b_lo,
				1 => b_hi,
				2 => a_lo,
				3 => a_hi,
				_ => z,
			}
		};

		// `_mm_alignr_epi8`'s shift is a compile-time immediate, but `r` is
		// only known-constant *within* one `IMM8` instantiation, not
		// expressible as a derived const-generic argument on stable Rust
		// (no `generic_const_exprs`): so route through a literal-per-arm
		// match instead. Dead arms fold away at monomorphization time.
		// Revisit if `generic_const_exprs` (or an equivalent) stabilizes -
		// `_mm_alignr_epi8::<{ r }>(...)` would replace this whole macro.
		macro_rules! alignr16 {
			($r:expr, $a:expr, $b:expr) => {
				match $r {
					0 => _mm_alignr_epi8::<0>($a, $b),
					1 => _mm_alignr_epi8::<1>($a, $b),
					2 => _mm_alignr_epi8::<2>($a, $b),
					3 => _mm_alignr_epi8::<3>($a, $b),
					4 => _mm_alignr_epi8::<4>($a, $b),
					5 => _mm_alignr_epi8::<5>($a, $b),
					6 => _mm_alignr_epi8::<6>($a, $b),
					7 => _mm_alignr_epi8::<7>($a, $b),
					8 => _mm_alignr_epi8::<8>($a, $b),
					9 => _mm_alignr_epi8::<9>($a, $b),
					10 => _mm_alignr_epi8::<10>($a, $b),
					11 => _mm_alignr_epi8::<11>($a, $b),
					12 => _mm_alignr_epi8::<12>($a, $b),
					13 => _mm_alignr_epi8::<13>($a, $b),
					14 => _mm_alignr_epi8::<14>($a, $b),
					15 => _mm_alignr_epi8::<15>($a, $b),
					_ => unreachable!(),
				}
			};
		}

		let out_lo = alignr16!(r, pick(k + 1), pick(k));
		let out_hi = alignr16!(r, pick(k + 2), pick(k + 1));
		let vr = _mm256_set_m128i(out_hi, out_lo);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn slli_si256<const IMM8: i32>(a: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vr = _mm256_slli_si256::<IMM8>(va);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// `vpbroadcastb` takes its source from a 128-bit register (only byte 0
/// matters), so build one with `byte` in lane 0; the other 15 bytes are
/// dead, never read by the instruction.
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn broadcastb(byte: u8) -> [u8; 32] {
	unsafe {
		let mut src = [0u8; 16];
		src[0] = byte;
		let va: __m128i = _mm_loadu_si128(src.as_ptr().cast());
		let vr = _mm256_broadcastb_epi8(va);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Software reference: `vpalignr`'s per-128-bit-lane semantics, built by
/// applying [`super::super::sse::ssse3::alignr_scalar`] to each 16-byte
/// lane independently.
#[cfg(test)]
fn alignr_u8x32_scalar(a: &[u8; 32], b: &[u8; 32], imm: i32) -> Vec<u8> {
	use super::super::sse::ssse3::alignr_scalar;
	let mut out = Vec::with_capacity(32);
	out.extend(alignr_scalar(&a[0..16], &b[0..16], imm));
	out.extend(alignr_scalar(&a[16..32], &b[16..32], imm));
	out
}

/// Software reference for [`Avx2::alignr_u8x32_full`]: the true 64-byte
/// window `concat(b, a)`, no lane splitting. Shares the underlying model
/// with [`super::super::sse::ssse3::alignr_scalar`], just at 32/64 instead
/// of 16/32 bytes (reimplemented rather than reused since that helper is
/// `[u8]`-generic on a shared length, not the asymmetric 32-in/64-window
/// shape here).
#[cfg(test)]
fn alignr_u8x32_full_scalar(a: &[u8; 32], b: &[u8; 32], imm: i32) -> Vec<u8> {
	let window: Vec<u8> = b.iter().chain(a.iter()).copied().collect();
	(0..32)
		.map(|i| {
			let pos = imm as i64 + i as i64;
			if pos < 0 || pos as usize >= window.len() { 0 } else { window[pos as usize] }
		})
		.collect()
}

macro_rules! avx2_i32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		avx2_i32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Avx2, lift_target_feature = "avx2,avx512f,avx512vl",
			lifted_fn = $lifted_fn, lift_proof = Avx512FVl,
			width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal, lifted_fn = $lifted_fn:ident,) => {
		avx2_u32_binop!($fixed_fn, $slice_fn, $intrinsic_fn, $intrinsic, $scalar, $fixed_doc, $slice_doc);
		simd_binop_lifted! {
			token = Avx2, lift_target_feature = "avx2,avx512f,avx512vl",
			lifted_fn = $lifted_fn, lift_proof = Avx512FVl,
			width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			lifted_doc = $slice_doc,
		}
	};
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i32_binop!(
	add_i32x8, add_i32_slice, paddd, _mm256_add_epi32, |x: i32, y: i32| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddd`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide `add_i32x8` chunks, scalar remainder.",
	lifted_fn = add_i32_slice_lifted,
);
avx2_i32_binop!(
	sub_i32x8, sub_i32_slice, psubd, _mm256_sub_epi32, |x: i32, y: i32| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubd`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide `sub_i32x8` chunks, scalar remainder.",
	lifted_fn = sub_i32_slice_lifted,
);
avx2_i32_binop!(
	mul_i32x8, mul_i32_slice, pmulld, _mm256_mullo_epi32, |x: i32, y: i32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`vpmulld`, 256-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide `mul_i32x8` chunks, scalar remainder.",
	lifted_fn = mul_i32_slice_lifted,
);
avx2_i32_binop!(
	min_i32x8, min_i32_slice, pminsd, _mm256_min_epi32, |x, y| x.min(y),
	"Per-lane signed min (`vpminsd`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_i32x8` chunks, scalar remainder."
);
avx2_i32_binop!(
	max_i32x8, max_i32_slice, pmaxsd, _mm256_max_epi32, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsd`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_i32x8` chunks, scalar remainder."
);
scalar_only_binop! {
	token = Avx2,
	fixed_fn = div_i32x8, slice_fn = div_i32_slice,
	width = 8, elem = i32,
	scalar = |x: i32, y: i32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor or `i32::MIN / -1`, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

avx2_u32_binop!(
	add_u32x8, add_u32_slice, paddd_u, _mm256_add_epi32, |x: u32, y: u32| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddd`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 8-wide `add_u32x8` chunks, scalar remainder.",
	lifted_fn = add_u32_slice_lifted,
);
avx2_u32_binop!(
	sub_u32x8, sub_u32_slice, psubd_u, _mm256_sub_epi32, |x: u32, y: u32| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubd`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 8-wide `sub_u32x8` chunks, scalar remainder.",
	lifted_fn = sub_u32_slice_lifted,
);
avx2_u32_binop!(
	mul_u32x8, mul_u32_slice, pmulld_u, _mm256_mullo_epi32, |x: u32, y: u32| x.wrapping_mul(y),
	"`a * b` per lane, low 32 bits (`vpmulld`, 256-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 8-wide `mul_u32x8` chunks, scalar remainder.",
	lifted_fn = mul_u32_slice_lifted,
);
avx2_u32_binop!(
	min_u32x8, min_u32_slice, pminud, _mm256_min_epu32, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminud`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_u32x8` chunks, scalar remainder."
);
avx2_u32_binop!(
	max_u32x8, max_u32_slice, pmaxud, _mm256_max_epu32, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxud`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_u32x8` chunks, scalar remainder."
);
scalar_only_binop! {
	token = Avx2,
	fixed_fn = div_u32x8, slice_fn = div_u32_slice,
	width = 8, elem = u32,
	scalar = |x: u32, y: u32| x / y,
	fixed_doc = "`a / b` per lane. No hardware SIMD integer divide exists on x86 at any width; this is a plain scalar loop, not vectorized. Panics on zero divisor, matching Rust's `/`.",
	slice_doc = "`out[i] = a[i] / b[i]`. Scalar loop, no chunking (nothing to align to).",
}

avx2_i32_binop!(
	and_i32x8, and_i32_slice, vpand, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = and_i32_slice_lifted,
);
avx2_i32_binop!(
	or_i32x8, or_i32_slice, vpor, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = or_i32_slice_lifted,
);
avx2_i32_binop!(
	xor_i32x8, xor_i32_slice, vpxor, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = xor_i32_slice_lifted,
);
avx2_i32_binop!(
	andnot_i32x8, andnot_i32_slice, vpandn, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = andnot_i32_slice_lifted,
);
avx2_i32_binop!(
	cmpeq_i32x8, cmpeq_i32_slice, vpcmpeqd, _mm256_cmpeq_epi32,
	|x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`vpcmpeqd`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder.",
	lifted_fn = cmpeq_i32_slice_lifted,
);

avx2_u32_binop!(
	and_u32x8, and_u32_slice, vpand_u, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = and_u32_slice_lifted,
);
avx2_u32_binop!(
	or_u32x8, or_u32_slice, vpor_u, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = or_u32_slice_lifted,
);
avx2_u32_binop!(
	xor_u32x8, xor_u32_slice, vpxor_u, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = xor_u32_slice_lifted,
);
avx2_u32_binop!(
	andnot_u32x8, andnot_u32_slice, vpandn_u, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 8-wide chunks, scalar remainder.",
	lifted_fn = andnot_u32_slice_lifted,
);
avx2_u32_binop!(
	cmpeq_u32x8, cmpeq_u32_slice, vpcmpeqd_u, _mm256_cmpeq_epi32,
	|x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`vpcmpeqd`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder.",
	lifted_fn = cmpeq_u32_slice_lifted,
);

// i64/u64 add/sub: native AVX2. min/max: composed via cmpgt+blendv (no native
// pminsq/pmaxsq below AVX-512F): see impl block below add/sub.
macro_rules! avx2_i64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i64_binop!(
	add_i64x4, add_i64_slice, paddq, _mm256_add_epi64, |x: i64, y: i64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddq`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 4-wide `add_i64x4` chunks, scalar remainder."
);
avx2_i64_binop!(
	sub_i64x4, sub_i64_slice, psubq, _mm256_sub_epi64, |x: i64, y: i64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubq`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 4-wide `sub_i64x4` chunks, scalar remainder."
);
avx2_u64_binop!(
	add_u64x4, add_u64_slice, paddq_u, _mm256_add_epi64, |x: u64, y: u64| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddq`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 4-wide `add_u64x4` chunks, scalar remainder."
);
avx2_u64_binop!(
	sub_u64x4, sub_u64_slice, psubq_u, _mm256_sub_epi64, |x: u64, y: u64| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubq`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 4-wide `sub_u64x4` chunks, scalar remainder."
);
// i64/u64 bitwise: same si256 ops as i32 (view-only).
avx2_i64_binop!(
	and_i64x4, and_i64_slice, vpand_i64, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	or_i64x4, or_i64_slice, vpor_i64, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	xor_i64x4, xor_i64_slice, vpxor_i64, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	andnot_i64x4, andnot_i64_slice, vpandn_i64, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	and_u64x4, and_u64_slice, vpand_u64, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	or_u64x4, or_u64_slice, vpor_u64, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	xor_u64x4, xor_u64_slice, vpxor_u64, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	andnot_u64x4, andnot_u64_slice, vpandn_u64, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	cmpeq_i64x4, cmpeq_i64_slice, vpcmpeqq, _mm256_cmpeq_epi64,
	|x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`vpcmpeqq`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	cmpeq_u64x4, cmpeq_u64_slice, vpcmpeqq_u, _mm256_cmpeq_epi64,
	|x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`vpcmpeqq`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	cmpgt_i64x4, cmpgt_i64_slice, vpcmpgtq, _mm256_cmpgt_epi64,
	|x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`vpcmpgtq`, 256-bit): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 4-wide chunks, scalar remainder."
);

impl Avx2 {
	/// Unsigned greater-than mask (all-1s if `a>b`). Sign-bit flip + [`cmpgt_i64x4`].
	#[inline]
	pub fn cmpgt_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		let ai: [i64; 4] = core::array::from_fn(|i| (a[i] ^ 0x8000_0000_0000_0000) as i64);
		let bi: [i64; 4] = core::array::from_fn(|i| (b[i] ^ 0x8000_0000_0000_0000) as i64);
		let r = self.cmpgt_i64x4(ai, bi);
		core::array::from_fn(|i| r[i] as u64)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i64x4`].
	#[inline]
	pub fn cmplt_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		self.cmpgt_i64x4(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Avx2::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i64x4`].
	#[inline]
	pub fn cmple_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		let gt = self.cmpgt_i64x4(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Avx2::cmpgt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmpgt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i64x4`].
	#[inline]
	pub fn cmpge_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		let lt = self.cmplt_i64x4(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Avx2::cmplt_i64_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		self.cmplt_i64_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u64x4`].
	#[inline]
	pub fn cmplt_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		self.cmpgt_u64x4(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u64` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		self.cmpgt_u64_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u64x4`].
	#[inline]
	pub fn cmple_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		let gt = self.cmpgt_u64x4(a, b);
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

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u64x4`].
	#[inline]
	pub fn cmpge_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		let lt = self.cmplt_u64x4(a, b);
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

	/// Per-lane signed min (`vpcmpgtq` + `vpblendvb`, 256-bit; no native `vpminsq` below AVX-512F).
	#[inline]
	pub fn min_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		unsafe { min_i64x4_composed(&a, &b) }
	}

	/// `out[i] = min(a[i], b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_i64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane signed max (`vpcmpgtq` + `vpblendvb`, 256-bit; no native `vpmaxsq` below AVX-512F).
	#[inline]
	pub fn max_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		unsafe { max_i64x4_composed(&a, &b) }
	}

	/// `out[i] = max(a[i], b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_i64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}

	/// Per-lane unsigned min: sign-bit-flip compare + blend on the original unflipped values.
	#[inline]
	pub fn min_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		unsafe { min_u64x4_composed(&a, &b) }
	}

	/// `out[i] = min(a[i], b[i])` (`u64`). 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn min_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.min_u64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.min(y);
		}
	}

	/// Per-lane unsigned max: sign-bit-flip compare + blend on the original unflipped values.
	#[inline]
	pub fn max_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		unsafe { max_u64x4_composed(&a, &b) }
	}

	/// `out[i] = max(a[i], b[i])` (`u64`). 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn max_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.max_u64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.max(y);
		}
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn min_i64x4_composed(a: &[i64; 4], b: &[i64; 4]) -> [i64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let gt = _mm256_cmpgt_epi64(va, vb);
		let vr = _mm256_blendv_epi8(va, vb, gt);
		let mut out = [0i64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn max_i64x4_composed(a: &[i64; 4], b: &[i64; 4]) -> [i64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let gt = _mm256_cmpgt_epi64(vb, va);
		let vr = _mm256_blendv_epi8(va, vb, gt);
		let mut out = [0i64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn min_u64x4_composed(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let sign = _mm256_set1_epi64x(i64::MIN);
		let af = _mm256_xor_si256(va, sign);
		let bf = _mm256_xor_si256(vb, sign);
		let gt = _mm256_cmpgt_epi64(af, bf);
		let vr = _mm256_blendv_epi8(va, vb, gt);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn max_u64x4_composed(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let sign = _mm256_set1_epi64x(i64::MIN);
		let af = _mm256_xor_si256(va, sign);
		let bf = _mm256_xor_si256(vb, sign);
		let gt = _mm256_cmpgt_epi64(bf, af);
		let vr = _mm256_blendv_epi8(va, vb, gt);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

// i64/u64 mullo: schoolbook via pmuludq halves (256-bit mirror of the SSE2 composition,
// same reasoning: no native 64x64->64 multiply below AVX-512DQ). i64 abs: branchless
// sign-broadcast (256-bit mirror of the SSE2 composition; no native 64-bit arithmetic
// shift below AVX-512).
impl Avx2 {
	/// Per-lane low-64-bit multiply, wrapping: schoolbook decomposition into 32-bit
	/// half-lane products (`vpmuludq`+shifts+adds, 256-bit; no native 64x64->64 multiply
	/// exists below AVX-512DQ).
	#[inline]
	pub fn mullo_u64x4(self, a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
		unsafe { mullo_u64x4_composed(&a, &b) }
	}

	/// `out[i] = a[i].wrapping_mul(b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn mullo_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [u64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.mullo_u64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.wrapping_mul(y);
		}
	}

	/// Signed view of [`mullo_u64x4`](Self::mullo_u64x4): wrapping low-64 multiply is
	/// bit-identical for signed and unsigned operands.
	#[inline]
	pub fn mullo_i64x4(self, a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
		let au: [u64; 4] = core::array::from_fn(|i| a[i] as u64);
		let bu: [u64; 4] = core::array::from_fn(|i| b[i] as u64);
		let r = self.mullo_u64x4(au, bu);
		core::array::from_fn(|i| r[i] as i64)
	}

	/// `out[i] = a[i].wrapping_mul(b[i])`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn mullo_i64_slice(self, a: &[i64], b: &[i64], out: &mut [i64]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let b_chunks = b.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [i64; 4] = ac.try_into().expect("chunks_exact width");
			let bv: [i64; 4] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.mullo_i64x4(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = x.wrapping_mul(y);
		}
	}

	/// Per-lane absolute value, wrapping (`i64::MIN` stays `i64::MIN`): branchless
	/// sign-broadcast mask (`shuffle`+`srai`, 256-bit) + `(a XOR mask) - mask`. No native
	/// 64-bit arithmetic shift exists below AVX-512.
	#[inline]
	pub fn abs_i64x4(self, a: [i64; 4]) -> [i64; 4] {
		unsafe { abs_i64x4_composed(&a) }
	}

	/// `out[i] = a[i].wrapping_abs()`. 4-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub(crate) fn abs_i64_slice(self, a: &[i64], out: &mut [i64]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(4);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(4);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i64; 4] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.abs_i64x4(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_abs();
		}
	}
}

/// 256-bit mirror of the SSE2 schoolbook composition (`mullo_u64x2_composed`).
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mullo_u64x4_composed(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let ac = _mm256_mul_epu32(va, vb);
		let a_hi = _mm256_srli_epi64::<32>(va);
		let b_hi = _mm256_srli_epi64::<32>(vb);
		let cross = _mm256_add_epi64(_mm256_mul_epu32(a_hi, vb), _mm256_mul_epu32(va, b_hi));
		let vr = _mm256_add_epi64(ac, _mm256_slli_epi64::<32>(cross));
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// 256-bit mirror of the SSE2 sign-broadcast composition (`abs_i64x2_composed`).
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn abs_i64x4_composed(a: &[i64; 4]) -> [i64; 4] {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		// 0xF5 = _MM_SHUFFLE(3,3,1,1): duplicate each lane's high dword into both dwords.
		let hi_dup = _mm256_shuffle_epi32::<0xF5>(va);
		let mask = _mm256_srai_epi32::<31>(hi_dup);
		let vr = _mm256_sub_epi64(_mm256_xor_si256(va, mask), mask);
		let mut out = [0i64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

const WIDENING_MUL_ODD_LANES: i32 = 0b1010_1010;

impl Avx2 {
	/// Widening multiply: full 64-bit product of each of the 8 lanes, split
	/// into low/high 32-bit halves: same 8-lane count on both outputs,
	/// unlike [`Avx2::mul_i32x8`] which narrows back down to 32 bits.
	/// `pmuludq` only reads each 64-bit slot's low half, so the even lanes
	/// come from one pass and the odd lanes need a second pass on the
	/// shifted-down input, then both get re-interleaved with an immediate
	/// blend (`vpmuludq`+`vpsrlq`+`vpsllq`+`vpblendd`, 256-bit).
	#[inline]
	pub fn widening_mul_u32x8(self, a: [u32; 8], b: [u32; 8]) -> ([u32; 8], [u32; 8]) {
		unsafe { widening_mul_u32x8_composed(&a, &b) }
	}

	/// Signed sibling of [`Avx2::widening_mul_u32x8`] (`vpmuldq` for the
	/// even-lane pass instead of `vpmuludq`; the odd-lane pass and
	/// re-interleave are identical since the low 32 bits of a product don't
	/// depend on signedness).
	#[inline]
	pub fn widening_mul_i32x8(self, a: [i32; 8], b: [i32; 8]) -> ([i32; 8], [i32; 8]) {
		unsafe { widening_mul_i32x8_composed(&a, &b) }
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widening_mul_u32x8_composed(a: &[u32; 8], b: &[u32; 8]) -> ([u32; 8], [u32; 8]) {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let ab_evens = _mm256_mul_epu32(va, vb);
		let ab_odds = _mm256_mul_epu32(_mm256_srli_epi64::<32>(va), _mm256_srli_epi64::<32>(vb));
		let lo = _mm256_blend_epi32::<WIDENING_MUL_ODD_LANES>(ab_evens, _mm256_slli_epi64::<32>(ab_odds));
		let hi = _mm256_blend_epi32::<WIDENING_MUL_ODD_LANES>(_mm256_srli_epi64::<32>(ab_evens), ab_odds);
		let mut lo_out = [0u32; 8];
		let mut hi_out = [0u32; 8];
		_mm256_storeu_si256(lo_out.as_mut_ptr().cast(), lo);
		_mm256_storeu_si256(hi_out.as_mut_ptr().cast(), hi);
		(lo_out, hi_out)
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widening_mul_i32x8_composed(a: &[i32; 8], b: &[i32; 8]) -> ([i32; 8], [i32; 8]) {
	unsafe {
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let vb: __m256i = _mm256_loadu_si256(b.as_ptr().cast());
		let ab_evens = _mm256_mul_epi32(va, vb);
		let ab_odds = _mm256_mul_epi32(_mm256_srli_epi64::<32>(va), _mm256_srli_epi64::<32>(vb));
		let lo = _mm256_blend_epi32::<WIDENING_MUL_ODD_LANES>(ab_evens, _mm256_slli_epi64::<32>(ab_odds));
		let hi = _mm256_blend_epi32::<WIDENING_MUL_ODD_LANES>(_mm256_srli_epi64::<32>(ab_evens), ab_odds);
		let mut lo_out = [0i32; 8];
		let mut hi_out = [0i32; 8];
		_mm256_storeu_si256(lo_out.as_mut_ptr().cast(), lo);
		_mm256_storeu_si256(hi_out.as_mut_ptr().cast(), hi);
		(lo_out, hi_out)
	}
}

avx2_i32_binop!(
	cmpgt_i32x8, cmpgt_i32_slice, vpcmpgtd, _mm256_cmpgt_epi32,
	|x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`vpcmpgtd`, 256-bit): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 8-wide chunks, scalar remainder.",
	lifted_fn = cmpgt_i32_slice_lifted,
);

impl Avx2 {
	/// Unsigned greater-than mask (all-1s if `a>b`). Sign-bit flip + [`cmpgt_i32x8`].
	#[inline]
	pub fn cmpgt_u32x8(self, a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
		let ai: [i32; 8] = core::array::from_fn(|i| (a[i] ^ 0x8000_0000) as i32);
		let bi: [i32; 8] = core::array::from_fn(|i| (b[i] ^ 0x8000_0000) as i32);
		let r = self.cmpgt_i32x8(ai, bi);
		core::array::from_fn(|i| r[i] as u32)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(8);
		let b_chunks = b.chunks_exact(8);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(8);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u32; 8] = ac.try_into().expect("chunks_exact width");
			let bv: [u32; 8] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u32x8(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Lane less-than mask (all-1s if `a<b`): operand-swapped [`cmpgt_i32x8`].
	#[inline]
	pub fn cmplt_i32x8(self, a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
		self.cmpgt_i32x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0`. Operand-swapped [`Avx2::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(b, a, out);
	}

	/// Lane less-equal mask (all-1s if `a<=b`): bitwise NOT of [`cmpgt_i32x8`].
	#[inline]
	pub fn cmple_i32x8(self, a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
		let gt = self.cmpgt_i32x8(a, b);
		core::array::from_fn(|i| !gt[i])
	}

	/// `out[i] = all-1s if a[i]<=b[i] else 0`. NOT of [`Avx2::cmpgt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmple_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmpgt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane greater-equal mask (all-1s if `a>=b`): bitwise NOT of [`cmplt_i32x8`].
	#[inline]
	pub fn cmpge_i32x8(self, a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
		let lt = self.cmplt_i32x8(a, b);
		core::array::from_fn(|i| !lt[i])
	}

	/// `out[i] = all-1s if a[i]>=b[i] else 0`. NOT of [`Avx2::cmplt_i32_slice`].
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpge_i32_slice(self, a: &[i32], b: &[i32], out: &mut [i32]) {
		self.cmplt_i32_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	/// Lane less-than mask, unsigned: operand-swapped [`cmpgt_u32x8`].
	#[inline]
	pub fn cmplt_u32x8(self, a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
		self.cmpgt_u32x8(b, a)
	}

	/// `out[i] = all-1s if a[i]<b[i] else 0` (`u32` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmplt_u32_slice(self, a: &[u32], b: &[u32], out: &mut [u32]) {
		self.cmpgt_u32_slice(b, a, out);
	}

	/// Lane less-equal mask, unsigned: bitwise NOT of [`cmpgt_u32x8`].
	#[inline]
	pub fn cmple_u32x8(self, a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
		let gt = self.cmpgt_u32x8(a, b);
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

	/// Lane greater-equal mask, unsigned: bitwise NOT of [`cmplt_u32x8`].
	#[inline]
	pub fn cmpge_u32x8(self, a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
		let lt = self.cmplt_u32x8(a, b);
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
}

// select: blendv (i32/u32: all-0/1 mask; f32: sign bit, same as Sse41::blend_f32x4).
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_i32x8, slice_fn = select_i32_slice, intrinsic_fn = vpblendvb_i32,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: i32| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_u32x8, slice_fn = select_u32_slice, intrinsic_fn = vpblendvb_u32,
	width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: u32| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_f32x8, slice_fn = select_f32_slice, intrinsic_fn = vblendvps256,
	width = 8, elem = f32, vec = __m256, loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
	intrinsic = _mm256_blendv_ps, scalar = |a: f32, b: f32, m: f32| if m.is_sign_negative() { b } else { a },
	fixed_doc = "Per-lane select (`vblendvps`): mask sign bit picks `b` (same as `Sse41::blend_f32x4`).",
	slice_doc = "`out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 8-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_i64x4, slice_fn = select_i64_slice, intrinsic_fn = vpblendvb_i64,
	width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: i64| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_u64x4, slice_fn = select_u64_slice, intrinsic_fn = vpblendvb_u64,
	width = 4, elem = u64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: u64| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_f64x4, slice_fn = select_f64_slice, intrinsic_fn = vblendvpd256,
	width = 4, elem = f64, vec = __m256d, loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
	intrinsic = _mm256_blendv_pd, scalar = |a: f64, b: f64, m: f64| if m.is_sign_negative() { b } else { a },
	fixed_doc = "Per-lane select (`vblendvpd`): mask sign bit picks `b`.",
	slice_doc = "`out[i] = if mask[i].is_sign_negative() { b[i] } else { a[i] }`. 4-wide chunks, scalar remainder.",
}

// select narrow: same vpblendvb. blendv tests each byte's sign bit, not whole-lane
// nonzero; only agrees with `!= 0` for all-0/all-1 masks (see avx512f select doc).
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_i8x32, slice_fn = select_i8_slice, intrinsic_fn = vpblendvb_i8,
	width = 32, elem = i8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: i8| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 32-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_u8x32, slice_fn = select_u8_slice, intrinsic_fn = vpblendvb_u8,
	width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: u8| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 32-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_i16x16, slice_fn = select_i16_slice, intrinsic_fn = vpblendvb_i16,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: i16| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.",
}
simd_ternop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = select_u16x16, slice_fn = select_u16_slice, intrinsic_fn = vpblendvb_u16,
	width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_blendv_epi8, scalar = |a, b, m: u16| if m != 0 { b } else { a },
	fixed_doc = "Per-lane select (`vpblendvb`). `mask`: all-0/1 lanes; picks `b` where set.",
	slice_doc = "`out[i] = if mask[i] != 0 { b[i] } else { a[i] }`. 16-wide chunks, scalar remainder.",
}

// Per-lane variable shift (count is a vector, not a broadcast IMM): plain `simd_binop!`
// fits directly. x86 semantics: count>=32 zeroes the result (sllv/srlv) or fills with the
// sign bit (srav); NOT the same as Rust's `wrapping_shl`/`wrapping_shr`, which wrap the
// count instead. No SSE2 form exists at any width; AVX2 is the true bottom rung.
macro_rules! avx2_i32_varshift {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u32_varshift {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i32_varshift!(
	sllv_i32x8, sllv_i32_slice, vpsllvd, _mm256_sllv_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector not a broadcast IMM (`vpsllvd`, 256-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] << count[i]`. 8-wide chunks, scalar remainder."
);
avx2_i32_varshift!(
	srlv_i32x8, srlv_i32_slice, vpsrlvd, _mm256_srlv_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { ((x as u32).wrapping_shr(count as u32)) as i32 },
	"`a >> count` logical per lane, `count` a vector not a broadcast IMM (`vpsrlvd`, 256-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] logical >> count[i]`. 8-wide chunks, scalar remainder."
);
avx2_i32_varshift!(
	srav_i32x8, srav_i32_slice, vpsravd, _mm256_srav_epi32,
	|x: i32, count: i32| if (count as u32) >= 32 { x >> 31 } else { x.wrapping_shr(count as u32) },
	"`a >> count` arithmetic per lane, `count` a vector not a broadcast IMM (`vpsravd`, 256-bit).",
	"`out[i] = sign-fill if count[i]>=32 else a[i] arithmetic >> count[i]`. 8-wide chunks, scalar remainder."
);
avx2_u32_varshift!(
	sllv_u32x8, sllv_u32_slice, vpsllvd_u, _mm256_sllv_epi32,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shl(count) },
	"`a << count` per lane, `count` a vector not a broadcast IMM (`vpsllvd`, 256-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] << count[i]`. 8-wide chunks, scalar remainder."
);
avx2_u32_varshift!(
	srlv_u32x8, srlv_u32_slice, vpsrlvd_u, _mm256_srlv_epi32,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shr(count) },
	"`a >> count` per lane, `count` a vector not a broadcast IMM (`vpsrlvd`, 256-bit).",
	"`out[i] = 0 if count[i]>=32 else a[i] >> count[i]`. 8-wide chunks, scalar remainder."
);
// i64/u64 variable shift: AVX2 has sllv/srlv only (no sravq until AVX-512F).
avx2_i64_binop!(
	sllv_i64x4, sllv_i64_slice, vpsllvq, _mm256_sllv_epi64,
	|x: i64, count: i64| if (count as u64) >= 64 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector (`vpsllvq`, 256-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] << count[i]`. 4-wide chunks, scalar remainder."
);
avx2_i64_binop!(
	srlv_i64x4, srlv_i64_slice, vpsrlvq, _mm256_srlv_epi64,
	|x: i64, count: i64| if (count as u64) >= 64 { 0 } else { ((x as u64).wrapping_shr(count as u32)) as i64 },
	"`a >> count` logical per lane, `count` a vector (`vpsrlvq`, 256-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] logical >> count[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	sllv_u64x4, sllv_u64_slice, vpsllvq_u, _mm256_sllv_epi64,
	|x: u64, count: u64| if count >= 64 { 0 } else { x.wrapping_shl(count as u32) },
	"`a << count` per lane, `count` a vector (`vpsllvq`, 256-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] << count[i]`. 4-wide chunks, scalar remainder."
);
avx2_u64_binop!(
	srlv_u64x4, srlv_u64_slice, vpsrlvq_u, _mm256_srlv_epi64,
	|x: u64, count: u64| if count >= 64 { 0 } else { x.wrapping_shr(count as u32) },
	"`a >> count` per lane, `count` a vector (`vpsrlvq`, 256-bit).",
	"`out[i] = 0 if count[i]>=64 else a[i] >> count[i]`. 4-wide chunks, scalar remainder."
);

macro_rules! avx2_i32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u32_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = u32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i32_shift_imm!(
	shl_i32x8, shl_i32_slice, vpslld, _mm256_sll_epi32, |x: i32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpslld`, 256-bit).",
	"`out[i] = a[i] << IMM`. 8-wide chunks, scalar remainder."
);
avx2_i32_shift_imm!(
	shr_i32x8, shr_i32_slice, vpsrld, _mm256_srl_epi32, |x: i32, imm| ((x as u32).wrapping_shr(imm)) as i32,
	"`a >> IMM` logical per lane (`vpsrld`, 256-bit).",
	"`out[i] = a[i] logical >> IMM`. 8-wide chunks, scalar remainder."
);
avx2_i32_shift_imm!(
	sra_i32x8, sra_i32_slice, vpsrad, _mm256_sra_epi32, |x: i32, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`vpsrad`, 256-bit).",
	"`out[i] = a[i] arithmetic >> IMM`. 8-wide chunks, scalar remainder."
);
avx2_u32_shift_imm!(
	shl_u32x8, shl_u32_slice, vpslld_u, _mm256_sll_epi32, |x: u32, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpslld`, 256-bit).",
	"`out[i] = a[i] << IMM`. 8-wide chunks, scalar remainder."
);
avx2_u32_shift_imm!(
	shr_u32x8, shr_u32_slice, vpsrld_u, _mm256_srl_epi32, |x: u32, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`vpsrld`, 256-bit).",
	"`out[i] = a[i] >> IMM`. 8-wide chunks, scalar remainder."
);

// i8/u8/i16/u16: `pub(crate)` default (auto_up cascade calls these; see
// module doc). i8/u8 mul/shifts stay `vis = pub` below: no HW path below
// AVX2 exists for those, so no cascade, so no auto_up re-export.

macro_rules! avx2_i8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = i8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u8_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_i16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u16_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i8_binop!(
	add_i8x32, add_i8_slice, paddb, _mm256_add_epi8, |x: i8, y: i8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddb`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	sub_i8x32, sub_i8_slice, psubb, _mm256_sub_epi8, |x: i8, y: i8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubb`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	adds_i8x32, adds_i8_slice, paddsb, _mm256_adds_epi8, |x: i8, y: i8| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddsb`, 256-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	subs_i8x32, subs_i8_slice, psubsb, _mm256_subs_epi8, |x: i8, y: i8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubsb`, 256-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	min_i8x32, min_i8_slice, pminsb, _mm256_min_epi8, |x, y| x.min(y),
	"Per-lane signed min (`vpminsb`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	max_i8x32, max_i8_slice, pmaxsb, _mm256_max_epi8, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsb`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	cmpeq_i8x32, cmpeq_i8_slice, vpcmpeqb, _mm256_cmpeq_epi8,
	|x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`vpcmpeqb`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	cmpgt_i8x32, cmpgt_i8_slice, vpcmpgtb, _mm256_cmpgt_epi8,
	|x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`vpcmpgtb`, 256-bit): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	and_i8x32, and_i8_slice, vpand_i8, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	or_i8x32, or_i8_slice, vpor_i8, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	xor_i8x32, xor_i8_slice, vpxor_i8, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 32-wide chunks, scalar remainder."
);
avx2_i8_binop!(
	andnot_i8x32, andnot_i8_slice, vpandn_i8, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 32-wide chunks, scalar remainder."
);
// mul_i8/u8, shl/shr/sra_i8/u8: no native 8-bit SIMD multiply or
// byte-granularity shift exists on x86 at any tier, but composed forms beat
// a scalar loop. 256-bit mirror of `sse2.rs`'s composition (same
// per-128-bit-lane cancellation reasoning for `unpacklo/hi`+`pack*` as
// `alignr_u8x32`'s lane-lock analysis
simd_binop! {
	token = Avx2, vis = pub, target_feature = "avx2",
	fixed_fn = mul_i8x32, slice_fn = mul_i8_slice, intrinsic_fn = mul_i8x32_intrinsic,
	width = 32, elem = i8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = mullo_epi8x32_composed, scalar = |x: i8, y: i8| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, wrapping: composed via zero-extend+`vpmullw`+`vpackuswb` (no native 8-bit SIMD multiply on x86 at any tier).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 32-wide chunks, scalar remainder.",
}

impl Avx2 {
	/// Per-lane logical left shift by `IMM` (masked to `IMM & 7`, matching
	/// [`u8::wrapping_shl`]), wrapping: composed via widening to 16-bit lanes
	/// (`vpsllw`, register-count form) + a byte-repeated mask that clears the
	/// bits shifted in from each byte's neighbor
	#[inline]
	pub fn shl_u8x32<const IMM: u32>(self, a: [u8; 32]) -> [u8; 32] {
		unsafe { shl_u8x32_composed::<IMM>(&a) }
	}

	/// `out[i] = a[i].wrapping_shl(IMM)`. 32-wide chunks
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shl_u8_slice<const IMM: u32>(self, a: &[u8], out: &mut [u8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u8; 32] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shl_u8x32::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shl(IMM);
		}
	}

	#[inline]
	pub fn shl_i8x32<const IMM: u32>(self, a: [i8; 32]) -> [i8; 32] {
		let au: [u8; 32] = core::array::from_fn(|i| a[i] as u8);
		let r = self.shl_u8x32::<IMM>(au);
		core::array::from_fn(|i| r[i] as i8)
	}

	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shl_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 32] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shl_i8x32::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shl(IMM);
		}
	}

	/// Per-lane logical right shift by `IMM` (masked to `IMM & 7`, matching
	/// [`u8::wrapping_shr`]): composed via widening to 16-bit lanes (`vpsrlw`,
	/// register-count form) + a byte-repeated mask that clears the bits
	/// shifted in from each byte's neighbor
	#[inline]
	pub fn shr_u8x32<const IMM: u32>(self, a: [u8; 32]) -> [u8; 32] {
		unsafe { shr_u8x32_composed::<IMM>(&a) }
	}

	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shr_u8_slice<const IMM: u32>(self, a: &[u8], out: &mut [u8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [u8; 32] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shr_u8x32::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shr(IMM);
		}
	}

	/// Logical (unsigned) view of [`shr_u8x32`](Self::shr_u8x32): matches
	/// scalar-only `shr_i8x32`'s semantics (`(a as u8) >> IMM as i8`).
	#[inline]
	pub fn shr_i8x32<const IMM: u32>(self, a: [i8; 32]) -> [i8; 32] {
		let au: [u8; 32] = core::array::from_fn(|i| a[i] as u8);
		let r = self.shr_u8x32::<IMM>(au);
		core::array::from_fn(|i| r[i] as i8)
	}

	/// # Panics
	/// `out.len() != a.len()`.
	pub fn shr_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 32] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.shr_i8x32::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = ((x as u8).wrapping_shr(IMM)) as i8;
		}
	}

	/// Per-lane arithmetic right shift by `IMM` (masked to `IMM & 7`, matching
	/// [`i8::wrapping_shr`])
	#[inline]
	pub fn sra_i8x32<const IMM: u32>(self, a: [i8; 32]) -> [i8; 32] {
		unsafe { sra_i8x32_composed::<IMM>(&a) }
	}

	/// # Panics
	/// `out.len() != a.len()`.
	pub fn sra_i8_slice<const IMM: u32>(self, a: &[i8], out: &mut [i8]) {
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
			let av: [i8; 32] = ac.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.sra_i8x32::<IMM>(av));
		}
		for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
			*o = x.wrapping_shr(IMM);
		}
	}
}

/// Composed 8-bit wrapping multiply, 256-bit mirror of `sse2.rs`'s
/// `mullo_epi8x16_composed`
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mullo_epi8x32_composed(a: __m256i, b: __m256i) -> __m256i {
	let zero = _mm256_setzero_si256();
	let a_lo = _mm256_unpacklo_epi8(a, zero);
	let a_hi = _mm256_unpackhi_epi8(a, zero);
	let b_lo = _mm256_unpacklo_epi8(b, zero);
	let b_hi = _mm256_unpackhi_epi8(b, zero);
	let p_lo = _mm256_mullo_epi16(a_lo, b_lo);
	let p_hi = _mm256_mullo_epi16(a_hi, b_hi);
	let mask = _mm256_set1_epi16(0x00FF);
	_mm256_packus_epi16(_mm256_and_si256(p_lo, mask), _mm256_and_si256(p_hi, mask))
}

/// Composed per-byte logical left shift, 256-bit mirror of `sse2.rs`'s
/// `shl_u8x16_composed`
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shl_u8x32_composed<const IMM: u32>(a: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let shift = IMM & 7;
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let count = _mm_cvtsi32_si128(shift as i32);
		let wide = _mm256_sll_epi16(va, count);
		let mask_byte = ((0xFFu32 << shift) & 0xFF) as u8;
		let mask = _mm256_set1_epi8(mask_byte as i8);
		let vr = _mm256_and_si256(wide, mask);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Composed per-byte logical right shift, 256-bit mirror of
/// [`shl_u8x32_composed`] via `vpsrlw` + a `0xFF>>(IMM&7)` byte mask
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shr_u8x32_composed<const IMM: u32>(a: &[u8; 32]) -> [u8; 32] {
	unsafe {
		let shift = IMM & 7;
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let count = _mm_cvtsi32_si128(shift as i32);
		let wide = _mm256_srl_epi16(va, count);
		let mask_byte = (0xFFu32 >> shift) as u8;
		let mask = _mm256_set1_epi8(mask_byte as i8);
		let vr = _mm256_and_si256(wide, mask);
		let mut out = [0u8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// Composed per-byte arithmetic right shift, 256-bit mirror of `sse2.rs`'s
/// `sra_i8x16_composed`: sign-extend each byte to a full 16-bit lane
/// (`vpunpcklbw`/`vpunpckhbw` with self, then `vpsraw` by the literal 8
///
/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sra_i8x32_composed<const IMM: u32>(a: &[i8; 32]) -> [i8; 32] {
	unsafe {
		let shift = IMM & 7;
		let va: __m256i = _mm256_loadu_si256(a.as_ptr().cast());
		let lo_ext = _mm256_srai_epi16::<8>(_mm256_unpacklo_epi8(va, va));
		let hi_ext = _mm256_srai_epi16::<8>(_mm256_unpackhi_epi8(va, va));
		let count = _mm_cvtsi32_si128(shift as i32);
		let lo_shifted = _mm256_sra_epi16(lo_ext, count);
		let hi_shifted = _mm256_sra_epi16(hi_ext, count);
		let vr = _mm256_packs_epi16(lo_shifted, hi_shifted);
		let mut out = [0i8; 32];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), vr);
		out
	}
}
simd_unop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = abs_i8x32, slice_fn = abs_i8_slice, intrinsic_fn = pabsb,
	width = 32, elem = i8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_abs_epi8, scalar = |x: i8| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsb`, 256-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 32-wide chunks, scalar remainder.",
}

avx2_u8_binop!(
	add_u8x32, add_u8_slice, paddb_u, _mm256_add_epi8, |x: u8, y: u8| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddb`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	sub_u8x32, sub_u8_slice, psubb_u, _mm256_sub_epi8, |x: u8, y: u8| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubb`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	adds_u8x32, adds_u8_slice, paddusb, _mm256_adds_epu8, |x: u8, y: u8| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddusb`, 256-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	subs_u8x32, subs_u8_slice, psubusb, _mm256_subs_epu8, |x: u8, y: u8| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubusb`, 256-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	min_u8x32, min_u8_slice, pminub, _mm256_min_epu8, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminub`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	max_u8x32, max_u8_slice, pmaxub, _mm256_max_epu8, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxub`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	avg_u8x32, avg_u8_slice, pavgb, _mm256_avg_epu8, |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8,
	"Per-lane rounded unsigned average, `(a+b+1)/2` (`vpavgb`, 256-bit). No signed form exists in the ISA.",
	"`out[i] = (a[i] as u16 + b[i] as u16 + 1) / 2`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	cmpeq_u8x32, cmpeq_u8_slice, vpcmpeqb_u, _mm256_cmpeq_epi8,
	|x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`vpcmpeqb`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	and_u8x32, and_u8_slice, vpand_u8, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	or_u8x32, or_u8_slice, vpor_u8, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	xor_u8x32, xor_u8_slice, vpxor_u8, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 32-wide chunks, scalar remainder."
);
avx2_u8_binop!(
	andnot_u8x32, andnot_u8_slice, vpandn_u8, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 32-wide chunks, scalar remainder."
);

simd_binop! {
	token = Avx2, vis = pub, target_feature = "avx2",
	fixed_fn = mul_u8x32, slice_fn = mul_u8_slice, intrinsic_fn = mul_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = mullo_epi8x32_composed, scalar = |x: u8, y: u8| x.wrapping_mul(y),
	fixed_doc = "`a * b` per lane, wrapping: composed via zero-extend+`vpmullw`+`vpackuswb` (no native 8-bit SIMD multiply on x86 at any tier).",
	slice_doc = "`out[i] = a[i].wrapping_mul(b[i])`. 32-wide chunks, scalar remainder.",
}

avx2_i16_binop!(
	add_i16x16, add_i16_slice, paddw, _mm256_add_epi16, |x: i16, y: i16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddw`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	sub_i16x16, sub_i16_slice, psubw, _mm256_sub_epi16, |x: i16, y: i16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubw`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	adds_i16x16, adds_i16_slice, paddsw, _mm256_adds_epi16, |x: i16, y: i16| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddsw`, 256-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	subs_i16x16, subs_i16_slice, psubsw, _mm256_subs_epi16, |x: i16, y: i16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubsw`, 256-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	mul_i16x16, mul_i16_slice, pmullw, _mm256_mullo_epi16, |x: i16, y: i16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`vpmullw`, 256-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	min_i16x16, min_i16_slice, pminsw, _mm256_min_epi16, |x, y| x.min(y),
	"Per-lane signed min (`vpminsw`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	max_i16x16, max_i16_slice, pmaxsw, _mm256_max_epi16, |x, y| x.max(y),
	"Per-lane signed max (`vpmaxsw`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	cmpeq_i16x16, cmpeq_i16_slice, vpcmpeqw, _mm256_cmpeq_epi16,
	|x, y| if x == y { -1 } else { 0 },
	"Lane equality mask (`vpcmpeqw`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	cmpgt_i16x16, cmpgt_i16_slice, vpcmpgtw, _mm256_cmpgt_epi16,
	|x, y| if x > y { -1 } else { 0 },
	"Lane greater-than mask (`vpcmpgtw`, 256-bit): all-1s if `a>b`, else 0.",
	"`out[i] = all-1s if a[i]>b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	and_i16x16, and_i16_slice, vpand_i16, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	or_i16x16, or_i16_slice, vpor_i16, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	xor_i16x16, xor_i16_slice, vpxor_i16, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
avx2_i16_binop!(
	andnot_i16x16, andnot_i16_slice, vpandn_i16, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
simd_unop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = abs_i16x16, slice_fn = abs_i16_slice, intrinsic_fn = pabsw,
	width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_abs_epi16, scalar = |x: i16| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsw`, 256-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 16-wide chunks, scalar remainder.",
}
simd_unop! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = abs_i32x8, slice_fn = abs_i32_slice, intrinsic_fn = pabsd,
	width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
	intrinsic = _mm256_abs_epi32, scalar = |x: i32| x.wrapping_abs(),
	fixed_doc = "Per-lane absolute value (`vpabsd`, 256-bit).",
	slice_doc = "`out[i] = a[i].wrapping_abs()`. 8-wide chunks, scalar remainder.",
}

avx2_u16_binop!(
	add_u16x16, add_u16_slice, paddw_u, _mm256_add_epi16, |x: u16, y: u16| x.wrapping_add(y),
	"`a + b` per lane, wrapping (`vpaddw`, 256-bit).",
	"`out[i] = a[i].wrapping_add(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	sub_u16x16, sub_u16_slice, psubw_u, _mm256_sub_epi16, |x: u16, y: u16| x.wrapping_sub(y),
	"`a - b` per lane, wrapping (`vpsubw`, 256-bit).",
	"`out[i] = a[i].wrapping_sub(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	adds_u16x16, adds_u16_slice, paddusw, _mm256_adds_epu16, |x: u16, y: u16| x.saturating_add(y),
	"`a + b` per lane, saturating (`vpaddusw`, 256-bit).",
	"`out[i] = a[i].saturating_add(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	subs_u16x16, subs_u16_slice, psubusw, _mm256_subs_epu16, |x: u16, y: u16| x.saturating_sub(y),
	"`a - b` per lane, saturating (`vpsubusw`, 256-bit).",
	"`out[i] = a[i].saturating_sub(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	mul_u16x16, mul_u16_slice, pmullw_u, _mm256_mullo_epi16, |x: u16, y: u16| x.wrapping_mul(y),
	"`a * b` per lane, low 16 bits (`vpmullw`, 256-bit).",
	"`out[i] = a[i].wrapping_mul(b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	min_u16x16, min_u16_slice, pminuw, _mm256_min_epu16, |x, y| x.min(y),
	"Per-lane unsigned min (`vpminuw`, 256-bit).",
	"`out[i] = min(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	max_u16x16, max_u16_slice, pmaxuw, _mm256_max_epu16, |x, y| x.max(y),
	"Per-lane unsigned max (`vpmaxuw`, 256-bit).",
	"`out[i] = max(a[i], b[i])`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	avg_u16x16, avg_u16_slice, pavgw, _mm256_avg_epu16, |x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16,
	"Per-lane rounded unsigned average, `(a+b+1)/2` (`vpavgw`, 256-bit). No signed form exists in the ISA.",
	"`out[i] = (a[i] as u32 + b[i] as u32 + 1) / 2`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	cmpeq_u16x16, cmpeq_u16_slice, vpcmpeqw_u, _mm256_cmpeq_epi16,
	|x, y| if x == y { !0 } else { 0 },
	"Lane equality mask (`vpcmpeqw`, 256-bit): all-1s if equal, else 0.",
	"`out[i] = all-1s if a[i]==b[i] else 0`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	and_u16x16, and_u16_slice, vpand_u16, _mm256_and_si256, |x, y| x & y,
	"`a & b` per lane (`vpand`, 256-bit).",
	"`out[i] = a[i] & b[i]`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	or_u16x16, or_u16_slice, vpor_u16, _mm256_or_si256, |x, y| x | y,
	"`a | b` per lane (`vpor`, 256-bit).",
	"`out[i] = a[i] | b[i]`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	xor_u16x16, xor_u16_slice, vpxor_u16, _mm256_xor_si256, |x, y| x ^ y,
	"`a ^ b` per lane (`vpxor`, 256-bit).",
	"`out[i] = a[i] ^ b[i]`. 16-wide chunks, scalar remainder."
);
avx2_u16_binop!(
	andnot_u16x16, andnot_u16_slice, vpandn_u16, _mm256_andnot_si256, |x, y| !x & y,
	"`!a & b` per lane (`vpandn`, 256-bit).",
	"`out[i] = !a[i] & b[i]`. 16-wide chunks, scalar remainder."
);

// Narrow ordering: signed native; unsigned sign-bit flip; lt/le/ge = swap/NOT of cmpgt.
impl Avx2 {
	/// Unsigned greater-than mask. Sign-bit flip + [`cmpgt_i8x32`].
	#[inline]
	pub fn cmpgt_u8x32(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		let ai: [i8; 32] = core::array::from_fn(|i| (a[i] ^ 0x80) as i8);
		let bi: [i8; 32] = core::array::from_fn(|i| (b[i] ^ 0x80) as i8);
		let r = self.cmpgt_i8x32(ai, bi);
		core::array::from_fn(|i| r[i] as u8)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u8` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(32);
		let b_chunks = b.chunks_exact(32);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(32);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u8; 32] = ac.try_into().expect("chunks_exact width");
			let bv: [u8; 32] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u8x32(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	/// Unsigned greater-than mask. Sign-bit flip + [`cmpgt_i16x16`].
	#[inline]
	pub fn cmpgt_u16x16(self, a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
		let ai: [i16; 16] = core::array::from_fn(|i| (a[i] ^ 0x8000) as i16);
		let bi: [i16; 16] = core::array::from_fn(|i| (b[i] ^ 0x8000) as i16);
		let r = self.cmpgt_i16x16(ai, bi);
		core::array::from_fn(|i| r[i] as u16)
	}

	/// `out[i] = all-1s if a[i]>b[i] else 0` (`u16` view).
	///
	/// # Panics
	/// `a.len() != b.len() || out.len() != a.len()`.
	pub(crate) fn cmpgt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		assert_eq!(a.len(), b.len());
		assert_eq!(out.len(), a.len());
		let a_chunks = a.chunks_exact(16);
		let b_chunks = b.chunks_exact(16);
		let a_rem = a_chunks.remainder();
		let b_rem = b_chunks.remainder();
		let mut out_chunks = out.chunks_exact_mut(16);
		for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
			let av: [u16; 16] = ac.try_into().expect("chunks_exact width");
			let bv: [u16; 16] = bc.try_into().expect("chunks_exact width");
			oc.copy_from_slice(&self.cmpgt_u16x16(av, bv));
		}
		for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
			*o = if x > y { !0 } else { 0 };
		}
	}

	#[inline]
	pub fn cmplt_i8x32(self, a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
		self.cmpgt_i8x32(b, a)
	}
	pub(crate) fn cmplt_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_i8x32(self, a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
		core::array::from_fn(|i| !self.cmpgt_i8x32(a, b)[i])
	}
	pub(crate) fn cmple_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmpgt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_i8x32(self, a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
		core::array::from_fn(|i| !self.cmplt_i8x32(a, b)[i])
	}
	pub(crate) fn cmpge_i8_slice(self, a: &[i8], b: &[i8], out: &mut [i8]) {
		self.cmplt_i8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_u8x32(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		self.cmpgt_u8x32(b, a)
	}
	pub(crate) fn cmplt_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_u8x32(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		core::array::from_fn(|i| !self.cmpgt_u8x32(a, b)[i])
	}
	pub(crate) fn cmple_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmpgt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_u8x32(self, a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
		core::array::from_fn(|i| !self.cmplt_u8x32(a, b)[i])
	}
	pub(crate) fn cmpge_u8_slice(self, a: &[u8], b: &[u8], out: &mut [u8]) {
		self.cmplt_u8_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_i16x16(self, a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
		self.cmpgt_i16x16(b, a)
	}
	pub(crate) fn cmplt_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_i16x16(self, a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
		core::array::from_fn(|i| !self.cmpgt_i16x16(a, b)[i])
	}
	pub(crate) fn cmple_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmpgt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_i16x16(self, a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
		core::array::from_fn(|i| !self.cmplt_i16x16(a, b)[i])
	}
	pub(crate) fn cmpge_i16_slice(self, a: &[i16], b: &[i16], out: &mut [i16]) {
		self.cmplt_i16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}

	#[inline]
	pub fn cmplt_u16x16(self, a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
		self.cmpgt_u16x16(b, a)
	}
	pub(crate) fn cmplt_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(b, a, out);
	}
	#[inline]
	pub fn cmple_u16x16(self, a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
		core::array::from_fn(|i| !self.cmpgt_u16x16(a, b)[i])
	}
	pub(crate) fn cmple_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmpgt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
	#[inline]
	pub fn cmpge_u16x16(self, a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
		core::array::from_fn(|i| !self.cmplt_u16x16(a, b)[i])
	}
	pub(crate) fn cmpge_u16_slice(self, a: &[u16], b: &[u16], out: &mut [u16]) {
		self.cmplt_u16_slice(a, b, out);
		for o in out.iter_mut() {
			*o = !*o;
		}
	}
}

macro_rules! avx2_i16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx2_u16_shift_imm {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $shift:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_shift_imm! {
			token = Avx2, target_feature = "avx2",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 16, elem = u16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
			shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx2_i16_shift_imm!(
	shl_i16x16, shl_i16_slice, vpsllw, _mm256_sll_epi16, |x: i16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpsllw`, 256-bit).",
	"`out[i] = a[i] << IMM`. 16-wide chunks, scalar remainder."
);
avx2_i16_shift_imm!(
	shr_i16x16, shr_i16_slice, vpsrlw, _mm256_srl_epi16, |x: i16, imm| ((x as u16).wrapping_shr(imm)) as i16,
	"`a >> IMM` logical per lane (`vpsrlw`, 256-bit).",
	"`out[i] = a[i] logical >> IMM`. 16-wide chunks, scalar remainder."
);
avx2_i16_shift_imm!(
	sra_i16x16, sra_i16_slice, vpsraw, _mm256_sra_epi16, |x: i16, imm| x.wrapping_shr(imm),
	"`a >> IMM` arithmetic per lane (`vpsraw`, 256-bit).",
	"`out[i] = a[i] arithmetic >> IMM`. 16-wide chunks, scalar remainder."
);
avx2_u16_shift_imm!(
	shl_u16x16, shl_u16_slice, vpsllw_u, _mm256_sll_epi16, |x: u16, imm| x.wrapping_shl(imm),
	"`a << IMM` per lane (`vpsllw`, 256-bit).",
	"`out[i] = a[i] << IMM`. 16-wide chunks, scalar remainder."
);
avx2_u16_shift_imm!(
	shr_u16x16, shr_u16_slice, vpsrlw_u, _mm256_srl_epi16, |x: u16, imm| x.wrapping_shr(imm),
	"`a >> IMM` logical per lane (`vpsrlw`, 256-bit).",
	"`out[i] = a[i] >> IMM`. 16-wide chunks, scalar remainder."
);

simd_movemask! {
	token = Avx2, target_feature = "avx2",
	fixed_fn = movemask_i8x32, intrinsic_fn = movemask_epi8,
	width = 32, elem = i8, vec = __m256i, mask = u32,
	loadu = _mm256_loadu_si256, intrinsic = _mm256_movemask_epi8,
	doc = "Sign-bit mask, one bit per lane (`vpmovmskb`).",
}

// Partial (ragged-tail) load/store, AVX2's `VPMASKMOVD`/`VPMASKMOVQ` (the
// integer counterpart of `Avx`'s `VMASKMOVPS`/`PD`; see that module's doc
// for why the mask has to be built as a per-lane compare here instead of
// AVX-512's scalar bit-arithmetic trick).
const PARTIAL_LANE_IDX_I32X8: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const PARTIAL_LANE_IDX_I64X4: [i64; 4] = [0, 1, 2, 3];

impl Avx2 {
	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_i32x8(self, slice: &[i32]) -> [i32; 8] {
		unsafe { partial_load_i32x8_intrinsic(slice.as_ptr(), slice.len().min(8) as u32) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_i32x8(self, v: [i32; 8], slice: &mut [i32]) {
		unsafe { partial_store_i32x8_intrinsic(slice.as_mut_ptr(), slice.len().min(8) as u32, &v) }
	}

	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_u32x8(self, slice: &[u32]) -> [u32; 8] {
		unsafe { partial_load_u32x8_intrinsic(slice.as_ptr(), slice.len().min(8) as u32) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_u32x8(self, v: [u32; 8], slice: &mut [u32]) {
		unsafe { partial_store_u32x8_intrinsic(slice.as_mut_ptr(), slice.len().min(8) as u32, &v) }
	}

	/// Loads `slice.len().min(4)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_i64x4(self, slice: &[i64]) -> [i64; 4] {
		unsafe { partial_load_i64x4_intrinsic(slice.as_ptr(), slice.len().min(4) as u32) }
	}

	/// Writes `slice.len().min(4)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_i64x4(self, v: [i64; 4], slice: &mut [i64]) {
		unsafe { partial_store_i64x4_intrinsic(slice.as_mut_ptr(), slice.len().min(4) as u32, &v) }
	}

	/// Loads `slice.len().min(4)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_u64x4(self, slice: &[u64]) -> [u64; 4] {
		unsafe { partial_load_u64x4_intrinsic(slice.as_ptr(), slice.len().min(4) as u32) }
	}

	/// Writes `slice.len().min(4)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_u64x4(self, v: [u64; 4], slice: &mut [u64]) {
		unsafe { partial_store_u64x4_intrinsic(slice.as_mut_ptr(), slice.len().min(4) as u32, &v) }
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_mask_i32x8(n: u32) -> __m256i {
	unsafe {
		let idx = _mm256_loadu_si256(PARTIAL_LANE_IDX_I32X8.as_ptr().cast());
		let n_bcast = [n as i32; 8];
		let n_bcast = _mm256_loadu_si256(n_bcast.as_ptr().cast());
		_mm256_cmpgt_epi32(n_bcast, idx)
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_load_i32x8_intrinsic(ptr: *const i32, n: u32) -> [i32; 8] {
	unsafe {
		let mask = partial_mask_i32x8(n);
		let v = _mm256_maskload_epi32(ptr, mask);
		let mut out = [0i32; 8];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_store_i32x8_intrinsic(ptr: *mut i32, n: u32, v: &[i32; 8]) {
	unsafe {
		let mask = partial_mask_i32x8(n);
		let vv = _mm256_loadu_si256(v.as_ptr().cast());
		_mm256_maskstore_epi32(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_load_u32x8_intrinsic(ptr: *const u32, n: u32) -> [u32; 8] {
	unsafe {
		let mask = partial_mask_i32x8(n);
		let v = _mm256_maskload_epi32(ptr.cast(), mask);
		let mut out = [0u32; 8];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_store_u32x8_intrinsic(ptr: *mut u32, n: u32, v: &[u32; 8]) {
	unsafe {
		let mask = partial_mask_i32x8(n);
		let vv = _mm256_loadu_si256(v.as_ptr().cast());
		_mm256_maskstore_epi32(ptr.cast(), mask, vv);
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_mask_i64x4(n: u32) -> __m256i {
	unsafe {
		let idx = _mm256_loadu_si256(PARTIAL_LANE_IDX_I64X4.as_ptr().cast());
		let n_bcast = [n as i64; 4];
		let n_bcast = _mm256_loadu_si256(n_bcast.as_ptr().cast());
		_mm256_cmpgt_epi64(n_bcast, idx)
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_load_i64x4_intrinsic(ptr: *const i64, n: u32) -> [i64; 4] {
	unsafe {
		let mask = partial_mask_i64x4(n);
		let v = _mm256_maskload_epi64(ptr, mask);
		let mut out = [0i64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_store_i64x4_intrinsic(ptr: *mut i64, n: u32, v: &[i64; 4]) {
	unsafe {
		let mask = partial_mask_i64x4(n);
		let vv = _mm256_loadu_si256(v.as_ptr().cast());
		_mm256_maskstore_epi64(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_load_u64x4_intrinsic(ptr: *const u64, n: u32) -> [u64; 4] {
	unsafe {
		let mask = partial_mask_i64x4(n);
		let v = _mm256_maskload_epi64(ptr.cast(), mask);
		let mut out = [0u64; 4];
		_mm256_storeu_si256(out.as_mut_ptr().cast(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX2 via [`Avx2`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn partial_store_u64x4_intrinsic(ptr: *mut u64, n: u32, v: &[u64; 4]) {
	unsafe {
		let mask = partial_mask_i64x4(n);
		let vv = _mm256_loadu_si256(v.as_ptr().cast());
		_mm256_maskstore_epi64(ptr.cast(), mask, vv);
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx2.rs"]
mod tests;
