//! AVX512VBMI2 (Ice Lake, 2019): 512-bit lane shifts (`shldv`/`shrdv`, imm forms) and masked
//! compress/expand for 8/16-bit lanes. Token: [`Avx512Vbmi2`] (`"avx512vbmi2"`).
//! VL forms: `super::avx512vl`. Pointer `compressstoreu`/`expandloadu` deferred.

use core::arch::x86_64::{
	__m512i, _mm512_loadu_si512, _mm512_mask_compress_epi16, _mm512_mask_compress_epi8, _mm512_mask_expand_epi16,
	_mm512_mask_expand_epi8, _mm512_mask_compressstoreu_epi16, _mm512_mask_compressstoreu_epi8,
	_mm512_mask_expandloadu_epi16, _mm512_mask_expandloadu_epi8, _mm512_maskz_expandloadu_epi16,
	_mm512_maskz_expandloadu_epi8, _mm512_mask_shldi_epi16, _mm512_mask_shldi_epi32, _mm512_mask_shldi_epi64,
	_mm512_mask_shldv_epi16, _mm512_mask_shldv_epi32, _mm512_mask_shldv_epi64, _mm512_mask_shrdi_epi16,
	_mm512_mask_shrdi_epi32, _mm512_mask_shrdi_epi64, _mm512_mask_shrdv_epi16, _mm512_mask_shrdv_epi32,
	_mm512_mask_shrdv_epi64, _mm512_maskz_compress_epi16, _mm512_maskz_compress_epi8, _mm512_maskz_expand_epi16,
	_mm512_maskz_expand_epi8, _mm512_maskz_shldi_epi16, _mm512_maskz_shldi_epi32, _mm512_maskz_shldi_epi64,
	_mm512_maskz_shldv_epi16, _mm512_maskz_shldv_epi32, _mm512_maskz_shldv_epi64, _mm512_maskz_shrdi_epi16,
	_mm512_maskz_shrdi_epi32, _mm512_maskz_shrdi_epi64, _mm512_maskz_shrdv_epi16, _mm512_maskz_shrdv_epi32,
	_mm512_maskz_shrdv_epi64, _mm512_shldi_epi16, _mm512_shldi_epi32, _mm512_shldi_epi64, _mm512_shldv_epi16,
	_mm512_shldv_epi32, _mm512_shldv_epi64, _mm512_shrdi_epi16, _mm512_shrdi_epi32, _mm512_shrdi_epi64,
	_mm512_shrdv_epi16, _mm512_shrdv_epi32, _mm512_shrdv_epi64, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{
	simd_binop_imm, simd_binop_imm_masked, simd_compressstoreu, simd_expandloadu, simd_ternop, simd_ternop_masked,
	simd_unop_masked,
};

/// Proof token: AVX512VBMI2, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Vbmi2(());

impl Avx512Vbmi2 {
	/// `None` if the CPU (or the compile-time target) lacks AVX512VBMI2.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512vbmi2).then_some(Avx512Vbmi2(()))
	}
}

/// `(((a as wide) << W) | b) << (c & (W-1))`, upper `W` bits. Shared with
/// `Avx512Vbmi2Vl`'s ops in `super::avx512vl`.
pub(crate) fn shldv_u16_scalar(a: u16, b: u16, c: u16) -> u16 {
	let cat: u32 = ((a as u32) << 16) | (b as u32);
	(cat << ((c & 15) as u32) >> 16) as u16
}

/// Signed sibling of [`shldv_u16_scalar`] (bit pattern only, no sign meaning).
pub(crate) fn shldv_i16_scalar(a: i16, b: i16, c: i16) -> i16 {
	shldv_u16_scalar(a as u16, b as u16, c as u16) as i16
}

/// `((b as wide) << W) | a) >> (c & (W-1))`, lower `W` bits.
pub(crate) fn shrdv_u16_scalar(a: u16, b: u16, c: u16) -> u16 {
	let cat: u32 = ((b as u32) << 16) | (a as u32);
	(cat >> ((c & 15) as u32)) as u16
}

/// Signed sibling of [`shrdv_u16_scalar`].
pub(crate) fn shrdv_i16_scalar(a: i16, b: i16, c: i16) -> i16 {
	shrdv_u16_scalar(a as u16, b as u16, c as u16) as i16
}

/// 32-bit sibling of [`shldv_u16_scalar`].
pub(crate) fn shldv_u32_scalar(a: u32, b: u32, c: u32) -> u32 {
	let cat: u64 = ((a as u64) << 32) | (b as u64);
	(cat << (c & 31) >> 32) as u32
}

/// Signed sibling of [`shldv_u32_scalar`].
pub(crate) fn shldv_i32_scalar(a: i32, b: i32, c: i32) -> i32 {
	shldv_u32_scalar(a as u32, b as u32, c as u32) as i32
}

/// 32-bit sibling of [`shrdv_u16_scalar`].
pub(crate) fn shrdv_u32_scalar(a: u32, b: u32, c: u32) -> u32 {
	let cat: u64 = ((b as u64) << 32) | (a as u64);
	(cat >> (c & 31)) as u32
}

/// Signed sibling of [`shrdv_u32_scalar`].
pub(crate) fn shrdv_i32_scalar(a: i32, b: i32, c: i32) -> i32 {
	shrdv_u32_scalar(a as u32, b as u32, c as u32) as i32
}

/// 64-bit sibling of [`shldv_u16_scalar`] (128-bit software intermediate).
pub(crate) fn shldv_u64_scalar(a: u64, b: u64, c: u64) -> u64 {
	let cat: u128 = ((a as u128) << 64) | (b as u128);
	(cat << (c & 63) >> 64) as u64
}

/// Signed sibling of [`shldv_u64_scalar`].
pub(crate) fn shldv_i64_scalar(a: i64, b: i64, c: i64) -> i64 {
	shldv_u64_scalar(a as u64, b as u64, c as u64) as i64
}

/// 64-bit sibling of [`shrdv_u16_scalar`] (128-bit software intermediate).
pub(crate) fn shrdv_u64_scalar(a: u64, b: u64, c: u64) -> u64 {
	let cat: u128 = ((b as u128) << 64) | (a as u128);
	(cat >> (c & 63)) as u64
}

/// Signed sibling of [`shrdv_u64_scalar`].
pub(crate) fn shrdv_i64_scalar(a: i64, b: i64, c: i64) -> i64 {
	shrdv_u64_scalar(a as u64, b as u64, c as u64) as i64
}

// shldi/shrdi: same math as shldv/shrdv, just with a compile-time `imm`
// instead of a per-lane runtime `c`: reuse the shldv/shrdv scalar refs
// directly rather than re-deriving the bit math.

/// `shldi(a, b, imm) == shldv(a, b, imm)` (per-lane `c` is just `imm`
/// broadcast). Shared with `Avx512Vbmi2Vl`'s ops in `super::avx512vl`.
pub(crate) fn shldi_u16_scalar(a: u16, b: u16, imm: i32) -> u16 {
	shldv_u16_scalar(a, b, imm as u16)
}

/// Signed sibling of [`shldi_u16_scalar`].
pub(crate) fn shldi_i16_scalar(a: i16, b: i16, imm: i32) -> i16 {
	shldv_i16_scalar(a, b, imm as i16)
}

/// `shrdi(a, b, imm) == shrdv(a, b, imm)`.
pub(crate) fn shrdi_u16_scalar(a: u16, b: u16, imm: i32) -> u16 {
	shrdv_u16_scalar(a, b, imm as u16)
}

/// Signed sibling of [`shrdi_u16_scalar`].
pub(crate) fn shrdi_i16_scalar(a: i16, b: i16, imm: i32) -> i16 {
	shrdv_i16_scalar(a, b, imm as i16)
}

/// 32-bit sibling of [`shldi_u16_scalar`].
pub(crate) fn shldi_u32_scalar(a: u32, b: u32, imm: i32) -> u32 {
	shldv_u32_scalar(a, b, imm as u32)
}

/// Signed sibling of [`shldi_u32_scalar`].
pub(crate) fn shldi_i32_scalar(a: i32, b: i32, imm: i32) -> i32 {
	shldv_i32_scalar(a, b, imm)
}

/// 32-bit sibling of [`shrdi_u16_scalar`].
pub(crate) fn shrdi_u32_scalar(a: u32, b: u32, imm: i32) -> u32 {
	shrdv_u32_scalar(a, b, imm as u32)
}

/// Signed sibling of [`shrdi_u32_scalar`].
pub(crate) fn shrdi_i32_scalar(a: i32, b: i32, imm: i32) -> i32 {
	shrdv_i32_scalar(a, b, imm)
}

/// 64-bit sibling of [`shldi_u16_scalar`].
pub(crate) fn shldi_u64_scalar(a: u64, b: u64, imm: i32) -> u64 {
	shldv_u64_scalar(a, b, imm as u64)
}

/// Signed sibling of [`shldi_u64_scalar`].
pub(crate) fn shldi_i64_scalar(a: i64, b: i64, imm: i32) -> i64 {
	shldv_i64_scalar(a, b, imm as i64)
}

/// 64-bit sibling of [`shrdi_u16_scalar`].
pub(crate) fn shrdi_u64_scalar(a: u64, b: u64, imm: i32) -> u64 {
	shrdv_u64_scalar(a, b, imm as u64)
}

/// Signed sibling of [`shrdi_u64_scalar`].
pub(crate) fn shrdi_i64_scalar(a: i64, b: i64, imm: i32) -> i64 {
	shrdv_i64_scalar(a, b, imm as i64)
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_i16x32, slice_fn = shldv_i16_slice, intrinsic_fn = shldv_i16x32_intrinsic,
	width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi16, scalar = shldv_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 32-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_u16x32, slice_fn = shldv_u16_slice, intrinsic_fn = shldv_u16x32_intrinsic,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi16, scalar = shldv_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 32-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_i16x32, slice_fn = shrdv_i16_slice, intrinsic_fn = shrdv_i16x32_intrinsic,
	width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi16, scalar = shrdv_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 32-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_u16x32, slice_fn = shrdv_u16_slice, intrinsic_fn = shrdv_u16x32_intrinsic,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi16, scalar = shrdv_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 32-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_i32x16, slice_fn = shldv_i32_slice, intrinsic_fn = shldv_i32x16_intrinsic,
	width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi32, scalar = shldv_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_u32x16, slice_fn = shldv_u32_slice, intrinsic_fn = shldv_u32x16_intrinsic,
	width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi32, scalar = shldv_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_i32x16, slice_fn = shrdv_i32_slice, intrinsic_fn = shrdv_i32x16_intrinsic,
	width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi32, scalar = shrdv_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_u32x16, slice_fn = shrdv_u32_slice, intrinsic_fn = shrdv_u32x16_intrinsic,
	width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi32, scalar = shrdv_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 16-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_i64x8, slice_fn = shldv_i64_slice, intrinsic_fn = shldv_i64x8_intrinsic,
	width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi64, scalar = shldv_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldv_u64x8, slice_fn = shldv_u64_slice, intrinsic_fn = shldv_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldv_epi64, scalar = shldv_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `c` (`vpshldvq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_i64x8, slice_fn = shrdv_i64_slice, intrinsic_fn = shrdv_i64x8_intrinsic,
	width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi64, scalar = shrdv_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

simd_ternop! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdv_u64x8, slice_fn = shrdv_u64_slice, intrinsic_fn = shrdv_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdv_epi64, scalar = shrdv_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `c` (`vpshrdvq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdv(a[i], b[i], c[i])`. 8-wide chunks, software scalar rem.",
}

// Merge/zero-masked funnel shift: `a` doubles as the merge fallback, same
// shape as FMA (`_mm512_mask_shldv_epi16(a, k, b, c)`, no separate `src`).

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_i16x32_merge_masked, zero_fn = shldv_i16x32_zero_masked,
	merge_intrinsic_fn = mask_shldv_epi16_intrinsic, zero_intrinsic_fn = maskz_shldv_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi16, zero_intrinsic = _mm512_maskz_shldv_epi16,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvw`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvw`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_u16x32_merge_masked, zero_fn = shldv_u16x32_zero_masked,
	merge_intrinsic_fn = mask_shldv_epu16_intrinsic, zero_intrinsic_fn = maskz_shldv_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi16, zero_intrinsic = _mm512_maskz_shldv_epi16,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvw`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvw`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_i16x32_merge_masked, zero_fn = shrdv_i16x32_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epi16_intrinsic, zero_intrinsic_fn = maskz_shrdv_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi16, zero_intrinsic = _mm512_maskz_shrdv_epi16,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvw`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvw`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_u16x32_merge_masked, zero_fn = shrdv_u16x32_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epu16_intrinsic, zero_intrinsic_fn = maskz_shrdv_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi16, zero_intrinsic = _mm512_maskz_shrdv_epi16,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvw`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvw`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_i32x16_merge_masked, zero_fn = shldv_i32x16_zero_masked,
	merge_intrinsic_fn = mask_shldv_epi32_intrinsic, zero_intrinsic_fn = maskz_shldv_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi32, zero_intrinsic = _mm512_maskz_shldv_epi32,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvd`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvd`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_u32x16_merge_masked, zero_fn = shldv_u32x16_zero_masked,
	merge_intrinsic_fn = mask_shldv_epu32_intrinsic, zero_intrinsic_fn = maskz_shldv_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi32, zero_intrinsic = _mm512_maskz_shldv_epi32,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvd`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvd`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_i32x16_merge_masked, zero_fn = shrdv_i32x16_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epi32_intrinsic, zero_intrinsic_fn = maskz_shrdv_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi32, zero_intrinsic = _mm512_maskz_shrdv_epi32,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvd`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvd`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_u32x16_merge_masked, zero_fn = shrdv_u32x16_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epu32_intrinsic, zero_intrinsic_fn = maskz_shrdv_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi32, zero_intrinsic = _mm512_maskz_shrdv_epi32,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvd`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvd`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_i64x8_merge_masked, zero_fn = shldv_i64x8_zero_masked,
	merge_intrinsic_fn = mask_shldv_epi64_intrinsic, zero_intrinsic_fn = maskz_shldv_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi64, zero_intrinsic = _mm512_maskz_shldv_epi64,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvq`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvq`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldv_u64x8_merge_masked, zero_fn = shldv_u64x8_zero_masked,
	merge_intrinsic_fn = mask_shldv_epu64_intrinsic, zero_intrinsic_fn = maskz_shldv_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldv_epi64, zero_intrinsic = _mm512_maskz_shldv_epi64,
	merge_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else `a` (`vpshldvq`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `c` where `mask` bit is set, else zero (`vpshldvq`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_i64x8_merge_masked, zero_fn = shrdv_i64x8_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epi64_intrinsic, zero_intrinsic_fn = maskz_shrdv_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi64, zero_intrinsic = _mm512_maskz_shrdv_epi64,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvq`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvq`, zero-masked).",
}

simd_ternop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdv_u64x8_merge_masked, zero_fn = shrdv_u64x8_zero_masked,
	merge_intrinsic_fn = mask_shrdv_epu64_intrinsic, zero_intrinsic_fn = maskz_shrdv_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdv_epi64, zero_intrinsic = _mm512_maskz_shrdv_epi64,
	merge_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else `a` (`vpshrdvq`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `c` where `mask` bit is set, else zero (`vpshrdvq`, zero-masked).",
}

// shldi/shrdi: immediate funnel shift. Same bit math as shldv/shrdv, but a
// genuine immediate-form instruction (`_mm512_shldi_epi64::<IMM8>(a, b)`,
// not a runtime-vector `c`): `simd_binop_imm`, not `simd_ternop`.

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_i16x32, slice_fn = shldi_i16_slice, intrinsic_fn = shldi_i16x32_intrinsic,
	width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi16, scalar = shldi_i16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 32-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_u16x32, slice_fn = shldi_u16_slice, intrinsic_fn = shldi_u16x32_intrinsic,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi16, scalar = shldi_u16_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 32-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_i16x32, slice_fn = shrdi_i16_slice, intrinsic_fn = shrdi_i16x32_intrinsic,
	width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi16, scalar = shrdi_i16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 32-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_u16x32, slice_fn = shrdi_u16_slice, intrinsic_fn = shrdi_u16x32_intrinsic,
	width = 32, elem = u16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi16, scalar = shrdi_u16_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdw`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 32-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_i32x16, slice_fn = shldi_i32_slice, intrinsic_fn = shldi_i32x16_intrinsic,
	width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi32, scalar = shldi_i32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_u32x16, slice_fn = shldi_u32_slice, intrinsic_fn = shldi_u32x16_intrinsic,
	width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi32, scalar = shldi_u32_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_i32x16, slice_fn = shrdi_i32_slice, intrinsic_fn = shrdi_i32x16_intrinsic,
	width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi32, scalar = shrdi_i32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_u32x16, slice_fn = shrdi_u32_slice, intrinsic_fn = shrdi_u32x16_intrinsic,
	width = 16, elem = u32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi32, scalar = shrdi_u32_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdd`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 16-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_i64x8, slice_fn = shldi_i64_slice, intrinsic_fn = shldi_i64x8_intrinsic,
	width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi64, scalar = shldi_i64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shldi_u64x8, slice_fn = shldi_u64_slice, intrinsic_fn = shldi_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shldi_epi64, scalar = shldi_u64_scalar,
	fixed_doc = "Funnel shift left `a:b` by `IMM8` (`vpshldq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shldi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_i64x8, slice_fn = shrdi_i64_slice, intrinsic_fn = shrdi_i64x8_intrinsic,
	width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi64, scalar = shrdi_i64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

simd_binop_imm! {
	token = Avx512Vbmi2, vis = pub, target_feature = "avx512vbmi2",
	fixed_fn = shrdi_u64x8, slice_fn = shrdi_u64_slice, intrinsic_fn = shrdi_u64x8_intrinsic,
	width = 8, elem = u64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_shrdi_epi64, scalar = shrdi_u64_scalar,
	fixed_doc = "Funnel shift right `b:a` by `IMM8` (`vpshrdq`, 512-bit). See module docs for the exact bit math.",
	slice_doc = "`out[i] = shrdi(a[i], b[i], IMM8)`. 8-wide chunks, software scalar rem.",
}

// shldi/shrdi merge/zero-masked: `_mm512_mask_shldi_epi64(src, k, a, b)` has
// a genuinely separate `src` (not fused with `a` like shldv's masked form) -
// `simd_binop_imm_masked`, not `simd_ternop_masked`.

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_i16x32_merge_masked, zero_fn = shldi_i16x32_zero_masked,
	merge_intrinsic_fn = mask_shldi_epi16_intrinsic, zero_intrinsic_fn = maskz_shldi_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi16, zero_intrinsic = _mm512_maskz_shldi_epi16,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldw`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldw`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_u16x32_merge_masked, zero_fn = shldi_u16x32_zero_masked,
	merge_intrinsic_fn = mask_shldi_epu16_intrinsic, zero_intrinsic_fn = maskz_shldi_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi16, zero_intrinsic = _mm512_maskz_shldi_epi16,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldw`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldw`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_i16x32_merge_masked, zero_fn = shrdi_i16x32_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epi16_intrinsic, zero_intrinsic_fn = maskz_shrdi_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi16, zero_intrinsic = _mm512_maskz_shrdi_epi16,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdw`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdw`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_u16x32_merge_masked, zero_fn = shrdi_u16x32_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epu16_intrinsic, zero_intrinsic_fn = maskz_shrdi_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi16, zero_intrinsic = _mm512_maskz_shrdi_epi16,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdw`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdw`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_i32x16_merge_masked, zero_fn = shldi_i32x16_zero_masked,
	merge_intrinsic_fn = mask_shldi_epi32_intrinsic, zero_intrinsic_fn = maskz_shldi_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi32, zero_intrinsic = _mm512_maskz_shldi_epi32,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldd`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldd`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_u32x16_merge_masked, zero_fn = shldi_u32x16_zero_masked,
	merge_intrinsic_fn = mask_shldi_epu32_intrinsic, zero_intrinsic_fn = maskz_shldi_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi32, zero_intrinsic = _mm512_maskz_shldi_epi32,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldd`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldd`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_i32x16_merge_masked, zero_fn = shrdi_i32x16_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epi32_intrinsic, zero_intrinsic_fn = maskz_shrdi_epi32_intrinsic,
	width = 16, elem = i32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi32, zero_intrinsic = _mm512_maskz_shrdi_epi32,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdd`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdd`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_u32x16_merge_masked, zero_fn = shrdi_u32x16_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epu32_intrinsic, zero_intrinsic_fn = maskz_shrdi_epu32_intrinsic,
	width = 16, elem = u32, vec = __m512i, mask = u16,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi32, zero_intrinsic = _mm512_maskz_shrdi_epi32,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdd`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdd`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_i64x8_merge_masked, zero_fn = shldi_i64x8_zero_masked,
	merge_intrinsic_fn = mask_shldi_epi64_intrinsic, zero_intrinsic_fn = maskz_shldi_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi64, zero_intrinsic = _mm512_maskz_shldi_epi64,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldq`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldq`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shldi_u64x8_merge_masked, zero_fn = shldi_u64x8_zero_masked,
	merge_intrinsic_fn = mask_shldi_epu64_intrinsic, zero_intrinsic_fn = maskz_shldi_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shldi_epi64, zero_intrinsic = _mm512_maskz_shldi_epi64,
	merge_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else `src` (`vpshldq`, merge-masked).",
	zero_doc = "Funnel shift left `a:b` by `IMM8` where `mask` bit is set, else zero (`vpshldq`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_i64x8_merge_masked, zero_fn = shrdi_i64x8_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epi64_intrinsic, zero_intrinsic_fn = maskz_shrdi_epi64_intrinsic,
	width = 8, elem = i64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi64, zero_intrinsic = _mm512_maskz_shrdi_epi64,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdq`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdq`, zero-masked).",
}

simd_binop_imm_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = shrdi_u64x8_merge_masked, zero_fn = shrdi_u64x8_zero_masked,
	merge_intrinsic_fn = mask_shrdi_epu64_intrinsic, zero_intrinsic_fn = maskz_shrdi_epu64_intrinsic,
	width = 8, elem = u64, vec = __m512i, mask = u8,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_shrdi_epi64, zero_intrinsic = _mm512_maskz_shrdi_epi64,
	merge_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else `src` (`vpshrdq`, merge-masked).",
	zero_doc = "Funnel shift right `b:a` by `IMM8` where `mask` bit is set, else zero (`vpshrdq`, zero-masked).",
}

// compress/expand: i8/u8/i16/u16 only (epi32/64/ps/pd live in avx512f.rs,
// AVX-512F-only). No unmasked base op exists in the ISA.

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = compress_i8x64_merge_masked, zero_fn = compress_i8x64_zero_masked,
	merge_intrinsic_fn = mask_compress_epi8_intrinsic, zero_intrinsic_fn = maskz_compress_epi8_intrinsic,
	width = 64, elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi8, zero_intrinsic = _mm512_maskz_compress_epi8,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressb`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = compress_u8x64_merge_masked, zero_fn = compress_u8x64_zero_masked,
	merge_intrinsic_fn = mask_compress_epu8_intrinsic, zero_intrinsic_fn = maskz_compress_epu8_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi8, zero_intrinsic = _mm512_maskz_compress_epi8,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressb`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = compress_i16x32_merge_masked, zero_fn = compress_i16x32_zero_masked,
	merge_intrinsic_fn = mask_compress_epi16_intrinsic, zero_intrinsic_fn = maskz_compress_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi16, zero_intrinsic = _mm512_maskz_compress_epi16,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressw`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressw`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = compress_u16x32_merge_masked, zero_fn = compress_u16x32_zero_masked,
	merge_intrinsic_fn = mask_compress_epu16_intrinsic, zero_intrinsic_fn = maskz_compress_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_compress_epi16, zero_intrinsic = _mm512_maskz_compress_epi16,
	merge_doc = "Left-pack lanes where `mask` bit is set to the front (increasing index), rest copied from `src` (`vpcompressw`, merge-masked).",
	zero_doc = "Left-pack lanes where `mask` bit is set to the front, rest zero (`vpcompressw`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expand_i8x64_merge_masked, zero_fn = expand_i8x64_zero_masked,
	merge_intrinsic_fn = mask_expand_epi8_intrinsic, zero_intrinsic_fn = maskz_expand_epi8_intrinsic,
	width = 64, elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi8, zero_intrinsic = _mm512_maskz_expand_epi8,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandb`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expand_u8x64_merge_masked, zero_fn = expand_u8x64_zero_masked,
	merge_intrinsic_fn = mask_expand_epu8_intrinsic, zero_intrinsic_fn = maskz_expand_epu8_intrinsic,
	width = 64, elem = u8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi8, zero_intrinsic = _mm512_maskz_expand_epi8,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandb`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandb`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expand_i16x32_merge_masked, zero_fn = expand_i16x32_zero_masked,
	merge_intrinsic_fn = mask_expand_epi16_intrinsic, zero_intrinsic_fn = maskz_expand_epi16_intrinsic,
	width = 32, elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi16, zero_intrinsic = _mm512_maskz_expand_epi16,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandw`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandw`, zero-masked).",
}

simd_unop_masked! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expand_u16x32_merge_masked, zero_fn = expand_u16x32_zero_masked,
	merge_intrinsic_fn = mask_expand_epu16_intrinsic, zero_intrinsic_fn = maskz_expand_epu16_intrinsic,
	width = 32, elem = u16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expand_epi16, zero_intrinsic = _mm512_maskz_expand_epi16,
	merge_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else copy from `src` (`vpexpandw`, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next lane from `a` in increasing-index order, else zero (`vpexpandw`, zero-masked).",
}

// Memory forms of the same two ops: `compressstoreu` writes only the selected
// lanes (no merge/zero split: unselected lanes produce no store), while
// `expandloadu` reads only as many elements as the mask selects. Both are
// pointer-based in the ISA; the safe wrappers bound them with a popcount
// length assert.
simd_compressstoreu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	fixed_fn = compressstoreu_i8x64, intrinsic_fn = compressstoreu_i8x64_intrinsic,
	width = 64, elem = i8, ptr_elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi8,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressb`, store form).",
}

simd_compressstoreu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	fixed_fn = compressstoreu_u8x64, intrinsic_fn = compressstoreu_u8x64_intrinsic,
	width = 64, elem = u8, ptr_elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi8,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressb`, store form).",
}

simd_compressstoreu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	fixed_fn = compressstoreu_i16x32, intrinsic_fn = compressstoreu_i16x32_intrinsic,
	width = 32, elem = i16, ptr_elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi16,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressw`, store form).",
}

simd_compressstoreu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	fixed_fn = compressstoreu_u16x32, intrinsic_fn = compressstoreu_u16x32_intrinsic,
	width = 32, elem = u16, ptr_elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, intrinsic = _mm512_mask_compressstoreu_epi16,
	doc = "Left-pack the lanes whose `mask` bit is set and store them to the front of `out` (`vpcompressw`, store form).",
}

simd_expandloadu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expandloadu_i8x64_merge_masked, zero_fn = expandloadu_i8x64_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_i8_intrinsic, zero_intrinsic_fn = maskz_expandloadu_i8_intrinsic,
	width = 64, elem = i8, ptr_elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi8, zero_intrinsic = _mm512_maskz_expandloadu_epi8,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandb`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandb`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expandloadu_u8x64_merge_masked, zero_fn = expandloadu_u8x64_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_u8_intrinsic, zero_intrinsic_fn = maskz_expandloadu_u8_intrinsic,
	width = 64, elem = u8, ptr_elem = i8, vec = __m512i, mask = u64,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi8, zero_intrinsic = _mm512_maskz_expandloadu_epi8,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandb`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandb`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expandloadu_i16x32_merge_masked, zero_fn = expandloadu_i16x32_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_i16_intrinsic, zero_intrinsic_fn = maskz_expandloadu_i16_intrinsic,
	width = 32, elem = i16, ptr_elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi16, zero_intrinsic = _mm512_maskz_expandloadu_epi16,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandw`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandw`, load form, zero-masked).",
}

simd_expandloadu! {
	token = Avx512Vbmi2, target_feature = "avx512vbmi2",
	merge_fn = expandloadu_u16x32_merge_masked, zero_fn = expandloadu_u16x32_zero_masked,
	merge_intrinsic_fn = mask_expandloadu_u16_intrinsic, zero_intrinsic_fn = maskz_expandloadu_u16_intrinsic,
	width = 32, elem = u16, ptr_elem = i16, vec = __m512i, mask = u32,
	loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_expandloadu_epi16, zero_intrinsic = _mm512_maskz_expandloadu_epi16,
	merge_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else copy from `src` (`vpexpandw`, load form, merge-masked).",
	zero_doc = "Where `mask` bit is set, consume the next element from `mem` in increasing-index order, else zero (`vpexpandw`, load form, zero-masked).",
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512vbmi2.rs"]
mod tests;
