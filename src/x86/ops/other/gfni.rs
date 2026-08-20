//! GFNI (Ice Lake, 2019): byte-wise GF(2^8), poly `0x11B` (`"gfni"`). Tokens: [`Gfni`] (128/256),
//! [`Gfni512`] (512, needs `"gfni,avx512f"`). `gf2p8mul` via [`simd_binop`]; affine forms hand-written.
//! 128-bit affine requires `"avx"` (VEX form; legacy encoding needs 16-byte-aligned mem).

use core::arch::x86_64::{
	__m128i, __m256i, __m512i, _mm256_gf2p8affine_epi64_epi8, _mm256_gf2p8affineinv_epi64_epi8, _mm256_gf2p8mul_epi8,
	_mm256_loadu_si256, _mm256_storeu_si256, _mm512_gf2p8affine_epi64_epi8, _mm512_gf2p8affineinv_epi64_epi8,
	_mm512_gf2p8mul_epi8, _mm512_loadu_si512, _mm512_storeu_si512, _mm_gf2p8affine_epi64_epi8, _mm_gf2p8affineinv_epi64_epi8,
	_mm_gf2p8mul_epi8, _mm_loadu_si128, _mm_storeu_si128,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::simd_binop;

/// Proof token: GFNI available (128/256-bit ops). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Gfni(());

impl Gfni {
	/// `None` if the CPU (or the compile-time target) lacks GFNI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Gfni).then_some(Gfni(()))
	}
}

/// Proof token: GFNI *and* AVX-512F, both required for the 512-bit forms.
/// Real hardware has GFNI without AVX-512F (Alder Lake+ client), so this is
/// a genuine second check, not sugar. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Gfni512(());

impl Gfni512 {
	/// `None` unless the CPU has both GFNI and AVX-512F.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Gfni) && set.contains(Feature::Avx512f)).then_some(Gfni512(()))
	}
}

/// Per-byte GF(2^8) multiply, reduction polynomial `0x11B` (same field as
/// the AES S-box). Also the slice-remainder scalar fallback for
/// [`Gfni::gf2p8mul_epi8_slice_u8x16`]/`_u8x32` (see [`simd_binop`]).
fn gf2p8mul_byte(a: u8, b: u8) -> u8 {
	let mut tword: u32 = 0;
	for i in 0..8 {
		if (b >> i) & 1 != 0 {
			tword ^= (a as u32) << i;
		}
	}
	for i in (8..=14).rev() {
		if (tword >> i) & 1 != 0 {
			tword ^= 0x11B << (i - 8);
		}
	}
	tword as u8
}

simd_binop! {
	token = Gfni, vis = pub, target_feature = "gfni",
	fixed_fn = gf2p8mul_epi8_u8x16, slice_fn = gf2p8mul_epi8_slice_u8x16, intrinsic_fn = gf2p8mul_epi8_u8x16_intrinsic,
	width = 16, elem = u8, vec = __m128i,
	loadu = _mm_loadu_si128, storeu = _mm_storeu_si128, intrinsic = _mm_gf2p8mul_epi8,
	scalar = gf2p8mul_byte,
	fixed_doc = "Per-byte GF(2^8) multiply, GF(2^8) reduction `0x11B` (`vgf2p8mulb`, 128-bit).",
	slice_doc = "Slice form of [`Gfni::gf2p8mul_epi8_u8x16`].",
}
simd_binop! {
	token = Gfni, vis = pub, target_feature = "gfni,avx",
	fixed_fn = gf2p8mul_epi8_u8x32, slice_fn = gf2p8mul_epi8_slice_u8x32, intrinsic_fn = gf2p8mul_epi8_u8x32_intrinsic,
	width = 32, elem = u8, vec = __m256i,
	loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_gf2p8mul_epi8,
	scalar = gf2p8mul_byte,
	fixed_doc = "Per-byte GF(2^8) multiply, GF(2^8) reduction `0x11B` (`vgf2p8mulb`, 256-bit).",
	slice_doc = "Slice form of [`Gfni::gf2p8mul_epi8_u8x32`].",
}

impl Gfni512 {
	/// Per-byte GF(2^8) multiply, GF(2^8) reduction `0x11B` (`vgf2p8mulb`, 512-bit).
	#[inline]
	pub fn gf2p8mul_epi8_u8x64(self, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
		unsafe { gf2p8mul_epi8_u8x64_intrinsic(&a, &b) }
	}
}

/// # Safety
/// Caller proved GFNI + AVX-512F via [`Gfni512`].
#[inline]
#[target_feature(enable = "gfni,avx512f")]
unsafe fn gf2p8mul_epi8_u8x64_intrinsic(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
	unsafe {
		let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
		let vb: __m512i = _mm512_loadu_si512(b.as_ptr().cast());
		let vr = _mm512_gf2p8mul_epi8(va, vb);
		let mut out = [0u8; 64];
		_mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
		out
	}
}

/// `parity(x)`: XOR of all 8 bits of `x` (0 or 1).
#[cfg(test)]
fn parity(x: u8) -> u8 {
	x.count_ones() as u8 & 1
}

/// One output byte of `gf2p8affine_epi64_epi8`: `retbyte.bit[i] =
/// parity(matrix.byte[7-i] AND src) XOR imm8.bit[i]`.
#[cfg(test)]
fn gf2p8affine_byte(matrix: u64, src: u8, imm8: u8) -> u8 {
	let mut out = 0u8;
	for i in 0..8u32 {
		let matrix_byte = (matrix >> (8 * (7 - i))) as u8;
		let bit = parity(matrix_byte & src) ^ ((imm8 >> i) & 1);
		out |= bit << i;
	}
	out
}

/// One output byte of `gf2p8affineinv_epi64_epi8`: same as
/// [`gf2p8affine_byte`], but `src` is first replaced by its GF(2^8)
/// multiplicative inverse (`0 -> 0`, brute-force search: test-only, not a
/// hot path).
#[cfg(test)]
fn gf2p8affineinv_byte(matrix: u64, src: u8, imm8: u8) -> u8 {
	let inv = if src == 0 { 0 } else { (1..=255u16).map(|x| x as u8).find(|&x| gf2p8mul_byte(src, x) == 1).expect("GF(2^8) nonzero elements are invertible") };
	gf2p8affine_byte(matrix, inv, imm8)
}

/// Applies [`gf2p8affine_byte`]/[`gf2p8affineinv_byte`] to every byte of an
/// 8-byte (one 64-bit lane's worth of) input, given that lane's matrix qword.
#[cfg(test)]
fn affine_lane(matrix: u64, x: [u8; 8], imm8: u8, inv: bool) -> [u8; 8] {
	core::array::from_fn(|i| if inv { gf2p8affineinv_byte(matrix, x[i], imm8) } else { gf2p8affine_byte(matrix, x[i], imm8) })
}

macro_rules! gfni_affine {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, vec = $Vec:ty, loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const B: i32>(self, x: [u8; $width], matrix: [u8; $width]) -> [u8; $width] {
				unsafe { $intrinsic_fn::<B>(&x, &matrix) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const B: i32>(x: &[u8; $width], matrix: &[u8; $width]) -> [u8; $width] {
			unsafe {
				let vx: $Vec = $loadu(x.as_ptr().cast());
				let vm: $Vec = $loadu(matrix.as_ptr().cast());
				let vr = $intrinsic::<B>(vx, vm);
				let mut out = [0u8; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

gfni_affine! {
	token = Gfni, target_feature = "gfni,avx",
	fixed_fn = gf2p8affine_epi64_epi8_u8x16, intrinsic_fn = gf2p8affine_epi64_epi8_u8x16_intrinsic,
	width = 16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128, intrinsic = _mm_gf2p8affine_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform per byte, `B`-bit-per-lane matrix `matrix` (`vgf2p8affineqb`, 128-bit).",
}
gfni_affine! {
	token = Gfni, target_feature = "gfni,avx",
	fixed_fn = gf2p8affine_epi64_epi8_u8x32, intrinsic_fn = gf2p8affine_epi64_epi8_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_gf2p8affine_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform per byte, `B`-bit-per-lane matrix `matrix` (`vgf2p8affineqb`, 256-bit).",
}
gfni_affine! {
	token = Gfni512, target_feature = "gfni,avx512f",
	fixed_fn = gf2p8affine_epi64_epi8_u8x64, intrinsic_fn = gf2p8affine_epi64_epi8_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_gf2p8affine_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform per byte, `B`-bit-per-lane matrix `matrix` (`vgf2p8affineqb`, 512-bit).",
}
gfni_affine! {
	token = Gfni, target_feature = "gfni,avx",
	fixed_fn = gf2p8affineinv_epi64_epi8_u8x16, intrinsic_fn = gf2p8affineinv_epi64_epi8_u8x16_intrinsic,
	width = 16, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128, intrinsic = _mm_gf2p8affineinv_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform of the multiplicative inverse per byte (`vgf2p8affineinvqb`, 128-bit).",
}
gfni_affine! {
	token = Gfni, target_feature = "gfni,avx",
	fixed_fn = gf2p8affineinv_epi64_epi8_u8x32, intrinsic_fn = gf2p8affineinv_epi64_epi8_u8x32_intrinsic,
	width = 32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256, intrinsic = _mm256_gf2p8affineinv_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform of the multiplicative inverse per byte (`vgf2p8affineinvqb`, 256-bit).",
}
gfni_affine! {
	token = Gfni512, target_feature = "gfni,avx512f",
	fixed_fn = gf2p8affineinv_epi64_epi8_u8x64, intrinsic_fn = gf2p8affineinv_epi64_epi8_u8x64_intrinsic,
	width = 64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512, intrinsic = _mm512_gf2p8affineinv_epi64_epi8,
	fixed_doc = "GF(2^8) affine transform of the multiplicative inverse per byte (`vgf2p8affineinvqb`, 512-bit).",
}

#[cfg(test)]
#[path = "../../test/ops/other/gfni.rs"]
mod tests;
