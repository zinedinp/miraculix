//! Ops that mixed-precision and byte/bitset kernels need, grouped by token
//! rather than by Intel extension: f32 <-> f64 conversion, byte/word/dword
//! interleave, byte shuffle at 256/512 bits, zero-extend, runtime-count
//! shifts, partial loads, cross-lane fixups and AVX-512 byte masks.
//!
//! The methods live on the usual tokens ([`Sse2`], [`Sse41`], [`Avx`],
//! [`Avx2`], [`Avx512f`], [`Avx512Bw`]); this file only holds the code.
//!
//! **Lane order.** `unpack*`, `shuffle_i8x32/x64` and `permute2x128` follow
//! the Intel semantics: unpack and byte shuffle work **inside each 128-bit
//! lane**, so 256/512-bit results are not a plain concatenation of the 128-bit
//! results. Put the lanes back in order with [`Avx2::permute2x128_i64x4`] or
//! [`Avx512f::permutex2var_u64x8`].

use core::arch::x86_64::*;

use super::avx::avx::Avx;
use super::avx::avx2::Avx2;
use super::avx512::avx512bw::Avx512Bw;
use super::avx512::avx512f::Avx512f;
use super::macros::{
    simd_binop_fixed, simd_binop_imm_fixed, simd_broadcast, simd_cvt_narrow, simd_cvt_widen,
};
use super::sse::sse2::Sse2;
use super::sse::sse41::Sse41;

// f32 <-> f64

simd_cvt_widen! {
    token = Sse2, target_feature = "sse2",
    fixed_fn = cvt_f32x4_to_f64x2, intrinsic_fn = cvt_f32x4_to_f64x2_intrinsic,
    in_width = 4, out_width = 2,
    in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
    out_elem = f64, out_vec = __m128d, out_storeu = _mm_storeu_pd,
    intrinsic = _mm_cvtps_pd,
    fixed_doc = "Widens the **low two** f32 lanes to f64 (`cvtps2pd`); lanes 2 and 3 are ignored.",
}
simd_cvt_narrow! {
    token = Sse2, target_feature = "sse2",
    fixed_fn = cvt_f64x2_to_f32x4, intrinsic_fn = cvt_f64x2_to_f32x4_intrinsic,
    in_width = 2, out_width = 4,
    in_elem = f64, in_vec = __m128d, in_loadu = _mm_loadu_pd,
    out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
    intrinsic = _mm_cvtpd_ps,
    fixed_doc = "Narrows two f64 lanes to f32 with round-to-nearest (`cvtpd2ps`); lanes 2 and 3 are zero.",
}
simd_cvt_widen! {
    token = Avx, target_feature = "avx",
    fixed_fn = cvt_f32x4_to_f64x4, intrinsic_fn = cvt_f32x4_to_f64x4_intrinsic,
    in_width = 4, out_width = 4,
    in_elem = f32, in_vec = __m128, in_loadu = _mm_loadu_ps,
    out_elem = f64, out_vec = __m256d, out_storeu = _mm256_storeu_pd,
    intrinsic = _mm256_cvtps_pd,
    fixed_doc = "Widens four f32 lanes to f64 (`vcvtps2pd`, 256-bit result).",
}
simd_cvt_narrow! {
    token = Avx, target_feature = "avx",
    fixed_fn = cvt_f64x4_to_f32x4, intrinsic_fn = cvt_f64x4_to_f32x4_intrinsic,
    in_width = 4, out_width = 4,
    in_elem = f64, in_vec = __m256d, in_loadu = _mm256_loadu_pd,
    out_elem = f32, out_vec = __m128, out_storeu = _mm_storeu_ps,
    intrinsic = _mm256_cvtpd_ps,
    fixed_doc = "Narrows four f64 lanes to f32 with round-to-nearest (`vcvtpd2ps`).",
}
simd_cvt_widen! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = cvt_f32x8_to_f64x8, intrinsic_fn = cvt_f32x8_to_f64x8_intrinsic,
    in_width = 8, out_width = 8,
    in_elem = f32, in_vec = __m256, in_loadu = _mm256_loadu_ps,
    out_elem = f64, out_vec = __m512d, out_storeu = _mm512_storeu_pd,
    intrinsic = _mm512_cvtps_pd,
    fixed_doc = "Widens eight f32 lanes to f64 (`vcvtps2pd`, 512-bit result).",
}
simd_cvt_narrow! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = cvt_f64x8_to_f32x8, intrinsic_fn = cvt_f64x8_to_f32x8_intrinsic,
    in_width = 8, out_width = 8,
    in_elem = f64, in_vec = __m512d, in_loadu = _mm512_loadu_pd,
    out_elem = f32, out_vec = __m256, out_storeu = _mm256_storeu_ps,
    intrinsic = _mm512_cvtpd_ps,
    fixed_doc = "Narrows eight f64 lanes to f32 with round-to-nearest (`vcvtpd2ps`).",
}

// Interleave (unpack)

simd_binop_fixed! {
    token = Sse2, target_feature = "sse2",
    fixed_fn = unpacklo_u8x16, intrinsic_fn = unpacklo_epi8_fixed,
    width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
    intrinsic = _mm_unpacklo_epi8,
    fixed_doc = "Interleaves the low 8 bytes of `a`/`b`: `[a0, b0, a1, b1, ...]` (`punpcklbw`).",
}
simd_binop_fixed! {
    token = Sse2, target_feature = "sse2",
    fixed_fn = unpackhi_u8x16, intrinsic_fn = unpackhi_epi8_fixed,
    width = 16, elem = u8, vec = __m128i, loadu = _mm_loadu_si128, storeu = _mm_storeu_si128,
    intrinsic = _mm_unpackhi_epi8,
    fixed_doc = "Interleaves the high 8 bytes of `a`/`b`: `[a8, b8, a9, b9, ...]` (`punpckhbw`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpacklo_u8x32, intrinsic_fn = unpacklo_epi8_256,
    width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpacklo_epi8,
    fixed_doc = "Interleaves the low 8 bytes of **each 128-bit lane** of `a`/`b` (`vpunpcklbw`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpackhi_u8x32, intrinsic_fn = unpackhi_epi8_256,
    width = 32, elem = u8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpackhi_epi8,
    fixed_doc = "Interleaves the high 8 bytes of **each 128-bit lane** of `a`/`b` (`vpunpckhbw`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpacklo_i16x16, intrinsic_fn = unpacklo_epi16_256,
    width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpacklo_epi16,
    fixed_doc = "Interleaves the low 4 words of **each 128-bit lane** of `a`/`b` (`vpunpcklwd`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpackhi_i16x16, intrinsic_fn = unpackhi_epi16_256,
    width = 16, elem = i16, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpackhi_epi16,
    fixed_doc = "Interleaves the high 4 words of **each 128-bit lane** of `a`/`b` (`vpunpckhwd`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpacklo_i32x8, intrinsic_fn = unpacklo_epi32_256,
    width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpacklo_epi32,
    fixed_doc = "Interleaves the low 2 dwords of **each 128-bit lane** of `a`/`b` (`vpunpckldq`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpackhi_i32x8, intrinsic_fn = unpackhi_epi32_256,
    width = 8, elem = i32, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpackhi_epi32,
    fixed_doc = "Interleaves the high 2 dwords of **each 128-bit lane** of `a`/`b` (`vpunpckhdq`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpacklo_i64x4, intrinsic_fn = unpacklo_epi64_256,
    width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpacklo_epi64,
    fixed_doc = "Per 128-bit lane `[a0, b0]`, `[a2, b2]` (`vpunpcklqdq`).",
}
simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = unpackhi_i64x4, intrinsic_fn = unpackhi_epi64_256,
    width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_unpackhi_epi64,
    fixed_doc = "Per 128-bit lane `[a1, b1]`, `[a3, b3]` (`vpunpckhqdq`).",
}
simd_binop_fixed! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = unpacklo_u8x64, intrinsic_fn = unpacklo_epi8_512,
    width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpacklo_epi8,
    fixed_doc = "Interleaves the low 8 bytes of **each 128-bit lane** of `a`/`b` (`vpunpcklbw`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = unpackhi_u8x64, intrinsic_fn = unpackhi_epi8_512,
    width = 64, elem = u8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpackhi_epi8,
    fixed_doc = "Interleaves the high 8 bytes of **each 128-bit lane** of `a`/`b` (`vpunpckhbw`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = unpacklo_i16x32, intrinsic_fn = unpacklo_epi16_512,
    width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpacklo_epi16,
    fixed_doc = "Interleaves the low 4 words of **each 128-bit lane** of `a`/`b` (`vpunpcklwd`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = unpackhi_i16x32, intrinsic_fn = unpackhi_epi16_512,
    width = 32, elem = i16, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpackhi_epi16,
    fixed_doc = "Interleaves the high 4 words of **each 128-bit lane** of `a`/`b` (`vpunpckhwd`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = unpacklo_i32x16, intrinsic_fn = unpacklo_epi32_512,
    width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpacklo_epi32,
    fixed_doc = "Interleaves the low 2 dwords of **each 128-bit lane** of `a`/`b` (`vpunpckldq`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = unpackhi_i32x16, intrinsic_fn = unpackhi_epi32_512,
    width = 16, elem = i32, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpackhi_epi32,
    fixed_doc = "Interleaves the high 2 dwords of **each 128-bit lane** of `a`/`b` (`vpunpckhdq`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = unpacklo_i64x8, intrinsic_fn = unpacklo_epi64_512,
    width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpacklo_epi64,
    fixed_doc = "Per 128-bit lane `[a0, b0]`, `[a2, b2]`, ... (`vpunpcklqdq`, 512-bit).",
}
simd_binop_fixed! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = unpackhi_i64x8, intrinsic_fn = unpackhi_epi64_512,
    width = 8, elem = i64, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_unpackhi_epi64,
    fixed_doc = "Per 128-bit lane `[a1, b1]`, `[a3, b3]`, ... (`vpunpckhqdq`, 512-bit).",
}

// Byte shuffle at 256/512 bits (`b` holds the indices; in-lane, like `Ssse3::shuffle_i8x16`)

simd_binop_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = shuffle_i8x32, intrinsic_fn = shuffle_epi8_256,
    width = 32, elem = i8, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_shuffle_epi8,
    fixed_doc = "Byte LUT (`vpshufb`): `shuffle_i8x32(table, indices)`. **Each 128-bit lane indexes only its own 16 table bytes** (`indices[i] & 15` selects within the lane); an index with the high bit set gives 0.",
}
simd_binop_fixed! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = shuffle_i8x64, intrinsic_fn = shuffle_epi8_512,
    width = 64, elem = i8, vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_shuffle_epi8,
    fixed_doc = "Byte LUT (`vpshufb`): `shuffle_i8x64(table, indices)`. **Each 128-bit lane indexes only its own 16 table bytes**; an index with the high bit set gives 0. For a full 64-entry lookup use `Avx512Vbmi::permutexvar_u8x64`.",
}

// Zero-extend u8 -> u16

simd_cvt_widen! {
    token = Sse41, target_feature = "sse4.1",
    fixed_fn = cvtepu8_u16x8, intrinsic_fn = cvtepu8_epi16_128,
    in_width = 16, out_width = 8,
    in_elem = u8, in_vec = __m128i, in_loadu = _mm_loadu_si128,
    out_elem = u16, out_vec = __m128i, out_storeu = _mm_storeu_si128,
    intrinsic = _mm_cvtepu8_epi16,
    fixed_doc = "Zero-extends the **low 8** bytes to u16 (`pmovzxbw`); bytes 8..16 are ignored.",
}
simd_cvt_widen! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = cvtepu8_u16x16, intrinsic_fn = cvtepu8_epi16_256,
    in_width = 16, out_width = 16,
    in_elem = u8, in_vec = __m128i, in_loadu = _mm_loadu_si128,
    out_elem = u16, out_vec = __m256i, out_storeu = _mm256_storeu_si256,
    intrinsic = _mm256_cvtepu8_epi16,
    fixed_doc = "Zero-extends 16 bytes to u16 (`vpmovzxbw`), in order (no lane split).",
}
simd_cvt_widen! {
    token = Avx512Bw, target_feature = "avx512bw",
    fixed_fn = cvtepu8_u16x32, intrinsic_fn = cvtepu8_epi16_512,
    in_width = 32, out_width = 32,
    in_elem = u8, in_vec = __m256i, in_loadu = _mm256_loadu_si256,
    out_elem = u16, out_vec = __m512i, out_storeu = _mm512_storeu_si512,
    intrinsic = _mm512_cvtepu8_epi16,
    fixed_doc = "Zero-extends 32 bytes to u16 (`vpmovzxbw`), in order (no lane split).",
}

// Cross-lane fixups and 128-bit broadcasts

simd_binop_imm_fixed! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = permute2x128_i64x4, intrinsic_fn = permute2x128_si256_fixed,
    width = 4, elem = i64, vec = __m256i, loadu = _mm256_loadu_si256, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_permute2x128_si256,
    fixed_doc = "Selects one 128-bit half from `a` and one from `b` by `IMM8` (`vperm2i128`). Low 2 bits pick the source for the low half (0/1 = `a` lo/hi, 2/3 = `b` lo/hi), bits 4-5 for the high half; bits 3 and 7 zero a half.",
}
simd_broadcast! {
    token = Avx2, target_feature = "avx2",
    fixed_fn = broadcast_u8x16_to_x32, intrinsic_fn = broadcast_u8x16_to_x32_intrinsic,
    narrow_width = 16, wide_width = 32, elem = u8,
    narrow_vec = __m128i, wide_vec = __m256i,
    narrow_loadu = _mm_loadu_si128, storeu = _mm256_storeu_si256,
    intrinsic = _mm256_broadcastsi128_si256,
    fixed_doc = "Copies the 128-bit input into both halves (`vbroadcasti128`).",
}
simd_broadcast! {
    token = Avx512f, target_feature = "avx512f",
    fixed_fn = broadcast_u8x16_to_x64, intrinsic_fn = broadcast_u8x16_to_x64_intrinsic,
    narrow_width = 16, wide_width = 64, elem = u8,
    narrow_vec = __m128i, wide_vec = __m512i,
    narrow_loadu = _mm_loadu_si128, storeu = _mm512_storeu_si512,
    intrinsic = _mm512_broadcast_i32x4,
    fixed_doc = "Copies the 128-bit input into all four 128-bit lanes (`vbroadcasti32x4`).",
}

impl Avx512f {
    /// Two-source 64-bit permute (`vpermt2q`): `out[i] = (idx[i] & 8 ? b : a)[idx[i] & 7]`.
    #[inline]
    pub fn permutex2var_u64x8(self, a: [u64; 8], idx: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        unsafe { permutex2var_epi64_512(&a, &idx, &b) }
    }
}

/// # Safety
/// Caller proved AVX-512F via [`Avx512f`].
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn permutex2var_epi64_512(a: &[u64; 8], idx: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
    unsafe {
        let va = _mm512_loadu_si512(a.as_ptr().cast());
        let vi = _mm512_loadu_si512(idx.as_ptr().cast());
        let vb = _mm512_loadu_si512(b.as_ptr().cast());
        let vr = _mm512_permutex2var_epi64(va, vi, vb);
        let mut out = [0u64; 8];
        _mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
        out
    }
}

// Runtime-count shifts (one count for all lanes; count >= bit width gives 0)

macro_rules! shift_var {
    (
        $Token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident,
        $width:literal, $vec:ty, $loadu:path, $storeu:path, $intrinsic:path, $doc:literal
    ) => {
        impl $Token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [u16; $width], count: u32) -> [u16; $width] {
                unsafe { $intrinsic_fn(&a, count) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[u16; $width], count: u32) -> [u16; $width] {
            unsafe {
                let va: $vec = $loadu(a.as_ptr().cast());
                let vc = _mm_cvtsi32_si128(count as i32);
                let vr = $intrinsic(va, vc);
                let mut out = [0u16; $width];
                $storeu(out.as_mut_ptr().cast(), vr);
                out
            }
        }
    };
}

shift_var!(
    Sse2,
    "sse2",
    sll_u16x8,
    sll_epi16_128,
    8,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_sll_epi16,
    "`a << count` per u16 lane, count from a runtime value (`psllw`). `count >= 16` gives 0."
);
shift_var!(
    Sse2,
    "sse2",
    srl_u16x8,
    srl_epi16_128,
    8,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_srl_epi16,
    "`a >> count` logical per u16 lane, runtime count (`psrlw`). `count >= 16` gives 0."
);
shift_var!(
    Avx2,
    "avx2",
    sll_u16x16,
    sll_epi16_256,
    16,
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_sll_epi16,
    "`a << count` per u16 lane, runtime count (`vpsllw`). `count >= 16` gives 0."
);
shift_var!(
    Avx2,
    "avx2",
    srl_u16x16,
    srl_epi16_256,
    16,
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_srl_epi16,
    "`a >> count` logical per u16 lane, runtime count (`vpsrlw`). `count >= 16` gives 0."
);
shift_var!(
    Avx512Bw,
    "avx512bw",
    sll_u16x32,
    sll_epi16_512,
    32,
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_sll_epi16,
    "`a << count` per u16 lane, runtime count (`vpsllw`). `count >= 16` gives 0."
);
shift_var!(
    Avx512Bw,
    "avx512bw",
    srl_u16x32,
    srl_epi16_512,
    32,
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_srl_epi16,
    "`a >> count` logical per u16 lane, runtime count (`vpsrlw`). `count >= 16` gives 0."
);

// Partial loads / scalar-to-vector (upper lanes zeroed)

impl Sse2 {
    /// Loads 8 bytes into the low half, upper 8 bytes zero (`movq`).
    #[inline]
    pub fn loadl_u8x8(self, a: [u8; 8]) -> [u8; 16] {
        unsafe { loadl_epi64_fixed(&a) }
    }

    /// Puts `x` in dword 0, the rest zero (`movd`).
    #[inline]
    pub fn cvtsi32_u32x4(self, x: u32) -> [u32; 4] {
        unsafe { cvtsi32_si128_fixed(x) }
    }

    /// Puts `x` in qword 0, the upper qword zero (`movq`).
    #[inline]
    pub fn cvtsi64_u64x2(self, x: u64) -> [u64; 2] {
        unsafe { cvtsi64_si128_fixed(x) }
    }
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn loadl_epi64_fixed(a: &[u8; 8]) -> [u8; 16] {
    unsafe {
        let v = _mm_loadl_epi64(a.as_ptr().cast());
        let mut out = [0u8; 16];
        _mm_storeu_si128(out.as_mut_ptr().cast(), v);
        out
    }
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn cvtsi32_si128_fixed(x: u32) -> [u32; 4] {
    unsafe {
        let v = _mm_cvtsi32_si128(x as i32);
        let mut out = [0u32; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), v);
        out
    }
}

/// # Safety
/// Caller proved SSE2 via [`Sse2`].
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn cvtsi64_si128_fixed(x: u64) -> [u64; 2] {
    unsafe {
        let v = _mm_cvtsi64_si128(x as i64);
        let mut out = [0u64; 2];
        _mm_storeu_si128(out.as_mut_ptr().cast(), v);
        out
    }
}

// AVX-512BW byte masks as u64

impl Avx512Bw {
    /// Per-byte bit test: bit `i` of the result is `(a[i] & b[i]) != 0`
    /// (`vptestmb`). With `b = [1 << k; 64]` this is bit `k` of every byte
    /// as a 64-bit mask. (`movepi8_mask` only reads the sign bit.)
    #[inline]
    pub fn test_u8x64_mask(self, a: [u8; 64], b: [u8; 64]) -> u64 {
        unsafe { test_epi8_mask_512(&a, &b) }
    }

    /// Mask select (`vpblendmb`): `out[i] = if k >> i & 1 != 0 { b[i] } else { a[i] }`.
    /// Counterpart of `select_i8x64` for a `u64` mask instead of a vector mask.
    #[inline]
    pub fn mask_blend_u8x64(self, k: u64, a: [u8; 64], b: [u8; 64]) -> [u8; 64] {
        unsafe { mask_blend_epi8_512(k, &a, &b) }
    }
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn test_epi8_mask_512(a: &[u8; 64], b: &[u8; 64]) -> u64 {
    unsafe {
        let va = _mm512_loadu_si512(a.as_ptr().cast());
        let vb = _mm512_loadu_si512(b.as_ptr().cast());
        _mm512_test_epi8_mask(va, vb)
    }
}

/// # Safety
/// Caller proved AVX-512BW via [`Avx512Bw`].
#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn mask_blend_epi8_512(k: u64, a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
    unsafe {
        let va = _mm512_loadu_si512(a.as_ptr().cast());
        let vb = _mm512_loadu_si512(b.as_ptr().cast());
        let vr = _mm512_mask_blend_epi8(k, va, vb);
        let mut out = [0u8; 64];
        _mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
        out
    }
}

#[cfg(test)]
#[path = "../test/ops/extra.rs"]
mod tests;
