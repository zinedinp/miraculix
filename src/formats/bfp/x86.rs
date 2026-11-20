//! AVX-512 rows (x86-64-v4): encode one row per `zmm`, decode one row per `zmm`.

use super::{BFP_ROW, BfpElem, check_decode, check_encode, scalar};
use crate::x86::bundles::V4;
use core::arch::x86_64::*;

/// # Safety
/// AVX-512 F/BW/VL and AVX2 available. `src` points to 16 `f32` (`BF == false`)
/// or 16 `u16` (`BF == true`); `out` to `16 * (BITS + 1) / 8` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
unsafe fn row<const BITS: u32, const BF: bool>(src: *const u8, exp_out: *mut u8, out: *mut u8) {
    unsafe {
        let words = if BF {
            _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                src as *const __m256i,
            )))
        } else {
            _mm512_loadu_si512(src as *const _)
        };
        let exps = _mm512_and_si512(_mm512_srli_epi32::<23>(words), _mm512_set1_epi32(255));
        let max8 = _mm256_max_epu32(
            _mm512_castsi512_si256(exps),
            _mm512_extracti64x4_epi64::<1>(exps),
        );
        let mut max4 = _mm_max_epu32(
            _mm256_castsi256_si128(max8),
            _mm256_extracti128_si256::<1>(max8),
        );
        max4 = _mm_max_epu32(max4, _mm_shuffle_epi32::<0x4e>(max4));
        max4 = _mm_max_epu32(max4, _mm_shuffle_epi32::<0xb1>(max4));
        *exp_out = _mm_cvtsi128_si32(max4) as u8;
        let mut m = _mm512_or_si512(
            _mm512_and_si512(words, _mm512_set1_epi32(0x7F_FFFF)),
            _mm512_set1_epi32(0x80_0000),
        );
        m = _mm512_srlv_epi32(m, _mm512_sub_epi32(_mm512_broadcastd_epi32(max4), exps));
        let sh = _mm512_set1_epi32(24 - BITS as i32);
        let odd = _mm512_and_si512(_mm512_srlv_epi32(m, sh), _mm512_set1_epi32(1));
        let rnd = _mm512_add_epi32(odd, _mm512_set1_epi32((1 << (23 - BITS)) - 1));
        m = _mm512_srlv_epi32(_mm512_add_epi32(m, rnd), sh);
        m = _mm512_min_epu32(m, _mm512_set1_epi32((1 << BITS) - 1));
        let valid = _mm512_cmpneq_epi32_mask(exps, _mm512_setzero_si512())
            & _mm512_cmpneq_epi32_mask(m, _mm512_setzero_si512());
        let sign = _mm512_sllv_epi32(
            _mm512_srli_epi32::<31>(words),
            _mm512_set1_epi32(BITS as i32),
        );
        let codes = _mm512_maskz_mov_epi32(valid, _mm512_or_si512(m, sign));
        let bytes = _mm512_cvtepi32_epi8(codes);
        if BITS == 7 {
            _mm_storeu_si128(out as *mut __m128i, bytes);
        } else if BITS == 3 {
            let pairs = _mm_maddubs_epi16(bytes, _mm_set1_epi16(0x1001));
            _mm_storel_epi64(out as *mut __m128i, _mm_packus_epi16(pairs, pairs));
        } else {
            let pairs = _mm_maddubs_epi16(bytes, _mm_set1_epi16(0x0401));
            let quads = _mm_madd_epi16(pairs, _mm_set1_epi32(0x0010_0001));
            let shorts = _mm_packus_epi32(quads, quads);
            let v = _mm_cvtsi128_si32(_mm_packus_epi16(shorts, shorts));
            (out as *mut i32).write_unaligned(v);
        }
    }
}

/// The row's 16 codes, one per `u32` lane, element `k` of a byte at bit `k * (BITS + 1)`.
///
/// # Safety
/// AVX-512 F/BW/VL available; `d` points to `16 * (BITS + 1) / 8` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
unsafe fn codes<const BITS: u32>(d: *const u8) -> __m512i {
    unsafe {
        if BITS == 7 {
            _mm512_cvtepu8_epi32(_mm_loadu_si128(d as *const __m128i))
        } else if BITS == 3 {
            let b = _mm_loadl_epi64(d as *const __m128i);
            let rep = _mm_shuffle_epi8(
                b,
                _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
            );
            let sh = _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);
            _mm512_and_si512(
                _mm512_srlv_epi32(_mm512_cvtepu8_epi32(rep), sh),
                _mm512_set1_epi32(15),
            )
        } else {
            let b = _mm_cvtsi32_si128((d as *const i32).read_unaligned());
            let rep = _mm_shuffle_epi8(
                b,
                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
            );
            let sh = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);
            _mm512_and_si512(
                _mm512_srlv_epi32(_mm512_cvtepu8_epi32(rep), sh),
                _mm512_set1_epi32(3),
            )
        }
    }
}

/// A row's values as `f32` bits: `sign * magnitude * unit`, as [`scalar::decode_row`].
///
/// # Safety
/// As [`codes`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
unsafe fn values<const BITS: u32>(d: *const u8, unit: f32) -> __m512i {
    unsafe {
        let c = codes::<BITS>(d);
        let mag = _mm512_and_si512(c, _mm512_set1_epi32((1 << BITS) - 1));
        let sign = _mm512_slli_epi32::<31>(_mm512_srlv_epi32(c, _mm512_set1_epi32(BITS as i32)));
        let nonzero = _mm512_cmpneq_epi32_mask(mag, _mm512_setzero_si512());
        let v = _mm512_maskz_mul_ps(nonzero, _mm512_cvtepi32_ps(mag), _mm512_set1_ps(unit));
        // XOR, as scalar `-x` does.
        _mm512_xor_si512(_mm512_castps_si512(v), sign)
    }
}

/// A row as the Tensix unpacker's BF16 patterns in the low halves of the lanes, as
/// [`scalar::tensix_bf16`].
///
/// # Safety
/// As [`codes`], plus AVX-512 CD.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512cd,avx2")]
unsafe fn tensix<const BITS: u32>(d: *const u8, shared: u8) -> __m512i {
    unsafe {
        let c = codes::<BITS>(d);
        let sign = _mm512_srlv_epi32(c, _mm512_set1_epi32(BITS as i32));
        let mag8 = _mm512_and_si512(
            _mm512_sllv_epi32(c, _mm512_set1_epi32(8 - BITS as i32)),
            _mm512_set1_epi32(0xff),
        );
        let lz = _mm512_sub_epi32(_mm512_lzcnt_epi32(mag8), _mm512_set1_epi32(24));
        let mant = _mm512_and_si512(_mm512_sllv_epi32(mag8, lz), _mm512_set1_epi32(0x7e));
        let exp = _mm512_and_si512(
            _mm512_sub_epi32(_mm512_set1_epi32(shared as i32), lz),
            _mm512_set1_epi32(0xff),
        );
        let normal = _mm512_or_si512(
            _mm512_or_si512(_mm512_slli_epi32::<15>(sign), _mm512_slli_epi32::<7>(exp)),
            mant,
        );
        let zero = _mm512_cmpeq_epi32_mask(mag8, _mm512_setzero_si512());
        let neg_inf = _mm512_and_si512(
            _mm512_sub_epi32(_mm512_setzero_si512(), sign),
            _mm512_set1_epi32(0xff80),
        );
        _mm512_mask_blend_epi32(zero, normal, neg_inf)
    }
}

/// What a decode writes: [`F32`] and [`BF16`] values, [`TENSIX`] and [`TENSIX_F32`] the
/// unpacker's patterns.
const F32: u8 = 0;
const BF16: u8 = 1;
const TENSIX: u8 = 2;
const TENSIX_F32: u8 = 3;

/// Decodes `rows` rows, the loop inside the kernel so the row body inlines.
///
/// # Safety
/// AVX-512 F/BW/VL/CD available; `data` has `rows * 16 * (BITS + 1) / 8` bytes, `exps` `rows`
/// bytes and `out` room for `rows * 16` outputs of `OUT` (4 bytes for F32 kinds, 2 otherwise).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512cd,avx2")]
unsafe fn decode_rows<const BITS: u32, const OUT: u8>(
    elem: BfpElem,
    exps: *const u8,
    data: *const u8,
    out: *mut u8,
    rows: usize,
) {
    let rb = 16 * (BITS as usize + 1) / 8;
    let ob = if OUT == F32 || OUT == TENSIX_F32 {
        64
    } else {
        32
    };
    for r in 0..rows {
        unsafe {
            let shared = *exps.add(r);
            let d = data.add(r * rb);
            let o = out.add(r * ob);
            match OUT {
                F32 => {
                    _mm512_storeu_si512(o as *mut _, values::<BITS>(d, scalar::unit(elem, shared)))
                }
                BF16 => {
                    let v = _mm512_srli_epi32::<16>(values::<BITS>(d, scalar::unit(elem, shared)));
                    _mm256_storeu_si256(o as *mut __m256i, _mm512_cvtepi32_epi16(v));
                }
                TENSIX => {
                    let v = tensix::<BITS>(d, shared);
                    _mm256_storeu_si256(o as *mut __m256i, _mm512_cvtepi32_epi16(v));
                }
                _ => {
                    let v = _mm512_slli_epi32::<16>(tensix::<BITS>(d, shared));
                    _mm512_storeu_si512(o as *mut _, v);
                }
            }
        }
    }
}

fn decode<T, const OUT: u8>(elem: BfpElem, exps: &[u8], data: &[u8], dst: &mut [T]) {
    check_decode(elem, exps.len(), data.len(), dst.len());
    let (e, d, o, n) = (
        exps.as_ptr(),
        data.as_ptr(),
        dst.as_mut_ptr() as *mut u8,
        exps.len(),
    );
    // SAFETY: the caller holds a V4 token (AVX-512 F/BW/CD/DQ/VL + AVX2); `check_decode`
    // proved `data` has `n` rows of bytes and `dst` `n * 16` elements of `T`, the size
    // `OUT` writes.
    unsafe {
        match elem {
            BfpElem::M7 => decode_rows::<7, OUT>(elem, e, d, o, n),
            BfpElem::M3 => decode_rows::<3, OUT>(elem, e, d, o, n),
            BfpElem::M1 => decode_rows::<1, OUT>(elem, e, d, o, n),
        }
    }
}

#[inline]
fn encode<T, const BF: bool>(elem: BfpElem, src: &[T], exps: &mut [u8], data: &mut [u8]) {
    let rb = elem.row_bytes();
    for (r, e) in exps.iter_mut().enumerate() {
        let s = src[r * BFP_ROW..][..BFP_ROW].as_ptr() as *const u8;
        let d = data[r * rb..][..rb].as_mut_ptr();
        // SAFETY: the caller holds a V4 token (AVX-512 F/BW/VL + AVX2); `s` has 16 elements
        // and `d` has `rb` bytes (both slices were bounds-checked just above).
        unsafe {
            match elem {
                BfpElem::M7 => row::<7, BF>(s, e, d),
                BfpElem::M3 => row::<3, BF>(s, e, d),
                BfpElem::M1 => row::<1, BF>(s, e, d),
            }
        }
    }
}

impl V4 {
    /// [`super::bfp_encode_rows`] with the AVX-512 kernel and no per-call detection:
    /// detect once, pass the token. Bit-exact with the portable encoder.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_encode_rows(self, elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
        check_encode(elem, src.len(), exps.len(), data.len());
        encode::<f32, false>(elem, src, exps, data);
    }

    /// [`super::bfp_encode_rows_bf16`] with a token (see [`Self::bfp_encode_rows`]).
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_encode_rows_bf16(
        self,
        elem: BfpElem,
        src: &[u16],
        exps: &mut [u8],
        data: &mut [u8],
    ) {
        check_encode(elem, src.len(), exps.len(), data.len());
        encode::<u16, true>(elem, src, exps, data);
    }

    /// [`super::bfp_decode_rows`] with the AVX-512 kernel. Bit-exact with the portable decoder.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
        decode::<f32, F32>(elem, exps, data, out);
    }

    /// [`super::bfp_decode_rows_bf16`] with the AVX-512 kernel.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows_bf16(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
        decode::<u16, BF16>(elem, exps, data, out);
    }

    /// [`super::bfp_decode_rows_tensix`] with the AVX-512 kernel.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows_tensix(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
        decode::<u16, TENSIX>(elem, exps, data, out);
    }

    /// [`super::bfp_decode_rows_tensix_f32`] with the AVX-512 kernel.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows_tensix_f32(
        self,
        elem: BfpElem,
        exps: &[u8],
        data: &[u8],
        out: &mut [f32],
    ) {
        decode::<f32, TENSIX_F32>(elem, exps, data, out);
    }
}
