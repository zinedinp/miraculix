//! Tenstorrent block floating point `BFP_B` (`Bfp8_b`, `Bfp4_b`, `Bfp2_b`).
//!
//! A row is 16 values (one row of a 16x16 face) with one shared exponent:
//! the maximum biased `f32` exponent of the row, stored raw as a byte.
//! Each element keeps sign + magnitude: the significand with its hidden 1,
//! shifted right by `shared - exponent`, rounded to nearest even and
//! saturated to `BITS` bits. Inputs with exponent 0 flush to zero and a zero
//! magnitude drops its sign. Bit-exact with tt-metal's `bfp_simd::row_scalar`
//! (`scripts/tt-oracle`).
//!
//! The exponent stream (`exps`, 1 byte per row) and the element stream
//! (`data`) stay separate. The tile and face layout (64 exponent bytes first,
//! then mantissa bytes) is the caller's job.
//!
//! Element packing inside a row: element `k` of a byte sits at bit
//! `k * (BITS + 1)`, low bits first. A row takes 16 bytes (`M7`), 8 (`M3`)
//! or 4 (`M1`) data bytes.
//!
//! Decoding gives `sign * magnitude * 2^(shared - 127 - (BITS - 1))`.
//!
//! On x86_64 with AVX-512 (F, BW, VL) encoding does one row per `zmm`; on RISC-V
//! with `V`, one row per vector loop trip (`Rvv::bfp_encode_rows`).

use super::minifloat::pow2;

/// Values per row (and per shared exponent).
pub const BFP_ROW: usize = 16;

/// Element width of a `BFP_B` format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BfpElem {
    /// `Bfp8_b`: sign + 7 mantissa bits.
    M7,
    /// `Bfp4_b`: sign + 3 mantissa bits.
    M3,
    /// `Bfp2_b`: sign + 1 mantissa bit.
    M1,
}

impl BfpElem {
    /// Magnitude bits (7, 3 or 1).
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            BfpElem::M7 => 7,
            BfpElem::M3 => 3,
            BfpElem::M1 => 1,
        }
    }

    /// Data bytes per row of [`BFP_ROW`] values (16, 8 or 4).
    pub const fn row_bytes(self) -> usize {
        BFP_ROW * (self.mantissa_bits() as usize + 1) / 8
    }
}

fn row_scalar<const BITS: u32>(words: &[u32; 16], exp_out: &mut u8, out: &mut [u8]) {
    let shared = words.iter().map(|w| (w >> 23) & 255).max().unwrap_or(0);
    *exp_out = shared as u8;
    let shift = 24 - BITS;
    let limit = (1u32 << BITS) - 1;
    let mut codes = [0u32; 16];
    for (c, &w) in codes.iter_mut().zip(words) {
        let exponent = (w >> 23) & 255;
        let diff = shared - exponent;
        let mut m = (w & 0x7F_FFFF) | 0x80_0000;
        m = if diff >= 32 { 0 } else { m >> diff };
        m = (m + ((1u32 << (shift - 1)) - 1) + ((m >> shift) & 1)) >> shift;
        m = if exponent != 0 { m.min(limit) } else { 0 };
        *c = if m != 0 { m | ((w >> 31) << BITS) } else { 0 };
    }
    let per_byte = 8 / (BITS as usize + 1);
    for (j, o) in out.iter_mut().take(16 / per_byte).enumerate() {
        let mut v = 0u32;
        for k in 0..per_byte {
            v |= codes[j * per_byte + k] << (k as u32 * (BITS + 1));
        }
        *o = v as u8;
    }
}

fn load_f32(row: &[f32]) -> [u32; 16] {
    core::array::from_fn(|i| row[i].to_bits())
}

fn load_bf16(row: &[u16]) -> [u32; 16] {
    core::array::from_fn(|i| (row[i] as u32) << 16)
}

fn encode_scalar<const BITS: u32, T: Copy>(
    src: &[T],
    exps: &mut [u8],
    data: &mut [u8],
    load: fn(&[T]) -> [u32; 16],
) {
    let rb = BFP_ROW * (BITS as usize + 1) / 8;
    for ((row, e), d) in src.chunks_exact(BFP_ROW).zip(exps.iter_mut()).zip(data.chunks_exact_mut(rb)) {
        row_scalar::<BITS>(&load(row), e, d);
    }
}

fn check_encode(elem: BfpElem, n: usize, exps: usize, data: usize) {
    assert_eq!(n % BFP_ROW, 0, "length must be a multiple of {BFP_ROW}");
    let rows = n / BFP_ROW;
    assert_eq!(exps, rows, "one exponent byte per row");
    assert_eq!(data, rows * elem.row_bytes(), "data length");
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    /// # Safety
    /// AVX-512 F/BW/VL and AVX2 available. `src` points to 16 `f32` (`BF == false`)
    /// or 16 `u16` (`BF == true`); `out` to `16 * (BITS + 1) / 8` bytes.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
    pub(super) unsafe fn row<const BITS: u32, const BF: bool>(
        src: *const u8,
        exp_out: *mut u8,
        out: *mut u8,
    ) {
        unsafe {
            let words = if BF {
                _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(src as *const __m256i)))
            } else {
                _mm512_loadu_si512(src as *const _)
            };
            let exps = _mm512_and_si512(_mm512_srli_epi32::<23>(words), _mm512_set1_epi32(255));
            let max8 = _mm256_max_epu32(_mm512_castsi512_si256(exps), _mm512_extracti64x4_epi64::<1>(exps));
            let mut max4 = _mm_max_epu32(_mm256_castsi256_si128(max8), _mm256_extracti128_si256::<1>(max8));
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
            let sign = _mm512_sllv_epi32(_mm512_srli_epi32::<31>(words), _mm512_set1_epi32(BITS as i32));
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
}

fn encode_scalar_dispatch<T: Copy>(
    elem: BfpElem,
    src: &[T],
    exps: &mut [u8],
    data: &mut [u8],
    load: fn(&[T]) -> [u32; 16],
) {
    match elem {
        BfpElem::M7 => encode_scalar::<7, T>(src, exps, data, load),
        BfpElem::M3 => encode_scalar::<3, T>(src, exps, data, load),
        BfpElem::M1 => encode_scalar::<1, T>(src, exps, data, load),
    }
}

#[cfg(target_arch = "x86_64")]
fn encode_x86<T, const BF: bool>(elem: BfpElem, src: &[T], exps: &mut [u8], data: &mut [u8]) -> bool {
    match crate::x86::bundles::V4::detect() {
        Some(t) => {
            encode_v4::<T, BF>(t, elem, src, exps, data);
            true
        }
        None => false,
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn encode_v4<T, const BF: bool>(
    _t: crate::x86::bundles::V4,
    elem: BfpElem,
    src: &[T],
    exps: &mut [u8],
    data: &mut [u8],
) {
    let rb = elem.row_bytes();
    for (r, e) in exps.iter_mut().enumerate() {
        let s = src[r * BFP_ROW..][..BFP_ROW].as_ptr() as *const u8;
        let d = data[r * rb..][..rb].as_mut_ptr();
        // SAFETY: V4 proves AVX-512 F/BW/VL + AVX2; `s` has 16 elements and
        // `d` has `rb` bytes (both slices were bounds-checked just above).
        unsafe {
            match elem {
                BfpElem::M7 => x86::row::<7, BF>(s, e, d),
                BfpElem::M3 => x86::row::<3, BF>(s, e, d),
                BfpElem::M1 => x86::row::<1, BF>(s, e, d),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl crate::x86::bundles::V4 {
    /// [`bfp_encode_rows`] with the AVX-512 kernel and no per-call detection:
    /// detect once, pass the token. Bit-exact with the portable encoder.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_encode_rows(self, elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
        check_encode(elem, src.len(), exps.len(), data.len());
        encode_v4::<f32, false>(self, elem, src, exps, data);
    }

    /// [`bfp_encode_rows_bf16`] with a token (see [`Self::bfp_encode_rows`]).
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_encode_rows_bf16(self, elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
        check_encode(elem, src.len(), exps.len(), data.len());
        encode_v4::<u16, true>(self, elem, src, exps, data);
    }

    /// [`bfp_decode_rows`] taking a token (decode has no AVX-512 kernel yet,
    /// so this is the portable decoder).
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
        bfp_decode_rows(elem, exps, data, out);
    }

    /// [`bfp_decode_rows_bf16`] taking a token.
    ///
    /// # Panics
    /// If the lengths do not match.
    #[inline]
    pub fn bfp_decode_rows_bf16(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
        bfp_decode_rows_bf16(elem, exps, data, out);
    }
}

/// Encodes `f32` rows. `src.len()` must be a multiple of [`BFP_ROW`];
/// `exps` gets one byte per row and `data` [`BfpElem::row_bytes`] per row.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_encode_rows(elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    #[cfg(target_arch = "x86_64")]
    if encode_x86::<f32, false>(elem, src, exps, data) {
        return;
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    if let Some(t) = crate::riscv::ops::rvv::Rvv::detect() {
        return t.bfp_encode_rows(elem, src, exps, data);
    }
    encode_scalar_dispatch(elem, src, exps, data, load_f32);
}

/// [`bfp_encode_rows`] for bfloat16 input (`u16` bit patterns).
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_encode_rows_bf16(elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    #[cfg(target_arch = "x86_64")]
    if encode_x86::<u16, true>(elem, src, exps, data) {
        return;
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    if let Some(t) = crate::riscv::ops::rvv::Rvv::detect() {
        return t.bfp_encode_rows_bf16(elem, src, exps, data);
    }
    encode_scalar_dispatch(elem, src, exps, data, load_bf16);
}

/// Portable encoder, no SIMD. The reference the hardware paths are tested against.
#[doc(hidden)]
pub fn bfp_encode_rows_scalar(elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    encode_scalar_dispatch(elem, src, exps, data, load_f32);
}

/// Portable bf16 encoder, no SIMD.
#[doc(hidden)]
pub fn bfp_encode_rows_bf16_scalar(elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    encode_scalar_dispatch(elem, src, exps, data, load_bf16);
}

fn decode_row(elem: BfpElem, shared: u8, d: &[u8], mut put: impl FnMut(usize, f32)) {
    let bits = elem.mantissa_bits();
    let unit = pow2(shared as i32 - 127 - (bits as i32 - 1));
    let w = bits + 1;
    let per = 8 / w as usize;
    let mask = (1u32 << w) - 1;
    for i in 0..BFP_ROW {
        let code = (d[i / per] as u32 >> ((i % per) as u32 * w)) & mask;
        let mag = (code & ((1 << bits) - 1)) as f32 * unit;
        put(i, if code >> bits != 0 { -mag } else { mag });
    }
}

fn check_decode(elem: BfpElem, exps: usize, data: usize, out: usize) {
    assert_eq!(out, exps * BFP_ROW, "output length");
    assert_eq!(data, exps * elem.row_bytes(), "data length");
}

/// Decodes rows to `f32`: `exps` has one byte per row, `out` [`BFP_ROW`] values per row.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
    check_decode(elem, exps.len(), data.len(), out.len());
    let rb = elem.row_bytes();
    for (r, &e) in exps.iter().enumerate() {
        let o = &mut out[r * BFP_ROW..][..BFP_ROW];
        decode_row(elem, e, &data[r * rb..][..rb], |i, v| o[i] = v);
    }
}

/// [`bfp_decode_rows`] to bfloat16 bits. Exact: at most 7 significant bits per value.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows_bf16(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
    check_decode(elem, exps.len(), data.len(), out.len());
    let rb = elem.row_bytes();
    for (r, &e) in exps.iter().enumerate() {
        let o = &mut out[r * BFP_ROW..][..BFP_ROW];
        decode_row(elem, e, &data[r * rb..][..rb], |i, v| o[i] = (v.to_bits() >> 16) as u16);
    }
}
