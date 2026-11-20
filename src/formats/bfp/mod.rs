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
//! Two decodes:
//!
//! | Function | Result |
//! |---|---|
//! | [`bfp_decode_rows`], [`bfp_decode_rows_bf16`] | the value, `sign * magnitude * 2^(shared - 127 - (BITS - 1))`, as `f32` (BF16: its top half) |
//! | [`bfp_decode_rows_tensix`], [`bfp_decode_rows_tensix_f32`] | the BF16 bit pattern the Tensix unpacker produces (`FloatBitPatterns.md:119-139`) |
//!
//! They differ in three places, none of which an encoder reaches from finite inputs
//! above ~2^-120: a sign without magnitude (`-0` against `-inf`); a normalised
//! exponent (`shared` less the magnitude's leading zeros) of 0 or below, where the
//! unpacker's `u8` exponent wraps; and a normalised exponent of 255 with a mantissa,
//! where the value overflows to `inf` and the unpacker writes a NaN pattern.
//!
//! | Module | Contents |
//! |---|---|
//! | `scalar` | Portable rows: the reference every SIMD path is tested against |
//! | `x86` | AVX-512 (x86-64-v4): one row per `zmm` for encode and both decodes; [`crate::x86::bundles::V4`] token methods |
//!
//! On RISC-V with `V` encoding runs one row per vector loop trip
//! (`Rvv::bfp_encode_rows`); decoding is portable there.

mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

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

fn check_encode(elem: BfpElem, n: usize, exps: usize, data: usize) {
    assert_eq!(n % BFP_ROW, 0, "length must be a multiple of {BFP_ROW}");
    let rows = n / BFP_ROW;
    assert_eq!(exps, rows, "one exponent byte per row");
    assert_eq!(data, rows * elem.row_bytes(), "data length");
}

fn check_decode(elem: BfpElem, exps: usize, data: usize, out: usize) {
    assert_eq!(out, exps * BFP_ROW, "output length");
    assert_eq!(data, exps * elem.row_bytes(), "data length");
}

#[cfg(target_arch = "x86_64")]
fn v4() -> Option<crate::x86::bundles::V4> {
    crate::x86::bundles::V4::detect()
}

/// Encodes `f32` rows. `src.len()` must be a multiple of [`BFP_ROW`];
/// `exps` gets one byte per row and `data` [`BfpElem::row_bytes`] per row.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_encode_rows(elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_encode_rows(elem, src, exps, data);
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    if let Some(t) = crate::riscv::ops::rvv::Rvv::detect() {
        return t.bfp_encode_rows(elem, src, exps, data);
    }
    bfp_encode_rows_scalar(elem, src, exps, data);
}

/// [`bfp_encode_rows`] for bfloat16 input (`u16` bit patterns).
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_encode_rows_bf16(elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_encode_rows_bf16(elem, src, exps, data);
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    if let Some(t) = crate::riscv::ops::rvv::Rvv::detect() {
        return t.bfp_encode_rows_bf16(elem, src, exps, data);
    }
    bfp_encode_rows_bf16_scalar(elem, src, exps, data);
}

/// Portable encoder, no SIMD. The reference the hardware paths are tested against.
#[doc(hidden)]
pub fn bfp_encode_rows_scalar(elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    scalar::encode_scalar_dispatch(elem, src, exps, data, scalar::load_f32);
}

/// Portable bf16 encoder, no SIMD.
#[doc(hidden)]
pub fn bfp_encode_rows_bf16_scalar(elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
    check_encode(elem, src.len(), exps.len(), data.len());
    scalar::encode_scalar_dispatch(elem, src, exps, data, scalar::load_bf16);
}

/// Decodes rows to `f32`: `exps` has one byte per row, `out` [`BFP_ROW`] values per row.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_decode_rows(elem, exps, data, out);
    }
    bfp_decode_rows_scalar(elem, exps, data, out);
}

/// [`bfp_decode_rows`] to bfloat16 bits. Exact: at most 7 significant bits per value.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows_bf16(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_decode_rows_bf16(elem, exps, data, out);
    }
    bfp_decode_rows_bf16_scalar(elem, exps, data, out);
}

/// Decodes rows to the BF16 bit patterns the Tensix unpacker produces from them
/// (`FloatBitPatterns.md:119-139`): bit-exact with the card, wrapping exponent and
/// `-inf` for a sign without magnitude included.
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows_tensix(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_decode_rows_tensix(elem, exps, data, out);
    }
    bfp_decode_rows_tensix_scalar(elem, exps, data, out);
}

/// [`bfp_decode_rows_tensix`] widened to `f32` (the BF16 pattern in the top half).
///
/// # Panics
/// If the lengths do not match.
pub fn bfp_decode_rows_tensix_f32(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = v4() {
        return t.bfp_decode_rows_tensix_f32(elem, exps, data, out);
    }
    bfp_decode_rows_tensix_f32_scalar(elem, exps, data, out);
}

fn decode_scalar<T>(
    elem: BfpElem,
    exps: &[u8],
    data: &[u8],
    out: &mut [T],
    row: impl Fn(BfpElem, u8, &[u8], &mut [T]),
) {
    check_decode(elem, exps.len(), data.len(), out.len());
    let rb = elem.row_bytes();
    for (r, &e) in exps.iter().enumerate() {
        row(
            elem,
            e,
            &data[r * rb..][..rb],
            &mut out[r * BFP_ROW..][..BFP_ROW],
        );
    }
}

/// Portable [`bfp_decode_rows`].
#[doc(hidden)]
pub fn bfp_decode_rows_scalar(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
    decode_scalar(elem, exps, data, out, |elem, e, d, o| {
        scalar::decode_row(elem, e, d, |i, v| o[i] = v)
    });
}

/// Portable [`bfp_decode_rows_bf16`].
#[doc(hidden)]
pub fn bfp_decode_rows_bf16_scalar(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
    decode_scalar(elem, exps, data, out, |elem, e, d, o| {
        scalar::decode_row(elem, e, d, |i, v| o[i] = (v.to_bits() >> 16) as u16)
    });
}

/// Portable [`bfp_decode_rows_tensix`].
#[doc(hidden)]
pub fn bfp_decode_rows_tensix_scalar(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
    decode_scalar(elem, exps, data, out, |elem, e, d, o| {
        scalar::decode_row_tensix(elem, e, d, |i, v| o[i] = v)
    });
}

/// Portable [`bfp_decode_rows_tensix_f32`].
#[doc(hidden)]
pub fn bfp_decode_rows_tensix_f32_scalar(elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
    decode_scalar(elem, exps, data, out, |elem, e, d, o| {
        scalar::decode_row_tensix(elem, e, d, |i, v| o[i] = f32::from_bits((v as u32) << 16))
    });
}
