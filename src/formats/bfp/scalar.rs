//! Portable rows: the encoder every SIMD path is tested against, and both decodes.

use super::{BFP_ROW, BfpElem};
use crate::formats::minifloat::pow2;

pub(super) fn row_scalar<const BITS: u32>(words: &[u32; 16], exp_out: &mut u8, out: &mut [u8]) {
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

pub(super) fn load_f32(row: &[f32]) -> [u32; 16] {
    core::array::from_fn(|i| row[i].to_bits())
}

pub(super) fn load_bf16(row: &[u16]) -> [u32; 16] {
    core::array::from_fn(|i| (row[i] as u32) << 16)
}

fn encode_scalar<const BITS: u32, T: Copy>(
    src: &[T],
    exps: &mut [u8],
    data: &mut [u8],
    load: fn(&[T]) -> [u32; 16],
) {
    let rb = BFP_ROW * (BITS as usize + 1) / 8;
    for ((row, e), d) in src
        .as_chunks::<BFP_ROW>()
        .0
        .iter()
        .zip(exps.iter_mut())
        .zip(data.chunks_exact_mut(rb))
    {
        row_scalar::<BITS>(&load(row), e, d);
    }
}

pub(super) fn encode_scalar_dispatch<T: Copy>(
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

/// The unit of a row: one step of the magnitude, `2^(shared - 127 - (BITS - 1))`.
pub(super) fn unit(elem: BfpElem, shared: u8) -> f32 {
    pow2(shared as i32 - 127 - (elem.mantissa_bits() as i32 - 1))
}

/// Element `i` of a row's data bytes: sign in bit `BITS`, magnitude below.
#[inline]
pub(super) fn code(elem: BfpElem, d: &[u8], i: usize) -> u32 {
    let w = elem.mantissa_bits() + 1;
    let per = 8 / w as usize;
    (d[i / per] as u32 >> ((i % per) as u32 * w)) & ((1 << w) - 1)
}

/// One row as values: `sign * magnitude * unit`.
pub(super) fn decode_row(elem: BfpElem, shared: u8, d: &[u8], mut put: impl FnMut(usize, f32)) {
    let bits = elem.mantissa_bits();
    let unit = unit(elem, shared);
    for i in 0..BFP_ROW {
        let code = code(elem, d, i);
        let m = code & ((1 << bits) - 1);
        // A zero magnitude is zero even where the unit overflows (BFP2 at shared 255).
        let mag = if m == 0 { 0.0 } else { m as f32 * unit };
        put(i, if code >> bits != 0 { -mag } else { mag });
    }
}

/// One code as the Tensix unpacker turns it into BF16 (`FloatBitPatterns.md:119-139`): the
/// code moves to the top of a byte, the magnitude is normalised with its leading zeros taken
/// off the shared exponent as a `u8` (so it wraps), and a sign without magnitude is `-inf`.
#[inline]
pub(super) const fn tensix_bf16(bits: u32, code: u32, shared: u8) -> u16 {
    let datum = (code << (7 - bits)) as u8;
    let sign = (datum >> 7) as u16;
    let mag = datum << 1;
    if mag == 0 {
        if sign != 0 { 0xff80 } else { 0 }
    } else {
        let lz = mag.leading_zeros();
        let mag = mag << lz;
        let exp = shared.wrapping_sub(lz as u8);
        (sign << 15) | ((exp as u16) << 7) | ((mag & 0x7e) as u16)
    }
}

/// One row as the unpacker's BF16 bit patterns.
pub(super) fn decode_row_tensix(
    elem: BfpElem,
    shared: u8,
    d: &[u8],
    mut put: impl FnMut(usize, u16),
) {
    let bits = elem.mantissa_bits();
    for i in 0..BFP_ROW {
        put(i, tensix_bf16(bits, code(elem, d, i), shared));
    }
}
