//! Block-scaled formats: OCP MX (MXFP4/6/8, MXINT8) and NVFP4.
//!
//! The CPU equivalent of a block-scaled matrix engine is "decode the
//! elements, multiply by the block scale, then use `f32` (or BF16 via
//! `dpbf16`) math". These functions do the decode and the matching quantize.
//!
//! Memory layout (little-endian bit order): elements are packed LSB-first in
//! one bitstream per block, so FP4 is two per byte (element 0 in the low
//! nibble) and FP6 is four per three bytes.

use super::minifloat::{MiniFloat, Overflow, pow2};

/// Elements per MX block.
pub const MX_BLOCK: usize = 32;
/// Elements per NVFP4 block.
pub const NVFP4_BLOCK: usize = 16;

/// Element type of an MX block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MxElem {
    /// MXFP4 (E2M1).
    Fp4,
    /// MXFP6 E2M3.
    Fp6E2M3,
    /// MXFP6 E3M2.
    Fp6E3M2,
    /// MXFP8 E4M3 (OCP FN).
    Fp8E4M3,
    /// MXFP8 E5M2.
    Fp8E5M2,
    /// MXINT8: two's complement, value = `i8 / 64`.
    Int8,
}

impl MxElem {
    fn float(self) -> Option<MiniFloat> {
        match self {
            MxElem::Fp4 => Some(MiniFloat::E2M1),
            MxElem::Fp6E2M3 => Some(MiniFloat::E2M3),
            MxElem::Fp6E3M2 => Some(MiniFloat::E3M2),
            MxElem::Fp8E4M3 => Some(MiniFloat::E4M3Fn),
            MxElem::Fp8E5M2 => Some(MiniFloat::E5M2),
            MxElem::Int8 => None,
        }
    }

    /// Bits per element.
    pub const fn bits(self) -> usize {
        match self {
            MxElem::Fp4 => 4,
            MxElem::Fp6E2M3 | MxElem::Fp6E3M2 => 6,
            MxElem::Fp8E4M3 | MxElem::Fp8E5M2 | MxElem::Int8 => 8,
        }
    }

    /// Bytes of element data per 32-element block.
    pub const fn block_bytes(self) -> usize {
        MX_BLOCK * self.bits() / 8
    }

    fn emax(self) -> i32 {
        self.float().map_or(0, MiniFloat::emax)
    }
}

fn get_bits(data: &[u8], i: usize, w: usize) -> u32 {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    let mut v = data[byte] as u32 >> off;
    if off + w > 8 {
        v |= (data[byte + 1] as u32) << (8 - off);
    }
    v & ((1 << w) - 1)
}

fn put_bits(data: &mut [u8], i: usize, w: usize, v: u32) {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    data[byte] |= (v << off) as u8;
    if off + w > 8 {
        data[byte + 1] |= (v >> (8 - off)) as u8;
    }
}

/// E8M0 scale to `f32`: `2^(s - 127)`, `0xFF` is NaN.
pub fn e8m0_to_f32(s: u8) -> f32 {
    if s == 0xFF { f32::NAN } else { pow2(s as i32 - 127) }
}

fn round_even(v: f32) -> f32 {
    const M: f32 = 12582912.0; // 1.5 * 2^23, valid for |v| < 2^22
    (v + M) - M
}

/// Dequantizes MX blocks into `out`. `out.len()` must be a multiple of
/// [`MX_BLOCK`]; `scales` holds one E8M0 byte per block, `data` the packed
/// elements.
///
/// # Panics
/// If the lengths do not match the block count.
pub fn mx_dequantize(elem: MxElem, scales: &[u8], data: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % MX_BLOCK, 0);
    let blocks = out.len() / MX_BLOCK;
    assert!(scales.len() >= blocks && data.len() >= blocks * elem.block_bytes());
    let (w, bb) = (elem.bits(), elem.block_bytes());
    for (b, ob) in out.chunks_exact_mut(MX_BLOCK).enumerate() {
        let scale = e8m0_to_f32(scales[b]);
        let d = &data[b * bb..(b + 1) * bb];
        for (i, o) in ob.iter_mut().enumerate() {
            let c = get_bits(d, i, w);
            *o = match elem.float() {
                Some(f) => f.decode(c) * scale,
                None => (c as u8 as i8) as f32 * (1.0 / 64.0) * scale,
            };
        }
    }
}

/// Quantizes `src` (length a multiple of [`MX_BLOCK`]) into MX blocks using
/// the OCP rule: shared exponent `floor(log2(amax)) - emax(elem)`, elements
/// rounded to nearest even and saturated.
///
/// # Panics
/// If the lengths do not match the block count.
pub fn mx_quantize(elem: MxElem, src: &[f32], scales: &mut [u8], data: &mut [u8]) {
    assert_eq!(src.len() % MX_BLOCK, 0);
    let blocks = src.len() / MX_BLOCK;
    assert!(scales.len() >= blocks && data.len() >= blocks * elem.block_bytes());
    let (w, bb) = (elem.bits(), elem.block_bytes());
    for (b, sb) in src.chunks_exact(MX_BLOCK).enumerate() {
        let amax = sb.iter().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m });
        let s = if amax == 0.0 || !amax.is_finite() {
            127
        } else {
            let e = ((amax.to_bits() >> 23) as i32 - 127) - elem.emax();
            (e + 127).clamp(0, 254)
        } as u8;
        scales[b] = s;
        let inv = 1.0 / e8m0_to_f32(s);
        let d = &mut data[b * bb..(b + 1) * bb];
        d.fill(0);
        for (i, &x) in sb.iter().enumerate() {
            let v = x * inv;
            let c = match elem.float() {
                Some(f) => f.encode(v, Overflow::Saturate),
                None => round_even((v * 64.0).clamp(-127.0, 127.0)) as i32 as u8 as u32,
            };
            put_bits(d, i, w, c);
        }
    }
}

/// NVFP4 per-tensor scale for `src`: `amax / (6 * 448)`, so block scales fit E4M3.
/// Returns `1.0` for an all-zero tensor.
pub fn nvfp4_tensor_scale(src: &[f32]) -> f32 {
    let amax = src.iter().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m });
    if amax == 0.0 { 1.0 } else { amax / (6.0 * 448.0) }
}

/// Dequantizes NVFP4: 16-element FP4 blocks, each with an E4M3 scale byte,
/// times one per-tensor `f32` scale. `out.len()` must be a multiple of [`NVFP4_BLOCK`].
///
/// # Panics
/// If the lengths do not match the block count.
pub fn nvfp4_dequantize(scales: &[u8], tensor_scale: f32, data: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % NVFP4_BLOCK, 0);
    let blocks = out.len() / NVFP4_BLOCK;
    assert!(scales.len() >= blocks && data.len() >= blocks * NVFP4_BLOCK / 2);
    for (b, ob) in out.chunks_exact_mut(NVFP4_BLOCK).enumerate() {
        let s = MiniFloat::E4M3Fn.decode(scales[b] as u32) * tensor_scale;
        let d = &data[b * 8..(b + 1) * 8];
        for (i, o) in ob.iter_mut().enumerate() {
            *o = MiniFloat::E2M1.decode(get_bits(d, i, 4)) * s;
        }
    }
}

/// Quantizes to NVFP4 with the given `tensor_scale` (see [`nvfp4_tensor_scale`]).
///
/// # Panics
/// If the lengths do not match the block count.
pub fn nvfp4_quantize(src: &[f32], tensor_scale: f32, scales: &mut [u8], data: &mut [u8]) {
    assert_eq!(src.len() % NVFP4_BLOCK, 0);
    let blocks = src.len() / NVFP4_BLOCK;
    assert!(scales.len() >= blocks && data.len() >= blocks * NVFP4_BLOCK / 2);
    for (b, sb) in src.chunks_exact(NVFP4_BLOCK).enumerate() {
        let amax = sb.iter().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m });
        let sc = MiniFloat::E4M3Fn.encode(amax / 6.0 / tensor_scale, Overflow::Saturate);
        scales[b] = sc as u8;
        let real = MiniFloat::E4M3Fn.decode(sc) * tensor_scale;
        let d = &mut data[b * 8..(b + 1) * 8];
        d.fill(0);
        for (i, &x) in sb.iter().enumerate() {
            let c = if real == 0.0 { 0 } else { MiniFloat::E2M1.encode(x / real, Overflow::Saturate) };
            put_bits(d, i, 4, c);
        }
    }
}
