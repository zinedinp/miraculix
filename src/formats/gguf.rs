//! GGUF / llama.cpp quantized blocks, decode only (for quality comparison).
//!
//! Layouts and arithmetic follow `dequantize_row_q8_0` / `dequantize_row_q6_K`
//! in ggml (`scripts/gguf-oracle` checks the output bit for bit).

use super::MiniFloat;

/// Elements per `Q8_0` block.
pub const Q8_0_BLOCK: usize = 32;
/// Bytes per `Q8_0` block: `f16` scale + 32 `i8`.
pub const Q8_0_BYTES: usize = 2 + Q8_0_BLOCK;
/// Elements per `Q6_K` super-block.
pub const Q6_K_BLOCK: usize = 256;
/// Bytes per `Q6_K` super-block: 128 low nibbles, 64 high 2-bit pairs, 16 `i8` scales, `f16` scale.
pub const Q6_K_BYTES: usize = 128 + 64 + 16 + 2;

fn half(lo: u8, hi: u8) -> f32 {
    MiniFloat::F16.decode(u16::from_le_bytes([lo, hi]) as u32)
}

/// Decodes `Q8_0` blocks: `y = d * q`.
///
/// # Panics
/// If `out.len()` is not a multiple of [`Q8_0_BLOCK`] or `blocks` is not
/// `out.len() / 32 * 34` bytes.
pub fn dequantize_q8_0(blocks: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % Q8_0_BLOCK, 0);
    assert_eq!(blocks.len(), out.len() / Q8_0_BLOCK * Q8_0_BYTES);
    for (b, o) in blocks
        .chunks_exact(Q8_0_BYTES)
        .zip(out.chunks_exact_mut(Q8_0_BLOCK))
    {
        let d = half(b[0], b[1]);
        for (y, &q) in o.iter_mut().zip(&b[2..]) {
            *y = (q as i8) as f32 * d;
        }
    }
}

/// Decodes `Q6_K` super-blocks: 6-bit values (`ql` low nibble + `qh` two bits, minus 32)
/// times an `i8` sub-scale (16 values each) times the `f16` super-block scale.
///
/// # Panics
/// If `out.len()` is not a multiple of [`Q6_K_BLOCK`] or `blocks` is not
/// `out.len() / 256 * 210` bytes.
pub fn dequantize_q6_k(blocks: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % Q6_K_BLOCK, 0);
    assert_eq!(blocks.len(), out.len() / Q6_K_BLOCK * Q6_K_BYTES);
    for (b, y) in blocks
        .chunks_exact(Q6_K_BYTES)
        .zip(out.chunks_exact_mut(Q6_K_BLOCK))
    {
        let (ql_all, rest) = b.split_at(128);
        let (qh_all, rest) = rest.split_at(64);
        let (sc_all, dh) = rest.split_at(16);
        let d = half(dh[0], dh[1]);
        for n in 0..2 {
            let (ql, qh, sc) = (&ql_all[n * 64..], &qh_all[n * 32..], &sc_all[n * 8..]);
            let yy = &mut y[n * 128..];
            for l in 0..32 {
                let is = l / 16;
                let q = |lo: u8, hi: u8| ((lo | (hi << 4)) as i8) as i32 - 32;
                let q1 = q(ql[l] & 0xF, (qh[l] >> 0) & 3);
                let q2 = q(ql[l + 32] & 0xF, (qh[l] >> 2) & 3);
                let q3 = q(ql[l] >> 4, (qh[l] >> 4) & 3);
                let q4 = q(ql[l + 32] >> 4, (qh[l] >> 6) & 3);
                yy[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                yy[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                yy[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                yy[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}
