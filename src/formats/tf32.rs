//! TF32: f32 range, 10 mantissa bits. On a CPU it is an `f32` whose low 13
//! mantissa bits are zero, so no new type is needed.

/// Rounds `x` to TF32 precision (round to nearest even), keeping `f32` storage.
pub fn round_to_tf32(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7F80_0000) == 0x7F80_0000 {
        return x; // inf / NaN
    }
    let r = b.wrapping_add(0xFFF + ((b >> 13) & 1)) & !0x1FFF;
    // Rounding past f32::MAX carries into infinity, like hardware.
    f32::from_bits(r)
}

/// In-place [`round_to_tf32`] over a slice.
pub fn round_to_tf32_slice(data: &mut [f32]) {
    data.iter_mut().for_each(|v| *v = round_to_tf32(*v));
}
