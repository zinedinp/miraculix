//! SVE `FEAT_F32MM` (v8.6): single-precision 2×2 matrix multiply-accumulate.
//! Token: [`SveF32mm`].
//!
//! Every 128-bit segment (4 `f32` lanes) holds one 2×2 matrix, row-major:
//! `out[2i+j] = acc[2i+j] + Σₖ a[2i+k]·b[2j+k]` - the `A·Bᵀ` layout of the
//! integer and `bf16` matrix ops. No Neon counterpart.

use super::super::macros::{arm_token, sve_stream};
use super::check_acc_shape;

arm_token! {
    /// Proof that SVE with `FEAT_F32MM` (`FMMLA z.s`) is available.
    token = SveF32mm, requires = [Sve, Sve2F32mm],
}

sve_stream! {
    target_feature = "sve,f32mm", inner_fn = sve_fmmla_f32,
    size = ["s", "ld1w", "st1w", ", lsl #2", "incw"],
    ins = [acc: f32 => "0", a: f32 => "1", b: f32 => "2"], out = f32, result = "0",
    op = ["fmmla z0.s, z1.s, z2.s"],
}

impl SveF32mm {
    /// 2×2 · 2×2 matrix multiply-accumulate per 4-lane segment:
    /// `out[2i+j] = acc[2i+j] + a[2i]·b[2j] + a[2i+1]·b[2j+1]` (`fmmla z.s`).
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 4.
    #[inline]
    pub fn mmla_f32_slice(self, acc: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
        check_acc_shape(acc.len(), out.len(), &[a.len(), b.len()], 1, 4);
        unsafe {
            sve_fmmla_f32(
                acc.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                out.len(),
            )
        }
    }
}

#[cfg(test)]
#[path = "../../test/ops/sve/sve_f32mm.rs"]
mod tests;
