//! SVE `FEAT_F64MM` (v8.6): double-precision 2×2 matrix multiply-accumulate.
//! Token: [`SveF64mm`].
//!
//! Same layout as [`super::sve_f32mm`], but a 2×2 `f64` matrix fills a
//! **256-bit** segment. That is why FEAT_F64MM needs a vector length of at
//! least 256 bits: the kernel (and QEMU with `sve256=off`) withholds the
//! feature bit on a 128-bit machine, so this token never exists there.

use super::super::macros::{arm_token, sve_stream};
use super::check_acc_shape;

arm_token! {
    /// Proof that SVE with `FEAT_F64MM` (`FMMLA z.d`) is available - and so a
    /// vector length of at least 256 bits.
    token = SveF64mm, requires = [Sve, Sve2F64mm],
}

sve_stream! {
    target_feature = "sve,f64mm", inner_fn = sve_fmmla_f64,
    size = ["d", "ld1d", "st1d", ", lsl #3", "incd"],
    ins = [acc: f64 => "0", a: f64 => "1", b: f64 => "2"], out = f64, result = "0",
    op = ["fmmla z0.d, z1.d, z2.d"],
}

impl SveF64mm {
    /// 2×2 · 2×2 matrix multiply-accumulate per 4-lane (256-bit) segment:
    /// `out[2i+j] = acc[2i+j] + a[2i]·b[2j] + a[2i+1]·b[2j+1]` (`fmmla z.d`).
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 4.
    #[inline]
    pub fn mmla_f64_slice(self, acc: &[f64], a: &[f64], b: &[f64], out: &mut [f64]) {
        check_acc_shape(acc.len(), out.len(), &[a.len(), b.len()], 1, 4);
        unsafe {
            sve_fmmla_f64(
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
#[path = "../../test/ops/sve/sve_f64mm.rs"]
mod tests;
