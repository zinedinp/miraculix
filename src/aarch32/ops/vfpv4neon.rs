//! Neon + VFPv4 FMA (`+FMA`, GCC `-mfpu=neon-vfpv4`). Token: [`Vfpv4Neon`].
//! Separate from [`super::neon::Neon`]; needs both
//! [`Feature::Neon`](super::super::Feature::Neon) and
//! [`Feature::Vfpv4`](super::super::Feature::Vfpv4). Upstream:
//! `core::arch::arm::vfmaq_f32`.
//!
//! Two operand orders, told apart by name: the fixed-width [`Vfpv4Neon::fma_f32x4`]
//! keeps ACLE's `vfmaq_f32` order (`b * c + a`, addend first), the slice form
//! `fmadd_f32_slice` uses the crate-wide `fmadd` order (`a * b + c`), as on
//! aarch64 and x86.

use crate::arm_shared::macros::{arm_token, neon_fma_slice};
use crate::arm_shared::simd_arch::{float32x4_t, vfmaq_f32, vld1q_f32, vst1q_f32};

arm_token! {
    /// Proof that Neon + VFPv4 fused multiply-add is available.
    token = Vfpv4Neon, requires = [Neon, Vfpv4],
}

impl Vfpv4Neon {
    /// `VFMA.F32`: per-lane fused `b*c + a` (single rounding). `a` is the
    /// accumulator (matches `vfmaq_f32` / `simd_fma(b, c, a)`, not x86
    /// `fmadd(a,b,c)`), so this one is hand-written rather than
    /// `neon_fma!`, which uses the `fmadd` order.
    #[inline]
    pub fn fma_f32x4(self, a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
        #[target_feature(enable = "neon,v7,vfp4")]
        unsafe fn imp(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
            unsafe {
                let rv = vfmaq_f32(
                    vld1q_f32(a.as_ptr()),
                    vld1q_f32(b.as_ptr()),
                    vld1q_f32(c.as_ptr()),
                );
                let mut out = [0f32; 4];
                vst1q_f32(out.as_mut_ptr(), rv);
                out
            }
        }
        unsafe { imp(a, b, c) }
    }
}

neon_fma_slice! {
    token = Vfpv4Neon, vis = pub, target_feature = "neon,v7,vfp4",
    slice_fn = fmadd_f32_slice,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vfmaq_f32,
    scalar = |x, y, z| x * y + z,
    slice_doc = "`VFMA.F32` over whole slices: `out = a * b + c` (the `fmadd` order, not [`Vfpv4Neon::fma_f32x4`]'s). The vector body is fused; a tail shorter than 4 lanes is not (it evaluates `a * b + c` as written), matching `auto_up::fmadd_f32`.",
}

#[cfg(test)]
#[path = "../test/ops/vfpv4neon.rs"]
mod tests;
