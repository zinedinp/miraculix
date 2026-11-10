//! ARMv8.2-A `FEAT_FHM`: widening half-precision multiply-accumulate into
//! single precision. Token: [`Fhm`].
//!
//! `FMLAL` takes four `f16` products (exact in `f32`) and adds each to an
//! `f32` accumulator lane with **one** rounding; `_low` reads lanes 0-3 of
//! the `f16` operands, `_high` lanes 4-7. `FMLSL` subtracts. `f16` operands
//! are bit patterns (`[u16; 8]`), as in [`super::fp16`].
//!
//! Intrinsics stable since Rust 1.94 (`stdarch_neon_fp16`). aarch64-only:
//! `core::arch::arm` has no `vfmlal*`, which is why `aarch32::ops` lists
//! `Feature::Fhm` as detect-only.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    float32x4_t, vfmlalq_high_f16, vfmlalq_low_f16, vfmlslq_high_f16, vfmlslq_low_f16, vld1q_f32,
    vld1q_u16, vreinterpretq_f16_u16, vst1q_f32,
};

arm_token! {
    /// Proof that `FEAT_FHM` (`FMLAL`/`FMLSL`) is available. Requires the
    /// `FEAT_FP16` bits too: the instructions operate on half-precision data.
    token = Fhm, requires = [Fhm, Fp16, AsimdHp],
}

macro_rules! fhm_op {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Fhm {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, acc: [f32; 4], a: [u16; 8], b: [u16; 8]) -> [f32; 4] {
                unsafe { $intrinsic_fn(&acc, &a, &b) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,fp16,fhm")]
        unsafe fn $intrinsic_fn(acc: &[f32; 4], a: &[u16; 8], b: &[u16; 8]) -> [f32; 4] {
            unsafe {
                let vacc: float32x4_t = vld1q_f32(acc.as_ptr());
                let va = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
                let vb = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr()));
                let mut out = [0f32; 4];
                vst1q_f32(out.as_mut_ptr(), $intrinsic(vacc, va, vb));
                out
            }
        }
    };
}

fhm_op!(
    fmlal_low_f32x4,
    fmlal_low_f32q_fixed,
    vfmlalq_low_f16,
    "`acc[i] + a[i]·b[i]` for `f16` lanes 0-3, widened to `f32`, one rounding (`fmlal v.4s, v.4h, v.4h`)."
);
fhm_op!(
    fmlal_high_f32x4,
    fmlal_high_f32q_fixed,
    vfmlalq_high_f16,
    "`acc[i] + a[i+4]·b[i+4]` for `f16` lanes 4-7, widened to `f32`, one rounding (`fmlal2 v.4s, v.4h, v.4h`)."
);
fhm_op!(
    fmlsl_low_f32x4,
    fmlsl_low_f32q_fixed,
    vfmlslq_low_f16,
    "`acc[i] - a[i]·b[i]` for `f16` lanes 0-3, widened to `f32`, one rounding (`fmlsl v.4s, v.4h, v.4h`)."
);
fhm_op!(
    fmlsl_high_f32x4,
    fmlsl_high_f32q_fixed,
    vfmlslq_high_f16,
    "`acc[i] - a[i+4]·b[i+4]` for `f16` lanes 4-7, widened to `f32`, one rounding (`fmlsl2 v.4s, v.4h, v.4h`)."
);

#[cfg(test)]
#[path = "../../test/ops/neon/fhm.rs"]
mod tests;
