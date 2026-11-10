//! ARMv8.1-A `FEAT_RDM`: saturating rounding doubling multiply
//! accumulate / subtract, returning the high half. Token: [`Rdm`].
//!
//! The fixed-point Q15/Q31 multiply-accumulate that DSP code otherwise builds
//! from `SQRDMULH` plus a separate saturating add - here fused, so the add
//! happens before the single saturation. Intrinsics stable since Rust 1.62
//! (`rdm_intrinsics`). No aarch32 counterpart: 32-bit Linux exports no HWCAP
//! bit for it (see `aarch32::ops`' "Not wrapped" ledger).
//!
//! `a` is the accumulator, as in ACLE's `vqrdmlahq_s16(a, b, c)`: the result
//! is `sat((a << esize) ± 2·b·c + (1 << (esize - 1))) >> esize`.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    int16x4_t, int16x8_t, int32x2_t, int32x4_t, vld1_s16, vld1_s32, vld1q_s16, vld1q_s32,
    vqrdmlah_s16, vqrdmlah_s32, vqrdmlahq_s16, vqrdmlahq_s32, vqrdmlsh_s16, vqrdmlsh_s32,
    vqrdmlshq_s16, vqrdmlshq_s32, vst1_s16, vst1_s32, vst1q_s16, vst1q_s32,
};

arm_token! {
    /// Proof that `FEAT_RDM` (ARMv8.1-A rounding doubling multiply-accumulate)
    /// is available.
    token = Rdm, requires = [Rdm],
}

/// Accumulator-first ternop, operand order passed through verbatim.
macro_rules! rdm_ternop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $Elem:ty, $lanes:literal, $Vec:ty, $ld:path, $st:path, $intrinsic:path, $doc:literal) => {
        impl Rdm {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(
                self,
                a: [$Elem; $lanes],
                b: [$Elem; $lanes],
                c: [$Elem; $lanes],
            ) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn(&a, &b, &c) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,rdm")]
        unsafe fn $intrinsic_fn(
            a: &[$Elem; $lanes],
            b: &[$Elem; $lanes],
            c: &[$Elem; $lanes],
        ) -> [$Elem; $lanes] {
            unsafe {
                let va: $Vec = $ld(a.as_ptr());
                let vb: $Vec = $ld(b.as_ptr());
                let vc: $Vec = $ld(c.as_ptr());
                let vr = $intrinsic(va, vb, vc);
                let mut out = [0 as $Elem; $lanes];
                $st(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

rdm_ternop!(
    qrdmlah_i16x8,
    sqrdmlah_s16q_fixed,
    i16,
    8,
    int16x8_t,
    vld1q_s16,
    vst1q_s16,
    vqrdmlahq_s16,
    "Q15 multiply-accumulate: `a + round(2·b·c / 2^16)`, saturated once at the end (`sqrdmlah v.8h`)."
);
rdm_ternop!(
    qrdmlah_i32x4,
    sqrdmlah_s32q_fixed,
    i32,
    4,
    int32x4_t,
    vld1q_s32,
    vst1q_s32,
    vqrdmlahq_s32,
    "Q31 multiply-accumulate: `a + round(2·b·c / 2^32)`, saturated once at the end (`sqrdmlah v.4s`)."
);
rdm_ternop!(
    qrdmlsh_i16x8,
    sqrdmlsh_s16q_fixed,
    i16,
    8,
    int16x8_t,
    vld1q_s16,
    vst1q_s16,
    vqrdmlshq_s16,
    "Q15 multiply-subtract: `a - round(2·b·c / 2^16)`, saturated once at the end (`sqrdmlsh v.8h`)."
);
rdm_ternop!(
    qrdmlsh_i32x4,
    sqrdmlsh_s32q_fixed,
    i32,
    4,
    int32x4_t,
    vld1q_s32,
    vst1q_s32,
    vqrdmlshq_s32,
    "Q31 multiply-subtract: `a - round(2·b·c / 2^32)`, saturated once at the end (`sqrdmlsh v.4s`)."
);
rdm_ternop!(
    qrdmlah_i16x4,
    sqrdmlah_s16d_fixed,
    i16,
    4,
    int16x4_t,
    vld1_s16,
    vst1_s16,
    vqrdmlah_s16,
    "Q15 multiply-accumulate on a 64-bit D register (`sqrdmlah v.4h`)."
);
rdm_ternop!(
    qrdmlah_i32x2,
    sqrdmlah_s32d_fixed,
    i32,
    2,
    int32x2_t,
    vld1_s32,
    vst1_s32,
    vqrdmlah_s32,
    "Q31 multiply-accumulate on a 64-bit D register (`sqrdmlah v.2s`)."
);
rdm_ternop!(
    qrdmlsh_i16x4,
    sqrdmlsh_s16d_fixed,
    i16,
    4,
    int16x4_t,
    vld1_s16,
    vst1_s16,
    vqrdmlsh_s16,
    "Q15 multiply-subtract on a 64-bit D register (`sqrdmlsh v.4h`)."
);
rdm_ternop!(
    qrdmlsh_i32x2,
    sqrdmlsh_s32d_fixed,
    i32,
    2,
    int32x2_t,
    vld1_s32,
    vst1_s32,
    vqrdmlsh_s32,
    "Q31 multiply-subtract on a 64-bit D register (`sqrdmlsh v.2s`)."
);

#[cfg(test)]
#[path = "../../test/ops/neon/rdm.rs"]
mod tests;
