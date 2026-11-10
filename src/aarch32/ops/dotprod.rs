//! ARMv8.2-A Dot Product: 16-lane i8/u8 sum-of-4-products into 4-lane
//! i32/u32. Token: [`Dotprod`]. Upstream: `vdotq_s32`/`vdotq_u32`. Detect:
//! [`Feature::Dotprod`](super::super::Feature::Dotprod) (`ASIMDDP`). Lane `i`:
//! `acc[i] + sum(a[4*i+k]*b[4*i+k] for k in 0..4)`. `USDOT` lives in
//! [`super::i8mm`] (needs `I8mm`, not Dotprod). No plain vector `SUDOT` in
//! stdarch (lane forms only). Same method names and macro as
//! `aarch64::ops::neon::dotprod` (shared DSL, `crate::arm_shared`).

use crate::arm_shared::macros::{arm_token, neon_dot_acc};
use crate::arm_shared::simd_arch::{
    int8x16_t, int32x4_t, uint8x16_t, uint32x4_t, vdotq_s32, vdotq_u32, vld1q_s8, vld1q_s32,
    vld1q_u8, vld1q_u32, vst1q_s32, vst1q_u32,
};

arm_token! {
    /// Proof that the Dot Product extension is available.
    token = Dotprod, requires = [Neon, Dotprod],
}

neon_dot_acc! {
    token = Dotprod, target_feature = "neon,v7,v8,dotprod",
    fixed_fn = dot_s32, intrinsic_fn = dot_s32_imp, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = i8, a_vec = int8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_s8, b_loadu = vld1q_s8,
    intrinsic = vdotq_s32,
    fixed_doc = "`VSDOT.S8`: signed 8-bit dot product, accumulated into `i32` lanes.",
}

neon_dot_acc! {
    token = Dotprod, target_feature = "neon,v7,v8,dotprod",
    fixed_fn = dot_u32, intrinsic_fn = dot_u32_imp, in_lanes = 16,
    acc = u32, acc_vec = uint32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = u8, b_vec = uint8x16_t,
    acc_loadu = vld1q_u32, acc_storeu = vst1q_u32, a_loadu = vld1q_u8, b_loadu = vld1q_u8,
    intrinsic = vdotq_u32,
    fixed_doc = "`VUDOT.U8`: unsigned 8-bit dot product, accumulated into `u32` lanes.",
}

#[cfg(test)]
#[path = "../test/ops/dotprod.rs"]
mod tests;
