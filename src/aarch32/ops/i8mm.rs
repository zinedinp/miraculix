//! ARMv8.6-A Int8 Matrix Multiply: mixed-sign `USDOT` plus 8-bit 2x2 block
//! matmul-accumulate (`SMMLA`/`UMMLA`/`USMMLA`). Token: [`I8mm`]. Detect:
//! [`Feature::I8mm`](super::super::Feature::I8mm). `MMLA`: `a`/`b` are two
//! packed 8-byte rows each; the 4 accumulator lanes are a row-major 2x2 of
//! 8-element dots (not a Dotprod shape despite the shared `(acc,a,b)->acc`
//! call form, and the shared `neon_dot_acc!` macro).

use crate::arm_shared::macros::{arm_token, neon_dot_acc};
use crate::arm_shared::simd_arch::{
    int8x16_t, int32x4_t, uint8x16_t, uint32x4_t, vld1q_s8, vld1q_s32, vld1q_u8, vld1q_u32,
    vmmlaq_s32, vmmlaq_u32, vst1q_s32, vst1q_u32, vusdotq_s32, vusmmlaq_s32,
};

arm_token! {
    /// Proof that the Int8 Matrix Multiply extension is available.
    token = I8mm, requires = [Neon, I8mm],
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,v7,v8,i8mm",
    fixed_fn = dot_us32, intrinsic_fn = dot_us32_imp, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_u8, b_loadu = vld1q_s8,
    intrinsic = vusdotq_s32,
    fixed_doc = "`VUSDOT.S8`: mixed-sign 8-bit dot (`a` unsigned, `b` signed) into `i32` lanes. Gated by `Feature::I8mm`, not `Feature::Dotprod` (stdarch attribute; see [`super::dotprod`]).",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,v7,v8,i8mm",
    fixed_fn = mmla_s32, intrinsic_fn = mmla_s32_imp, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = i8, a_vec = int8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_s8, b_loadu = vld1q_s8,
    intrinsic = vmmlaq_s32,
    fixed_doc = "`VSMMLA.S8`: signed 8-bit 2x2 block matmul-accumulate (module doc).",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,v7,v8,i8mm",
    fixed_fn = mmla_u32, intrinsic_fn = mmla_u32_imp, in_lanes = 16,
    acc = u32, acc_vec = uint32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = u8, b_vec = uint8x16_t,
    acc_loadu = vld1q_u32, acc_storeu = vst1q_u32, a_loadu = vld1q_u8, b_loadu = vld1q_u8,
    intrinsic = vmmlaq_u32,
    fixed_doc = "`VUMMLA.U8`: unsigned 8-bit 2x2 block matmul-accumulate; same lane layout as [`I8mm::mmla_s32`].",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,v7,v8,i8mm",
    fixed_fn = mmla_us32, intrinsic_fn = mmla_us32_imp, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_u8, b_loadu = vld1q_s8,
    intrinsic = vusmmlaq_s32,
    fixed_doc = "`VUSMMLA.S8`: mixed-sign (`a` unsigned, `b` signed) 2x2 block matmul-accumulate; same lane layout as [`I8mm::mmla_s32`].",
}

#[cfg(test)]
#[path = "../test/ops/i8mm.rs"]
mod tests;
