//! ARMv8.6-A `FEAT_I8MM`: 8-bit integer matrix multiply-accumulate and the
//! mixed-sign dot product. Token: [`I8mm`].
//!
//! The `mmla_*` ops treat each 16-byte operand as a **2×8 matrix, row-major**:
//! `a` is the left matrix `A`, `b` holds the rows of `Bᵀ`, and the 2×2 result
//! `A·Bᵀ` is added to `acc` (row-major, `acc[2i + j] += Σₖ a[8i+k]·b[8j+k]`).
//!
//! **Why `asm!`:** the I8MM intrinsics are still unstable
//! (`stdarch_neon_i8mm`) while the `i8mm` target feature is stable - same
//! arrangement as [`super::dotprod`]. Method names match
//! `aarch32::ops::i8mm`.

use super::super::macros::{arm_token, neon_dot_acc};
use super::super::simd_arch::{
    int8x16_t, int32x4_t, uint8x16_t, uint32x4_t, vld1q_s8, vld1q_s32, vld1q_u8, vld1q_u32,
    vst1q_s32, vst1q_u32,
};

arm_token! {
    /// Proof that `FEAT_I8MM` (`SMMLA`/`UMMLA`/`USMMLA`/`USDOT`) is available.
    token = I8mm, requires = [I8mm],
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,i8mm",
    fixed_fn = mmla_s32, intrinsic_fn = smmla_s32q_fixed, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = i8, a_vec = int8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_s8, b_loadu = vld1q_s8,
    asm = "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
    fixed_doc = "Signed 2×8 · 8×2 matrix multiply-accumulate: `acc[2i+j] += Σₖ a[8i+k]·b[8j+k]` (`smmla v.4s, v.16b, v.16b`).",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,i8mm",
    fixed_fn = mmla_u32, intrinsic_fn = ummla_u32q_fixed, in_lanes = 16,
    acc = u32, acc_vec = uint32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = u8, b_vec = uint8x16_t,
    acc_loadu = vld1q_u32, acc_storeu = vst1q_u32, a_loadu = vld1q_u8, b_loadu = vld1q_u8,
    asm = "ummla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
    fixed_doc = "Unsigned 2×8 · 8×2 matrix multiply-accumulate (`ummla v.4s, v.16b, v.16b`).",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,i8mm",
    fixed_fn = mmla_us32, intrinsic_fn = usmmla_s32q_fixed, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_u8, b_loadu = vld1q_s8,
    asm = "usmmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
    fixed_doc = "Mixed-sign matrix multiply-accumulate: unsigned `a`, signed `b`, signed `i32` result (`usmmla v.4s, v.16b, v.16b`).",
}

neon_dot_acc! {
    token = I8mm, target_feature = "neon,i8mm",
    fixed_fn = dot_us32, intrinsic_fn = usdot_s32q_fixed, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_u8, b_loadu = vld1q_s8,
    asm = "usdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
    fixed_doc = "Mixed-sign 4-way dot product accumulate: unsigned `a`, signed `b` (`usdot v.4s, v.16b, v.16b`). The shape of `Dotprod::dot_s32`, for quantised activations × signed weights.",
}

#[cfg(test)]
#[path = "../../test/ops/neon/i8mm.rs"]
mod tests;
