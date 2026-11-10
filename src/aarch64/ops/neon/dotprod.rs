//! ARMv8.2-A `FEAT_DotProd` (mandatory from v8.4): 4-way 8-bit dot product
//! into 32-bit lanes. Token: [`Dotprod`].
//!
//! `acc[i] += a[4i]·b[4i] + a[4i+1]·b[4i+1] + a[4i+2]·b[4i+2] + a[4i+3]·b[4i+3]`,
//! products widened, sum wrapping in 32 bits.
//!
//! Intrinsics `vdotq_s32`/`vdotq_u32`, stable since Rust 1.98
//! (`stdarch_neon_dotprod`) - the release that sets this crate's MSRV. Until
//! then this file emitted the instruction through `asm!`; the API did not
//! change with the switch.
//!
//! Same method names as `aarch32::ops::dotprod` (`dot_s32`/`dot_u32`); fixed
//! width only, since the op reduces across lanes (no `auto` rung, by the same
//! policy as x86 and aarch32).

use super::super::macros::{arm_token, neon_dot_acc};
use super::super::simd_arch::{
    int8x16_t, int32x4_t, uint8x16_t, uint32x4_t, vdotq_s32, vdotq_u32, vld1q_s8, vld1q_s32,
    vld1q_u8, vld1q_u32, vst1q_s32, vst1q_u32,
};

arm_token! {
    /// Proof that `FEAT_DotProd` (`SDOT`/`UDOT`) is available.
    token = Dotprod, requires = [Dotprod],
}

neon_dot_acc! {
    token = Dotprod, target_feature = "neon,dotprod",
    fixed_fn = dot_s32, intrinsic_fn = sdot_s32q_fixed, in_lanes = 16,
    acc = i32, acc_vec = int32x4_t, a_elem = i8, a_vec = int8x16_t, b_elem = i8, b_vec = int8x16_t,
    acc_loadu = vld1q_s32, acc_storeu = vst1q_s32, a_loadu = vld1q_s8, b_loadu = vld1q_s8,
    intrinsic = vdotq_s32,
    fixed_doc = "Signed 8-bit dot product accumulate: each `i32` lane of `acc` gains the sum of four `i8 × i8` products (`sdot v.4s, v.16b, v.16b`).",
}

neon_dot_acc! {
    token = Dotprod, target_feature = "neon,dotprod",
    fixed_fn = dot_u32, intrinsic_fn = udot_u32q_fixed, in_lanes = 16,
    acc = u32, acc_vec = uint32x4_t, a_elem = u8, a_vec = uint8x16_t, b_elem = u8, b_vec = uint8x16_t,
    acc_loadu = vld1q_u32, acc_storeu = vst1q_u32, a_loadu = vld1q_u8, b_loadu = vld1q_u8,
    intrinsic = vdotq_u32,
    fixed_doc = "Unsigned 8-bit dot product accumulate (`udot v.4s, v.16b, v.16b`).",
}

#[cfg(test)]
#[path = "../../test/ops/neon/dotprod.rs"]
mod tests;
