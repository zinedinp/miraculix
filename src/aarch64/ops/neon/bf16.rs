//! ARMv8.6-A `FEAT_BF16`: bfloat16 dot product, matrix multiply,
//! widening multiply-add, and `f32` → `bf16` narrowing. Token: [`Bf16`].
//!
//! `bf16` values cross the API as bit patterns (`[u16; N]`): the top 16 bits
//! of an `f32`. Two rounding regimes, and the difference matters:
//!
//! - `BFMLALB`/`BFMLALT` and `BFCVTN` are ordinary IEEE operations
//!   (fused, round-to-nearest-even).
//! - `BFDOT` and `BFMMLA` are **not**: without `FEAT_EBF16` they round the
//!   intermediate products and sums to odd and flush denormals. They are
//!   built for ML throughput, and only results that are exact (small
//!   integers, as in this module's tests) are bit-reproducible against a
//!   plain `f32` computation.
//!
//! **Why `asm!`:** no BF16 intrinsics are exported by `core::arch::aarch64`
//! at all in the installed toolchain; the `bf16` target feature is stable.

use core::arch::asm;

use super::super::macros::{arm_token, neon_dot_acc};
use super::super::simd_arch::{float32x4_t, uint16x8_t, vld1q_f32, vld1q_u16, vst1q_f32};

arm_token! {
    /// Proof that `FEAT_BF16` (`BFDOT`/`BFMMLA`/`BFMLALB`/`BFMLALT`/`BFCVTN`)
    /// is available.
    token = Bf16, requires = [Bf16],
}

neon_dot_acc! {
    token = Bf16, target_feature = "neon,bf16",
    fixed_fn = bfdot_f32x4, intrinsic_fn = bfdot_f32q_fixed, in_lanes = 8,
    acc = f32, acc_vec = float32x4_t, a_elem = u16, a_vec = uint16x8_t, b_elem = u16, b_vec = uint16x8_t,
    acc_loadu = vld1q_f32, acc_storeu = vst1q_f32, a_loadu = vld1q_u16, b_loadu = vld1q_u16,
    asm = "bfdot {acc:v}.4s, {a:v}.8h, {b:v}.8h",
    fixed_doc = "2-way `bf16` dot product accumulate: `acc[i] += a[2i]·b[2i] + a[2i+1]·b[2i+1]` (`bfdot v.4s, v.8h, v.8h`). Non-IEEE intermediate rounding, see the module doc.",
}

neon_dot_acc! {
    token = Bf16, target_feature = "neon,bf16",
    fixed_fn = bfmmla_f32x4, intrinsic_fn = bfmmla_f32q_fixed, in_lanes = 8,
    acc = f32, acc_vec = float32x4_t, a_elem = u16, a_vec = uint16x8_t, b_elem = u16, b_vec = uint16x8_t,
    acc_loadu = vld1q_f32, acc_storeu = vst1q_f32, a_loadu = vld1q_u16, b_loadu = vld1q_u16,
    asm = "bfmmla {acc:v}.4s, {a:v}.8h, {b:v}.8h",
    fixed_doc = "`bf16` 2×4 · 4×2 matrix multiply-accumulate, row-major like `I8mm::mmla_s32`: `acc[2i+j] += Σₖ a[4i+k]·b[4j+k]` (`bfmmla v.4s, v.8h, v.8h`). Non-IEEE intermediate rounding.",
}

neon_dot_acc! {
    token = Bf16, target_feature = "neon,bf16",
    fixed_fn = bfmlalb_f32x4, intrinsic_fn = bfmlalb_f32q_fixed, in_lanes = 8,
    acc = f32, acc_vec = float32x4_t, a_elem = u16, a_vec = uint16x8_t, b_elem = u16, b_vec = uint16x8_t,
    acc_loadu = vld1q_f32, acc_storeu = vst1q_f32, a_loadu = vld1q_u16, b_loadu = vld1q_u16,
    asm = "bfmlalb {acc:v}.4s, {a:v}.8h, {b:v}.8h",
    fixed_doc = "Widening fused multiply-add on the **even** `bf16` lanes: `acc[i] + a[2i]·b[2i]`, one IEEE rounding (`bfmlalb v.4s, v.8h, v.8h`).",
}

neon_dot_acc! {
    token = Bf16, target_feature = "neon,bf16",
    fixed_fn = bfmlalt_f32x4, intrinsic_fn = bfmlalt_f32q_fixed, in_lanes = 8,
    acc = f32, acc_vec = float32x4_t, a_elem = u16, a_vec = uint16x8_t, b_elem = u16, b_vec = uint16x8_t,
    acc_loadu = vld1q_f32, acc_storeu = vst1q_f32, a_loadu = vld1q_u16, b_loadu = vld1q_u16,
    asm = "bfmlalt {acc:v}.4s, {a:v}.8h, {b:v}.8h",
    fixed_doc = "Widening fused multiply-add on the **odd** `bf16` lanes: `acc[i] + a[2i+1]·b[2i+1]`, one IEEE rounding (`bfmlalt v.4s, v.8h, v.8h`).",
}

impl Bf16 {
    /// Narrow four `f32` lanes to `bf16` bit patterns, round-to-nearest-even
    /// (`bfcvtn v.4h, v.4s`). Unlike truncating the low 16 bits, this rounds.
    #[inline]
    pub fn cvt_f32x4_to_bf16x4(self, a: [f32; 4]) -> [u16; 4] {
        unsafe { bfcvtn_f32q_fixed(&a) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,bf16")]
unsafe fn bfcvtn_f32q_fixed(a: &[f32; 4]) -> [u16; 4] {
    unsafe {
        let va: float32x4_t = vld1q_f32(a.as_ptr());
        let r: u64;
        // BFCVTN writes the low 64 bits of the destination; read them as one GPR.
        asm!(
            "bfcvtn {t:v}.4h, {a:v}.4s",
            "fmov {r}, {t:d}",
            a = in(vreg) va,
            t = out(vreg) _,
            r = lateout(reg) r,
            options(pure, nomem, nostack, preserves_flags),
        );
        core::array::from_fn(|i| (r >> (16 * i)) as u16)
    }
}

#[cfg(test)]
#[path = "../../test/ops/neon/bf16.rs"]
mod tests;
