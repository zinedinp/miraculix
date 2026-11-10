//! ARMv8-A `FEAT_PMULL`: 64 × 64 → 128-bit carry-less (polynomial over
//! GF(2)) multiply. Token: [`Pmull`].
//!
//! The GHASH / CRC-folding primitive, x86's `pclmulqdq`. Separate token from
//! [`super::aes::Aes`] because Linux reports it as its own HWCAP bit, even
//! though LLVM gates both behind the single `aes` target feature.
//! `vmull_p64` is stable since Rust 1.59 (`neon_intrinsics`). aarch32 lists
//! `Feature::Pmull` as detect-only (no `vmull_p64` in `core::arch::arm`).

use super::super::macros::arm_token;
use super::super::simd_arch::{vld1q_p64, vmull_high_p64, vmull_p64};

arm_token! {
    /// Proof that `FEAT_PMULL` (`PMULL`/`PMULL2` on 64-bit lanes) is available.
    token = Pmull, requires = [Pmull],
}

impl Pmull {
    /// Carry-less multiply of two 64-bit polynomials into a 128-bit product
    /// (`pmull v.1q, v.1d, v.1d`). x86 counterpart: `pclmulqdq` with
    /// immediate `0x00`.
    #[inline]
    pub fn pmull_u64(self, a: u64, b: u64) -> u128 {
        unsafe { pmull_fixed(a, b) }
    }

    /// Carry-less multiply of the **high** 64-bit lanes of `a` and `b`
    /// (`pmull2 v.1q, v.2d, v.2d`); x86 `pclmulqdq` immediate `0x11`.
    #[inline]
    pub fn pmull_high_u64x2(self, a: [u64; 2], b: [u64; 2]) -> u128 {
        unsafe { pmull2_fixed(&a, &b) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull_fixed(a: u64, b: u64) -> u128 {
    vmull_p64(a, b)
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull2_fixed(a: &[u64; 2], b: &[u64; 2]) -> u128 {
    unsafe { vmull_high_p64(vld1q_p64(a.as_ptr()), vld1q_p64(b.as_ptr())) }
}

#[cfg(test)]
#[path = "../../test/ops/crypto/pmull.rs"]
mod tests;
