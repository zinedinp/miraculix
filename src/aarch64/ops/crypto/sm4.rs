//! ARMv8.2-A `FEAT_SM4`: Chinese national-standard SM4 block cipher. Token:
//! [`Sm4`].
//!
//! `SM4E` runs **four** SM4 rounds on the state words `X0..X3` (lane 0 =
//! `X0`) with round keys `rk0..rk3`, giving `X4..X7`:
//! `X[i+4] = X[i] ^ T(X[i+1] ^ X[i+2] ^ X[i+3] ^ rk[i])`. `SM4EKEY` is the
//! same unbalanced-Feistel step on the key schedule with `T'` and the `CK`
//! constants.
//!
//! **Why `asm!`:** intrinsics unstable (`stdarch_neon_sm4`); the `sm4` target
//! feature is stable.

use core::arch::asm;

use super::super::macros::arm_token;
use super::super::simd_arch::{uint32x4_t, vld1q_u32, vst1q_u32};

arm_token! {
    /// Proof that `FEAT_SM4` (`SM4E`/`SM4EKEY`) is available.
    token = Sm4, requires = [Sm4],
}

impl Sm4 {
    /// Four SM4 encryption rounds on `state` with `round_keys`
    /// (`sm4e v.4s, v.4s`). Decryption is the same op with the words of
    /// both operands reversed.
    #[inline]
    pub fn sm4e(self, state: [u32; 4], round_keys: [u32; 4]) -> [u32; 4] {
        unsafe { sm4e_fixed(&state, &round_keys) }
    }

    /// Four steps of the SM4 key schedule: next four round keys from the
    /// previous four and the `CK` constants (`sm4ekey v.4s, v.4s, v.4s`).
    #[inline]
    pub fn sm4ekey(self, keys: [u32; 4], ck: [u32; 4]) -> [u32; 4] {
        unsafe { sm4ekey_fixed(&keys, &ck) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sm4")]
unsafe fn sm4e_fixed(state: &[u32; 4], rk: &[u32; 4]) -> [u32; 4] {
    unsafe {
        let mut d: uint32x4_t = vld1q_u32(state.as_ptr());
        asm!(
            "sm4e {d:v}.4s, {k:v}.4s",
            d = inout(vreg) d,
            k = in(vreg) vld1q_u32(rk.as_ptr()),
            options(pure, nomem, nostack, preserves_flags),
        );
        let mut out = [0u32; 4];
        vst1q_u32(out.as_mut_ptr(), d);
        out
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sm4")]
unsafe fn sm4ekey_fixed(keys: &[u32; 4], ck: &[u32; 4]) -> [u32; 4] {
    unsafe {
        let r: uint32x4_t;
        asm!(
            "sm4ekey {r:v}.4s, {k:v}.4s, {c:v}.4s",
            r = lateout(vreg) r,
            k = in(vreg) vld1q_u32(keys.as_ptr()),
            c = in(vreg) vld1q_u32(ck.as_ptr()),
            options(pure, nomem, nostack, preserves_flags),
        );
        let mut out = [0u32; 4];
        vst1q_u32(out.as_mut_ptr(), r);
        out
    }
}

#[cfg(test)]
#[path = "../../test/ops/crypto/sm4.rs"]
mod tests;
