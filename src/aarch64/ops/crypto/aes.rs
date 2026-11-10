//! ARMv8-A `FEAT_AES`: one AES round, split the way the hardware splits it.
//! Token: [`Aes`].
//!
//! `AESE` is `AddRoundKey` + `SubBytes` + `ShiftRows`; `AESMC` is
//! `MixColumns`. A full encryption round is `aesmc(aese(state, key))`, and
//! the last round skips `aesmc`. Decryption mirrors it with `AESD`/`AESIMC`.
//! Intrinsics stable since Rust 1.72 (`aarch64_neon_crypto_intrinsics`).
//! Same method names and shapes as `aarch32::ops::crypto::Aes`.
//!
//! `PMULL` (carry-less multiply, used by AES-GCM) has its own token in
//! [`super::pmull`]: Linux reports it as a separate HWCAP bit.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    uint8x16_t, vaesdq_u8, vaeseq_u8, vaesimcq_u8, vaesmcq_u8, vld1q_u8, vst1q_u8,
};

arm_token! {
    /// Proof that `FEAT_AES` (`AESE`/`AESD`/`AESMC`/`AESIMC`) is available.
    token = Aes, requires = [Aes],
}

macro_rules! aes_binop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Aes {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, data: [u8; 16], key: [u8; 16]) -> [u8; 16] {
                unsafe { $intrinsic_fn(&data, &key) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,aes")]
        unsafe fn $intrinsic_fn(data: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
            unsafe {
                let vd: uint8x16_t = vld1q_u8(data.as_ptr());
                let vk: uint8x16_t = vld1q_u8(key.as_ptr());
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), $intrinsic(vd, vk));
                out
            }
        }
    };
}

macro_rules! aes_unop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Aes {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, data: [u8; 16]) -> [u8; 16] {
                unsafe { $intrinsic_fn(&data) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,aes")]
        unsafe fn $intrinsic_fn(data: &[u8; 16]) -> [u8; 16] {
            unsafe {
                let vd: uint8x16_t = vld1q_u8(data.as_ptr());
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), $intrinsic(vd));
                out
            }
        }
    };
}

aes_binop!(
    aese,
    aese_fixed,
    vaeseq_u8,
    "AES encryption step: `SubBytes(ShiftRows(data ^ key))` - no `MixColumns`, chain [`Aes::aesmc`] for a full round (`aese v.16b`)."
);
aes_binop!(
    aesd,
    aesd_fixed,
    vaesdq_u8,
    "AES decryption step: `InvSubBytes(InvShiftRows(data ^ key))` (`aesd v.16b`)."
);
aes_unop!(
    aesmc,
    aesmc_fixed,
    vaesmcq_u8,
    "AES `MixColumns` (`aesmc v.16b`)."
);
aes_unop!(
    aesimc,
    aesimc_fixed,
    vaesimcq_u8,
    "AES `InvMixColumns` (`aesimc v.16b`)."
);

#[cfg(test)]
#[path = "../../test/ops/crypto/aes.rs"]
mod tests;
