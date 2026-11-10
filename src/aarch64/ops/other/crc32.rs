//! ARMv8.0-A `FEAT_CRC32` (mandatory from v8.1): CRC-32 (ISO-HDLC, the zlib /
//! Ethernet polynomial) and CRC-32C (Castagnoli) over 1/2/4/8 bytes per
//! instruction. Token: [`Crc32`].
//!
//! Same method names as `aarch32::ops::crc32`, plus whole-buffer
//! [`Crc32::crc32_slice`] / [`Crc32::crc32c_slice`] that fold 8 bytes per
//! step. The raw instructions neither pre- nor post-invert; the slice forms
//! do not either, so the standard check value is
//! `!crc32_slice(!0, b"123456789") == 0xCBF43926`.
//!
//! Intrinsics stable since Rust 1.80 (`stdarch_aarch64_crc32`).

use super::super::macros::arm_token;
use super::super::simd_arch::{
    __crc32b, __crc32cb, __crc32cd, __crc32ch, __crc32cw, __crc32d, __crc32h, __crc32w,
};

arm_token! {
    /// Proof that `FEAT_CRC32` (`CRC32*`/`CRC32C*`) is available.
    token = Crc32, requires = [Crc32],
}

macro_rules! crc_step {
    ($fixed_fn:ident, $intrinsic_fn:ident, $T:ty, $intrinsic:path, $doc:literal) => {
        impl Crc32 {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, crc: u32, data: $T) -> u32 {
                unsafe { $intrinsic_fn(crc, data) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "crc")]
        unsafe fn $intrinsic_fn(crc: u32, data: $T) -> u32 {
            $intrinsic(crc, data)
        }
    };
}

crc_step!(
    crc32b,
    crc32b_fixed,
    u8,
    __crc32b,
    "CRC-32 over one byte (`crc32b`)."
);
crc_step!(
    crc32h,
    crc32h_fixed,
    u16,
    __crc32h,
    "CRC-32 over two bytes, little-endian (`crc32h`)."
);
crc_step!(
    crc32w,
    crc32w_fixed,
    u32,
    __crc32w,
    "CRC-32 over four bytes, little-endian (`crc32w`)."
);
crc_step!(
    crc32d,
    crc32d_fixed,
    u64,
    __crc32d,
    "CRC-32 over eight bytes, little-endian (`crc32x`)."
);
crc_step!(
    crc32cb,
    crc32cb_fixed,
    u8,
    __crc32cb,
    "CRC-32C over one byte (`crc32cb`)."
);
crc_step!(
    crc32ch,
    crc32ch_fixed,
    u16,
    __crc32ch,
    "CRC-32C over two bytes, little-endian (`crc32ch`)."
);
crc_step!(
    crc32cw,
    crc32cw_fixed,
    u32,
    __crc32cw,
    "CRC-32C over four bytes, little-endian (`crc32cw`)."
);
crc_step!(
    crc32cd,
    crc32cd_fixed,
    u64,
    __crc32cd,
    "CRC-32C over eight bytes, little-endian (`crc32cx`)."
);

/// Whole-buffer fold, 8 bytes per instruction, then the byte tail. The loop
/// lives inside the `#[target_feature]` fn so the per-step call inlines.
macro_rules! crc_slice {
    ($slice_fn:ident, $impl_fn:ident, $d:path, $b:path, $doc:literal) => {
        impl Crc32 {
            #[doc = $doc]
            pub fn $slice_fn(self, crc: u32, data: &[u8]) -> u32 {
                unsafe { $impl_fn(crc, data) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[target_feature(enable = "crc")]
        unsafe fn $impl_fn(mut crc: u32, data: &[u8]) -> u32 {
            let mut chunks = data.chunks_exact(8);
            for c in chunks.by_ref() {
                crc = $d(crc, u64::from_le_bytes(c.try_into().unwrap()));
            }
            for &byte in chunks.remainder() {
                crc = $b(crc, byte);
            }
            crc
        }
    };
}

crc_slice!(
    crc32_slice,
    crc32_slice_impl,
    __crc32d,
    __crc32b,
    "CRC-32 over a whole buffer, no pre/post inversion (`crc32x` per 8 bytes, `crc32b` for the tail)."
);
crc_slice!(
    crc32c_slice,
    crc32c_slice_impl,
    __crc32cd,
    __crc32cb,
    "CRC-32C over a whole buffer, no pre/post inversion (`crc32cx` per 8 bytes, `crc32cb` for the tail)."
);

#[cfg(test)]
#[path = "../../test/ops/other/crc32.rs"]
mod tests;
