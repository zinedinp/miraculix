//! ARMv8.2-A `FEAT_SHA3`: the four Keccak helper instructions. Token:
//! [`Sha3`].
//!
//! These are plain bitwise ops - `EOR3`, `RAX1`, `XAR`, `BCAX` - that
//! collapse two or three instructions of a Keccak-f[1600] round into one.
//! Useful well beyond SHA-3 (`EOR3` is a three-way XOR for any data).
//! Intrinsics stable since Rust 1.79 (`stdarch_neon_sha3`). LLVM's `sha3`
//! target feature also covers SHA-512 (see [`super::sha512`]).

use super::super::macros::arm_token;
use super::super::simd_arch::{
    uint8x16_t, uint64x2_t, vbcaxq_u8, vbcaxq_u64, veor3q_u8, veor3q_u64, vld1q_u8, vld1q_u64,
    vrax1q_u64, vst1q_u8, vst1q_u64, vxarq_u64,
};

arm_token! {
    /// Proof that `FEAT_SHA3` (`EOR3`/`RAX1`/`XAR`/`BCAX`) is available.
    token = Sha3, requires = [Sha3],
}

macro_rules! sha3_ternop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $Elem:ty, $lanes:literal, $Vec:ty, $ld:path, $st:path, $intrinsic:path, $doc:literal) => {
        impl Sha3 {
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
        #[target_feature(enable = "neon,sha3")]
        unsafe fn $intrinsic_fn(
            a: &[$Elem; $lanes],
            b: &[$Elem; $lanes],
            c: &[$Elem; $lanes],
        ) -> [$Elem; $lanes] {
            unsafe {
                let (va, vb, vc): ($Vec, $Vec, $Vec) =
                    ($ld(a.as_ptr()), $ld(b.as_ptr()), $ld(c.as_ptr()));
                let mut out = [0 as $Elem; $lanes];
                $st(out.as_mut_ptr(), $intrinsic(va, vb, vc));
                out
            }
        }
    };
}

sha3_ternop!(
    eor3_u8x16,
    eor3_u8q_fixed,
    u8,
    16,
    uint8x16_t,
    vld1q_u8,
    vst1q_u8,
    veor3q_u8,
    "Three-way XOR `a ^ b ^ c` (`eor3 v.16b`)."
);
sha3_ternop!(
    eor3_u64x2,
    eor3_u64q_fixed,
    u64,
    2,
    uint64x2_t,
    vld1q_u64,
    vst1q_u64,
    veor3q_u64,
    "Three-way XOR `a ^ b ^ c` (`eor3 v.16b`, viewed as 64-bit lanes)."
);
sha3_ternop!(
    bcax_u8x16,
    bcax_u8q_fixed,
    u8,
    16,
    uint8x16_t,
    vld1q_u8,
    vst1q_u8,
    vbcaxq_u8,
    "Bit-clear and XOR: `a ^ (b & !c)` - Keccak's χ step in one instruction (`bcax v.16b`)."
);
sha3_ternop!(
    bcax_u64x2,
    bcax_u64q_fixed,
    u64,
    2,
    uint64x2_t,
    vld1q_u64,
    vst1q_u64,
    vbcaxq_u64,
    "Bit-clear and XOR: `a ^ (b & !c)` on 64-bit lanes (`bcax v.16b`)."
);

impl Sha3 {
    /// Rotate and XOR: `a ^ b.rotate_left(1)` per 64-bit lane - Keccak's θ
    /// column parity step (`rax1 v.2d`).
    #[inline]
    pub fn rax1_u64x2(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        unsafe { rax1_u64q_fixed(&a, &b) }
    }

    /// XOR and rotate: `(a ^ b).rotate_right(IMM)` per 64-bit lane, `IMM` in
    /// `0..64` - Keccak's ρ step fused with the preceding XOR (`xar v.2d`).
    #[inline]
    pub fn xar_u64x2<const IMM: i32>(self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        unsafe { xar_u64q_fixed::<IMM>(&a, &b) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sha3")]
unsafe fn rax1_u64q_fixed(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
    unsafe {
        let mut out = [0u64; 2];
        vst1q_u64(
            out.as_mut_ptr(),
            vrax1q_u64(vld1q_u64(a.as_ptr()), vld1q_u64(b.as_ptr())),
        );
        out
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sha3")]
unsafe fn xar_u64q_fixed<const IMM: i32>(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
    unsafe {
        let mut out = [0u64; 2];
        vst1q_u64(
            out.as_mut_ptr(),
            vxarq_u64::<IMM>(vld1q_u64(a.as_ptr()), vld1q_u64(b.as_ptr())),
        );
        out
    }
}

#[cfg(test)]
#[path = "../../test/ops/crypto/sha3.rs"]
mod tests;
