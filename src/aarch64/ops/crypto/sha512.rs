//! ARMv8.2-A `FEAT_SHA512`: SHA-512 compression and message-schedule steps.
//! Token: [`Sha512`].
//!
//! Its own HWCAP bit, but LLVM gates it behind the `sha3` target feature -
//! hence `target_feature(enable = "neon,sha3")` on a token that checks
//! `Feature::Sha512`. Intrinsics stable since Rust 1.79 (`stdarch_neon_sha3`).
//! Each `sha512h`/`sha512h2` pair performs **two** rounds on 64-bit words.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    uint64x2_t, vld1q_u64, vsha512h2q_u64, vsha512hq_u64, vsha512su0q_u64, vsha512su1q_u64,
    vst1q_u64,
};

arm_token! {
    /// Proof that `FEAT_SHA512` (`SHA512H`/`SHA512H2`/`SHA512SU0`/`SHA512SU1`)
    /// is available.
    token = Sha512, requires = [Sha512],
}

#[inline(always)]
unsafe fn ld(a: &[u64; 2]) -> uint64x2_t {
    unsafe { vld1q_u64(a.as_ptr()) }
}

#[inline(always)]
unsafe fn st(v: uint64x2_t) -> [u64; 2] {
    let mut out = [0u64; 2];
    unsafe { vst1q_u64(out.as_mut_ptr(), v) };
    out
}

macro_rules! sha512_ternop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $a:ident, $b:ident, $c:ident, $doc:literal) => {
        impl Sha512 {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, $a: [u64; 2], $b: [u64; 2], $c: [u64; 2]) -> [u64; 2] {
                unsafe { $intrinsic_fn(&$a, &$b, &$c) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,sha3")]
        unsafe fn $intrinsic_fn($a: &[u64; 2], $b: &[u64; 2], $c: &[u64; 2]) -> [u64; 2] {
            unsafe { st($intrinsic(ld($a), ld($b), ld($c))) }
        }
    };
}

sha512_ternop!(
    sha512h,
    sha512h_fixed,
    vsha512hq_u64,
    hash_ed,
    hash_gf,
    kwh_kwh2,
    "SHA-512 two rounds, first half (`sha512h q, q, v.2d`). Operand roles follow ACLE's `vsha512hq_u64`."
);
sha512_ternop!(
    sha512h2,
    sha512h2_fixed,
    vsha512h2q_u64,
    sum_ab,
    hash_c_,
    hash_ab,
    "SHA-512 two rounds, second half (`sha512h2 q, q, v.2d`). Operand roles follow ACLE's `vsha512h2q_u64`."
);
sha512_ternop!(
    sha512su1,
    sha512su1_fixed,
    vsha512su1q_u64,
    s01_s02,
    w14_15,
    w9_10,
    "SHA-512 message schedule update, part 2 of 2: `s01_s02 + σ1(w14_15) + w9_10` per lane (`sha512su1 v.2d`)."
);

impl Sha512 {
    /// SHA-512 message schedule update, part 1 of 2: `w0_1 + σ0(w1_2)` per
    /// lane, where `w1_2` is the upper word of `w0_1` and the lower word of
    /// `w2_3` (`sha512su0 v.2d, v.2d`).
    #[inline]
    pub fn sha512su0(self, w0_1: [u64; 2], w2_3: [u64; 2]) -> [u64; 2] {
        unsafe { sha512su0_fixed(&w0_1, &w2_3) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sha3")]
unsafe fn sha512su0_fixed(w0_1: &[u64; 2], w2_3: &[u64; 2]) -> [u64; 2] {
    unsafe { st(vsha512su0q_u64(ld(w0_1), ld(w2_3))) }
}

#[cfg(test)]
#[path = "../../test/ops/crypto/sha512.rs"]
mod tests;
