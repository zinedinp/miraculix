//! ARMv8-A `FEAT_SHA1` + `FEAT_SHA256`: SHA-1 and SHA-256 compression and
//! message-schedule steps. Token: [`Sha`].
//!
//! One token for both hashes, as on aarch32: LLVM's single `sha2` target
//! feature covers `SHA1*` and `SHA256*`, so [`Sha::from_features`] requires
//! both HWCAP bits. Intrinsics stable since Rust 1.72.
//!
//! Each `*h`/`sha1{c,p,m}` call performs **four** rounds on a 4-word state
//! half; `*su0`/`*su1` together produce the next four schedule words. The
//! tests drive complete SHA-1 and SHA-256 compressions through these and
//! check the FIPS 180-4 `"abc"` digests.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    uint32x4_t, vld1q_u32, vsha1cq_u32, vsha1h_u32, vsha1mq_u32, vsha1pq_u32, vsha1su0q_u32,
    vsha1su1q_u32, vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
};

arm_token! {
    /// Proof that `FEAT_SHA1` and `FEAT_SHA256` are both available.
    token = Sha, requires = [Sha1, Sha2],
}

#[inline(always)]
unsafe fn ld(a: &[u32; 4]) -> uint32x4_t {
    unsafe { vld1q_u32(a.as_ptr()) }
}

#[inline(always)]
unsafe fn st(v: uint32x4_t) -> [u32; 4] {
    let mut out = [0u32; 4];
    unsafe { vst1q_u32(out.as_mut_ptr(), v) };
    out
}

/// SHA-1 four-round step: `hash_abcd`, the fifth state word `hash_e`, and the
/// round inputs `wk = w + K`.
macro_rules! sha1_round {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Sha {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
                unsafe { $intrinsic_fn(&hash_abcd, hash_e, &wk) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,sha2")]
        unsafe fn $intrinsic_fn(hash_abcd: &[u32; 4], hash_e: u32, wk: &[u32; 4]) -> [u32; 4] {
            unsafe { st($intrinsic(ld(hash_abcd), hash_e, ld(wk))) }
        }
    };
}

/// Three 4-word vectors in, one out.
macro_rules! sha_ternop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $a:ident, $b:ident, $c:ident, $doc:literal) => {
        impl Sha {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, $a: [u32; 4], $b: [u32; 4], $c: [u32; 4]) -> [u32; 4] {
                unsafe { $intrinsic_fn(&$a, &$b, &$c) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,sha2")]
        unsafe fn $intrinsic_fn($a: &[u32; 4], $b: &[u32; 4], $c: &[u32; 4]) -> [u32; 4] {
            unsafe { st($intrinsic(ld($a), ld($b), ld($c))) }
        }
    };
}

/// Two 4-word vectors in, one out.
macro_rules! sha_binop {
    ($fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $a:ident, $b:ident, $doc:literal) => {
        impl Sha {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, $a: [u32; 4], $b: [u32; 4]) -> [u32; 4] {
                unsafe { $intrinsic_fn(&$a, &$b) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,sha2")]
        unsafe fn $intrinsic_fn($a: &[u32; 4], $b: &[u32; 4]) -> [u32; 4] {
            unsafe { st($intrinsic(ld($a), ld($b))) }
        }
    };
}

impl Sha {
    /// SHA-1 fixed rotate: `hash_e.rotate_left(30)` (`sha1h s, s`). Computes
    /// the next round's `e` from the current `a`.
    #[inline]
    pub fn sha1h(self, hash_e: u32) -> u32 {
        unsafe { sha1h_fixed(hash_e) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "sha2")]
unsafe fn sha1h_fixed(hash_e: u32) -> u32 {
    vsha1h_u32(hash_e)
}

sha1_round!(
    sha1c,
    sha1c_fixed,
    vsha1cq_u32,
    "SHA-1 four rounds with the *choose* function, rounds 0-19 (`sha1c q, s, v.4s`)."
);
sha1_round!(
    sha1p,
    sha1p_fixed,
    vsha1pq_u32,
    "SHA-1 four rounds with the *parity* function, rounds 20-39 and 60-79 (`sha1p q, s, v.4s`)."
);
sha1_round!(
    sha1m,
    sha1m_fixed,
    vsha1mq_u32,
    "SHA-1 four rounds with the *majority* function, rounds 40-59 (`sha1m q, s, v.4s`)."
);
sha_ternop!(
    sha1su0,
    sha1su0_fixed,
    vsha1su0q_u32,
    w0_3,
    w4_7,
    w8_11,
    "SHA-1 message schedule update, part 1 of 2 (`sha1su0 v.4s`)."
);
sha_binop!(
    sha1su1,
    sha1su1_fixed,
    vsha1su1q_u32,
    tw0_3,
    w12_15,
    "SHA-1 message schedule update, part 2 of 2 (`sha1su1 v.4s`)."
);
sha_ternop!(
    sha256h,
    sha256h_fixed,
    vsha256hq_u32,
    hash_abcd,
    hash_efgh,
    wk,
    "SHA-256 four rounds, new `abcd` half (`sha256h q, q, v.4s`)."
);
sha_ternop!(
    sha256h2,
    sha256h2_fixed,
    vsha256h2q_u32,
    hash_efgh,
    hash_abcd,
    wk,
    "SHA-256 four rounds, new `efgh` half (`sha256h2 q, q, v.4s`). Takes `efgh` first and the `abcd` from **before** the matching [`Sha::sha256h`] call - the ACLE order, whatever upstream's parameter names say."
);
sha_binop!(
    sha256su0,
    sha256su0_fixed,
    vsha256su0q_u32,
    w0_3,
    w4_7,
    "SHA-256 message schedule update, part 1 of 2 (`sha256su0 v.4s`)."
);
sha_ternop!(
    sha256su1,
    sha256su1_fixed,
    vsha256su1q_u32,
    tw0_3,
    w8_11,
    w12_15,
    "SHA-256 message schedule update, part 2 of 2 (`sha256su1 v.4s`)."
);

#[cfg(test)]
#[path = "../../test/ops/crypto/sha2.rs"]
mod tests;
