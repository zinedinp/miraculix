//! k-mask flag predicates (`kortest`/`ktest`, `u8`/`u16`/`u32`/`u64`).
//! CF/ZF are returned as `bool` rather than a mask value. See [`super`]
//! for feature gating.

use core::arch::x86_64::{
    _kortestc_mask8_u8, _kortestc_mask16_u8, _kortestc_mask32_u8, _kortestc_mask64_u8,
    _kortestz_mask8_u8, _kortestz_mask16_u8, _kortestz_mask32_u8, _kortestz_mask64_u8,
    _ktestc_mask8_u8, _ktestc_mask16_u8, _ktestc_mask32_u8, _ktestc_mask64_u8, _ktestz_mask8_u8,
    _ktestz_mask16_u8, _ktestz_mask32_u8, _ktestz_mask64_u8,
};

use super::super::avx512bw::Avx512Bw;
use super::super::avx512dq::Avx512Dq;
use super::super::avx512f::Avx512f;

macro_rules! k_test {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: $mask, b: $mask) -> bool {
                unsafe { $intrinsic_fn(a, b) != 0 }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: $mask, b: $mask) -> u8 {
            $intrinsic(a, b)
        }
    };
}

k_test!(
    Avx512f,
    "avx512f",
    kortestc_mask16,
    kortestc_mask16_intrinsic,
    _kortestc_mask16_u8,
    u16,
    "`true` iff `a | b` is all-ones, 16-bit masks (`kortestw`, CF)."
);
k_test!(
    Avx512f,
    "avx512f",
    kortestz_mask16,
    kortestz_mask16_intrinsic,
    _kortestz_mask16_u8,
    u16,
    "`true` iff `a | b` is all-zero, 16-bit masks (`kortestw`, ZF)."
);

k_test!(
    Avx512Dq,
    "avx512dq",
    kortestc_mask8,
    kortestc_mask8_intrinsic,
    _kortestc_mask8_u8,
    u8,
    "`true` iff `a | b` is all-ones, 8-bit masks (`kortestb`, CF)."
);
k_test!(
    Avx512Dq,
    "avx512dq",
    kortestz_mask8,
    kortestz_mask8_intrinsic,
    _kortestz_mask8_u8,
    u8,
    "`true` iff `a | b` is all-zero, 8-bit masks (`kortestb`, ZF)."
);
k_test!(
    Avx512Dq,
    "avx512dq",
    ktestc_mask8,
    ktestc_mask8_intrinsic,
    _ktestc_mask8_u8,
    u8,
    "`true` iff `!a & b` is all-zero, 8-bit masks (`ktestb`, CF)."
);
k_test!(
    Avx512Dq,
    "avx512dq",
    ktestz_mask8,
    ktestz_mask8_intrinsic,
    _ktestz_mask8_u8,
    u8,
    "`true` iff `a & b` is all-zero, 8-bit masks (`ktestb`, ZF)."
);

// mask16 extras that need AVX-512DQ, not just AVX-512F (KTESTW).
k_test!(
    Avx512Dq,
    "avx512dq",
    ktestc_mask16,
    ktestc_mask16_intrinsic,
    _ktestc_mask16_u8,
    u16,
    "`true` iff `!a & b` is all-zero, 16-bit masks (`ktestw`, CF); needs AVX-512DQ."
);
k_test!(
    Avx512Dq,
    "avx512dq",
    ktestz_mask16,
    ktestz_mask16_intrinsic,
    _ktestz_mask16_u8,
    u16,
    "`true` iff `a & b` is all-zero, 16-bit masks (`ktestw`, ZF); needs AVX-512DQ."
);

k_test!(
    Avx512Bw,
    "avx512bw",
    kortestc_mask32,
    kortestc_mask32_intrinsic,
    _kortestc_mask32_u8,
    u32,
    "`true` iff `a | b` is all-ones, 32-bit masks (`kortestd`, CF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    kortestz_mask32,
    kortestz_mask32_intrinsic,
    _kortestz_mask32_u8,
    u32,
    "`true` iff `a | b` is all-zero, 32-bit masks (`kortestd`, ZF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    ktestc_mask32,
    ktestc_mask32_intrinsic,
    _ktestc_mask32_u8,
    u32,
    "`true` iff `!a & b` is all-zero, 32-bit masks (`ktestd`, CF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    ktestz_mask32,
    ktestz_mask32_intrinsic,
    _ktestz_mask32_u8,
    u32,
    "`true` iff `a & b` is all-zero, 32-bit masks (`ktestd`, ZF)."
);

k_test!(
    Avx512Bw,
    "avx512bw",
    kortestc_mask64,
    kortestc_mask64_intrinsic,
    _kortestc_mask64_u8,
    u64,
    "`true` iff `a | b` is all-ones, 64-bit masks (`kortestq`, CF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    kortestz_mask64,
    kortestz_mask64_intrinsic,
    _kortestz_mask64_u8,
    u64,
    "`true` iff `a | b` is all-zero, 64-bit masks (`kortestq`, ZF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    ktestc_mask64,
    ktestc_mask64_intrinsic,
    _ktestc_mask64_u8,
    u64,
    "`true` iff `!a & b` is all-zero, 64-bit masks (`ktestq`, CF)."
);
k_test!(
    Avx512Bw,
    "avx512bw",
    ktestz_mask64,
    ktestz_mask64_intrinsic,
    _ktestz_mask64_u8,
    u64,
    "`true` iff `a & b` is all-zero, 64-bit masks (`ktestq`, ZF)."
);

#[cfg(test)]
#[path = "../../../test/ops/avx512/kmask/ktest.rs"]
mod tests;
