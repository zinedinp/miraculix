//! k-mask bitwise logic (`kand`/`kor`/`kxor`/`kxnor`/`kandn`/`knot`)
//! plus wrapping add (`kadd`), `u8`/`u16`/`u32`/`u64`. See [`super`]
//! for feature gating.

use core::arch::x86_64::{
    _kadd_mask8, _kadd_mask16, _kadd_mask32, _kadd_mask64, _kand_mask8, _kand_mask16, _kand_mask32,
    _kand_mask64, _kandn_mask8, _kandn_mask16, _kandn_mask32, _kandn_mask64, _knot_mask8,
    _knot_mask16, _knot_mask32, _knot_mask64, _kor_mask8, _kor_mask16, _kor_mask32, _kor_mask64,
    _kxnor_mask8, _kxnor_mask16, _kxnor_mask32, _kxnor_mask64, _kxor_mask8, _kxor_mask16,
    _kxor_mask32, _kxor_mask64,
};

use super::super::avx512bw::Avx512Bw;
use super::super::avx512dq::Avx512Dq;
use super::super::avx512f::Avx512f;

macro_rules! k_binop {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: $mask, b: $mask) -> $mask {
                unsafe { $intrinsic_fn(a, b) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: $mask, b: $mask) -> $mask {
            $intrinsic(a, b)
        }
    };
}

macro_rules! k_unop {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: $mask) -> $mask {
                unsafe { $intrinsic_fn(a) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: $mask) -> $mask {
            $intrinsic(a)
        }
    };
}

// mask16, AVX-512F

k_binop!(
    Avx512f,
    "avx512f",
    kand_mask16,
    kand_mask16_intrinsic,
    _kand_mask16,
    u16,
    "Bitwise AND of two 16-bit masks (`kandw`)."
);
k_binop!(
    Avx512f,
    "avx512f",
    kor_mask16,
    kor_mask16_intrinsic,
    _kor_mask16,
    u16,
    "Bitwise OR of two 16-bit masks (`korw`)."
);
k_binop!(
    Avx512f,
    "avx512f",
    kxor_mask16,
    kxor_mask16_intrinsic,
    _kxor_mask16,
    u16,
    "Bitwise XOR of two 16-bit masks (`kxorw`)."
);
k_binop!(
    Avx512f,
    "avx512f",
    kxnor_mask16,
    kxnor_mask16_intrinsic,
    _kxnor_mask16,
    u16,
    "Bitwise XNOR of two 16-bit masks (`kxnorw`)."
);
k_binop!(
    Avx512f,
    "avx512f",
    kandn_mask16,
    kandn_mask16_intrinsic,
    _kandn_mask16,
    u16,
    "Bitwise NOT of `a` ANDed with `b`, 16-bit masks (`kandnw`)."
);
k_unop!(
    Avx512f,
    "avx512f",
    knot_mask16,
    knot_mask16_intrinsic,
    _knot_mask16,
    u16,
    "Bitwise NOT of a 16-bit mask (`knotw`)."
);

// mask8, AVX-512DQ

k_binop!(
    Avx512Dq,
    "avx512dq",
    kand_mask8,
    kand_mask8_intrinsic,
    _kand_mask8,
    u8,
    "Bitwise AND of two 8-bit masks (`kandb`)."
);
k_binop!(
    Avx512Dq,
    "avx512dq",
    kor_mask8,
    kor_mask8_intrinsic,
    _kor_mask8,
    u8,
    "Bitwise OR of two 8-bit masks (`korb`)."
);
k_binop!(
    Avx512Dq,
    "avx512dq",
    kxor_mask8,
    kxor_mask8_intrinsic,
    _kxor_mask8,
    u8,
    "Bitwise XOR of two 8-bit masks (`kxorb`)."
);
k_binop!(
    Avx512Dq,
    "avx512dq",
    kxnor_mask8,
    kxnor_mask8_intrinsic,
    _kxnor_mask8,
    u8,
    "Bitwise XNOR of two 8-bit masks (`kxnorb`)."
);
k_binop!(
    Avx512Dq,
    "avx512dq",
    kandn_mask8,
    kandn_mask8_intrinsic,
    _kandn_mask8,
    u8,
    "Bitwise NOT of `a` ANDed with `b`, 8-bit masks (`kandnb`)."
);
k_binop!(
    Avx512Dq,
    "avx512dq",
    kadd_mask8,
    kadd_mask8_intrinsic,
    _kadd_mask8,
    u8,
    "Add two 8-bit masks, wrapping (`kaddb`)."
);
k_unop!(
    Avx512Dq,
    "avx512dq",
    knot_mask8,
    knot_mask8_intrinsic,
    _knot_mask8,
    u8,
    "Bitwise NOT of an 8-bit mask (`knotb`)."
);

// mask16 extras that need AVX-512DQ, not just AVX-512F (KADDW).
k_binop!(
    Avx512Dq,
    "avx512dq",
    kadd_mask16,
    kadd_mask16_intrinsic,
    _kadd_mask16,
    u16,
    "Add two 16-bit masks, wrapping (`kaddw`); needs AVX-512DQ."
);

// mask32/mask64, AVX-512BW

k_binop!(
    Avx512Bw,
    "avx512bw",
    kand_mask32,
    kand_mask32_intrinsic,
    _kand_mask32,
    u32,
    "Bitwise AND of two 32-bit masks (`kandd`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kor_mask32,
    kor_mask32_intrinsic,
    _kor_mask32,
    u32,
    "Bitwise OR of two 32-bit masks (`kord`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kxor_mask32,
    kxor_mask32_intrinsic,
    _kxor_mask32,
    u32,
    "Bitwise XOR of two 32-bit masks (`kxord`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kxnor_mask32,
    kxnor_mask32_intrinsic,
    _kxnor_mask32,
    u32,
    "Bitwise XNOR of two 32-bit masks (`kxnord`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kandn_mask32,
    kandn_mask32_intrinsic,
    _kandn_mask32,
    u32,
    "Bitwise NOT of `a` ANDed with `b`, 32-bit masks (`kandnd`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kadd_mask32,
    kadd_mask32_intrinsic,
    _kadd_mask32,
    u32,
    "Add two 32-bit masks, wrapping (`kaddd`)."
);
k_unop!(
    Avx512Bw,
    "avx512bw",
    knot_mask32,
    knot_mask32_intrinsic,
    _knot_mask32,
    u32,
    "Bitwise NOT of a 32-bit mask (`knotd`)."
);

k_binop!(
    Avx512Bw,
    "avx512bw",
    kand_mask64,
    kand_mask64_intrinsic,
    _kand_mask64,
    u64,
    "Bitwise AND of two 64-bit masks (`kandq`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kor_mask64,
    kor_mask64_intrinsic,
    _kor_mask64,
    u64,
    "Bitwise OR of two 64-bit masks (`korq`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kxor_mask64,
    kxor_mask64_intrinsic,
    _kxor_mask64,
    u64,
    "Bitwise XOR of two 64-bit masks (`kxorq`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kxnor_mask64,
    kxnor_mask64_intrinsic,
    _kxnor_mask64,
    u64,
    "Bitwise XNOR of two 64-bit masks (`kxnorq`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kandn_mask64,
    kandn_mask64_intrinsic,
    _kandn_mask64,
    u64,
    "Bitwise NOT of `a` ANDed with `b`, 64-bit masks (`kandnq`)."
);
k_binop!(
    Avx512Bw,
    "avx512bw",
    kadd_mask64,
    kadd_mask64_intrinsic,
    _kadd_mask64,
    u64,
    "Add two 64-bit masks, wrapping (`kaddq`)."
);
k_unop!(
    Avx512Bw,
    "avx512bw",
    knot_mask64,
    knot_mask64_intrinsic,
    _knot_mask64,
    u64,
    "Bitwise NOT of a 64-bit mask (`knotq`)."
);

#[cfg(test)]
#[path = "../../../test/ops/avx512/kmask/logic.rs"]
mod tests;
