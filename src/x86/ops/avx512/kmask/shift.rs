//! k-mask shifts (`kshiftl`/`kshiftr`, zero-filled) for `u8`/`u16`/`u32`/`u64`.
//! See [`super`] for feature gating.

use core::arch::x86_64::{
    _kshiftli_mask8, _kshiftli_mask16, _kshiftli_mask32, _kshiftli_mask64, _kshiftri_mask8,
    _kshiftri_mask16, _kshiftri_mask32, _kshiftri_mask64,
};

use super::super::avx512bw::Avx512Bw;
use super::super::avx512dq::Avx512Dq;
use super::super::avx512f::Avx512f;

macro_rules! k_shift {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:ident, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn<const IMM: u32>(self, a: $mask) -> $mask {
                unsafe { $intrinsic_fn::<IMM>(a) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn<const IMM: u32>(a: $mask) -> $mask {
            $intrinsic::<IMM>(a)
        }
    };
}

k_shift!(
    Avx512f,
    "avx512f",
    kshiftli_mask16,
    kshiftli_mask16_intrinsic,
    _kshiftli_mask16,
    u16,
    "Shift a 16-bit mask left by `IMM` bits, zero-filled (`kshiftlw`)."
);
k_shift!(
    Avx512f,
    "avx512f",
    kshiftri_mask16,
    kshiftri_mask16_intrinsic,
    _kshiftri_mask16,
    u16,
    "Shift a 16-bit mask right by `IMM` bits, zero-filled (`kshiftrw`)."
);

k_shift!(
    Avx512Dq,
    "avx512dq",
    kshiftli_mask8,
    kshiftli_mask8_intrinsic,
    _kshiftli_mask8,
    u8,
    "Shift an 8-bit mask left by `IMM` bits, zero-filled (`kshiftlb`)."
);
k_shift!(
    Avx512Dq,
    "avx512dq",
    kshiftri_mask8,
    kshiftri_mask8_intrinsic,
    _kshiftri_mask8,
    u8,
    "Shift an 8-bit mask right by `IMM` bits, zero-filled (`kshiftrb`)."
);

k_shift!(
    Avx512Bw,
    "avx512bw",
    kshiftli_mask32,
    kshiftli_mask32_intrinsic,
    _kshiftli_mask32,
    u32,
    "Shift a 32-bit mask left by `IMM` bits, zero-filled (`kshiftld`)."
);
k_shift!(
    Avx512Bw,
    "avx512bw",
    kshiftri_mask32,
    kshiftri_mask32_intrinsic,
    _kshiftri_mask32,
    u32,
    "Shift a 32-bit mask right by `IMM` bits, zero-filled (`kshiftrd`)."
);
k_shift!(
    Avx512Bw,
    "avx512bw",
    kshiftli_mask64,
    kshiftli_mask64_intrinsic,
    _kshiftli_mask64,
    u64,
    "Shift a 64-bit mask left by `IMM` bits, zero-filled (`kshiftlq`)."
);
k_shift!(
    Avx512Bw,
    "avx512bw",
    kshiftri_mask64,
    kshiftri_mask64_intrinsic,
    _kshiftri_mask64,
    u64,
    "Shift a 64-bit mask right by `IMM` bits, zero-filled (`kshiftrq`)."
);

#[cfg(test)]
#[path = "../../../test/ops/avx512/kmask/shift.rs"]
mod tests;
