//! Mask construction from a lane count (`mask_between`): a k-mask with the
//! low `n` bits set (lanes `[0, n)` active), the shape a masked partial/tail
//! load or store needs. Pure bit arithmetic: no HW instruction to wrap, but
//! still token-gated so it only appears alongside the masked ops it feeds.
//! See [`super`] for feature gating.

use super::super::avx512bw::Avx512Bw;
use super::super::avx512dq::Avx512Dq;
use super::super::avx512f::Avx512f;

macro_rules! k_mask_between {
    ($token:ty, $fixed_fn:ident, $mask:ty, $width:literal, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            ///
            /// `n >= $width` returns all-ones (every lane active).
            #[inline]
            pub fn $fixed_fn(self, n: u32) -> $mask {
                if n >= $width {
                    <$mask>::MAX
                } else {
                    (1 as $mask << n) - 1
                }
            }
        }
    };
}

k_mask_between!(Avx512Dq, mask_between_mask8, u8, 8, "8-bit mask with lanes `[0, n)` active (pairs with `Avx512Dq`'s mask8 ops).");
k_mask_between!(Avx512f, mask_between_mask16, u16, 16, "16-bit mask with lanes `[0, n)` active (pairs with `Avx512f`'s mask16 ops).");
k_mask_between!(Avx512Bw, mask_between_mask32, u32, 32, "32-bit mask with lanes `[0, n)` active (pairs with `Avx512Bw`'s mask32 ops).");
k_mask_between!(Avx512Bw, mask_between_mask64, u64, 64, "64-bit mask with lanes `[0, n)` active (pairs with `Avx512Bw`'s mask64 ops).");

#[cfg(test)]
#[path = "../../../test/ops/avx512/kmask/build.rs"]
mod tests;
