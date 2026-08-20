//! Mask<->vector bridge (`movepi*_mask` / `movm_epi*`) for 512-bit
//! (`Avx512Bw`/`Avx512Dq`) and 128/256-bit (`Avx512BwVl`/`Avx512DqVl`).
//! See [`super`] for feature gating.

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_movepi8_mask, _mm_movepi16_mask,
    _mm_movepi32_mask, _mm_movepi64_mask, _mm_movm_epi8, _mm_movm_epi16, _mm_movm_epi32,
    _mm_movm_epi64, _mm_storeu_si128, _mm256_loadu_si256, _mm256_movepi8_mask,
    _mm256_movepi16_mask, _mm256_movepi32_mask, _mm256_movepi64_mask, _mm256_movm_epi8,
    _mm256_movm_epi16, _mm256_movm_epi32, _mm256_movm_epi64, _mm256_storeu_si256,
    _mm512_loadu_si512, _mm512_movepi8_mask, _mm512_movepi16_mask, _mm512_movepi32_mask,
    _mm512_movepi64_mask, _mm512_movm_epi8, _mm512_movm_epi16, _mm512_movm_epi32,
    _mm512_movm_epi64, _mm512_storeu_si512,
};

use super::super::avx512bw::Avx512Bw;
use super::super::avx512dq::Avx512Dq;
use super::super::avx512vl::{Avx512BwVl, Avx512DqVl};

macro_rules! k_vec_to_mask {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $width:literal, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$elem; $width]) -> $mask {
                unsafe { $intrinsic_fn(&a) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$elem; $width]) -> $mask {
            unsafe {
                let va: __m512i = _mm512_loadu_si512(a.as_ptr().cast());
                $intrinsic(va)
            }
        }
    };
}

macro_rules! k_mask_to_vec {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $width:literal, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, k: $mask) -> [$elem; $width] {
                unsafe { $intrinsic_fn(k) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(k: $mask) -> [$elem; $width] {
            unsafe {
                let vr: __m512i = $intrinsic(k);
                let mut out: [$elem; $width] = [Default::default(); $width];
                _mm512_storeu_si512(out.as_mut_ptr().cast(), vr);
                out
            }
        }
    };
}

// mask <-> vector bridge, 512-bit only

k_vec_to_mask!(
    Avx512Bw,
    "avx512bw",
    movepi8_mask,
    movepi8_mask_intrinsic,
    _mm512_movepi8_mask,
    i8,
    64,
    u64,
    "Extract a 64-bit mask from each lane's sign bit (`vpmovb2m`)."
);
k_vec_to_mask!(
    Avx512Bw,
    "avx512bw",
    movepi16_mask,
    movepi16_mask_intrinsic,
    _mm512_movepi16_mask,
    i16,
    32,
    u32,
    "Extract a 32-bit mask from each lane's sign bit (`vpmovw2m`)."
);
k_vec_to_mask!(
    Avx512Dq,
    "avx512dq",
    movepi32_mask,
    movepi32_mask_intrinsic,
    _mm512_movepi32_mask,
    i32,
    16,
    u16,
    "Extract a 16-bit mask from each lane's sign bit (`vpmovd2m`)."
);
k_vec_to_mask!(
    Avx512Dq,
    "avx512dq",
    movepi64_mask,
    movepi64_mask_intrinsic,
    _mm512_movepi64_mask,
    i64,
    8,
    u8,
    "Extract an 8-bit mask from each lane's sign bit (`vpmovq2m`)."
);

k_mask_to_vec!(
    Avx512Bw,
    "avx512bw",
    movm_epi8,
    movm_epi8_intrinsic,
    _mm512_movm_epi8,
    i8,
    64,
    u64,
    "Broadcast a 64-bit mask into lanes, each all-0 or all-1 (`vpmovm2b`)."
);
k_mask_to_vec!(
    Avx512Bw,
    "avx512bw",
    movm_epi16,
    movm_epi16_intrinsic,
    _mm512_movm_epi16,
    i16,
    32,
    u32,
    "Broadcast a 32-bit mask into lanes, each all-0 or all-1 (`vpmovm2w`)."
);
k_mask_to_vec!(
    Avx512Dq,
    "avx512dq",
    movm_epi32,
    movm_epi32_intrinsic,
    _mm512_movm_epi32,
    i32,
    16,
    u16,
    "Broadcast a 16-bit mask into lanes, each all-0 or all-1 (`vpmovm2d`)."
);
k_mask_to_vec!(
    Avx512Dq,
    "avx512dq",
    movm_epi64,
    movm_epi64_intrinsic,
    _mm512_movm_epi64,
    i64,
    8,
    u8,
    "Broadcast an 8-bit mask into lanes, each all-0 or all-1 (`vpmovm2q`)."
);

// mask <-> vector bridge, 128/256-bit (`Avx512BwVl`/`Avx512DqVl`)
//
// The 512-bit macros hardcode `_mm512_loadu_si512`/`_mm512_storeu_si512`,
// so 128/256-bit forms need separate macros parameterized by `vec`,
// `loadu`, and `storeu`.

macro_rules! k_vec_to_mask_w {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $width:literal, $vec:ty, $loadu:path, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$elem; $width]) -> $mask {
                unsafe { $intrinsic_fn(&a) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$elem; $width]) -> $mask {
            unsafe {
                let va: $vec = $loadu(a.as_ptr().cast());
                $intrinsic(va)
            }
        }
    };
}

macro_rules! k_mask_to_vec_w {
    ($token:ty, $tf:literal, $fixed_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $elem:ty, $width:literal, $vec:ty, $storeu:path, $mask:ty, $doc:literal) => {
        impl $token {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, k: $mask) -> [$elem; $width] {
                unsafe { $intrinsic_fn(k) }
            }
        }
        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(k: $mask) -> [$elem; $width] {
            unsafe {
                let vr: $vec = $intrinsic(k);
                let mut out: [$elem; $width] = [Default::default(); $width];
                $storeu(out.as_mut_ptr().cast(), vr);
                out
            }
        }
    };
}

k_vec_to_mask_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movepi8_mask_x16,
    movepi8_mask_x16_intrinsic,
    _mm_movepi8_mask,
    i8,
    16,
    __m128i,
    _mm_loadu_si128,
    u16,
    "Extract a 16-bit mask from each lane's sign bit (`vpmovb2m`, 128-bit)."
);
k_vec_to_mask_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movepi8_mask_x32,
    movepi8_mask_x32_intrinsic,
    _mm256_movepi8_mask,
    i8,
    32,
    __m256i,
    _mm256_loadu_si256,
    u32,
    "Extract a 32-bit mask from each lane's sign bit (`vpmovb2m`, 256-bit)."
);
k_vec_to_mask_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movepi16_mask_x8,
    movepi16_mask_x8_intrinsic,
    _mm_movepi16_mask,
    i16,
    8,
    __m128i,
    _mm_loadu_si128,
    u8,
    "Extract an 8-bit mask from each lane's sign bit (`vpmovw2m`, 128-bit)."
);
k_vec_to_mask_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movepi16_mask_x16,
    movepi16_mask_x16_intrinsic,
    _mm256_movepi16_mask,
    i16,
    16,
    __m256i,
    _mm256_loadu_si256,
    u16,
    "Extract a 16-bit mask from each lane's sign bit (`vpmovw2m`, 256-bit)."
);
k_vec_to_mask_w!(Avx512DqVl, "avx512dq,avx512vl", movepi32_mask_x4, movepi32_mask_x4_intrinsic, _mm_movepi32_mask, i32, 4, __m128i, _mm_loadu_si128, u8, "Extract a mask (bottom 4 bits significant) from each lane's sign bit (`vpmovd2m`, 128-bit; `__mmask8`, the AVX-512 minimum, despite 4 lanes).");
k_vec_to_mask_w!(
    Avx512DqVl,
    "avx512dq,avx512vl",
    movepi32_mask_x8,
    movepi32_mask_x8_intrinsic,
    _mm256_movepi32_mask,
    i32,
    8,
    __m256i,
    _mm256_loadu_si256,
    u8,
    "Extract an 8-bit mask from each lane's sign bit (`vpmovd2m`, 256-bit)."
);
k_vec_to_mask_w!(Avx512DqVl, "avx512dq,avx512vl", movepi64_mask_x2, movepi64_mask_x2_intrinsic, _mm_movepi64_mask, i64, 2, __m128i, _mm_loadu_si128, u8, "Extract a mask (bottom 2 bits significant) from each lane's sign bit (`vpmovq2m`, 128-bit; `__mmask8`, the AVX-512 minimum, despite 2 lanes).");
k_vec_to_mask_w!(Avx512DqVl, "avx512dq,avx512vl", movepi64_mask_x4, movepi64_mask_x4_intrinsic, _mm256_movepi64_mask, i64, 4, __m256i, _mm256_loadu_si256, u8, "Extract a mask (bottom 4 bits significant) from each lane's sign bit (`vpmovq2m`, 256-bit; `__mmask8`, the AVX-512 minimum, despite 4 lanes).");

k_mask_to_vec_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movm_epi8_x16,
    movm_epi8_x16_intrinsic,
    _mm_movm_epi8,
    i8,
    16,
    __m128i,
    _mm_storeu_si128,
    u16,
    "Broadcast a 16-bit mask into lanes, each all-0 or all-1 (`vpmovm2b`, 128-bit)."
);
k_mask_to_vec_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movm_epi8_x32,
    movm_epi8_x32_intrinsic,
    _mm256_movm_epi8,
    i8,
    32,
    __m256i,
    _mm256_storeu_si256,
    u32,
    "Broadcast a 32-bit mask into lanes, each all-0 or all-1 (`vpmovm2b`, 256-bit)."
);
k_mask_to_vec_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movm_epi16_x8,
    movm_epi16_x8_intrinsic,
    _mm_movm_epi16,
    i16,
    8,
    __m128i,
    _mm_storeu_si128,
    u8,
    "Broadcast an 8-bit mask into lanes, each all-0 or all-1 (`vpmovm2w`, 128-bit)."
);
k_mask_to_vec_w!(
    Avx512BwVl,
    "avx512bw,avx512vl",
    movm_epi16_x16,
    movm_epi16_x16_intrinsic,
    _mm256_movm_epi16,
    i16,
    16,
    __m256i,
    _mm256_storeu_si256,
    u16,
    "Broadcast a 16-bit mask into lanes, each all-0 or all-1 (`vpmovm2w`, 256-bit)."
);
k_mask_to_vec_w!(
    Avx512DqVl,
    "avx512dq,avx512vl",
    movm_epi32_x4,
    movm_epi32_x4_intrinsic,
    _mm_movm_epi32,
    i32,
    4,
    __m128i,
    _mm_storeu_si128,
    u8,
    "Broadcast the low 4 mask bits into lanes, each all-0 or all-1 (`vpmovm2d`, 128-bit)."
);
k_mask_to_vec_w!(
    Avx512DqVl,
    "avx512dq,avx512vl",
    movm_epi32_x8,
    movm_epi32_x8_intrinsic,
    _mm256_movm_epi32,
    i32,
    8,
    __m256i,
    _mm256_storeu_si256,
    u8,
    "Broadcast an 8-bit mask into lanes, each all-0 or all-1 (`vpmovm2d`, 256-bit)."
);
k_mask_to_vec_w!(
    Avx512DqVl,
    "avx512dq,avx512vl",
    movm_epi64_x2,
    movm_epi64_x2_intrinsic,
    _mm_movm_epi64,
    i64,
    2,
    __m128i,
    _mm_storeu_si128,
    u8,
    "Broadcast the low 2 mask bits into lanes, each all-0 or all-1 (`vpmovm2q`, 128-bit)."
);
k_mask_to_vec_w!(
    Avx512DqVl,
    "avx512dq,avx512vl",
    movm_epi64_x4,
    movm_epi64_x4_intrinsic,
    _mm256_movm_epi64,
    i64,
    4,
    __m256i,
    _mm256_storeu_si256,
    u8,
    "Broadcast the low 4 mask bits into lanes, each all-0 or all-1 (`vpmovm2q`, 256-bit)."
);

#[cfg(test)]
#[path = "../../../test/ops/avx512/kmask/bridge.rs"]
mod tests;
