//! Bundled tokens for whole x86-64 levels.
//!
//! A kernel that needs AVX2 + FMA + AVX-512BW + VBMI otherwise has to carry
//! four tokens. [`V3`], [`V4`] and [`V4Vbmi`] hold the tokens of one level
//! together and prove the whole [`GenericLevel`] (including the BMI/LZCNT/
//! MOVBE bits that have no token), so one `detect()` replaces the per-token
//! probes. Fields are public tokens: copy out the ones a call needs.
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::bundles::V3;
//! miraculix::avx_v3_fn! {
//!     fn fma(v: V3, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
//!         v.fma.fmadd_f32x8(a, b, c)
//!     }
//! }
//! if let Some(v3) = V3::detect() {
//!     let _ = fma(v3, [1.0; 8], [2.0; 8], [3.0; 8]);
//! }
//! # }
//! ```
//!
//! Pair a bundle with its macro: [`avx_v3_fn!`](crate::avx_v3_fn) for [`V3`],
//! [`avx512_v4_fn!`](crate::avx512_v4_fn) for [`V4`],
//! [`avx512_v4_vbmi_fn!`](crate::avx512_v4_vbmi_fn) for [`V4Vbmi`].
//!
//! Detection uses the cached [`crate::x86::detect_features`], so it folds to a
//! constant when the binary is built with `-C target-cpu` at or above the level.

use super::ops::avx::{avx::Avx, avx2::Avx2, f16c::F16c, fma::Fma};
use super::ops::avx512::{
    avx512bw::Avx512Bw, avx512cd::Avx512cd, avx512dq::Avx512Dq, avx512f::Avx512f,
    avx512vbmi::Avx512Vbmi, avx512vl::Avx512FVl,
};
use super::ops::sse::{sse2::Sse2, sse41::Sse41, sse42::Sse42, ssse3::Ssse3};
use super::{FeatureSet, GenericLevel, detect_features};

/// Tokens of `-march=x86-64-v3`: SSE2..SSE4.2, AVX, AVX2, FMA, F16C.
#[derive(Debug, Clone, Copy)]
pub struct V3 {
    pub sse2: Sse2,
    pub ssse3: Ssse3,
    pub sse41: Sse41,
    pub sse42: Sse42,
    pub avx: Avx,
    pub avx2: Avx2,
    pub fma: Fma,
    pub f16c: F16c,
}

impl V3 {
    /// Cached host probe: `Some` if the whole V3 level is present.
    pub fn detect() -> Option<Self> {
        Self::from_features(detect_features())
    }

    /// `Some` only if `set` covers every feature of [`GenericLevel::V3`].
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        if !set.contains_all(GenericLevel::V3.required_features()) {
            return None;
        }
        Some(V3 {
            sse2: Sse2::from_features(set)?,
            ssse3: Ssse3::from_features(set)?,
            sse41: Sse41::from_features(set)?,
            sse42: Sse42::from_features(set)?,
            avx: Avx::from_features(set)?,
            avx2: Avx2::from_features(set)?,
            fma: Fma::from_features(set)?,
            f16c: F16c::from_features(set)?,
        })
    }
}

/// Tokens of `-march=x86-64-v4`: everything in [`V3`] plus AVX-512
/// F/BW/CD/DQ/VL.
#[derive(Debug, Clone, Copy)]
pub struct V4 {
    pub v3: V3,
    pub avx512f: Avx512f,
    pub avx512bw: Avx512Bw,
    pub avx512cd: Avx512cd,
    pub avx512dq: Avx512Dq,
    /// 128/256-bit AVX-512F forms.
    pub avx512f_vl: Avx512FVl,
}

impl V4 {
    /// Cached host probe: `Some` if the whole V4 level is present.
    pub fn detect() -> Option<Self> {
        Self::from_features(detect_features())
    }

    /// `Some` only if `set` covers every feature of [`GenericLevel::V4`].
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        if !set.contains_all(GenericLevel::V4.required_features()) {
            return None;
        }
        Some(V4 {
            v3: V3::from_features(set)?,
            avx512f: Avx512f::from_features(set)?,
            avx512bw: Avx512Bw::from_features(set)?,
            avx512cd: Avx512cd::from_features(set)?,
            avx512dq: Avx512Dq::from_features(set)?,
            avx512f_vl: Avx512FVl::from_features(set)?,
        })
    }
}

/// [`V4`] plus AVX-512 VBMI (Ice Lake / Zen 4 and later). VBMI is not part of
/// the psABI V4 level, so it is probed separately.
#[derive(Debug, Clone, Copy)]
pub struct V4Vbmi {
    pub v4: V4,
    pub vbmi: Avx512Vbmi,
}

impl V4Vbmi {
    /// Cached host probe: `Some` if V4 and VBMI are both present.
    pub fn detect() -> Option<Self> {
        Self::from_features(detect_features())
    }

    /// `Some` only if `set` has the V4 level and VBMI.
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        Some(V4Vbmi {
            v4: V4::from_features(set)?,
            vbmi: Avx512Vbmi::from_features(set)?,
        })
    }
}

#[cfg(test)]
#[path = "test/bundles.rs"]
mod tests;
