//! x86 / x86_64: detect, auto slice ops, tokens, kernel macros.
//!
//! # What should I call?
//!
//! | Goal | Call |
//! |---|---|
//! | Fast whole-buffer math | [`auto_up`] functions (`add_i32`, `fmadd_f32`, ...) |
//! | "How good is this CPU?" | [`detect_level`] (tier) or [`detect_features`] (raw bits) |
//! | Custom fixed-width kernel | [`ops`] tokens (`Avx2::detect()`, then methods) |
//! | Multi-op SIMD body | macros in [`fn_macros`] (`avx2_fn!`, ...) |
//! | Startup cache fill | [`warm_up`] (optional) |
//! | Binary vs CPU mismatch | [`shortpath::verify_or_panic`] |
//!
//! No app `init` is required. First detect fills a process-wide cache.

use crate::level_cache::CachedU8;

/// The target's intrinsics: `core::arch::x86` on 32-bit x86,
/// `core::arch::x86_64` on x86-64. The op tables import from here so one
/// source serves both; what exists on x86-64 only is gated where it's used.
#[cfg(target_arch = "x86")]
pub(crate) mod arch {
    pub use core::arch::x86::*;

    // The x86-64-only intrinsics the op tables use, from 32-bit ones with
    // the same result, so the tables compile unchanged.

    /// `_mm_cvtsi64_si128`: `a` in the low lane, the high lane zero.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn _mm_cvtsi64_si128(a: i64) -> __m128i {
        _mm_set_epi64x(0, a)
    }

    /// `_mm_cvtsi128_si64`: the low 64-bit lane.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn _mm_cvtsi128_si64(a: __m128i) -> i64 {
        let lo = _mm_cvtsi128_si32(a).cast_unsigned();
        let hi = _mm_cvtsi128_si32(_mm_srli_si128::<4>(a)).cast_unsigned();
        ((u64::from(hi) << 32) | u64::from(lo)).cast_signed()
    }

    /// `_popcnt64`: both halves counted.
    #[inline]
    #[target_feature(enable = "popcnt")]
    pub fn _popcnt64(x: i64) -> i32 {
        #[allow(clippy::cast_possible_truncation, reason = "the two halves")]
        let (lo, hi) = (x as i32, (x >> 32) as i32);
        _popcnt32(lo) + _popcnt32(hi)
    }

    /// `_mm_crc32_u64`: CRC32-C of the eight bytes, low dword first, on the
    /// low 32 bits of `crc`; the result zero-extended.
    #[inline]
    #[target_feature(enable = "sse4.2")]
    pub fn _mm_crc32_u64(crc: u64, v: u64) -> u64 {
        #[allow(clippy::cast_possible_truncation, reason = "the low halves")]
        let low = _mm_crc32_u32(crc as u32, v as u32);
        #[allow(clippy::cast_possible_truncation, reason = "the high half")]
        let high = (v >> 32) as u32;
        u64::from(_mm_crc32_u32(low, high))
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) use core::arch::x86_64 as arch;
use features::CachedFeatureSet;

mod auto_down;
/// Ready-made elementwise slice ops: pick the best SIMD tier automatically.
pub mod auto_up;
/// Short alias for [`auto_up`] (same module). Prefer the `auto_up` name in new code.
pub use auto_up as auto;
/// ISA feature enum and bitset ([`Feature`], [`FeatureSet`]).
pub mod bundles;
pub mod features;
/// `avx_fn!` / `avx2_fn!` / ... trampolines for multi-op kernels.
pub mod fn_macros;
/// Token-gated ops (SSE, AVX, AVX-512, ...), fixed-width and slice methods.
pub mod ops;
pub mod reduce;
/// Compile-time shortpath and optional startup verify.
pub mod shortpath;
/// Coarse [`GenericLevel`] V1..V4 and [`Avx10`] helpers.
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::{Avx10, GenericLevel};

static CACHED_LEVEL: CachedU8 = CachedU8::new();
static CACHED_FEATURES: CachedFeatureSet = CachedFeatureSet::new();

fn level_from_u8(v: u8) -> Option<GenericLevel> {
    match v {
        1 => Some(GenericLevel::V1),
        2 => Some(GenericLevel::V2),
        3 => Some(GenericLevel::V3),
        4 => Some(GenericLevel::V4),
        _ => None,
    }
}

/// Best x86-64 **tier** for this process (`V1`..`V4`).
///
/// Call this in normal application code. The first call probes (or uses a
/// compile-time shortpath); later calls are a cheap cache hit. No global
/// `init` is required.
///
/// Prefer this over [`detect_level_fresh`] unless you are writing tests.
///
/// # Example
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
/// use miraculix::x86::{detect_level, GenericLevel};
/// let level = detect_level();
/// if level >= GenericLevel::V3 {
///     // Full x86-64-v3 bundle is available (AVX2, FMA, ...).
/// }
/// # }
/// ```
pub fn detect_level() -> GenericLevel {
    let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
    level_from_u8(v).expect("cached GenericLevel discriminant")
}

/// Same answer as [`detect_level`], but **always** re-runs shortpath / `CPUID`.
///
/// Does not read or write the process cache. Use in tests or re-audits.
/// Prefer [`detect_level`] in hot paths.
pub fn detect_level_fresh() -> GenericLevel {
    match shortpath::resolve() {
        shortpath::ResolvedPath::Assumed(level) => {
            // Compile flags are a floor, not a ceiling: the host may run higher.
            let host = GenericLevel::detect(FeatureSet::detect());
            if host > level { host } else { level }
        }
        shortpath::ResolvedPath::RuntimeDispatch => GenericLevel::detect(FeatureSet::detect()),
    }
}

/// Full **per-feature** capability bitset for this process.
///
/// Unlike [`detect_level`], this is not folded into a coarse V1..V4 bucket.
/// A host can have `Avx512f` without every V4 flag; auto dispatch uses this
/// raw set so those features still light up.
///
/// Cached after the first call (union of compile-time lower bound and one
/// `CPUID` probe).
///
/// # Example
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
/// use miraculix::x86::{detect_features, Feature};
/// use miraculix::x86::ops::avx::avx2::Avx2;
///
/// let set = detect_features();
/// if set.contains(Feature::Avx2) {
///     let t = Avx2::from_features(set).unwrap();
///     let _ = t;
/// }
/// # }
/// ```
pub fn detect_features() -> FeatureSet {
    shortpath::compile_time_features().union(CACHED_FEATURES.get_or_init(FeatureSet::detect))
}

/// Same answer as [`detect_features`], but always re-probes `CPUID`.
///
/// Does not read or write the process cache. Prefer [`detect_features`] in
/// normal code.
pub fn detect_features_fresh() -> FeatureSet {
    shortpath::compile_time_features().union(FeatureSet::detect())
}

/// Optional: run detect once at startup so the first hot path is a cache hit.
///
/// Not required. Safe to call multiple times.
pub fn warm_up() {
    let _ = detect_level();
    let _ = detect_features();
}

#[cfg(test)]
#[path = "test/x86_mod.rs"]
mod tests;
