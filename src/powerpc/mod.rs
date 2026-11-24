//! PowerPC and Power ISA: game consoles, G4/G5 desktops, OpenPOWER and IBM POWER.
//!
//! - [`features`]: [`Feature`] bits from auxv (Linux/Android/FreeBSD), the core
//!   registers (bare metal) or the compile flags.
//! - [`tiers`]: [`PowerLevel`], named after the open ISA versions.
//! - [`class`]: PVR decoding, [`class::PowerClass`] (which console / desktop /
//!   server), cache-line size.
//! - [`ops`]: tokens and slice kernels in stable `asm!`: `Vmx` (AltiVec,
//!   big-endian), `PairedSingle` (GameCube/Wii/Wii U), `Vmx128` (Xbox 360).
//! - [`auto`]: one call per op, best kernel picked at run time.
//!
//! Detection is cached once per process: [`detect_level`], [`detect_features`].

use crate::level_cache::CachedU8;

pub mod auto;
pub mod class;
pub mod features;
pub mod ops;
// The register-level bare-metal probe is only reached with `target_os = "none"`.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
mod pvr;
pub mod shortpath;
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::PowerLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();

static CACHED_FEATURES: features::CachedFeatureSet = features::CachedFeatureSet::new();

/// Every detected feature bit for this process, cached after the first call.
///
/// The runtime probe is unioned with the compile-time features, so anything
/// the binary was built to assume counts even where the OS probe cannot see
/// it. Prefer this over [`FeatureSet::detect`], which probes every time.
#[inline]
pub fn detect_features() -> FeatureSet {
    FeatureSet::compile_time().union(CACHED_FEATURES.get_or_init(FeatureSet::detect))
}

/// Same answer as [`detect_features`], but always probes again (no cache).
/// Tests and diagnostics only.
pub fn detect_features_fresh() -> FeatureSet {
    FeatureSet::compile_time().union(FeatureSet::detect())
}

fn level_from_u8(v: u8) -> Option<PowerLevel> {
    match v {
        0 => Some(PowerLevel::Scalar),
        1 => Some(PowerLevel::Altivec),
        2 => Some(PowerLevel::Vsx),
        3 => Some(PowerLevel::Isa207),
        4 => Some(PowerLevel::Isa300),
        5 => Some(PowerLevel::Isa31),
        _ => None,
    }
}

/// Best capability tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
pub fn detect_level() -> PowerLevel {
    let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
    level_from_u8(v).expect("cached PowerLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
pub fn detect_level_fresh() -> PowerLevel {
    let host = PowerLevel::detect(detect_features_fresh());
    match shortpath::resolve() {
        // Compile flags are a floor, not a ceiling: the host may run higher.
        shortpath::ResolvedPath::Assumed(level) if level > host => level,
        _ => host,
    }
}

/// Optional: fill the detect cache at startup. Not required.
pub fn warm_up() {
    let _ = detect_level();
    let _ = detect_features();
}

#[cfg(test)]
#[path = "test/ops.rs"]
mod ops_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_feature_has_a_unique_bit() {
        let mut seen = std::collections::HashSet::new();
        for &feature in Feature::ALL {
            assert!(seen.insert(feature.bit()), "duplicate bit for {feature:?}");
        }
    }

    #[test]
    fn power_levels_are_cumulative() {
        for pair in PowerLevel::ALL.windows(2) {
            let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
            assert!(
                lower.iter().all(|f| higher.contains(f)),
                "{:?} must be a superset of {:?}",
                pair[1],
                pair[0]
            );
        }
    }

    #[test]
    fn every_level_survives_the_cache_encoding() {
        for &level in PowerLevel::ALL {
            assert_eq!(level_from_u8(level as u8), Some(level));
        }
    }

    /// No `+altivec`/`+vsx` in plain `cargo test`: shortpath baseline, verify no-op.
    #[test]
    fn verify_or_panic_is_a_no_op_without_a_compile_time_power_level() {
        shortpath::verify_or_panic();
    }

    #[test]
    fn detect_level_matches_fresh() {
        assert_eq!(detect_level(), detect_level_fresh());
    }

    #[test]
    fn detect_level_is_stable_across_calls() {
        let a = detect_level();
        let b = detect_level();
        assert_eq!(a, b);
    }

    #[test]
    fn warm_up_then_detect_agrees_with_fresh() {
        warm_up();
        assert_eq!(detect_level(), detect_level_fresh());
    }
}
