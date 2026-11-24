//! AArch64 NEON/SVE/SME: [`features`], [`tiers`] [`ArmLevel`], [`shortpath`].
//! Linux/Android `getauxval`; FreeBSD `elf_aux_info`; bare/other compile-time floor.
//! macOS [`AppleLevel`]; Windows [`SnapdragonLevel`]. Once-global: [`detect_level`].

// ArmLevel path (not Apple/Windows tiers).
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use crate::level_cache::CachedU8;
use features::CachedFeatureSet;

pub mod features;
/// Token-gated Neon / crypto / CRC32 ops. No Cargo feature and no nightly
/// toolchain needed: `core::arch::aarch64`'s Neon intrinsics are stable
/// (unlike `core::arch::arm`'s, which gate `aarch32::ops` behind
/// `nightly-arm-neon`).
pub mod ops;
pub mod reduce;
pub mod shortpath;
pub mod tiers;

/// Whole-buffer slice ops without picking a token by hand (see [`auto_up`]).
mod auto_down;
pub mod auto_up;
/// Short alias for [`auto_up`] (same module). Prefer `auto_up` in new code.
pub use auto_up as auto;

#[cfg(target_os = "macos")]
pub mod apple;

#[cfg(target_os = "windows")]
pub mod windows;

pub use features::{Feature, FeatureSet};
pub use tiers::ArmLevel;

#[cfg(target_os = "macos")]
pub use apple::AppleLevel;

#[cfg(target_os = "windows")]
pub use windows::SnapdragonLevel;

static CACHED_FEATURES: CachedFeatureSet = CachedFeatureSet::new();

/// Raw feature bits for this process (cached after first call).
///
/// The runtime probe (auxv / `sysctlbyname` / `IsProcessorFeaturePresent`) is
/// `union`ed with [`shortpath::compile_time_features`], so anything the binary
/// was *compiled* to assume counts even where the OS probe cannot see it.
/// Prefer this over [`detect_features_fresh`] in normal code; [`auto_up`] uses
/// it so cascades do not re-read `auxv` on every call.
///
/// ```
/// # #[cfg(target_arch = "aarch64")] {
/// use miraculix::aarch64::{detect_features, Feature};
///
/// let set = detect_features();
/// assert!(set.contains(Feature::Neon)); // mandatory on AArch64
/// # }
/// ```
#[inline]
pub fn detect_features() -> FeatureSet {
    shortpath::compile_time_features().union(CACHED_FEATURES.get_or_init(FeatureSet::detect))
}

/// Same answer as [`detect_features`], but always re-probes the OS.
///
/// Does not read or write the process cache. Prefer [`detect_features`] in
/// normal code.
pub fn detect_features_fresh() -> FeatureSet {
    shortpath::compile_time_features().union(FeatureSet::detect())
}

// ArmLevel API (Linux, Android, FreeBSD, bare-metal, other).

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
static CACHED_LEVEL: CachedU8 = CachedU8::new();

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn arm_level_from_u8(v: u8) -> Option<ArmLevel> {
    match v {
        0 => Some(ArmLevel::V8_0),
        1 => Some(ArmLevel::V8_1),
        2 => Some(ArmLevel::V8_2),
        3 => Some(ArmLevel::V8_3),
        4 => Some(ArmLevel::V8_4),
        5 => Some(ArmLevel::V8_5),
        6 => Some(ArmLevel::V8_6),
        7 => Some(ArmLevel::V8_7),
        8 => Some(ArmLevel::V8_8),
        9 => Some(ArmLevel::V8_9),
        10 => Some(ArmLevel::V9_0),
        11 => Some(ArmLevel::V9_1),
        12 => Some(ArmLevel::V9_2),
        13 => Some(ArmLevel::V9_3),
        14 => Some(ArmLevel::V9_4),
        15 => Some(ArmLevel::V9_5),
        _ => None,
    }
}

/// Best Arm tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[inline]
pub fn detect_level() -> ArmLevel {
    let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
    arm_level_from_u8(v).expect("cached ArmLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn detect_level_fresh() -> ArmLevel {
    match shortpath::resolve() {
        shortpath::ResolvedPath::Assumed(level) => {
            // Compile flags are a floor, not a ceiling: the host may run higher.
            let host = ArmLevel::detect(FeatureSet::detect());
            if host > level { host } else { level }
        }
        shortpath::ResolvedPath::Baseline => ArmLevel::detect(FeatureSet::detect()),
    }
}

/// Optional: fill the detect cache at startup. Not required.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn warm_up() {
    let _ = detect_level();
    let _ = detect_features();
}

// macOS: forward AppleLevel.

/// Best Apple Silicon **tier** for this process (cached).
#[cfg(target_os = "macos")]
#[inline]
pub fn detect_level() -> AppleLevel {
    apple::detect_level()
}

/// Same as [`detect_level`] but always re-probes. Tests / re-audit only.
#[cfg(target_os = "macos")]
pub fn detect_level_fresh() -> AppleLevel {
    apple::detect_level_fresh()
}

/// Optional: fill the detect cache at startup. Not required.
#[cfg(target_os = "macos")]
pub fn warm_up() {
    apple::warm_up();
    let _ = detect_features();
}

// Windows: forward SnapdragonLevel.

/// Best Windows-on-Arm **tier** for this process (cached).
#[cfg(target_os = "windows")]
#[inline]
pub fn detect_level() -> SnapdragonLevel {
    windows::detect_level()
}

/// Same as [`detect_level`] but always re-probes. Tests / re-audit only.
#[cfg(target_os = "windows")]
pub fn detect_level_fresh() -> SnapdragonLevel {
    windows::detect_level_fresh()
}

/// Optional: fill the detect cache at startup. Not required.
#[cfg(target_os = "windows")]
pub fn warm_up() {
    windows::warm_up();
    let _ = detect_features();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NEON mandatory on AArch64.
    #[test]
    fn neon_is_always_available() {
        assert!(FeatureSet::detect().contains(Feature::Neon));
    }

    /// Distinct bit indices for the Feature bitset.
    #[test]
    fn every_feature_has_a_unique_bit() {
        let mut seen = std::collections::HashSet::new();
        for &feature in Feature::ALL {
            assert!(seen.insert(feature.bit()), "duplicate bit for {feature:?}");
        }
    }

    /// `detect` matches manual required-features on the same set.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    fn arm_level_matches_manual_check() {
        let set = FeatureSet::detect();
        let detected = ArmLevel::detect(set);

        for &level in ArmLevel::ALL {
            let should_hold = set.contains_all(level.required_features());
            if !level.is_runtime_detectable() {
                assert!(
                    level > detected,
                    "detect() returned compile-time-only {detected:?}"
                );
                continue;
            }
            assert_eq!(
                level <= detected,
                should_hold || level == ArmLevel::V8_0,
                "level {level:?} required-features check disagrees with detect() result"
            );
        }
    }

    /// Higher tiers supersets of lower (copy-paste guard).
    #[test]
    fn arm_levels_are_cumulative() {
        for pair in ArmLevel::ALL.windows(2) {
            let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
            assert!(
                lower.iter().all(|f| higher.contains(f)),
                "{:?} must be a superset of {:?}",
                pair[1],
                pair[0]
            );
        }
    }

    /// Every discriminant round-trips through the level cache decoder.
    #[test]
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    fn arm_level_from_u8_round_trips_every_level() {
        for (i, &level) in ArmLevel::ALL.iter().enumerate() {
            assert_eq!(
                level as u8 as usize, i,
                "ALL out of discriminant order at {level:?}"
            );
            assert_eq!(arm_level_from_u8(level as u8), Some(level));
        }
        assert_eq!(arm_level_from_u8(ArmLevel::ALL.len() as u8), None);
    }

    /// No march/feature in plain `cargo test`: verify no-op.
    #[test]
    fn verify_or_panic_is_a_no_op_without_a_compile_time_arch_level() {
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
        let c = detect_level();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn warm_up_then_detect_agrees_with_fresh() {
        warm_up();
        assert_eq!(detect_level(), detect_level_fresh());
    }

    /// Cached path matches a fresh probe.
    #[test]
    fn detect_features_matches_fresh() {
        assert_eq!(detect_features(), detect_features_fresh());
    }

    /// Repeated calls are stable (process cache).
    #[test]
    fn detect_features_is_stable_across_calls() {
        let a = detect_features();
        let b = detect_features();
        assert_eq!(a, b);
    }

    /// The compile-time floor is a *lower* bound: it can only add bits.
    #[test]
    fn compile_time_features_is_a_floor_not_a_ceiling() {
        let probed = FeatureSet::detect();
        let combined = detect_features();
        for &feature in Feature::ALL {
            if probed.contains(feature) {
                assert!(combined.contains(feature), "{feature:?} lost by the union");
            }
        }
    }

    /// Non-vacuity guard for `scripts/qemu-test.sh`. Every token test opens
    /// with `detect() else return`, so a CPU model that silently lacked a
    /// feature would pass its tests without running them. Under QEMU the
    /// model name arrives in `QEMU_CPU` (qemu-user passes the environment
    /// through); this pins what each matrix row must report, and must not.
    /// Outside QEMU the variable is unset and the test is a no-op.
    ///
    /// Observed with QEMU 11.1: `cortex-a53` **does** report AES/PMULL/SHA/CRC32,
    /// so no row demonstrates crypto-token refusal; the refusal rows prove it
    /// for FP16, DotProd, RDM, SVE, I8MM and BF16 instead.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn qemu_cpu_model_reports_the_expected_features() {
        use Feature::*;
        let Ok(cpu) = std::env::var("QEMU_CPU") else {
            return;
        };
        let set = detect_features();
        let (must, must_not): (&[Feature], &[Feature]) = match cpu.split(',').next().unwrap_or("") {
            "cortex-a53" | "cortex-a72" => (
                &[Neon, Aes, Pmull, Sha1, Sha2, Crc32],
                &[
                    Fp16, AsimdHp, Dotprod, Rdm, Fhm, Sve, Sve2, I8mm, Bf16, Sha3, Sha512, Sm3,
                    Sm4, Jscvt, Fcma, Frint, Rng,
                ],
            ),
            "cortex-a76" => (
                &[Fp16, AsimdHp, Dotprod, Rdm, Lse],
                &[Sve, Fhm, I8mm, Bf16, Sha3, Jscvt, Fcma, Frint],
            ),
            "neoverse-v1" => (
                &[
                    Sve, Sha3, Sha512, Sm3, Sm4, Fhm, I8mm, Jscvt, Fcma, Rng, Dotprod,
                ],
                &[Sve2],
            ),
            "neoverse-n2" => (
                &[Sve, Sve2, Sve2Aes, Sve2Bitperm, Bf16, Frint, I8mm, Fhm],
                &[Sme, Sme2, Sve2p1, Cssc, Mops],
            ),
            "a64fx" => (&[Sve, Fp16, Fcma], &[Sve2, Dotprod]),
            "max" => (
                &[
                    Neon,
                    Aes,
                    Pmull,
                    Sha1,
                    Sha2,
                    Sha3,
                    Sha512,
                    Sm3,
                    Sm4,
                    Crc32,
                    Rdm,
                    Fp16,
                    AsimdHp,
                    Dotprod,
                    Fhm,
                    Fcma,
                    Jscvt,
                    Frint,
                    I8mm,
                    Bf16,
                    Rng,
                    Sve,
                    Sve2,
                    Sve2Aes,
                    Sve2Bitperm,
                    Sve2I8mm,
                    Sve2F32mm,
                    Sve2F64mm,
                    Sve2Bf16,
                    Sme,
                    // AT_HWCAP2 34..59, observed on QEMU 11.1.
                    Sme2,
                    Sme2p1,
                    Sve2p1,
                    Cssc,
                    Mops,
                    Hbc,
                    Lut,
                    Faminmax,
                    F8cvt,
                    F8fma,
                    SveB16b16,
                ],
                &[],
            ),
            _ => return,
        };
        for f in must {
            // FEAT_F64MM needs a vector length of at least 256 bits; QEMU
            // withdraws it when `max` is capped at 128 (`sve256=off`).
            if *f == Sve2F64mm && cpu.contains("sve256=off") {
                continue;
            }
            assert!(
                set.contains(*f),
                "QEMU_CPU={cpu} should report {f:?}; its token tests would be vacuous"
            );
        }
        for f in must_not {
            assert!(
                !set.contains(*f),
                "QEMU_CPU={cpu} should NOT report {f:?}; the refusal row proves nothing"
            );
        }
    }

    /// NEON/FP are in the compile-time floor even with no `-C target-feature`.
    #[test]
    fn compile_time_features_carries_the_aarch64_baseline() {
        let set = shortpath::compile_time_features();
        assert!(set.contains(Feature::Neon));
        assert!(set.contains(Feature::Fp));
    }
}
