//! Compile-time collapse of runtime dispatch. Baseline [`ArmLevel::V8_0`]
//! (NEON always). Cascade on rustc `v8.1a`..`v9.5a` cfgs (authoritative, nightly-only; see below).

use super::tiers::ArmLevel;
use super::{Feature, FeatureSet};

// The `v8.Xa`/`v9.Xa` cfgs are **unstable** target features (verified on
// 1.98.1: `-C target-feature=+v8.1a` warns "unstable feature", and
// `--print cfg` never lists them on stable). On a stable toolchain this
// cascade therefore always lands on V8_0 and `resolve()` returns `Baseline`;
// it only fires on nightly. [`compile_time_features`] below uses the stable
// per-feature cfgs instead and works on both.
#[cfg(target_feature = "v9.5a")]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_5;

#[cfg(all(target_feature = "v9.4a", not(any(target_feature = "v9.5a"))))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_4;

#[cfg(all(
    target_feature = "v9.3a",
    not(any(target_feature = "v9.5a", target_feature = "v9.4a"))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_3;

#[cfg(all(
    target_feature = "v9.2a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_2;

#[cfg(all(
    target_feature = "v9.1a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_1;

#[cfg(all(
    target_feature = "v9a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_0;

#[cfg(all(
    target_feature = "v8.9a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_9;

#[cfg(all(
    target_feature = "v8.8a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_8;

#[cfg(all(
    target_feature = "v8.7a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_7;

#[cfg(all(
    target_feature = "v8.6a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_6;

#[cfg(all(
    target_feature = "v8.5a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a",
        target_feature = "v8.6a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_5;

#[cfg(all(
    target_feature = "v8.4a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a",
        target_feature = "v8.6a",
        target_feature = "v8.5a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_4;

#[cfg(all(
    target_feature = "v8.3a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a",
        target_feature = "v8.6a",
        target_feature = "v8.5a",
        target_feature = "v8.4a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_3;

#[cfg(all(
    target_feature = "v8.2a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a",
        target_feature = "v8.6a",
        target_feature = "v8.5a",
        target_feature = "v8.4a",
        target_feature = "v8.3a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_2;

#[cfg(all(
    target_feature = "v8.1a",
    not(any(
        target_feature = "v9.5a",
        target_feature = "v9.4a",
        target_feature = "v9.3a",
        target_feature = "v9.2a",
        target_feature = "v9.1a",
        target_feature = "v9a",
        target_feature = "v8.9a",
        target_feature = "v8.8a",
        target_feature = "v8.7a",
        target_feature = "v8.6a",
        target_feature = "v8.5a",
        target_feature = "v8.4a",
        target_feature = "v8.3a",
        target_feature = "v8.2a"
    ))
))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_1;

#[cfg(not(any(
    target_feature = "v9.5a",
    target_feature = "v9.4a",
    target_feature = "v9.3a",
    target_feature = "v9.2a",
    target_feature = "v9.1a",
    target_feature = "v9a",
    target_feature = "v8.9a",
    target_feature = "v8.8a",
    target_feature = "v8.7a",
    target_feature = "v8.6a",
    target_feature = "v8.5a",
    target_feature = "v8.4a",
    target_feature = "v8.3a",
    target_feature = "v8.2a",
    target_feature = "v8.1a"
)))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_0;

/// Compile-time feature **floor** from `-C target-feature` / `-C target-cpu` /
/// `-march`, as a [`FeatureSet`]. A *lower* bound, never an upper one: it is
/// `union`ed into [`super::detect_features`] so a feature the compiler was
/// told to assume is reported even where the runtime probe cannot see it
/// (bare-metal, or a HWCAP bit the running kernel does not export).
///
/// Mirrors x86's [`crate::x86::shortpath::compile_time_features`]. Covers every
/// [`Feature`] an `aarch64::ops` token checks, plus the `v8.x`/`v9.x`
/// architecture cfgs' mandatory implications - unlike `COMPILE_TIME_LEVEL`
/// above, which only buckets into [`ArmLevel`].
///
/// `cfg!` (not `#[cfg]`) so this stays one straight-line function; `Feature::bit`
/// is not `const`-callable, which is why this is a plain `fn` and not a `const`.
#[inline]
pub fn compile_time_features() -> FeatureSet {
    let mut set = FeatureSet::default();

    // AArch64 architectural floor: scalar FP and NEON are mandatory.
    set = set.with(Feature::Fp);
    set = set.with(Feature::Neon);

    if cfg!(target_feature = "aes") {
        // LLVM's `aes` covers AES + PMULL as one unit (Arm FEAT_AES/FEAT_PMULL).
        set = set.with(Feature::Aes);
        set = set.with(Feature::Pmull);
    }
    if cfg!(target_feature = "sha2") {
        // LLVM's `sha2` is FEAT_SHA1 + FEAT_SHA256 together.
        set = set.with(Feature::Sha1);
        set = set.with(Feature::Sha2);
    }
    if cfg!(target_feature = "sha3") {
        // LLVM's `sha3` is FEAT_SHA3 + FEAT_SHA512 together.
        set = set.with(Feature::Sha3);
        set = set.with(Feature::Sha512);
    }
    if cfg!(target_feature = "sm4") {
        // LLVM's `sm4` is FEAT_SM3 + FEAT_SM4 together.
        set = set.with(Feature::Sm3);
        set = set.with(Feature::Sm4);
    }
    if cfg!(target_feature = "crc") {
        set = set.with(Feature::Crc32);
    }
    if cfg!(target_feature = "lse") {
        set = set.with(Feature::Lse);
    }
    if cfg!(target_feature = "rdm") {
        set = set.with(Feature::Rdm);
    }
    if cfg!(target_feature = "fp16") {
        // FEAT_FP16 is both the scalar and the NEON half-precision bit.
        set = set.with(Feature::Fp16);
        set = set.with(Feature::AsimdHp);
    }
    if cfg!(target_feature = "fhm") {
        set = set.with(Feature::Fhm);
    }
    if cfg!(target_feature = "dotprod") {
        set = set.with(Feature::Dotprod);
    }
    if cfg!(target_feature = "jsconv") {
        set = set.with(Feature::Jscvt);
    }
    if cfg!(target_feature = "fcma") {
        set = set.with(Feature::Fcma);
    }
    if cfg!(target_feature = "rcpc") {
        set = set.with(Feature::Rcpc);
    }
    if cfg!(target_feature = "rcpc2") {
        set = set.with(Feature::Rcpc2);
    }
    if cfg!(target_feature = "dpb") {
        set = set.with(Feature::Dcpop);
    }
    if cfg!(target_feature = "dpb2") {
        set = set.with(Feature::Dcpodp);
    }
    if cfg!(target_feature = "dit") {
        set = set.with(Feature::Dit);
    }
    if cfg!(target_feature = "flagm") {
        set = set.with(Feature::Flagm);
    }
    if cfg!(target_feature = "flagm2") {
        set = set.with(Feature::Flagm2);
    }
    if cfg!(target_feature = "ssbs") {
        set = set.with(Feature::Ssbs);
    }
    if cfg!(target_feature = "sb") {
        set = set.with(Feature::Sb);
    }
    if cfg!(target_feature = "bti") {
        set = set.with(Feature::Bti);
    }
    if cfg!(target_feature = "paca") {
        set = set.with(Feature::Paca);
    }
    if cfg!(target_feature = "pacg") {
        set = set.with(Feature::Pacg);
    }
    if cfg!(target_feature = "frintts") {
        set = set.with(Feature::Frint);
    }
    if cfg!(target_feature = "rand") {
        set = set.with(Feature::Rng);
    }
    if cfg!(target_feature = "i8mm") {
        set = set.with(Feature::I8mm);
    }
    if cfg!(target_feature = "bf16") {
        set = set.with(Feature::Bf16);
    }
    if cfg!(target_feature = "ecv") {
        set = set.with(Feature::Ecv);
    }
    if cfg!(target_feature = "wfxt") {
        set = set.with(Feature::Wfxt);
    }
    if cfg!(target_feature = "mte") {
        set = set.with(Feature::Mte);
    }
    if cfg!(target_feature = "sve") {
        set = set.with(Feature::Sve);
    }
    if cfg!(target_feature = "sve2") {
        set = set.with(Feature::Sve);
        set = set.with(Feature::Sve2);
    }
    if cfg!(target_feature = "sve2-aes") {
        set = set.with(Feature::Sve2Aes);
        set = set.with(Feature::Sve2Pmull);
    }
    if cfg!(target_feature = "sve2-bitperm") {
        set = set.with(Feature::Sve2Bitperm);
    }
    if cfg!(target_feature = "sve2-sha3") {
        set = set.with(Feature::Sve2Sha3);
    }
    if cfg!(target_feature = "sve2-sm4") {
        set = set.with(Feature::Sve2Sm4);
    }
    if cfg!(target_feature = "i8mm") && cfg!(target_feature = "sve") {
        set = set.with(Feature::Sve2I8mm);
    }
    if cfg!(target_feature = "f32mm") {
        set = set.with(Feature::Sve2F32mm);
    }
    if cfg!(target_feature = "f64mm") {
        set = set.with(Feature::Sve2F64mm);
    }
    if cfg!(target_feature = "bf16") && cfg!(target_feature = "sve") {
        set = set.with(Feature::Sve2Bf16);
    }
    if cfg!(target_feature = "sme") {
        set = set.with(Feature::Sme);
    }

    set
}

/// Assumed above V8_0 vs still need runtime `getauxval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
    /// Assumed via march/target-cpu/feature.
    Assumed(ArmLevel),
    /// Only V8_0/NEON known; use [`ArmLevel::detect`].
    Baseline,
}

/// Skip runtime dispatch above NEON baseline if assumed.
pub const fn resolve() -> ResolvedPath {
    match COMPILE_TIME_LEVEL {
        ArmLevel::V8_0 => ResolvedPath::Baseline,
        level => ResolvedPath::Assumed(level),
    }
}

/// Process-start re-check via auxv/sysctl/PF. No-op on Baseline and on
/// compile-time-only platforms (no runtime probe to disprove Assumed).
///
/// # Panics
/// Assumed level missing on this CPU (when a runtime probe exists).
pub fn verify_or_panic() {
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_vendor = "apple",
        target_os = "windows"
    )))]
    {
        return;
    }
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_vendor = "apple",
        target_os = "windows"
    ))]
    if let ResolvedPath::Assumed(level) = resolve() {
        let set = FeatureSet::detect();
        if !set.contains_all(level.required_features()) {
            panic!(
                "miraculix: this binary was compiled assuming AArch64 {level:?} (via `-march=...`, \
				 `-C target-cpu=native`, or explicit `-C target-feature=...`), but the CPU it is \
				 running on does not support that level. Recompile for a lower level (or plain \
				 AArch64 baseline), or run this binary on compatible hardware."
            );
        }
    }
}
