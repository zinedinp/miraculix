//! Compile-time collapse: the level the build already guarantees.
//!
//! Keys are `altivec` / `vsx` / `power8-vector` / `power9-vector` /
//! `power10-vector` (all still unstable, so only visible on nightly with
//! `-C target-feature` or `-C target-cpu`) and the PS3 target, whose PPE always
//! has AltiVec. On stable everything else resolves at run time.

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use super::FeatureSet;
use super::tiers::PowerLevel;

const COMPILE_TIME_LEVEL: PowerLevel = if cfg!(target_feature = "power10-vector") {
    PowerLevel::Isa31
} else if cfg!(target_feature = "power9-vector") {
    PowerLevel::Isa300
} else if cfg!(target_feature = "power8-vector") {
    PowerLevel::Isa207
} else if cfg!(target_feature = "vsx") {
    PowerLevel::Vsx
} else if cfg!(any(target_feature = "altivec", target_os = "ps3")) {
    PowerLevel::Altivec
} else {
    PowerLevel::Scalar
};

/// Assumed above Scalar vs still need runtime probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
    /// Assumed via target-feature / target-cpu / the PS3 target.
    Assumed(PowerLevel),
    /// Only Scalar known from compile-time; use [`PowerLevel::detect`].
    Baseline,
}

/// Skip runtime dispatch above Scalar if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
    match COMPILE_TIME_LEVEL {
        PowerLevel::Scalar => ResolvedPath::Baseline,
        level => ResolvedPath::Assumed(level),
    }
}

/// Re-check Assumed level when a runtime probe exists. No-op on Baseline and
/// on platforms without an OS probe.
///
/// # Panics
/// Assumed level missing on this CPU (Linux/Android/FreeBSD).
pub fn verify_or_panic() {
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
    {
        return;
    }
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    if let ResolvedPath::Assumed(level) = resolve() {
        let set = FeatureSet::detect();
        if !set.contains_all(level.required_features()) {
            panic!(
                "miraculix: this binary was compiled assuming Power {level:?} (via \
                 `-C target-feature` or a matching target-cpu), but the CPU it is \
                 running on does not support that level. Recompile without those \
                 features, or run on compatible POWER hardware."
            );
        }
    }
}
