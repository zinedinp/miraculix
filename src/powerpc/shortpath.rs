//! Compile-time collapse. Keys: `altivec` / `vsx` (stable target features).
//! POWER8/9 need runtime (unstable cfg names in rustc).

use super::tiers::PowerLevel;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use super::FeatureSet;

#[cfg(target_feature = "vsx")]
const COMPILE_TIME_LEVEL: PowerLevel = PowerLevel::Vsx;

#[cfg(all(target_feature = "altivec", not(target_feature = "vsx")))]
const COMPILE_TIME_LEVEL: PowerLevel = PowerLevel::Altivec;

#[cfg(not(target_feature = "altivec"))]
const COMPILE_TIME_LEVEL: PowerLevel = PowerLevel::Scalar;

/// Assumed above Scalar vs still need runtime probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// Assumed via target-feature / target-cpu.
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
/// on compile-time-only platforms.
///
/// # Panics
/// Assumed AltiVec/VSX missing on this CPU (Linux/Android/FreeBSD).
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
				 `-C target-feature=+altivec/+vsx` or a matching target-cpu), but the CPU it is \
				 running on does not support that level. Recompile without those features, or \
				 run on compatible POWER hardware."
			);
		}
	}
}
