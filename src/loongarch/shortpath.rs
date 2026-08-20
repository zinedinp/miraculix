//! Compile-time collapse. Keys: `lsx` / `lasx` target features.

use super::tiers::LoongArchLevel;
#[cfg(any(target_os = "linux", target_os = "android"))]
use super::FeatureSet;

#[cfg(target_feature = "lasx")]
const COMPILE_TIME_LEVEL: LoongArchLevel = LoongArchLevel::Lasx;

#[cfg(all(target_feature = "lsx", not(target_feature = "lasx")))]
const COMPILE_TIME_LEVEL: LoongArchLevel = LoongArchLevel::Lsx;

#[cfg(not(target_feature = "lsx"))]
const COMPILE_TIME_LEVEL: LoongArchLevel = LoongArchLevel::Scalar;

/// Assumed above Scalar vs still need runtime (or cfg-only) probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// Assumed via target-feature / target-cpu.
	Assumed(LoongArchLevel),
	/// Only Scalar known from compile-time; use [`LoongArchLevel::detect`].
	Baseline,
}

/// Skip runtime dispatch above Scalar if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		LoongArchLevel::Scalar => ResolvedPath::Baseline,
		level => ResolvedPath::Assumed(level),
	}
}

/// Re-check Assumed level when a runtime probe exists. No-op on Baseline and
/// on compile-time-only platforms (no auxv).
///
/// # Panics
/// Assumed LSX/LASX missing on this CPU (Linux/Android only).
pub fn verify_or_panic() {
	#[cfg(not(any(target_os = "linux", target_os = "android")))]
	{
		return;
	}
	#[cfg(any(target_os = "linux", target_os = "android"))]
	if let ResolvedPath::Assumed(level) = resolve() {
		let set = FeatureSet::detect();
		if !set.contains_all(level.required_features()) {
			panic!(
				"miraculix: this binary was compiled assuming LoongArch {level:?} (via \
				 `-C target-feature=+lsx/+lasx` or a matching target-cpu), but the CPU it is \
				 running on does not support that level. Recompile without those features, or \
				 run on hardware with LSX/LASX enabled in the kernel."
			);
		}
	}
}
