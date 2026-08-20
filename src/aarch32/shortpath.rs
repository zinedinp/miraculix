//! Compile-time collapse of runtime dispatch. Caps at [`FpuLevel::Vfpv4Neon`]:
//! Crypto needs runtime (cfg has aes, no pmull). One cascade on target_feature.

use super::tiers::FpuLevel;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use super::FeatureSet;

#[cfg(all(target_feature = "vfp4", target_feature = "d32", target_feature = "neon"))]
const COMPILE_TIME_LEVEL: Option<FpuLevel> = Some(FpuLevel::Vfpv4Neon);

#[cfg(all(
	not(all(target_feature = "vfp4", target_feature = "d32", target_feature = "neon")),
	target_feature = "vfp3",
	target_feature = "d32",
	target_feature = "neon",
))]
const COMPILE_TIME_LEVEL: Option<FpuLevel> = Some(FpuLevel::Vfpv3Neon);

#[cfg(not(all(target_feature = "vfp3", target_feature = "d32", target_feature = "neon")))]
const COMPILE_TIME_LEVEL: Option<FpuLevel> = None;

/// Assumed level vs still need runtime `getauxval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// Assumed via march/target-cpu/feature.
	Assumed(FpuLevel),
	/// Nothing assumed; use [`FpuLevel::detect`].
	RuntimeDispatch,
}

/// Skip runtime dispatch if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		Some(level) => ResolvedPath::Assumed(level),
		None => ResolvedPath::RuntimeDispatch,
	}
}

/// Process-start re-check via auxv. No-op on RuntimeDispatch and on
/// compile-time-only platforms.
///
/// # Panics
/// Assumed level missing on this CPU (when a runtime probe exists).
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
				"miraculix: this binary was compiled assuming AArch32 {level:?} (via `-march=...`, \
				 `-C target-cpu=native`, or explicit `-C target-feature=...`), but the CPU it is \
				 running on does not support that level. Recompile for a lower level (or plain \
				 AArch32 baseline), or run this binary on compatible hardware."
			);
		}
	}
}
