//! Compile-time collapse of runtime dispatch. Baseline [`ArmLevel::V8_0`]
//! (NEON always). Cascade on rustc `v8.1a`..`v9a` cfgs (authoritative).

use super::tiers::ArmLevel;
#[cfg(any(
	target_os = "linux",
	target_os = "android",
	target_os = "freebsd",
	target_os = "macos",
	target_os = "windows"
))]
use super::FeatureSet;

#[cfg(target_feature = "v9a")]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V9_0;

#[cfg(all(not(target_feature = "v9a"), target_feature = "v8.7a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_7;

#[cfg(all(not(target_feature = "v8.7a"), target_feature = "v8.6a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_6;

#[cfg(all(not(target_feature = "v8.6a"), target_feature = "v8.5a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_5;

#[cfg(all(not(target_feature = "v8.5a"), target_feature = "v8.4a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_4;

#[cfg(all(not(target_feature = "v8.4a"), target_feature = "v8.3a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_3;

#[cfg(all(not(target_feature = "v8.3a"), target_feature = "v8.2a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_2;

#[cfg(all(not(target_feature = "v8.2a"), target_feature = "v8.1a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_1;

#[cfg(not(target_feature = "v8.1a"))]
const COMPILE_TIME_LEVEL: ArmLevel = ArmLevel::V8_0;

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
		target_os = "macos",
		target_os = "windows"
	)))]
	{
		return;
	}
	#[cfg(any(
		target_os = "linux",
		target_os = "android",
		target_os = "freebsd",
		target_os = "macos",
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
