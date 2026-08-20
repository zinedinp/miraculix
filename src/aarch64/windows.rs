//! Windows on Arm tiers + shortpath (Snapdragon / Oryon). Not [`super::ArmLevel`]:
//! no `PF_ARM_*` for RDM, so ArmLevel sticks at V8_0. `Base` = NEON floor;
//! `XElite` = all windows probe can prove. Shortpath key: `target_feature=sm4`
//! (only `oryon-1` among listed aarch64 target-cpus).

use crate::level_cache::CachedU8;
use super::features::{Feature, FeatureSet};

/// Enum list for Windows on Arm levels. `Base` = no assumption beyond NEON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SnapdragonLevel {
	/// No assumption beyond NEON (the whole target's floor).
	Base = 0,
	/// Everything provable via `IsProcessorFeaturePresent` today: classic
	/// AES/PMULL/SHA1/SHA2 crypto bundle, CRC32, LSE, DotProd, JSCVT, RCPC.
	XElite = 1,
}

impl SnapdragonLevel {
	pub const ALL: &'static [SnapdragonLevel] = &[SnapdragonLevel::Base, SnapdragonLevel::XElite];

	/// Cumulative required features. `Base` = empty (NEON always).
	pub fn required_features(self) -> &'static [Feature] {
		const X_ELITE: &[Feature] = &[
			Feature::Aes,
			Feature::Pmull,
			Feature::Sha1,
			Feature::Sha2,
			Feature::Crc32,
			Feature::Lse,
			Feature::Dotprod,
			Feature::Jscvt,
			Feature::Rcpc,
		];

		match self {
			SnapdragonLevel::Base => &[],
			SnapdragonLevel::XElite => X_ELITE,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		SnapdragonLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(SnapdragonLevel::Base)
	}
}

#[cfg(target_feature = "sm4")]
const COMPILE_TIME_LEVEL: SnapdragonLevel = SnapdragonLevel::XElite;

#[cfg(not(target_feature = "sm4"))]
const COMPILE_TIME_LEVEL: SnapdragonLevel = SnapdragonLevel::Base;

/// Assumed above `Base` vs still need runtime `IsProcessorFeaturePresent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// `XElite` via `-C target-cpu=oryon-1`; no runtime check for that level.
	Assumed(SnapdragonLevel),
	/// Only `Base`/NEON known; use [`SnapdragonLevel::detect`].
	Baseline,
}

/// Skip runtime `IsProcessorFeaturePresent` if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		SnapdragonLevel::Base => ResolvedPath::Baseline,
		level => ResolvedPath::Assumed(level),
	}
}

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<SnapdragonLevel> {
	match v {
		0 => Some(SnapdragonLevel::Base),
		1 => Some(SnapdragonLevel::XElite),
		_ => None,
	}
}

/// Best level (process cache). First call: shortpath or `IsProcessorFeaturePresent`.
pub fn detect_level() -> SnapdragonLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached SnapdragonLevel discriminant")
}

/// Always shortpath or WinAPI now; ignores process cache.
pub fn detect_level_fresh() -> SnapdragonLevel {
	match resolve() {
		ResolvedPath::Assumed(level) => level,
		ResolvedPath::Baseline => SnapdragonLevel::detect(FeatureSet::detect()),
	}
}

/// Optional early fill so the first hot path is a cache hit.
pub fn warm_up() {
	let _ = detect_level();
}

/// Process-start re-check via `IsProcessorFeaturePresent`. No-op on Baseline.
///
/// # Panics
/// Assumed level missing (emulation, or a future WinSDK/oryon-N mismatch).
pub fn verify_or_panic() {
	if let ResolvedPath::Assumed(level) = resolve() {
		let set = FeatureSet::detect();
		if !set.contains_all(level.required_features()) {
			panic!(
				"miraculix: this binary was compiled assuming Windows on Arm {level:?} (via `-C \
				 target-cpu=oryon-1`), but the CPU it is running on does not support that level. \
				 Recompile without a Snapdragon-specific target-cpu (or plain aarch64 baseline), or \
				 run this binary on Snapdragon X Elite/Plus hardware."
			);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Higher tiers supersets of lower (copy-paste guard).
	#[test]
	fn snapdragon_levels_are_cumulative() {
		for pair in SnapdragonLevel::ALL.windows(2) {
			let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
			assert!(
				lower.iter().all(|f| higher.contains(f)),
				"{:?} must be a superset of {:?}",
				pair[1],
				pair[0]
			);
		}
	}

	/// `detect` matches manual required-features on the same set.
	#[test]
	fn snapdragon_level_matches_manual_check() {
		let set = FeatureSet::detect();
		let detected = SnapdragonLevel::detect(set);

		for &level in SnapdragonLevel::ALL {
			let should_hold = set.contains_all(level.required_features());
			assert_eq!(
				level <= detected,
				should_hold || level == SnapdragonLevel::Base,
				"level {level:?} required-features check disagrees with detect() result"
			);
		}
	}

	/// No oryon-1 target-cpu in plain `cargo test`: verify no-op.
	#[test]
	fn verify_or_panic_is_a_no_op_without_a_compile_time_snapdragon_level() {
		verify_or_panic();
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
