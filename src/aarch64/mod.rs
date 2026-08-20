//! AArch64 NEON/SVE/SME: [`features`], [`tiers`] [`ArmLevel`], [`shortpath`].
//! Linux/Android `getauxval`; FreeBSD `elf_aux_info`; bare/other compile-time floor.
//! macOS [`AppleLevel`]; Windows [`SnapdragonLevel`]. Once-global: [`detect_level`].

// ArmLevel path (not Apple/Windows tiers).
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use crate::level_cache::CachedU8;

pub mod features;
pub mod shortpath;
pub mod tiers;

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
		8 => Some(ArmLevel::V9_0),
		_ => None,
	}
}

/// Best Arm tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn detect_level() -> ArmLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	arm_level_from_u8(v).expect("cached ArmLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn detect_level_fresh() -> ArmLevel {
	match shortpath::resolve() {
		shortpath::ResolvedPath::Assumed(level) => level,
		shortpath::ResolvedPath::Baseline => ArmLevel::detect(FeatureSet::detect()),
	}
}

/// Optional: fill the detect cache at startup. Not required.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn warm_up() {
	let _ = detect_level();
}

// macOS: forward AppleLevel.

/// Best Apple Silicon **tier** for this process (cached).
#[cfg(target_os = "macos")]
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
}

// Windows: forward SnapdragonLevel.

/// Best Windows-on-Arm **tier** for this process (cached).
#[cfg(target_os = "windows")]
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
}
