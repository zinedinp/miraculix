//! powerpc64 (powerpc64le first): [`features`] AltiVec/VSX, [`tiers`]
//! [`PowerLevel`], [`shortpath`]. Auxv on Linux/Android/FreeBSD; bare compile-time.
//! BE powerpc64 shares bits; dedicated BE policy is TODO. Once-global: [`detect_level`].
//! [`ops`]: per-extension token + wrapper API (AltiVec, behind `nightly-altivec`).

use crate::level_cache::CachedU8;

pub mod features;
pub mod ops;
pub mod shortpath;
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::PowerLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<PowerLevel> {
	match v {
		0 => Some(PowerLevel::Scalar),
		1 => Some(PowerLevel::Altivec),
		2 => Some(PowerLevel::Vsx),
		3 => Some(PowerLevel::Power8),
		4 => Some(PowerLevel::Power9),
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
	match shortpath::resolve() {
		shortpath::ResolvedPath::Assumed(level) => level,
		shortpath::ResolvedPath::Baseline => PowerLevel::detect(FeatureSet::detect()),
	}
}

/// Optional: fill the detect cache at startup. Not required.
pub fn warm_up() {
	let _ = detect_level();
}

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
