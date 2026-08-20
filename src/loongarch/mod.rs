//! LoongArch64: [`features`] (LSX/LASX), [`tiers`] [`LoongArchLevel`],
//! [`shortpath`]. Linux `getauxval`; bare-metal compile-time only.
//! Once-global: [`detect_level`].

use crate::level_cache::CachedU8;

pub mod features;
pub mod shortpath;
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::LoongArchLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<LoongArchLevel> {
	match v {
		0 => Some(LoongArchLevel::Scalar),
		1 => Some(LoongArchLevel::Lsx),
		2 => Some(LoongArchLevel::Lasx),
		_ => None,
	}
}

/// Best capability tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
pub fn detect_level() -> LoongArchLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached LoongArchLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
pub fn detect_level_fresh() -> LoongArchLevel {
	match shortpath::resolve() {
		shortpath::ResolvedPath::Assumed(level) => level,
		shortpath::ResolvedPath::Baseline => LoongArchLevel::detect(FeatureSet::detect()),
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
	fn loongarch_levels_are_cumulative() {
		for pair in LoongArchLevel::ALL.windows(2) {
			let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
			assert!(
				lower.iter().all(|f| higher.contains(f)),
				"{:?} must be a superset of {:?}",
				pair[1],
				pair[0]
			);
		}
	}

	/// No `+lsx`/`+lasx` in plain `cargo test`: shortpath baseline, verify no-op.
	#[test]
	fn verify_or_panic_is_a_no_op_without_a_compile_time_loongarch_level() {
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
