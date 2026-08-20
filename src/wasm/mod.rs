//! WASM32/64 SIMD: [`features`] (simd128/relaxed-simd), [`tiers`] [`WasmLevel`],
//! [`shortpath`]. Compile-time only (no runtime probe; host validates opcodes).
//! One module for both widths. Once-global API for parity with other archs.

use crate::level_cache::CachedU8;

pub mod features;
pub mod shortpath;
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::WasmLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<WasmLevel> {
	match v {
		0 => Some(WasmLevel::Scalar),
		1 => Some(WasmLevel::Simd128),
		2 => Some(WasmLevel::RelaxedSimd),
		_ => None,
	}
}

/// WASM SIMD tier for this binary (compile-time only; cached for API parity).
///
/// There is no runtime probe on WASM. Prefer this over [`detect_level_fresh`].
pub fn detect_level() -> WasmLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached WasmLevel discriminant")
}

/// Same as [`detect_level`] (always re-resolves const shortpath; no HW cost).
pub fn detect_level_fresh() -> WasmLevel {
	shortpath::resolve()
}

/// Optional: fill the detect cache at startup. Not required.
pub fn warm_up() {
	let _ = detect_level();
}

#[cfg(test)]
mod tests {
	use super::*;

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
	fn wasm_level_matches_manual_check() {
		let set = FeatureSet::detect();
		let detected = WasmLevel::detect(set);

		for &level in WasmLevel::ALL {
			let should_hold = set.contains_all(level.required_features());
			assert_eq!(
				level <= detected,
				should_hold || level == WasmLevel::Scalar,
				"level {level:?} required-features check disagrees with detect() result"
			);
		}
	}

	/// Higher tiers supersets of lower (copy-paste guard).
	#[test]
	fn wasm_levels_are_cumulative() {
		for pair in WasmLevel::ALL.windows(2) {
			let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
			assert!(
				lower.iter().all(|f| higher.contains(f)),
				"{:?} must be a superset of {:?}",
				pair[1],
				pair[0]
			);
		}
	}

	/// fresh path is shortpath::resolve().
	#[test]
	fn detect_level_fresh_matches_shortpath_resolve() {
		assert_eq!(detect_level_fresh(), shortpath::resolve());
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
