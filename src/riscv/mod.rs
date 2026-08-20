//! RISC-V RV32/RV64: [`features`] (M/A/F/D/C/V), [`tiers`] [`RiscvLevel`],
//! [`shortpath`]. Auxv on Linux/Android/FreeBSD; compile-time elsewhere. One
//! module both widths. Finer ISA needs `riscv_hwprobe(2)` (not here).

use crate::level_cache::CachedU8;

pub mod features;
pub mod shortpath;
pub mod tiers;

pub use features::{Feature, FeatureSet};
pub use tiers::RiscvLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<RiscvLevel> {
	match v {
		0 => Some(RiscvLevel::Gc),
		1 => Some(RiscvLevel::Vector),
		_ => None,
	}
}

/// Best capability tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
pub fn detect_level() -> RiscvLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached RiscvLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
pub fn detect_level_fresh() -> RiscvLevel {
	match shortpath::resolve() {
		shortpath::ResolvedPath::Assumed(level) => level,
		shortpath::ResolvedPath::Baseline => RiscvLevel::detect(FeatureSet::detect()),
	}
}

/// Optional: fill the detect cache at startup. Not required.
pub fn warm_up() {
	let _ = detect_level();
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Gc floor: M/A/F/D/C on `riscv{32,64}gc-*-linux-*`.
	#[test]
	fn gc_is_always_available() {
		assert!(FeatureSet::detect().contains_all(RiscvLevel::Gc.required_features()));
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
	fn riscv_level_matches_manual_check() {
		let set = FeatureSet::detect();
		let detected = RiscvLevel::detect(set);

		for &level in RiscvLevel::ALL {
			let should_hold = set.contains_all(level.required_features());
			assert_eq!(
				level <= detected,
				should_hold || level == RiscvLevel::Gc,
				"level {level:?} required-features check disagrees with detect() result"
			);
		}
	}

	/// Higher tiers supersets of lower (copy-paste guard).
	#[test]
	fn riscv_levels_are_cumulative() {
		for pair in RiscvLevel::ALL.windows(2) {
			let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
			assert!(
				lower.iter().all(|f| higher.contains(f)),
				"{:?} must be a superset of {:?}",
				pair[1],
				pair[0]
			);
		}
	}

	/// No `+v` in plain `cargo test`: shortpath baseline, verify no-op.
	#[test]
	fn verify_or_panic_is_a_no_op_without_a_compile_time_riscv_level() {
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
