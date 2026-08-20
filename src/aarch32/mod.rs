//! AArch32 VFP/NEON: [`features`], [`tiers`] FPU levels, [`shortpath`].
//! Auxv on Linux/Android/FreeBSD; compile-time only elsewhere. No Windows CE
//! rustc target (TODO). Once-global: [`detect_level`]. Token-gated ops
//! behind `nightly-arm-neon` (unstable `core::arch::arm`): [`ops`].

use crate::level_cache::CachedU8;
use features::CachedFeatureSet;

pub mod features;
/// Token-gated DSP/SIMD32 and Neon ops. Needs `nightly-arm-neon` and a
/// nightly toolchain (`core::arch::arm` DSP/SIMD32 and Neon are unstable).
#[cfg(feature = "nightly-arm-neon")]
pub mod ops;
pub mod shortpath;
pub mod tiers;

/// Elementwise slice work without picking a token by hand (see [`auto_up`]).
/// Needs [`ops::neon`], so shares its `v7` compile-time gate. A sub-`v7`
/// build only gets [`ops::dsp`] directly (no `auto` yet; the `i8` family
/// could stand alone without Neon if something needs that later).
#[cfg(all(feature = "nightly-arm-neon", any(target_feature = "v7", doc)))]
mod auto_down;
#[cfg(all(feature = "nightly-arm-neon", any(target_feature = "v7", doc)))]
pub mod auto_up;
/// Short alias for [`auto_up`] (same module). Prefer `auto_up` in new code.
#[cfg(all(feature = "nightly-arm-neon", any(target_feature = "v7", doc)))]
pub use auto_up as auto;

pub use features::{Feature, FeatureSet};
pub use tiers::FpuLevel;

static CACHED_LEVEL: CachedU8 = CachedU8::new();
static CACHED_FEATURES: CachedFeatureSet = CachedFeatureSet::new();

/// Raw feature bits for this process (cached after first call). Prefer over
/// [`detect_features_fresh`] in normal code; used by [`auto_up`] so cascades
/// do not re-read `auxv` on every call.
pub fn detect_features() -> FeatureSet {
	CACHED_FEATURES.get_or_init(FeatureSet::detect)
}

/// Same answer as [`detect_features`], but always re-probes `auxv`.
///
/// Does not read or write the process cache. Prefer [`detect_features`] in
/// normal code.
pub fn detect_features_fresh() -> FeatureSet {
	FeatureSet::detect()
}

fn level_from_u8(v: u8) -> Option<FpuLevel> {
	match v {
		0 => Some(FpuLevel::None),
		1 => Some(FpuLevel::Vfpv3Neon),
		2 => Some(FpuLevel::Vfpv4Neon),
		3 => Some(FpuLevel::Crypto),
		_ => None,
	}
}

/// Best FPU/NEON tier for this process (cached after first call).
///
/// No app `init` required. Prefer over [`detect_level_fresh`] in normal code.
pub fn detect_level() -> FpuLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached FpuLevel discriminant")
}

/// Same as [`detect_level`] but always re-probes (no cache). Tests / re-audit only.
pub fn detect_level_fresh() -> FpuLevel {
	match shortpath::resolve() {
		shortpath::ResolvedPath::Assumed(level) => level,
		shortpath::ResolvedPath::RuntimeDispatch => FpuLevel::detect(FeatureSet::detect()),
	}
}

/// Optional: fill the detect caches at startup. Not required.
pub fn warm_up() {
	let _ = detect_level();
	let _ = detect_features();
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `Feature::bit` keys the bitset; every variant needs a distinct index.
	#[test]
	fn every_feature_has_a_unique_bit() {
		let mut seen = std::collections::HashSet::new();
		for &feature in Feature::ALL {
			assert!(seen.insert(feature.bit()), "duplicate bit for {feature:?}");
		}
	}

	/// `detect` matches manual required-features on the same set.
	#[test]
	fn fpu_level_matches_manual_check() {
		let set = FeatureSet::detect();
		let detected = FpuLevel::detect(set);

		for &level in FpuLevel::ALL {
			let should_hold = set.contains_all(level.required_features());
			assert_eq!(
				level <= detected,
				should_hold || level == FpuLevel::None,
				"level {level:?} required-features check disagrees with detect() result"
			);
		}
	}

	/// Higher tiers supersets of lower (copy-paste guard).
	#[test]
	fn fpu_levels_are_cumulative() {
		for pair in FpuLevel::ALL.windows(2) {
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
	fn verify_or_panic_is_a_no_op_without_a_compile_time_fpu_level() {
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

	/// Cached path matches a fresh probe.
	#[test]
	fn detect_features_matches_fresh() {
		assert_eq!(detect_features(), detect_features_fresh());
	}

	/// Repeated calls are stable (process cache).
	#[test]
	fn detect_features_is_stable_across_calls() {
		let a = detect_features();
		let b = detect_features();
		assert_eq!(a, b);
	}
}

#[cfg(all(test, feature = "nightly-arm-neon", any(target_feature = "v7", doc)))]
mod auto_tests {
	use super::{auto, auto_up};

	/// `aarch32::auto` is only a re-export of `auto_up` (same functions).
	#[test]
	fn auto_alias_matches_auto_up() {
		let a = [1i32, -2, 3, -4];
		let b = [10, 20, -30, 40];
		let mut via_alias = [0i32; 4];
		let mut via_auto_up = [0i32; 4];
		auto::add_i32(&a, &b, &mut via_alias);
		auto_up::add_i32(&a, &b, &mut via_auto_up);
		assert_eq!(via_alias, via_auto_up);
	}
}
