//! Apple Silicon tiers + shortpath. Not [`super::ArmLevel`]: `apple-mN` never
//! sets `v8.xa`/`v9a` cfgs. M1 is darwin floor; M2 means "M2 or later" (LLVM
//! shares m2/m3/m4 feature sets). New m3/m4-only ISA needs a new tier.

use crate::level_cache::CachedU8;
use super::features::{Feature, FeatureSet};

/// Enum list for Apple Silicon levels. M1 always; no empty state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AppleLevel {
	/// M1+: crypto, CRC32, LSE, FP16, RDM, JSCVT, FCMA, RCPC(2), DPB(2),
	/// DotProd, FHM, DIT, FlagM(2), SSBS, SB, PAC, FRINTTS.
	M1 = 0,
	/// M2+: + BF16, BTI, I8MM.
	M2 = 1,
}

impl AppleLevel {
	pub const ALL: &'static [AppleLevel] = &[AppleLevel::M1, AppleLevel::M2];

	/// Cumulative required features. M1 = floor.
	pub fn required_features(self) -> &'static [Feature] {
		const M1: &[Feature] = &[
			Feature::Aes, Feature::Pmull, Feature::Sha1, Feature::Sha2, Feature::Sha3,
			Feature::Sha512, Feature::Crc32, Feature::Lse, Feature::Fp16, Feature::Rdm,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Rcpc2, Feature::Dcpop,
			Feature::Dcpodp, Feature::Dotprod, Feature::Fhm, Feature::Dit, Feature::Flagm,
			Feature::Flagm2, Feature::Ssbs, Feature::Sb, Feature::Frint, Feature::Paca,
			Feature::Pacg,
		];
		const M2: &[Feature] = &[
			Feature::Aes, Feature::Pmull, Feature::Sha1, Feature::Sha2, Feature::Sha3,
			Feature::Sha512, Feature::Crc32, Feature::Lse, Feature::Fp16, Feature::Rdm,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Rcpc2, Feature::Dcpop,
			Feature::Dcpodp, Feature::Dotprod, Feature::Fhm, Feature::Dit, Feature::Flagm,
			Feature::Flagm2, Feature::Ssbs, Feature::Sb, Feature::Frint, Feature::Paca,
			Feature::Pacg, Feature::Bf16, Feature::Bti, Feature::I8mm,
		];

		match self {
			AppleLevel::M1 => M1,
			AppleLevel::M2 => M2,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		AppleLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(AppleLevel::M1)
	}
}

#[cfg(all(target_feature = "bf16", target_feature = "bti", target_feature = "i8mm"))]
const COMPILE_TIME_LEVEL: Option<AppleLevel> = Some(AppleLevel::M2);

#[cfg(not(all(target_feature = "bf16", target_feature = "bti", target_feature = "i8mm")))]
const COMPILE_TIME_LEVEL: Option<AppleLevel> = None;

/// Assumed level vs still need runtime `sysctlbyname`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// M2+ via `target-cpu=apple-m2/m3/m4`; no runtime check for that level.
	Assumed(AppleLevel),
	/// Only M1 floor at compile time; use [`AppleLevel::detect`].
	RuntimeDispatch,
}

/// Skip runtime `sysctlbyname` if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		Some(level) => ResolvedPath::Assumed(level),
		None => ResolvedPath::RuntimeDispatch,
	}
}

static CACHED_LEVEL: CachedU8 = CachedU8::new();

fn level_from_u8(v: u8) -> Option<AppleLevel> {
	match v {
		0 => Some(AppleLevel::M1),
		1 => Some(AppleLevel::M2),
		_ => None,
	}
}

/// Best level (process cache). First call: shortpath or `sysctlbyname`.
pub fn detect_level() -> AppleLevel {
	let v = CACHED_LEVEL.get_or_init(|| detect_level_fresh() as u8);
	level_from_u8(v).expect("cached AppleLevel discriminant")
}

/// Always shortpath or `sysctlbyname` now; ignores process cache.
pub fn detect_level_fresh() -> AppleLevel {
	match resolve() {
		ResolvedPath::Assumed(level) => level,
		ResolvedPath::RuntimeDispatch => AppleLevel::detect(FeatureSet::detect()),
	}
}

/// Optional early fill so the first hot path is a cache hit.
pub fn warm_up() {
	let _ = detect_level();
}

/// Process-start re-check via `sysctlbyname`. No-op on RuntimeDispatch.
///
/// # Panics
/// Assumed level missing (emulation / future sysctl change; not normal M2 metal).
pub fn verify_or_panic() {
	if let ResolvedPath::Assumed(level) = resolve() {
		let set = FeatureSet::detect();
		if !set.contains_all(level.required_features()) {
			panic!(
				"miraculix: this binary was compiled assuming Apple Silicon {level:?} (via \
				 `-C target-cpu=apple-m2/apple-m3/apple-m4`), but the CPU it is running on does \
				 not support that level. Recompile for `apple-m1` (or plain aarch64-apple-darwin \
				 baseline), or run this binary on newer Apple Silicon hardware."
			);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Higher tiers supersets of lower (copy-paste guard).
	#[test]
	fn apple_levels_are_cumulative() {
		for pair in AppleLevel::ALL.windows(2) {
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
	fn apple_level_matches_manual_check() {
		let set = FeatureSet::detect();
		let detected = AppleLevel::detect(set);

		for &level in AppleLevel::ALL {
			let should_hold = set.contains_all(level.required_features());
			assert_eq!(
				level <= detected,
				should_hold || level == AppleLevel::M1,
				"level {level:?} required-features check disagrees with detect() result"
			);
		}
	}

	/// No apple-mN in plain `cargo test`: verify no-op.
	#[test]
	fn verify_or_panic_is_a_no_op_without_a_compile_time_apple_level() {
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
