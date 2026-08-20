//! Coarse tiers over [`super::Feature`]. Aimed at powerpc64le; same bits on BE
//! powerpc64 when that path is enabled. POWER8/9 fold vector crypto into gen.

use super::features::{Feature, FeatureSet};

/// Enum list for Power vector levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PowerLevel {
	/// No AltiVec/VSX proven.
	Scalar = 0,
	/// AltiVec / VMX.
	Altivec = 1,
	/// + VSX.
	Vsx = 2,
	/// + POWER8 / v2.07 vector.
	Power8 = 3,
	/// + POWER9 / v3.0 vector.
	Power9 = 4,
}

impl PowerLevel {
	pub const ALL: &'static [PowerLevel] = &[
		PowerLevel::Scalar,
		PowerLevel::Altivec,
		PowerLevel::Vsx,
		PowerLevel::Power8,
		PowerLevel::Power9,
	];

	/// Cumulative required features. Scalar = empty.
	pub fn required_features(self) -> &'static [Feature] {
		const ALTIVEC: &[Feature] = &[Feature::Altivec];
		const VSX: &[Feature] = &[Feature::Altivec, Feature::Vsx];
		const POWER8: &[Feature] = &[
			Feature::Altivec,
			Feature::Vsx,
			Feature::Power8,
			Feature::Power8Altivec,
			Feature::Power8Vector,
		];
		const POWER9: &[Feature] = &[
			Feature::Altivec,
			Feature::Vsx,
			Feature::Power8,
			Feature::Power8Altivec,
			Feature::Power8Vector,
			Feature::Power9,
			Feature::Power9Altivec,
			Feature::Power9Vector,
		];

		match self {
			PowerLevel::Scalar => &[],
			PowerLevel::Altivec => ALTIVEC,
			PowerLevel::Vsx => VSX,
			PowerLevel::Power8 => POWER8,
			PowerLevel::Power9 => POWER9,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		PowerLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(PowerLevel::Scalar)
	}
}
