//! Coarse tiers over [`super::Feature`]. LSX is the usual la64 Linux floor
//! (`loongarch64-unknown-linux-gnu` requires LSX); Scalar is for soft/none.

use super::features::{Feature, FeatureSet};

/// Enum list for LoongArch SIMD levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LoongArchLevel {
	/// No LSX/LASX proven (scalar / softfloat).
	Scalar = 0,
	/// + LSX 128-bit.
	Lsx = 1,
	/// + LASX 256-bit.
	Lasx = 2,
}

impl LoongArchLevel {
	pub const ALL: &'static [LoongArchLevel] =
		&[LoongArchLevel::Scalar, LoongArchLevel::Lsx, LoongArchLevel::Lasx];

	/// Cumulative required features. Scalar = empty.
	pub fn required_features(self) -> &'static [Feature] {
		const LSX: &[Feature] = &[Feature::Lsx];
		const LASX: &[Feature] = &[Feature::Lsx, Feature::Lasx];

		match self {
			LoongArchLevel::Scalar => &[],
			LoongArchLevel::Lsx => LSX,
			LoongArchLevel::Lasx => LASX,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		LoongArchLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(LoongArchLevel::Scalar)
	}
}
