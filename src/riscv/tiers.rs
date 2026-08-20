//! Coarse tiers over [`super::Feature`]. On `riscv*gc-*-linux-*`, M/A/F/D/C
//! are ABI floor (like NEON on AArch64). Only [`Feature::V`] is HW-dependent.
//! No RVA22/RVA23 table: Zba/Zbb/... not in Feature.

use super::features::{Feature, FeatureSet};

/// Enum list for RISC-V levels. Gc always (gc targets); no empty state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum RiscvLevel {
	/// gc floor: M A F D C.
	Gc = 0,
	/// + V.
	Vector = 1,
}

impl RiscvLevel {
	pub const ALL: &'static [RiscvLevel] = &[RiscvLevel::Gc, RiscvLevel::Vector];

	/// Cumulative required features. Gc = ABI floor.
	pub fn required_features(self) -> &'static [Feature] {
		const GC: &[Feature] = &[Feature::M, Feature::A, Feature::F, Feature::D, Feature::C];
		const VECTOR: &[Feature] =
			&[Feature::M, Feature::A, Feature::F, Feature::D, Feature::C, Feature::V];

		match self {
			RiscvLevel::Gc => GC,
			RiscvLevel::Vector => VECTOR,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		RiscvLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(RiscvLevel::Gc)
	}
}
