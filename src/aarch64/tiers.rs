//! Coarse arch tiers over [`super::Feature`] (~ GCC `-march=armv8.x-a`/`v9a`).
//! Approx (optionals != clean versions); truth is FeatureSet. macOS caps without
//! Uscat: use [`super::AppleLevel`]. Windows lacks RDM `PF_ARM_*`: use
//! [`super::SnapdragonLevel`].

use super::features::{Feature, FeatureSet};

/// Enum list for ARMv8/v9. NEON always => >= V8_0; no empty state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ArmLevel {
	/// ARMv8.0-A: NEON; opt AES/PMULL/SHA/CRC32.
	V8_0 = 0,
	/// ARMv8.1-A: +LSE, RDM, mand. CRC32.
	V8_1 = 1,
	/// ARMv8.2-A: +DC clean to PoP.
	V8_2 = 2,
	/// ARMv8.3-A: +JSCVT, FCMA, RCPC, PAC.
	V8_3 = 3,
	/// ARMv8.4-A: +DIT, Uscat, RCPC2, FLAGM, mand. DotProd.
	V8_4 = 4,
	/// ARMv8.5-A: +SSBS, SB, BTI, RNG, FRINT, FLAGM2, DCPODP.
	V8_5 = 5,
	/// ARMv8.6-A: +I8MM, BF16, ECV.
	V8_6 = 6,
	/// ARMv8.7-A: +WFET/WFIT, AFP, RPRES.
	V8_7 = 7,
	/// ARMv9.0-A: mand. SVE2 (SVE opt since 8.2).
	V9_0 = 8,
}

impl ArmLevel {
	pub const ALL: &'static [ArmLevel] = &[
		ArmLevel::V8_0, ArmLevel::V8_1, ArmLevel::V8_2, ArmLevel::V8_3, ArmLevel::V8_4,
		ArmLevel::V8_5, ArmLevel::V8_6, ArmLevel::V8_7, ArmLevel::V9_0,
	];

	/// Cumulative required features. V8_0 = empty (NEON always).
	pub fn required_features(self) -> &'static [Feature] {
		const V8_1: &[Feature] = &[Feature::Lse, Feature::Rdm, Feature::Crc32];
		const V8_2: &[Feature] = &[Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop];
		const V8_3: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
		];
		const V8_4: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
			Feature::Dit, Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Dotprod,
		];
		const V8_5: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
			Feature::Dit, Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Dotprod,
			Feature::Ssbs, Feature::Sb, Feature::Bti, Feature::Rng, Feature::Frint,
			Feature::Flagm2, Feature::Dcpodp,
		];
		const V8_6: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
			Feature::Dit, Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Dotprod,
			Feature::Ssbs, Feature::Sb, Feature::Bti, Feature::Rng, Feature::Frint,
			Feature::Flagm2, Feature::Dcpodp,
			Feature::I8mm, Feature::Bf16, Feature::Ecv,
		];
		const V8_7: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
			Feature::Dit, Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Dotprod,
			Feature::Ssbs, Feature::Sb, Feature::Bti, Feature::Rng, Feature::Frint,
			Feature::Flagm2, Feature::Dcpodp,
			Feature::I8mm, Feature::Bf16, Feature::Ecv,
			Feature::Wfxt, Feature::Afp, Feature::Rpres,
		];
		const V9_0: &[Feature] = &[
			Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop,
			Feature::Jscvt, Feature::Fcma, Feature::Rcpc, Feature::Paca, Feature::Pacg,
			Feature::Dit, Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Dotprod,
			Feature::Ssbs, Feature::Sb, Feature::Bti, Feature::Rng, Feature::Frint,
			Feature::Flagm2, Feature::Dcpodp,
			Feature::I8mm, Feature::Bf16, Feature::Ecv,
			Feature::Wfxt, Feature::Afp, Feature::Rpres,
			Feature::Sve, Feature::Sve2,
		];

		match self {
			ArmLevel::V8_0 => &[],
			ArmLevel::V8_1 => V8_1,
			ArmLevel::V8_2 => V8_2,
			ArmLevel::V8_3 => V8_3,
			ArmLevel::V8_4 => V8_4,
			ArmLevel::V8_5 => V8_5,
			ArmLevel::V8_6 => V8_6,
			ArmLevel::V8_7 => V8_7,
			ArmLevel::V9_0 => V9_0,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		ArmLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(ArmLevel::V8_0)
	}
}
