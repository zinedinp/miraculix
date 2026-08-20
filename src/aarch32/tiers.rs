//! Coarse FPU/SIMD tiers over [`super::Feature`] (~ GCC `-mfpu=`). Only levels
//! with distinct cumulative HWCAP patterns (no vfpv3-d16 split). Approx.
//! [`FpuLevel::None`] = scalar int (no "no-FPU" bit).

use super::features::{Feature, FeatureSet};

/// Enum list for AArch32 FPU levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum FpuLevel {
	/// Scalar integer only.
	None = 0,
	/// `-mfpu=neon`: VFPv3 + 32 D-regs + NEON.
	Vfpv3Neon = 1,
	/// `-mfpu=neon-vfpv4`: +FMA.
	Vfpv4Neon = 2,
	/// ~ `crypto-neon-fp-armv8`: +AES/PMULL/SHA.
	Crypto = 3,
}

impl FpuLevel {
	pub const ALL: &'static [FpuLevel] =
		&[FpuLevel::None, FpuLevel::Vfpv3Neon, FpuLevel::Vfpv4Neon, FpuLevel::Crypto];

	/// Cumulative required features. None = empty.
	pub fn required_features(self) -> &'static [Feature] {
		const VFPV3_NEON: &[Feature] = &[Feature::Vfp, Feature::Vfpv3, Feature::Vfpd32, Feature::Neon];
		const VFPV4_NEON: &[Feature] =
			&[Feature::Vfp, Feature::Vfpv3, Feature::Vfpd32, Feature::Neon, Feature::Vfpv4];
		const CRYPTO: &[Feature] = &[
			Feature::Vfp, Feature::Vfpv3, Feature::Vfpd32, Feature::Neon, Feature::Vfpv4,
			Feature::Aes, Feature::Pmull, Feature::Sha1, Feature::Sha2,
		];

		match self {
			FpuLevel::None => &[],
			FpuLevel::Vfpv3Neon => VFPV3_NEON,
			FpuLevel::Vfpv4Neon => VFPV4_NEON,
			FpuLevel::Crypto => CRYPTO,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		FpuLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(FpuLevel::None)
	}
}
