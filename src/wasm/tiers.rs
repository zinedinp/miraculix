//! Coarse tiers over [`super::Feature`]. No hardware floor here (unlike NEON
//! on AArch64 or gc on RISC-V): plain `wasm32-unknown-unknown` has neither
//! bit by default, so Scalar is a real, reachable state, not just a sentinel.

use super::features::{Feature, FeatureSet};

/// Enum list for WASM SIMD tiers. Scalar is the real floor; no empty-but-unreachable state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum WasmLevel {
	/// No SIMD proposal compiled in; plain scalar wasm ops only.
	Scalar = 0,
	/// + `simd128`.
	Simd128 = 1,
	/// + `relaxed-simd` (implies simd128).
	RelaxedSimd = 2,
}

impl WasmLevel {
	pub const ALL: &'static [WasmLevel] =
		&[WasmLevel::Scalar, WasmLevel::Simd128, WasmLevel::RelaxedSimd];

	/// Cumulative required features. Scalar = empty.
	pub fn required_features(self) -> &'static [Feature] {
		const SIMD128: &[Feature] = &[Feature::Simd128];
		const RELAXED_SIMD: &[Feature] = &[Feature::Simd128, Feature::RelaxedSimd];

		match self {
			WasmLevel::Scalar => &[],
			WasmLevel::Simd128 => SIMD128,
			WasmLevel::RelaxedSimd => RELAXED_SIMD,
		}
	}

	/// Highest level fully covered by `set`.
	pub fn detect(set: FeatureSet) -> Self {
		WasmLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(WasmLevel::Scalar)
	}
}
