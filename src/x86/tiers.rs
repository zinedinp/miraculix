//! Coarse x86-64 tiers ([`GenericLevel`] V1..V4) and [`Avx10`] helpers.
//!
//! End users usually call [`crate::x86::detect_level`] rather than
//! [`GenericLevel::detect`] directly.

use super::features::{Feature, FeatureSet};

/// Coarse x86-64 capability tier (psABI generic levels, SIMD/bitmanip subset).
///
/// Ordered: `V1 < V2 < V3 < V4`.
/// Obtained via [`crate::x86::detect_level`].
///
/// | Level | Rough meaning |
/// |---|---|
/// | [`V1`](GenericLevel::V1) | Baseline x86-64 |
/// | [`V2`](GenericLevel::V2) | SSE3..SSE4.2 + POPCNT |
/// | [`V3`](GenericLevel::V3) | + AVX / AVX2 / FMA / F16C / BMI... |
/// | [`V4`](GenericLevel::V4) | + AVX-512 F/BW/CD/DQ/VL |
///
/// For individual flags (e.g. AVX-512F without full V4), use
/// [`crate::x86::detect_features`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum GenericLevel {
	/// Baseline x86-64.
	V1 = 1,
	/// `-march=x86-64-v2`: SSE3/SSSE3/SSE4.1/SSE4.2/POPCNT.
	V2 = 2,
	/// `-march=x86-64-v3`: +AVX/AVX2/BMI1/BMI2/F16C/FMA/LZCNT/MOVBE/XSAVE.
	V3 = 3,
	/// `-march=x86-64-v4`: +AVX-512F/BW/CD/DQ/VL.
	V4 = 4,
}

impl GenericLevel {
	/// All levels low to high.
	pub const ALL: &'static [GenericLevel] =
		&[GenericLevel::V1, GenericLevel::V2, GenericLevel::V3, GenericLevel::V4];

	/// Features this level requires (higher levels include lower ones).
	pub fn required_features(self) -> &'static [Feature] {
		const V2: &[Feature] =
			&[Feature::Popcnt, Feature::Sse3, Feature::Ssse3, Feature::Sse41, Feature::Sse42];
		const V3: &[Feature] = &[
			Feature::Popcnt, Feature::Sse3, Feature::Ssse3, Feature::Sse41, Feature::Sse42,
			Feature::Avx, Feature::Avx2, Feature::Bmi1, Feature::Bmi2, Feature::F16c,
			Feature::Fma, Feature::Lzcnt, Feature::Movbe, Feature::Xsave,
		];
		const V4: &[Feature] = &[
			Feature::Popcnt, Feature::Sse3, Feature::Ssse3, Feature::Sse41, Feature::Sse42,
			Feature::Avx, Feature::Avx2, Feature::Bmi1, Feature::Bmi2, Feature::F16c,
			Feature::Fma, Feature::Lzcnt, Feature::Movbe, Feature::Xsave,
			Feature::Avx512f, Feature::Avx512bw, Feature::Avx512cd, Feature::Avx512dq,
			Feature::Avx512vl,
		];

		match self {
			GenericLevel::V1 => &[],
			GenericLevel::V2 => V2,
			GenericLevel::V3 => V3,
			GenericLevel::V4 => V4,
		}
	}

	/// Highest level whose full feature list is covered by `set`.
	///
	/// Prefer [`crate::x86::detect_level`] for the process-cached answer.
	pub fn detect(set: FeatureSet) -> Self {
		GenericLevel::ALL
			.iter()
			.rev()
			.copied()
			.find(|&level| set.contains_all(level.required_features()))
			.unwrap_or(GenericLevel::V1)
	}
}

/// Helpers for the AVX10 feature bit (CPUID leaf `0x24`).
///
/// Presence is reliable via [`Feature::Avx10`]. Version is best-effort for
/// diagnostics only, not for correctness gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Avx10;

impl Avx10 {
	/// `true` if [`Feature::Avx10`] is set.
	pub fn is_present(set: FeatureSet) -> bool {
		set.contains(Feature::Avx10)
	}

	/// Best-effort AVX10 version from leaf `0x24` EBX[7:0], or `None` if absent.
	///
	/// Diagnostics only; do not gate correctness on this value.
	pub fn version(set: FeatureSet) -> Option<u8> {
		if !Self::is_present(set) {
			return None;
		}
		#[cfg(target_arch = "x86_64")]
		use core::arch::x86_64::__cpuid_count;
		#[cfg(target_arch = "x86")]
		use core::arch::x86::__cpuid_count;

		let leaf = __cpuid_count(0x24, 0);
		Some((leaf.ebx & 0xff) as u8)
	}
}
