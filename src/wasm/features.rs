//! WASM SIMD: fixed at compile time. Host validates opcodes at instantiate;
//! `detect()` restates `cfg(target_feature)`, not a runtime hardware read.

/// Enum list for WASM SIMD proposals. `relaxed-simd` implies `simd128` (LLVM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
	/// Fixed-width 128-bit SIMD (`v128`, `i32x4_add`, ...). Stable since 1.54.
	Simd128,
	/// Relaxed-SIMD: implementation-defined-precision ops (fma, swizzle, dot).
	/// Stable since 1.82.
	RelaxedSimd,
}

impl Feature {
	/// Index = bit in [`FeatureSet`].
	pub const ALL: &'static [Feature] = &[Feature::Simd128, Feature::RelaxedSimd];

	/// Bit index in [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Compile-time only: mirrors this binary's `cfg(target_feature)`, not the
	/// host engine.
	pub fn detect() -> Self {
		#[allow(unused_mut, reason = "mutated only when simd128/relaxed-simd cfg is active")]
		let mut set = 0u128;
		#[cfg(target_feature = "simd128")]
		{
			set |= 1 << Feature::Simd128.bit();
		}
		#[cfg(target_feature = "relaxed-simd")]
		{
			set |= 1 << Feature::RelaxedSimd.bit();
		}
		Self(set)
	}

	pub fn contains(self, feature: Feature) -> bool {
		self.0 & (1 << feature.bit()) != 0
	}

	pub fn contains_all(self, required: &[Feature]) -> bool {
		required.iter().all(|&f| self.contains(f))
	}
}
