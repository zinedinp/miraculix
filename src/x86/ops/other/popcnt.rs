//! POPCNT: scalar GPR `popcnt` (no packed form). Stable `core::arch`.
//! Token: [`Popcnt::detect`]. Used by `auto` under AVX512VPOPCNTDQ and AVX512BITALG.
//! Only 32/64-bit GPR forms are wrapped.

use core::arch::x86_64::{_popcnt32, _popcnt64};

use super::super::super::{Feature, FeatureSet};

/// Proof token: POPCNT available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Popcnt(());

impl Popcnt {
	/// `None` if the CPU (or the compile-time target) lacks POPCNT.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Popcnt).then_some(Popcnt(()))
	}

	/// Number of set bits in `x` (`popcnt`, 32-bit GPR form).
	#[inline]
	pub fn popcnt_u32(self, x: u32) -> u32 {
		unsafe { popcnt32(x) }
	}

	/// Number of set bits in `x` (`popcnt`, 64-bit GPR form).
	#[inline]
	pub fn popcnt_u64(self, x: u64) -> u64 {
		unsafe { popcnt64(x) }
	}

	/// `out[i] = a[i].count_ones()`. No hardware SIMD popcount exists on
	/// x86 at any width (only the scalar GPR form); plain per-element loop.
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn popcnt_u32_slice(self, a: &[u32], out: &mut [u32]) {
		assert_eq!(out.len(), a.len());
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = self.popcnt_u32(x);
		}
	}

	/// `out[i] = a[i].count_ones()` (as `u64`). Same no-SIMD-form limitation
	/// as [`Popcnt::popcnt_u32_slice`].
	///
	/// # Panics
	/// `out.len() != a.len()`.
	pub fn popcnt_u64_slice(self, a: &[u64], out: &mut [u64]) {
		assert_eq!(out.len(), a.len());
		for (&x, o) in a.iter().zip(out.iter_mut()) {
			*o = self.popcnt_u64(x);
		}
	}
}

/// # Safety
/// Caller proved POPCNT via [`Popcnt`].
#[inline]
#[target_feature(enable = "popcnt")]
unsafe fn popcnt32(x: u32) -> u32 {
	_popcnt32(x as i32) as u32
}

/// # Safety
/// Caller proved POPCNT via [`Popcnt`].
#[inline]
#[target_feature(enable = "popcnt")]
unsafe fn popcnt64(x: u64) -> u64 {
	_popcnt64(x as i64) as u64
}

#[cfg(test)]
#[path = "../../test/ops/other/popcnt.rs"]
mod tests;
