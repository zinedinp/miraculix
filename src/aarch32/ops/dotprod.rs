//! ARMv8.2-A Dot Product: 16-lane i8/u8 sum-of-4-products into 4-lane
//! i32/u32. Token: [`Dotprod`]. Upstream: `vdotq_s32`/`vdotq_u32`. Detect:
//! [`Feature::Dotprod`] (`ASIMDDP`). Lane `i`: `acc[i] + sum(a[4*i+k]*b[4*i+k]
//! for k in 0..4)`. `USDOT` lives in [`super::i8mm`] (needs `I8mm`, not
//! Dotprod). No plain vector `SUDOT` in stdarch (lane forms only).

use super::super::{Feature, FeatureSet};
use super::macros::neon_dot_acc_x4;

/// Proof that the Dot Product extension is available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Dotprod(());

impl Dotprod {
	/// Probe once: `Some(token)` if Dot Product is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` unless both `Feature::Neon` and `Feature::Dotprod` are present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Neon) && set.contains(Feature::Dotprod)).then_some(Dotprod(()))
	}

	neon_dot_acc_x4!(
		/// `VSDOT.S8`: signed 8-bit dot product, accumulated into `i32` lanes.
		dot_s32,
		i32,
		i8,
		i8,
		vld1q_s32,
		vst1q_s32,
		vld1q_s8,
		vld1q_s8,
		"dotprod",
		vdotq_s32
	);
	neon_dot_acc_x4!(
		/// `VUDOT.U8`: unsigned 8-bit dot product, accumulated into `u32` lanes.
		dot_u32,
		u32,
		u8,
		u8,
		vld1q_u32,
		vst1q_u32,
		vld1q_u8,
		vld1q_u8,
		"dotprod",
		vdotq_u32
	);
}

#[cfg(test)]
#[path = "../test/ops/dotprod.rs"]
mod tests;
