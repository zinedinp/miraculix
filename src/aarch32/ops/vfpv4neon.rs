//! Neon + VFPv4 FMA (`+FMA`, GCC `-mfpu=neon-vfpv4`). Token: [`Vfpv4Neon`].
//! Separate from [`super::neon::Neon`]; needs both [`Feature::Neon`] and
//! [`Feature::Vfpv4`]. Upstream: `core::arch::arm::vfmaq_f32`.

use super::super::{Feature, FeatureSet};

/// Proof that Neon + VFPv4 fused multiply-add is available. Zero-sized, `Copy`.
///
/// Obtain via [`Vfpv4Neon::detect`] or [`Vfpv4Neon::from_features`], then
/// call methods on the token.
#[derive(Debug, Clone, Copy)]
pub struct Vfpv4Neon(());

impl Vfpv4Neon {
	/// Probe once: `Some(token)` if Neon + VFPv4 FMA is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` unless both `Feature::Neon` and `Feature::Vfpv4` are present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Neon) && set.contains(Feature::Vfpv4)).then_some(Vfpv4Neon(()))
	}

	/// `VFMA.F32`: per-lane fused `b*c + a` (single rounding). `a` is the
	/// accumulator (matches `vfmaq_f32` / `simd_fma(b, c, a)`, not x86
	/// `fmadd(a,b,c)`). Uses `enable = "vfp4"` on arm (not `"v7"`), so this
	/// method is hand-written rather than [`super::macros::neon_ternop_x4`].
	#[inline]
	pub fn fma_f32x4(self, a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
		#[target_feature(enable = "neon")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "vfp4"))]
		unsafe fn imp(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
			let av = unsafe { core::arch::arm::vld1q_f32(a.as_ptr()) };
			let bv = unsafe { core::arch::arm::vld1q_f32(b.as_ptr()) };
			let cv = unsafe { core::arch::arm::vld1q_f32(c.as_ptr()) };
			let rv = core::arch::arm::vfmaq_f32(av, bv, cv);
			let mut out = [0f32; 4];
			unsafe { core::arch::arm::vst1q_f32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(a, b, c) }
	}
}

#[cfg(test)]
#[path = "../test/ops/vfpv4neon.rs"]
mod tests;
