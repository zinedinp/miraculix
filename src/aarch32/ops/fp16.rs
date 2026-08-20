//! ARMv8.2-A FullFP16 Neon (vector half-precision). Tokens: [`Fp16Neon`],
//! [`Fp16Fma`]. Detect on [`Feature::AsimdHp`] (`ASIMDHP`), not
//! [`Feature::Fp16`] (`FPHP` = scalar VFP half). Public surface is `[u16; 8]`
//! bit patterns (same as x86 BF16 here). `Fp16Fma` also requires
//! [`Feature::Vfpv4`] for `vfmaq_f16` (extra bit combo = own token).

use super::super::{Feature, FeatureSet};
use super::macros::{neon_binop_f16x8, neon_cmp_f16x8, neon_unop_f16x8};

/// Proof that the FullFP16 Neon extension is available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Fp16Neon(());

/// Proof that FullFP16 Neon and VFPv4 (`vfmaq_f16`) are both available.
/// Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Fp16Fma(());

impl Fp16Neon {
	/// Probe once: `Some(token)` if FullFP16 Neon is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` unless both `Feature::Neon` and `Feature::AsimdHp` are present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Neon) && set.contains(Feature::AsimdHp)).then_some(Fp16Neon(()))
	}

	neon_binop_f16x8!(
		/// `VADD.F16`: per-lane `f16` addition.
		add_f16x8,
		vaddq_f16
	);
	neon_binop_f16x8!(
		/// `VSUB.F16`: per-lane `f16` subtraction.
		sub_f16x8,
		vsubq_f16
	);
	neon_binop_f16x8!(
		/// `VMUL.F16`: per-lane `f16` multiplication.
		mul_f16x8,
		vmulq_f16
	);
	neon_binop_f16x8!(
		/// `VMAX.F16`: per-lane `f16` maximum. NaN follows the `VMAX`
		/// instruction, not a scalar `max` (same caveat as
		/// [`super::neon::Neon::max_f32x4`]).
		max_f16x8,
		vmaxq_f16
	);
	neon_binop_f16x8!(
		/// `VMIN.F16`: per-lane `f16` minimum. Same NaN caveat as [`Fp16Neon::max_f16x8`].
		min_f16x8,
		vminq_f16
	);
	neon_unop_f16x8!(
		/// `VABS.F16`: per-lane `f16` absolute value.
		abs_f16x8,
		vabsq_f16
	);
	neon_unop_f16x8!(
		/// `VNEG.F16`: per-lane `f16` negation.
		neg_f16x8,
		vnegq_f16
	);

	neon_cmp_f16x8!(
		/// `VCEQ.F16`: per-lane `f16` equality, `[u16; 8]` lane mask (all-1s
		/// or 0, not `bool`). NaN never equals (mask 0).
		cmpeq_f16x8,
		vceqq_f16
	);
	neon_cmp_f16x8!(
		/// `VCGT.F16`: per-lane `f16` greater-than (ordered; false if either
		/// lane is NaN), `[u16; 8]` lane mask.
		cmpgt_f16x8,
		vcgtq_f16
	);
	neon_cmp_f16x8!(
		/// `VCGE.F16`: per-lane `f16` greater-or-equal (ordered), `[u16; 8]` lane mask.
		cmpge_f16x8,
		vcgeq_f16
	);
	neon_cmp_f16x8!(
		/// `VCLT.F16`: per-lane `f16` less-than (ordered), `[u16; 8]` lane mask.
		cmplt_f16x8,
		vcltq_f16
	);
	neon_cmp_f16x8!(
		/// `VCLE.F16`: per-lane `f16` less-or-equal (ordered), `[u16; 8]` lane mask.
		cmple_f16x8,
		vcleq_f16
	);
}

impl Fp16Fma {
	/// Probe once: `Some(token)` if FullFP16 Neon + VFPv4 are available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` unless `Feature::Neon`, `Feature::AsimdHp` and
	/// `Feature::Vfpv4` are all present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Neon) && set.contains(Feature::AsimdHp) && set.contains(Feature::Vfpv4))
			.then_some(Fp16Fma(()))
	}

	/// `VFMA.F16`: per-lane `f16` fused multiply-add, `b * c + a` (`a` is the
	/// accumulator, matching `vfmaq_f16` and
	/// [`super::vfpv4neon::Vfpv4Neon::fma_f32x4`], not x86 `fmadd`). No
	/// `fms_f16x8` (same asymmetry as `fma_f32x4`).
	#[inline]
	pub fn fma_f16x8(self, a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
		#[target_feature(enable = "neon,fp16")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "vfp4"))]
		unsafe fn imp(a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
			let av = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(a.as_ptr()) });
			let bv = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(b.as_ptr()) });
			let cv = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(c.as_ptr()) });
			let rv = core::arch::arm::vreinterpretq_u16_f16(core::arch::arm::vfmaq_f16(av, bv, cv));
			let mut out = [0u16; 8];
			unsafe { core::arch::arm::vst1q_u16(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(a, b, c) }
	}
}

#[cfg(test)]
#[path = "../test/ops/fp16.rs"]
mod tests;
