//! ARMv8.6-A Int8 Matrix Multiply: mixed-sign `USDOT` plus 8-bit 2x2 block
//! matmul-accumulate (`SMMLA`/`UMMLA`/`USMMLA`). Token: [`I8mm`]. Detect:
//! [`Feature::I8mm`]. `MMLA`: `a`/`b` are two packed 8-byte rows each; the 4
//! accumulator lanes are a row-major 2x2 of 8-element dots (not a Dotprod
//! shape despite the shared `(acc,a,b)->acc` call form).

use super::super::{Feature, FeatureSet};
use super::macros::neon_dot_acc_x4;

/// Proof that the Int8 Matrix Multiply extension is available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct I8mm(());

impl I8mm {
	/// Probe once: `Some(token)` if Int8 Matrix Multiply is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` unless both `Feature::Neon` and `Feature::I8mm` are present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Neon) && set.contains(Feature::I8mm)).then_some(I8mm(()))
	}

	neon_dot_acc_x4!(
		/// `VUSDOT.S8`: mixed-sign 8-bit dot (`a` unsigned, `b` signed) into
		/// `i32` lanes. Gated by `Feature::I8mm`, not `Feature::Dotprod`
		/// (stdarch attribute; see [`super::dotprod`]).
		dot_us32,
		i32,
		u8,
		i8,
		vld1q_s32,
		vst1q_s32,
		vld1q_u8,
		vld1q_s8,
		"i8mm",
		vusdotq_s32
	);
	neon_dot_acc_x4!(
		/// `VSMMLA.S8`: signed 8-bit 2x2 block matmul-accumulate (module doc).
		mmla_s32,
		i32,
		i8,
		i8,
		vld1q_s32,
		vst1q_s32,
		vld1q_s8,
		vld1q_s8,
		"i8mm",
		vmmlaq_s32
	);
	neon_dot_acc_x4!(
		/// `VUMMLA.U8`: unsigned 8-bit 2x2 block matmul-accumulate; same lane
		/// layout as [`I8mm::mmla_s32`].
		mmla_u32,
		u32,
		u8,
		u8,
		vld1q_u32,
		vst1q_u32,
		vld1q_u8,
		vld1q_u8,
		"i8mm",
		vmmlaq_u32
	);
	neon_dot_acc_x4!(
		/// `VUSMMLA.S8`: mixed-sign (`a` unsigned, `b` signed) 2x2 block
		/// matmul-accumulate; same lane layout as [`I8mm::mmla_s32`].
		mmla_us32,
		i32,
		u8,
		i8,
		vld1q_s32,
		vst1q_s32,
		vld1q_u8,
		vld1q_s8,
		"i8mm",
		vusmmlaq_s32
	);
}

#[cfg(test)]
#[path = "../test/ops/i8mm.rs"]
mod tests;
