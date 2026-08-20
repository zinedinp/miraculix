//! ARMv7-A Neon vector ops (SSE/AVX analog). Token: [`Neon`]. Upstream
//! `int32x4_t`/`float32x4_t`/`uint32x4_t` under `stdarch_arm_neon_intrinsics`
//! (unstable on arm; stable on aarch64). On arm every intrinsic needs both
//! `"neon"` and `"v7"` on the impl fn ([`neon_binop_x4`](super::macros)).
//! Detect: [`Feature::Neon`].

use super::super::{Feature, FeatureSet};
use super::macros::{neon_binop_x4, neon_cmp_x4, neon_ternop_x4, neon_unop_x4};

/// Proof that Neon is available. Zero-sized, `Copy`.
///
/// Obtain via [`Neon::detect`] or [`Neon::from_features`], then call
/// methods on the token.
#[derive(Debug, Clone, Copy)]
pub struct Neon(());

impl Neon {
	/// Probe once: `Some(token)` if Neon is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` if `Feature::Neon` is missing.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Neon).then_some(Neon(()))
	}

	neon_binop_x4!(
		/// `VADD.S32`: per-lane `i32` addition.
		add_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vaddq_s32
	);
	neon_binop_x4!(
		/// `VSUB.S32`: per-lane `i32` subtraction.
		sub_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vsubq_s32
	);
	neon_binop_x4!(
		/// `VMUL.S32`: per-lane `i32` multiplication (low 32 bits of the product).
		mul_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vmulq_s32
	);

	neon_binop_x4!(
		/// `VADD.F32`: per-lane `f32` addition.
		add_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vaddq_f32
	);
	neon_binop_x4!(
		/// `VSUB.F32`: per-lane `f32` subtraction.
		sub_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vsubq_f32
	);
	neon_binop_x4!(
		/// `VMUL.F32`: per-lane `f32` multiplication.
		mul_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vmulq_f32
	);

	neon_binop_x4!(
		/// `VAND`: per-lane `u32` bitwise AND.
		and_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		vandq_u32
	);
	neon_binop_x4!(
		/// `VORR`: per-lane `u32` bitwise OR.
		or_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		vorrq_u32
	);
	neon_binop_x4!(
		/// `VEOR`: per-lane `u32` bitwise XOR.
		xor_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		veorq_u32
	);
	neon_binop_x4!(
		/// `VBIC`: per-lane `u32` bitwise AND-NOT, `a & !b` (native `vbicq_u32`
		/// operand order; x86's `andnot` convention is the mirror image, `!a &
		/// b` - callers wanting that swap arguments at the call site).
		andnot_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		vbicq_u32
	);

	neon_cmp_x4!(
		/// `VCEQ.S32`: per-lane `i32` equality, `[u32; 4]` lane mask
		/// (all-1s or 0, not `bool`).
		cmpeq_i32x4,
		i32,
		vld1q_s32,
		vceqq_s32
	);
	neon_cmp_x4!(
		/// `VCGT.S32`: per-lane `i32` greater-than, `[u32; 4]` lane mask.
		cmpgt_i32x4,
		i32,
		vld1q_s32,
		vcgtq_s32
	);
	neon_cmp_x4!(
		/// `VCGE.S32`: per-lane `i32` greater-or-equal, `[u32; 4]` lane mask.
		cmpge_i32x4,
		i32,
		vld1q_s32,
		vcgeq_s32
	);
	neon_cmp_x4!(
		/// `VCLT.S32`: per-lane `i32` less-than, `[u32; 4]` lane mask.
		cmplt_i32x4,
		i32,
		vld1q_s32,
		vcltq_s32
	);
	neon_cmp_x4!(
		/// `VCLE.S32`: per-lane `i32` less-or-equal, `[u32; 4]` lane mask.
		cmple_i32x4,
		i32,
		vld1q_s32,
		vcleq_s32
	);

	neon_cmp_x4!(
		/// `VCEQ.F32`: per-lane `f32` equality, `[u32; 4]` lane mask. NaN
		/// never equals (mask 0), matching this crate's x86 auto convention.
		cmpeq_f32x4,
		f32,
		vld1q_f32,
		vceqq_f32
	);
	neon_cmp_x4!(
		/// `VCGT.F32`: per-lane `f32` greater-than (ordered; false if either
		/// lane is NaN), `[u32; 4]` lane mask.
		cmpgt_f32x4,
		f32,
		vld1q_f32,
		vcgtq_f32
	);
	neon_cmp_x4!(
		/// `VCGE.F32`: per-lane `f32` greater-or-equal (ordered), `[u32; 4]`
		/// lane mask.
		cmpge_f32x4,
		f32,
		vld1q_f32,
		vcgeq_f32
	);
	neon_cmp_x4!(
		/// `VCLT.F32`: per-lane `f32` less-than (ordered), `[u32; 4]` lane mask.
		cmplt_f32x4,
		f32,
		vld1q_f32,
		vcltq_f32
	);
	neon_cmp_x4!(
		/// `VCLE.F32`: per-lane `f32` less-or-equal (ordered), `[u32; 4]`
		/// lane mask.
		cmple_f32x4,
		f32,
		vld1q_f32,
		vcleq_f32
	);

	neon_binop_x4!(
		/// `VMAX.S32`: per-lane `i32` maximum.
		max_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vmaxq_s32
	);
	neon_binop_x4!(
		/// `VMIN.S32`: per-lane `i32` minimum.
		min_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vminq_s32
	);
	neon_binop_x4!(
		/// `VMAX.F32`: per-lane `f32` maximum. NaN follows the `VMAX`
		/// instruction, not Rust `f32::max` (same SIMD-vs-scalar caveat as
		/// x86 `auto_up`).
		max_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vmaxq_f32
	);
	neon_binop_x4!(
		/// `VMIN.F32`: per-lane `f32` minimum. Same NaN caveat as
		/// [`Neon::max_f32x4`].
		min_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vminq_f32
	);

	neon_unop_x4!(
		/// `VABS.S32`: per-lane `i32` absolute value.
		abs_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vabsq_s32
	);
	neon_unop_x4!(
		/// `VNEG.S32`: per-lane `i32` negation.
		neg_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vnegq_s32
	);
	neon_unop_x4!(
		/// `VABS.F32`: per-lane `f32` absolute value.
		abs_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vabsq_f32
	);
	neon_unop_x4!(
		/// `VNEG.F32`: per-lane `f32` negation.
		neg_f32x4,
		f32,
		vld1q_f32,
		vst1q_f32,
		vnegq_f32
	);
	neon_unop_x4!(
		/// `VMVN`: per-lane `u32` bitwise NOT.
		not_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		vmvnq_u32
	);

	neon_binop_x4!(
		/// `VSHL.S32`: per-lane variable `i32` shift. Positive `b[i]` shifts
		/// left, negative shifts right (arithmetic); magnitudes `>= 32`
		/// saturate per `VSHL`, not Rust shift-amount panics.
		shl_i32x4,
		i32,
		vld1q_s32,
		vst1q_s32,
		vshlq_s32
	);

	neon_ternop_x4!(
		/// `VBSL`: per-lane bit-select `(a & b) | (!a & c)`; `a` picks bits
		/// of `b` where set, of `c` where clear.
		select_u32x4,
		u32,
		vld1q_u32,
		vst1q_u32,
		vbslq_u32
	);
}

#[cfg(test)]
#[path = "../test/ops/neon.rs"]
mod tests;
