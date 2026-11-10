//! ARMv8.5-A `FEAT_FRINTTS`: round to an integral value that is guaranteed to
//! fit a 32- or 64-bit integer. Token: [`Frintts`].
//!
//! `FRINT32Z`/`FRINT64Z` round toward zero, `FRINT32X`/`FRINT64X` use the
//! current rounding mode (round-to-nearest-even unless code changed FPCR).
//! A result the target integer range cannot hold - including NaN and ±∞ -
//! becomes the **most negative** value of that range (`-2³¹` / `-2⁶³`) as a
//! float. The result stays floating point; this is the "round, then the
//! later `fcvtzs` cannot trap or saturate" building block.
//!
//! **Why `asm!`:** the FRINTTS intrinsics are still unstable
//! (`stdarch_neon_ftts`); the `frintts` target feature is stable.

use core::arch::asm;

use super::super::macros::arm_token;
use super::super::simd_arch::{
    float32x4_t, float64x2_t, vld1q_f32, vld1q_f64, vst1q_f32, vst1q_f64,
};

arm_token! {
    /// Proof that `FEAT_FRINTTS` (`FRINT32Z`/`FRINT32X`/`FRINT64Z`/`FRINT64X`)
    /// is available.
    token = Frintts, requires = [Frint],
}

macro_rules! frint {
	($fixed_fn:ident, $intrinsic_fn:ident, $Elem:ty, $lanes:literal, $Vec:ty, $ld:path, $st:path, $asm:literal, $doc:literal) => {
		impl Frintts {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $lanes]) -> [$Elem; $lanes] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,frintts")]
		unsafe fn $intrinsic_fn(a: &[$Elem; $lanes]) -> [$Elem; $lanes] {
			unsafe {
				let mut va: $Vec = $ld(a.as_ptr());
				asm!($asm, a = inout(vreg) va, options(pure, nomem, nostack, preserves_flags));
				let mut out = [0 as $Elem; $lanes];
				$st(out.as_mut_ptr(), va);
				out
			}
		}
	};
}

frint!(
    rnd32z_f32x4,
    frint32z_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "frint32z {a:v}.4s, {a:v}.4s",
    "Round toward zero into `i32` range; out of range, NaN or ±∞ → `-2³¹` (`frint32z v.4s`)."
);
frint!(
    rnd32x_f32x4,
    frint32x_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "frint32x {a:v}.4s, {a:v}.4s",
    "Round with the current mode (nearest-even by default) into `i32` range; out of range → `-2³¹` (`frint32x v.4s`)."
);
frint!(
    rnd64z_f32x4,
    frint64z_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "frint64z {a:v}.4s, {a:v}.4s",
    "Round toward zero into `i64` range; out of range, NaN or ±∞ → `-2⁶³` (`frint64z v.4s`)."
);
frint!(
    rnd64x_f32x4,
    frint64x_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "frint64x {a:v}.4s, {a:v}.4s",
    "Round with the current mode into `i64` range; out of range → `-2⁶³` (`frint64x v.4s`)."
);
frint!(
    rnd32z_f64x2,
    frint32z_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "frint32z {a:v}.2d, {a:v}.2d",
    "Round toward zero into `i32` range; out of range → `-2³¹` (`frint32z v.2d`)."
);
frint!(
    rnd32x_f64x2,
    frint32x_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "frint32x {a:v}.2d, {a:v}.2d",
    "Round with the current mode into `i32` range; out of range → `-2³¹` (`frint32x v.2d`)."
);
frint!(
    rnd64z_f64x2,
    frint64z_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "frint64z {a:v}.2d, {a:v}.2d",
    "Round toward zero into `i64` range; out of range → `-2⁶³` (`frint64z v.2d`)."
);
frint!(
    rnd64x_f64x2,
    frint64x_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "frint64x {a:v}.2d, {a:v}.2d",
    "Round with the current mode into `i64` range; out of range → `-2⁶³` (`frint64x v.2d`)."
);

#[cfg(test)]
#[path = "../../test/ops/neon/frintts.rs"]
mod tests;
