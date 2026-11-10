//! ARMv8.3-A `FEAT_FCMA`: complex-number add and multiply-accumulate on
//! interleaved `[re, im, re, im, ...]` lanes. Token: [`Fcma`].
//!
//! `FCADD #rot` adds `b` rotated by 90° or 270° in the complex plane.
//! `FCMLA #rot` is **half** a complex multiply: `#0` adds `re(a)·b`, `#90`
//! adds `i·im(a)·b`, so `#0` followed by `#90` is the full `acc + a·b`
//! ([`Fcma::cmla_f32x4`] does exactly that). Each partial product is fused
//! into the accumulator with one rounding.
//!
//! **Why `asm!`:** the FCMA intrinsics are still unstable
//! (`stdarch_neon_fcma`); the `fcma` target feature is stable. No aarch32
//! counterpart (`aarch32::ops` ledger: "no stdarch on any arch").

use core::arch::asm;

use super::super::macros::arm_token;
use super::super::simd_arch::{
    float32x4_t, float64x2_t, vld1q_f32, vld1q_f64, vst1q_f32, vst1q_f64,
};

arm_token! {
    /// Proof that `FEAT_FCMA` (`FCADD`/`FCMLA`) is available.
    token = Fcma, requires = [Fcma],
}

/// `FCADD`: two operands, rotation baked into the template.
macro_rules! fcadd {
	($fixed_fn:ident, $intrinsic_fn:ident, $Elem:ty, $lanes:literal, $Vec:ty, $ld:path, $st:path, $asm:literal, $doc:literal) => {
		impl Fcma {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $lanes], b: [$Elem; $lanes]) -> [$Elem; $lanes] {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,fcma")]
		unsafe fn $intrinsic_fn(a: &[$Elem; $lanes], b: &[$Elem; $lanes]) -> [$Elem; $lanes] {
			unsafe {
				let mut va: $Vec = $ld(a.as_ptr());
				let vb: $Vec = $ld(b.as_ptr());
				asm!($asm, a = inout(vreg) va, b = in(vreg) vb, options(pure, nomem, nostack, preserves_flags));
				let mut out = [0 as $Elem; $lanes];
				$st(out.as_mut_ptr(), va);
				out
			}
		}
	};
}

/// `FCMLA`: accumulator plus two operands.
macro_rules! fcmla {
	($fixed_fn:ident, $intrinsic_fn:ident, $Elem:ty, $lanes:literal, $Vec:ty, $ld:path, $st:path, $asm:literal, $doc:literal) => {
		impl Fcma {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, acc: [$Elem; $lanes], a: [$Elem; $lanes], b: [$Elem; $lanes]) -> [$Elem; $lanes] {
				unsafe { $intrinsic_fn(&acc, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,fcma")]
		unsafe fn $intrinsic_fn(acc: &[$Elem; $lanes], a: &[$Elem; $lanes], b: &[$Elem; $lanes]) -> [$Elem; $lanes] {
			unsafe {
				let mut vacc: $Vec = $ld(acc.as_ptr());
				let va: $Vec = $ld(a.as_ptr());
				let vb: $Vec = $ld(b.as_ptr());
				asm!(
					$asm,
					acc = inout(vreg) vacc,
					a = in(vreg) va,
					b = in(vreg) vb,
					options(pure, nomem, nostack, preserves_flags),
				);
				let mut out = [0 as $Elem; $lanes];
				$st(out.as_mut_ptr(), vacc);
				out
			}
		}
	};
}

fcadd!(
    cadd_rot90_f32x4,
    fcadd90_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcadd {a:v}.4s, {a:v}.4s, {b:v}.4s, #90",
    "Complex add with `b` rotated by 90°: `re = a.re - b.im`, `im = a.im + b.re` per pair (`fcadd v.4s, #90`)."
);
fcadd!(
    cadd_rot270_f32x4,
    fcadd270_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcadd {a:v}.4s, {a:v}.4s, {b:v}.4s, #270",
    "Complex add with `b` rotated by 270°: `re = a.re + b.im`, `im = a.im - b.re` per pair (`fcadd v.4s, #270`)."
);
fcadd!(
    cadd_rot90_f64x2,
    fcadd90_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "fcadd {a:v}.2d, {a:v}.2d, {b:v}.2d, #90",
    "Complex add with `b` rotated by 90° on one `f64` pair (`fcadd v.2d, #90`)."
);
fcadd!(
    cadd_rot270_f64x2,
    fcadd270_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "fcadd {a:v}.2d, {a:v}.2d, {b:v}.2d, #270",
    "Complex add with `b` rotated by 270° on one `f64` pair (`fcadd v.2d, #270`)."
);

fcmla!(
    cmla_rot0_f32x4,
    fcmla0_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcmla {acc:v}.4s, {a:v}.4s, {b:v}.4s, #0",
    "Half complex multiply-accumulate, `#0`: `re += a.re·b.re`, `im += a.re·b.im`, each fused (`fcmla v.4s, #0`)."
);
fcmla!(
    cmla_rot90_f32x4,
    fcmla90_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcmla {acc:v}.4s, {a:v}.4s, {b:v}.4s, #90",
    "Half complex multiply-accumulate, `#90`: `re -= a.im·b.im`, `im += a.im·b.re` (`fcmla v.4s, #90`)."
);
fcmla!(
    cmla_rot180_f32x4,
    fcmla180_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcmla {acc:v}.4s, {a:v}.4s, {b:v}.4s, #180",
    "Half complex multiply-accumulate, `#180`: `re -= a.re·b.re`, `im -= a.re·b.im` (`fcmla v.4s, #180`)."
);
fcmla!(
    cmla_rot270_f32x4,
    fcmla270_f32q_fixed,
    f32,
    4,
    float32x4_t,
    vld1q_f32,
    vst1q_f32,
    "fcmla {acc:v}.4s, {a:v}.4s, {b:v}.4s, #270",
    "Half complex multiply-accumulate, `#270`: `re += a.im·b.im`, `im -= a.im·b.re` (`fcmla v.4s, #270`)."
);
fcmla!(
    cmla_rot0_f64x2,
    fcmla0_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "fcmla {acc:v}.2d, {a:v}.2d, {b:v}.2d, #0",
    "Half complex multiply-accumulate, `#0`, on one `f64` pair (`fcmla v.2d, #0`)."
);
fcmla!(
    cmla_rot90_f64x2,
    fcmla90_f64q_fixed,
    f64,
    2,
    float64x2_t,
    vld1q_f64,
    vst1q_f64,
    "fcmla {acc:v}.2d, {a:v}.2d, {b:v}.2d, #90",
    "Half complex multiply-accumulate, `#90`, on one `f64` pair (`fcmla v.2d, #90`)."
);

impl Fcma {
    /// Full complex multiply-accumulate `acc + a·b` on two `[re, im]` pairs:
    /// `FCMLA #0` then `FCMLA #90`. Each of the four partial products is fused
    /// into the running sum, so the result can differ in the last bit from a
    /// textbook `(ac - bd, ad + bc)` evaluated with separate roundings.
    #[inline]
    pub fn cmla_f32x4(self, acc: [f32; 4], a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        self.cmla_rot90_f32x4(self.cmla_rot0_f32x4(acc, a, b), a, b)
    }

    /// Full complex multiply-accumulate `acc + a·b` on one `f64` pair; see
    /// [`Fcma::cmla_f32x4`].
    #[inline]
    pub fn cmla_f64x2(self, acc: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        self.cmla_rot90_f64x2(self.cmla_rot0_f64x2(acc, a, b), a, b)
    }
}

#[cfg(test)]
#[path = "../../test/ops/neon/fcma.rs"]
mod tests;
