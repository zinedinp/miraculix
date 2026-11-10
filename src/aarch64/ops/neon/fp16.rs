//! ARMv8.2-A `FEAT_FP16`: half-precision Neon arithmetic. Token:
//! [`Fp16Neon`].
//!
//! Values cross the API as **IEEE binary16 bit patterns** (`[u16; 8]`), the
//! same convention as `aarch32::ops::fp16`: the `f16` primitive is still
//! unstable, and `vld1q_f16` with it, so every load is `vld1q_u16` plus a
//! `vreinterpretq_f16_u16` (both stable since Rust 1.94).
//!
//! Slice tails need no scalar `f16` either: the last partial chunk is copied
//! into a zero-padded 8-lane buffer, run through the **same** vector
//! instruction, and only the live lanes are copied back. So every lane of a
//! slice result - body and tail - comes from the hardware op, with its exact
//! rounding and NaN behaviour.
//!
//! Unlike aarch32 there is no separate `Fp16Fma` token: `FMLA` on half
//! precision is part of `FEAT_FP16` on AArch64.

use super::super::macros::arm_token;
use super::super::simd_arch::{
    float16x8_t, uint16x8_t, vabsq_f16, vaddq_f16, vceqq_f16, vcgeq_f16, vcgtq_f16, vcleq_f16,
    vcltq_f16, vdivq_f16, vfmaq_f16, vfmsq_f16, vld1q_u16, vmaxq_f16, vminq_f16, vmulq_f16,
    vnegq_f16, vreinterpretq_f16_u16, vreinterpretq_u16_f16, vsqrtq_f16, vst1q_u16, vsubq_f16,
};

arm_token! {
    /// Proof that `FEAT_FP16` half-precision Neon arithmetic is available
    /// (both the scalar `FPHP` and the vector `ASIMDHP` bit).
    token = Fp16Neon, requires = [Fp16, AsimdHp],
}

/// Run `f` over whole 8-lane chunks, then once more over a zero-padded copy of
/// the tail. `N` is the number of input slices.
#[inline(always)]
fn padded_chunks<const N: usize>(
    ins: [&[u16]; N],
    out: &mut [u16],
    mut f: impl FnMut([&[u16; 8]; N]) -> [u16; 8],
) {
    let len = out.len();
    let whole = len - len % 8;
    for base in (0..whole).step_by(8) {
        let args: [&[u16; 8]; N] =
            core::array::from_fn(|k| ins[k][base..base + 8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&f(args));
    }
    if whole < len {
        let rest = len - whole;
        let pads: [[u16; 8]; N] = core::array::from_fn(|k| {
            let mut p = [0u16; 8];
            p[..rest].copy_from_slice(&ins[k][whole..]);
            p
        });
        let args: [&[u16; 8]; N] = core::array::from_fn(|k| &pads[k]);
        out[whole..].copy_from_slice(&f(args)[..rest]);
    }
}

macro_rules! fp16_binop {
    ($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Fp16Neon {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
                unsafe { $intrinsic_fn(&a, &b) }
            }

            #[doc = $doc]
            ///
            /// Whole slices; the tail runs through the same instruction on a
            /// zero-padded chunk (see the module doc).
            ///
            /// # Panics
            /// `a.len() != b.len() || out.len() != a.len()`.
            pub fn $slice_fn(self, a: &[u16], b: &[u16], out: &mut [u16]) {
                assert_eq!(a.len(), b.len());
                assert_eq!(out.len(), a.len());
                padded_chunks([a, b], out, |[x, y]| unsafe { $intrinsic_fn(x, y) });
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,fp16")]
        unsafe fn $intrinsic_fn(a: &[u16; 8], b: &[u16; 8]) -> [u16; 8] {
            unsafe {
                let va: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
                let vb: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr()));
                let mut out = [0u16; 8];
                vst1q_u16(out.as_mut_ptr(), vreinterpretq_u16_f16($intrinsic(va, vb)));
                out
            }
        }
    };
}

macro_rules! fp16_unop {
    ($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Fp16Neon {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [u16; 8]) -> [u16; 8] {
                unsafe { $intrinsic_fn(&a) }
            }

            #[doc = $doc]
            ///
            /// Whole slices; the tail runs through the same instruction on a
            /// zero-padded chunk.
            ///
            /// # Panics
            /// `out.len() != a.len()`.
            pub fn $slice_fn(self, a: &[u16], out: &mut [u16]) {
                assert_eq!(out.len(), a.len());
                padded_chunks([a], out, |[x]| unsafe { $intrinsic_fn(x) });
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,fp16")]
        unsafe fn $intrinsic_fn(a: &[u16; 8]) -> [u16; 8] {
            unsafe {
                let va: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
                let mut out = [0u16; 8];
                vst1q_u16(out.as_mut_ptr(), vreinterpretq_u16_f16($intrinsic(va)));
                out
            }
        }
    };
}

/// Compare: the intrinsic already returns `uint16x8_t` lane masks, so there is
/// no reinterpret on the way out.
macro_rules! fp16_cmp {
    ($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $doc:literal) => {
        impl Fp16Neon {
            #[doc = $doc]
            #[inline]
            pub fn $fixed_fn(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
                unsafe { $intrinsic_fn(&a, &b) }
            }

            #[doc = $doc]
            ///
            /// # Panics
            /// `a.len() != b.len() || out.len() != a.len()`.
            pub fn $slice_fn(self, a: &[u16], b: &[u16], out: &mut [u16]) {
                assert_eq!(a.len(), b.len());
                assert_eq!(out.len(), a.len());
                padded_chunks([a, b], out, |[x, y]| unsafe { $intrinsic_fn(x, y) });
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = "neon,fp16")]
        unsafe fn $intrinsic_fn(a: &[u16; 8], b: &[u16; 8]) -> [u16; 8] {
            unsafe {
                let va: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
                let vb: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr()));
                let vr: uint16x8_t = $intrinsic(va, vb);
                let mut out = [0u16; 8];
                vst1q_u16(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

fp16_binop!(
    add_f16x8,
    add_f16_slice,
    fadd_f16q_fixed,
    vaddq_f16,
    "Per-lane `f16` addition (`fadd v.8h`)."
);
fp16_binop!(
    sub_f16x8,
    sub_f16_slice,
    fsub_f16q_fixed,
    vsubq_f16,
    "Per-lane `f16` subtraction (`fsub v.8h`)."
);
fp16_binop!(
    mul_f16x8,
    mul_f16_slice,
    fmul_f16q_fixed,
    vmulq_f16,
    "Per-lane `f16` multiplication (`fmul v.8h`)."
);
fp16_binop!(
    div_f16x8,
    div_f16_slice,
    fdiv_f16q_fixed,
    vdivq_f16,
    "Per-lane `f16` division (`fdiv v.8h`)."
);
fp16_binop!(
    max_f16x8,
    max_f16_slice,
    fmax_f16q_fixed,
    vmaxq_f16,
    "Per-lane `f16` maximum (`fmax v.8h`). A NaN lane propagates, as the instruction does - not Rust's `max`, which prefers the number."
);
fp16_binop!(
    min_f16x8,
    min_f16_slice,
    fmin_f16q_fixed,
    vminq_f16,
    "Per-lane `f16` minimum (`fmin v.8h`). Same NaN caveat as [`Fp16Neon::max_f16x8`]."
);
fp16_unop!(
    abs_f16x8,
    abs_f16_slice,
    fabs_f16q_fixed,
    vabsq_f16,
    "Per-lane `f16` absolute value (`fabs v.8h`)."
);
fp16_unop!(
    neg_f16x8,
    neg_f16_slice,
    fneg_f16q_fixed,
    vnegq_f16,
    "Per-lane `f16` negation (`fneg v.8h`)."
);
fp16_unop!(
    sqrt_f16x8,
    sqrt_f16_slice,
    fsqrt_f16q_fixed,
    vsqrtq_f16,
    "Per-lane `f16` square root (`fsqrt v.8h`)."
);
fp16_cmp!(
    cmpeq_f16x8,
    cmpeq_f16_slice,
    fcmeq_f16q_fixed,
    vceqq_f16,
    "Per-lane `f16` equality as an all-ones-or-zero `u16` mask (`fcmeq v.8h`). NaN never equal."
);
fp16_cmp!(
    cmpgt_f16x8,
    cmpgt_f16_slice,
    fcmgt_f16q_fixed,
    vcgtq_f16,
    "Per-lane ordered `f16` greater-than as a `u16` mask (`fcmgt v.8h`)."
);
fp16_cmp!(
    cmpge_f16x8,
    cmpge_f16_slice,
    fcmge_f16q_fixed,
    vcgeq_f16,
    "Per-lane ordered `f16` greater-or-equal as a `u16` mask (`fcmge v.8h`)."
);
fp16_cmp!(
    cmplt_f16x8,
    cmplt_f16_slice,
    fcmlt_f16q_fixed,
    vcltq_f16,
    "Per-lane ordered `f16` less-than as a `u16` mask (`fcmgt v.8h`, operands swapped)."
);
fp16_cmp!(
    cmple_f16x8,
    cmple_f16_slice,
    fcmle_f16q_fixed,
    vcleq_f16,
    "Per-lane ordered `f16` less-or-equal as a `u16` mask (`fcmge v.8h`, operands swapped)."
);

impl Fp16Neon {
    /// Fused multiply-add `a * b + c` on `f16` lanes, one rounding (`fmla v.8h`).
    ///
    /// aarch32 spells this `Fp16Fma::fma_f16x8`; the name here follows the
    /// crate-wide `fmadd_*` (see `SIMD_NAME_MAP.md`).
    #[inline]
    pub fn fmadd_f16x8(self, a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
        unsafe { fmla_f16q_fixed(&a, &b, &c) }
    }

    /// Fused multiply-add over whole slices; unlike the `f32` form the tail is
    /// fused too (it goes through the same instruction on a padded chunk).
    ///
    /// # Panics
    /// `a`, `b`, `c` and `out` do not all have the same length.
    pub fn fmadd_f16_slice(self, a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        assert_eq!(out.len(), a.len());
        padded_chunks([a, b, c], out, |[x, y, z]| unsafe {
            fmla_f16q_fixed(x, y, z)
        });
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,fp16")]
unsafe fn fmla_f16q_fixed(a: &[u16; 8], b: &[u16; 8], c: &[u16; 8]) -> [u16; 8] {
    unsafe {
        let va = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
        let vb = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr()));
        let vc = vreinterpretq_f16_u16(vld1q_u16(c.as_ptr()));
        // ACLE: vfmaq_f16(acc, x, y) = acc + x * y - addend first.
        let vr = vfmaq_f16(vc, va, vb);
        let mut out = [0u16; 8];
        vst1q_u16(out.as_mut_ptr(), vreinterpretq_u16_f16(vr));
        out
    }
}

// ---- the other three fused forms (x86 `fmsub`/`fnmadd`/`fnmsub`)
//
// `FMLA`/`FMLS` plus an exact `FNEG` of the addend: negation never rounds, so
// each form still rounds once. Same sign conventions as `Neon::fmsub_f32x4`.

macro_rules! fp16_fma_variant {
	($fixed:ident, $slice:ident, $inner:ident, $formula:literal, $mnem:literal, |$a:ident, $b:ident, $c:ident| $body:expr) => {
		impl Fp16Neon {
			#[doc = concat!("Fused `", $formula, "` on `f16` lanes, one rounding (`", $mnem, "`).")]
			#[inline]
			pub fn $fixed(self, a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
				unsafe { $inner(&a, &b, &c) }
			}

			#[doc = concat!("[`Fp16Neon::", stringify!($fixed), "`] over whole slices; the padded tail is fused too.")]
			///
			/// # Panics
			/// `a`, `b`, `c` and `out` do not all have the same length.
			pub fn $slice(self, a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(a.len(), c.len());
				assert_eq!(out.len(), a.len());
				padded_chunks([a, b, c], out, |[x, y, z]| unsafe { $inner(x, y, z) });
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,fp16")]
		unsafe fn $inner(a: &[u16; 8], b: &[u16; 8], c: &[u16; 8]) -> [u16; 8] {
			unsafe {
				let $a: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(a.as_ptr()));
				let $b: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(b.as_ptr()));
				let $c: float16x8_t = vreinterpretq_f16_u16(vld1q_u16(c.as_ptr()));
				let vr = $body;
				let mut out = [0u16; 8];
				vst1q_u16(out.as_mut_ptr(), vreinterpretq_u16_f16(vr));
				out
			}
		}
	};
}

// ACLE: vfmaq_f16(acc, x, y) = acc + x·y, vfmsq_f16(acc, x, y) = acc - x·y.
fp16_fma_variant!(
    fmsub_f16x8,
    fmsub_f16_slice,
    fmsub_f16q_fixed,
    "a * b - c",
    "fneg + fmla v.8h",
    |a, b, c| vfmaq_f16(vnegq_f16(c), a, b)
);
fp16_fma_variant!(
    fnmadd_f16x8,
    fnmadd_f16_slice,
    fnmadd_f16q_fixed,
    "-(a * b) + c",
    "fmls v.8h",
    |a, b, c| vfmsq_f16(c, a, b)
);
fp16_fma_variant!(
    fnmsub_f16x8,
    fnmsub_f16_slice,
    fnmsub_f16q_fixed,
    "-(a * b) - c",
    "fneg + fmls v.8h",
    |a, b, c| vfmsq_f16(vnegq_f16(c), a, b)
);

#[cfg(test)]
#[path = "../../test/ops/neon/fp16.rs"]
mod tests;
