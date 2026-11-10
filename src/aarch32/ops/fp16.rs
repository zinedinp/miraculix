//! ARMv8.2-A FullFP16 Neon (vector half-precision). Tokens: [`Fp16Neon`],
//! [`Fp16Fma`]. Detect on [`Feature::AsimdHp`](super::super::Feature::AsimdHp)
//! (`ASIMDHP`), not [`Feature::Fp16`](super::super::Feature::Fp16) (`FPHP` =
//! scalar VFP half). Public surface is `[u16; 8]` bit patterns (same as x86
//! BF16 here). `Fp16Fma` also requires
//! [`Feature::Vfpv4`](super::super::Feature::Vfpv4) for `vfmaq_f16` (extra
//! bit combo = own token).
//!
//! Written on the shared Neon DSL (`crate::arm_shared`); `load_f16` /
//! `store_f16` bridge the `u16` lanes to `float16x8_t`. Slice tails run
//! through scalar `f16` (as `auto_up` always did), unlike aarch64's
//! `Fp16Neon`, which pads the tail into one more vector op.

use crate::arm_shared::macros::{arm_token, neon_binop, neon_cmp, neon_fma_slice, neon_unop};
use crate::arm_shared::simd_arch::{
    float16x8_t, uint16x8_t, vabsq_f16, vaddq_f16, vceqq_f16, vcgeq_f16, vcgtq_f16, vcleq_f16,
    vcltq_f16, vfmaq_f16, vld1q_u16, vmaxq_f16, vminq_f16, vmulq_f16, vnegq_f16,
    vreinterpretq_f16_u16, vreinterpretq_u16_f16, vst1q_u16, vsubq_f16,
};

arm_token! {
    /// Proof that the FullFP16 Neon extension is available.
    token = Fp16Neon, requires = [Neon, AsimdHp],
}

arm_token! {
    /// Proof that FullFP16 Neon and VFPv4 (`vfmaq_f16`) are both available.
    token = Fp16Fma, requires = [Neon, AsimdHp, Vfpv4],
}

/// Scalar tails of the slice forms, the same per-lane formulas as
/// `auto_down`'s `f16` fallbacks.
///
/// Plain fns on purpose, passed by path: a closure written inside the
/// `#[target_feature]` slice body would inherit `fp16`, and in a debug build
/// its `f16` values then cross into `core`'s non-`fp16` helpers
/// (`f16::from_bits`, ...) in a different register than they are expected,
/// which returned the input unchanged for `neg`. Only `u16` crosses here.
mod tail {
    fn h(x: u16) -> f16 {
        f16::from_bits(x)
    }
    fn mask(b: bool) -> u16 {
        if b { u16::MAX } else { 0 }
    }
    pub(super) fn add(x: u16, y: u16) -> u16 {
        (h(x) + h(y)).to_bits()
    }
    pub(super) fn sub(x: u16, y: u16) -> u16 {
        (h(x) - h(y)).to_bits()
    }
    pub(super) fn mul(x: u16, y: u16) -> u16 {
        (h(x) * h(y)).to_bits()
    }
    pub(super) fn max(x: u16, y: u16) -> u16 {
        if h(x) > h(y) { x } else { y }
    }
    pub(super) fn min(x: u16, y: u16) -> u16 {
        if h(x) < h(y) { x } else { y }
    }
    pub(super) fn abs(x: u16) -> u16 {
        let v = h(x);
        (if v < 0.0 { -v } else { v }).to_bits()
    }
    pub(super) fn neg(x: u16) -> u16 {
        (-h(x)).to_bits()
    }
    pub(super) fn cmpeq(x: u16, y: u16) -> u16 {
        mask(h(x) == h(y))
    }
    pub(super) fn cmpgt(x: u16, y: u16) -> u16 {
        mask(h(x) > h(y))
    }
    pub(super) fn cmpge(x: u16, y: u16) -> u16 {
        mask(h(x) >= h(y))
    }
    pub(super) fn cmplt(x: u16, y: u16) -> u16 {
        mask(h(x) < h(y))
    }
    pub(super) fn cmple(x: u16, y: u16) -> u16 {
        mask(h(x) <= h(y))
    }
    pub(super) fn fmadd(x: u16, y: u16, z: u16) -> u16 {
        (h(x) * h(y) + h(z)).to_bits()
    }
}

/// `vld1q_u16` + reinterpret: 8 `f16` bit patterns into a `float16x8_t`.
///
/// # Safety
/// `p` is valid for 8 `u16` reads; caller proved FullFP16 Neon.
#[inline]
#[target_feature(enable = "neon,v7,fp16")]
unsafe fn load_f16(p: *const u16) -> float16x8_t {
    unsafe { vreinterpretq_f16_u16(vld1q_u16(p)) }
}

/// Reinterpret + `vst1q_u16`: the inverse of [`load_f16`].
///
/// # Safety
/// `p` is valid for 8 `u16` writes; caller proved FullFP16 Neon.
#[inline]
#[target_feature(enable = "neon,v7,fp16")]
unsafe fn store_f16(p: *mut u16, v: float16x8_t) {
    unsafe { vst1q_u16(p, vreinterpretq_u16_f16(v)) }
}

neon_binop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = add_f16x8, slice_fn = add_f16_slice, intrinsic_fn = add_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vaddq_f16,
    scalar = tail::add,
    fixed_doc = "`VADD.F16`: per-lane `f16` addition.",
    slice_doc = "`VADD.F16`: per-lane `f16` addition. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_binop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = sub_f16x8, slice_fn = sub_f16_slice, intrinsic_fn = sub_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vsubq_f16,
    scalar = tail::sub,
    fixed_doc = "`VSUB.F16`: per-lane `f16` subtraction.",
    slice_doc = "`VSUB.F16`: per-lane `f16` subtraction. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_binop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = mul_f16x8, slice_fn = mul_f16_slice, intrinsic_fn = mul_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vmulq_f16,
    scalar = tail::mul,
    fixed_doc = "`VMUL.F16`: per-lane `f16` multiplication.",
    slice_doc = "`VMUL.F16`: per-lane `f16` multiplication. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_binop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = max_f16x8, slice_fn = max_f16_slice, intrinsic_fn = max_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vmaxq_f16,
    scalar = tail::max,
    fixed_doc = "`VMAX.F16`: per-lane `f16` maximum. NaN follows the `VMAX` instruction, not a scalar `max` (same caveat as [`super::neon::Neon::max_f32x4`]).",
    slice_doc = "`VMAX.F16`: per-lane `f16` maximum. NaN follows the `VMAX` instruction, not a scalar `max` (same caveat as [`super::neon::Neon::max_f32x4`]). A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_binop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = min_f16x8, slice_fn = min_f16_slice, intrinsic_fn = min_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vminq_f16,
    scalar = tail::min,
    fixed_doc = "`VMIN.F16`: per-lane `f16` minimum. Same NaN caveat as [`Fp16Neon::max_f16x8`].",
    slice_doc = "`VMIN.F16`: per-lane `f16` minimum. Same NaN caveat as [`Fp16Neon::max_f16x8`]. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_unop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = abs_f16x8, slice_fn = abs_f16_slice, intrinsic_fn = abs_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vabsq_f16,
    scalar = tail::abs,
    fixed_doc = "`VABS.F16`: per-lane `f16` absolute value.",
    slice_doc = "`VABS.F16`: per-lane `f16` absolute value. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_unop! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = neg_f16x8, slice_fn = neg_f16_slice, intrinsic_fn = neg_f16x8_imp,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vnegq_f16,
    scalar = tail::neg,
    fixed_doc = "`VNEG.F16`: per-lane `f16` negation.",
    slice_doc = "`VNEG.F16`: per-lane `f16` negation. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_cmp! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = cmpeq_f16x8, slice_fn = cmpeq_f16_slice, intrinsic_fn = cmpeq_f16x8_imp,
    lanes = 8, elem = u16, mask = u16,
    vec = float16x8_t, mask_vec = uint16x8_t,
    loadu = load_f16, mask_storeu = vst1q_u16, intrinsic = vceqq_f16,
    scalar = tail::cmpeq,
    fixed_doc = "`VCEQ.F16`: per-lane `f16` equality, `[u16; 8]` lane mask (all-1s or 0, not `bool`). NaN never equals (mask 0).",
    slice_doc = "`VCEQ.F16`: per-lane `f16` equality, `[u16; 8]` lane mask (all-1s or 0, not `bool`). NaN never equals (mask 0). A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_cmp! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = cmpgt_f16x8, slice_fn = cmpgt_f16_slice, intrinsic_fn = cmpgt_f16x8_imp,
    lanes = 8, elem = u16, mask = u16,
    vec = float16x8_t, mask_vec = uint16x8_t,
    loadu = load_f16, mask_storeu = vst1q_u16, intrinsic = vcgtq_f16,
    scalar = tail::cmpgt,
    fixed_doc = "`VCGT.F16`: per-lane `f16` greater-than (ordered; false if either lane is NaN), `[u16; 8]` lane mask.",
    slice_doc = "`VCGT.F16`: per-lane `f16` greater-than (ordered; false if either lane is NaN), `[u16; 8]` lane mask. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_cmp! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = cmpge_f16x8, slice_fn = cmpge_f16_slice, intrinsic_fn = cmpge_f16x8_imp,
    lanes = 8, elem = u16, mask = u16,
    vec = float16x8_t, mask_vec = uint16x8_t,
    loadu = load_f16, mask_storeu = vst1q_u16, intrinsic = vcgeq_f16,
    scalar = tail::cmpge,
    fixed_doc = "`VCGE.F16`: per-lane `f16` greater-or-equal (ordered), `[u16; 8]` lane mask.",
    slice_doc = "`VCGE.F16`: per-lane `f16` greater-or-equal (ordered), `[u16; 8]` lane mask. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_cmp! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = cmplt_f16x8, slice_fn = cmplt_f16_slice, intrinsic_fn = cmplt_f16x8_imp,
    lanes = 8, elem = u16, mask = u16,
    vec = float16x8_t, mask_vec = uint16x8_t,
    loadu = load_f16, mask_storeu = vst1q_u16, intrinsic = vcltq_f16,
    scalar = tail::cmplt,
    fixed_doc = "`VCLT.F16`: per-lane `f16` less-than (ordered), `[u16; 8]` lane mask.",
    slice_doc = "`VCLT.F16`: per-lane `f16` less-than (ordered), `[u16; 8]` lane mask. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

neon_cmp! {
    token = Fp16Neon, vis = pub, target_feature = "neon,v7,v8,fp16",
    fixed_fn = cmple_f16x8, slice_fn = cmple_f16_slice, intrinsic_fn = cmple_f16x8_imp,
    lanes = 8, elem = u16, mask = u16,
    vec = float16x8_t, mask_vec = uint16x8_t,
    loadu = load_f16, mask_storeu = vst1q_u16, intrinsic = vcleq_f16,
    scalar = tail::cmple,
    fixed_doc = "`VCLE.F16`: per-lane `f16` less-or-equal (ordered), `[u16; 8]` lane mask.",
    slice_doc = "`VCLE.F16`: per-lane `f16` less-or-equal (ordered), `[u16; 8]` lane mask. A tail shorter than 8 lanes is finished with scalar `f16`.",
}

impl Fp16Fma {
    /// `VFMA.F16`: per-lane `f16` fused multiply-add, `b * c + a` (`a` is the
    /// accumulator, matching `vfmaq_f16` and
    /// [`super::vfpv4neon::Vfpv4Neon::fma_f32x4`], not x86 `fmadd`). No
    /// `fms_f16x8` (same asymmetry as `fma_f32x4`).
    #[inline]
    pub fn fma_f16x8(self, a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
        #[target_feature(enable = "neon,v7,fp16,vfp4")]
        unsafe fn imp(a: [u16; 8], b: [u16; 8], c: [u16; 8]) -> [u16; 8] {
            unsafe {
                let rv = vfmaq_f16(
                    load_f16(a.as_ptr()),
                    load_f16(b.as_ptr()),
                    load_f16(c.as_ptr()),
                );
                let mut out = [0u16; 8];
                store_f16(out.as_mut_ptr(), rv);
                out
            }
        }
        unsafe { imp(a, b, c) }
    }
}

neon_fma_slice! {
    token = Fp16Fma, vis = pub, target_feature = "neon,v7,fp16,vfp4",
    slice_fn = fmadd_f16_slice,
    lanes = 8, elem = u16, vec = float16x8_t,
    loadu = load_f16, storeu = store_f16, intrinsic = vfmaq_f16,
    scalar = tail::fmadd,
    slice_doc = "`VFMA.F16` over whole slices: `out = a * b + c` (the `fmadd` order, not [`Fp16Fma::fma_f16x8`]'s). The vector body is fused; a tail shorter than 8 lanes is not (scalar `f16`, `a * b + c` as written), matching `auto_up::fmadd_f16`.",
}

#[cfg(test)]
#[path = "../test/ops/fp16.rs"]
mod tests;
