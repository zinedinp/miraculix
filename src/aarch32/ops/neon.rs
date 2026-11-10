//! ARMv7-A Neon vector ops (SSE/AVX analog). Token: [`Neon`]. Upstream
//! `int32x4_t`/`float32x4_t`/`uint32x4_t` under `stdarch_arm_neon_intrinsics`
//! (unstable on arm; stable on aarch64). Detect: [`Feature::Neon`](super::super::Feature::Neon).
//!
//! Written on the Neon DSL shared with aarch64 (`crate::arm_shared`):
//! each op has a fixed-width `[T; 4]` form and a whole-slice `_slice` form
//! whose tail (`< 4` lanes) is finished scalar-wise, exactly like
//! `auto_up`'s fallback. On arm every intrinsic needs `"neon"` plus the
//! `"v7"` baseline, so the feature string is `"neon,v7"`.

use crate::arm_shared::macros::{
    arm_token, neon_binop, neon_cmp, neon_select, neon_select_slice, neon_unop,
};
use crate::arm_shared::simd_arch::{
    float32x4_t, int32x4_t, uint32x4_t, vabsq_f32, vabsq_s32, vaddq_f32, vaddq_s32, vandq_u32,
    vbicq_u32, vbslq_f32, vbslq_s32, vbslq_u32, vceqq_f32, vceqq_s32, vcgeq_f32, vcgeq_s32,
    vcgtq_f32, vcgtq_s32, vcleq_f32, vcleq_s32, vcltq_f32, vcltq_s32, veorq_u32, vld1q_f32,
    vld1q_s32, vld1q_u32, vmaxq_f32, vmaxq_s32, vminq_f32, vminq_s32, vmulq_f32, vmulq_s32,
    vmvnq_u32, vnegq_f32, vnegq_s32, vorrq_u32, vshlq_s32, vst1q_f32, vst1q_s32, vst1q_u32,
    vsubq_f32, vsubq_s32,
};

arm_token! {
    /// Proof that Neon is available.
    token = Neon, requires = [Neon],
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = add_i32x4, slice_fn = add_i32_slice, intrinsic_fn = add_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vaddq_s32,
    scalar = |x, y| x.wrapping_add(y),
    fixed_doc = "`VADD.S32`: per-lane `i32` addition.",
    slice_doc = "`VADD.S32`: per-lane `i32` addition. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = sub_i32x4, slice_fn = sub_i32_slice, intrinsic_fn = sub_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vsubq_s32,
    scalar = |x, y| x.wrapping_sub(y),
    fixed_doc = "`VSUB.S32`: per-lane `i32` subtraction.",
    slice_doc = "`VSUB.S32`: per-lane `i32` subtraction. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = mul_i32x4, slice_fn = mul_i32_slice, intrinsic_fn = mul_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vmulq_s32,
    scalar = |x, y| x.wrapping_mul(y),
    fixed_doc = "`VMUL.S32`: per-lane `i32` multiplication (low 32 bits of the product).",
    slice_doc = "`VMUL.S32`: per-lane `i32` multiplication (low 32 bits of the product). A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = add_f32x4, slice_fn = add_f32_slice, intrinsic_fn = add_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vaddq_f32,
    scalar = |x, y| x + y,
    fixed_doc = "`VADD.F32`: per-lane `f32` addition.",
    slice_doc = "`VADD.F32`: per-lane `f32` addition. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = sub_f32x4, slice_fn = sub_f32_slice, intrinsic_fn = sub_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vsubq_f32,
    scalar = |x, y| x - y,
    fixed_doc = "`VSUB.F32`: per-lane `f32` subtraction.",
    slice_doc = "`VSUB.F32`: per-lane `f32` subtraction. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = mul_f32x4, slice_fn = mul_f32_slice, intrinsic_fn = mul_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vmulq_f32,
    scalar = |x, y| x * y,
    fixed_doc = "`VMUL.F32`: per-lane `f32` multiplication.",
    slice_doc = "`VMUL.F32`: per-lane `f32` multiplication. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = and_u32x4, slice_fn = and_u32_slice, intrinsic_fn = and_u32x4_imp,
    lanes = 4, elem = u32, vec = uint32x4_t,
    loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = vandq_u32,
    scalar = |x, y| x & y,
    fixed_doc = "`VAND`: per-lane `u32` bitwise AND.",
    slice_doc = "`VAND`: per-lane `u32` bitwise AND. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = or_u32x4, slice_fn = or_u32_slice, intrinsic_fn = or_u32x4_imp,
    lanes = 4, elem = u32, vec = uint32x4_t,
    loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = vorrq_u32,
    scalar = |x, y| x | y,
    fixed_doc = "`VORR`: per-lane `u32` bitwise OR.",
    slice_doc = "`VORR`: per-lane `u32` bitwise OR. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = xor_u32x4, slice_fn = xor_u32_slice, intrinsic_fn = xor_u32x4_imp,
    lanes = 4, elem = u32, vec = uint32x4_t,
    loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = veorq_u32,
    scalar = |x, y| x ^ y,
    fixed_doc = "`VEOR`: per-lane `u32` bitwise XOR.",
    slice_doc = "`VEOR`: per-lane `u32` bitwise XOR. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = andnot_u32x4, slice_fn = andnot_u32_slice, intrinsic_fn = andnot_u32x4_imp,
    lanes = 4, elem = u32, vec = uint32x4_t,
    loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = vbicq_u32,
    scalar = |x, y| x & !y,
    fixed_doc = "`VBIC`: per-lane `u32` bitwise AND-NOT, `a & !b` (native `vbicq_u32` operand order; x86's `andnot` convention is the mirror image, `!a & b` - callers wanting that swap arguments at the call site).",
    slice_doc = "`VBIC`: per-lane `u32` bitwise AND-NOT, `a & !b` (native `vbicq_u32` operand order; x86's `andnot` convention is the mirror image, `!a & b` - callers wanting that swap arguments at the call site). A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpeq_i32x4, slice_fn = cmpeq_i32_slice, intrinsic_fn = cmpeq_i32x4_imp,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_storeu = vst1q_u32, intrinsic = vceqq_s32,
    scalar = |x, y| if x == y { u32::MAX } else { 0 },
    fixed_doc = "`VCEQ.S32`: per-lane `i32` equality, `[u32; 4]` lane mask (all-1s or 0, not `bool`).",
    slice_doc = "`VCEQ.S32`: per-lane `i32` equality, `[u32; 4]` lane mask (all-1s or 0, not `bool`). A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpgt_i32x4, slice_fn = cmpgt_i32_slice, intrinsic_fn = cmpgt_i32x4_imp,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_storeu = vst1q_u32, intrinsic = vcgtq_s32,
    scalar = |x, y| if x > y { u32::MAX } else { 0 },
    fixed_doc = "`VCGT.S32`: per-lane `i32` greater-than, `[u32; 4]` lane mask.",
    slice_doc = "`VCGT.S32`: per-lane `i32` greater-than, `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpge_i32x4, slice_fn = cmpge_i32_slice, intrinsic_fn = cmpge_i32x4_imp,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_storeu = vst1q_u32, intrinsic = vcgeq_s32,
    scalar = |x, y| if x >= y { u32::MAX } else { 0 },
    fixed_doc = "`VCGE.S32`: per-lane `i32` greater-or-equal, `[u32; 4]` lane mask.",
    slice_doc = "`VCGE.S32`: per-lane `i32` greater-or-equal, `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmplt_i32x4, slice_fn = cmplt_i32_slice, intrinsic_fn = cmplt_i32x4_imp,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_storeu = vst1q_u32, intrinsic = vcltq_s32,
    scalar = |x, y| if x < y { u32::MAX } else { 0 },
    fixed_doc = "`VCLT.S32`: per-lane `i32` less-than, `[u32; 4]` lane mask.",
    slice_doc = "`VCLT.S32`: per-lane `i32` less-than, `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmple_i32x4, slice_fn = cmple_i32_slice, intrinsic_fn = cmple_i32x4_imp,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_storeu = vst1q_u32, intrinsic = vcleq_s32,
    scalar = |x, y| if x <= y { u32::MAX } else { 0 },
    fixed_doc = "`VCLE.S32`: per-lane `i32` less-or-equal, `[u32; 4]` lane mask.",
    slice_doc = "`VCLE.S32`: per-lane `i32` less-or-equal, `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpeq_f32x4, slice_fn = cmpeq_f32_slice, intrinsic_fn = cmpeq_f32x4_imp,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_storeu = vst1q_u32, intrinsic = vceqq_f32,
    scalar = |x, y| if x == y { u32::MAX } else { 0 },
    fixed_doc = "`VCEQ.F32`: per-lane `f32` equality, `[u32; 4]` lane mask. NaN never equals (mask 0), matching this crate's x86 auto convention.",
    slice_doc = "`VCEQ.F32`: per-lane `f32` equality, `[u32; 4]` lane mask. NaN never equals (mask 0), matching this crate's x86 auto convention. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpgt_f32x4, slice_fn = cmpgt_f32_slice, intrinsic_fn = cmpgt_f32x4_imp,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_storeu = vst1q_u32, intrinsic = vcgtq_f32,
    scalar = |x, y| if x > y { u32::MAX } else { 0 },
    fixed_doc = "`VCGT.F32`: per-lane `f32` greater-than (ordered; false if either lane is NaN), `[u32; 4]` lane mask.",
    slice_doc = "`VCGT.F32`: per-lane `f32` greater-than (ordered; false if either lane is NaN), `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmpge_f32x4, slice_fn = cmpge_f32_slice, intrinsic_fn = cmpge_f32x4_imp,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_storeu = vst1q_u32, intrinsic = vcgeq_f32,
    scalar = |x, y| if x >= y { u32::MAX } else { 0 },
    fixed_doc = "`VCGE.F32`: per-lane `f32` greater-or-equal (ordered), `[u32; 4]` lane mask.",
    slice_doc = "`VCGE.F32`: per-lane `f32` greater-or-equal (ordered), `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmplt_f32x4, slice_fn = cmplt_f32_slice, intrinsic_fn = cmplt_f32x4_imp,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_storeu = vst1q_u32, intrinsic = vcltq_f32,
    scalar = |x, y| if x < y { u32::MAX } else { 0 },
    fixed_doc = "`VCLT.F32`: per-lane `f32` less-than (ordered), `[u32; 4]` lane mask.",
    slice_doc = "`VCLT.F32`: per-lane `f32` less-than (ordered), `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_cmp! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = cmple_f32x4, slice_fn = cmple_f32_slice, intrinsic_fn = cmple_f32x4_imp,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_storeu = vst1q_u32, intrinsic = vcleq_f32,
    scalar = |x, y| if x <= y { u32::MAX } else { 0 },
    fixed_doc = "`VCLE.F32`: per-lane `f32` less-or-equal (ordered), `[u32; 4]` lane mask.",
    slice_doc = "`VCLE.F32`: per-lane `f32` less-or-equal (ordered), `[u32; 4]` lane mask. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = max_i32x4, slice_fn = max_i32_slice, intrinsic_fn = max_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vmaxq_s32,
    scalar = |x, y| x.max(y),
    fixed_doc = "`VMAX.S32`: per-lane `i32` maximum.",
    slice_doc = "`VMAX.S32`: per-lane `i32` maximum. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = min_i32x4, slice_fn = min_i32_slice, intrinsic_fn = min_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vminq_s32,
    scalar = |x, y| x.min(y),
    fixed_doc = "`VMIN.S32`: per-lane `i32` minimum.",
    slice_doc = "`VMIN.S32`: per-lane `i32` minimum. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = max_f32x4, slice_fn = max_f32_slice, intrinsic_fn = max_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vmaxq_f32,
    scalar = |x, y| x.max(y),
    fixed_doc = "`VMAX.F32`: per-lane `f32` maximum. NaN follows the `VMAX` instruction, not Rust `f32::max` (same SIMD-vs-scalar caveat as x86 `auto_up`).",
    slice_doc = "`VMAX.F32`: per-lane `f32` maximum. NaN follows the `VMAX` instruction, not Rust `f32::max` (same SIMD-vs-scalar caveat as x86 `auto_up`). A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = min_f32x4, slice_fn = min_f32_slice, intrinsic_fn = min_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vminq_f32,
    scalar = |x, y| x.min(y),
    fixed_doc = "`VMIN.F32`: per-lane `f32` minimum. Same NaN caveat as [`Neon::max_f32x4`].",
    slice_doc = "`VMIN.F32`: per-lane `f32` minimum. Same NaN caveat as [`Neon::max_f32x4`]. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_unop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = abs_i32x4, slice_fn = abs_i32_slice, intrinsic_fn = abs_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vabsq_s32,
    scalar = |x| x.wrapping_abs(),
    fixed_doc = "`VABS.S32`: per-lane `i32` absolute value.",
    slice_doc = "`VABS.S32`: per-lane `i32` absolute value. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_unop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = neg_i32x4, slice_fn = neg_i32_slice, intrinsic_fn = neg_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vnegq_s32,
    scalar = |x| x.wrapping_neg(),
    fixed_doc = "`VNEG.S32`: per-lane `i32` negation.",
    slice_doc = "`VNEG.S32`: per-lane `i32` negation. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_unop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = abs_f32x4, slice_fn = abs_f32_slice, intrinsic_fn = abs_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vabsq_f32,
    scalar = |x| x.abs(),
    fixed_doc = "`VABS.F32`: per-lane `f32` absolute value.",
    slice_doc = "`VABS.F32`: per-lane `f32` absolute value. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_unop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = neg_f32x4, slice_fn = neg_f32_slice, intrinsic_fn = neg_f32x4_imp,
    lanes = 4, elem = f32, vec = float32x4_t,
    loadu = vld1q_f32, storeu = vst1q_f32, intrinsic = vnegq_f32,
    scalar = |x| -x,
    fixed_doc = "`VNEG.F32`: per-lane `f32` negation.",
    slice_doc = "`VNEG.F32`: per-lane `f32` negation. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_unop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = not_u32x4, slice_fn = not_u32_slice, intrinsic_fn = not_u32x4_imp,
    lanes = 4, elem = u32, vec = uint32x4_t,
    loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = vmvnq_u32,
    scalar = |x| !x,
    fixed_doc = "`VMVN`: per-lane `u32` bitwise NOT.",
    slice_doc = "`VMVN`: per-lane `u32` bitwise NOT. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_binop! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = shl_i32x4, slice_fn = shl_i32_slice, intrinsic_fn = shl_i32x4_imp,
    lanes = 4, elem = i32, vec = int32x4_t,
    loadu = vld1q_s32, storeu = vst1q_s32, intrinsic = vshlq_s32,
    scalar = |x, y| if y >= 32 { 0 } else if y <= -32 { if x < 0 { -1 } else { 0 } } else if y >= 0 { x.wrapping_shl(y as u32) } else { x >> (-y) as u32 },
    fixed_doc = "`VSHL.S32`: per-lane variable `i32` shift. Positive `b[i]` shifts left, negative shifts right (arithmetic); magnitudes `>= 32` saturate per `VSHL`, not Rust shift-amount panics.",
    slice_doc = "`VSHL.S32`: per-lane variable `i32` shift. Positive `b[i]` shifts left, negative shifts right (arithmetic); magnitudes `>= 32` saturate per `VSHL`, not Rust shift-amount panics. A tail shorter than 4 lanes is finished scalar-wise.",
}

neon_select! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    fixed_fn = select_u32x4, slice_fn = select_u32_slice, intrinsic_fn = select_u32x4_imp,
    lanes = 4, elem = u32, mask = u32,
    vec = uint32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_u32, mask_loadu = vld1q_u32, storeu = vst1q_u32, intrinsic = vbslq_u32,
    scalar = |m, x, y| if m != 0 { x } else { y },
    fixed_doc = "`VBSL`: per-lane bit-select; `mask` picks bits of `a` where set, of `b` where clear.",
    slice_doc = "`VBSL`: per-lane bit-select; `mask` picks bits of `a` where set, of `b` where clear. The scalar tail takes `a` where the mask lane is non-zero, else `b` (same as `auto_up::select_*`), so it matches the vector body for all-1s/0 masks.",
}

neon_select_slice! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    slice_fn = select_i32_slice,
    lanes = 4, elem = i32, mask = u32,
    vec = int32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_s32, mask_loadu = vld1q_u32, storeu = vst1q_s32, intrinsic = vbslq_s32,
    scalar = |m, x, y| if m != 0 { x } else { y },
    slice_doc = "`VBSL`: per-lane bit-select; `mask` picks bits of `a` where set, of `b` where clear. `i32` lanes, `u32` mask. The scalar tail takes `a` where the mask lane is non-zero, else `b` (same as `auto_up::select_*`), so it matches the vector body for all-1s/0 masks.",
}

neon_select_slice! {
    token = Neon, vis = pub, target_feature = "neon,v7",
    slice_fn = select_f32_slice,
    lanes = 4, elem = f32, mask = u32,
    vec = float32x4_t, mask_vec = uint32x4_t,
    loadu = vld1q_f32, mask_loadu = vld1q_u32, storeu = vst1q_f32, intrinsic = vbslq_f32,
    scalar = |m, x, y| if m != 0 { x } else { y },
    slice_doc = "`VBSL`: per-lane bit-select; `mask` picks bits of `a` where set, of `b` where clear. `f32` lanes, `u32` mask. The scalar tail takes `a` where the mask lane is non-zero, else `b` (same as `auto_up::select_*`), so it matches the vector body for all-1s/0 masks.",
}

#[cfg(test)]
#[path = "../test/ops/neon.rs"]
mod tests;
