//! # Auto slice ops (PowerPC)
//!
//! Same names and contract as `riscv::auto` / `x86::reduce`: each function
//! reads the cached [`super::detect_features`] once and runs the best kernel:
//!
//! | Op group | Order |
//! |---|---|
//! | element-wise f32/i32 | `Isa300` > `Vsx` > `Vmx` > `PairedSingle` (f32) > portable |
//! | `dot_f32`, `reduce_sum_f32` | `Isa300` > `Vsx` > `Vmx128` > `Vmx` > `PairedSingle` > portable |
//! | max/min/argmax, `bf16_to_f32` | `Vsx` > `Vmx` > portable |
//! | `affine_u16_f32` | `Vsx` > `Vmx` > `PairedSingle` > portable |
//! | `gemm_f32_kernel` | `Mma` > portable |
//! | everything else | portable (`reduce_common`) |
//!
//! `Vmx` is built for big-endian targets (consoles, G4/G5); POWER7+ in either
//! endianness takes the VSX kernels. NaN is ignored by max/min/argmax/top-k/
//! compress, ties pick the lowest index, and mismatched lengths panic. Sums and
//! dot products add in a kernel-specific order, so the last bits can differ
//! between kernels.

#[allow(unused_imports, reason = "unused on targets without any kernel")]
use super::{FeatureSet, detect_features};

/// Declares one rung: the token type where the `cfg` holds, an uninhabited
/// stand-in elsewhere, and `fn $f(features) -> Option<token>`.
macro_rules! rung {
    ($f:ident, $alias:ident, $path:path, $($cfg:tt)*) => {
        #[cfg($($cfg)*)]
        type $alias = $path;
        #[cfg(not($($cfg)*))]
        type $alias = Never;
        #[inline]
        #[allow(unused_variables)]
        fn $f(features: FeatureSet) -> Option<$alias> {
            <$alias>::from_features(features)
        }
    };
}

rung!(
    vsx,
    VsxRung,
    super::ops::vsx::Vsx,
    any(target_arch = "powerpc", target_arch = "powerpc64")
);
rung!(
    isa300,
    Isa300Rung,
    super::ops::isa300::Isa300,
    target_arch = "powerpc64"
);
rung!(
    mma,
    MmaRung,
    super::ops::mma::Mma,
    target_arch = "powerpc64"
);
rung!(vmx, VmxRung, super::ops::vmx::Vmx, target_endian = "big");
rung!(
    paired,
    PairedRung,
    super::ops::paired::PairedSingle,
    target_arch = "powerpc"
);
rung!(
    vmx128,
    Vmx128Rung,
    super::ops::vmx128::Vmx128,
    all(target_arch = "powerpc64", target_endian = "big")
);

/// Uninhabited token for kernels a target does not build.
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Never {}

#[allow(dead_code, unused_variables)]
impl Never {
    fn from_features(features: FeatureSet) -> Option<Self> {
        None
    }
    fn add_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        match self {}
    }
    fn sub_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        match self {}
    }
    fn mul_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        match self {}
    }
    fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        match self {}
    }
    fn add_i32(self, a: &[i32], b: &[i32], out: &mut [i32]) {
        match self {}
    }
    fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        match self {}
    }
    fn reduce_max_f32(self, a: &[f32]) -> f32 {
        match self {}
    }
    fn reduce_min_f32(self, a: &[f32]) -> f32 {
        match self {}
    }
    fn argmax_f32(self, a: &[f32]) -> Option<(usize, f32)> {
        match self {}
    }
    fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        match self {}
    }
    fn bf16_to_f32(self, a: &[u16], out: &mut [f32]) {
        match self {}
    }
    fn affine_u16_f32(self, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
        match self {}
    }
    fn gemm_f32_kernel(self, kc: usize, a: &[f32], b: &[f32], acc: &mut [[f32; GEMM_NR]; GEMM_MR]) {
        match self {}
    }
}

/// `out[i] = a[i] + b[i]`.
///
/// # Panics
/// If the lengths differ.
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.add_f32(a, b, out);
    }
    if let Some(t) = vsx(f) {
        return t.add_f32(a, b, out);
    }
    if let Some(t) = vmx(f) {
        return t.add_f32(a, b, out);
    }
    if let Some(t) = paired(f) {
        return t.add_f32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// `out[i] = a[i] - b[i]`.
///
/// # Panics
/// If the lengths differ.
pub fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.sub_f32(a, b, out);
    }
    if let Some(t) = vsx(f) {
        return t.sub_f32(a, b, out);
    }
    if let Some(t) = vmx(f) {
        return t.sub_f32(a, b, out);
    }
    if let Some(t) = paired(f) {
        return t.sub_f32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// `out[i] = a[i] * b[i]`.
///
/// # Panics
/// If the lengths differ.
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.mul_f32(a, b, out);
    }
    if let Some(t) = vsx(f) {
        return t.mul_f32(a, b, out);
    }
    if let Some(t) = vmx(f) {
        return t.mul_f32(a, b, out);
    }
    if let Some(t) = paired(f) {
        return t.mul_f32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// `out[i] = a[i] * b[i] + c[i]`. Fused (one rounding) in the VMX vector part;
/// the portable path rounds twice.
///
/// # Panics
/// If the lengths differ.
pub fn fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.fmadd_f32(a, b, c, out);
    }
    if let Some(t) = vsx(f) {
        return t.fmadd_f32(a, b, c, out);
    }
    if let Some(t) = vmx(f) {
        return t.fmadd_f32(a, b, c, out);
    }
    if let Some(t) = paired(f) {
        return t.fmadd_f32(a, b, c, out);
    }
    assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i] + c[i];
    }
}

/// `out[i] = a[i] + b[i]`, wrapping.
///
/// # Panics
/// If the lengths differ.
pub fn add_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.add_i32(a, b, out);
    }
    if let Some(t) = vsx(f) {
        return t.add_i32(a, b, out);
    }
    if let Some(t) = vmx(f) {
        return t.add_i32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i].wrapping_add(b[i]);
    }
}

/// Sum of all elements (order is implementation-defined). NaN propagates.
pub fn reduce_sum_f32(a: &[f32]) -> f32 {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.reduce_sum_f32(a);
    }
    if let Some(t) = vsx(f) {
        return t.reduce_sum_f32(a);
    }
    if let Some(t) = vmx128(f) {
        return t.reduce_sum_f32(a);
    }
    if let Some(t) = vmx(f) {
        return t.reduce_sum_f32(a);
    }
    if let Some(t) = paired(f) {
        return t.reduce_sum_f32(a);
    }
    crate::reduce_common::sum4(a)
}

/// Largest element, NaN ignored; `-inf` if there is none.
pub fn reduce_max_f32(a: &[f32]) -> f32 {
    let f = detect_features();
    if let Some(t) = vsx(f) {
        return t.reduce_max_f32(a);
    }
    if let Some(t) = vmx(f) {
        return t.reduce_max_f32(a);
    }
    a.iter()
        .fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m })
}

/// Smallest element, NaN ignored; `+inf` if there is none.
pub fn reduce_min_f32(a: &[f32]) -> f32 {
    let f = detect_features();
    if let Some(t) = vsx(f) {
        return t.reduce_min_f32(a);
    }
    if let Some(t) = vmx(f) {
        return t.reduce_min_f32(a);
    }
    a.iter()
        .fold(f32::INFINITY, |m, &x| if x < m { x } else { m })
}

/// First index of the largest element and its value, NaN ignored.
/// `None` if the slice is empty or all NaN.
pub fn argmax_f32(a: &[f32]) -> Option<(usize, f32)> {
    let f = detect_features();
    if let Some(t) = vsx(f) {
        return t.argmax_f32(a);
    }
    if let Some(t) = vmx(f) {
        return t.argmax_f32(a);
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in a.iter().enumerate() {
        if x > best.map_or(f32::NEG_INFINITY, |b| b.1) || (best.is_none() && x == f32::NEG_INFINITY)
        {
            best = Some((i, x));
        }
    }
    best
}

/// Indices of all `a[i] > threshold`, ascending; returns the count. NaN never
/// matches. Portable code on every PowerPC so far.
///
/// # Panics
/// If `out.len() < a.len()`.
pub fn compress_gt_f32(a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    assert!(out.len() >= a.len(), "out must hold a.len() indices");
    let mut n = 0;
    for (i, &x) in a.iter().enumerate() {
        if x > threshold {
            out[n] = i as u32;
            n += 1;
        }
    }
    n
}

/// The `k = idx.len()` largest elements (NaN ignored), descending, ties to the
/// lower index. Returns how many were found.
///
/// # Panics
/// If `idx.len() != val.len()`.
pub fn topk_f32(a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    crate::reduce_common::topk_with(a, idx, val, compress_gt_f32)
}

/// Dot product of two `i8` slices into `i32`, wrapping. Portable code: VMX's
/// byte multiply-sum (`vmsummbm`) is signed x **unsigned** only.
///
/// # Panics
/// If the lengths differ.
pub fn dot_i8_i32(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32))
}

/// bfloat16 bits to `f32`, exact.
///
/// # Panics
/// If the lengths differ.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = vsx(f) {
        return t.bf16_to_f32(a, out);
    }
    if let Some(t) = vmx(f) {
        return t.bf16_to_f32(a, out);
    }
    assert_eq!(a.len(), out.len());
    crate::reduce_common::bf16_to_f32_scalar(a, out, 0);
}

/// `out[i] = exp(a[i])`, an approximation (relative error <= 3e-7, see
/// `x86::reduce::exp_f32`). Portable code.
///
/// # Panics
/// If the lengths differ.
pub fn exp_f32(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    crate::reduce_common::exp_f32_scalar(a, out, 0);
}

/// `out[i] = a * q[i] + b` with `q[i]` an unsigned 16-bit integer, two
/// roundings in every kernel and the portable code alike.
///
/// # Panics
/// If the lengths differ.
pub fn affine_u16_f32(q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    let f = detect_features();
    if let Some(t) = vsx(f) {
        return t.affine_u16_f32(q, a, b, out);
    }
    if let Some(t) = vmx(f) {
        return t.affine_u16_f32(q, a, b, out);
    }
    if let Some(t) = paired(f) {
        return t.affine_u16_f32(q, a, b, out);
    }
    assert_eq!(q.len(), out.len());
    crate::reduce_common::affine_u16_f32_scalar(q, a, b, out, 0);
}

/// Dot product `sum(a[i] * b[i])`. The kernels fuse per lane and add their
/// accumulators in their own order; the portable code keeps four lane sums
/// and rounds each product. NaN and inf propagate.
///
/// # Panics
/// If the lengths differ.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let f = detect_features();
    if let Some(t) = isa300(f) {
        return t.dot_f32(a, b);
    }
    if let Some(t) = vsx(f) {
        return t.dot_f32(a, b);
    }
    if let Some(t) = vmx128(f) {
        return t.dot_f32(a, b);
    }
    if let Some(t) = vmx(f) {
        return t.dot_f32(a, b);
    }
    if let Some(t) = paired(f) {
        return t.dot_f32(a, b);
    }
    assert_eq!(a.len(), b.len());
    crate::reduce_common::dot4(a, b, 0)
}

/// `ln(sum(exp(a[i])))` without overflow, same contract as
/// `x86::reduce::logsumexp_f32`, built from this module's max, `exp_f32` and sum.
pub fn logsumexp_f32(a: &[f32]) -> f32 {
    crate::reduce_common::logsumexp_with(a, reduce_max_f32, exp_f32, reduce_sum_f32)
}

/// `out[i] = a[i] - logsumexp_f32(a)`.
///
/// # Panics
/// If the lengths differ.
pub fn log_softmax_f32(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let l = logsumexp_f32(a);
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x - l;
    }
}

/// `out[i] = sum_k(signal[i + k] * taps[k])`, same contract as
/// `x86::reduce::sliding_dot_f32`. Portable code, k ascending.
///
/// # Panics
/// If `out` is not empty and `signal.len() + 1 < out.len() + taps.len()`.
pub fn sliding_dot_f32(signal: &[f32], taps: &[f32], out: &mut [f32]) {
    crate::reduce_common::sliding_check(signal, taps, out);
    crate::reduce_common::sliding_scalar(signal, taps, out, 0);
}

/// `out[i] = (a[i] * (den - num) + b[i] * num) / den` rounded to the nearest
/// (halves up). Same contract as `x86::reduce::lerp_u8`, exact. Portable code:
/// VMX has no exact vector divide (`vrefp` is an estimate), and how the
/// paired-single `u8` store rounds is not documented well enough to rely on.
///
/// # Panics
/// If the lengths differ, `den` is 0 or over 32767, or `num > den`.
pub fn lerp_u8(a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8]) {
    crate::reduce_common::lerp_check(a, b, num, den, out);
    crate::reduce_common::lerp_scalar(a, b, num, den, out, 0);
}

pub use crate::reduce_common::{GEMM_MR, GEMM_NR};

/// GEMM micro-kernel, same contract as `x86::reduce::gemm_f32_kernel`
/// (`acc[i][j] += sum_k a[k*MR + i] * b[k*NR + j]`, k ascending). Portable code
/// (product and sum rounded separately) everywhere except POWER10/11, where
/// the MMA accumulators run it with one rounding per step.
///
/// # Panics
/// If `a.len() < kc * GEMM_MR` or `b.len() < kc * GEMM_NR`.
pub fn gemm_f32_kernel(kc: usize, a: &[f32], b: &[f32], acc: &mut [[f32; GEMM_NR]; GEMM_MR]) {
    if let Some(t) = mma(detect_features()) {
        return t.gemm_f32_kernel(kc, a, b, acc);
    }
    crate::reduce_common::gemm_check(kc, a, b);
    crate::reduce_common::gemm_kernel_scalar(kc, a, b, acc)
}
