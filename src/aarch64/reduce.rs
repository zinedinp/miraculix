//! # Slice reductions and searches (`f32`), Neon
//!
//! Same names and rules as `x86::reduce` and `riscv::auto`: **NaN is ignored**
//! by max/min/argmax/top-k/compress (the `FMAXNM`/`FMINNM` family), ties pick
//! the lowest index, `reduce_sum_f32` propagates NaN and adds in a
//! vector-lane order (so its last bits differ from a sequential sum).
//!
//! Neon has no compress instruction, so [`compress_gt_f32`] compares 16
//! elements at a time and only visits the lanes of blocks that have a hit;
//! after a top-k threshold warms up almost every block is skipped.
//!
//! ```
//! # #[cfg(target_arch = "aarch64")] {
//! use miraculix::aarch64::reduce;
//! let logits = [0.5f32, -1.0, 7.25, 3.0, f32::NAN];
//! assert_eq!(reduce::argmax_f32(&logits), Some((2, 7.25)));
//! # }
//! ```

use core::arch::aarch64::*;

use super::ops::neon::neon::Neon;
use crate::reduce_common::{EXP_HI, EXP_LO, LN2_HI, LN2_LO, LOG2E, P};

#[inline]
fn hw() -> Option<Neon> {
    // Every aarch64 target except `*-softfloat` is built with `neon`, so the
    // binary already assumes it: no cache read per call.
    if cfg!(target_feature = "neon") {
        Some(Neon::assume_baseline())
    } else {
        Neon::detect()
    }
}

// ---------------------------------------------------------------- bf16 -> f32

/// `out[i] = f32::from_bits((a[i] as u32) << 16)`: bfloat16 bits to `f32`. Exact.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    bf16_to_f32_with(hw(), a, out)
}

/// [`bf16_to_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn bf16_to_f32_with(neon: Option<Neon>, a: &[u16], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    // SAFETY (all `unsafe` here): `neon` is `Some` only with a `Neon` token, which only detection hands out;
    // pointers stay inside the slices.
    let done = if neon.is_some() {
        unsafe { bf16_neon(a, out) }
    } else {
        0
    };
    crate::reduce_common::bf16_to_f32_scalar(a, out, done);
}

#[target_feature(enable = "neon")]
unsafe fn bf16_neon(a: &[u16], out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        while i + 8 <= a.len() {
            let v = vld1q_u16(a.as_ptr().add(i));
            vst1q_u32(
                out.as_mut_ptr().add(i) as *mut u32,
                vshll_n_u16::<16>(vget_low_u16(v)),
            );
            vst1q_u32(
                out.as_mut_ptr().add(i + 4) as *mut u32,
                vshll_high_n_u16::<16>(v),
            );
            i += 8;
        }
    }
    i
}

// ------------------------------------------------------------ sum / max / min

/// Sum of all elements (lane order; NaN propagates).
pub fn reduce_sum_f32(a: &[f32]) -> f32 {
    reduce_sum_f32_with(hw(), a)
}

/// [`reduce_sum_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn reduce_sum_f32_with(neon: Option<Neon>, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut s, done) = if neon.is_some() {
        unsafe { sum_neon(a) }
    } else {
        (0.0, 0)
    };
    s += crate::reduce_common::sum4(&a[done..]);
    s
}

#[target_feature(enable = "neon")]
unsafe fn sum_neon(a: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                *ac = vaddq_f32(*ac, vld1q_f32(a.as_ptr().add(i + 4 * k)));
            }
            i += 16;
        }
        while i + 4 <= a.len() {
            acc[0] = vaddq_f32(acc[0], vld1q_f32(a.as_ptr().add(i)));
            i += 4;
        }
        let t = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        (vaddvq_f32(t), i)
    }
}

macro_rules! ext_kernel {
    ($name:ident, $vop:ident, $vred:ident, $init:expr) => {
        #[target_feature(enable = "neon")]
        unsafe fn $name(a: &[f32]) -> (f32, usize) {
            unsafe {
                // `FMAXNM`/`FMINNM`: a quiet NaN loses to a number.
                let mut acc = [vdupq_n_f32($init); 4];
                let mut i = 0;
                while i + 16 <= a.len() {
                    for (k, ac) in acc.iter_mut().enumerate() {
                        *ac = $vop(*ac, vld1q_f32(a.as_ptr().add(i + 4 * k)));
                    }
                    i += 16;
                }
                while i + 4 <= a.len() {
                    acc[0] = $vop(acc[0], vld1q_f32(a.as_ptr().add(i)));
                    i += 4;
                }
                ($vred($vop($vop(acc[0], acc[1]), $vop(acc[2], acc[3]))), i)
            }
        }
    };
}
ext_kernel!(max_neon, vmaxnmq_f32, vmaxnmvq_f32, f32::NEG_INFINITY);
ext_kernel!(min_neon, vminnmq_f32, vminnmvq_f32, f32::INFINITY);

/// Largest element, NaN ignored; `-inf` if there is none.
pub fn reduce_max_f32(a: &[f32]) -> f32 {
    reduce_max_f32_with(hw(), a)
}

/// Smallest element, NaN ignored; `+inf` if there is none.
pub fn reduce_min_f32(a: &[f32]) -> f32 {
    reduce_min_f32_with(hw(), a)
}

/// [`reduce_max_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn reduce_max_f32_with(neon: Option<Neon>, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut m, done) = if neon.is_some() {
        unsafe { max_neon(a) }
    } else {
        (f32::NEG_INFINITY, 0)
    };
    for &x in &a[done..] {
        if x > m {
            m = x;
        }
    }
    m
}

/// [`reduce_min_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn reduce_min_f32_with(neon: Option<Neon>, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut m, done) = if neon.is_some() {
        unsafe { min_neon(a) }
    } else {
        (f32::INFINITY, 0)
    };
    for &x in &a[done..] {
        if x < m {
            m = x;
        }
    }
    m
}

// ------------------------------------------------------------------- argmax

/// First index of the largest element and its value, NaN ignored.
/// `None` if the slice is empty or all NaN.
pub fn argmax_f32(a: &[f32]) -> Option<(usize, f32)> {
    argmax_f32_with(hw(), a)
}

/// [`argmax_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn argmax_f32_with(neon: Option<Neon>, a: &[f32]) -> Option<(usize, f32)> {
    let m = reduce_max_f32_with(neon, a);
    // SAFETY: see `bf16_to_f32_with`.
    let start = match if neon.is_some() {
        unsafe { find_neon(a, m) }
    } else {
        Ok(0)
    } {
        Ok(s) => s,
        Err(i) => return Some((i, m)),
    };
    a[start..]
        .iter()
        .position(|&x| x == m)
        .map(|p| (p + start, m))
}

/// `Err(index)` when found, `Ok(first_unscanned)` otherwise.
#[target_feature(enable = "neon")]
unsafe fn find_neon(a: &[f32], v: f32) -> Result<usize, usize> {
    unsafe {
        let t = vdupq_n_f32(v);
        let mut i = 0;
        while i + 4 <= a.len() {
            let m = vceqq_f32(vld1q_f32(a.as_ptr().add(i)), t);
            if vmaxvq_u32(m) != 0 {
                let mut lanes = [0u32; 4];
                vst1q_u32(lanes.as_mut_ptr(), m);
                return Err(i + lanes.iter().position(|&l| l != 0).unwrap());
            }
            i += 4;
        }
        Ok(i)
    }
}

// ---------------------------------------------------------------------- exp

/// `out[i] = exp(a[i])`, an approximation. Same algorithm, constants and
/// contract as `x86::reduce::exp_f32`: relative error <= 3e-7 for inputs in
/// `[-87.3, 88.7]`, `0` below `-87.33655`, `+inf` above `88.72283`, NaN stays NaN.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn exp_f32(a: &[f32], out: &mut [f32]) {
    exp_f32_with(hw(), a, out)
}

/// [`exp_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn exp_f32_with(neon: Option<Neon>, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    // SAFETY: see `bf16_to_f32_with`.
    let done = if neon.is_some() {
        unsafe { exp_neon(a, out) }
    } else {
        0
    };
    crate::reduce_common::exp_f32_scalar(a, out, done);
}

#[target_feature(enable = "neon")]
unsafe fn exp_neon(a: &[f32], out: &mut [f32]) -> usize {
    unsafe {
        let mut i = 0;
        while i + 4 <= a.len() {
            let x = vld1q_f32(a.as_ptr().add(i));
            let xc = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
            let fx = vrndnq_f32(vmulq_f32(xc, vdupq_n_f32(LOG2E)));
            let mut r = vfmsq_f32(xc, fx, vdupq_n_f32(LN2_HI));
            r = vfmsq_f32(r, fx, vdupq_n_f32(LN2_LO));
            let mut y = vdupq_n_f32(P[0]);
            for &c in &P[1..] {
                y = vfmaq_f32(vdupq_n_f32(c), y, r);
            }
            let z = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), y, vmulq_f32(r, r));
            let n = vcvtnq_s32_f32(fx);
            let n1 = vshrq_n_s32::<1>(n);
            let b = vdupq_n_s32(127);
            let s1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, b)));
            let s2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vsubq_s32(n, n1), b)));
            let mut res = vmulq_f32(vmulq_f32(z, s1), s2);
            res = vbslq_f32(vcltq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(0.0), res);
            res = vbslq_f32(
                vcgtq_f32(x, vdupq_n_f32(EXP_HI)),
                vdupq_n_f32(f32::INFINITY),
                res,
            );
            // NaN lanes compare false on `x == x`.
            res = vbslq_f32(vceqq_f32(x, x), res, x);
            vst1q_f32(out.as_mut_ptr().add(i), res);
            i += 4;
        }
        i
    }
}

// ------------------------------------------------- affine u16, dot, logsumexp

/// `out[i] = a * q[i] + b` with `q[i]` an unsigned 16-bit integer. Neon
/// (`uxtl` + `ucvtf` + `fmla`, one rounding) or scalar (two roundings), so the
/// last bit can differ between the two.
///
/// # Panics
/// If `q.len() != out.len()`.
pub fn affine_u16_f32(q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    affine_u16_f32_with(hw(), q, a, b, out)
}

/// [`affine_u16_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn affine_u16_f32_with(neon: Option<Neon>, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    assert_eq!(q.len(), out.len());
    // SAFETY: see `bf16_to_f32_with`.
    let done = if neon.is_some() {
        unsafe { affine_neon(q, a, b, out) }
    } else {
        0
    };
    crate::reduce_common::affine_u16_f32_scalar(q, a, b, out, done);
}

#[target_feature(enable = "neon")]
unsafe fn affine_neon(q: &[u16], a: f32, b: f32, out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let (va, vb) = (vdupq_n_f32(a), vdupq_n_f32(b));
        while i + 8 <= q.len() {
            let v = vld1q_u16(q.as_ptr().add(i));
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v)));
            let hi = vcvtq_f32_u32(vmovl_high_u16(v));
            vst1q_f32(out.as_mut_ptr().add(i), vfmaq_f32(vb, lo, va));
            vst1q_f32(out.as_mut_ptr().add(i + 4), vfmaq_f32(vb, hi, va));
            i += 8;
        }
    }
    i
}

/// Dot product `sum(a[i] * b[i])` (four `fmla` accumulators on Neon; lane
/// order, so the last bits differ from a sequential sum). NaN and inf propagate.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_with(hw(), a, b)
}

/// [`dot_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn dot_f32_with(neon: Option<Neon>, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: see `bf16_to_f32_with`.
    let (mut s, done) = if neon.is_some() {
        unsafe { dot_neon(a, b) }
    } else {
        (0.0, 0)
    };
    s += crate::reduce_common::dot4(a, b, done);
    s
}

/// `out[i] = sum_k(signal[i + k] * taps[k])` for every `i < out.len()`: an FIR
/// filter with its taps in reverse order, or a cross-correlation. Same
/// contract as `x86::reduce::sliding_dot_f32`: vectorised across the outputs
/// (eight NEON accumulators, fused), k ascending, the scalar tail
/// (`out.len() % 4`) rounds each product. NaN and inf propagate.
///
/// # Panics
/// If `out` is not empty and `signal.len() + 1 < out.len() + taps.len()`.
pub fn sliding_dot_f32(signal: &[f32], taps: &[f32], out: &mut [f32]) {
    sliding_dot_f32_with(hw(), signal, taps, out)
}

/// [`sliding_dot_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn sliding_dot_f32_with(neon: Option<Neon>, signal: &[f32], taps: &[f32], out: &mut [f32]) {
    crate::reduce_common::sliding_check(signal, taps, out);
    // SAFETY: see `bf16_to_f32_with`; `sliding_check` keeps every window in bounds.
    let done = if neon.is_some() {
        unsafe { sliding_neon(signal, taps, out) }
    } else {
        0
    };
    crate::reduce_common::sliding_scalar(signal, taps, &mut out[done..], done);
}

/// Whole vectors of outputs; returns how many it wrote.
#[target_feature(enable = "neon")]
unsafe fn sliding_neon(signal: &[f32], taps: &[f32], out: &mut [f32]) -> usize {
    unsafe {
        let (s, n) = (signal.as_ptr(), out.len());
        let mut i = 0;
        while i + 32 <= n {
            let mut acc = [vdupq_n_f32(0.0); 8];
            for (k, &t) in taps.iter().enumerate() {
                for (r, ac) in acc.iter_mut().enumerate() {
                    *ac = vfmaq_n_f32(*ac, vld1q_f32(s.add(i + k + 4 * r)), t);
                }
            }
            for (r, ac) in acc.iter().enumerate() {
                vst1q_f32(out.as_mut_ptr().add(i + 4 * r), *ac);
            }
            i += 32;
        }
        while i + 4 <= n {
            let mut ac = vdupq_n_f32(0.0);
            for (k, &t) in taps.iter().enumerate() {
                ac = vfmaq_n_f32(ac, vld1q_f32(s.add(i + k)), t);
            }
            vst1q_f32(out.as_mut_ptr().add(i), ac);
            i += 4;
        }
        i
    }
}

/// `out[i] = (a[i] * (den - num) + b[i] * num) / den` rounded to the nearest
/// (halves up). Same contract as `x86::reduce::lerp_u8`: exact, 16 bytes a
/// loop (four `f32` vectors) on Neon.
///
/// # Panics
/// If the lengths differ, `den` is 0 or over 32767, or `num > den`.
pub fn lerp_u8(a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8]) {
    lerp_u8_with(hw(), a, b, num, den, out)
}

/// [`lerp_u8`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn lerp_u8_with(neon: Option<Neon>, a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8]) {
    crate::reduce_common::lerp_check(a, b, num, den, out);
    let w = crate::reduce_common::lerp_operands(num, den);
    // SAFETY: see `bf16_to_f32_with`; `lerp_check` makes the three lengths equal.
    let done = if neon.is_some() {
        unsafe { lerp_neon(a, b, w, out) }
    } else {
        0
    };
    crate::reduce_common::lerp_scalar(a, b, num, den, out, done);
}

/// Whole blocks of 16; returns how many bytes it wrote.
#[target_feature(enable = "neon")]
unsafe fn lerp_neon(a: &[u8], b: &[u8], [wa, wb, den, den2]: [f32; 4], out: &mut [u8]) -> usize {
    unsafe {
        let (wa, wb) = (vdupq_n_f32(wa), vdupq_n_f32(wb));
        let (den, den2) = (vdupq_n_f32(den), vdupq_n_f32(den2));
        let blend = |x: uint32x4_t, y: uint32x4_t| {
            let s = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(x), wa),
                vmulq_f32(vcvtq_f32_u32(y), wb),
            );
            vmovn_u32(vcvtq_u32_f32(vdivq_f32(
                vaddq_f32(vaddq_f32(s, s), den),
                den2,
            )))
        };
        // Eight u16 lanes of `x` and of `y`, blended.
        let half = |x: uint16x8_t, y: uint16x8_t| {
            let low = blend(vmovl_u16(vget_low_u16(x)), vmovl_u16(vget_low_u16(y)));
            let high = blend(vmovl_high_u16(x), vmovl_high_u16(y));
            vmovn_u16(vcombine_u16(low, high))
        };
        let mut i = 0;
        while i + 16 <= out.len() {
            let x = vld1q_u8(a.as_ptr().add(i));
            let y = vld1q_u8(b.as_ptr().add(i));
            let low = half(vmovl_u8(vget_low_u8(x)), vmovl_u8(vget_low_u8(y)));
            let high = half(vmovl_high_u8(x), vmovl_high_u8(y));
            vst1q_u8(out.as_mut_ptr().add(i), vcombine_u8(low, high));
            i += 16;
        }
        i
    }
}

#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                let o = i + 4 * k;
                *ac = vfmaq_f32(
                    *ac,
                    vld1q_f32(a.as_ptr().add(o)),
                    vld1q_f32(b.as_ptr().add(o)),
                );
            }
            i += 16;
        }
        while i + 4 <= a.len() {
            acc[0] = vfmaq_f32(
                acc[0],
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            i += 4;
        }
        let t = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        (vaddvq_f32(t), i)
    }
}

/// `ln(sum(exp(a[i])))` without overflow; same contract as
/// `x86::reduce::logsumexp_f32` (max ignores NaN, NaN makes the result NaN,
/// empty or all `-inf` gives `-inf`, `+inf` gives `+inf`).
pub fn logsumexp_f32(a: &[f32]) -> f32 {
    logsumexp_f32_with(hw(), a)
}

/// [`logsumexp_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn logsumexp_f32_with(neon: Option<Neon>, a: &[f32]) -> f32 {
    crate::reduce_common::logsumexp_with(
        a,
        |x| reduce_max_f32_with(neon, x),
        |x, o| exp_f32_with(neon, x, o),
        |x| reduce_sum_f32_with(neon, x),
    )
}

/// `out[i] = a[i] - logsumexp_f32(a)`.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn log_softmax_f32(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let l = logsumexp_f32(a);
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x - l;
    }
}

// ------------------------------------------------------- GEMM micro-kernel

pub use crate::reduce_common::{GEMM_MR, GEMM_NR};

/// GEMM micro-kernel, same contract as `x86::reduce::gemm_f32_kernel`:
/// `acc[i][j] += sum_k a[k*MR + i] * b[k*NR + j]`, k ascending. Neon: four
/// passes of 8 rows × 8 columns (16 q accumulators, `vfmaq`, one rounding per
/// step); the scalar tier rounds twice.
///
/// # Panics
/// If `a.len() < kc * GEMM_MR` or `b.len() < kc * GEMM_NR`.
pub fn gemm_f32_kernel(kc: usize, a: &[f32], b: &[f32], acc: &mut [[f32; GEMM_NR]; GEMM_MR]) {
    gemm_f32_kernel_with(hw(), kc, a, b, acc)
}

/// [`gemm_f32_kernel`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn gemm_f32_kernel_with(
    neon: Option<Neon>,
    kc: usize,
    a: &[f32],
    b: &[f32],
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    crate::reduce_common::gemm_check(kc, a, b);
    if neon.is_some() {
        // SAFETY: Neon detected; lengths checked above.
        unsafe { gemm_kernel_neon(kc, a.as_ptr(), b.as_ptr(), acc) }
    } else {
        crate::reduce_common::gemm_kernel_scalar(kc, a, b, acc)
    }
}

/// # Safety
/// Neon present; `a`/`b` valid for `kc*MR` / `kc*NR` floats.
#[target_feature(enable = "neon")]
unsafe fn gemm_kernel_neon(
    kc: usize,
    a: *const f32,
    b: *const f32,
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    unsafe {
        for c0 in [0usize, 8, 16, 24] {
            let mut c = [[vdupq_n_f32(0.0); 2]; GEMM_MR];
            for (ci, row) in c.iter_mut().zip(acc.iter()) {
                ci[0] = vld1q_f32(row.as_ptr().add(c0));
                ci[1] = vld1q_f32(row.as_ptr().add(c0 + 4));
            }
            for k in 0..kc {
                let bk = b.add(k * GEMM_NR + c0);
                let (b0, b1) = (vld1q_f32(bk), vld1q_f32(bk.add(4)));
                let ak = a.add(k * GEMM_MR);
                for (i, ci) in c.iter_mut().enumerate() {
                    let ai = *ak.add(i);
                    ci[0] = vfmaq_n_f32(ci[0], b0, ai);
                    ci[1] = vfmaq_n_f32(ci[1], b1, ai);
                }
            }
            for (ci, row) in c.iter().zip(acc.iter_mut()) {
                vst1q_f32(row.as_mut_ptr().add(c0), ci[0]);
                vst1q_f32(row.as_mut_ptr().add(c0 + 4), ci[1]);
            }
        }
    }
}

// ------------------------------------------------------- compress / top-k

/// Writes the indices of all `a[i] > threshold` to `out`, ascending, and
/// returns how many. NaN never matches.
///
/// # Panics
/// If `out.len() < a.len()`.
pub fn compress_gt_f32(a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    compress_gt_f32_with(hw(), a, threshold, out)
}

/// [`compress_gt_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn compress_gt_f32_with(
    neon: Option<Neon>,
    a: &[f32],
    threshold: f32,
    out: &mut [u32],
) -> usize {
    assert!(out.len() >= a.len(), "out must hold a.len() indices");
    // SAFETY: see `bf16_to_f32_with`; writes stay below `a.len() <= out.len()`.
    let (mut n, done) = if neon.is_some() {
        unsafe { compress_neon(a, threshold, out) }
    } else {
        (0, 0)
    };
    for (j, &x) in a[done..].iter().enumerate() {
        if x > threshold {
            out[n] = (done + j) as u32;
            n += 1;
        }
    }
    n
}

#[target_feature(enable = "neon")]
unsafe fn compress_neon(a: &[f32], thr: f32, out: &mut [u32]) -> (usize, usize) {
    unsafe {
        let t = vdupq_n_f32(thr);
        let (mut n, mut i) = (0usize, 0usize);
        while i + 16 <= a.len() {
            let m = [
                vcgtq_f32(vld1q_f32(a.as_ptr().add(i)), t),
                vcgtq_f32(vld1q_f32(a.as_ptr().add(i + 4)), t),
                vcgtq_f32(vld1q_f32(a.as_ptr().add(i + 8)), t),
                vcgtq_f32(vld1q_f32(a.as_ptr().add(i + 12)), t),
            ];
            if vmaxvq_u32(vorrq_u32(vorrq_u32(m[0], m[1]), vorrq_u32(m[2], m[3]))) != 0 {
                for (k, mk) in m.iter().enumerate() {
                    if vmaxvq_u32(*mk) != 0 {
                        let mut lanes = [0u32; 4];
                        vst1q_u32(lanes.as_mut_ptr(), *mk);
                        for (l, &lane) in lanes.iter().enumerate() {
                            if lane != 0 {
                                out[n] = (i + 4 * k + l) as u32;
                                n += 1;
                            }
                        }
                    }
                }
            }
            i += 16;
        }
        (n, i)
    }
}

/// The `k = idx.len()` largest elements, NaN ignored, descending, ties to the
/// lower index. Returns how many were found. See `x86::reduce::topk_f32`.
///
/// # Panics
/// If `idx.len() != val.len()`.
pub fn topk_f32(a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    topk_f32_with(hw(), a, idx, val)
}

/// [`topk_f32`] with Neon resolved once: pass `Neon::detect()` (or `None` for
/// the portable path).
pub fn topk_f32_with(neon: Option<Neon>, a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    crate::reduce_common::topk_with(a, idx, val, |c, t, b| compress_gt_f32_with(neon, c, t, b))
}

#[cfg(test)]
#[path = "test/reduce.rs"]
mod tests;
