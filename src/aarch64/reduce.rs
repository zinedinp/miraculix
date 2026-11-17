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
use crate::reduce_common::{EXP_HI, EXP_LO, LN2_HI, LN2_LO, LOG2E, P, exp_scalar};

fn hw() -> bool {
    Neon::detect().is_some()
}

// ---------------------------------------------------------------- bf16 -> f32

/// `out[i] = f32::from_bits((a[i] as u32) << 16)`: bfloat16 bits to `f32`. Exact.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    bf16_to_f32_with(hw(), a, out)
}

#[doc(hidden)]
pub fn bf16_to_f32_with(neon: bool, a: &[u16], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    // SAFETY (all `unsafe` here): `neon` is true only when Neon was detected;
    // pointers stay inside the slices.
    let done = if neon { unsafe { bf16_neon(a, out) } } else { 0 };
    for i in done..a.len() {
        out[i] = f32::from_bits((a[i] as u32) << 16);
    }
}

#[target_feature(enable = "neon")]
unsafe fn bf16_neon(a: &[u16], out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        while i + 8 <= a.len() {
            let v = vld1q_u16(a.as_ptr().add(i));
            vst1q_u32(out.as_mut_ptr().add(i) as *mut u32, vshll_n_u16::<16>(vget_low_u16(v)));
            vst1q_u32(out.as_mut_ptr().add(i + 4) as *mut u32, vshll_high_n_u16::<16>(v));
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

#[doc(hidden)]
pub fn reduce_sum_f32_with(neon: bool, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut s, done) = if neon { unsafe { sum_neon(a) } } else { (0.0, 0) };
    let mut lanes = [0f32; 4];
    for (i, &x) in a[done..].iter().enumerate() {
        lanes[i % 4] += x;
    }
    s += (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
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

#[doc(hidden)]
pub fn reduce_max_f32_with(neon: bool, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut m, done) = if neon { unsafe { max_neon(a) } } else { (f32::NEG_INFINITY, 0) };
    for &x in &a[done..] {
        if x > m {
            m = x;
        }
    }
    m
}

#[doc(hidden)]
pub fn reduce_min_f32_with(neon: bool, a: &[f32]) -> f32 {
    // SAFETY: see `bf16_to_f32_with`.
    let (mut m, done) = if neon { unsafe { min_neon(a) } } else { (f32::INFINITY, 0) };
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

#[doc(hidden)]
pub fn argmax_f32_with(neon: bool, a: &[f32]) -> Option<(usize, f32)> {
    let m = reduce_max_f32_with(neon, a);
    // SAFETY: see `bf16_to_f32_with`.
    let start = match if neon { unsafe { find_neon(a, m) } } else { Ok(0) } {
        Ok(s) => s,
        Err(i) => return Some((i, m)),
    };
    a[start..].iter().position(|&x| x == m).map(|p| (p + start, m))
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

#[doc(hidden)]
pub fn exp_f32_with(neon: bool, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    // SAFETY: see `bf16_to_f32_with`.
    let done = if neon { unsafe { exp_neon(a, out) } } else { 0 };
    for i in done..a.len() {
        out[i] = exp_scalar(a[i]);
    }
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
            res = vbslq_f32(vcgtq_f32(x, vdupq_n_f32(EXP_HI)), vdupq_n_f32(f32::INFINITY), res);
            // NaN lanes compare false on `x == x`.
            res = vbslq_f32(vceqq_f32(x, x), res, x);
            vst1q_f32(out.as_mut_ptr().add(i), res);
            i += 4;
        }
        i
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

#[doc(hidden)]
pub fn compress_gt_f32_with(neon: bool, a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    assert!(out.len() >= a.len(), "out must hold a.len() indices");
    // SAFETY: see `bf16_to_f32_with`; writes stay below `a.len() <= out.len()`.
    let (mut n, done) = if neon { unsafe { compress_neon(a, threshold, out) } } else { (0, 0) };
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

#[doc(hidden)]
pub fn topk_f32_with(neon: bool, a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    crate::reduce_common::topk_with(a, idx, val, |c, t, b| compress_gt_f32_with(neon, c, t, b))
}

#[cfg(test)]
#[path = "test/reduce.rs"]
mod tests;
