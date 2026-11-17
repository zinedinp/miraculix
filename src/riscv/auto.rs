//! # Auto slice ops (RISC-V)
//!
//! Same contract and names as `x86::reduce` / `x86::auto_up`: each function
//! uses RVV when the CPU has `V` (cached [`super::detect_features`]) and
//! portable code otherwise. NaN is ignored by max/min/argmax/top-k/compress,
//! ties pick the lowest index, and mismatched lengths panic.
//!
//! The `*_f32` reductions add in an order that depends on VLEN, so sums can
//! differ in the last bits between CPUs (and from the scalar path).

use super::detect_features;
use super::ops::rvv::Rvv;
use super::ops::zvfbf::Zvfbfmin;

fn rvv() -> Option<Rvv> {
    Rvv::from_features(detect_features())
}

/// `out[i] = a[i] + b[i]`.
///
/// # Panics
/// If the lengths differ.
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    if let Some(t) = rvv() {
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
    if let Some(t) = rvv() {
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
    if let Some(t) = rvv() {
        return t.mul_f32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// `out[i] = a[i] * b[i] + c[i]`. Fused (one rounding) with RVV; the portable
/// path rounds twice (`core` has no fast fused multiply-add without hardware).
///
/// # Panics
/// If the lengths differ.
pub fn fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    if let Some(t) = rvv() {
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
    if let Some(t) = rvv() {
        return t.add_i32(a, b, out);
    }
    assert!(a.len() == b.len() && b.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i].wrapping_add(b[i]);
    }
}

/// Sum of all elements (order is implementation-defined). NaN propagates.
pub fn reduce_sum_f32(a: &[f32]) -> f32 {
    if let Some(t) = rvv() {
        return t.reduce_sum_f32(a);
    }
    let mut lanes = [0f32; 4];
    for (i, &x) in a.iter().enumerate() {
        lanes[i % 4] += x;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Largest element, NaN ignored; `-inf` if there is none.
pub fn reduce_max_f32(a: &[f32]) -> f32 {
    if let Some(t) = rvv() {
        return t.reduce_max_f32(a);
    }
    a.iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m })
}

/// Smallest element, NaN ignored; `+inf` if there is none.
pub fn reduce_min_f32(a: &[f32]) -> f32 {
    if let Some(t) = rvv() {
        return t.reduce_min_f32(a);
    }
    a.iter().fold(f32::INFINITY, |m, &x| if x < m { x } else { m })
}

/// First index of the largest element and its value, NaN ignored.
/// `None` if the slice is empty or all NaN.
pub fn argmax_f32(a: &[f32]) -> Option<(usize, f32)> {
    if let Some(t) = rvv() {
        return t.argmax_f32(a);
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in a.iter().enumerate() {
        if x > best.map_or(f32::NEG_INFINITY, |b| b.1) || (best.is_none() && x == f32::NEG_INFINITY) {
            best = Some((i, x));
        }
    }
    best
}

/// Indices of all `a[i] > threshold`, ascending; returns the count. NaN never matches.
///
/// # Panics
/// If `out.len() < a.len()`.
pub fn compress_gt_f32(a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    if let Some(t) = rvv() {
        return t.compress_gt_f32(a, threshold, out);
    }
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
    crate::topk::topk_with(a, idx, val, compress_gt_f32)
}

/// Dot product of two `i8` slices into `i32`, wrapping.
///
/// # Panics
/// If the lengths differ.
pub fn dot_i8_i32(a: &[i8], b: &[i8]) -> i32 {
    if let Some(t) = rvv() {
        return t.dot_i8_i32(a, b);
    }
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32))
}

/// bfloat16 bits to `f32`, exact. Uses `Zvfbfmin` when present.
///
/// # Panics
/// If the lengths differ.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    if let Some(t) = Zvfbfmin::from_features(detect_features()) {
        return t.cvt_bf16_to_f32_slice(a, out);
    }
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = f32::from_bits((a[i] as u32) << 16);
    }
}
