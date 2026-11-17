//! # Slice reductions and searches (`f32`)
//!
//! Whole-slice operations that return a value or an index: the shape of
//! sampling over a logit vector. Like [`super::auto_up`], each function picks
//! AVX-512F, AVX2 or scalar from [`super::detect_features`] (cached) and falls
//! back to portable code. Also available as [`super::auto`]-style free
//! functions; names are prefixed `reduce_` where [`super::auto_up`] already
//! has an elementwise op of the same name.
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::reduce;
//! let logits = [0.5f32, -1.0, 7.25, 3.0, f32::NAN];
//! assert_eq!(reduce::reduce_max_f32(&logits), 7.25);
//! assert_eq!(reduce::argmax_f32(&logits), Some((2, 7.25)));
//! let mut idx = [0u32; 2];
//! let mut val = [0f32; 2];
//! assert_eq!(reduce::topk_f32(&logits, &mut idx, &mut val), 2);
//! assert_eq!((idx, val), ([2, 3], [7.25, 3.0]));
//! # }
//! ```
//!
//! ## Contract
//!
//! - **NaN is ignored** by `reduce_max_f32`, `reduce_min_f32`, `argmax_f32`,
//!   `topk_f32` and `compress_gt_f32` (it never compares greater). `reduce_sum_f32`
//!   propagates it.
//! - `reduce_max_f32` of an empty or all-NaN slice is `-inf`, `reduce_min_f32`
//!   gives `+inf`.
//! - **Ties** pick the lowest index (`argmax_f32`, `topk_f32`).
//! - `reduce_sum_f32` adds in a tier-dependent order (several vector
//!   accumulators), so its last bits differ between CPUs. The relative
//!   error is far below that of one sequential `f32` accumulator.
//! - [`exp_f32`] is an approximation, bound documented there.

use super::{Feature, FeatureSet, detect_features};

/// Execution tier. Crate-visible so tests can run every rung the CPU has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum Tier {
    Scalar,
    Avx2,
    Avx2Fma,
    Avx512,
}

#[doc(hidden)]
pub fn best_tier() -> Tier {
    tier_of(detect_features())
}

fn tier_of(f: FeatureSet) -> Tier {
    if f.contains(Feature::Avx512f) {
        Tier::Avx512
    } else if f.contains(Feature::Avx2) && f.contains(Feature::Fma) {
        Tier::Avx2Fma
    } else if f.contains(Feature::Avx2) {
        Tier::Avx2
    } else {
        Tier::Scalar
    }
}

/// Tiers the running CPU can execute, best last. For tests and benchmarks.
#[doc(hidden)]
pub fn available_tiers() -> [Option<Tier>; 4] {
    let f = detect_features();
    [
        Some(Tier::Scalar),
        f.contains(Feature::Avx2).then_some(Tier::Avx2),
        (f.contains(Feature::Avx2) && f.contains(Feature::Fma)).then_some(Tier::Avx2Fma),
        f.contains(Feature::Avx512f).then_some(Tier::Avx512),
    ]
}

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------- bf16 -> f32

/// `out[i] = f32::from_bits((a[i] as u32) << 16)`: bfloat16 bit patterns to `f32`.
/// Exact. AVX-512F / AVX2 (`vpmovzxwd` + `vpslld`) or scalar.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    bf16_to_f32_with(best_tier(), a, out)
}

#[doc(hidden)]
pub fn bf16_to_f32_with(tier: Tier, a: &[u16], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let done = match tier {
        // SAFETY (all `unsafe` below): the caller picked a tier the CPU has
        // (`best_tier` / `available_tiers`), and slice bounds are respected.
        Tier::Avx512 => unsafe { bf16_avx512(a, out) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { bf16_avx2(a, out) },
        Tier::Scalar => 0,
    };
    for i in done..a.len() {
        out[i] = f32::from_bits((a[i] as u32) << 16);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn bf16_avx512(a: &[u16], out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        while i + 16 <= a.len() {
            let v = _mm512_cvtepu16_epi32(_mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i));
            _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut _, _mm512_slli_epi32::<16>(v));
            i += 16;
        }
    }
    i
}

#[target_feature(enable = "avx2")]
unsafe fn bf16_avx2(a: &[u16], out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        while i + 8 <= a.len() {
            let v = _mm256_cvtepu16_epi32(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, _mm256_slli_epi32::<16>(v));
            i += 8;
        }
    }
    i
}

// ------------------------------------------------------------ sum / max / min

/// Sum of all elements. Order is tier-dependent (see module docs); NaN propagates.
pub fn reduce_sum_f32(a: &[f32]) -> f32 {
    reduce_sum_f32_with(best_tier(), a)
}

#[doc(hidden)]
pub fn reduce_sum_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut s, done) = match tier {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        Tier::Avx512 => unsafe { sum_avx512(a) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { sum_avx2(a) },
        Tier::Scalar => (0.0, 0),
    };
    let mut lanes = [0f32; 4];
    for (i, &x) in a[done..].iter().enumerate() {
        lanes[i % 4] += x;
    }
    s += (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    s
}

#[target_feature(enable = "avx512f")]
unsafe fn sum_avx512(a: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [_mm512_setzero_ps(); 4];
        let mut i = 0;
        while i + 64 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                *ac = _mm512_add_ps(*ac, _mm512_loadu_ps(a.as_ptr().add(i + 16 * k)));
            }
            i += 64;
        }
        while i + 16 <= a.len() {
            acc[0] = _mm512_add_ps(acc[0], _mm512_loadu_ps(a.as_ptr().add(i)));
            i += 16;
        }
        let t = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        (_mm512_reduce_add_ps(t), i)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(a: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                *ac = _mm256_add_ps(*ac, _mm256_loadu_ps(a.as_ptr().add(i + 8 * k)));
            }
            i += 32;
        }
        while i + 8 <= a.len() {
            acc[0] = _mm256_add_ps(acc[0], _mm256_loadu_ps(a.as_ptr().add(i)));
            i += 8;
        }
        let t = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let h = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps::<1>(t));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        let h = _mm_add_ss(h, _mm_shuffle_ps::<1>(h, h));
        (_mm_cvtss_f32(h), i)
    }
}

macro_rules! ext_kernels {
    ($avx512:ident, $avx2:ident, $op512:ident, $op256:ident, $op128:ident, $red512:ident, $init:expr) => {
        #[target_feature(enable = "avx512f")]
        unsafe fn $avx512(a: &[f32]) -> (f32, usize) {
            unsafe {
                // `op(x, acc)`: x86 returns the second operand on NaN, so NaN lanes keep `acc`.
                let mut acc = [_mm512_set1_ps($init); 4];
                let mut i = 0;
                while i + 64 <= a.len() {
                    for (k, ac) in acc.iter_mut().enumerate() {
                        *ac = $op512(_mm512_loadu_ps(a.as_ptr().add(i + 16 * k)), *ac);
                    }
                    i += 64;
                }
                while i + 16 <= a.len() {
                    acc[0] = $op512(_mm512_loadu_ps(a.as_ptr().add(i)), acc[0]);
                    i += 16;
                }
                let t = $op512($op512(acc[0], acc[1]), $op512(acc[2], acc[3]));
                ($red512(t), i)
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $avx2(a: &[f32]) -> (f32, usize) {
            unsafe {
                let mut acc = [_mm256_set1_ps($init); 4];
                let mut i = 0;
                while i + 32 <= a.len() {
                    for (k, ac) in acc.iter_mut().enumerate() {
                        *ac = $op256(_mm256_loadu_ps(a.as_ptr().add(i + 8 * k)), *ac);
                    }
                    i += 32;
                }
                while i + 8 <= a.len() {
                    acc[0] = $op256(_mm256_loadu_ps(a.as_ptr().add(i)), acc[0]);
                    i += 8;
                }
                let t = $op256($op256(acc[0], acc[1]), $op256(acc[2], acc[3]));
                let h = $op128(_mm256_castps256_ps128(t), _mm256_extractf128_ps::<1>(t));
                let h = $op128(h, _mm_movehl_ps(h, h));
                let h = $op128(h, _mm_shuffle_ps::<1>(h, h));
                (_mm_cvtss_f32(h), i)
            }
        }
    };
}
ext_kernels!(max_avx512, max_avx2, _mm512_max_ps, _mm256_max_ps, _mm_max_ps, _mm512_reduce_max_ps, f32::NEG_INFINITY);
ext_kernels!(min_avx512, min_avx2, _mm512_min_ps, _mm256_min_ps, _mm_min_ps, _mm512_reduce_min_ps, f32::INFINITY);

/// Largest element, ignoring NaN. `-inf` if there is none.
pub fn reduce_max_f32(a: &[f32]) -> f32 {
    reduce_max_f32_with(best_tier(), a)
}

/// Smallest element, ignoring NaN. `+inf` if there is none.
pub fn reduce_min_f32(a: &[f32]) -> f32 {
    reduce_min_f32_with(best_tier(), a)
}

#[doc(hidden)]
pub fn reduce_max_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut m, done) = match tier {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        Tier::Avx512 => unsafe { max_avx512(a) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { max_avx2(a) },
        Tier::Scalar => (f32::NEG_INFINITY, 0),
    };
    for &x in &a[done..] {
        if x > m {
            m = x;
        }
    }
    m
}

#[doc(hidden)]
pub fn reduce_min_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut m, done) = match tier {
        // SAFETY: as above.
        Tier::Avx512 => unsafe { min_avx512(a) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { min_avx2(a) },
        Tier::Scalar => (f32::INFINITY, 0),
    };
    for &x in &a[done..] {
        if x < m {
            m = x;
        }
    }
    m
}

// ------------------------------------------------------------------- argmax

/// First index of the largest element and its value, ignoring NaN.
/// `None` if the slice is empty or all NaN.
pub fn argmax_f32(a: &[f32]) -> Option<(usize, f32)> {
    argmax_f32_with(best_tier(), a)
}

#[doc(hidden)]
pub fn argmax_f32_with(tier: Tier, a: &[f32]) -> Option<(usize, f32)> {
    let m = reduce_max_f32_with(tier, a);
    find_first_eq(tier, a, m).map(|i| (i, m))
}

fn find_first_eq(tier: Tier, a: &[f32], v: f32) -> Option<usize> {
    let done = match tier {
        // SAFETY: as above.
        Tier::Avx512 => unsafe { find_avx512(a, v) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { find_avx2(a, v) },
        Tier::Scalar => Ok(0),
    };
    match done {
        Ok(start) => a[start..].iter().position(|&x| x == v).map(|p| p + start),
        Err(i) => Some(i),
    }
}

/// `Err(index)` when found, `Ok(first_unscanned)` otherwise.
#[target_feature(enable = "avx512f")]
unsafe fn find_avx512(a: &[f32], v: f32) -> Result<usize, usize> {
    unsafe {
        let t = _mm512_set1_ps(v);
        let mut i = 0;
        while i + 16 <= a.len() {
            let m = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(_mm512_loadu_ps(a.as_ptr().add(i)), t);
            if m != 0 {
                return Err(i + m.trailing_zeros() as usize);
            }
            i += 16;
        }
        Ok(i)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn find_avx2(a: &[f32], v: f32) -> Result<usize, usize> {
    unsafe {
        let t = _mm256_set1_ps(v);
        let mut i = 0;
        while i + 8 <= a.len() {
            let c = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_loadu_ps(a.as_ptr().add(i)), t);
            let m = _mm256_movemask_ps(c) as u32;
            if m != 0 {
                return Err(i + m.trailing_zeros() as usize);
            }
            i += 8;
        }
        Ok(i)
    }
}

// ---------------------------------------------------------------------- exp

/// `out[i] = exp(a[i])`, an approximation.
///
/// Cephes-style: range reduction by `ln 2` (split constant), degree-6
/// polynomial, scaling by a power of two. **Relative error <= 3e-7** (about
/// 2.5 ulp) for inputs in `[-87.3, 88.7]`, on every tier (FMA tiers fuse the
/// polynomial, so results can differ from the scalar tier in the last bit).
/// Inputs below `-87.33655` return `0` (the subnormal range is flushed), above
/// `88.72283` return `+inf`, NaN returns NaN. Not bit-exact with `libm`.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn exp_f32(a: &[f32], out: &mut [f32]) {
    exp_f32_with(best_tier(), a, out)
}

use crate::reduce_common::{EXP_HI, EXP_LO, LN2_HI, LN2_LO, LOG2E, P, exp_scalar};

#[doc(hidden)]
pub fn exp_f32_with(tier: Tier, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let done = match tier {
        // SAFETY: as above (`Avx2Fma` requires FMA, which `available_tiers` checks).
        Tier::Avx512 => unsafe { exp_avx512(a, out) },
        Tier::Avx2Fma => unsafe { exp_avx2_fma(a, out) },
        Tier::Avx2 | Tier::Scalar => 0,
    };
    for i in done..a.len() {
        out[i] = exp_scalar(a[i]);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn exp_avx512(a: &[f32], out: &mut [f32]) -> usize {
    unsafe {
        let mut i = 0;
        while i + 16 <= a.len() {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let xc = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(EXP_LO)), _mm512_set1_ps(EXP_HI));
            let fx = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm512_mul_ps(xc, _mm512_set1_ps(LOG2E)),
            );
            let mut r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_HI), xc);
            r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_LO), r);
            let mut y = _mm512_set1_ps(P[0]);
            for &c in &P[1..] {
                y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(c));
            }
            let z = _mm512_fmadd_ps(y, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0)));
            let n = _mm512_cvtps_epi32(fx);
            let n1 = _mm512_srai_epi32::<1>(n);
            let b = _mm512_set1_epi32(127);
            let s1 = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(n1, b)));
            let s2 = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_sub_epi32(n, n1), b)));
            let mut res = _mm512_mul_ps(_mm512_mul_ps(z, s1), s2);
            res = _mm512_mask_mov_ps(res, _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(EXP_LO)), _mm512_setzero_ps());
            res = _mm512_mask_mov_ps(
                res,
                _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, _mm512_set1_ps(EXP_HI)),
                _mm512_set1_ps(f32::INFINITY),
            );
            res = _mm512_mask_mov_ps(res, _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(x, x), x);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), res);
            i += 16;
        }
        i
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2_fma(a: &[f32], out: &mut [f32]) -> usize {
    unsafe {
        let mut i = 0;
        while i + 8 <= a.len() {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(EXP_LO)), _mm256_set1_ps(EXP_HI));
            let fx = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
                xc,
                _mm256_set1_ps(LOG2E),
            ));
            let mut r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), xc);
            r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), r);
            let mut y = _mm256_set1_ps(P[0]);
            for &c in &P[1..] {
                y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(c));
            }
            let z = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0)));
            let n = _mm256_cvtps_epi32(fx);
            let n1 = _mm256_srai_epi32::<1>(n);
            let b = _mm256_set1_epi32(127);
            let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, b)));
            let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_sub_epi32(n, n1), b)));
            let mut res = _mm256_mul_ps(_mm256_mul_ps(z, s1), s2);
            res = _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(EXP_LO)), res);
            let hi = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(EXP_HI));
            res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::INFINITY), hi);
            res = _mm256_blendv_ps(res, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
            i += 8;
        }
        i
    }
}


// affine u16, dot, logsumexp

/// `out[i] = a * q[i] + b`, with `q[i]` read as an unsigned 16-bit integer
/// (dequantising a `u16` log-prob or scale table). AVX-512F / AVX2 (`vpmovzxwd`
/// + `vcvtdq2ps` + multiply-add) or scalar. The FMA tiers round once, the
/// others twice, so results can differ in the last bit between tiers.
///
/// # Panics
/// If `q.len() != out.len()`.
pub fn affine_u16_f32(q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    affine_u16_f32_with(best_tier(), q, a, b, out)
}

#[doc(hidden)]
pub fn affine_u16_f32_with(tier: Tier, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    assert_eq!(q.len(), out.len());
    let done = match tier {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        Tier::Avx512 => unsafe { affine_avx512(q, a, b, out) },
        Tier::Avx2Fma => unsafe { affine_avx2_fma(q, a, b, out) },
        Tier::Avx2 => unsafe { affine_avx2(q, a, b, out) },
        Tier::Scalar => 0,
    };
    for i in done..q.len() {
        out[i] = a * (q[i] as f32) + b;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn affine_avx512(q: &[u16], a: f32, b: f32, out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let (va, vb) = (_mm512_set1_ps(a), _mm512_set1_ps(b));
        while i + 16 <= q.len() {
            let v = _mm512_cvtepu16_epi32(_mm256_loadu_si256(q.as_ptr().add(i) as *const __m256i));
            let r = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), r);
            i += 16;
        }
    }
    i
}

#[target_feature(enable = "avx2,fma")]
unsafe fn affine_avx2_fma(q: &[u16], a: f32, b: f32, out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let (va, vb) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
        while i + 8 <= q.len() {
            let v = _mm256_cvtepu16_epi32(_mm_loadu_si128(q.as_ptr().add(i) as *const __m128i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), va, vb));
            i += 8;
        }
    }
    i
}

#[target_feature(enable = "avx2")]
unsafe fn affine_avx2(q: &[u16], a: f32, b: f32, out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let (va, vb) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
        while i + 8 <= q.len() {
            let v = _mm256_cvtepu16_epi32(_mm_loadu_si128(q.as_ptr().add(i) as *const __m128i));
            let r = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v), va), vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
            i += 8;
        }
    }
    i
}

/// Dot product `sum(a[i] * b[i])`. Four vector accumulators of fused
/// multiply-adds on AVX-512F and AVX2+FMA (AVX2 alone multiplies then adds);
/// the order depends on the tier, so the last bits differ between CPUs, and
/// the scalar tail and the scalar tier round the product separately. NaN and
/// inf propagate.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_with(best_tier(), a, b)
}

#[doc(hidden)]
pub fn dot_f32_with(tier: Tier, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (mut s, done) = match tier {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        Tier::Avx512 => unsafe { dot_avx512(a, b) },
        Tier::Avx2Fma => unsafe { dot_avx2_fma(a, b) },
        Tier::Avx2 => unsafe { dot_avx2(a, b) },
        Tier::Scalar => (0.0, 0),
    };
    let mut lanes = [0f32; 4];
    for i in done..a.len() {
        lanes[i % 4] += a[i] * b[i];
    }
    s += (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    s
}

#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [_mm512_setzero_ps(); 4];
        let mut i = 0;
        while i + 64 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                let o = i + 16 * k;
                *ac = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(o)), _mm512_loadu_ps(b.as_ptr().add(o)), *ac);
            }
            i += 64;
        }
        while i + 16 <= a.len() {
            acc[0] = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), acc[0]);
            i += 16;
        }
        let t = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        (_mm512_reduce_add_ps(t), i)
    }
}

macro_rules! dot_avx2_kernel {
    ($name:ident, $feat:literal, $mac:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name(a: &[f32], b: &[f32]) -> (f32, usize) {
            unsafe {
                let mac = $mac;
                let mut acc = [_mm256_setzero_ps(); 4];
                let mut i = 0;
                while i + 32 <= a.len() {
                    for (k, ac) in acc.iter_mut().enumerate() {
                        let o = i + 8 * k;
                        *ac = mac(_mm256_loadu_ps(a.as_ptr().add(o)), _mm256_loadu_ps(b.as_ptr().add(o)), *ac);
                    }
                    i += 32;
                }
                while i + 8 <= a.len() {
                    acc[0] = mac(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc[0]);
                    i += 8;
                }
                let t = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
                let h = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps::<1>(t));
                let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
                let h = _mm_add_ss(h, _mm_shuffle_ps::<1>(h, h));
                (_mm_cvtss_f32(h), i)
            }
        }
    };
}
dot_avx2_kernel!(dot_avx2_fma, "avx2,fma", |x, y, z| _mm256_fmadd_ps(x, y, z));
dot_avx2_kernel!(dot_avx2, "avx2", |x, y, z| _mm256_add_ps(_mm256_mul_ps(x, y), z));

/// `ln(sum(exp(a[i])))` without overflow (shifted by the maximum), built from
/// [`reduce_max_f32`], [`exp_f32`] (error <= 3e-7) and [`reduce_sum_f32`]
/// (blocks of 1024 added in `f64`; the final `ln` is exact to `f32`).
/// NaN elements are ignored by the maximum but make the result NaN; an empty
/// slice or all `-inf` gives `-inf`; a `+inf` element gives `+inf`.
pub fn logsumexp_f32(a: &[f32]) -> f32 {
    logsumexp_f32_with(best_tier(), a)
}

#[doc(hidden)]
pub fn logsumexp_f32_with(tier: Tier, a: &[f32]) -> f32 {
    crate::reduce_common::logsumexp_with(
        a,
        |x| reduce_max_f32_with(tier, x),
        |x, o| exp_f32_with(tier, x, o),
        |x| reduce_sum_f32_with(tier, x),
    )
}

/// `out[i] = a[i] - logsumexp_f32(a)`: log-softmax over the whole slice.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn log_softmax_f32(a: &[f32], out: &mut [f32]) {
    log_softmax_f32_with(best_tier(), a, out)
}

#[doc(hidden)]
pub fn log_softmax_f32_with(tier: Tier, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let l = logsumexp_f32_with(tier, a);
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x - l;
    }
}

// ------------------------------------------------------- compress / top-k

/// Writes the indices of all `a[i] > threshold` to `out`, ascending, and
/// returns how many there are. NaN never matches.
///
/// On AVX-512 this is `vcompressps` on an index vector (register form, fast
/// on AMD too); AVX2 uses `movemask`; otherwise scalar.
///
/// # Panics
/// If `out.len() < a.len()` (the worst case is every element matching).
pub fn compress_gt_f32(a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    compress_gt_f32_with(best_tier(), a, threshold, out)
}

#[doc(hidden)]
pub fn compress_gt_f32_with(tier: Tier, a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    assert!(out.len() >= a.len(), "out must hold a.len() indices");
    let (mut n, done) = match tier {
        // SAFETY: as above. The kernels write at most `done` indices plus 16 slack lanes
        // that stay below `a.len() <= out.len()` (see `compress_avx512`).
        Tier::Avx512 => unsafe { compress_avx512(a, threshold, out) },
        Tier::Avx2 | Tier::Avx2Fma => unsafe { compress_avx2(a, threshold, out) },
        Tier::Scalar => (0, 0),
    };
    for (j, &x) in a[done..].iter().enumerate() {
        if x > threshold {
            out[n] = (done + j) as u32;
            n += 1;
        }
    }
    n
}

#[target_feature(enable = "avx512f")]
unsafe fn compress_avx512(a: &[f32], thr: f32, out: &mut [u32]) -> (usize, usize) {
    unsafe {
        let t = _mm512_set1_ps(thr);
        let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let (mut n, mut i) = (0usize, 0usize);
        // Writing 16 lanes at out[n..] is in bounds: n <= i and i + 16 <= a.len() <= out.len().
        while i + 16 <= a.len() {
            let m = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_loadu_ps(a.as_ptr().add(i)), t);
            let idx = _mm512_add_epi32(iota, _mm512_set1_epi32(i as i32));
            _mm512_storeu_si512(out.as_mut_ptr().add(n) as *mut _, _mm512_maskz_compress_epi32(m, idx));
            n += m.count_ones() as usize;
            i += 16;
        }
        (n, i)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn compress_avx2(a: &[f32], thr: f32, out: &mut [u32]) -> (usize, usize) {
    unsafe {
        let t = _mm256_set1_ps(thr);
        let (mut n, mut i) = (0usize, 0usize);
        while i + 8 <= a.len() {
            let mut m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_loadu_ps(a.as_ptr().add(i)), t)) as u32;
            while m != 0 {
                out[n] = (i + m.trailing_zeros() as usize) as u32;
                n += 1;
                m &= m - 1;
            }
            i += 8;
        }
        (n, i)
    }
}

/// The `k = idx.len()` largest elements, NaN ignored, sorted by value
/// descending (ties: lower index first). Writes their indices to `idx` and
/// values to `val`, and returns how many were found (`< k` only if the slice
/// has fewer non-NaN elements).
///
/// After the first `k` candidates, whole blocks are skipped with
/// [`compress_gt_f32`] against the current `k`-th value, so a large slice
/// costs close to one pass of vector compares. No allocation.
///
/// # Panics
/// If `idx.len() != val.len()`.
pub fn topk_f32(a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    topk_f32_with(best_tier(), a, idx, val)
}

#[doc(hidden)]
pub fn topk_f32_with(tier: Tier, a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    crate::reduce_common::topk_with(a, idx, val, |chunk, thr, buf| compress_gt_f32_with(tier, chunk, thr, buf))
}

#[cfg(test)]
#[path = "test/reduce.rs"]
mod tests;
