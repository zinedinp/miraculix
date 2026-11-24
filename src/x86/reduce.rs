//! # Slice reductions and searches (`f32`)
//!
//! Whole-slice operations that return a value or an index: the shape of
//! sampling over a logit vector. Like [`super::auto_up`], each function picks
//! AVX-512F, AVX2, AVX or SSE2 from [`super::detect_features`] (cached) and falls
//! back to portable code (only 32-bit x86 without SSE2 gets there). Also available as [`super::auto`]-style free
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

/// A dispatch decision for the `*_with` functions, resolved once.
///
/// A `Tier` is a **proof**: the only ways to get one are [`best_tier`] and
/// [`available_tiers`], which hand out only tiers this CPU can run. That
/// makes every `*_with(tier, ..)` call safe, and lets a hot loop skip the
/// per-call feature check:
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
/// use miraculix::x86::reduce;
/// let tier = reduce::best_tier(); // once, at startup
/// let rows = [[1.0f32; 64]; 4];
/// for r in &rows {
///     let _ = reduce::dot_f32_with(tier, r, r); // no detection here
/// }
/// # }
/// ```
///
/// It cannot be built by hand, so it cannot claim a tier the CPU lacks:
///
/// ```compile_fail
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
/// let forged = miraculix::x86::reduce::Tier(Default::default());
/// # }
/// # #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
/// # compile_error!("x86 only");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier(TierKind);

/// The rung behind a [`Tier`]. Private, so a `Tier` cannot be forged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TierKind {
    Scalar,
    /// Every x86_64: 128-bit, no FMA.
    Sse2,
    /// AVX without AVX2 (Sandy Bridge, Ivy Bridge, Bulldozer): 256-bit
    /// floats, 128-bit integers, no FMA.
    Avx,
    Avx2,
    Avx2Fma,
    Avx512,
}

/// The fastest [`Tier`] this CPU can run (from the cached
/// [`super::detect_features`]).
pub fn best_tier() -> Tier {
    tier_of(detect_features())
}

fn tier_of(f: FeatureSet) -> Tier {
    Tier(if f.contains(Feature::Avx512f) {
        TierKind::Avx512
    } else if f.contains(Feature::Avx2) && f.contains(Feature::Fma) {
        TierKind::Avx2Fma
    } else if f.contains(Feature::Avx2) {
        TierKind::Avx2
    } else if f.contains(Feature::Avx) {
        TierKind::Avx
    } else if f.contains(Feature::Sse2) {
        TierKind::Sse2
    } else {
        TierKind::Scalar
    })
}

/// Every [`Tier`] this CPU can run, slowest first (`None` where the CPU
/// lacks it). For tests and benchmarks that compare rungs.
pub fn available_tiers() -> [Option<Tier>; 6] {
    let f = detect_features();
    [
        Some(TierKind::Scalar),
        f.contains(Feature::Sse2).then_some(TierKind::Sse2),
        f.contains(Feature::Avx).then_some(TierKind::Avx),
        f.contains(Feature::Avx2).then_some(TierKind::Avx2),
        (f.contains(Feature::Avx2) && f.contains(Feature::Fma)).then_some(TierKind::Avx2Fma),
        f.contains(Feature::Avx512f).then_some(TierKind::Avx512),
    ]
    .map(|k| k.map(Tier))
}

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------- bf16 -> f32

/// `out[i] = f32::from_bits((a[i] as u32) << 16)`: bfloat16 bit patterns to `f32`.
/// Exact. AVX-512F / AVX2 (`vpmovzxwd` + `vpslld`), SSE2 (`punpck*wd` with zero) or scalar.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn bf16_to_f32(a: &[u16], out: &mut [f32]) {
    bf16_to_f32_with(best_tier(), a, out)
}

/// [`bf16_to_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn bf16_to_f32_with(tier: Tier, a: &[u16], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let done = match tier.0 {
        // SAFETY (all `unsafe` below): a `Tier` only comes from `best_tier` /
        // `available_tiers`, which hand out tiers the CPU has (private
        // field, so it cannot be forged), and slice bounds are respected.
        TierKind::Avx512 => unsafe { bf16_avx512(a, out) },
        TierKind::Avx2 | TierKind::Avx2Fma => unsafe { bf16_avx2(a, out) },
        TierKind::Avx | TierKind::Sse2 => unsafe { bf16_sse2(a, out) },
        TierKind::Scalar => 0,
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
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i) as *mut _,
                _mm512_slli_epi32::<16>(v),
            );
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
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_slli_epi32::<16>(v),
            );
            i += 8;
        }
    }
    i
}

#[target_feature(enable = "sse2")]
unsafe fn bf16_sse2(a: &[u16], out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let zero = _mm_setzero_si128();
        while i + 8 <= a.len() {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            // Zero below each half word: the half word shifted up by 16.
            let o = out.as_mut_ptr().add(i) as *mut __m128i;
            _mm_storeu_si128(o, _mm_unpacklo_epi16(zero, v));
            _mm_storeu_si128(o.add(1), _mm_unpackhi_epi16(zero, v));
            i += 8;
        }
    }
    i
}

// ------------------------------------------------------------ sum / max / min

/// Horizontal sum of four lanes.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hsum_sse2(t: __m128) -> f32 {
    let h = _mm_add_ps(t, _mm_movehl_ps(t, t));
    _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps::<1>(h, h)))
}

/// Sum of all elements. Order is tier-dependent (see module docs); NaN propagates.
pub fn reduce_sum_f32(a: &[f32]) -> f32 {
    reduce_sum_f32_with(best_tier(), a)
}

/// [`reduce_sum_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn reduce_sum_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut s, done) = match tier.0 {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        TierKind::Avx512 => unsafe { sum_avx512(a) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Avx2Fma => unsafe { sum_avx(a) },
        TierKind::Sse2 => unsafe { sum_sse2(a) },
        TierKind::Scalar => (0.0, 0),
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

#[target_feature(enable = "avx")]
unsafe fn sum_avx(a: &[f32]) -> (f32, usize) {
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

#[target_feature(enable = "sse2")]
unsafe fn sum_sse2(a: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [_mm_setzero_ps(); 4];
        let mut i = 0;
        while i + 16 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                *ac = _mm_add_ps(*ac, _mm_loadu_ps(a.as_ptr().add(i + 4 * k)));
            }
            i += 16;
        }
        while i + 4 <= a.len() {
            acc[0] = _mm_add_ps(acc[0], _mm_loadu_ps(a.as_ptr().add(i)));
            i += 4;
        }
        let t = _mm_add_ps(_mm_add_ps(acc[0], acc[1]), _mm_add_ps(acc[2], acc[3]));
        (hsum_sse2(t), i)
    }
}

macro_rules! ext_kernels {
    ($avx512:ident, $avx:ident, $sse2:ident, $op512:ident, $op256:ident, $op128:ident, $red512:ident, $init:expr) => {
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

        #[target_feature(enable = "avx")]
        unsafe fn $avx(a: &[f32]) -> (f32, usize) {
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

        #[target_feature(enable = "sse2")]
        unsafe fn $sse2(a: &[f32]) -> (f32, usize) {
            unsafe {
                let mut acc = [_mm_set1_ps($init); 4];
                let mut i = 0;
                while i + 16 <= a.len() {
                    for (k, ac) in acc.iter_mut().enumerate() {
                        *ac = $op128(_mm_loadu_ps(a.as_ptr().add(i + 4 * k)), *ac);
                    }
                    i += 16;
                }
                while i + 4 <= a.len() {
                    acc[0] = $op128(_mm_loadu_ps(a.as_ptr().add(i)), acc[0]);
                    i += 4;
                }
                let h = $op128($op128(acc[0], acc[1]), $op128(acc[2], acc[3]));
                let h = $op128(h, _mm_movehl_ps(h, h));
                let h = $op128(h, _mm_shuffle_ps::<1>(h, h));
                (_mm_cvtss_f32(h), i)
            }
        }
    };
}
ext_kernels!(
    max_avx512,
    max_avx,
    max_sse2,
    _mm512_max_ps,
    _mm256_max_ps,
    _mm_max_ps,
    _mm512_reduce_max_ps,
    f32::NEG_INFINITY
);
ext_kernels!(
    min_avx512,
    min_avx,
    min_sse2,
    _mm512_min_ps,
    _mm256_min_ps,
    _mm_min_ps,
    _mm512_reduce_min_ps,
    f32::INFINITY
);

/// Largest element, ignoring NaN. `-inf` if there is none.
pub fn reduce_max_f32(a: &[f32]) -> f32 {
    reduce_max_f32_with(best_tier(), a)
}

/// Smallest element, ignoring NaN. `+inf` if there is none.
pub fn reduce_min_f32(a: &[f32]) -> f32 {
    reduce_min_f32_with(best_tier(), a)
}

/// [`reduce_max_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn reduce_max_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut m, done) = match tier.0 {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        TierKind::Avx512 => unsafe { max_avx512(a) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Avx2Fma => unsafe { max_avx(a) },
        TierKind::Sse2 => unsafe { max_sse2(a) },
        TierKind::Scalar => (f32::NEG_INFINITY, 0),
    };
    for &x in &a[done..] {
        if x > m {
            m = x;
        }
    }
    m
}

/// [`reduce_min_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn reduce_min_f32_with(tier: Tier, a: &[f32]) -> f32 {
    let (mut m, done) = match tier.0 {
        // SAFETY: as above.
        TierKind::Avx512 => unsafe { min_avx512(a) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Avx2Fma => unsafe { min_avx(a) },
        TierKind::Sse2 => unsafe { min_sse2(a) },
        TierKind::Scalar => (f32::INFINITY, 0),
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

/// [`argmax_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn argmax_f32_with(tier: Tier, a: &[f32]) -> Option<(usize, f32)> {
    let m = reduce_max_f32_with(tier, a);
    find_first_eq(tier, a, m).map(|i| (i, m))
}

fn find_first_eq(tier: Tier, a: &[f32], v: f32) -> Option<usize> {
    let done = match tier.0 {
        // SAFETY: as above.
        TierKind::Avx512 => unsafe { find_avx512(a, v) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Avx2Fma => unsafe { find_avx(a, v) },
        TierKind::Sse2 => unsafe { find_sse2(a, v) },
        TierKind::Scalar => Ok(0),
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

#[target_feature(enable = "avx")]
unsafe fn find_avx(a: &[f32], v: f32) -> Result<usize, usize> {
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

#[target_feature(enable = "sse2")]
unsafe fn find_sse2(a: &[f32], v: f32) -> Result<usize, usize> {
    unsafe {
        let t = _mm_set1_ps(v);
        let mut i = 0;
        while i + 4 <= a.len() {
            let m = _mm_movemask_ps(_mm_cmpeq_ps(_mm_loadu_ps(a.as_ptr().add(i)), t)) as u32;
            if m != 0 {
                return Err(i + m.trailing_zeros() as usize);
            }
            i += 4;
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
/// polynomial, so results can differ from the scalar tier in the last bit;
/// SSE2, which also serves AVX and AVX2 without FMA, rounds `x * log2(e)` by the
/// MXCSR mode, to nearest unless it was changed).
/// Inputs below `-87.33655` return `0` (the subnormal range is flushed), above
/// `88.72283` return `+inf`, NaN returns NaN. Not bit-exact with `libm`.
///
/// # Panics
/// If `a.len() != out.len()`.
pub fn exp_f32(a: &[f32], out: &mut [f32]) {
    exp_f32_with(best_tier(), a, out)
}

use crate::reduce_common::{EXP_HI, EXP_LO, LN2_HI, LN2_LO, LOG2E, P, exp_scalar};

/// [`exp_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn exp_f32_with(tier: Tier, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let done = match tier.0 {
        // SAFETY: as above (`Avx2Fma` requires FMA, which `available_tiers` checks).
        TierKind::Avx512 => unsafe { exp_avx512(a, out) },
        TierKind::Avx2Fma => unsafe { exp_avx2_fma(a, out) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Sse2 => unsafe { exp_sse2(a, out) },
        TierKind::Scalar => 0,
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
            let xc = _mm512_min_ps(
                _mm512_max_ps(x, _mm512_set1_ps(EXP_LO)),
                _mm512_set1_ps(EXP_HI),
            );
            let fx = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm512_mul_ps(xc, _mm512_set1_ps(LOG2E)),
            );
            let mut r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_HI), xc);
            r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_LO), r);
            let mut y = _mm512_set1_ps(P[0]);
            for &c in &P[1..] {
                y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(c));
            }
            let z = _mm512_fmadd_ps(
                y,
                _mm512_mul_ps(r, r),
                _mm512_add_ps(r, _mm512_set1_ps(1.0)),
            );
            let n = _mm512_cvtps_epi32(fx);
            let n1 = _mm512_srai_epi32::<1>(n);
            let b = _mm512_set1_epi32(127);
            let s1 = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(n1, b)));
            let s2 = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
                _mm512_sub_epi32(n, n1),
                b,
            )));
            let mut res = _mm512_mul_ps(_mm512_mul_ps(z, s1), s2);
            res = _mm512_mask_mov_ps(
                res,
                _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(EXP_LO)),
                _mm512_setzero_ps(),
            );
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
            let xc = _mm256_min_ps(
                _mm256_max_ps(x, _mm256_set1_ps(EXP_LO)),
                _mm256_set1_ps(EXP_HI),
            );
            let fx = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm256_mul_ps(xc, _mm256_set1_ps(LOG2E)),
            );
            let mut r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), xc);
            r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), r);
            let mut y = _mm256_set1_ps(P[0]);
            for &c in &P[1..] {
                y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(c));
            }
            let z = _mm256_fmadd_ps(
                y,
                _mm256_mul_ps(r, r),
                _mm256_add_ps(r, _mm256_set1_ps(1.0)),
            );
            let n = _mm256_cvtps_epi32(fx);
            let n1 = _mm256_srai_epi32::<1>(n);
            let b = _mm256_set1_epi32(127);
            let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, b)));
            let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                _mm256_sub_epi32(n, n1),
                b,
            )));
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

#[target_feature(enable = "sse2")]
unsafe fn exp_sse2(a: &[f32], out: &mut [f32]) -> usize {
    unsafe {
        let select = |mask: __m128, yes: __m128, no: __m128| {
            _mm_or_ps(_mm_and_ps(mask, yes), _mm_andnot_ps(mask, no))
        };
        let mut i = 0;
        while i + 4 <= a.len() {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let xc = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(EXP_LO)), _mm_set1_ps(EXP_HI));
            // No `roundps` before SSE4.1: rounded through an integer (`|xc * log2 e| < 128`).
            let n = _mm_cvtps_epi32(_mm_mul_ps(xc, _mm_set1_ps(LOG2E)));
            let fx = _mm_cvtepi32_ps(n);
            let mut r = _mm_sub_ps(xc, _mm_mul_ps(fx, _mm_set1_ps(LN2_HI)));
            r = _mm_sub_ps(r, _mm_mul_ps(fx, _mm_set1_ps(LN2_LO)));
            let mut y = _mm_set1_ps(P[0]);
            for &c in &P[1..] {
                y = _mm_add_ps(_mm_mul_ps(y, r), _mm_set1_ps(c));
            }
            let z = _mm_add_ps(
                _mm_mul_ps(y, _mm_mul_ps(r, r)),
                _mm_add_ps(r, _mm_set1_ps(1.0)),
            );
            let n1 = _mm_srai_epi32::<1>(n);
            let b = _mm_set1_epi32(127);
            let s1 = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n1, b)));
            let s2 = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(_mm_sub_epi32(n, n1), b)));
            let mut res = _mm_mul_ps(_mm_mul_ps(z, s1), s2);
            res = _mm_andnot_ps(_mm_cmplt_ps(x, _mm_set1_ps(EXP_LO)), res);
            res = select(
                _mm_cmpgt_ps(x, _mm_set1_ps(EXP_HI)),
                _mm_set1_ps(f32::INFINITY),
                res,
            );
            res = select(_mm_cmpunord_ps(x, x), x, res);
            _mm_storeu_ps(out.as_mut_ptr().add(i), res);
            i += 4;
        }
        i
    }
}

// affine u16, dot, logsumexp

/// `out[i] = a * q[i] + b`, with `q[i]` read as an unsigned 16-bit integer
/// (dequantising a `u16` log-prob or scale table). AVX-512F / AVX2
/// (`vpmovzxwd`, `vcvtdq2ps`, multiply-add), SSE2 (`punpck*wd`, `cvtdq2ps`) or scalar. The FMA tiers round once, the
/// others twice, so results can differ in the last bit between tiers.
///
/// # Panics
/// If `q.len() != out.len()`.
pub fn affine_u16_f32(q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    affine_u16_f32_with(best_tier(), q, a, b, out)
}

/// [`affine_u16_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn affine_u16_f32_with(tier: Tier, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
    assert_eq!(q.len(), out.len());
    let done = match tier.0 {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        TierKind::Avx512 => unsafe { affine_avx512(q, a, b, out) },
        TierKind::Avx2Fma => unsafe { affine_avx2_fma(q, a, b, out) },
        TierKind::Avx2 => unsafe { affine_avx2(q, a, b, out) },
        TierKind::Avx | TierKind::Sse2 => unsafe { affine_sse2(q, a, b, out) },
        TierKind::Scalar => 0,
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
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), va, vb),
            );
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

#[target_feature(enable = "sse2")]
unsafe fn affine_sse2(q: &[u16], a: f32, b: f32, out: &mut [f32]) -> usize {
    let mut i = 0;
    unsafe {
        let (va, vb, zero) = (_mm_set1_ps(a), _mm_set1_ps(b), _mm_setzero_si128());
        while i + 8 <= q.len() {
            let v = _mm_loadu_si128(q.as_ptr().add(i) as *const __m128i);
            for (half, wide) in [_mm_unpacklo_epi16(v, zero), _mm_unpackhi_epi16(v, zero)]
                .into_iter()
                .enumerate()
            {
                let r = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(wide), va), vb);
                _mm_storeu_ps(out.as_mut_ptr().add(i + 4 * half), r);
            }
            i += 8;
        }
    }
    i
}

/// Dot product `sum(a[i] * b[i])`. Four vector accumulators of fused
/// multiply-adds on AVX-512F and AVX2+FMA (AVX, AVX2 alone and SSE2 multiply then add);
/// the order depends on the tier, so the last bits differ between CPUs, and
/// the scalar tail and the scalar tier round the product separately. NaN and
/// inf propagate.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_with(best_tier(), a, b)
}

/// [`dot_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn dot_f32_with(tier: Tier, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (mut s, done) = match tier.0 {
        // SAFETY: tier chosen from detection; see `bf16_to_f32_with`.
        TierKind::Avx512 => unsafe { dot_avx512(a, b) },
        TierKind::Avx2Fma => unsafe { dot_avx2_fma(a, b) },
        TierKind::Avx | TierKind::Avx2 => unsafe { dot_avx(a, b) },
        TierKind::Sse2 => unsafe { dot_sse2(a, b) },
        TierKind::Scalar => (0.0, 0),
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
                *ac = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.as_ptr().add(o)),
                    _mm512_loadu_ps(b.as_ptr().add(o)),
                    *ac,
                );
            }
            i += 64;
        }
        while i + 16 <= a.len() {
            acc[0] = _mm512_fmadd_ps(
                _mm512_loadu_ps(a.as_ptr().add(i)),
                _mm512_loadu_ps(b.as_ptr().add(i)),
                acc[0],
            );
            i += 16;
        }
        let t = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        (_mm512_reduce_add_ps(t), i)
    }
}

macro_rules! dot_ymm_kernel {
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
                        *ac = mac(
                            _mm256_loadu_ps(a.as_ptr().add(o)),
                            _mm256_loadu_ps(b.as_ptr().add(o)),
                            *ac,
                        );
                    }
                    i += 32;
                }
                while i + 8 <= a.len() {
                    acc[0] = mac(
                        _mm256_loadu_ps(a.as_ptr().add(i)),
                        _mm256_loadu_ps(b.as_ptr().add(i)),
                        acc[0],
                    );
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
dot_ymm_kernel!(dot_avx2_fma, "avx2,fma", |x, y, z| _mm256_fmadd_ps(x, y, z));
dot_ymm_kernel!(dot_avx, "avx", |x, y, z| _mm256_add_ps(
    _mm256_mul_ps(x, y),
    z
));

#[target_feature(enable = "sse2")]
unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> (f32, usize) {
    unsafe {
        let mut acc = [_mm_setzero_ps(); 4];
        let mut i = 0;
        while i + 16 <= a.len() {
            for (k, ac) in acc.iter_mut().enumerate() {
                let o = i + 4 * k;
                *ac = _mm_add_ps(
                    _mm_mul_ps(
                        _mm_loadu_ps(a.as_ptr().add(o)),
                        _mm_loadu_ps(b.as_ptr().add(o)),
                    ),
                    *ac,
                );
            }
            i += 16;
        }
        while i + 4 <= a.len() {
            acc[0] = _mm_add_ps(
                _mm_mul_ps(
                    _mm_loadu_ps(a.as_ptr().add(i)),
                    _mm_loadu_ps(b.as_ptr().add(i)),
                ),
                acc[0],
            );
            i += 4;
        }
        let t = _mm_add_ps(_mm_add_ps(acc[0], acc[1]), _mm_add_ps(acc[2], acc[3]));
        (hsum_sse2(t), i)
    }
}

// ------------------------------------------------------------- sliding dot

/// `out[i] = sum_k(signal[i + k] * taps[k])` for every `i < out.len()`: an FIR
/// filter with its taps in reverse order (the newest sample's tap last), or a
/// cross-correlation. One dispatch for the whole slice, vectorised across the
/// outputs (eight vector accumulators, each tap broadcast once per pass), so
/// short filters run as fast as long ones. Every output adds its products in
/// ascending `k`; AVX-512F and AVX2+FMA fuse them, AVX, AVX2 alone, SSE2, the
/// scalar tier and the scalar tail (`out.len() % lanes`) round each product,
/// so the last bits differ between tiers. NaN and inf propagate.
///
/// # Panics
/// If `out` is not empty and `signal.len() + 1 < out.len() + taps.len()`.
pub fn sliding_dot_f32(signal: &[f32], taps: &[f32], out: &mut [f32]) {
    sliding_dot_f32_with(best_tier(), signal, taps, out)
}

/// [`sliding_dot_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn sliding_dot_f32_with(tier: Tier, signal: &[f32], taps: &[f32], out: &mut [f32]) {
    crate::reduce_common::sliding_check(signal, taps, out);
    let done = match tier.0 {
        // SAFETY: tier chosen from detection; `sliding_check` keeps every
        // window `signal[i..i + taps.len()]` with `i < out.len()` in bounds.
        TierKind::Avx512 => unsafe { sliding_avx512(signal, taps, out) },
        TierKind::Avx2Fma => unsafe { sliding_avx2_fma(signal, taps, out) },
        TierKind::Avx | TierKind::Avx2 => unsafe { sliding_avx(signal, taps, out) },
        TierKind::Sse2 => unsafe { sliding_sse2(signal, taps, out) },
        TierKind::Scalar => 0,
    };
    crate::reduce_common::sliding_scalar(signal, taps, &mut out[done..], done);
}

macro_rules! sliding_kernel {
    ($name:ident, $feat:literal, $w:literal, $zero:ident, $set1:ident, $load:ident, $store:ident, $mac:expr) => {
        /// Whole vectors of outputs; returns how many it wrote.
        #[target_feature(enable = $feat)]
        unsafe fn $name(signal: &[f32], taps: &[f32], out: &mut [f32]) -> usize {
            unsafe {
                let mac = $mac;
                let (s, n) = (signal.as_ptr(), out.len());
                let mut i = 0;
                while i + 8 * $w <= n {
                    let mut acc = [$zero(); 8];
                    for (k, &t) in taps.iter().enumerate() {
                        let t = $set1(t);
                        for (r, ac) in acc.iter_mut().enumerate() {
                            *ac = mac(t, $load(s.add(i + k + r * $w)), *ac);
                        }
                    }
                    for (r, ac) in acc.iter().enumerate() {
                        $store(out.as_mut_ptr().add(i + r * $w), *ac);
                    }
                    i += 8 * $w;
                }
                while i + $w <= n {
                    let mut ac = $zero();
                    for (k, &t) in taps.iter().enumerate() {
                        ac = mac($set1(t), $load(s.add(i + k)), ac);
                    }
                    $store(out.as_mut_ptr().add(i), ac);
                    i += $w;
                }
                i
            }
        }
    };
}
sliding_kernel!(
    sliding_avx512,
    "avx512f",
    16,
    _mm512_setzero_ps,
    _mm512_set1_ps,
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    |x, y, z| _mm512_fmadd_ps(x, y, z)
);
sliding_kernel!(
    sliding_avx2_fma,
    "avx2,fma",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    |x, y, z| _mm256_fmadd_ps(x, y, z)
);
sliding_kernel!(
    sliding_avx,
    "avx",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    |x, y, z| _mm256_add_ps(_mm256_mul_ps(x, y), z)
);
sliding_kernel!(
    sliding_sse2,
    "sse2",
    4,
    _mm_setzero_ps,
    _mm_set1_ps,
    _mm_loadu_ps,
    _mm_storeu_ps,
    |x, y, z| _mm_add_ps(_mm_mul_ps(x, y), z)
);

// -------------------------------------------------------------- byte blend

/// `out[i] = (a[i] * (den - num) + b[i] * num) / den` rounded to the nearest
/// (halves up): `a` blended into `b` by `num / den`, as a GPU's `mix` writes a
/// UNORM8 target. Exact on every tier (the lanes work in `f32`, where every
/// step is exact or provably rounds the same). AVX-512F 64 bytes a loop
/// (16 lanes), AVX2 32 (8 lanes), SSE2 and AVX 16 (4 lanes).
///
/// # Panics
/// If the lengths differ, `den` is 0 or over 32767, or `num > den`.
pub fn lerp_u8(a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8]) {
    lerp_u8_with(best_tier(), a, b, num, den, out)
}

/// [`lerp_u8`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn lerp_u8_with(tier: Tier, a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8]) {
    crate::reduce_common::lerp_check(a, b, num, den, out);
    let w = crate::reduce_common::lerp_operands(num, den);
    let done = match tier.0 {
        // SAFETY: tier chosen from detection; `lerp_check` makes the three lengths equal.
        TierKind::Avx512 => unsafe { lerp_avx512(a, b, w, out) },
        TierKind::Avx2 | TierKind::Avx2Fma => unsafe { lerp_avx2(a, b, w, out) },
        TierKind::Avx | TierKind::Sse2 => unsafe { lerp_sse2(a, b, w, out) },
        TierKind::Scalar => 0,
    };
    crate::reduce_common::lerp_scalar(a, b, num, den, out, done);
}

/// Whole blocks of 64; returns how many bytes it wrote.
#[target_feature(enable = "avx512f")]
unsafe fn lerp_avx512(a: &[u8], b: &[u8], [wa, wb, den, den2]: [f32; 4], out: &mut [u8]) -> usize {
    unsafe {
        let (wa, wb) = (_mm512_set1_ps(wa), _mm512_set1_ps(wb));
        let (den, den2) = (_mm512_set1_ps(den), _mm512_set1_ps(den2));
        let widen =
            |p: *const u8| _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(p.cast())));
        let mut i = 0;
        while i + 64 <= out.len() {
            for g in (i..i + 64).step_by(16) {
                let s = _mm512_add_ps(
                    _mm512_mul_ps(widen(a.as_ptr().add(g)), wa),
                    _mm512_mul_ps(widen(b.as_ptr().add(g)), wb),
                );
                let q = _mm512_div_ps(_mm512_add_ps(_mm512_add_ps(s, s), den), den2);
                let bytes = _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(q));
                _mm_storeu_si128(out.as_mut_ptr().add(g).cast(), bytes);
            }
            i += 64;
        }
        i
    }
}

/// Whole blocks of 32; returns how many bytes it wrote.
#[target_feature(enable = "avx2")]
unsafe fn lerp_avx2(a: &[u8], b: &[u8], [wa, wb, den, den2]: [f32; 4], out: &mut [u8]) -> usize {
    unsafe {
        let (wa, wb) = (_mm256_set1_ps(wa), _mm256_set1_ps(wb));
        let (den, den2) = (_mm256_set1_ps(den), _mm256_set1_ps(den2));
        let widen =
            |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(p.cast())));
        let blend = |g: usize| {
            let s = _mm256_add_ps(
                _mm256_mul_ps(widen(a.as_ptr().add(g)), wa),
                _mm256_mul_ps(widen(b.as_ptr().add(g)), wb),
            );
            _mm256_cvttps_epi32(_mm256_div_ps(_mm256_add_ps(_mm256_add_ps(s, s), den), den2))
        };
        // `packs`/`packus` work per 128-bit half: dwords 0-7 come out as
        // r0 lo, r1 lo, r2 lo, r3 lo, r0 hi, r1 hi, r2 hi, r3 hi.
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let mut i = 0;
        while i + 32 <= out.len() {
            let low = _mm256_packs_epi32(blend(i), blend(i + 8));
            let high = _mm256_packs_epi32(blend(i + 16), blend(i + 24));
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(low, high), order);
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), bytes);
            i += 32;
        }
        i
    }
}

/// Whole blocks of 16; returns how many bytes it wrote.
#[target_feature(enable = "sse2")]
unsafe fn lerp_sse2(a: &[u8], b: &[u8], [wa, wb, den, den2]: [f32; 4], out: &mut [u8]) -> usize {
    unsafe {
        let (wa, wb) = (_mm_set1_ps(wa), _mm_set1_ps(wb));
        let (den, den2) = (_mm_set1_ps(den), _mm_set1_ps(den2));
        let zero = _mm_setzero_si128();
        let blend = |x: __m128i, y: __m128i| {
            let s = _mm_add_ps(
                _mm_mul_ps(_mm_cvtepi32_ps(x), wa),
                _mm_mul_ps(_mm_cvtepi32_ps(y), wb),
            );
            _mm_cvttps_epi32(_mm_div_ps(_mm_add_ps(_mm_add_ps(s, s), den), den2))
        };
        // Four dwords of `x` (u16 lanes) and of `y`, blended.
        let quarter = |x: __m128i, y: __m128i, high: bool| {
            if high {
                blend(_mm_unpackhi_epi16(x, zero), _mm_unpackhi_epi16(y, zero))
            } else {
                blend(_mm_unpacklo_epi16(x, zero), _mm_unpacklo_epi16(y, zero))
            }
        };
        let mut i = 0;
        while i + 16 <= out.len() {
            let x = _mm_loadu_si128(a.as_ptr().add(i).cast());
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let (xl, xh) = (_mm_unpacklo_epi8(x, zero), _mm_unpackhi_epi8(x, zero));
            let (yl, yh) = (_mm_unpacklo_epi8(y, zero), _mm_unpackhi_epi8(y, zero));
            let low = _mm_packs_epi32(quarter(xl, yl, false), quarter(xl, yl, true));
            let high = _mm_packs_epi32(quarter(xh, yh, false), quarter(xh, yh, true));
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_packus_epi16(low, high));
            i += 16;
        }
        i
    }
}

/// `ln(sum(exp(a[i])))` without overflow (shifted by the maximum), built from
/// [`reduce_max_f32`], [`exp_f32`] (error <= 3e-7) and [`reduce_sum_f32`]
/// (blocks of 1024 added in `f64`; the final `ln` is exact to `f32`).
/// NaN elements are ignored by the maximum but make the result NaN; an empty
/// slice or all `-inf` gives `-inf`; a `+inf` element gives `+inf`.
pub fn logsumexp_f32(a: &[f32]) -> f32 {
    logsumexp_f32_with(best_tier(), a)
}

/// [`logsumexp_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
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

/// [`log_softmax_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn log_softmax_f32_with(tier: Tier, a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let l = logsumexp_f32_with(tier, a);
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x - l;
    }
}

// ------------------------------------------------------- GEMM micro-kernel

pub use crate::reduce_common::{GEMM_MR, GEMM_NR};

/// GEMM micro-kernel: `acc[i][j] += sum_k a[k*MR + i] * b[k*NR + j]` for
/// `k in 0..kc` – one `GEMM_MR × GEMM_NR` (8 × 32) block of `C = A·B`, with A
/// and B packed into panels (A: `kc` groups of `MR` row values, B: `kc` groups
/// of `NR` column values). The caller packs, blocks and threads; this keeps
/// the block in registers (AVX-512F: 16 zmm accumulators; AVX/AVX2: four passes of
/// 4 rows × 16 columns; SSE2: eight passes of 4 rows × 8 columns). k runs in
/// ascending order on every tier; the FMA tiers round once per step, AVX, AVX2
/// without FMA, SSE2 and the scalar tier twice, so the last bits differ between
/// tiers. NaN and inf propagate.
///
/// # Panics
/// If `a.len() < kc * GEMM_MR` or `b.len() < kc * GEMM_NR`.
pub fn gemm_f32_kernel(kc: usize, a: &[f32], b: &[f32], acc: &mut [[f32; GEMM_NR]; GEMM_MR]) {
    gemm_f32_kernel_with(best_tier(), kc, a, b, acc)
}

/// [`gemm_f32_kernel`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn gemm_f32_kernel_with(
    tier: Tier,
    kc: usize,
    a: &[f32],
    b: &[f32],
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    crate::reduce_common::gemm_check(kc, a, b);
    match tier.0 {
        // SAFETY: tier chosen from detection; lengths checked above.
        TierKind::Avx512 => unsafe { gemm_kernel_avx512(kc, a.as_ptr(), b.as_ptr(), acc) },
        TierKind::Avx2Fma => unsafe { gemm_kernel_avx2_fma(kc, a.as_ptr(), b.as_ptr(), acc) },
        TierKind::Avx | TierKind::Avx2 => unsafe {
            gemm_kernel_avx(kc, a.as_ptr(), b.as_ptr(), acc)
        },
        TierKind::Sse2 => unsafe { gemm_kernel_sse2(kc, a.as_ptr(), b.as_ptr(), acc) },
        TierKind::Scalar => crate::reduce_common::gemm_kernel_scalar(kc, a, b, acc),
    }
}

/// # Safety
/// AVX-512F present; `a`/`b` valid for `kc*MR` / `kc*NR` floats.
#[target_feature(enable = "avx512f")]
unsafe fn gemm_kernel_avx512(
    kc: usize,
    a: *const f32,
    b: *const f32,
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    unsafe {
        let mut c = [[_mm512_setzero_ps(); 2]; GEMM_MR];
        for (ci, row) in c.iter_mut().zip(acc.iter()) {
            ci[0] = _mm512_loadu_ps(row.as_ptr());
            ci[1] = _mm512_loadu_ps(row.as_ptr().add(16));
        }
        for k in 0..kc {
            let b0 = _mm512_loadu_ps(b.add(k * GEMM_NR));
            let b1 = _mm512_loadu_ps(b.add(k * GEMM_NR + 16));
            let ak = a.add(k * GEMM_MR);
            for (i, ci) in c.iter_mut().enumerate() {
                let ai = _mm512_set1_ps(*ak.add(i));
                ci[0] = _mm512_fmadd_ps(ai, b0, ci[0]);
                ci[1] = _mm512_fmadd_ps(ai, b1, ci[1]);
            }
        }
        for (ci, row) in c.iter().zip(acc.iter_mut()) {
            _mm512_storeu_ps(row.as_mut_ptr(), ci[0]);
            _mm512_storeu_ps(row.as_mut_ptr().add(16), ci[1]);
        }
    }
}

macro_rules! gemm_ymm_kernel {
    ($name:ident, $feat:literal, $mac:expr) => {
        /// # Safety
        /// Features present; `a`/`b` valid for `kc*MR` / `kc*NR` floats.
        #[target_feature(enable = $feat)]
        unsafe fn $name(
            kc: usize,
            a: *const f32,
            b: *const f32,
            acc: &mut [[f32; GEMM_NR]; GEMM_MR],
        ) {
            unsafe {
                let mac = $mac;
                // 4 passes (row half × column half) of 4 rows × 2 ymm = 8 accumulators.
                for r0 in [0usize, 4] {
                    for c0 in [0usize, 16] {
                        let mut c = [[_mm256_setzero_ps(); 2]; 4];
                        for (i, ci) in c.iter_mut().enumerate() {
                            let row = acc[r0 + i].as_ptr().add(c0);
                            ci[0] = _mm256_loadu_ps(row);
                            ci[1] = _mm256_loadu_ps(row.add(8));
                        }
                        for k in 0..kc {
                            let bk = b.add(k * GEMM_NR + c0);
                            let (b0, b1) = (_mm256_loadu_ps(bk), _mm256_loadu_ps(bk.add(8)));
                            let ak = a.add(k * GEMM_MR + r0);
                            for (i, ci) in c.iter_mut().enumerate() {
                                let ai = _mm256_set1_ps(*ak.add(i));
                                ci[0] = mac(ai, b0, ci[0]);
                                ci[1] = mac(ai, b1, ci[1]);
                            }
                        }
                        for (i, ci) in c.iter().enumerate() {
                            let row = acc[r0 + i].as_mut_ptr().add(c0);
                            _mm256_storeu_ps(row, ci[0]);
                            _mm256_storeu_ps(row.add(8), ci[1]);
                        }
                    }
                }
            }
        }
    };
}
gemm_ymm_kernel!(gemm_kernel_avx2_fma, "avx2,fma", |x, y, z| _mm256_fmadd_ps(
    x, y, z
));
gemm_ymm_kernel!(gemm_kernel_avx, "avx", |x, y, z| _mm256_add_ps(
    _mm256_mul_ps(x, y),
    z
));

/// # Safety
/// SSE2 present; `a`/`b` valid for `kc*MR` / `kc*NR` floats.
#[target_feature(enable = "sse2")]
unsafe fn gemm_kernel_sse2(
    kc: usize,
    a: *const f32,
    b: *const f32,
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    unsafe {
        // 8 passes (row half × column quarter) of 4 rows × 2 xmm = 8 accumulators.
        for r0 in [0usize, 4] {
            for c0 in [0usize, 8, 16, 24] {
                let mut c = [[_mm_setzero_ps(); 2]; 4];
                for (i, ci) in c.iter_mut().enumerate() {
                    let row = acc[r0 + i].as_ptr().add(c0);
                    ci[0] = _mm_loadu_ps(row);
                    ci[1] = _mm_loadu_ps(row.add(4));
                }
                for k in 0..kc {
                    let bk = b.add(k * GEMM_NR + c0);
                    let (b0, b1) = (_mm_loadu_ps(bk), _mm_loadu_ps(bk.add(4)));
                    let ak = a.add(k * GEMM_MR + r0);
                    for (i, ci) in c.iter_mut().enumerate() {
                        let ai = _mm_set1_ps(*ak.add(i));
                        ci[0] = _mm_add_ps(_mm_mul_ps(ai, b0), ci[0]);
                        ci[1] = _mm_add_ps(_mm_mul_ps(ai, b1), ci[1]);
                    }
                }
                for (i, ci) in c.iter().enumerate() {
                    let row = acc[r0 + i].as_mut_ptr().add(c0);
                    _mm_storeu_ps(row, ci[0]);
                    _mm_storeu_ps(row.add(4), ci[1]);
                }
            }
        }
    }
}

// ------------------------------------------------------- compress / top-k

/// Writes the indices of all `a[i] > threshold` to `out`, ascending, and
/// returns how many there are. NaN never matches.
///
/// On AVX-512 this is `vcompressps` on an index vector (register form, fast
/// on AMD too); AVX and SSE2 use `movemask`; otherwise scalar.
///
/// # Panics
/// If `out.len() < a.len()` (the worst case is every element matching).
pub fn compress_gt_f32(a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    compress_gt_f32_with(best_tier(), a, threshold, out)
}

/// [`compress_gt_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn compress_gt_f32_with(tier: Tier, a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
    assert!(out.len() >= a.len(), "out must hold a.len() indices");
    let (mut n, done) = match tier.0 {
        // SAFETY: as above. The kernels write at most `done` indices plus 16 slack lanes
        // that stay below `a.len() <= out.len()` (see `compress_avx512`).
        TierKind::Avx512 => unsafe { compress_avx512(a, threshold, out) },
        TierKind::Avx | TierKind::Avx2 | TierKind::Avx2Fma => unsafe {
            compress_avx(a, threshold, out)
        },
        TierKind::Sse2 => unsafe { compress_sse2(a, threshold, out) },
        TierKind::Scalar => (0, 0),
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
            _mm512_storeu_si512(
                out.as_mut_ptr().add(n) as *mut _,
                _mm512_maskz_compress_epi32(m, idx),
            );
            n += m.count_ones() as usize;
            i += 16;
        }
        (n, i)
    }
}

#[target_feature(enable = "avx")]
unsafe fn compress_avx(a: &[f32], thr: f32, out: &mut [u32]) -> (usize, usize) {
    unsafe {
        let t = _mm256_set1_ps(thr);
        let (mut n, mut i) = (0usize, 0usize);
        while i + 8 <= a.len() {
            let mut m = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                t,
            )) as u32;
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

#[target_feature(enable = "sse2")]
unsafe fn compress_sse2(a: &[f32], thr: f32, out: &mut [u32]) -> (usize, usize) {
    unsafe {
        let t = _mm_set1_ps(thr);
        let (mut n, mut i) = (0usize, 0usize);
        while i + 4 <= a.len() {
            let mut m = _mm_movemask_ps(_mm_cmpgt_ps(_mm_loadu_ps(a.as_ptr().add(i)), t)) as u32;
            while m != 0 {
                out[n] = (i + m.trailing_zeros() as usize) as u32;
                n += 1;
                m &= m - 1;
            }
            i += 4;
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

/// [`topk_f32`] with a [`Tier`] resolved once (see [`best_tier`]).
pub fn topk_f32_with(tier: Tier, a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
    crate::reduce_common::topk_with(a, idx, val, |chunk, thr, buf| {
        compress_gt_f32_with(tier, chunk, thr, buf)
    })
}

#[cfg(test)]
#[path = "test/reduce.rs"]
mod tests;
