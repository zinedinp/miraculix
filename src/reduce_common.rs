//! Arch-neutral pieces shared by the per-architecture slice reductions
//! (`x86::reduce`, `aarch64::reduce`, `riscv::auto`): the top-k driver and
//! the scalar `exp` with its constants.
//!
//! Top-k driver. Each architecture supplies a `compress` kernel
//! (indices of elements greater than a threshold) and this does the rest.

/// Inserts `x` (at index `i`) into the sorted prefix `val[..*n]`. Ranks after
/// everything `>= x` already stored (earlier index wins ties); drops it if the
/// list is full and `x` is not larger than the last entry.
fn insert(idx: &mut [u32], val: &mut [f32], n: &mut usize, i: usize, x: f32) {
    let k = val.len();
    let mut p = if *n < k {
        *n += 1;
        *n - 1
    } else if x > val[k - 1] {
        k - 1
    } else {
        return;
    };
    while p > 0 && val[p - 1] < x {
        val[p] = val[p - 1];
        idx[p] = idx[p - 1];
        p -= 1;
    }
    val[p] = x;
    idx[p] = i as u32;
}

/// Top `idx.len()` elements of `a` by value (NaN ignored), descending, ties to
/// the lower index. Returns the count found. `compress(chunk, threshold, buf)`
/// must write the ascending indices of `chunk[i] > threshold` into `buf`
/// (`buf.len() >= chunk.len()`) and return how many.
pub(crate) fn topk_with(
    a: &[f32],
    idx: &mut [u32],
    val: &mut [f32],
    mut compress: impl FnMut(&[f32], f32, &mut [u32]) -> usize,
) -> usize {
    assert_eq!(idx.len(), val.len());
    let k = idx.len();
    if k == 0 {
        return 0;
    }
    let (mut n, mut pos) = (0usize, 0usize);
    while pos < a.len() && n < k {
        if !a[pos].is_nan() {
            insert(idx, val, &mut n, pos, a[pos]);
        }
        pos += 1;
    }
    let mut buf = [0u32; 256];
    while pos < a.len() {
        let end = (pos + 256).min(a.len());
        // The k-th value only grows, so a stale threshold just admits extra candidates.
        let c = compress(&a[pos..end], val[k - 1], &mut buf);
        for &j in &buf[..c] {
            let i = pos + j as usize;
            insert(idx, val, &mut n, i, a[i]);
        }
        pos = end;
    }
    n
}

// ---- exp (Cephes-style, see `x86::reduce::exp_f32` for the contract) ----

pub(crate) const EXP_HI: f32 = 88.722_83;
pub(crate) const EXP_LO: f32 = -87.336_55;
pub(crate) const LOG2E: f32 = core::f32::consts::LOG2_E;
pub(crate) const LN2_HI: f32 = 0.693_359_4;
pub(crate) const LN2_LO: f32 = -2.121_944_4e-4;
pub(crate) const P: [f32; 6] = [
    1.987_569_15e-4,
    1.398_199_950_7e-3,
    8.333_451_907_3e-3,
    4.166_579_589_4e-2,
    1.666_666_545_9e-1,
    5.000_000_120_1e-1,
];

pub(crate) fn exp_scalar(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x < EXP_LO {
        return 0.0;
    }
    if x > EXP_HI {
        return f32::INFINITY;
    }
    let fx = round_ties_even(x * LOG2E);
    let r = (x - fx * LN2_HI) - fx * LN2_LO;
    let mut y = P[0];
    for &c in &P[1..] {
        y = y * r + c;
    }
    let z = y * (r * r) + r + 1.0;
    // Scale in two halves: 2^128 alone is not representable, but z * 2^128 can be.
    let n = fx as i32;
    let n1 = n >> 1;
    z * f32::from_bits(((n1 + 127) as u32) << 23) * f32::from_bits(((n - n1 + 127) as u32) << 23)
}

/// `round_ties_even` without relying on `std`: magic-number rounding, valid for |x| < 2^22.
pub(crate) fn round_ties_even(x: f32) -> f32 {
    pub(crate) const M: f32 = 12_582_912.0;
    (x + M) - M
}

