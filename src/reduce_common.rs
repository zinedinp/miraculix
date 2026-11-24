//! Arch-neutral pieces shared by the per-architecture slice reductions
//! (`x86::reduce`, `aarch64::reduce`, `riscv::auto`): the top-k driver and
//! the scalar `exp` with its constants, the GEMM and sliding-dot scalar paths.
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

// ---- logsumexp / log_softmax driver ----

/// Natural logarithm in `f64` (`no_std`, no `libm`): `x = m * 2^e` with
/// `m` in `[sqrt(1/2), sqrt(2))`, then `ln m = 2 atanh((m-1)/(m+1))` as an odd
/// series (|s| <= 0.1716, truncation error < 1e-14). Handles 0, negative, inf, NaN.
pub(crate) fn ln_f64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    // Bring subnormals into the normal range.
    let (x, adj) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, -54i64)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023 + adj;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 0.0;
    let mut k = 31.0;
    while k >= 1.0 {
        p = p * s2 + 1.0 / k;
        k -= 2.0;
    }
    2.0 * s * p + (e as f64) * core::f64::consts::LN_2
}

const LSE_BLOCK: usize = 1024;

/// `ln(sum(exp(a[i])))`, shifted by the maximum so it never overflows.
/// `max(a)` ignores NaN, an empty or all-NaN slice gives `-inf`, `+inf` gives
/// `+inf`; a NaN element makes the sum (and the result) NaN.
/// The architecture supplies `max` (slice maximum), `exp` (see `exp_f32`) and
/// `sum` (slice sum); blocks are added in `f64`.
pub(crate) fn logsumexp_with(
    a: &[f32],
    max: impl Fn(&[f32]) -> f32,
    exp: impl Fn(&[f32], &mut [f32]),
    sum: impl Fn(&[f32]) -> f32,
) -> f32 {
    let m = max(a);
    if m == f32::NEG_INFINITY || m == f32::INFINITY {
        // All -inf / empty -> -inf; a +inf element -> +inf (NaN stays NaN if any).
        return if a.iter().any(|x| x.is_nan()) {
            f32::NAN
        } else {
            m
        };
    }
    let mut buf = [0f32; LSE_BLOCK];
    let mut total = 0f64;
    for chunk in a.chunks(LSE_BLOCK) {
        let t = &mut buf[..chunk.len()];
        for (o, &x) in t.iter_mut().zip(chunk) {
            *o = x - m;
        }
        let mut e = [0f32; LSE_BLOCK];
        exp(t, &mut e[..chunk.len()]);
        total += sum(&e[..chunk.len()]) as f64;
    }
    (m as f64 + ln_f64(total)) as f32
}

// ------------------------------------------------------- GEMM micro-kernel

/// Rows of a packed A panel for `gemm_f32_kernel` (same on every architecture).
pub const GEMM_MR: usize = 8;
/// Columns of a packed B panel for `gemm_f32_kernel` (same on every architecture).
pub const GEMM_NR: usize = 32;

/// Checks the slice lengths of `gemm_f32_kernel`.
#[inline]
pub(crate) fn gemm_check(kc: usize, a: &[f32], b: &[f32]) {
    assert!(
        a.len() >= kc * GEMM_MR,
        "gemm_f32_kernel: A panel {} < kc*MR = {}",
        a.len(),
        kc * GEMM_MR
    );
    assert!(
        b.len() >= kc * GEMM_NR,
        "gemm_f32_kernel: B panel {} < kc*NR = {}",
        b.len(),
        kc * GEMM_NR
    );
}

/// Portable `gemm_f32_kernel`: `acc[i][j] += a[k][i] * b[k][j]`, k ascending,
/// product and sum rounded separately.
pub(crate) fn gemm_kernel_scalar(
    kc: usize,
    a: &[f32],
    b: &[f32],
    acc: &mut [[f32; GEMM_NR]; GEMM_MR],
) {
    for k in 0..kc {
        let (ak, bk) = (&a[k * GEMM_MR..][..GEMM_MR], &b[k * GEMM_NR..][..GEMM_NR]);
        for (row, &ai) in acc.iter_mut().zip(ak) {
            for (c, &bj) in row.iter_mut().zip(bk) {
                *c += ai * bj;
            }
        }
    }
}

// -------------------------------------------------------------- byte blend

/// The biggest `den` of `lerp_u8`: every sum `2 * (a * (den - num) + b * num) + den`
/// stays below 2^24, exact in `f32`.
pub(crate) const LERP_MAX_DEN: u16 = 32767;

/// Checks the arguments of `lerp_u8`.
#[inline]
pub(crate) fn lerp_check(a: &[u8], b: &[u8], num: u16, den: u16, out: &[u8]) {
    assert!(
        a.len() == b.len() && b.len() == out.len(),
        "lerp_u8: lengths {} {} {}",
        a.len(),
        b.len(),
        out.len()
    );
    assert!(
        (1..=LERP_MAX_DEN).contains(&den) && num <= den,
        "lerp_u8: {num} / {den}"
    );
}

/// Portable `lerp_u8` from `first` on.
pub(crate) fn lerp_scalar(a: &[u8], b: &[u8], num: u16, den: u16, out: &mut [u8], first: usize) {
    let (num, den) = (u32::from(num), u32::from(den));
    for ((&x, &y), o) in a[first..].iter().zip(&b[first..]).zip(&mut out[first..]) {
        let sum = u32::from(x) * (den - num) + u32::from(y) * num;
        *o = ((2 * sum + den) / (2 * den)) as u8;
    }
}

/// The `f32` operands of the vector `lerp_u8`: weights of `a` and `b`, `den`, `2 * den`.
/// Each lane computes `trunc((2 * (a * wa + b * wb) + den) / (2 * den))`: the products,
/// sums and the doubling are exact below 2^24, and the one rounded division is
/// never off by enough to cross an integer (a non-integer quotient under 256 is at
/// least `1 / (2 * den)` from one, more than half an ulp).
#[inline]
pub(crate) fn lerp_operands(num: u16, den: u16) -> [f32; 4] {
    let (num, den) = (f32::from(num), f32::from(den));
    [den - num, num, den, 2.0 * den]
}

// ------------------------------------------------------------- sliding dot

/// Checks the slice lengths of `sliding_dot_f32`: every window
/// `signal[i..i + taps.len()]` with `i < out.len()` is in bounds.
#[inline]
pub(crate) fn sliding_check(signal: &[f32], taps: &[f32], out: &[f32]) {
    assert!(
        out.is_empty() || signal.len() + 1 >= out.len() + taps.len(),
        "sliding_dot_f32: signal {} < out {} + taps {} - 1",
        signal.len(),
        out.len(),
        taps.len()
    );
}

/// Portable `sliding_dot_f32` for the outputs from `first` on (`out` is
/// `out[first..]` of the whole call): k ascending, product and sum rounded
/// separately.
pub(crate) fn sliding_scalar(signal: &[f32], taps: &[f32], out: &mut [f32], first: usize) {
    for (i, o) in out.iter_mut().enumerate() {
        let window = &signal[first + i..][..taps.len()];
        let mut acc = 0.0;
        for (&x, &t) in window.iter().zip(taps) {
            acc += x * t;
        }
        *o = acc;
    }
}
