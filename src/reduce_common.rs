//! Arch-neutral top-k driver. Each architecture supplies a `compress` kernel
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
