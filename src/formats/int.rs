//! Packed INT1 / INT2 / INT4 storage and dot products.
//!
//! Elements are packed little-endian by position: element 0 sits in the low
//! bits of byte 0. `B` is the bit width (1, 2 or 4).

const fn check<const B: usize>() {
    assert!(B == 1 || B == 2 || B == 4, "bit width must be 1, 2 or 4");
}

/// Bytes needed to hold `n` elements of `B` bits.
pub const fn packed_len<const B: usize>(n: usize) -> usize {
    check::<B>();
    (n * B).div_ceil(8)
}

/// Unpacks unsigned `B`-bit elements into `out` (one per byte).
///
/// # Panics
/// If `packed` is shorter than `packed_len::<B>(out.len())`.
pub fn unpack_uint<const B: usize>(packed: &[u8], out: &mut [u8]) {
    check::<B>();
    assert!(packed.len() >= packed_len::<B>(out.len()));
    let per = 8 / B;
    let mask = ((1u16 << B) - 1) as u8;
    for (i, o) in out.iter_mut().enumerate() {
        *o = (packed[i / per] >> ((i % per) * B)) & mask;
    }
}

/// Unpacks two's-complement `B`-bit elements into `out`, sign-extended.
///
/// # Panics
/// If `packed` is shorter than `packed_len::<B>(out.len())`.
pub fn unpack_int<const B: usize>(packed: &[u8], out: &mut [i8]) {
    check::<B>();
    assert!(packed.len() >= packed_len::<B>(out.len()));
    let per = 8 / B;
    let mask = ((1u16 << B) - 1) as u8;
    let shift = 8 - B as u32;
    for (i, o) in out.iter_mut().enumerate() {
        let v = (packed[i / per] >> ((i % per) * B)) & mask;
        *o = ((v << shift) as i8) >> shift;
    }
}

/// Packs unsigned elements, saturating to `0..2^B`. Unused tail bits are zeroed.
///
/// # Panics
/// If `packed` is shorter than `packed_len::<B>(vals.len())`.
pub fn pack_uint<const B: usize>(vals: &[u8], packed: &mut [u8]) {
    check::<B>();
    let n = packed_len::<B>(vals.len());
    assert!(packed.len() >= n);
    packed[..n].fill(0);
    let per = 8 / B;
    let max = ((1u16 << B) - 1) as u8;
    for (i, &v) in vals.iter().enumerate() {
        packed[i / per] |= v.min(max) << ((i % per) * B);
    }
}

/// Packs signed elements, saturating to `-2^(B-1)..2^(B-1)`. Unused tail bits are zeroed.
///
/// # Panics
/// If `packed` is shorter than `packed_len::<B>(vals.len())`.
pub fn pack_int<const B: usize>(vals: &[i8], packed: &mut [u8]) {
    check::<B>();
    let n = packed_len::<B>(vals.len());
    assert!(packed.len() >= n);
    packed[..n].fill(0);
    let per = 8 / B;
    let (lo, hi) = (-(1i16 << (B - 1)), (1i16 << (B - 1)) - 1);
    let mask = ((1u16 << B) - 1) as u8;
    for (i, &v) in vals.iter().enumerate() {
        let c = (v as i16).clamp(lo, hi) as u8 & mask;
        packed[i / per] |= c << ((i % per) * B);
    }
}

/// Dot product of two packed signed INT4 vectors of `n` elements, as `i32`.
///
/// # Panics
/// If either slice is shorter than `n.div_ceil(2)` bytes.
pub fn dot_i4(a: &[u8], b: &[u8], n: usize) -> i32 {
    assert!(a.len() >= n.div_ceil(2) && b.len() >= n.div_ceil(2));
    let nib = |x: u8| ((x << 4) as i8 >> 4) as i32;
    let mut acc = 0;
    for i in 0..n {
        let (x, y) = (a[i / 2] >> ((i % 2) * 4), b[i / 2] >> ((i % 2) * 4));
        acc += nib(x) * nib(y);
    }
    acc
}

/// Dot product of two packed unsigned INT4 vectors of `n` elements, as `u32`.
///
/// # Panics
/// If either slice is shorter than `n.div_ceil(2)` bytes.
pub fn dot_u4(a: &[u8], b: &[u8], n: usize) -> u32 {
    assert!(a.len() >= n.div_ceil(2) && b.len() >= n.div_ceil(2));
    let mut acc = 0;
    for i in 0..n {
        let s = (i % 2) * 4;
        acc += (((a[i / 2] >> s) & 15) as u32) * (((b[i / 2] >> s) & 15) as u32);
    }
    acc
}

/// INT1 AND-popcount: number of positions where both bits are set.
/// This is the binary-MMA `and.popc` operation.
pub fn popcount_and(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

/// INT1 XOR-popcount: Hamming distance. This is the binary-MMA `xor.popc` operation.
pub fn popcount_xor(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Dot product of two ±1 vectors stored as bits (1 = +1, 0 = -1), `n` elements:
/// `n - 2 * hamming`. Bits past `n` in the last byte are ignored.
pub fn dot_pm1(a: &[u8], b: &[u8], n: usize) -> i32 {
    let bytes = n.div_ceil(8);
    assert!(a.len() >= bytes && b.len() >= bytes);
    let mut ham = 0u32;
    for i in 0..bytes {
        let mut x = a[i] ^ b[i];
        if i == bytes - 1 && n % 8 != 0 {
            x &= (1u8 << (n % 8)) - 1;
        }
        ham += x.count_ones();
    }
    n as i32 - 2 * ham as i32
}
