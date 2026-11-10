//! Full SHA-1 and SHA-256 compressions driven through the instructions,
//! checked against the FIPS 180-4 `"abc"` example digests.
//!
//! The round constants are **derived** here (fractional parts of square and
//! cube roots of primes, FIPS 180-4 §4.2 / §5.3), not transcribed - the same
//! "no hand-copied tables" rule the AES tests follow. `f64` carries ~50
//! fractional bits for these small roots, well above the 32 needed; the
//! known-answer check would catch any flooring error anyway.

use super::Sha;

fn primes(n: usize) -> Vec<u32> {
    (2u32..)
        .filter(|&p| (2..p).take_while(|d| d * d <= p).all(|d| p % d != 0))
        .take(n)
        .collect()
}

fn frac32(x: f64) -> u32 {
    ((x - x.floor()) * 4_294_967_296.0) as u32
}

/// `"abc"` as one padded 512-bit block, big-endian words.
fn abc_block() -> [u32; 16] {
    let mut w = [0u32; 16];
    w[0] = 0x6162_6380;
    w[15] = 24;
    w
}

fn add4(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    core::array::from_fn(|i| a[i].wrapping_add(b[i]))
}

#[test]
fn sha256_abc_known_answer() {
    let Some(t) = Sha::detect() else { return };
    let p = primes(64);
    let k: Vec<u32> = p.iter().map(|&q| frac32((q as f64).cbrt())).collect();
    let h0: Vec<u32> = p[..8].iter().map(|&q| frac32((q as f64).sqrt())).collect();
    assert_eq!(k[0], 0x428a_2f98, "derived K[0]");
    assert_eq!(h0[0], 0x6a09_e667, "derived H0[0]");

    let blk = abc_block();
    let mut w: [[u32; 4]; 4] = core::array::from_fn(|i| core::array::from_fn(|j| blk[4 * i + j]));
    let (mut abcd, mut efgh) = ([h0[0], h0[1], h0[2], h0[3]], [h0[4], h0[5], h0[6], h0[7]]);
    for i in 0..16 {
        let wk = add4(
            w[i % 4],
            [k[4 * i], k[4 * i + 1], k[4 * i + 2], k[4 * i + 3]],
        );
        let abcd_old = abcd;
        abcd = t.sha256h(abcd, efgh, wk);
        efgh = t.sha256h2(efgh, abcd_old, wk);
        if i < 12 {
            w[i % 4] = t.sha256su1(
                t.sha256su0(w[i % 4], w[(i + 1) % 4]),
                w[(i + 2) % 4],
                w[(i + 3) % 4],
            );
        }
    }
    let digest: Vec<u32> = add4(abcd, [h0[0], h0[1], h0[2], h0[3]])
        .into_iter()
        .chain(add4(efgh, [h0[4], h0[5], h0[6], h0[7]]))
        .collect();
    assert_eq!(
        digest,
        [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad
        ]
    );
}

#[test]
fn sha1_abc_known_answer() {
    let Some(t) = Sha::detect() else { return };
    // K = floor(2^30 · sqrt(n)) for n = 2, 3, 5, 10 (FIPS 180-4 §4.2.1).
    let k = [2f64, 3.0, 5.0, 10.0].map(|n| (n.sqrt() * 1_073_741_824.0) as u32);
    assert_eq!(
        k,
        [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6],
        "derived K"
    );
    // H0: the byte counting pattern 01 23 45 .. EF FE DC .. 10, then F0 E1 D2 C3.
    let h0 = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];

    let blk = abc_block();
    let mut w: [[u32; 4]; 4] = core::array::from_fn(|i| core::array::from_fn(|j| blk[4 * i + j]));
    let mut abcd = [h0[0], h0[1], h0[2], h0[3]];
    let mut e = h0[4];
    for i in 0..20 {
        let wk = add4(w[i % 4], [k[i / 5]; 4]);
        let next_e = t.sha1h(abcd[0]);
        abcd = match i / 5 {
            0 => t.sha1c(abcd, e, wk),
            2 => t.sha1m(abcd, e, wk),
            _ => t.sha1p(abcd, e, wk),
        };
        e = next_e;
        if i < 16 {
            w[i % 4] = t.sha1su1(
                t.sha1su0(w[i % 4], w[(i + 1) % 4], w[(i + 2) % 4]),
                w[(i + 3) % 4],
            );
        }
    }
    let d = add4(abcd, [h0[0], h0[1], h0[2], h0[3]]);
    assert_eq!(
        [d[0], d[1], d[2], d[3], e.wrapping_add(h0[4])],
        [0xa9993e36, 0x4706816a, 0xba3e2571, 0x7850c26c, 0x9cd0d89d]
    );
}

#[test]
fn sha1h_is_rotate_left_30() {
    let Some(t) = Sha::detect() else { return };
    for x in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
        assert_eq!(t.sha1h(x), x.rotate_left(30));
    }
}
