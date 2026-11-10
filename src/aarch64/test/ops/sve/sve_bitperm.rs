//! `Sve2Bitperm` against a scalar `pext`/`pdep`, bit by bit.

use super::Sve2Bitperm;
use crate::aarch64::ops::macros::{SVE_LENS, mix};

/// Bits of `a` at `mask`'s set positions, packed low (x86 `pext`).
fn pext(a: u64, mask: u64) -> u64 {
    let (mut out, mut k) = (0u64, 0);
    for bit in 0..64 {
        if mask >> bit & 1 == 1 {
            out |= (a >> bit & 1) << k;
            k += 1;
        }
    }
    out
}

/// Low bits of `a` spread to `mask`'s set positions (x86 `pdep`).
fn pdep(a: u64, mask: u64) -> u64 {
    let (mut out, mut k) = (0u64, 0);
    for bit in 0..64 {
        if mask >> bit & 1 == 1 {
            out |= (a >> k & 1) << bit;
            k += 1;
        }
    }
    out
}

/// `bext(a, m)` low, then `bext(a, !m)` above it, within `bits`.
fn bgrp(a: u64, mask: u64, bits: u32) -> u64 {
    let all = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let (m, a) = (mask & all, a & all);
    let lo = pext(a, m);
    let hi = pext(a, !m & all);
    (lo | hi.checked_shl(m.count_ones()).unwrap_or(0)) & all
}

macro_rules! bitperm_test {
    ($test:ident, $E:ty, $bits:literal, $bext:ident, $bdep:ident, $bgrp:ident) => {
        #[test]
        fn $test() {
            let Some(t) = Sve2Bitperm::detect() else {
                return;
            };
            for len in SVE_LENS {
                let a: Vec<$E> = (0..len).map(|i| mix(i as u64) as $E).collect();
                // Sparse, dense and random masks.
                let m: Vec<$E> = (0..len)
                    .map(|i| match i % 3 {
                        0 => (mix(i as u64 ^ 9) & mix(i as u64 ^ 7)) as $E,
                        1 => !(mix(i as u64 ^ 5) & mix(i as u64 ^ 3)) as $E,
                        _ => mix(i as u64 ^ 1) as $E,
                    })
                    .collect();
                let check = |got: &[$E], f: &dyn Fn(u64, u64) -> u64, what: &str| {
                    for i in 0..len {
                        assert_eq!(
                            got[i] as u64,
                            f(a[i] as u64, m[i] as u64) as $E as u64,
                            "{what} len={len} i={i}"
                        );
                    }
                };
                let mut out = vec![0 as $E; len];
                t.$bext(&a, &m, &mut out);
                check(&out, &|x, y| pext(x, y), stringify!($bext));
                t.$bdep(&a, &m, &mut out);
                check(&out, &|x, y| pdep(x, y), stringify!($bdep));
                t.$bgrp(&a, &m, &mut out);
                check(&out, &|x, y| bgrp(x, y, $bits), stringify!($bgrp));
            }
        }
    };
}

bitperm_test!(
    bitperm_u8_matches_scalar,
    u8,
    8,
    bext_u8_slice,
    bdep_u8_slice,
    bgrp_u8_slice
);
bitperm_test!(
    bitperm_u16_matches_scalar,
    u16,
    16,
    bext_u16_slice,
    bdep_u16_slice,
    bgrp_u16_slice
);
bitperm_test!(
    bitperm_u32_matches_scalar,
    u32,
    32,
    bext_u32_slice,
    bdep_u32_slice,
    bgrp_u32_slice
);
bitperm_test!(
    bitperm_u64_matches_scalar,
    u64,
    64,
    bext_u64_slice,
    bdep_u64_slice,
    bgrp_u64_slice
);
