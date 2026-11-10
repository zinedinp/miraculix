//! `SveI8mm` against an exact `i64` oracle, segment by segment.

use super::SveI8mm;
use crate::aarch64::ops::macros::{SVE_LENS, mix};

fn mmla(acc: &[i64], a: &[i64], b: &[i64]) -> Vec<i64> {
    (0..acc.len())
        .map(|n| {
            let (seg, i, j) = (n / 4, n % 4 / 2, n % 2);
            (0..8).fold(acc[n], |s, k| {
                s + a[16 * seg + 8 * i + k] * b[16 * seg + 8 * j + k]
            })
        })
        .collect()
}

fn dot4(acc: &[i64], a: &[i64], b: &[i64]) -> Vec<i64> {
    (0..acc.len())
        .map(|l| (0..4).fold(acc[l], |s, k| s + a[4 * l + k] * b[4 * l + k]))
        .collect()
}

macro_rules! i8mm_slice_test {
    ($test:ident, $method:ident, $Acc:ty, $A:ty, $B:ty, $oracle:ident, $group:literal) => {
        #[test]
        fn $test() {
            let Some(t) = SveI8mm::detect() else { return };
            for n in SVE_LENS.map(|n| n / $group * $group) {
                let acc: Vec<$Acc> = (0..n).map(|i| mix(i as u64 ^ 11) as i16 as $Acc).collect();
                let a: Vec<$A> = (0..4 * n).map(|i| mix(i as u64 ^ 12) as $A).collect();
                let b: Vec<$B> = (0..4 * n).map(|i| mix(i as u64 ^ 13) as $B).collect();
                let mut out = vec![0 as $Acc; n];
                t.$method(&acc, &a, &b, &mut out);
                let wide = |v: &[$A]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();
                let want = $oracle(
                    &acc.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                    &wide(&a),
                    &b.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                );
                // The instruction wraps modulo 2^32; so does the cast.
                let want: Vec<$Acc> = want.iter().map(|&x| x as $Acc).collect();
                assert_eq!(out, want, "{} n={n}", stringify!($method));
            }
        }
    };
}

i8mm_slice_test!(
    mmla_s32_slice_matches_i64,
    mmla_s32_slice,
    i32,
    i8,
    i8,
    mmla,
    4
);
i8mm_slice_test!(
    mmla_u32_slice_matches_i64,
    mmla_u32_slice,
    u32,
    u8,
    u8,
    mmla,
    4
);
i8mm_slice_test!(
    mmla_us32_slice_matches_i64,
    mmla_us32_slice,
    i32,
    u8,
    i8,
    mmla,
    4
);
i8mm_slice_test!(
    dot_us32_slice_matches_i64,
    dot_us32_slice,
    i32,
    u8,
    i8,
    dot4,
    1
);

#[test]
#[should_panic(expected = "multiple of the 4-lane segment")]
fn mmla_rejects_a_partial_segment() {
    let Some(t) = SveI8mm::detect() else {
        panic!("multiple of the 4-lane segment (vacuous)")
    };
    let mut out = [0i32; 6];
    t.mmla_s32_slice(&[0; 6], &[0; 24], &[0; 24], &mut out);
}
