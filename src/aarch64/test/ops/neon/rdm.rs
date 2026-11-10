use super::Rdm;
use crate::aarch64::ops::macros::TestElem;

/// Arm pseudocode for `SQRDMLAH`/`SQRDMLSH`, evaluated in `i128` so nothing
/// can overflow before the single saturation.
fn oracle(a: i64, b: i64, c: i64, bits: u32, sub: bool) -> i64 {
    let prod = 2 * (b as i128) * (c as i128);
    let acc = ((a as i128) << bits) + if sub { -prod } else { prod } + (1i128 << (bits - 1));
    let r = acc >> bits;
    let (lo, hi) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
    r.clamp(lo, hi) as i64
}

macro_rules! rdm_test {
    ($test:ident, $fixed:ident, $Elem:ty, $lanes:literal, $sub:literal) => {
        #[test]
        fn $test() {
            let Some(t) = Rdm::detect() else { return };
            for r in 0..16usize {
                let a: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i));
                let b: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i + 40));
                let c: [$Elem; $lanes] = core::array::from_fn(|i| <$Elem>::sample(r * 97 + i + 80));
                let expect: [$Elem; $lanes] = core::array::from_fn(|i| {
                    oracle(a[i] as i64, b[i] as i64, c[i] as i64, <$Elem>::BITS, $sub) as $Elem
                });
                assert_eq!(
                    t.$fixed(a, b, c),
                    expect,
                    "{} round {r}",
                    stringify!($fixed)
                );
            }
            // MIN * MIN doubles to 2^(2·esize - 1): the case plain SQRDMULH
            // saturates on, and RDM only saturates after the accumulate.
            let m = <$Elem>::MIN;
            let got = t.$fixed([-1; $lanes], [m; $lanes], [m; $lanes]);
            let want = oracle(-1, m as i64, m as i64, <$Elem>::BITS, $sub) as $Elem;
            assert_eq!(got, [want; $lanes], "{} MIN*MIN", stringify!($fixed));
        }
    };
}

rdm_test!(
    qrdmlah_i16x8_matches_pseudocode,
    qrdmlah_i16x8,
    i16,
    8,
    false
);
rdm_test!(
    qrdmlah_i32x4_matches_pseudocode,
    qrdmlah_i32x4,
    i32,
    4,
    false
);
rdm_test!(
    qrdmlsh_i16x8_matches_pseudocode,
    qrdmlsh_i16x8,
    i16,
    8,
    true
);
rdm_test!(
    qrdmlsh_i32x4_matches_pseudocode,
    qrdmlsh_i32x4,
    i32,
    4,
    true
);
rdm_test!(
    qrdmlah_i16x4_matches_pseudocode,
    qrdmlah_i16x4,
    i16,
    4,
    false
);
rdm_test!(
    qrdmlah_i32x2_matches_pseudocode,
    qrdmlah_i32x2,
    i32,
    2,
    false
);
rdm_test!(
    qrdmlsh_i16x4_matches_pseudocode,
    qrdmlsh_i16x4,
    i16,
    4,
    true
);
rdm_test!(
    qrdmlsh_i32x2_matches_pseudocode,
    qrdmlsh_i32x2,
    i32,
    2,
    true
);
