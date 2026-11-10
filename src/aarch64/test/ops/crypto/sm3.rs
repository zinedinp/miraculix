//! `SM3SS1` has a one-line definition (GB/T 32905 `SS1`) and is checked
//! exactly. The `TT*`/`PARTW*` steps are round fragments without such an
//! identity; they get the documented determinism / input-sensitivity smoke
//! test, including a check that the lane immediate actually selects a lane.

use super::Sm3;
use crate::aarch64::ops::macros::mix;

fn v(i: u64) -> [u32; 4] {
    core::array::from_fn(|k| mix(i * 4 + k as u64) as u32)
}

#[test]
fn sm3ss1_matches_definition() {
    let Some(t) = Sm3::detect() else { return };
    for i in 0..300 {
        let (a, b, c) = (v(i), v(i + 1000), v(i + 2000));
        let ss1 = a[3]
            .rotate_left(12)
            .wrapping_add(b[3])
            .wrapping_add(c[3])
            .rotate_left(7);
        assert_eq!(t.sm3ss1(a, b, c), [0, 0, 0, ss1]);
    }
}

type Fragment = fn(Sm3, [u32; 4], [u32; 4], [u32; 4]) -> [u32; 4];

#[test]
fn sm3_round_fragments_are_deterministic_and_input_sensitive() {
    let Some(t) = Sm3::detect() else { return };
    let (d, a, b) = (v(1), v(2), v(3));
    let ops: [(&str, Fragment); 6] = [
        ("sm3tt1a", Sm3::sm3tt1a::<0>),
        ("sm3tt1b", Sm3::sm3tt1b::<0>),
        ("sm3tt2a", Sm3::sm3tt2a::<0>),
        ("sm3tt2b", Sm3::sm3tt2b::<0>),
        ("sm3partw1", Sm3::sm3partw1),
        ("sm3partw2", Sm3::sm3partw2),
    ];
    for (name, f) in ops {
        let r = f(t, d, a, b);
        assert_eq!(r, f(t, d, a, b), "{name} deterministic");
        assert_ne!(r, f(t, v(9), a, b), "{name} depends on acc");
        assert_ne!(r, f(t, d, v(9), b), "{name} depends on a");
        assert_ne!(r, f(t, d, a, v(9)), "{name} depends on b");
    }
    // The immediate picks one word of `b`: changing only that word changes
    // the result, changing a different word does not.
    let mut b_lane2 = b;
    b_lane2[2] ^= 0xFFFF_FFFF;
    assert_ne!(t.sm3tt1a::<2>(d, a, b), t.sm3tt1a::<2>(d, a, b_lane2));
    assert_eq!(t.sm3tt1a::<1>(d, a, b), t.sm3tt1a::<1>(d, a, b_lane2));
    assert_eq!(t.sm3tt2b::<3>(d, a, b), t.sm3tt2b::<3>(d, a, b_lane2));
}
