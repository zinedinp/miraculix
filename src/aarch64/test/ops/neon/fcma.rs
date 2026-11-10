//! Oracles from the Arm pseudocode, one fused `mul_add` per partial product.

use super::Fcma;
use crate::aarch64::ops::macros::TestElem;

fn pairs<T: TestElem>(r: usize, off: usize) -> [T; 4] {
    core::array::from_fn(|i| T::sample(r * 97 + i + off))
}

#[test]
fn cadd_f32x4_matches_rotation() {
    let Some(t) = Fcma::detect() else { return };
    for r in 0..8 {
        let (a, b) = (pairs::<f32>(r, 0), pairs::<f32>(r, 50));
        let rot90 = [a[0] - b[1], a[1] + b[0], a[2] - b[3], a[3] + b[2]];
        let rot270 = [a[0] + b[1], a[1] - b[0], a[2] + b[3], a[3] - b[2]];
        assert_eq!(t.cadd_rot90_f32x4(a, b), rot90);
        assert_eq!(t.cadd_rot270_f32x4(a, b), rot270);
        let (a2, b2) = ([a[0] as f64, a[1] as f64], [b[0] as f64, b[1] as f64]);
        assert_eq!(t.cadd_rot90_f64x2(a2, b2), [a2[0] - b2[1], a2[1] + b2[0]]);
        assert_eq!(t.cadd_rot270_f64x2(a2, b2), [a2[0] + b2[1], a2[1] - b2[0]]);
    }
}

#[test]
fn cmla_rotations_match_pseudocode() {
    let Some(t) = Fcma::detect() else { return };
    for r in 0..8 {
        // 1.000123 scaling makes the products inexact, so fusion is observable.
        let acc = pairs::<f32>(r, 0).map(|x| x * 1.000_123);
        let a = pairs::<f32>(r, 30).map(|x| x * 1.000_321);
        let b = pairs::<f32>(r, 60);
        let per = |f: &dyn Fn(usize) -> [f32; 2]| -> [f32; 4] {
            let (p, q) = (f(0), f(2));
            [p[0], p[1], q[0], q[1]]
        };
        let r0 = per(&|k| {
            [
                a[k].mul_add(b[k], acc[k]),
                a[k].mul_add(b[k + 1], acc[k + 1]),
            ]
        });
        let r90 = per(&|k| {
            [
                (-a[k + 1]).mul_add(b[k + 1], acc[k]),
                a[k + 1].mul_add(b[k], acc[k + 1]),
            ]
        });
        let r180 = per(&|k| {
            [
                (-a[k]).mul_add(b[k], acc[k]),
                (-a[k]).mul_add(b[k + 1], acc[k + 1]),
            ]
        });
        let r270 = per(&|k| {
            [
                a[k + 1].mul_add(b[k + 1], acc[k]),
                (-a[k + 1]).mul_add(b[k], acc[k + 1]),
            ]
        });
        assert_eq!(t.cmla_rot0_f32x4(acc, a, b), r0, "#0 round {r}");
        assert_eq!(t.cmla_rot90_f32x4(acc, a, b), r90, "#90 round {r}");
        assert_eq!(t.cmla_rot180_f32x4(acc, a, b), r180, "#180 round {r}");
        assert_eq!(t.cmla_rot270_f32x4(acc, a, b), r270, "#270 round {r}");
    }
}

#[test]
fn cmla_is_a_complex_multiply_accumulate() {
    let Some(t) = Fcma::detect() else { return };
    // (1 + 2i)(3 + 4i) = -5 + 10i, exact in both widths.
    assert_eq!(
        t.cmla_f32x4([0.0; 4], [1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 5.0, 0.0]),
        [-5.0, 10.0, 0.0, 5.0]
    );
    assert_eq!(
        t.cmla_f64x2([1.0, 1.0], [1.0, 2.0], [3.0, 4.0]),
        [-4.0, 11.0]
    );
}
