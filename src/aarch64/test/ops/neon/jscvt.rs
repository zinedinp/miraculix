use super::Jscvt;
use crate::aarch64::ops::macros::mix;

/// ECMAScript 2024 §7.1.6 ToInt32, spelled out.
fn to_int32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    x.trunc().rem_euclid(4_294_967_296.0) as u32 as i32
}

#[test]
fn jscvt_matches_ecmascript_to_int32() {
    let Some(t) = Jscvt::detect() else { return };
    let edge = [
        0.0,
        -0.0,
        -0.9,
        1.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_649.0,
        4_294_967_296.5,
        1e20,
        -1e20,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for x in edge {
        assert_eq!(t.jscvt(x), to_int32(x), "jscvt({x:e})");
    }
    for i in 0..1000u64 {
        let x = f64::from_bits(mix(i)).clamp(-1e30, 1e30);
        assert_eq!(t.jscvt(x), to_int32(x), "jscvt({x:e})");
    }
    // The point of the instruction: wrap, where `as` saturates.
    assert_eq!(t.jscvt(4_294_967_297.0), 1);
    assert_ne!(t.jscvt(4_294_967_297.0), 4_294_967_297.0f64 as i32);
}

#[test]
fn jscvt_slice_matches_scalar() {
    let Some(t) = Jscvt::detect() else { return };
    let a: Vec<f64> = (0..100u64).map(|i| (mix(i) as i64) as f64 / 3.0).collect();
    let mut out = vec![0; a.len()];
    t.jscvt_slice(&a, &mut out);
    assert_eq!(out, a.iter().map(|&x| to_int32(x)).collect::<Vec<_>>());
}
