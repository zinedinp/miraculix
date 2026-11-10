//! A hardware RNG has no oracle. What can be checked: reads mostly succeed,
//! successive values differ, and the bits are not stuck. A failure here means
//! the wrapper is wired wrong (e.g. always returning the same register), not
//! that the entropy is weak - that is not something a unit test can judge.

use super::Rng;

#[test]
fn rndr_values_vary_and_no_bit_is_stuck() {
    let Some(t) = Rng::detect() else { return };
    let vals: Vec<u64> = (0..64).filter_map(|_| t.rndr()).collect();
    assert!(
        vals.len() >= 32,
        "only {} of 64 RNDR reads succeeded",
        vals.len()
    );
    let (or, and) = vals.iter().fold((0u64, !0u64), |(o, a), &v| (o | v, a & v));
    assert_eq!(or, !0, "some bit was never 1");
    assert_eq!(and, 0, "some bit was never 0");
    assert!(vals.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn rndrrs_returns_values_too() {
    let Some(t) = Rng::detect() else { return };
    let vals: Vec<u64> = (0..16).filter_map(|_| t.rndrrs()).collect();
    assert!(
        vals.len() >= 8,
        "only {} of 16 RNDRRS reads succeeded",
        vals.len()
    );
    assert!(vals.windows(2).any(|w| w[0] != w[1]));
}
