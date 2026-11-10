use super::Pmull;
use crate::aarch64::ops::macros::mix;

/// Carry-less multiply by definition: XOR of shifted copies.
fn clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|i| b >> i & 1 == 1)
        .fold(0u128, |acc, i| acc ^ ((a as u128) << i))
}

#[test]
fn pmull_u64_matches_shift_xor_definition() {
    let Some(t) = Pmull::detect() else { return };
    for i in 0..500 {
        let (a, b) = (mix(i), mix(i + 7777));
        assert_eq!(t.pmull_u64(a, b), clmul(a, b), "{a:#x} * {b:#x}");
    }
    assert_eq!(t.pmull_u64(u64::MAX, u64::MAX), clmul(u64::MAX, u64::MAX));
    // x^63 * x^63 = x^126.
    assert_eq!(t.pmull_u64(1 << 63, 1 << 63), 1u128 << 126);
}

#[test]
fn pmull_high_uses_the_upper_lanes() {
    let Some(t) = Pmull::detect() else { return };
    for i in 0..100 {
        let a = [mix(i), mix(i + 1)];
        let b = [mix(i + 2), mix(i + 3)];
        assert_eq!(t.pmull_high_u64x2(a, b), clmul(a[1], b[1]));
    }
}
