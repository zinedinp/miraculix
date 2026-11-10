use super::Dotprod;
use crate::aarch64::ops::macros::TestElem;

#[test]
fn dot_s32_matches_widened_sum() {
    let Some(t) = Dotprod::detect() else { return };
    for r in 0..16usize {
        let acc: [i32; 4] = core::array::from_fn(|i| i32::sample(r * 31 + i));
        let a: [i8; 16] = core::array::from_fn(|i| i8::sample(r * 97 + i));
        let b: [i8; 16] = core::array::from_fn(|i| i8::sample(r * 97 + i + 50));
        let expect: [i32; 4] = core::array::from_fn(|l| {
            (0..4).fold(acc[l], |s, k| {
                s.wrapping_add(a[4 * l + k] as i32 * b[4 * l + k] as i32)
            })
        });
        assert_eq!(t.dot_s32(acc, a, b), expect, "round {r}");
    }
    // Extremes: four MIN*MIN products = 4 * 2^14 per lane, no 8/16-bit overflow.
    assert_eq!(
        t.dot_s32([0; 4], [i8::MIN; 16], [i8::MIN; 16]),
        [4 * 128 * 128; 4]
    );
}

#[test]
fn dot_u32_matches_widened_sum() {
    let Some(t) = Dotprod::detect() else { return };
    for r in 0..16usize {
        let acc: [u32; 4] = core::array::from_fn(|i| u32::sample(r * 31 + i));
        let a: [u8; 16] = core::array::from_fn(|i| u8::sample(r * 97 + i));
        let b: [u8; 16] = core::array::from_fn(|i| u8::sample(r * 97 + i + 50));
        let expect: [u32; 4] = core::array::from_fn(|l| {
            (0..4).fold(acc[l], |s, k| {
                s.wrapping_add(a[4 * l + k] as u32 * b[4 * l + k] as u32)
            })
        });
        assert_eq!(t.dot_u32(acc, a, b), expect, "round {r}");
    }
    assert_eq!(t.dot_u32([0; 4], [255; 16], [255; 16]), [4 * 255 * 255; 4]);
}
