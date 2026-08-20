use super::super::super::macros::{slice_binop_matches_scalar_test, slice_ternop_matches_scalar_test};
#[cfg(feature = "wider-bus-lift")]
use super::super::super::macros::slice_binop_lifted_matches_scalar_test;
use super::*;

#[test]
fn mul_i32x4_multiplies_lanes() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.mul_i32x4([1, 2, 3, 4], [2, 2, 2, 2]), [2, 4, 6, 8]);
}

#[test]
fn mul_i32x4_wraps_on_overflow() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.mul_i32x4([1 << 30, 0, 0, 0], [4, 0, 0, 0]), [0, 0, 0, 0]);
}

#[test]
fn min_i32x4_picks_smaller_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.min_i32x4([1, 20, -3, 4], [10, 2, 3, -4]), [1, 2, -3, -4]);
}

#[test]
fn max_i32x4_picks_larger_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.max_i32x4([1, 20, -3, 4], [10, 2, 3, -4]), [10, 20, 3, 4]);
}

#[test]
fn mul_u32x4_multiplies_lanes() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.mul_u32x4([1, 2, 3, 4], [2, 2, 2, 2]), [2, 4, 6, 8]);
}

#[test]
fn min_u32x4_picks_smaller_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.min_u32x4([1, 20, 3, 4], [10, 2, 30, 0]), [1, 2, 3, 0]);
}

#[test]
fn max_u32x4_picks_larger_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	assert_eq!(sse41.max_u32x4([1, 20, 3, 4], [10, 2, 30, 0]), [10, 20, 30, 4]);
}

slice_binop_matches_scalar_test!(mul_i32_slice_matches_scalar, Sse41, mul_i32_slice, |x: i32, y: i32| x.wrapping_mul(y), i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(mul_i32_slice_lifted_matches_scalar, Sse41, Avx, mul_i32_slice_lifted, |x: i32, y: i32| x.wrapping_mul(y), i32);
slice_binop_matches_scalar_test!(min_i32_slice_matches_scalar, Sse41, min_i32_slice, |x, y| x.min(y), i32);
slice_binop_matches_scalar_test!(max_i32_slice_matches_scalar, Sse41, max_i32_slice, |x, y| x.max(y), i32);

slice_binop_matches_scalar_test!(mul_u32_slice_matches_scalar, Sse41, mul_u32_slice, |x: u32, y: u32| x.wrapping_mul(y), u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(mul_u32_slice_lifted_matches_scalar, Sse41, Avx, mul_u32_slice_lifted, |x: u32, y: u32| x.wrapping_mul(y), u32);
slice_binop_matches_scalar_test!(min_u32_slice_matches_scalar, Sse41, min_u32_slice, |x, y| x.min(y), u32);
slice_binop_matches_scalar_test!(max_u32_slice_matches_scalar, Sse41, max_u32_slice, |x, y| x.max(y), u32);

#[test]
fn blend_picks_b_on_negative_mask_lanes() {
	let Some(sse41) = Sse41::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	let mask = [-1.0, 1.0, -1.0, 1.0];
	assert_eq!(sse41.blend_f32x4(a, b, mask), [10.0, 2.0, 30.0, 4.0]);
}

#[test]
fn all_positive_mask_returns_a_unchanged() {
	let Some(sse41) = Sse41::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	let mask = [1.0, 1.0, 1.0, 1.0];
	assert_eq!(sse41.blend_f32x4(a, b, mask), a);
}

#[test]
fn select_i32x4_picks_b_where_mask_set() {
	let Some(sse41) = Sse41::detect() else { return };
	let a = [1, 2, 3, 4];
	let b = [10, 20, 30, 40];
	let mask = [-1, 0, -1, 0];
	assert_eq!(sse41.select_i32x4(a, b, mask), [10, 2, 30, 4]);
}

#[test]
fn select_f32x4_uses_sign_bit_not_zero_test() {
	let Some(sse41) = Sse41::detect() else { return };
	let a = [1.0; 4];
	let b = [2.0; 4];
	let mask = [-0.0f32; 4];
	assert_eq!(sse41.select_f32x4(a, b, mask), [2.0; 4]);
}

#[test]
fn cmpeq_i64x2_matches_scalar() {
	let Some(t) = Sse41::detect() else { return };
	assert_eq!(t.cmpeq_i64x2([5, -1], [5, 2]), [-1, 0]);
}

#[test]
fn select_i64x2_picks_b_where_mask_set() {
	let Some(t) = Sse41::detect() else { return };
	assert_eq!(t.select_i64x2([1, 2], [10, 20], [-1, 0]), [10, 2]);
}

#[test]
fn select_f64x2_uses_sign_bit_not_zero_test() {
	let Some(t) = Sse41::detect() else { return };
	let a = [1.0f64; 2];
	let b = [2.0f64; 2];
	let mask = [-0.0f64; 2];
	assert_eq!(t.select_f64x2(a, b, mask), [2.0; 2]);
}

// select_i32/u32: no shared slice_ternop test (macro mask `2` is outside all-0/1 domain).
slice_ternop_matches_scalar_test!(
	select_f32_slice_matches_scalar, Sse41, select_f32_slice,
	|a: f32, b: f32, m: f32| if m.is_sign_negative() { b } else { a }, f32
);
slice_ternop_matches_scalar_test!(
	select_f64_slice_matches_scalar, Sse41, select_f64_slice,
	|a: f64, b: f64, m: f64| if m.is_sign_negative() { b } else { a }, f64
);

#[test]
fn select_i8x16_picks_b_where_mask_set() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8);
	let b: [i8; 16] = core::array::from_fn(|i| 100 - i as i8);
	let mask: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(sse41.select_i8x16(a, b, mask), expect);
}

#[test]
fn select_i16x8_picks_b_where_mask_set() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	let b: [i16; 8] = core::array::from_fn(|i| 100 - i as i16);
	let mask: [i16; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i16; 8] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(sse41.select_i16x8(a, b, mask), expect);
}

// select_i8/u8/i16/u16: no shared slice_ternop test, same out-of-domain reason as select_i32.

#[test]
fn min_i8x16_picks_smaller_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8 - 8);
	let b: [i8; 16] = core::array::from_fn(|i| 7 - i as i8);
	let expect: [i8; 16] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(sse41.min_i8x16(a, b), expect);
}

#[test]
fn max_i8x16_picks_larger_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8 - 8);
	let b: [i8; 16] = core::array::from_fn(|i| 7 - i as i8);
	let expect: [i8; 16] = core::array::from_fn(|i| a[i].max(b[i]));
	assert_eq!(sse41.max_i8x16(a, b), expect);
}

#[test]
fn min_u16x8_picks_smaller_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [u16; 8] = [1, 20, 3, 4000, 5, 6, 7, 8];
	let b: [u16; 8] = [10, 2, 30, 4, 50, 6, 700, 8];
	let expect: [u16; 8] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(sse41.min_u16x8(a, b), expect);
}

#[test]
fn max_u16x8_picks_larger_lane() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [u16; 8] = [1, 20, 3, 4000, 5, 6, 7, 8];
	let b: [u16; 8] = [10, 2, 30, 4, 50, 6, 700, 8];
	let expect: [u16; 8] = core::array::from_fn(|i| a[i].max(b[i]));
	assert_eq!(sse41.max_u16x8(a, b), expect);
}

slice_binop_matches_scalar_test!(min_i8_slice_matches_scalar, Sse41, min_i8_slice, |x, y| x.min(y), i8);
slice_binop_matches_scalar_test!(max_i8_slice_matches_scalar, Sse41, max_i8_slice, |x, y| x.max(y), i8);
slice_binop_matches_scalar_test!(min_u16_slice_matches_scalar, Sse41, min_u16_slice, |x, y| x.min(y), u16);
slice_binop_matches_scalar_test!(max_u16_slice_matches_scalar, Sse41, max_u16_slice, |x, y| x.max(y), u16);

#[test]
fn round_f32x4_rounds_to_nearest_int() {
	let Some(sse41) = Sse41::detect() else { return };
	use core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	let a = [1.5f32, 2.5, -1.5, 0.4];
	assert_eq!(sse41.round_f32x4::<_MM_FROUND_TO_NEAREST_INT>(a), [2.0, 2.0, -2.0, 0.0]);
}

#[test]
fn round_f32x4_truncates_toward_zero() {
	let Some(sse41) = Sse41::detect() else { return };
	use core::arch::x86_64::_MM_FROUND_TO_ZERO;
	let a = [1.9f32, 2.1, -1.9, -2.1];
	assert_eq!(sse41.round_f32x4::<_MM_FROUND_TO_ZERO>(a), [1.0, 2.0, -1.0, -2.0]);
}

#[test]
fn round_f64x2_rounds_up_to_pos_inf() {
	let Some(sse41) = Sse41::detect() else { return };
	use core::arch::x86_64::_MM_FROUND_TO_POS_INF;
	let a = [1.1f64, -1.1];
	assert_eq!(sse41.round_f64x2::<_MM_FROUND_TO_POS_INF>(a), [2.0, -1.0]);
}

#[test]
fn blend_i16x8_picks_b_where_bit_set() {
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	let b: [i16; 8] = core::array::from_fn(|i| 100 + i as i16);
	// 0xfc = 0b1111_1100: lanes 0-1 from a, lanes 2-7 from b.
	assert_eq!(sse41.blend_i16x8::<0xfc>(a, b), [0, 1, 102, 103, 104, 105, 106, 107]);
	assert_eq!(sse41.blend_i16x8::<0x00>(a, b), a);
	assert_eq!(sse41.blend_i16x8::<0xff>(a, b), b);
}
