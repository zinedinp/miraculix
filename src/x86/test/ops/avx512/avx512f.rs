use super::super::super::macros::{
	slice_binop_matches_scalar_test, slice_bitop_matches_scalar_bits_test, slice_shift_imm_matches_scalar_test,
	slice_ternop_matches_scalar_test,
};
use super::*;

#[test]
fn add_f32x16_sums_lanes() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
	let b: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * 10.0);
	let expect: [f32; 16] = core::array::from_fn(|i| a[i] + b[i]);
	assert_eq!(v4.add_f32x16(a, b), expect);
}

// Hand-worked per Intel's per-128-bit-lane semantics (SDM `VUNPCKLPS`/
// `VUNPCKHPS`/`VSHUFPS`): a 512-bit op treats each of the four 128-bit
// sub-lanes (elements 0-3, 4-7, 8-11, 12-15) independently, mirroring
// the 256-bit `Avx::unpacklo_f32x8`/`unpackhi_f32x8`/`shuffle_f32x8` tests.
#[test]
fn unpacklo_f32x16_interleaves_per_lane() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [f32; 16] = core::array::from_fn(|i| 100.0 + i as f32);
	assert_eq!(
		v4.unpacklo_f32x16(a, b),
		[0.0, 100.0, 1.0, 101.0, 4.0, 104.0, 5.0, 105.0, 8.0, 108.0, 9.0, 109.0, 12.0, 112.0, 13.0, 113.0]
	);
}

#[test]
fn unpackhi_f32x16_interleaves_per_lane() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [f32; 16] = core::array::from_fn(|i| 100.0 + i as f32);
	assert_eq!(
		v4.unpackhi_f32x16(a, b),
		[2.0, 102.0, 3.0, 103.0, 6.0, 106.0, 7.0, 107.0, 10.0, 110.0, 11.0, 111.0, 14.0, 114.0, 15.0, 115.0]
	);
}

#[test]
fn shuffle_f32x16_selects_per_lane() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [f32; 16] = core::array::from_fn(|i| 100.0 + i as f32);
	// 0x44 = 0b01_00_01_00: dst = [a0, a1, b0, b1] per 128-bit lane.
	assert_eq!(
		v4.shuffle_f32x16::<0x44>(a, b),
		[0.0, 1.0, 100.0, 101.0, 4.0, 5.0, 104.0, 105.0, 8.0, 9.0, 108.0, 109.0, 12.0, 13.0, 112.0, 113.0]
	);
}

#[test]
fn sub_f32x16_subtracts_lanes() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * 10.0);
	let b: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
	let expect: [f32; 16] = core::array::from_fn(|i| a[i] - b[i]);
	assert_eq!(v4.sub_f32x16(a, b), expect);
}

#[test]
fn mul_f32x16_multiplies_lanes() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
	let b: [f32; 16] = [2.0; 16];
	let expect: [f32; 16] = core::array::from_fn(|i| a[i] * 2.0);
	assert_eq!(v4.mul_f32x16(a, b), expect);
}

#[test]
fn div_f32x16_divides_lanes() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * 10.0);
	let b: [f32; 16] = [2.0; 16];
	let expect: [f32; 16] = core::array::from_fn(|i| a[i] / 2.0);
	assert_eq!(v4.div_f32x16(a, b), expect);
}

#[test]
fn min_f32x16_picks_smaller_lane() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) });
	let b: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 });
	let expect: [f32; 16] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(v4.min_f32x16(a, b), expect);
}

#[test]
fn max_f32x16_picks_larger_lane() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) });
	let b: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 });
	let expect: [f32; 16] = core::array::from_fn(|i| a[i].max(b[i]));
	assert_eq!(v4.max_f32x16(a, b), expect);
}

/// Lanes match scalar add/sub/mul/div/min/max.
#[test]
fn matches_scalar_on_random_lanes() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] =
		[17.5, -3.25, 0.0, 1e6, -1e-3, 42.0, 8.25, -8.25, 1.0, -1.0, 100.0, -100.0, 0.5, -0.5, 3.0, -3.0];
	let b: [f32; 16] =
		[-240.75, 10.0, -3.5, 255.0, 1.0, -42.0, 8.25, 8.25, -1.0, 1.0, -100.0, 100.0, -0.5, 0.5, -3.0, 3.0];

	let expect_add: [f32; 16] = core::array::from_fn(|i| a[i] + b[i]);
	let expect_sub: [f32; 16] = core::array::from_fn(|i| a[i] - b[i]);
	let expect_mul: [f32; 16] = core::array::from_fn(|i| a[i] * b[i]);
	let expect_div: [f32; 16] = core::array::from_fn(|i| a[i] / b[i]);
	let expect_min: [f32; 16] = core::array::from_fn(|i| a[i].min(b[i]));
	let expect_max: [f32; 16] = core::array::from_fn(|i| a[i].max(b[i]));

	assert_eq!(v4.add_f32x16(a, b), expect_add);
	assert_eq!(v4.sub_f32x16(a, b), expect_sub);
	assert_eq!(v4.mul_f32x16(a, b), expect_mul);
	assert_eq!(v4.div_f32x16(a, b), expect_div);
	assert_eq!(v4.min_f32x16(a, b), expect_min);
	assert_eq!(v4.max_f32x16(a, b), expect_max);
}

slice_binop_matches_scalar_test!(add_f32_slice_matches_scalar_for_various_lengths, Avx512f, add_f32_slice, |x, y| x + y, f32);
slice_binop_matches_scalar_test!(sub_f32_slice_matches_scalar_for_various_lengths, Avx512f, sub_f32_slice, |x, y| x - y, f32);
slice_binop_matches_scalar_test!(mul_f32_slice_matches_scalar_for_various_lengths, Avx512f, mul_f32_slice, |x, y| x * y, f32);
slice_binop_matches_scalar_test!(div_f32_slice_matches_scalar_for_various_lengths, Avx512f, div_f32_slice, |x, y| x / y, f32);
slice_binop_matches_scalar_test!(min_f32_slice_matches_scalar_for_various_lengths, Avx512f, min_f32_slice, |x, y| x.min(y), f32);
slice_binop_matches_scalar_test!(max_f32_slice_matches_scalar_for_various_lengths, Avx512f, max_f32_slice, |x, y| x.max(y), f32);

slice_binop_matches_scalar_test!(add_f64_slice_matches_scalar, Avx512f, add_f64_slice, |x, y| x + y, f64);
slice_binop_matches_scalar_test!(sub_f64_slice_matches_scalar, Avx512f, sub_f64_slice, |x, y| x - y, f64);
slice_binop_matches_scalar_test!(mul_f64_slice_matches_scalar, Avx512f, mul_f64_slice, |x, y| x * y, f64);
slice_binop_matches_scalar_test!(div_f64_slice_matches_scalar, Avx512f, div_f64_slice, |x, y| x / y, f64);
slice_binop_matches_scalar_test!(min_f64_slice_matches_scalar, Avx512f, min_f64_slice, |x, y| x.min(y), f64);
slice_binop_matches_scalar_test!(max_f64_slice_matches_scalar, Avx512f, max_f64_slice, |x, y| x.max(y), f64);

#[test]
fn and_f32x16_masks_off_sign_bit() {
	let Some(v4) = Avx512f::detect() else { return };
	let a = [-1.5f32; 16];
	let b = [f32::from_bits(0x7fff_ffff); 16];
	assert_eq!(v4.and_f32x16(a, b), [1.5f32; 16]);
}

slice_bitop_matches_scalar_bits_test!(
	and_f32_slice_matches_scalar, Avx512f, and_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	or_f32_slice_matches_scalar, Avx512f, or_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	xor_f32_slice_matches_scalar, Avx512f, xor_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f32_slice_matches_scalar, Avx512f, andnot_f32_slice,
	|x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	and_f64_slice_matches_scalar, Avx512f, and_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	or_f64_slice_matches_scalar, Avx512f, or_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	xor_f64_slice_matches_scalar, Avx512f, xor_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f64_slice_matches_scalar, Avx512f, andnot_f64_slice,
	|x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()), f64
);

slice_binop_matches_scalar_test!(add_i32_slice_matches_scalar, Avx512f, add_i32_slice, |x: i32, y: i32| x.wrapping_add(y), i32);
slice_binop_matches_scalar_test!(sub_i32_slice_matches_scalar, Avx512f, sub_i32_slice, |x: i32, y: i32| x.wrapping_sub(y), i32);
slice_binop_matches_scalar_test!(mul_i32_slice_matches_scalar, Avx512f, mul_i32_slice, |x: i32, y: i32| x.wrapping_mul(y), i32);
slice_binop_matches_scalar_test!(min_i32_slice_matches_scalar, Avx512f, min_i32_slice, |x, y| x.min(y), i32);
slice_binop_matches_scalar_test!(max_i32_slice_matches_scalar, Avx512f, max_i32_slice, |x, y| x.max(y), i32);
slice_binop_matches_scalar_test!(div_i32_slice_matches_scalar, Avx512f, div_i32_slice, |x: i32, y: i32| x / y, i32);

slice_binop_matches_scalar_test!(add_u32_slice_matches_scalar, Avx512f, add_u32_slice, |x: u32, y: u32| x.wrapping_add(y), u32);
slice_binop_matches_scalar_test!(sub_u32_slice_matches_scalar, Avx512f, sub_u32_slice, |x: u32, y: u32| x.wrapping_sub(y), u32);
slice_binop_matches_scalar_test!(mul_u32_slice_matches_scalar, Avx512f, mul_u32_slice, |x: u32, y: u32| x.wrapping_mul(y), u32);
slice_binop_matches_scalar_test!(min_u32_slice_matches_scalar, Avx512f, min_u32_slice, |x, y| x.min(y), u32);
slice_binop_matches_scalar_test!(max_u32_slice_matches_scalar, Avx512f, max_u32_slice, |x, y| x.max(y), u32);
slice_binop_matches_scalar_test!(div_u32_slice_matches_scalar, Avx512f, div_u32_slice, |x: u32, y: u32| x / y, u32);

#[test]
fn add_i32x16_wraps_on_overflow() {
	let Some(v4) = Avx512f::detect() else { return };
	let mut a = [0i32; 16];
	let mut b = [0i32; 16];
	a[0] = i32::MAX;
	b[0] = 1;
	let mut expect = [0i32; 16];
	expect[0] = i32::MIN;
	assert_eq!(v4.add_i32x16(a, b), expect);
}

#[test]
fn mul_i32x16_wraps_on_overflow() {
	let Some(v4) = Avx512f::detect() else { return };
	let mut a = [0i32; 16];
	let mut b = [0i32; 16];
	a[0] = 1 << 30;
	b[0] = 4;
	let expect = [0i32; 16];
	assert_eq!(v4.mul_i32x16(a, b), expect);
}

slice_binop_matches_scalar_test!(and_i32_slice_matches_scalar, Avx512f, and_i32_slice, |x, y| x & y, i32);
slice_binop_matches_scalar_test!(xor_u32_slice_matches_scalar, Avx512f, xor_u32_slice, |x, y| x ^ y, u32);
slice_binop_matches_scalar_test!(
	cmpeq_i32_slice_matches_scalar, Avx512f, cmpeq_i32_slice,
	|x, y| if x == y { -1 } else { 0 }, i32
);
slice_shift_imm_matches_scalar_test!(
	shl_i32_slice_matches_scalar, Avx512f, shl_i32_slice, 3,
	|x: i32, imm| x.wrapping_shl(imm), i32
);
slice_ternop_matches_scalar_test!(
	fmadd_f32_slice_matches_scalar, Avx512f, fmadd_f32_slice, |a, b, c| a * b + c, f32
);

#[test]
fn add_i64x8_wraps_on_overflow() {
	let Some(v4) = Avx512f::detect() else { return };
	let mut a = [0i64; 8];
	let mut b = [0i64; 8];
	a[0] = i64::MAX;
	b[0] = 1;
	let mut expect = [0i64; 8];
	expect[0] = i64::MIN;
	assert_eq!(v4.add_i64x8(a, b), expect);
}

slice_binop_matches_scalar_test!(add_i64_slice_matches_scalar, Avx512f, add_i64_slice, |x: i64, y: i64| x.wrapping_add(y), i64);
slice_binop_matches_scalar_test!(sub_i64_slice_matches_scalar, Avx512f, sub_i64_slice, |x: i64, y: i64| x.wrapping_sub(y), i64);
slice_binop_matches_scalar_test!(min_i64_slice_matches_scalar, Avx512f, min_i64_slice, |x, y| x.min(y), i64);
slice_binop_matches_scalar_test!(max_i64_slice_matches_scalar, Avx512f, max_i64_slice, |x, y| x.max(y), i64);
slice_binop_matches_scalar_test!(add_u64_slice_matches_scalar, Avx512f, add_u64_slice, |x: u64, y: u64| x.wrapping_add(y), u64);
slice_binop_matches_scalar_test!(sub_u64_slice_matches_scalar, Avx512f, sub_u64_slice, |x: u64, y: u64| x.wrapping_sub(y), u64);
slice_binop_matches_scalar_test!(min_u64_slice_matches_scalar, Avx512f, min_u64_slice, |x, y| x.min(y), u64);
slice_binop_matches_scalar_test!(max_u64_slice_matches_scalar, Avx512f, max_u64_slice, |x, y| x.max(y), u64);

#[test]
fn abs_i64x8_wrapping_abs_of_min() {
	let Some(t) = Avx512f::detect() else { return };
	let mut a = [0i64; 8];
	a[0] = i64::MIN;
	a[1] = -12345;
	let mut expect = [0i64; 8];
	expect[0] = i64::MIN;
	expect[1] = 12345;
	assert_eq!(t.abs_i64x8(a), expect);
}

#[test]
fn abs_i64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 16, 17, 100] {
		let a: Vec<i64> = (0..len).map(|i| (i as i64 - len as i64 / 2) * 0x1_0000_0007).collect();
		let mut out = vec![0i64; len];
		t.abs_i64_slice(&a, &mut out);
		let expect: Vec<i64> = a.iter().map(|&x| x.wrapping_abs()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn cmpgt_u32x16_treats_high_bit_as_large_not_negative() {
	let Some(v4) = Avx512f::detect() else { return };
	let mut a = [0u32; 16];
	let mut b = [0u32; 16];
	a[0] = 0xFFFF_FFFF;
	b[0] = 0;
	let mut expect = [0u32; 16];
	expect[0] = !0;
	assert_eq!(v4.cmpgt_u32x16(a, b), expect);
}

slice_binop_matches_scalar_test!(
	cmpgt_i32_slice_matches_scalar, Avx512f, cmpgt_i32_slice,
	|x, y| if x > y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpgt_u32_slice_matches_scalar, Avx512f, cmpgt_u32_slice,
	|x, y| if x > y { !0 } else { 0 }, u32
);

slice_binop_matches_scalar_test!(
	cmplt_i32_slice_matches_scalar, Avx512f, cmplt_i32_slice,
	|x, y| if x < y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmple_i32_slice_matches_scalar, Avx512f, cmple_i32_slice,
	|x, y| if x <= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpge_i32_slice_matches_scalar, Avx512f, cmpge_i32_slice,
	|x, y| if x >= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmplt_u32_slice_matches_scalar, Avx512f, cmplt_u32_slice,
	|x, y| if x < y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmple_u32_slice_matches_scalar, Avx512f, cmple_u32_slice,
	|x, y| if x <= y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmpge_u32_slice_matches_scalar, Avx512f, cmpge_u32_slice,
	|x, y| if x >= y { !0 } else { 0 }, u32
);

#[test]
fn sllv_i32x16_shifts_by_the_count_vector() {
	let Some(v4) = Avx512f::detect() else { return };
	let a = [1i32; 16];
	let count: [i32; 16] = core::array::from_fn(|i| i as i32 * 2);
	let expect: [i32; 16] = core::array::from_fn(|i| {
		let c = count[i] as u32;
		if c >= 32 { 0 } else { 1i32.wrapping_shl(c) }
	});
	assert_eq!(v4.sllv_i32x16(a, count), expect);
}

#[test]
fn srav_i32x16_sign_fills_past_bit_width() {
	let Some(v4) = Avx512f::detect() else { return };
	let a = [-8i32; 16];
	let count: [i32; 16] = core::array::from_fn(|i| i as i32 * 2);
	let expect: [i32; 16] = core::array::from_fn(|i| {
		let c = count[i] as u32;
		if c >= 32 { -8i32 >> 31 } else { (-8i32).wrapping_shr(c) }
	});
	assert_eq!(v4.srav_i32x16(a, count), expect);
}

slice_binop_matches_scalar_test!(
	sllv_i32_slice_matches_scalar, Avx512f, sllv_i32_slice,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { x.wrapping_shl(count as u32) }, i32
);
slice_binop_matches_scalar_test!(
	srlv_i32_slice_matches_scalar, Avx512f, srlv_i32_slice,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { ((x as u32).wrapping_shr(count as u32)) as i32 }, i32
);
slice_binop_matches_scalar_test!(
	sllv_u32_slice_matches_scalar, Avx512f, sllv_u32_slice,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shl(count) }, u32
);
slice_binop_matches_scalar_test!(
	srlv_u32_slice_matches_scalar, Avx512f, srlv_u32_slice,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shr(count) }, u32
);

#[test]
fn select_i32x16_picks_b_where_mask_set() {
	let Some(v4) = Avx512f::detect() else { return };
	let mut a = [1i32; 16];
	let mut b = [10i32; 16];
	let mut mask = [0i32; 16];
	mask[0] = -1;
	a[0] = 1;
	b[0] = 99;
	let expect: [i32; 16] = core::array::from_fn(|i| if i == 0 { 99 } else { 1 });
	assert_eq!(v4.select_i32x16(a, b, mask), expect);
}

#[test]
fn select_f32x16_uses_sign_bit_not_zero_test() {
	let Some(v4) = Avx512f::detect() else { return };
	let a = [1.0f32; 16];
	let b = [2.0f32; 16];
	let mask_positive = [1.0f32; 16];
	let mask_negzero = [-0.0f32; 16];
	assert_eq!(v4.select_f32x16(a, b, mask_positive), a, "positive nonzero mask must NOT select b");
	assert_eq!(v4.select_f32x16(a, b, mask_negzero), b, "negative zero (sign bit set) must select b");
}
slice_ternop_matches_scalar_test!(
	fmsub_f32_slice_matches_scalar, Avx512f, fmsub_f32_slice, |a, b, c| a * b - c, f32
);
slice_ternop_matches_scalar_test!(
	fnmadd_f32_slice_matches_scalar, Avx512f, fnmadd_f32_slice, |a: f32, b: f32, c: f32| -(a * b) + c, f32
);
slice_ternop_matches_scalar_test!(
	fnmsub_f32_slice_matches_scalar, Avx512f, fnmsub_f32_slice, |a: f32, b: f32, c: f32| -(a * b) - c, f32
);

#[test]
fn p4fmadd_f32x16_matches_manual_four_way_accumulate() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [[f32; 16]; 4] = core::array::from_fn(|n| core::array::from_fn(|i| (n * 16 + i) as f32 - 20.0));
	let c = [3.0f32, -1.5, 0.0, 2.0];
	let expect: [f32; 16] =
		core::array::from_fn(|i| a[i] + b[0][i] * c[0] + b[1][i] * c[1] + b[2][i] * c[2] + b[3][i] * c[3]);
	assert_eq!(t.p4fmadd_f32x16(a, b, c), expect);
}

#[test]
fn p4fnmadd_f32x16_matches_manual_four_way_subtract() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [[f32; 16]; 4] = core::array::from_fn(|n| core::array::from_fn(|i| (n * 16 + i) as f32 - 20.0));
	let c = [3.0f32, -1.5, 0.0, 2.0];
	let expect: [f32; 16] =
		core::array::from_fn(|i| a[i] - b[0][i] * c[0] - b[1][i] * c[1] - b[2][i] * c[2] - b[3][i] * c[3]);
	assert_eq!(t.p4fnmadd_f32x16(a, b, c), expect);
}

#[test]
fn p4fmadd_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let b: [Vec<f32>; 4] =
			core::array::from_fn(|n| (0..len).map(|i| (n * len + i) as f32 * 0.25 - 10.0).collect());
		let c = [1.0f32, -2.0, 0.5, 4.0];
		let mut out = vec![0f32; len];
		t.p4fmadd_f32_slice(&a, [&b[0], &b[1], &b[2], &b[3]], c, &mut out);
		let expect: Vec<f32> = (0..len)
			.map(|i| a[i] + b[0][i] * c[0] + b[1][i] * c[1] + b[2][i] * c[2] + b[3][i] * c[3])
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4fnmadd_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let b: [Vec<f32>; 4] =
			core::array::from_fn(|n| (0..len).map(|i| (n * len + i) as f32 * 0.25 - 10.0).collect());
		let c = [1.0f32, -2.0, 0.5, 4.0];
		let mut out = vec![0f32; len];
		t.p4fnmadd_f32_slice(&a, [&b[0], &b[1], &b[2], &b[3]], c, &mut out);
		let expect: Vec<f32> = (0..len)
			.map(|i| a[i] - b[0][i] * c[0] - b[1][i] * c[1] - b[2][i] * c[2] - b[3][i] * c[3])
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4fmadd_f32_slice_panics_on_length_mismatch() {
	let Some(t) = Avx512f::detect() else { return };
	let a = [0f32; 4];
	let b0 = [0f32; 4];
	let b1 = [0f32; 3];
	let b2 = [0f32; 4];
	let b3 = [0f32; 4];
	let mut out = [0f32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.p4fmadd_f32_slice(&a, [&b0, &b1, &b2, &b3], [0.0; 4], &mut out);
	}));
	assert!(result.is_err());
}

macro_rules! masked_binop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $b:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a = $a;
			let b = $b;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

masked_binop_test!(
	add_f32x16_masked_matches_scalar, add_f32x16_merge_masked, add_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x + y
);
masked_binop_test!(
	sub_f32x16_masked_matches_scalar, sub_f32x16_merge_masked, sub_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32 * 10.0), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x - y
);
masked_binop_test!(
	mul_f32x16_masked_matches_scalar, mul_f32x16_merge_masked, mul_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x * y
);
masked_binop_test!(
	div_f32x16_masked_matches_scalar, div_f32x16_merge_masked, div_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32 * 10.0), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x / y
);
masked_binop_test!(
	min_f32x16_masked_matches_scalar, min_f32x16_merge_masked, min_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), core::array::from_fn::<f32, 16, _>(|i| (16 - i) as f32),
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x.min(y)
);
masked_binop_test!(
	max_f32x16_masked_matches_scalar, max_f32x16_merge_masked, max_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), core::array::from_fn::<f32, 16, _>(|i| (16 - i) as f32),
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |x: f32, y: f32| x.max(y)
);

masked_binop_test!(
	add_f64x8_masked_matches_scalar, add_f64x8_merge_masked, add_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x + y
);
masked_binop_test!(
	sub_f64x8_masked_matches_scalar, sub_f64x8_merge_masked, sub_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64 * 10.0), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x - y
);
masked_binop_test!(
	mul_f64x8_masked_matches_scalar, mul_f64x8_merge_masked, mul_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x * y
);
masked_binop_test!(
	div_f64x8_masked_matches_scalar, div_f64x8_merge_masked, div_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64 * 10.0), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x / y
);
masked_binop_test!(
	min_f64x8_masked_matches_scalar, min_f64x8_merge_masked, min_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), core::array::from_fn::<f64, 8, _>(|i| (8 - i) as f64),
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x.min(y)
);
masked_binop_test!(
	max_f64x8_masked_matches_scalar, max_f64x8_merge_masked, max_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), core::array::from_fn::<f64, 8, _>(|i| (8 - i) as f64),
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |x: f64, y: f64| x.max(y)
);

masked_binop_test!(
	add_i32x16_masked_matches_scalar, add_i32x16_merge_masked, add_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| i as i32 + 1), [7i32; 16],
	core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 100), 0x5555u16, |x: i32, y: i32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i32x16_masked_matches_scalar, sub_i32x16_merge_masked, sub_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| i as i32 + 100), [7i32; 16],
	core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 100), 0x5555u16, |x: i32, y: i32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_i32x16_masked_matches_scalar, mul_i32x16_merge_masked, mul_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| i as i32 + 1), [3i32; 16],
	core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 100), 0x5555u16, |x: i32, y: i32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_i32x16_masked_matches_scalar, min_i32x16_merge_masked, min_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| i as i32 + 1), core::array::from_fn::<i32, 16, _>(|i| 16 - i as i32),
	core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 100), 0x5555u16, |x: i32, y: i32| x.min(y)
);
masked_binop_test!(
	max_i32x16_masked_matches_scalar, max_i32x16_merge_masked, max_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| i as i32 + 1), core::array::from_fn::<i32, 16, _>(|i| 16 - i as i32),
	core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 100), 0x5555u16, |x: i32, y: i32| x.max(y)
);

masked_binop_test!(
	add_u32x16_masked_matches_scalar, add_u32x16_merge_masked, add_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1), [7u32; 16],
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1000), 0x5555u16, |x: u32, y: u32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u32x16_masked_matches_scalar, sub_u32x16_merge_masked, sub_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 100), [7u32; 16],
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1000), 0x5555u16, |x: u32, y: u32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_u32x16_masked_matches_scalar, mul_u32x16_merge_masked, mul_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1), [3u32; 16],
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1000), 0x5555u16, |x: u32, y: u32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_u32x16_masked_matches_scalar, min_u32x16_merge_masked, min_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1), core::array::from_fn::<u32, 16, _>(|i| 16 - i as u32),
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1000), 0x5555u16, |x: u32, y: u32| x.min(y)
);
masked_binop_test!(
	max_u32x16_masked_matches_scalar, max_u32x16_merge_masked, max_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1), core::array::from_fn::<u32, 16, _>(|i| 16 - i as u32),
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 1000), 0x5555u16, |x: u32, y: u32| x.max(y)
);

masked_binop_test!(
	add_i64x8_masked_matches_scalar, add_i64x8_merge_masked, add_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| i as i64 + 1), [7i64; 8],
	core::array::from_fn::<i64, 8, _>(|i| -(i as i64) - 100), 0x55u8, |x: i64, y: i64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i64x8_masked_matches_scalar, sub_i64x8_merge_masked, sub_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| i as i64 + 100), [7i64; 8],
	core::array::from_fn::<i64, 8, _>(|i| -(i as i64) - 100), 0x55u8, |x: i64, y: i64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_i64x8_masked_matches_scalar, min_i64x8_merge_masked, min_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| i as i64 + 1), core::array::from_fn::<i64, 8, _>(|i| 8 - i as i64),
	core::array::from_fn::<i64, 8, _>(|i| -(i as i64) - 100), 0x55u8, |x: i64, y: i64| x.min(y)
);
masked_binop_test!(
	max_i64x8_masked_matches_scalar, max_i64x8_merge_masked, max_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| i as i64 + 1), core::array::from_fn::<i64, 8, _>(|i| 8 - i as i64),
	core::array::from_fn::<i64, 8, _>(|i| -(i as i64) - 100), 0x55u8, |x: i64, y: i64| x.max(y)
);

masked_binop_test!(
	add_u64x8_masked_matches_scalar, add_u64x8_merge_masked, add_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1), [7u64; 8],
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1000), 0x55u8, |x: u64, y: u64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u64x8_masked_matches_scalar, sub_u64x8_merge_masked, sub_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 100), [7u64; 8],
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1000), 0x55u8, |x: u64, y: u64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_u64x8_masked_matches_scalar, min_u64x8_merge_masked, min_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1), core::array::from_fn::<u64, 8, _>(|i| 8 - i as u64),
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1000), 0x55u8, |x: u64, y: u64| x.min(y)
);
masked_binop_test!(
	max_u64x8_masked_matches_scalar, max_u64x8_merge_masked, max_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1), core::array::from_fn::<u64, 8, _>(|i| 8 - i as u64),
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 1000), 0x55u8, |x: u64, y: u64| x.max(y)
);

macro_rules! masked_unop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a = $a;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

masked_unop_test!(
	abs_i32x16_masked_matches_scalar, abs_i32x16_merge_masked, abs_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| (i as i32 - 8) * 3), core::array::from_fn::<i32, 16, _>(|i| -(i as i32) - 1000),
	0x5555u16, |x: i32| x.wrapping_abs()
);
masked_unop_test!(
	abs_i64x8_masked_matches_scalar, abs_i64x8_merge_masked, abs_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| (i as i64 - 4) * 3), core::array::from_fn::<i64, 8, _>(|i| -(i as i64) - 1000),
	0x55u8, |x: i64| x.wrapping_abs()
);

// compress/expand pack/unpack by output position, not per-lane: not a
// fit for `masked_unop_test!`'s per-lane `op(a[i])` shape.
macro_rules! masked_compress_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $width:literal, $Elem:ty, $a:expr, $src:expr, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $mask = $mask_val;
			let mut merge_expect = src;
			let mut zero_expect: [$Elem; $width] = [Default::default(); $width];
			let mut j = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					merge_expect[j] = a[i];
					zero_expect[j] = a[i];
					j += 1;
				}
			}
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

macro_rules! masked_expand_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $width:literal, $Elem:ty, $a:expr, $src:expr, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $mask = $mask_val;
			let mut merge_expect = src;
			let mut zero_expect: [$Elem; $width] = [Default::default(); $width];
			let mut k = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					merge_expect[i] = a[k];
					zero_expect[i] = a[k];
					k += 1;
				}
			}
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

masked_compress_test!(
	compress_i32x16_masked_packs_selected, compress_i32x16_merge_masked, compress_i32x16_zero_masked, u16, 16, i32,
	core::array::from_fn(|i| (i as i32) * 3 - 20), core::array::from_fn(|i| -(i as i32) - 1000), 0x9A37u16
);
masked_compress_test!(
	compress_u32x16_masked_packs_selected, compress_u32x16_merge_masked, compress_u32x16_zero_masked, u16, 16, u32,
	core::array::from_fn(|i| (i as u32) * 3 + 1), core::array::from_fn(|i| (i as u32) + 9000), 0x9A37u16
);
masked_compress_test!(
	compress_i64x8_masked_packs_selected, compress_i64x8_merge_masked, compress_i64x8_zero_masked, u8, 8, i64,
	core::array::from_fn(|i| (i as i64) * 3 - 10), core::array::from_fn(|i| -(i as i64) - 1000), 0xA7u8
);
masked_compress_test!(
	compress_u64x8_masked_packs_selected, compress_u64x8_merge_masked, compress_u64x8_zero_masked, u8, 8, u64,
	core::array::from_fn(|i| (i as u64) * 3 + 1), core::array::from_fn(|i| (i as u64) + 9000), 0xA7u8
);
masked_compress_test!(
	compress_f32x16_masked_packs_selected, compress_f32x16_merge_masked, compress_f32x16_zero_masked, u16, 16, f32,
	core::array::from_fn(|i| (i as f32) * 3.0 - 20.0), core::array::from_fn(|i| -(i as f32) - 1000.0), 0x9A37u16
);
masked_compress_test!(
	compress_f64x8_masked_packs_selected, compress_f64x8_merge_masked, compress_f64x8_zero_masked, u8, 8, f64,
	core::array::from_fn(|i| (i as f64) * 3.0 - 10.0), core::array::from_fn(|i| -(i as f64) - 1000.0), 0xA7u8
);

masked_expand_test!(
	expand_i32x16_masked_unpacks_selected, expand_i32x16_merge_masked, expand_i32x16_zero_masked, u16, 16, i32,
	core::array::from_fn(|i| (i as i32) * 3 - 20), core::array::from_fn(|i| -(i as i32) - 1000), 0x9A37u16
);
masked_expand_test!(
	expand_u32x16_masked_unpacks_selected, expand_u32x16_merge_masked, expand_u32x16_zero_masked, u16, 16, u32,
	core::array::from_fn(|i| (i as u32) * 3 + 1), core::array::from_fn(|i| (i as u32) + 9000), 0x9A37u16
);
masked_expand_test!(
	expand_i64x8_masked_unpacks_selected, expand_i64x8_merge_masked, expand_i64x8_zero_masked, u8, 8, i64,
	core::array::from_fn(|i| (i as i64) * 3 - 10), core::array::from_fn(|i| -(i as i64) - 1000), 0xA7u8
);
masked_expand_test!(
	expand_u64x8_masked_unpacks_selected, expand_u64x8_merge_masked, expand_u64x8_zero_masked, u8, 8, u64,
	core::array::from_fn(|i| (i as u64) * 3 + 1), core::array::from_fn(|i| (i as u64) + 9000), 0xA7u8
);
masked_expand_test!(
	expand_f32x16_masked_unpacks_selected, expand_f32x16_merge_masked, expand_f32x16_zero_masked, u16, 16, f32,
	core::array::from_fn(|i| (i as f32) * 3.0 - 20.0), core::array::from_fn(|i| -(i as f32) - 1000.0), 0x9A37u16
);
masked_expand_test!(
	expand_f64x8_masked_unpacks_selected, expand_f64x8_merge_masked, expand_f64x8_zero_masked, u8, 8, f64,
	core::array::from_fn(|i| (i as f64) * 3.0 - 10.0), core::array::from_fn(|i| -(i as f64) - 1000.0), 0xA7u8
);

macro_rules! masked_ternop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $b:expr, $c:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a = $a;
			let b = $b;
			let c = $c;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i], c[i]) } else { a[i] });
			assert_eq!(t.$merge_fn(a, mask, b, c), merge_expect, "merge");
			let zero_expect = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 { op(a[i], b[i], c[i]) } else { Default::default() }
			});
			assert_eq!(t.$zero_fn(mask, a, b, c), zero_expect, "zero");
		}
	};
}

masked_ternop_test!(
	fmadd_f32x16_masked_matches_scalar, fmadd_f32x16_merge_masked, fmadd_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |a: f32, b: f32, c: f32| a * b + c
);
masked_ternop_test!(
	fmadd_f64x8_masked_matches_scalar, fmadd_f64x8_merge_masked, fmadd_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |a: f64, b: f64, c: f64| a * b + c
);
masked_ternop_test!(
	fmsub_f32x16_masked_matches_scalar, fmsub_f32x16_merge_masked, fmsub_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |a: f32, b: f32, c: f32| a * b - c
);
masked_ternop_test!(
	fmsub_f64x8_masked_matches_scalar, fmsub_f64x8_merge_masked, fmsub_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |a: f64, b: f64, c: f64| a * b - c
);
masked_ternop_test!(
	fnmadd_f32x16_masked_matches_scalar, fnmadd_f32x16_merge_masked, fnmadd_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |a: f32, b: f32, c: f32| -(a * b) + c
);
masked_ternop_test!(
	fnmadd_f64x8_masked_matches_scalar, fnmadd_f64x8_merge_masked, fnmadd_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |a: f64, b: f64, c: f64| -(a * b) + c
);
masked_ternop_test!(
	fnmsub_f32x16_masked_matches_scalar, fnmsub_f32x16_merge_masked, fnmsub_f32x16_zero_masked, u16,
	core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32), [2.0f32; 16],
	core::array::from_fn::<f32, 16, _>(|i| -(i as f32) - 100.0), 0x5555u16, |a: f32, b: f32, c: f32| -(a * b) - c
);
masked_ternop_test!(
	fnmsub_f64x8_masked_matches_scalar, fnmsub_f64x8_merge_masked, fnmsub_f64x8_zero_masked, u8,
	core::array::from_fn::<f64, 8, _>(|i| (i + 1) as f64), [2.0f64; 8],
	core::array::from_fn::<f64, 8, _>(|i| -(i as f64) - 100.0), 0x55u8, |a: f64, b: f64, c: f64| -(a * b) - c
);

// Bit-exact reference for `vpternlogd`/`vpternlogq`: per-bit 3-input lookup
// into `imm8` (index = a_bit<<2 | b_bit<<1 | c_bit). Unlike the FMA masked
// ternop above, the merge form's first operand (`src`) is both a logic
// input *and* the merge fallback (matches `simd_binop_masked`'s `src`
// shape, not `masked_ternop_test!`'s `a`-doubles-as-fallback shape).
fn ternarylogic_ref_u32(a: u32, b: u32, c: u32, imm8: u8) -> u32 {
	let imm8 = imm8 as u32;
	let mut r = 0u32;
	for bit in 0u32..32 {
		let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
		r |= ((imm8 >> idx) & 1) << bit;
	}
	r
}

fn ternarylogic_ref_u64(a: u64, b: u64, c: u64, imm8: u8) -> u64 {
	let imm8 = imm8 as u64;
	let mut r = 0u64;
	for bit in 0u64..64 {
		let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
		r |= ((imm8 >> idx) & 1) << bit;
	}
	r
}

macro_rules! ternarylogic_test {
	(
		$name:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $width:literal,
		$Elem:ty, $Uns:ty, $ref:ident, $a:expr, $b:expr, $c:expr, $src:expr, $mask_val:expr, $imm8:literal
	) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a: [$Elem; $width] = $a;
			let b: [$Elem; $width] = $b;
			let c: [$Elem; $width] = $c;
			let src: [$Elem; $width] = $src;
			let mask: $mask = $mask_val;

			let fixed_expect: [$Elem; $width] =
				core::array::from_fn(|i| $ref(a[i] as $Uns, b[i] as $Uns, c[i] as $Uns, $imm8) as $Elem);
			assert_eq!(t.$fixed_fn::<$imm8>(a, b, c), fixed_expect, "unmasked");

			let merge_expect: [$Elem; $width] = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 {
					$ref(src[i] as $Uns, a[i] as $Uns, b[i] as $Uns, $imm8) as $Elem
				} else {
					src[i]
				}
			});
			assert_eq!(t.$merge_fn::<$imm8>(src, mask, a, b), merge_expect, "merge");

			let zero_expect: [$Elem; $width] = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 {
					$ref(a[i] as $Uns, b[i] as $Uns, c[i] as $Uns, $imm8) as $Elem
				} else {
					Default::default()
				}
			});
			assert_eq!(t.$zero_fn::<$imm8>(mask, a, b, c), zero_expect, "zero");
		}
	};
}

ternarylogic_test!(
	ternarylogic_i32x16_matches_bit_lookup, ternarylogic_i32x16, ternarylogic_i32x16_merge_masked,
	ternarylogic_i32x16_zero_masked, u16, 16, i32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as i32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as i32) * -0x0789_0ABC - 3),
	core::array::from_fn(|i| (i as i32) ^ 0x5A5A_5A5A_u32 as i32), core::array::from_fn(|i| -(i as i32) - 1000),
	0x9A37u16, 0x96
);
ternarylogic_test!(
	ternarylogic_u32x16_matches_bit_lookup, ternarylogic_u32x16, ternarylogic_u32x16_merge_masked,
	ternarylogic_u32x16_zero_masked, u16, 16, u32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as u32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as u32) * 0x0789_0ABC + 3),
	core::array::from_fn(|i| (i as u32) ^ 0x5A5A_5A5A), core::array::from_fn(|i| (i as u32) + 9000),
	0x9A37u16, 0xE8
);
ternarylogic_test!(
	ternarylogic_i64x8_matches_bit_lookup, ternarylogic_i64x8, ternarylogic_i64x8_merge_masked,
	ternarylogic_i64x8_zero_masked, u8, 8, i64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as i64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as i64) * -0x0000_789A_BCDE - 3),
	core::array::from_fn(|i| (i as i64) ^ 0x5A5A_5A5A_5A5A_5A5A_u64 as i64), core::array::from_fn(|i| -(i as i64) - 1000),
	0xA7u8, 0x2D
);
ternarylogic_test!(
	ternarylogic_u64x8_matches_bit_lookup, ternarylogic_u64x8, ternarylogic_u64x8_merge_masked,
	ternarylogic_u64x8_zero_masked, u8, 8, u64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as u64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as u64) * 0x0000_789A_BCDE + 3),
	core::array::from_fn(|i| (i as u64) ^ 0x5A5A_5A5A_5A5A_5A5A), core::array::from_fn(|i| (i as u64) + 9000),
	0xA7u8, 0x71
);

#[test]
fn f16_to_f32x16_converts_known_values() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let expect: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	assert_eq!(t.f16_to_f32x16(a), expect);
}

#[test]
fn f32_to_f16x16_converts_known_values() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	assert_eq!(t.f32_to_f16x16::<ROUNDING>(a), expect);
}

#[test]
fn f16_f32_roundtrip_matches_f16c_128_bit_form() {
	let Some(t) = Avx512f::detect() else { return };
	let Some(f16c) = super::super::super::avx::f16c::F16c::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 - 8.0));
	let got = t.f16_to_f32x16(a);
	let expect_lo = f16c.f16_to_f32x8(a[..8].try_into().unwrap());
	let expect_hi = f16c.f16_to_f32x8(a[8..].try_into().unwrap());
	assert_eq!(got[..8], expect_lo);
	assert_eq!(got[8..], expect_hi);
}

#[test]
fn f16_to_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 3, 15, 16, 17, 33, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 - (len as f32 / 2.0))).collect();
		let mut out = vec![0f32; len];
		t.f16_to_f32_slice(&a, &mut out);
		let expect: Vec<f32> = a.iter().map(|&x| f16_to_f32_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn f32_to_f16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	for len in [0usize, 1, 3, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 - (len as f32 / 2.0)).collect();
		let mut out = vec![0u16; len];
		t.f32_to_f16_slice::<ROUNDING>(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_f16_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

macro_rules! compressstoreu_test {
	($name:ident, $fixed_fn:ident, $Elem:ty, $width:literal, $Mask:ty, $mask_val:expr, $a:expr, $sentinel:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a: [$Elem; $width] = $a;
			let mask: $Mask = $mask_val;
			let popcount = mask.count_ones() as usize;
			let mut out = [$sentinel; $width];
			assert_eq!(t.$fixed_fn(&mut out, mask, a), popcount, "written count");
			let mut j = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					assert_eq!(out[j], a[i], "packed lane {j}");
					j += 1;
				}
			}
			for (k, o) in out[popcount..].iter().enumerate() {
				assert_eq!(*o, $sentinel, "tail lane {k} must be untouched");
			}
		}
	};
}

// Cross-checked against the register forms above rather than a hand-written
// expectation: same lane selection, only the operand source differs.
macro_rules! expandloadu_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $reg_merge_fn:ident, $reg_zero_fn:ident,
	 $Elem:ty, $width:literal, $Mask:ty, $mask_val:expr, $a:expr, $src:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512f::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $Mask = $mask_val;
			assert_eq!(t.$merge_fn(src, mask, &a), t.$reg_merge_fn(src, mask, a), "merge");
			assert_eq!(t.$zero_fn(mask, &a), t.$reg_zero_fn(mask, a), "zero");
			let n = mask.count_ones() as usize;
			assert_eq!(t.$merge_fn(src, mask, &a[..n]), t.$reg_merge_fn(src, mask, a), "exact-length mem");
		}
	};
}

compressstoreu_test!(
	compressstoreu_i32x16_packs_selected_lanes, compressstoreu_i32x16, i32, 16, u16,
	0x9A37u16, core::array::from_fn(|i| i as i32 * 7 + 1), -1i32
);
compressstoreu_test!(
	compressstoreu_u32x16_packs_selected_lanes, compressstoreu_u32x16, u32, 16, u16,
	0x9A37u16, core::array::from_fn(|i| i as u32 * 7 + 1), u32::MAX
);
compressstoreu_test!(
	compressstoreu_i64x8_packs_selected_lanes, compressstoreu_i64x8, i64, 8, u8,
	0xA7u8, core::array::from_fn(|i| i as i64 * 7 + 1), -1i64
);
compressstoreu_test!(
	compressstoreu_u64x8_packs_selected_lanes, compressstoreu_u64x8, u64, 8, u8,
	0xA7u8, core::array::from_fn(|i| i as u64 * 7 + 1), u64::MAX
);
compressstoreu_test!(
	compressstoreu_f32x16_packs_selected_lanes, compressstoreu_f32x16, f32, 16, u16,
	0x9A37u16, core::array::from_fn(|i| i as f32 * 1.5 + 0.25), -1.0f32
);
compressstoreu_test!(
	compressstoreu_f64x8_packs_selected_lanes, compressstoreu_f64x8, f64, 8, u8,
	0xA7u8, core::array::from_fn(|i| i as f64 * 1.5 + 0.25), -1.0f64
);

expandloadu_test!(
	expandloadu_i32x16_matches_register_form,
	expandloadu_i32x16_merge_masked, expandloadu_i32x16_zero_masked,
	expand_i32x16_merge_masked, expand_i32x16_zero_masked,
	i32, 16, u16, 0x9A37u16, core::array::from_fn(|i| i as i32 * 7 + 1), core::array::from_fn(|i| -(i as i32) - 100)
);
expandloadu_test!(
	expandloadu_u32x16_matches_register_form,
	expandloadu_u32x16_merge_masked, expandloadu_u32x16_zero_masked,
	expand_u32x16_merge_masked, expand_u32x16_zero_masked,
	u32, 16, u16, 0x9A37u16, core::array::from_fn(|i| i as u32 * 7 + 1), core::array::from_fn(|i| i as u32 + 1000)
);
expandloadu_test!(
	expandloadu_i64x8_matches_register_form,
	expandloadu_i64x8_merge_masked, expandloadu_i64x8_zero_masked,
	expand_i64x8_merge_masked, expand_i64x8_zero_masked,
	i64, 8, u8, 0xA7u8, core::array::from_fn(|i| i as i64 * 7 + 1), core::array::from_fn(|i| -(i as i64) - 100)
);
expandloadu_test!(
	expandloadu_u64x8_matches_register_form,
	expandloadu_u64x8_merge_masked, expandloadu_u64x8_zero_masked,
	expand_u64x8_merge_masked, expand_u64x8_zero_masked,
	u64, 8, u8, 0xA7u8, core::array::from_fn(|i| i as u64 * 7 + 1), core::array::from_fn(|i| i as u64 + 1000)
);
expandloadu_test!(
	expandloadu_f32x16_matches_register_form,
	expandloadu_f32x16_merge_masked, expandloadu_f32x16_zero_masked,
	expand_f32x16_merge_masked, expand_f32x16_zero_masked,
	f32, 16, u16, 0x9A37u16, core::array::from_fn(|i| i as f32 * 1.5 + 0.25), core::array::from_fn(|i| -(i as f32) - 100.0)
);
expandloadu_test!(
	expandloadu_f64x8_matches_register_form,
	expandloadu_f64x8_merge_masked, expandloadu_f64x8_zero_masked,
	expand_f64x8_merge_masked, expand_f64x8_zero_masked,
	f64, 8, u8, 0xA7u8, core::array::from_fn(|i| i as f64 * 1.5 + 0.25), core::array::from_fn(|i| -(i as f64) - 100.0)
);

// The popcount length assert is the safe-API bridge for the raw pointer
// forms; `catch_unwind` rather than `#[should_panic]` so the check stays
// skippable on a host without AVX-512F, like every other test here.
#[test]
fn compressstoreu_panics_when_out_is_shorter_than_popcount() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32);
	let mut out = [0i32; 3];
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.compressstoreu_i32x16(&mut out, u16::MAX, a);
	}));
	assert!(r.is_err());
}

#[test]
fn expandloadu_panics_when_mem_is_shorter_than_popcount() {
	let Some(t) = Avx512f::detect() else { return };
	let src: [i32; 16] = [0; 16];
	let mem = [1i32, 2, 3];
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.expandloadu_i32x16_merge_masked(src, u16::MAX, &mem)
	}));
	assert!(r.is_err());
}

#[test]
fn sqrt_f32x16_matches_scalar_sqrt() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * (i + 1) as f32);
	let expect: [f32; 16] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(v4.sqrt_f32x16(a), expect);
}

#[test]
fn sqrt_f64x8_matches_scalar_sqrt() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64 * (i + 1) as f64);
	let expect: [f64; 8] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(v4.sqrt_f64x8(a), expect);
}

// `vrcp14`/`vrsqrt14` are hardware approximations, max relative error <= 2^-14 per SDM.
const APPROX_TOL14_F32: f32 = 0.000_061_035_156; // 2^-14
const APPROX_TOL14_F64: f64 = 0.00006103515625; // 2^-14

#[test]
fn rcp14_f32x16_approximates_reciprocal() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
	let got = v4.rcp14_f32x16(a);
	for i in 0..16 {
		let expect = 1.0 / a[i];
		assert!(
			(got[i] - expect).abs() <= APPROX_TOL14_F32 * expect.abs(),
			"lane {i}: got {}, expect ~{expect}",
			got[i]
		);
	}
}

#[test]
fn rcp14_f64x8_approximates_reciprocal() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
	let got = v4.rcp14_f64x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i];
		assert!(
			(got[i] - expect).abs() <= APPROX_TOL14_F64 * expect.abs(),
			"lane {i}: got {}, expect ~{expect}",
			got[i]
		);
	}
}

#[test]
fn rsqrt14_f32x16_approximates_reciprocal_sqrt() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32 * (i + 1) as f32);
	let got = v4.rsqrt14_f32x16(a);
	for i in 0..16 {
		let expect = 1.0 / a[i].sqrt();
		assert!(
			(got[i] - expect).abs() <= APPROX_TOL14_F32 * expect.abs(),
			"lane {i}: got {}, expect ~{expect}",
			got[i]
		);
	}
}

#[test]
fn rsqrt14_f64x8_approximates_reciprocal_sqrt() {
	let Some(v4) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64 * (i + 1) as f64);
	let got = v4.rsqrt14_f64x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i].sqrt();
		assert!(
			(got[i] - expect).abs() <= APPROX_TOL14_F64 * expect.abs(),
			"lane {i}: got {}, expect ~{expect}",
			got[i]
		);
	}
}

#[test]
fn extract_u8x16_from_x64_picks_selected_quarter() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	let expect1: [u8; 16] = core::array::from_fn(|i| (16 + i) as u8);
	let expect3: [u8; 16] = core::array::from_fn(|i| (48 + i) as u8);
	assert_eq!(t.extract_u8x16_from_x64::<1>(a), expect1);
	assert_eq!(t.extract_u8x16_from_x64::<3>(a), expect3);
}

#[test]
fn insert_u8x16_into_x64_overwrites_selected_quarter() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	let b: [u8; 16] = core::array::from_fn(|i| 200 + i as u8);
	let mut expect = a;
	expect[16..32].copy_from_slice(&b);
	assert_eq!(t.insert_u8x16_into_x64::<1>(a, b), expect);
}

#[test]
fn extract_insert_u8x16_x64_roundtrip() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	let quarter = t.extract_u8x16_from_x64::<2>(a);
	let rebuilt = t.insert_u8x16_into_x64::<2>([0u8; 64], quarter);
	assert_eq!(&rebuilt[32..48], &a[32..48]);
}

#[test]
fn rol_ror_i32x16_match_scalar_rotate() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| (i as i32).wrapping_mul(0x1111_1111u32 as i32).wrapping_add(1));
	let expect_l: [i32; 16] = core::array::from_fn(|i| a[i].rotate_left(7));
	let expect_r: [i32; 16] = core::array::from_fn(|i| a[i].rotate_right(7));
	assert_eq!(t.rol_i32x16::<7>(a), expect_l);
	assert_eq!(t.ror_i32x16::<7>(a), expect_r);
}

#[test]
fn rol_ror_u64x8_match_scalar_rotate() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u64; 8] = core::array::from_fn(|i| (i as u64 + 1) * 0x0123_4567_89ab_cdef);
	let expect_l: [u64; 8] = core::array::from_fn(|i| a[i].rotate_left(19));
	let expect_r: [u64; 8] = core::array::from_fn(|i| a[i].rotate_right(19));
	assert_eq!(t.rol_u64x8::<19>(a), expect_l);
	assert_eq!(t.ror_u64x8::<19>(a), expect_r);
}

#[test]
fn reduce_add_mul_i32x16_match_scalar_fold() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32 - 8);
	assert_eq!(t.reduce_add_i32x16(a), a.iter().fold(0i32, |acc, &x| acc.wrapping_add(x)));
	let b: [i32; 16] = core::array::from_fn(|i| (i as i32 % 3) + 1);
	assert_eq!(t.reduce_mul_i32x16(b), b.iter().fold(1i32, |acc, &x| acc.wrapping_mul(x)));
}

#[test]
fn reduce_max_min_u32x16_match_scalar_fold() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u32; 16] = core::array::from_fn(|i| ((i * 37 + 5) % 251) as u32);
	assert_eq!(t.reduce_max_u32x16(a), *a.iter().max().unwrap());
	assert_eq!(t.reduce_min_u32x16(a), *a.iter().min().unwrap());
}

#[test]
fn reduce_add_f64x8_matches_scalar_sum() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i as f64 + 1.0) * 1.5);
	assert_eq!(t.reduce_add_f64x8(a), a.iter().sum::<f64>());
}

#[test]
fn widening_mul_u32x16_matches_scalar_full_product() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [u32; 16] = core::array::from_fn(|i| (i as u32 + 1).wrapping_mul(0x1000_0001));
	let b: [u32; 16] = core::array::from_fn(|i| 0xFFFF_FFFF - i as u32);
	let (lo, hi) = t.widening_mul_u32x16(a, b);
	for i in 0..16 {
		let full = a[i] as u64 * b[i] as u64;
		assert_eq!(lo[i], full as u32, "lo[{i}]");
		assert_eq!(hi[i], (full >> 32) as u32, "hi[{i}]");
	}
}

#[test]
fn widening_mul_i32x16_matches_scalar_full_product() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| (i as i32 - 8) * 12345);
	let b: [i32; 16] = core::array::from_fn(|i| i32::MIN + i as i32 * 999);
	let (lo, hi) = t.widening_mul_i32x16(a, b);
	for i in 0..16 {
		let full = a[i] as i64 * b[i] as i64;
		assert_eq!(lo[i], full as i32, "lo[{i}]");
		assert_eq!(hi[i], (full >> 32) as i32, "hi[{i}]");
	}
}

/// Scalar complex multiply oracle over an interleaved `[re, im, re, im, ...]` slice.
fn scalar_mul_c(a: &[f32], b: &[f32], conj: bool, out: &mut [f32]) {
	for i in (0..a.len()).step_by(2) {
		let (ar, ai) = (a[i], if conj { -a[i + 1] } else { a[i + 1] });
		let (br, bi) = (b[i], b[i + 1]);
		out[i] = ar * br - ai * bi;
		out[i + 1] = ar * bi + ai * br;
	}
}

#[test]
fn conj_c32x16_negates_imaginary_lanes() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	let got = t.conj_c32x16(a);
	for i in 0..16 {
		let expect = if i % 2 == 0 { a[i] } else { -a[i] };
		assert_eq!(got[i], expect, "lane {i}");
	}
}

#[test]
fn mul_c32x16_matches_scalar_complex_multiply() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 1.5);
	let b: [f32; 16] = core::array::from_fn(|i| (i as f32 * 0.5) - 3.0);
	let mut expect = [0f32; 16];
	scalar_mul_c(&a, &b, false, &mut expect);
	assert_eq!(t.mul_c32x16(a, b), expect);
}

#[test]
fn conj_mul_c32x16_matches_scalar_conjugate_multiply() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 1.5);
	let b: [f32; 16] = core::array::from_fn(|i| (i as f32 * 0.5) - 3.0);
	let mut expect = [0f32; 16];
	scalar_mul_c(&a, &b, true, &mut expect);
	assert_eq!(t.conj_mul_c32x16(a, b), expect);
	assert_eq!(t.conj_mul_c32x16(a, b), t.mul_c32x16(t.conj_c32x16(a), b));
}

#[test]
fn abs2_c32x16_matches_scalar_squared_magnitude() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 1.5);
	let got = t.abs2_c32x16(a);
	for i in (0..16).step_by(2) {
		let expect = a[i] * a[i] + a[i + 1] * a[i + 1];
		assert_eq!(got[i], expect, "re lane {i}");
		assert_eq!(got[i + 1], expect, "im lane {}", i + 1);
	}
}

#[test]
fn conj_c64x8_negates_imaginary_lanes() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.0);
	let got = t.conj_c64x8(a);
	for i in 0..8 {
		let expect = if i % 2 == 0 { a[i] } else { -a[i] };
		assert_eq!(got[i], expect, "lane {i}");
	}
}

#[test]
fn mul_c64x8_matches_scalar_complex_multiply() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i as f64 - 4.0) * 1.5);
	let b: [f64; 8] = core::array::from_fn(|i| (i as f64 * 0.5) - 2.0);
	let mut expect = [0f64; 8];
	for i in (0..8).step_by(2) {
		expect[i] = a[i] * b[i] - a[i + 1] * b[i + 1];
		expect[i + 1] = a[i] * b[i + 1] + a[i + 1] * b[i];
	}
	assert_eq!(t.mul_c64x8(a, b), expect);
}

#[test]
fn conj_mul_c64x8_matches_mul_with_conjugated_a() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i as f64 - 4.0) * 1.5);
	let b: [f64; 8] = core::array::from_fn(|i| (i as f64 * 0.5) - 2.0);
	assert_eq!(t.conj_mul_c64x8(a, b), t.mul_c64x8(t.conj_c64x8(a), b));
}

#[test]
fn abs2_c64x8_matches_scalar_squared_magnitude() {
	let Some(t) = Avx512f::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| (i as f64 - 4.0) * 1.5);
	let got = t.abs2_c64x8(a);
	for i in (0..8).step_by(2) {
		let expect = a[i] * a[i] + a[i + 1] * a[i + 1];
		assert_eq!(got[i], expect, "re lane {i}");
		assert_eq!(got[i + 1], expect, "im lane {}", i + 1);
	}
}

#[test]
fn partial_load_f32x16_zero_pads_and_caps_at_width() {
	let Some(t) = Avx512f::detect() else { return };
	let src: Vec<f32> = (1..=5).map(|i| i as f32).collect();
	let got = t.partial_load_f32x16(&src);
	assert_eq!(&got[..5], src.as_slice());
	assert_eq!(&got[5..], [0.0; 11]);

	let long: Vec<f32> = (1..=20).map(|i| i as f32).collect();
	let got_long = t.partial_load_f32x16(&long);
	assert_eq!(got_long, core::array::from_fn::<f32, 16, _>(|i| (i + 1) as f32));
}

#[test]
fn partial_store_f32x16_writes_only_slice_len_elements() {
	let Some(t) = Avx512f::detect() else { return };
	let v: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	let mut dst = vec![-1.0f32; 5];
	t.partial_store_f32x16(v, &mut dst);
	assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn partial_load_store_f32x16_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 7, 8, 15, 16] {
		let src: Vec<f32> = (0..len).map(|i| i as f32 * 1.5 - 3.0).collect();
		let v = t.partial_load_f32x16(&src);
		let mut dst = vec![f32::NAN; len];
		t.partial_store_f32x16(v, &mut dst);
		assert_eq!(dst, src, "len {len}");
	}
}

#[test]
fn partial_load_store_f64x8_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 1, 4, 8, 12] {
		let src: Vec<f64> = (0..len).map(|i| i as f64 * 2.5 - 1.0).collect();
		let v = t.partial_load_f64x8(&src);
		assert_eq!(&v[len.min(8)..], &[0.0; 8][len.min(8)..]);
		let mut dst = vec![f64::NAN; len.min(8)];
		t.partial_store_f64x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}

#[test]
fn partial_load_store_i32x16_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 3, 16, 20] {
		let src: Vec<i32> = (0..len).map(|i| i as i32 * -7 + 3).collect();
		let v = t.partial_load_i32x16(&src);
		let mut dst = vec![-1i32; len.min(16)];
		t.partial_store_i32x16(v, &mut dst);
		assert_eq!(dst, &src[..len.min(16)], "len {len}");
	}
}

#[test]
fn partial_load_store_u32x16_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 3, 16, 20] {
		let src: Vec<u32> = (0..len).map(|i| i as u32 * 7 + 3).collect();
		let v = t.partial_load_u32x16(&src);
		let mut dst = vec![u32::MAX; len.min(16)];
		t.partial_store_u32x16(v, &mut dst);
		assert_eq!(dst, &src[..len.min(16)], "len {len}");
	}
}

#[test]
fn partial_load_store_i64x8_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 3, 8, 10] {
		let src: Vec<i64> = (0..len).map(|i| i as i64 * -7 + 3).collect();
		let v = t.partial_load_i64x8(&src);
		let mut dst = vec![-1i64; len.min(8)];
		t.partial_store_i64x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}

#[test]
fn partial_load_store_u64x8_roundtrip_various_lengths() {
	let Some(t) = Avx512f::detect() else { return };
	for len in [0usize, 3, 8, 10] {
		let src: Vec<u64> = (0..len).map(|i| i as u64 * 7 + 3).collect();
		let v = t.partial_load_u64x8(&src);
		let mut dst = vec![u64::MAX; len.min(8)];
		t.partial_store_u64x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}
