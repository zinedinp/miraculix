use super::super::super::macros::{
	slice_binop_matches_scalar_test, slice_bitop_matches_scalar_bits_test, slice_shift_imm_matches_scalar_test,
};
#[cfg(feature = "wider-bus-lift")]
use super::super::super::macros::slice_binop_lifted_matches_scalar_test;
use super::*;

/// x86-64 psABI baseline: SSE2 always present.
#[test]
#[cfg(target_arch = "x86_64")]
fn detect_finds_sse2_on_x86_64() {
	assert!(Sse2::detect().is_some());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn assume_baseline_matches_detect() {
	let via_detect = Sse2::detect().expect("x86_64 always has SSE2");
	let via_baseline = Sse2::assume_baseline();
	assert_eq!(via_detect.add_i32x4([1, 2, 3, 4], [1, 1, 1, 1]), via_baseline.add_i32x4([1, 2, 3, 4], [1, 1, 1, 1]));
}

#[test]
fn add_i32x4_sums_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.add_i32x4([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 33, 44]);
}

#[test]
fn sqrt_f64x2_matches_scalar_sqrt() {
	let Some(sse2) = Sse2::detect() else { return };
	let a = [4.0f64, 9.0];
	let expect: [f64; 2] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(sse2.sqrt_f64x2(a), expect);
}

#[test]
fn add_i32x4_wraps_on_overflow() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.add_i32x4([i32::MAX, 0, 0, 0], [1, 0, 0, 0]), [i32::MIN, 0, 0, 0]);
}

#[test]
fn sub_i32x4_subtracts_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.sub_i32x4([10, 20, 30, 40], [1, 2, 3, 4]), [9, 18, 27, 36]);
}

#[test]
fn sub_i32x4_wraps_on_underflow() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.sub_i32x4([i32::MIN, 0, 0, 0], [1, 0, 0, 0]), [i32::MAX, 0, 0, 0]);
}

#[test]
fn add_u32x4_wraps_on_overflow() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.add_u32x4([u32::MAX, 0, 0, 0], [1, 0, 0, 0]), [0, 0, 0, 0]);
}

#[test]
fn sub_u32x4_wraps_on_underflow() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.sub_u32x4([0, 0, 0, 0], [1, 0, 0, 0]), [u32::MAX, 0, 0, 0]);
}

#[test]
fn div_i32x4_divides_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.div_i32x4([10, -20, 30, 7], [2, 4, 5, 2]), [5, -5, 6, 3]);
}

#[test]
fn div_u32x4_divides_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.div_u32x4([10, 20, 30, 7], [2, 4, 5, 2]), [5, 5, 6, 3]);
}

#[test]
fn add_f64x2_sums_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.add_f64x2([1.5, -3.25], [10.0, 20.0]), [11.5, 16.75]);
}

#[test]
fn sub_f64x2_subtracts_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.sub_f64x2([10.0, 20.0], [1.5, 3.25]), [8.5, 16.75]);
}

#[test]
fn mul_f64x2_multiplies_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.mul_f64x2([1.5, -2.0], [2.0, 3.0]), [3.0, -6.0]);
}

#[test]
fn div_f64x2_divides_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.div_f64x2([9.0, -6.0], [2.0, 3.0]), [4.5, -2.0]);
}

#[test]
fn min_f64x2_picks_smaller_lane() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.min_f64x2([1.0, 20.0], [10.0, 2.0]), [1.0, 2.0]);
}

#[test]
fn max_f64x2_picks_larger_lane() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.max_f64x2([1.0, 20.0], [10.0, 2.0]), [10.0, 20.0]);
}

slice_binop_matches_scalar_test!(add_f64_slice_matches_scalar, Sse2, add_f64_slice, |x, y| x + y, f64);
slice_binop_matches_scalar_test!(sub_f64_slice_matches_scalar, Sse2, sub_f64_slice, |x, y| x - y, f64);
slice_binop_matches_scalar_test!(mul_f64_slice_matches_scalar, Sse2, mul_f64_slice, |x, y| x * y, f64);
slice_binop_matches_scalar_test!(div_f64_slice_matches_scalar, Sse2, div_f64_slice, |x, y| x / y, f64);
slice_binop_matches_scalar_test!(min_f64_slice_matches_scalar, Sse2, min_f64_slice, |x, y| x.min(y), f64);
slice_binop_matches_scalar_test!(max_f64_slice_matches_scalar, Sse2, max_f64_slice, |x, y| x.max(y), f64);

#[test]
fn and_f64x2_masks_off_sign_bit() {
	let Some(sse2) = Sse2::detect() else { return };
	let a = [-1.5f64; 2];
	let b = [f64::from_bits(0x7fff_ffff_ffff_ffff); 2];
	assert_eq!(sse2.and_f64x2(a, b), [1.5f64; 2]);
}

slice_bitop_matches_scalar_bits_test!(
	and_f64_slice_matches_scalar, Sse2, and_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	or_f64_slice_matches_scalar, Sse2, or_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	xor_f64_slice_matches_scalar, Sse2, xor_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f64_slice_matches_scalar, Sse2, andnot_f64_slice,
	|x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()), f64
);

slice_binop_matches_scalar_test!(add_i32_slice_matches_scalar, Sse2, add_i32_slice, |x: i32, y: i32| x.wrapping_add(y), i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(add_i32_slice_lifted_matches_scalar, Sse2, Avx, add_i32_slice_lifted, |x: i32, y: i32| x.wrapping_add(y), i32);
slice_binop_matches_scalar_test!(sub_i32_slice_matches_scalar, Sse2, sub_i32_slice, |x: i32, y: i32| x.wrapping_sub(y), i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(sub_i32_slice_lifted_matches_scalar, Sse2, Avx, sub_i32_slice_lifted, |x: i32, y: i32| x.wrapping_sub(y), i32);
slice_binop_matches_scalar_test!(div_i32_slice_matches_scalar, Sse2, div_i32_slice, |x: i32, y: i32| x / y, i32);
slice_binop_matches_scalar_test!(min_i32_slice_matches_scalar, Sse2, min_i32_slice, |x, y| x.min(y), i32);
slice_binop_matches_scalar_test!(max_i32_slice_matches_scalar, Sse2, max_i32_slice, |x, y| x.max(y), i32);

slice_binop_matches_scalar_test!(add_u32_slice_matches_scalar, Sse2, add_u32_slice, |x: u32, y: u32| x.wrapping_add(y), u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(add_u32_slice_lifted_matches_scalar, Sse2, Avx, add_u32_slice_lifted, |x: u32, y: u32| x.wrapping_add(y), u32);
slice_binop_matches_scalar_test!(sub_u32_slice_matches_scalar, Sse2, sub_u32_slice, |x: u32, y: u32| x.wrapping_sub(y), u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(sub_u32_slice_lifted_matches_scalar, Sse2, Avx, sub_u32_slice_lifted, |x: u32, y: u32| x.wrapping_sub(y), u32);
slice_binop_matches_scalar_test!(div_u32_slice_matches_scalar, Sse2, div_u32_slice, |x: u32, y: u32| x / y, u32);
slice_binop_matches_scalar_test!(min_u32_slice_matches_scalar, Sse2, min_u32_slice, |x, y| x.min(y), u32);
slice_binop_matches_scalar_test!(max_u32_slice_matches_scalar, Sse2, max_u32_slice, |x, y| x.max(y), u32);

slice_binop_matches_scalar_test!(and_i32_slice_matches_scalar, Sse2, and_i32_slice, |x, y| x & y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(and_i32_slice_lifted_matches_scalar, Sse2, Avx, and_i32_slice_lifted, |x, y| x & y, i32);
slice_binop_matches_scalar_test!(or_i32_slice_matches_scalar, Sse2, or_i32_slice, |x, y| x | y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(or_i32_slice_lifted_matches_scalar, Sse2, Avx, or_i32_slice_lifted, |x, y| x | y, i32);
slice_binop_matches_scalar_test!(xor_i32_slice_matches_scalar, Sse2, xor_i32_slice, |x, y| x ^ y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(xor_i32_slice_lifted_matches_scalar, Sse2, Avx, xor_i32_slice_lifted, |x, y| x ^ y, i32);
slice_binop_matches_scalar_test!(andnot_i32_slice_matches_scalar, Sse2, andnot_i32_slice, |x, y| !x & y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(andnot_i32_slice_lifted_matches_scalar, Sse2, Avx, andnot_i32_slice_lifted, |x, y| !x & y, i32);
slice_binop_matches_scalar_test!(
	cmpeq_i32_slice_matches_scalar, Sse2, cmpeq_i32_slice,
	|x, y| if x == y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(and_u32_slice_matches_scalar, Sse2, and_u32_slice, |x, y| x & y, u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(and_u32_slice_lifted_matches_scalar, Sse2, Avx, and_u32_slice_lifted, |x, y| x & y, u32);
slice_binop_matches_scalar_test!(or_u32_slice_matches_scalar, Sse2, or_u32_slice, |x, y| x | y, u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(or_u32_slice_lifted_matches_scalar, Sse2, Avx, or_u32_slice_lifted, |x, y| x | y, u32);
slice_binop_matches_scalar_test!(xor_u32_slice_matches_scalar, Sse2, xor_u32_slice, |x, y| x ^ y, u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(xor_u32_slice_lifted_matches_scalar, Sse2, Avx, xor_u32_slice_lifted, |x, y| x ^ y, u32);
slice_binop_matches_scalar_test!(andnot_u32_slice_matches_scalar, Sse2, andnot_u32_slice, |x, y| !x & y, u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(andnot_u32_slice_lifted_matches_scalar, Sse2, Avx, andnot_u32_slice_lifted, |x, y| !x & y, u32);
slice_binop_matches_scalar_test!(
	cmpeq_u32_slice_matches_scalar, Sse2, cmpeq_u32_slice,
	|x, y| if x == y { !0 } else { 0 }, u32
);

#[test]
fn add_i64x2_wraps_on_overflow() {
	let Some(t) = Sse2::detect() else { return };
	let a = [i64::MAX, 0];
	let b = [1, 0];
	assert_eq!(t.add_i64x2(a, b), [i64::MIN, 0]);
}

slice_binop_matches_scalar_test!(add_i64_slice_matches_scalar, Sse2, add_i64_slice, |x: i64, y: i64| x.wrapping_add(y), i64);
slice_binop_matches_scalar_test!(sub_i64_slice_matches_scalar, Sse2, sub_i64_slice, |x: i64, y: i64| x.wrapping_sub(y), i64);
slice_binop_matches_scalar_test!(add_u64_slice_matches_scalar, Sse2, add_u64_slice, |x: u64, y: u64| x.wrapping_add(y), u64);
slice_binop_matches_scalar_test!(sub_u64_slice_matches_scalar, Sse2, sub_u64_slice, |x: u64, y: u64| x.wrapping_sub(y), u64);

#[test]
fn mullo_u64x2_matches_scalar_for_cross_term_carry_values() {
	let Some(t) = Sse2::detect() else { return };
	// Both a_hi and b_hi nonzero, and the a_lo*b_hi+a_hi*b_lo cross term itself
	// overflows 32 bits: exercises every term of the schoolbook decomposition.
	let a = [0xFFFF_FFFF_FFFF_FFFFu64, 0x1_0000_0002];
	let b = [0xFFFF_FFFF_FFFF_FFFFu64, 0x1_0000_0003];
	let expect = [a[0].wrapping_mul(b[0]), a[1].wrapping_mul(b[1])];
	assert_eq!(t.mullo_u64x2(a, b), expect);
}

#[test]
fn mullo_i64x2_matches_scalar_for_negative_values() {
	let Some(t) = Sse2::detect() else { return };
	let a = [i64::MIN, -12345];
	let b = [-1i64, 6789];
	let expect = [a[0].wrapping_mul(b[0]), a[1].wrapping_mul(b[1])];
	assert_eq!(t.mullo_i64x2(a, b), expect);
}

#[test]
fn mullo_u64_slice_matches_scalar_for_various_lengths_and_large_values() {
	let Some(t) = Sse2::detect() else { return };
	for len in [0usize, 1, 2, 3, 5, 9] {
		let a: Vec<u64> = (0..len).map(|i| u64::MAX - i as u64 * 3).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64 + 1).wrapping_mul(0x1_0000_0001)).collect();
		let mut out = vec![0u64; len];
		t.mullo_u64_slice(&a, &b, &mut out);
		let expect: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_mul(y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

slice_binop_matches_scalar_test!(mullo_i64_slice_matches_scalar, Sse2, mullo_i64_slice, |x: i64, y: i64| x.wrapping_mul(y), i64);

#[test]
fn abs_i64x2_matches_scalar_including_i64_min() {
	let Some(t) = Sse2::detect() else { return };
	assert_eq!(t.abs_i64x2([i64::MIN, -42]), [i64::MIN, 42]);
	assert_eq!(t.abs_i64x2([7, 0]), [7, 0]);
}

#[test]
fn abs_i64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Sse2::detect() else { return };
	for len in [0usize, 1, 2, 3, 5, 9] {
		let a: Vec<i64> = (0..len).map(|i| (i as i64 - len as i64 / 2) * 0x1_0000_0007).collect();
		let mut out = vec![0i64; len];
		t.abs_i64_slice(&a, &mut out);
		let expect: Vec<i64> = a.iter().map(|&x| x.wrapping_abs()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn cmpgt_u32x4_treats_high_bit_as_large_not_negative() {
	let Some(t) = Sse2::detect() else { return };
	let a = [0xFFFF_FFFFu32, 0, 5, 5];
	let b = [0u32, 0xFFFF_FFFF, 4, 5];
	assert_eq!(t.cmpgt_u32x4(a, b), [!0, 0, !0, 0]);
}

slice_binop_matches_scalar_test!(
	cmpgt_i32_slice_matches_scalar, Sse2, cmpgt_i32_slice,
	|x, y| if x > y { -1 } else { 0 }, i32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	cmpgt_i32_slice_lifted_matches_scalar, Sse2, Avx, cmpgt_i32_slice_lifted,
	|x, y| if x > y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpgt_u32_slice_matches_scalar, Sse2, cmpgt_u32_slice,
	|x, y| if x > y { !0 } else { 0 }, u32
);

slice_binop_matches_scalar_test!(
	cmplt_i32_slice_matches_scalar, Sse2, cmplt_i32_slice,
	|x, y| if x < y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmple_i32_slice_matches_scalar, Sse2, cmple_i32_slice,
	|x, y| if x <= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpge_i32_slice_matches_scalar, Sse2, cmpge_i32_slice,
	|x, y| if x >= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmplt_u32_slice_matches_scalar, Sse2, cmplt_u32_slice,
	|x, y| if x < y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmple_u32_slice_matches_scalar, Sse2, cmple_u32_slice,
	|x, y| if x <= y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmpge_u32_slice_matches_scalar, Sse2, cmpge_u32_slice,
	|x, y| if x >= y { !0 } else { 0 }, u32
);

slice_shift_imm_matches_scalar_test!(
	shl_i32_slice_matches_scalar, Sse2, shl_i32_slice, 3,
	|x: i32, imm| x.wrapping_shl(imm), i32
);
slice_shift_imm_matches_scalar_test!(
	shr_i32_slice_matches_scalar, Sse2, shr_i32_slice, 2,
	|x: i32, imm| ((x as u32).wrapping_shr(imm)) as i32, i32
);
slice_shift_imm_matches_scalar_test!(
	sra_i32_slice_matches_scalar, Sse2, sra_i32_slice, 1,
	|x: i32, imm| x.wrapping_shr(imm), i32
);
slice_shift_imm_matches_scalar_test!(
	shl_u32_slice_matches_scalar, Sse2, shl_u32_slice, 3,
	|x: u32, imm| x.wrapping_shl(imm), u32
);
slice_shift_imm_matches_scalar_test!(
	shr_u32_slice_matches_scalar, Sse2, shr_u32_slice, 2,
	|x: u32, imm| x.wrapping_shr(imm), u32
);

slice_binop_matches_scalar_test!(add_i8_slice_matches_scalar, Sse2, add_i8_slice, |x: i8, y: i8| x.wrapping_add(y), i8);
slice_binop_matches_scalar_test!(sub_i8_slice_matches_scalar, Sse2, sub_i8_slice, |x: i8, y: i8| x.wrapping_sub(y), i8);
slice_binop_matches_scalar_test!(adds_i8_slice_matches_scalar, Sse2, adds_i8_slice, |x: i8, y: i8| x.saturating_add(y), i8);
slice_binop_matches_scalar_test!(subs_i8_slice_matches_scalar, Sse2, subs_i8_slice, |x: i8, y: i8| x.saturating_sub(y), i8);
slice_binop_matches_scalar_test!(
	cmpeq_i8_slice_matches_scalar, Sse2, cmpeq_i8_slice, |x, y| if x == y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmpgt_i8_slice_matches_scalar, Sse2, cmpgt_i8_slice, |x, y| if x > y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmplt_i8_slice_matches_scalar, Sse2, cmplt_i8_slice, |x, y| if x < y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmple_i8_slice_matches_scalar, Sse2, cmple_i8_slice, |x, y| if x <= y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmpge_i8_slice_matches_scalar, Sse2, cmpge_i8_slice, |x, y| if x >= y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(and_i8_slice_matches_scalar, Sse2, and_i8_slice, |x, y| x & y, i8);
slice_binop_matches_scalar_test!(or_i8_slice_matches_scalar, Sse2, or_i8_slice, |x, y| x | y, i8);
slice_binop_matches_scalar_test!(xor_i8_slice_matches_scalar, Sse2, xor_i8_slice, |x, y| x ^ y, i8);
slice_binop_matches_scalar_test!(andnot_i8_slice_matches_scalar, Sse2, andnot_i8_slice, |x, y| !x & y, i8);
slice_binop_matches_scalar_test!(mul_i8_slice_matches_scalar, Sse2, mul_i8_slice, |x: i8, y: i8| x.wrapping_mul(y), i8);

slice_binop_matches_scalar_test!(add_u8_slice_matches_scalar, Sse2, add_u8_slice, |x: u8, y: u8| x.wrapping_add(y), u8);
slice_binop_matches_scalar_test!(sub_u8_slice_matches_scalar, Sse2, sub_u8_slice, |x: u8, y: u8| x.wrapping_sub(y), u8);
slice_binop_matches_scalar_test!(adds_u8_slice_matches_scalar, Sse2, adds_u8_slice, |x: u8, y: u8| x.saturating_add(y), u8);
slice_binop_matches_scalar_test!(subs_u8_slice_matches_scalar, Sse2, subs_u8_slice, |x: u8, y: u8| x.saturating_sub(y), u8);
slice_binop_matches_scalar_test!(
	cmpeq_u8_slice_matches_scalar, Sse2, cmpeq_u8_slice, |x, y| if x == y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(
	cmpgt_u8_slice_matches_scalar, Sse2, cmpgt_u8_slice, |x, y| if x > y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(
	cmple_u8_slice_matches_scalar, Sse2, cmple_u8_slice, |x, y| if x <= y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(and_u8_slice_matches_scalar, Sse2, and_u8_slice, |x, y| x & y, u8);
slice_binop_matches_scalar_test!(or_u8_slice_matches_scalar, Sse2, or_u8_slice, |x, y| x | y, u8);
slice_binop_matches_scalar_test!(xor_u8_slice_matches_scalar, Sse2, xor_u8_slice, |x, y| x ^ y, u8);
slice_binop_matches_scalar_test!(andnot_u8_slice_matches_scalar, Sse2, andnot_u8_slice, |x, y| !x & y, u8);
slice_binop_matches_scalar_test!(min_u8_slice_matches_scalar, Sse2, min_u8_slice, |x, y| x.min(y), u8);
slice_binop_matches_scalar_test!(max_u8_slice_matches_scalar, Sse2, max_u8_slice, |x, y| x.max(y), u8);
slice_binop_matches_scalar_test!(mul_u8_slice_matches_scalar, Sse2, mul_u8_slice, |x: u8, y: u8| x.wrapping_mul(y), u8);
slice_binop_matches_scalar_test!(
	avg_u8_slice_matches_scalar, Sse2, avg_u8_slice, |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8, u8
);

slice_binop_matches_scalar_test!(add_i16_slice_matches_scalar, Sse2, add_i16_slice, |x: i16, y: i16| x.wrapping_add(y), i16);
slice_binop_matches_scalar_test!(sub_i16_slice_matches_scalar, Sse2, sub_i16_slice, |x: i16, y: i16| x.wrapping_sub(y), i16);
slice_binop_matches_scalar_test!(adds_i16_slice_matches_scalar, Sse2, adds_i16_slice, |x: i16, y: i16| x.saturating_add(y), i16);
slice_binop_matches_scalar_test!(subs_i16_slice_matches_scalar, Sse2, subs_i16_slice, |x: i16, y: i16| x.saturating_sub(y), i16);
slice_binop_matches_scalar_test!(
	cmpeq_i16_slice_matches_scalar, Sse2, cmpeq_i16_slice, |x, y| if x == y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmpgt_i16_slice_matches_scalar, Sse2, cmpgt_i16_slice, |x, y| if x > y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmplt_i16_slice_matches_scalar, Sse2, cmplt_i16_slice, |x, y| if x < y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmpge_i16_slice_matches_scalar, Sse2, cmpge_i16_slice, |x, y| if x >= y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(and_i16_slice_matches_scalar, Sse2, and_i16_slice, |x, y| x & y, i16);
slice_binop_matches_scalar_test!(or_i16_slice_matches_scalar, Sse2, or_i16_slice, |x, y| x | y, i16);
slice_binop_matches_scalar_test!(xor_i16_slice_matches_scalar, Sse2, xor_i16_slice, |x, y| x ^ y, i16);
slice_binop_matches_scalar_test!(andnot_i16_slice_matches_scalar, Sse2, andnot_i16_slice, |x, y| !x & y, i16);
slice_binop_matches_scalar_test!(min_i16_slice_matches_scalar, Sse2, min_i16_slice, |x, y| x.min(y), i16);
slice_binop_matches_scalar_test!(max_i16_slice_matches_scalar, Sse2, max_i16_slice, |x, y| x.max(y), i16);
slice_binop_matches_scalar_test!(mul_i16_slice_matches_scalar, Sse2, mul_i16_slice, |x: i16, y: i16| x.wrapping_mul(y), i16);

slice_binop_matches_scalar_test!(add_u16_slice_matches_scalar, Sse2, add_u16_slice, |x: u16, y: u16| x.wrapping_add(y), u16);
slice_binop_matches_scalar_test!(sub_u16_slice_matches_scalar, Sse2, sub_u16_slice, |x: u16, y: u16| x.wrapping_sub(y), u16);
slice_binop_matches_scalar_test!(adds_u16_slice_matches_scalar, Sse2, adds_u16_slice, |x: u16, y: u16| x.saturating_add(y), u16);
slice_binop_matches_scalar_test!(subs_u16_slice_matches_scalar, Sse2, subs_u16_slice, |x: u16, y: u16| x.saturating_sub(y), u16);
slice_binop_matches_scalar_test!(
	cmpeq_u16_slice_matches_scalar, Sse2, cmpeq_u16_slice, |x, y| if x == y { !0 } else { 0 }, u16
);
slice_binop_matches_scalar_test!(
	cmpgt_u16_slice_matches_scalar, Sse2, cmpgt_u16_slice, |x, y| if x > y { !0 } else { 0 }, u16
);
slice_binop_matches_scalar_test!(
	cmple_u16_slice_matches_scalar, Sse2, cmple_u16_slice, |x, y| if x <= y { !0 } else { 0 }, u16
);
slice_binop_matches_scalar_test!(and_u16_slice_matches_scalar, Sse2, and_u16_slice, |x, y| x & y, u16);
slice_binop_matches_scalar_test!(or_u16_slice_matches_scalar, Sse2, or_u16_slice, |x, y| x | y, u16);
slice_binop_matches_scalar_test!(xor_u16_slice_matches_scalar, Sse2, xor_u16_slice, |x, y| x ^ y, u16);
slice_binop_matches_scalar_test!(andnot_u16_slice_matches_scalar, Sse2, andnot_u16_slice, |x, y| !x & y, u16);
slice_binop_matches_scalar_test!(mul_u16_slice_matches_scalar, Sse2, mul_u16_slice, |x: u16, y: u16| x.wrapping_mul(y), u16);
slice_binop_matches_scalar_test!(
	avg_u16_slice_matches_scalar, Sse2, avg_u16_slice, |x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16, u16
);

slice_shift_imm_matches_scalar_test!(shl_i8_slice_matches_scalar, Sse2, shl_i8_slice, 3, |x: i8, imm| x.wrapping_shl(imm), i8);
slice_shift_imm_matches_scalar_test!(
	shr_i8_slice_matches_scalar, Sse2, shr_i8_slice, 2, |x: i8, imm| ((x as u8).wrapping_shr(imm)) as i8, i8
);
slice_shift_imm_matches_scalar_test!(sra_i8_slice_matches_scalar, Sse2, sra_i8_slice, 1, |x: i8, imm| x.wrapping_shr(imm), i8);
slice_shift_imm_matches_scalar_test!(shl_u8_slice_matches_scalar, Sse2, shl_u8_slice, 3, |x: u8, imm| x.wrapping_shl(imm), u8);
slice_shift_imm_matches_scalar_test!(shr_u8_slice_matches_scalar, Sse2, shr_u8_slice, 2, |x: u8, imm| x.wrapping_shr(imm), u8);

slice_shift_imm_matches_scalar_test!(
	shl_i16_slice_matches_scalar, Sse2, shl_i16_slice, 3, |x: i16, imm| x.wrapping_shl(imm), i16
);
slice_shift_imm_matches_scalar_test!(
	shr_i16_slice_matches_scalar, Sse2, shr_i16_slice, 2, |x: i16, imm| ((x as u16).wrapping_shr(imm)) as i16, i16
);
slice_shift_imm_matches_scalar_test!(
	sra_i16_slice_matches_scalar, Sse2, sra_i16_slice, 1, |x: i16, imm| x.wrapping_shr(imm), i16
);
slice_shift_imm_matches_scalar_test!(
	shl_u16_slice_matches_scalar, Sse2, shl_u16_slice, 3, |x: u16, imm| x.wrapping_shl(imm), u16
);
slice_shift_imm_matches_scalar_test!(
	shr_u16_slice_matches_scalar, Sse2, shr_u16_slice, 2, |x: u16, imm| x.wrapping_shr(imm), u16
);

#[test]
fn adds_i8x16_saturates_at_bounds() {
	let Some(sse2) = Sse2::detect() else { return };
	let mut a = [0i8; 16];
	let mut b = [0i8; 16];
	a[0] = i8::MAX;
	b[0] = 1;
	let mut expect = [0i8; 16];
	expect[0] = i8::MAX;
	assert_eq!(sse2.adds_i8x16(a, b), expect);
}

#[test]
fn adds_u16x8_saturates_at_bounds() {
	let Some(sse2) = Sse2::detect() else { return };
	let mut a = [0u16; 8];
	let mut b = [0u16; 8];
	a[0] = u16::MAX;
	b[0] = 1;
	let mut expect = [0u16; 8];
	expect[0] = u16::MAX;
	assert_eq!(sse2.adds_u16x8(a, b), expect);
}

#[test]
fn mul_i8x16_wraps_on_overflow() {
	let Some(sse2) = Sse2::detect() else { return };
	let mut a = [0i8; 16];
	let mut b = [0i8; 16];
	a[0] = 100;
	b[0] = 3;
	let expect: [i8; 16] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(sse2.mul_i8x16(a, b), expect);
}

// i8::MIN * i8::MIN and 0xFF * 0xFF exercise the composed mul's low-byte
// masking at the extreme ends of the 16-bit product range.
#[test]
fn mul_i8x16_min_times_min_matches_wrapping_mul() {
	let Some(sse2) = Sse2::detect() else { return };
	let a = [i8::MIN; 16];
	let b = [i8::MIN; 16];
	let expect = [i8::MIN.wrapping_mul(i8::MIN); 16];
	assert_eq!(sse2.mul_i8x16(a, b), expect);
}

#[test]
fn mul_u8x16_max_times_max_matches_wrapping_mul() {
	let Some(sse2) = Sse2::detect() else { return };
	let a = [0xFFu8; 16];
	let b = [0xFFu8; 16];
	let expect = [0xFFu8.wrapping_mul(0xFF); 16];
	assert_eq!(sse2.mul_u8x16(a, b), expect);
}

// IMM=0 (no-op) and IMM=7 (max meaningful byte shift) exercise the mask
// computation's edges.
#[test]
fn shl_u8x16_imm0_is_identity() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8 * 17);
	assert_eq!(sse2.shl_u8x16::<0>(a), a);
}

#[test]
fn shl_u8x16_imm7_matches_scalar() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8 * 17);
	let expect: [u8; 16] = core::array::from_fn(|i| a[i].wrapping_shl(7));
	assert_eq!(sse2.shl_u8x16::<7>(a), expect);
}

#[test]
fn shr_u8x16_imm7_matches_scalar() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8 * 17);
	let expect: [u8; 16] = core::array::from_fn(|i| a[i].wrapping_shr(7));
	assert_eq!(sse2.shr_u8x16::<7>(a), expect);
}

#[test]
fn sra_i8x16_of_i8_min_matches_scalar() {
	let Some(sse2) = Sse2::detect() else { return };
	let a = [i8::MIN; 16];
	let expect = [i8::MIN.wrapping_shr(3); 16];
	assert_eq!(sse2.sra_i8x16::<3>(a), expect);
}

#[test]
fn min_u8x16_picks_smaller_lane() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8);
	let b: [u8; 16] = core::array::from_fn(|i| (15 - i) as u8);
	let expect: [u8; 16] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(sse2.min_u8x16(a, b), expect);
}

#[test]
fn min_i16x8_picks_smaller_lane() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i16; 8] = [1, -20, 3, -4, 5, -6, 7, -8];
	let b: [i16; 8] = [-1, 20, -3, 4, -5, 6, -7, 8];
	let expect: [i16; 8] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(sse2.min_i16x8(a, b), expect);
}

#[test]
fn movemask_i8x16_matches_sign_bits() {
	let Some(sse2) = Sse2::detect() else { return };
	let mut a = [1i8; 16];
	let mut expected: u16 = 0;
	for i in (0..16).step_by(3) {
		a[i] = -1;
		expected |= 1 << i;
	}
	assert_eq!(sse2.movemask_i8x16(a), expected);
}

#[test]
fn movemask_f64x2_matches_sign_bits() {
	let Some(sse2) = Sse2::detect() else { return };
	assert_eq!(sse2.movemask_f64x2([-1.0, 1.0]), 0b01);
	assert_eq!(sse2.movemask_f64x2([1.0, -1.0]), 0b10);
	assert_eq!(sse2.movemask_f64x2([-1.0, -1.0]), 0b11);
	assert_eq!(sse2.movemask_f64x2([0.0, 1.0]), 0b00);
}

#[test]
fn slli_u8x16_shifts_bytes_left_zero_filled() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i + 1) as u8);
	let got = sse2.slli_u8x16::<3>(a);
	let mut expect = [0u8; 16];
	expect[3..].copy_from_slice(&a[..13]);
	assert_eq!(got, expect);
	assert_eq!(sse2.slli_u8x16::<0>(a), a);
	assert_eq!(sse2.slli_u8x16::<16>(a), [0u8; 16]);
}

#[test]
fn srli_u8x16_shifts_bytes_right_zero_filled() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i + 1) as u8);
	let got = sse2.srli_u8x16::<3>(a);
	let mut expect = [0u8; 16];
	expect[..13].copy_from_slice(&a[3..]);
	assert_eq!(got, expect);
	assert_eq!(sse2.srli_u8x16::<0>(a), a);
	assert_eq!(sse2.srli_u8x16::<16>(a), [0u8; 16]);
}

#[test]
fn unpacklo_hi_i16x8_interleave_halves() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	let b: [i16; 8] = core::array::from_fn(|i| 100 + i as i16);
	assert_eq!(sse2.unpacklo_i16x8(a, b), [0, 100, 1, 101, 2, 102, 3, 103]);
	assert_eq!(sse2.unpackhi_i16x8(a, b), [4, 104, 5, 105, 6, 106, 7, 107]);
}

#[test]
fn unpacklo_hi_i32x4_interleave_halves() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i32; 4] = [0, 1, 2, 3];
	let b: [i32; 4] = [100, 101, 102, 103];
	assert_eq!(sse2.unpacklo_i32x4(a, b), [0, 100, 1, 101]);
	assert_eq!(sse2.unpackhi_i32x4(a, b), [2, 102, 3, 103]);
}

#[test]
fn unpacklo_hi_i64x2_interleave_halves() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i64; 2] = [0, 1];
	let b: [i64; 2] = [100, 101];
	assert_eq!(sse2.unpacklo_i64x2(a, b), [0, 100]);
	assert_eq!(sse2.unpackhi_i64x2(a, b), [1, 101]);
}

#[test]
fn shufflelo_i16x8_reverses_low_lanes_passes_high_through() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	// 0x1b = 0b00_01_10_11: reversed low 4 lanes.
	assert_eq!(sse2.shufflelo_i16x8::<0x1b>(a), [3, 2, 1, 0, 4, 5, 6, 7]);
}

#[test]
fn shufflehi_i16x8_reverses_high_lanes_passes_low_through() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	assert_eq!(sse2.shufflehi_i16x8::<0x1b>(a), [0, 1, 2, 3, 7, 6, 5, 4]);
}

#[test]
fn shuffle_i32x4_reverses_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i32; 4] = [0, 1, 2, 3];
	// 0x1b = 0b00_01_10_11: full reversal.
	assert_eq!(sse2.shuffle_i32x4::<0x1b>(a), [3, 2, 1, 0]);
}

#[test]
fn pack_i32x4_to_i16x8_narrows_in_range_lanes() {
	let Some(sse2) = Sse2::detect() else { return };
	let a: [i32; 4] = [1, -1, 100, -100];
	let b: [i32; 4] = [i32::MIN, i32::MAX, 0, 32767];
	assert_eq!(sse2.pack_i32x4_to_i16x8(a, b), [1, -1, 100, -100, -32768, 32767, 0, 32767]);
}
