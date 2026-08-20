use super::super::super::macros::{
	slice_binop_matches_scalar_test, slice_shift_imm_matches_scalar_test, slice_ternop_matches_scalar_test,
};
#[cfg(feature = "wider-bus-lift")]
use super::super::super::macros::slice_binop_lifted_matches_scalar_test;
use super::*;

/// One-directional, not equality: `from_level` under-detects real hardware
/// outside its bucket (see the identical fix + rationale on
/// `Avx::from_level_agreeing_implies_detect_agrees`, `avx.rs`).
#[test]
fn from_level_agreeing_implies_detect_agrees() {
	let level = GenericLevel::detect(FeatureSet::detect());
	if Avx2::from_level(level).is_some() {
		assert!(Avx2::detect().is_some());
	}
}

#[test]
fn add_i32x8_wraps_on_overflow() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [i32::MAX, 0, 0, 0, 0, 0, 0, 0];
	let b = [1, 0, 0, 0, 0, 0, 0, 0];
	assert_eq!(v3.add_i32x8(a, b), [i32::MIN, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn mul_i32x8_wraps_on_overflow() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [1 << 30, 0, 0, 0, 0, 0, 0, 0];
	let b = [4, 0, 0, 0, 0, 0, 0, 0];
	assert_eq!(v3.mul_i32x8(a, b), [0, 0, 0, 0, 0, 0, 0, 0]);
}

slice_binop_matches_scalar_test!(add_i32_slice_matches_scalar, Avx2, add_i32_slice, |x: i32, y: i32| x.wrapping_add(y), i32);
slice_binop_matches_scalar_test!(sub_i32_slice_matches_scalar, Avx2, sub_i32_slice, |x: i32, y: i32| x.wrapping_sub(y), i32);
slice_binop_matches_scalar_test!(mul_i32_slice_matches_scalar, Avx2, mul_i32_slice, |x: i32, y: i32| x.wrapping_mul(y), i32);
slice_binop_matches_scalar_test!(min_i32_slice_matches_scalar, Avx2, min_i32_slice, |x, y| x.min(y), i32);
slice_binop_matches_scalar_test!(max_i32_slice_matches_scalar, Avx2, max_i32_slice, |x, y| x.max(y), i32);
slice_binop_matches_scalar_test!(div_i32_slice_matches_scalar, Avx2, div_i32_slice, |x: i32, y: i32| x / y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	add_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, add_i32_slice_lifted, |x: i32, y: i32| x.wrapping_add(y), i32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	sub_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, sub_i32_slice_lifted, |x: i32, y: i32| x.wrapping_sub(y), i32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	mul_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, mul_i32_slice_lifted, |x: i32, y: i32| x.wrapping_mul(y), i32
);

slice_binop_matches_scalar_test!(add_u32_slice_matches_scalar, Avx2, add_u32_slice, |x: u32, y: u32| x.wrapping_add(y), u32);
slice_binop_matches_scalar_test!(sub_u32_slice_matches_scalar, Avx2, sub_u32_slice, |x: u32, y: u32| x.wrapping_sub(y), u32);
slice_binop_matches_scalar_test!(mul_u32_slice_matches_scalar, Avx2, mul_u32_slice, |x: u32, y: u32| x.wrapping_mul(y), u32);
slice_binop_matches_scalar_test!(min_u32_slice_matches_scalar, Avx2, min_u32_slice, |x, y| x.min(y), u32);
slice_binop_matches_scalar_test!(max_u32_slice_matches_scalar, Avx2, max_u32_slice, |x, y| x.max(y), u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	add_u32_slice_lifted_matches_scalar, Avx2, Avx512FVl, add_u32_slice_lifted, |x: u32, y: u32| x.wrapping_add(y), u32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	sub_u32_slice_lifted_matches_scalar, Avx2, Avx512FVl, sub_u32_slice_lifted, |x: u32, y: u32| x.wrapping_sub(y), u32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	mul_u32_slice_lifted_matches_scalar, Avx2, Avx512FVl, mul_u32_slice_lifted, |x: u32, y: u32| x.wrapping_mul(y), u32
);
slice_binop_matches_scalar_test!(div_u32_slice_matches_scalar, Avx2, div_u32_slice, |x: u32, y: u32| x / y, u32);

slice_binop_matches_scalar_test!(and_i32_slice_matches_scalar, Avx2, and_i32_slice, |x, y| x & y, i32);
slice_binop_matches_scalar_test!(xor_u32_slice_matches_scalar, Avx2, xor_u32_slice, |x, y| x ^ y, u32);
slice_binop_matches_scalar_test!(
	cmpeq_i32_slice_matches_scalar, Avx2, cmpeq_i32_slice,
	|x, y| if x == y { -1 } else { 0 }, i32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(and_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, and_i32_slice_lifted, |x, y| x & y, i32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(xor_u32_slice_lifted_matches_scalar, Avx2, Avx512FVl, xor_u32_slice_lifted, |x, y| x ^ y, u32);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	cmpeq_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, cmpeq_i32_slice_lifted,
	|x, y| if x == y { -1 } else { 0 }, i32
);
slice_shift_imm_matches_scalar_test!(
	shl_i32_slice_matches_scalar, Avx2, shl_i32_slice, 3,
	|x: i32, imm| x.wrapping_shl(imm), i32
);
slice_shift_imm_matches_scalar_test!(
	sra_i32_slice_matches_scalar, Avx2, sra_i32_slice, 1,
	|x: i32, imm| x.wrapping_shr(imm), i32
);

#[test]
fn add_i64x4_wraps_on_overflow() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [i64::MAX, 0, 0, 0];
	let b = [1, 0, 0, 0];
	assert_eq!(v3.add_i64x4(a, b), [i64::MIN, 0, 0, 0]);
}

slice_binop_matches_scalar_test!(add_i64_slice_matches_scalar, Avx2, add_i64_slice, |x: i64, y: i64| x.wrapping_add(y), i64);
slice_binop_matches_scalar_test!(sub_i64_slice_matches_scalar, Avx2, sub_i64_slice, |x: i64, y: i64| x.wrapping_sub(y), i64);
slice_binop_matches_scalar_test!(min_i64_slice_matches_scalar, Avx2, min_i64_slice, |x, y| i64::min(x, y), i64);
slice_binop_matches_scalar_test!(max_i64_slice_matches_scalar, Avx2, max_i64_slice, |x, y| i64::max(x, y), i64);
slice_binop_matches_scalar_test!(min_u64_slice_matches_scalar, Avx2, min_u64_slice, |x, y| u64::min(x, y), u64);
slice_binop_matches_scalar_test!(max_u64_slice_matches_scalar, Avx2, max_u64_slice, |x, y| u64::max(x, y), u64);

#[test]
fn mullo_u64x4_matches_scalar_for_cross_term_carry_values() {
	let Some(t) = Avx2::detect() else { return };
	let a = [0xFFFF_FFFF_FFFF_FFFFu64, 0x1_0000_0002, 0x8000_0000_0000_0001, 3];
	let b = [0xFFFF_FFFF_FFFF_FFFFu64, 0x1_0000_0003, 2, 0xFFFF_FFFF_FFFF_FFFF];
	let expect: [u64; 4] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(t.mullo_u64x4(a, b), expect);
}

#[test]
fn mullo_i64x4_matches_scalar_for_negative_values() {
	let Some(t) = Avx2::detect() else { return };
	let a = [i64::MIN, -12345, 7, -1];
	let b = [-1i64, 6789, i64::MIN, i64::MAX];
	let expect: [i64; 4] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(t.mullo_i64x4(a, b), expect);
}

slice_binop_matches_scalar_test!(mullo_i64_slice_matches_scalar, Avx2, mullo_i64_slice, |x: i64, y: i64| x.wrapping_mul(y), i64);

#[test]
fn mullo_u64_slice_matches_scalar_for_various_lengths_and_large_values() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 8, 9, 17] {
		let a: Vec<u64> = (0..len).map(|i| u64::MAX - i as u64 * 3).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64 + 1).wrapping_mul(0x1_0000_0001)).collect();
		let mut out = vec![0u64; len];
		t.mullo_u64_slice(&a, &b, &mut out);
		let expect: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_mul(y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn abs_i64x4_matches_scalar_including_i64_min() {
	let Some(t) = Avx2::detect() else { return };
	assert_eq!(t.abs_i64x4([i64::MIN, -42, 7, 0]), [i64::MIN, 42, 7, 0]);
}

#[test]
fn abs_i64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 8, 9, 17] {
		let a: Vec<i64> = (0..len).map(|i| (i as i64 - len as i64 / 2) * 0x1_0000_0007).collect();
		let mut out = vec![0i64; len];
		t.abs_i64_slice(&a, &mut out);
		let expect: Vec<i64> = a.iter().map(|&x| x.wrapping_abs()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn min_max_i64x4_match_scalar() {
	let Some(t) = Avx2::detect() else { return };
	let a = [i64::MIN, 7, -3, 100];
	let b = [3, -7, -3, -100];
	assert_eq!(t.min_i64x4(a, b), [i64::MIN, -7, -3, -100]);
	assert_eq!(t.max_i64x4(a, b), [3, 7, -3, 100]);
}

#[test]
fn min_max_u64x4_match_scalar() {
	let Some(t) = Avx2::detect() else { return };
	let a = [u64::MAX, 7, 3, 100];
	let b = [3, 20, 3, 0];
	assert_eq!(t.min_u64x4(a, b), [3, 7, 3, 0]);
	assert_eq!(t.max_u64x4(a, b), [u64::MAX, 20, 3, 100]);
}
slice_binop_matches_scalar_test!(add_u64_slice_matches_scalar, Avx2, add_u64_slice, |x: u64, y: u64| x.wrapping_add(y), u64);
slice_binop_matches_scalar_test!(sub_u64_slice_matches_scalar, Avx2, sub_u64_slice, |x: u64, y: u64| x.wrapping_sub(y), u64);

#[test]
fn cmpgt_u32x8_treats_high_bit_as_large_not_negative() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [0u32; 8];
	let mut b = [0u32; 8];
	a[0] = 0xFFFF_FFFF;
	b[0] = 0;
	let mut expect = [0u32; 8];
	expect[0] = !0;
	assert_eq!(v3.cmpgt_u32x8(a, b), expect);
}

slice_binop_matches_scalar_test!(
	cmpgt_i32_slice_matches_scalar, Avx2, cmpgt_i32_slice,
	|x, y| if x > y { -1 } else { 0 }, i32
);
#[cfg(feature = "wider-bus-lift")]
slice_binop_lifted_matches_scalar_test!(
	cmpgt_i32_slice_lifted_matches_scalar, Avx2, Avx512FVl, cmpgt_i32_slice_lifted,
	|x, y| if x > y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpgt_u32_slice_matches_scalar, Avx2, cmpgt_u32_slice,
	|x, y| if x > y { !0 } else { 0 }, u32
);

slice_binop_matches_scalar_test!(
	cmplt_i32_slice_matches_scalar, Avx2, cmplt_i32_slice,
	|x, y| if x < y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmple_i32_slice_matches_scalar, Avx2, cmple_i32_slice,
	|x, y| if x <= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmpge_i32_slice_matches_scalar, Avx2, cmpge_i32_slice,
	|x, y| if x >= y { -1 } else { 0 }, i32
);
slice_binop_matches_scalar_test!(
	cmplt_u32_slice_matches_scalar, Avx2, cmplt_u32_slice,
	|x, y| if x < y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmple_u32_slice_matches_scalar, Avx2, cmple_u32_slice,
	|x, y| if x <= y { !0 } else { 0 }, u32
);
slice_binop_matches_scalar_test!(
	cmpge_u32_slice_matches_scalar, Avx2, cmpge_u32_slice,
	|x, y| if x >= y { !0 } else { 0 }, u32
);

#[test]
fn sllv_i32x8_shifts_by_the_count_vector() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [1i32; 8];
	let count = [0i32, 1, 2, 3, 4, 5, 31, 32];
	let expect: [i32; 8] = core::array::from_fn(|i| {
		let c = count[i] as u32;
		if c >= 32 { 0 } else { 1i32.wrapping_shl(c) }
	});
	assert_eq!(v3.sllv_i32x8(a, count), expect);
}

#[test]
fn srav_i32x8_sign_fills_past_bit_width() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [-8i32; 8];
	let count = [0i32, 1, 2, 3, 4, 5, 31, 32];
	let expect: [i32; 8] = core::array::from_fn(|i| {
		let c = count[i] as u32;
		if c >= 32 { -8i32 >> 31 } else { (-8i32).wrapping_shr(c) }
	});
	assert_eq!(v3.srav_i32x8(a, count), expect);
}

slice_binop_matches_scalar_test!(
	sllv_i32_slice_matches_scalar, Avx2, sllv_i32_slice,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { x.wrapping_shl(count as u32) }, i32
);
slice_binop_matches_scalar_test!(
	srlv_i32_slice_matches_scalar, Avx2, srlv_i32_slice,
	|x: i32, count: i32| if (count as u32) >= 32 { 0 } else { ((x as u32).wrapping_shr(count as u32)) as i32 }, i32
);
slice_binop_matches_scalar_test!(
	sllv_u32_slice_matches_scalar, Avx2, sllv_u32_slice,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shl(count) }, u32
);
slice_binop_matches_scalar_test!(
	srlv_u32_slice_matches_scalar, Avx2, srlv_u32_slice,
	|x: u32, count: u32| if count >= 32 { 0 } else { x.wrapping_shr(count) }, u32
);

#[test]
fn select_i32x8_picks_b_where_mask_set() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [1, 2, 3, 4, 5, 6, 7, 8];
	let b = [10, 20, 30, 40, 50, 60, 70, 80];
	let mask = [-1, 0, -1, 0, -1, 0, -1, 0];
	assert_eq!(v3.select_i32x8(a, b, mask), [10, 2, 30, 4, 50, 6, 70, 8]);
}

#[test]
fn select_f32x8_uses_sign_bit_not_zero_test() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [1.0; 8];
	let b = [2.0; 8];
	let mask = [-0.0f32; 8]; // negative zero: sign bit set, value == 0.0
	assert_eq!(v3.select_f32x8(a, b, mask), [2.0; 8]);
}

// select_i32/u32: no shared slice_ternop test (macro mask `2` is outside all-0/1 domain).
slice_ternop_matches_scalar_test!(
	select_f32_slice_matches_scalar, Avx2, select_f32_slice,
	|a: f32, b: f32, m: f32| if m.is_sign_negative() { b } else { a }, f32
);

#[test]
fn select_i8x32_picks_b_where_mask_set() {
	let Some(v3) = Avx2::detect() else { return };
	let a: [i8; 32] = core::array::from_fn(|i| i as i8);
	let b: [i8; 32] = core::array::from_fn(|i| 100 - i as i8);
	let mask: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(v3.select_i8x32(a, b, mask), expect);
}

#[test]
fn select_i16x16_picks_b_where_mask_set() {
	let Some(v3) = Avx2::detect() else { return };
	let a: [i16; 16] = core::array::from_fn(|i| i as i16);
	let b: [i16; 16] = core::array::from_fn(|i| 100 - i as i16);
	let mask: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(v3.select_i16x16(a, b, mask), expect);
}

// select_i8/u8/i16/u16: no shared slice_ternop test, same out-of-domain reason as select_i32.

#[test]
fn adds_i8x32_saturates_on_overflow() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [0i8; 32];
	let mut b = [0i8; 32];
	a[0] = i8::MAX;
	b[0] = 1;
	let mut expect = [0i8; 32];
	expect[0] = i8::MAX;
	assert_eq!(v3.adds_i8x32(a, b), expect);
}

#[test]
fn subs_u8x32_saturates_on_underflow() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [10u8; 32];
	let mut b = [20u8; 32];
	a[0] = 0;
	b[0] = 1;
	let mut expect = [0u8; 32];
	expect[0] = 0;
	assert_eq!(v3.subs_u8x32(a, b), expect);
}

#[test]
fn mul_i8x32_wraps_on_overflow() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [0i8; 32];
	let mut b = [0i8; 32];
	a[0] = 100;
	b[0] = 3;
	let mut expect = [0i8; 32];
	expect[0] = 100i8.wrapping_mul(3);
	assert_eq!(v3.mul_i8x32(a, b), expect);
}

// i8::MIN * i8::MIN and 0xFF * 0xFF exercise the composed mul's low-byte
// masking at the extreme ends of the 16-bit product range.
#[test]
fn mul_i8x32_min_times_min_matches_wrapping_mul() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [i8::MIN; 32];
	let b = [i8::MIN; 32];
	let expect = [i8::MIN.wrapping_mul(i8::MIN); 32];
	assert_eq!(v3.mul_i8x32(a, b), expect);
}

#[test]
fn mul_u8x32_max_times_max_matches_wrapping_mul() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [0xFFu8; 32];
	let b = [0xFFu8; 32];
	let expect = [0xFFu8.wrapping_mul(0xFF); 32];
	assert_eq!(v3.mul_u8x32(a, b), expect);
}

// AVX2's `unpacklo/hi_epi8`/`packus_epi16` are per-128-bit-lane
#[test]
fn mul_u8x32_low_and_high_lanes_are_independent() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [0u8; 32];
	let mut b = [0u8; 32];
	for i in 0..16 {
		a[i] = 3;
		b[i] = 5;
	}
	for i in 16..32 {
		a[i] = 200;
		b[i] = 7;
	}
	let got = v3.mul_u8x32(a, b);
	let expect: [u8; 32] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(got, expect);
	assert_eq!(&got[0..16], &[15u8; 16]);
	assert_eq!(&got[16..32], &[200u8.wrapping_mul(7); 16]);
}

// IMM=0 (no-op) and IMM=7 (max meaningful byte shift) exercise the mask
// computation's edges.
#[test]
fn shl_u8x32_imm0_is_identity() {
	let Some(v3) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 * 7);
	assert_eq!(v3.shl_u8x32::<0>(a), a);
}

#[test]
fn shl_u8x32_imm7_matches_scalar() {
	let Some(v3) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 * 7);
	let expect: [u8; 32] = core::array::from_fn(|i| a[i].wrapping_shl(7));
	assert_eq!(v3.shl_u8x32::<7>(a), expect);
}

#[test]
fn shr_u8x32_imm7_matches_scalar() {
	let Some(v3) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 * 7);
	let expect: [u8; 32] = core::array::from_fn(|i| a[i].wrapping_shr(7));
	assert_eq!(v3.shr_u8x32::<7>(a), expect);
}

#[test]
fn sra_i8x32_of_i8_min_matches_scalar() {
	let Some(v3) = Avx2::detect() else { return };
	let a = [i8::MIN; 32];
	let expect = [i8::MIN.wrapping_shr(3); 32];
	assert_eq!(v3.sra_i8x32::<3>(a), expect);
}

slice_binop_matches_scalar_test!(add_i8_slice_matches_scalar, Avx2, add_i8_slice, |x: i8, y: i8| x.wrapping_add(y), i8);
slice_binop_matches_scalar_test!(sub_i8_slice_matches_scalar, Avx2, sub_i8_slice, |x: i8, y: i8| x.wrapping_sub(y), i8);
slice_binop_matches_scalar_test!(adds_i8_slice_matches_scalar, Avx2, adds_i8_slice, |x: i8, y: i8| x.saturating_add(y), i8);
slice_binop_matches_scalar_test!(subs_i8_slice_matches_scalar, Avx2, subs_i8_slice, |x: i8, y: i8| x.saturating_sub(y), i8);
slice_binop_matches_scalar_test!(min_i8_slice_matches_scalar, Avx2, min_i8_slice, |x, y| x.min(y), i8);
slice_binop_matches_scalar_test!(max_i8_slice_matches_scalar, Avx2, max_i8_slice, |x, y| x.max(y), i8);
slice_binop_matches_scalar_test!(mul_i8_slice_matches_scalar, Avx2, mul_i8_slice, |x: i8, y: i8| x.wrapping_mul(y), i8);

#[test]
fn abs_i8x32_wrapping_abs_of_min() {
	let Some(t) = Avx2::detect() else { return };
	let mut a = [0i8; 32];
	a[0] = i8::MIN;
	a[1] = -5;
	let mut expect = [0i8; 32];
	expect[0] = i8::MIN; // wrapping_abs(MIN) == MIN
	expect[1] = 5;
	assert_eq!(t.abs_i8x32(a), expect);
}
slice_binop_matches_scalar_test!(
	cmpgt_i8_slice_matches_scalar, Avx2, cmpgt_i8_slice,
	|x, y| if x > y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmple_i8_slice_matches_scalar, Avx2, cmple_i8_slice,
	|x, y| if x <= y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmpgt_u8_slice_matches_scalar, Avx2, cmpgt_u8_slice,
	|x, y| if x > y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(
	cmpeq_i8_slice_matches_scalar, Avx2, cmpeq_i8_slice,
	|x, y| if x == y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(and_i8_slice_matches_scalar, Avx2, and_i8_slice, |x, y| x & y, i8);
slice_shift_imm_matches_scalar_test!(
	shl_i8_slice_matches_scalar, Avx2, shl_i8_slice, 3,
	|x: i8, imm| x.wrapping_shl(imm), i8
);
slice_shift_imm_matches_scalar_test!(
	shr_i8_slice_matches_scalar, Avx2, shr_i8_slice, 2,
	|x: i8, imm| ((x as u8).wrapping_shr(imm)) as i8, i8
);
slice_shift_imm_matches_scalar_test!(
	sra_i8_slice_matches_scalar, Avx2, sra_i8_slice, 1,
	|x: i8, imm| x.wrapping_shr(imm), i8
);

slice_binop_matches_scalar_test!(add_u8_slice_matches_scalar, Avx2, add_u8_slice, |x: u8, y: u8| x.wrapping_add(y), u8);
slice_binop_matches_scalar_test!(sub_u8_slice_matches_scalar, Avx2, sub_u8_slice, |x: u8, y: u8| x.wrapping_sub(y), u8);
slice_binop_matches_scalar_test!(adds_u8_slice_matches_scalar, Avx2, adds_u8_slice, |x: u8, y: u8| x.saturating_add(y), u8);
slice_binop_matches_scalar_test!(subs_u8_slice_matches_scalar, Avx2, subs_u8_slice, |x: u8, y: u8| x.saturating_sub(y), u8);
slice_binop_matches_scalar_test!(min_u8_slice_matches_scalar, Avx2, min_u8_slice, |x, y| x.min(y), u8);
slice_binop_matches_scalar_test!(max_u8_slice_matches_scalar, Avx2, max_u8_slice, |x, y| x.max(y), u8);
slice_binop_matches_scalar_test!(mul_u8_slice_matches_scalar, Avx2, mul_u8_slice, |x: u8, y: u8| x.wrapping_mul(y), u8);
slice_binop_matches_scalar_test!(
	avg_u8_slice_matches_scalar, Avx2, avg_u8_slice, |x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8, u8
);
slice_binop_matches_scalar_test!(
	cmpeq_u8_slice_matches_scalar, Avx2, cmpeq_u8_slice,
	|x, y| if x == y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(xor_u8_slice_matches_scalar, Avx2, xor_u8_slice, |x, y| x ^ y, u8);
slice_shift_imm_matches_scalar_test!(
	shl_u8_slice_matches_scalar, Avx2, shl_u8_slice, 3,
	|x: u8, imm| x.wrapping_shl(imm), u8
);
slice_shift_imm_matches_scalar_test!(
	shr_u8_slice_matches_scalar, Avx2, shr_u8_slice, 1,
	|x: u8, imm| x.wrapping_shr(imm), u8
);

slice_binop_matches_scalar_test!(add_i16_slice_matches_scalar, Avx2, add_i16_slice, |x: i16, y: i16| x.wrapping_add(y), i16);
slice_binop_matches_scalar_test!(sub_i16_slice_matches_scalar, Avx2, sub_i16_slice, |x: i16, y: i16| x.wrapping_sub(y), i16);
slice_binop_matches_scalar_test!(adds_i16_slice_matches_scalar, Avx2, adds_i16_slice, |x: i16, y: i16| x.saturating_add(y), i16);
slice_binop_matches_scalar_test!(subs_i16_slice_matches_scalar, Avx2, subs_i16_slice, |x: i16, y: i16| x.saturating_sub(y), i16);
slice_binop_matches_scalar_test!(mul_i16_slice_matches_scalar, Avx2, mul_i16_slice, |x: i16, y: i16| x.wrapping_mul(y), i16);
slice_binop_matches_scalar_test!(min_i16_slice_matches_scalar, Avx2, min_i16_slice, |x, y| x.min(y), i16);
slice_binop_matches_scalar_test!(max_i16_slice_matches_scalar, Avx2, max_i16_slice, |x, y| x.max(y), i16);

#[test]
fn abs_i16x16_wrapping_abs_of_min() {
	let Some(t) = Avx2::detect() else { return };
	let mut a = [0i16; 16];
	a[0] = i16::MIN;
	a[1] = -5;
	let mut expect = [0i16; 16];
	expect[0] = i16::MIN;
	expect[1] = 5;
	assert_eq!(t.abs_i16x16(a), expect);
}
slice_binop_matches_scalar_test!(
	cmpgt_i16_slice_matches_scalar, Avx2, cmpgt_i16_slice,
	|x, y| if x > y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmpge_i16_slice_matches_scalar, Avx2, cmpge_i16_slice,
	|x, y| if x >= y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmpgt_u16_slice_matches_scalar, Avx2, cmpgt_u16_slice,
	|x, y| if x > y { !0 } else { 0 }, u16
);
slice_binop_matches_scalar_test!(
	cmpeq_i16_slice_matches_scalar, Avx2, cmpeq_i16_slice,
	|x, y| if x == y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(and_i16_slice_matches_scalar, Avx2, and_i16_slice, |x, y| x & y, i16);
slice_shift_imm_matches_scalar_test!(
	shl_i16_slice_matches_scalar, Avx2, shl_i16_slice, 3,
	|x: i16, imm| x.wrapping_shl(imm), i16
);
slice_shift_imm_matches_scalar_test!(
	sra_i16_slice_matches_scalar, Avx2, sra_i16_slice, 1,
	|x: i16, imm| x.wrapping_shr(imm), i16
);

slice_binop_matches_scalar_test!(add_u16_slice_matches_scalar, Avx2, add_u16_slice, |x: u16, y: u16| x.wrapping_add(y), u16);
slice_binop_matches_scalar_test!(sub_u16_slice_matches_scalar, Avx2, sub_u16_slice, |x: u16, y: u16| x.wrapping_sub(y), u16);
slice_binop_matches_scalar_test!(adds_u16_slice_matches_scalar, Avx2, adds_u16_slice, |x: u16, y: u16| x.saturating_add(y), u16);
slice_binop_matches_scalar_test!(subs_u16_slice_matches_scalar, Avx2, subs_u16_slice, |x: u16, y: u16| x.saturating_sub(y), u16);
slice_binop_matches_scalar_test!(mul_u16_slice_matches_scalar, Avx2, mul_u16_slice, |x: u16, y: u16| x.wrapping_mul(y), u16);
slice_binop_matches_scalar_test!(
	avg_u16_slice_matches_scalar, Avx2, avg_u16_slice, |x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16, u16
);
slice_binop_matches_scalar_test!(min_u16_slice_matches_scalar, Avx2, min_u16_slice, |x, y| x.min(y), u16);
slice_binop_matches_scalar_test!(max_u16_slice_matches_scalar, Avx2, max_u16_slice, |x, y| x.max(y), u16);
slice_binop_matches_scalar_test!(
	cmpeq_u16_slice_matches_scalar, Avx2, cmpeq_u16_slice,
	|x, y| if x == y { !0 } else { 0 }, u16
);
slice_binop_matches_scalar_test!(xor_u16_slice_matches_scalar, Avx2, xor_u16_slice, |x, y| x ^ y, u16);
slice_shift_imm_matches_scalar_test!(
	shl_u16_slice_matches_scalar, Avx2, shl_u16_slice, 3,
	|x: u16, imm| x.wrapping_shl(imm), u16
);
slice_shift_imm_matches_scalar_test!(
	shr_u16_slice_matches_scalar, Avx2, shr_u16_slice, 1,
	|x: u16, imm| x.wrapping_shr(imm), u16
);

#[test]
fn movemask_i8x32_matches_sign_bits() {
	let Some(v3) = Avx2::detect() else { return };
	let mut a = [1i8; 32];
	let mut expected: u32 = 0;
	for i in (0..32).step_by(3) {
		a[i] = -1;
		expected |= 1 << i;
	}
	assert_eq!(v3.movemask_i8x32(a), expected);
}

#[test]
fn alignr_u8x32_imm0_returns_b_unchanged() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x32::<0>(a, b), b);
}

#[test]
fn alignr_u8x32_imm16_returns_a_unchanged() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x32::<16>(a, b), a);
}

#[test]
fn alignr_u8x32_lanes_are_independent() {
	// Each 128-bit lane sees only its own half of `a`/`b`: the high lane
	// must never pull bytes from the low lane's half, unlike a true
	// 256-bit concatenation would.
	let Some(t) = Avx2::detect() else { return };
	let mut a = [0u8; 32];
	let mut b = [0u8; 32];
	a[16] = 0xAA; // first byte of the high lane
	b[0] = 0xBB; // first byte of the low lane
	let out = t.alignr_u8x32::<1>(a, b);
	// Low lane: window=[b_lo,a_lo] shifted by 1 -> out[0..16][15] pulls a_lo[0]=0, not a[16].
	assert_eq!(out[15], 0);
	// High lane: window=[b_hi,a_hi] shifted by 1 -> out[16..32][15] pulls a_hi[0]=a[16]=0xAA.
	assert_eq!(out[31], 0xAA);
}

#[test]
fn alignr_u8x32_matches_scalar_reference() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	for imm in [1, 5, 15, 17, 24, 31] {
		let expect = alignr_u8x32_scalar(&a, &b, imm);
		let out = match imm {
			1 => t.alignr_u8x32::<1>(a, b),
			5 => t.alignr_u8x32::<5>(a, b),
			15 => t.alignr_u8x32::<15>(a, b),
			17 => t.alignr_u8x32::<17>(a, b),
			24 => t.alignr_u8x32::<24>(a, b),
			31 => t.alignr_u8x32::<31>(a, b),
			_ => unreachable!(),
		};
		assert_eq!(out.to_vec(), expect, "imm={imm}");
	}
}

#[test]
fn alignr_u8x32_full_imm0_returns_b_unchanged() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x32_full::<0>(a, b), b);
}

#[test]
fn alignr_u8x32_full_imm32_returns_a_unchanged() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x32_full::<32>(a, b), a);
}

#[test]
fn alignr_u8x32_full_imm64_or_more_is_all_zero() {
	let Some(t) = Avx2::detect() else { return };
	let a = [0xFFu8; 32];
	let b = [0xFFu8; 32];
	assert_eq!(t.alignr_u8x32_full::<64>(a, b), [0u8; 32]);
	assert_eq!(t.alignr_u8x32_full::<200>(a, b), [0u8; 32]);
}

#[test]
fn alignr_u8x32_full_matches_scalar_reference() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	for imm in [1, 5, 15, 16, 17, 24, 31, 33, 47, 48, 49, 63] {
		let expect = alignr_u8x32_full_scalar(&a, &b, imm);
		let out = match imm {
			1 => t.alignr_u8x32_full::<1>(a, b),
			5 => t.alignr_u8x32_full::<5>(a, b),
			15 => t.alignr_u8x32_full::<15>(a, b),
			16 => t.alignr_u8x32_full::<16>(a, b),
			17 => t.alignr_u8x32_full::<17>(a, b),
			24 => t.alignr_u8x32_full::<24>(a, b),
			31 => t.alignr_u8x32_full::<31>(a, b),
			33 => t.alignr_u8x32_full::<33>(a, b),
			47 => t.alignr_u8x32_full::<47>(a, b),
			48 => t.alignr_u8x32_full::<48>(a, b),
			49 => t.alignr_u8x32_full::<49>(a, b),
			63 => t.alignr_u8x32_full::<63>(a, b),
			_ => unreachable!(),
		};
		assert_eq!(out.to_vec(), expect, "imm={imm}");
	}
}

#[test]
fn alignr_u8x32_full_pulls_across_the_lane_boundary_the_native_form_cannot() {
	// imm=8: out[24] = window[32] = a[0]: a's LOW half feeding the
	// HIGH lane's output. The lane-locked `alignr_u8x32` can only feed
	// the high lane from a_hi/b_hi, so it must disagree here.
	let Some(t) = Avx2::detect() else { return };
	let mut a = [0u8; 32];
	a[0] = 0xAA;
	let b = [0u8; 32];
	assert_eq!(t.alignr_u8x32_full::<8>(a, b)[24], 0xAA);
	assert_ne!(t.alignr_u8x32::<8>(a, b)[24], 0xAA);
}

#[test]
fn slli_u8x32_shifts_each_128_bit_lane_independently() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i + 1) as u8);
	let got = t.slli_u8x32::<3>(a);
	let mut expect = [0u8; 32];
	expect[3..16].copy_from_slice(&a[0..13]);
	expect[19..32].copy_from_slice(&a[16..29]);
	assert_eq!(got, expect);
	assert_eq!(t.slli_u8x32::<0>(a), a);
	assert_eq!(t.slli_u8x32::<16>(a), [0u8; 32]);
}

#[test]
fn broadcast_u8x32_replicates_byte_across_all_lanes() {
	let Some(t) = Avx2::detect() else { return };
	assert_eq!(t.broadcast_u8x32(0x7A), [0x7Au8; 32]);
	assert_eq!(t.broadcast_u8x32(0), [0u8; 32]);
}

#[test]
fn extract_insert_u8x16_x32_roundtrip() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8);
	assert_eq!(t.extract_u8x16_from_x32::<0>(a).to_vec(), a[..16].to_vec());
	assert_eq!(t.extract_u8x16_from_x32::<1>(a).to_vec(), a[16..].to_vec());

	let b: [u8; 16] = core::array::from_fn(|i| 200 + i as u8);
	let mut expect = a;
	expect[16..].copy_from_slice(&b);
	assert_eq!(t.insert_u8x16_into_x32::<1>(a, b), expect);
}

#[test]
fn widening_mul_u32x8_matches_scalar_full_product() {
	let Some(t) = Avx2::detect() else { return };
	let a: [u32; 8] = core::array::from_fn(|i| (i as u32 + 1) * 0x1000_0001);
	let b: [u32; 8] = core::array::from_fn(|i| 0xFFFF_FFFF - i as u32);
	let (lo, hi) = t.widening_mul_u32x8(a, b);
	for i in 0..8 {
		let full = a[i] as u64 * b[i] as u64;
		assert_eq!(lo[i], full as u32, "lo[{i}]");
		assert_eq!(hi[i], (full >> 32) as u32, "hi[{i}]");
	}
}

#[test]
fn widening_mul_i32x8_matches_scalar_full_product() {
	let Some(t) = Avx2::detect() else { return };
	let a: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 12345);
	let b: [i32; 8] = core::array::from_fn(|i| i32::MIN + i as i32 * 999);
	let (lo, hi) = t.widening_mul_i32x8(a, b);
	for i in 0..8 {
		let full = a[i] as i64 * b[i] as i64;
		assert_eq!(lo[i], full as i32, "lo[{i}]");
		assert_eq!(hi[i], (full >> 32) as i32, "hi[{i}]");
	}
}

#[test]
fn partial_load_store_i32x8_roundtrip_various_lengths() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 8, 12] {
		let src: Vec<i32> = (0..len).map(|i| i as i32 * -7 + 3).collect();
		let v = t.partial_load_i32x8(&src);
		let mut dst = vec![-1i32; len.min(8)];
		t.partial_store_i32x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}

#[test]
fn partial_load_store_u32x8_roundtrip_various_lengths() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 8, 12] {
		let src: Vec<u32> = (0..len).map(|i| i as u32 * 7 + 3).collect();
		let v = t.partial_load_u32x8(&src);
		let mut dst = vec![u32::MAX; len.min(8)];
		t.partial_store_u32x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}

#[test]
fn partial_load_store_i64x4_roundtrip_various_lengths() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 4, 6] {
		let src: Vec<i64> = (0..len).map(|i| i as i64 * -7 + 3).collect();
		let v = t.partial_load_i64x4(&src);
		let mut dst = vec![-1i64; len.min(4)];
		t.partial_store_i64x4(v, &mut dst);
		assert_eq!(dst, &src[..len.min(4)], "len {len}");
	}
}

#[test]
fn partial_load_store_u64x4_roundtrip_various_lengths() {
	let Some(t) = Avx2::detect() else { return };
	for len in [0usize, 1, 3, 4, 6] {
		let src: Vec<u64> = (0..len).map(|i| i as u64 * 7 + 3).collect();
		let v = t.partial_load_u64x4(&src);
		let mut dst = vec![u64::MAX; len.min(4)];
		t.partial_store_u64x4(v, &mut dst);
		assert_eq!(dst, &src[..len.min(4)], "len {len}");
	}
}
