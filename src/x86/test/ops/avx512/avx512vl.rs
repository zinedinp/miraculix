use super::super::avx512bitalg::bitshuffle_scalar;
use super::super::avx512ifma::MASK52;
use super::super::avx512vbmi::{multishift_scalar, permutex2var_scalar, permutexvar_scalar};
use super::super::super::macros::{
	slice_binop_imm_matches_scalar_test, slice_binop_matches_scalar_test, slice_ternop_matches_scalar_test,
};
use super::*;

#[test]
fn popcnt_u8x16_matches_core() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let expect: [u8; 16] = core::array::from_fn(|i| a[i].count_ones() as u8);
	assert_eq!(t.popcnt_u8x16(a), expect);
}

#[test]
fn popcnt_u8x32_matches_core() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let expect: [u8; 32] = core::array::from_fn(|i| a[i].count_ones() as u8);
	assert_eq!(t.popcnt_u8x32(a), expect);
}

#[test]
fn popcnt_u16x8_matches_core() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| (i as u16).wrapping_mul(6151) ^ 0x5A5A);
	let expect: [u16; 8] = core::array::from_fn(|i| a[i].count_ones() as u16);
	assert_eq!(t.popcnt_u16x8(a), expect);
}

#[test]
fn popcnt_u16x16_matches_core() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| (i as u16).wrapping_mul(6151) ^ 0x5A5A);
	let expect: [u16; 16] = core::array::from_fn(|i| a[i].count_ones() as u16);
	assert_eq!(t.popcnt_u16x16(a), expect);
}

#[test]
fn popcnt_u8_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	for len in [0usize, 1, 31, 32, 33, 70, 200] {
		let a: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(53)).collect();
		let mut out = vec![0u8; len];
		t.popcnt_u8_slice_wide(&a, &mut out);
		let expect: Vec<u8> = a.iter().map(|x| x.count_ones() as u8).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u16_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 40, 100] {
		let a: Vec<u16> = (0..len).map(|i| (i as u16).wrapping_mul(6151)).collect();
		let mut out = vec![0u16; len];
		t.popcnt_u16_slice_wide(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|x| x.count_ones() as u16).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn bitshuffle_mask_u64x4_matches_scalar_reference() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let b: [u64; 4] = [0x0123_4567_89AB_CDEF, u64::MAX, 0, 0x8000_0000_0000_0000];
	let c: [u64; 4] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x0102_0304_0506_0708) ^ 0xDEAD_BEEF);
	let expect = bitshuffle_scalar(&b, &c) as u32;
	assert_eq!(t.bitshuffle_mask_u64x4(b, c), expect);
}

#[test]
fn bitshuffle_mask_u64x2_matches_scalar_reference() {
	let Some(t) = Avx512BitalgVl::detect() else { return };
	let b: [u64; 2] = [0x0123_4567_89AB_CDEF, u64::MAX];
	let c: [u64; 2] = [0xDEAD_BEEF_0011_2233, 0x1122_3344_5566_7788];
	let expect = bitshuffle_scalar(&b, &c) as u16;
	assert_eq!(t.bitshuffle_mask_u64x2(b, c), expect);
}

#[test]
fn detect_dq_vl_requires_both_bits() {
	let fs = FeatureSet::detect();
	let expect = fs.contains(Feature::Avx512dq) && fs.contains(Feature::Avx512vl);
	assert_eq!(Avx512DqVl::detect().is_some(), expect);
}

#[test]
fn mullo_i64x2_wraps_on_overflow() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [i64::MAX, 3];
	let b = [2, 4];
	let expect = [i64::MAX.wrapping_mul(2), 12];
	assert_eq!(t.mullo_i64x2(a, b), expect);
}

#[test]
fn mullo_u64x4_matches_core() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 4] = [0, u64::MAX, 1, 1 << 63];
	let b: [u64; 4] = [5, 2, u64::MAX, 2];
	let expect: [u64; 4] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(t.mullo_u64x4(a, b), expect);
}

slice_binop_matches_scalar_test!(
	mullo_i64_slice_vl_matches_scalar, Avx512DqVl, mullo_i64_slice, |x: i64, y: i64| x.wrapping_mul(y), i64
);
slice_binop_matches_scalar_test!(
	mullo_i64_slice_wide_matches_scalar, Avx512DqVl, mullo_i64_slice_wide, |x: i64, y: i64| x.wrapping_mul(y), i64
);
slice_binop_matches_scalar_test!(
	mullo_u64_slice_vl_matches_scalar, Avx512DqVl, mullo_u64_slice, |x: u64, y: u64| x.wrapping_mul(y), u64
);
slice_binop_matches_scalar_test!(
	mullo_u64_slice_wide_matches_scalar, Avx512DqVl, mullo_u64_slice_wide, |x: u64, y: u64| x.wrapping_mul(y), u64
);

#[test]
fn i64_to_f64x2_and_x4_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [i64; 2] = [-1000, 42];
	assert_eq!(t.i64_to_f64x2(a2), a2.map(|x| x as f64));
	let a4: [i64; 4] = [-1000, -1, 42, 123456];
	assert_eq!(t.i64_to_f64x4(a4), a4.map(|x| x as f64));
}

#[test]
fn u64_to_f64x2_and_x4_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [u64; 2] = [0, 42];
	assert_eq!(t.u64_to_f64x2(a2), a2.map(|x| x as f64));
	let a4: [u64; 4] = [0, 1, 42, 999999];
	assert_eq!(t.u64_to_f64x4(a4), a4.map(|x| x as f64));
}

#[test]
fn f64_to_i64x2_and_x4_match_round_ties_even() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f64; 2] = [-2.5, 2.5];
	assert_eq!(t.f64_to_i64x2(a2), a2.map(|x| x.round_ties_even() as i64));
	let a4: [f64; 4] = [-2.5, -0.5, 0.5, 3.5];
	assert_eq!(t.f64_to_i64x4(a4), a4.map(|x| x.round_ties_even() as i64));
}

#[test]
fn f64_to_i64x2_and_x4_trunc_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f64; 2] = [-2.9, 2.9];
	assert_eq!(t.f64_to_i64x2_trunc(a2), a2.map(|x| x as i64));
	let a4: [f64; 4] = [-2.9, -0.9, 0.9, 3.9];
	assert_eq!(t.f64_to_i64x4_trunc(a4), a4.map(|x| x as i64));
}

#[test]
fn f64_to_u64x2_and_x4_match_round_ties_even() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f64; 2] = [0.5, 2.5];
	assert_eq!(t.f64_to_u64x2(a2), a2.map(|x| x.round_ties_even() as u64));
	let a4: [f64; 4] = [0.0, 0.5, 2.5, 999999.5];
	assert_eq!(t.f64_to_u64x4(a4), a4.map(|x| x.round_ties_even() as u64));
}

#[test]
fn f64_to_u64x2_and_x4_trunc_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f64; 2] = [0.9, 2.9];
	assert_eq!(t.f64_to_u64x2_trunc(a2), a2.map(|x| x as u64));
	let a4: [f64; 4] = [0.0, 0.9, 2.9, 999999.9];
	assert_eq!(t.f64_to_u64x4_trunc(a4), a4.map(|x| x as u64));
}

#[test]
fn i64_to_f32x2_and_x4_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [i64; 2] = [-1000, 42];
	let expect2 = [a2[0] as f32, a2[1] as f32, 0.0, 0.0];
	assert_eq!(t.i64_to_f32x2(a2), expect2);
	let a4: [i64; 4] = [-1000, -1, 42, 123456];
	assert_eq!(t.i64_to_f32x4(a4), a4.map(|x| x as f32));
}

#[test]
fn u64_to_f32x2_and_x4_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [u64; 2] = [0, 42];
	let expect2 = [a2[0] as f32, a2[1] as f32, 0.0, 0.0];
	assert_eq!(t.u64_to_f32x2(a2), expect2);
	let a4: [u64; 4] = [0, 1, 42, 999999];
	assert_eq!(t.u64_to_f32x4(a4), a4.map(|x| x as f32));
}

#[test]
fn f32_to_i64x2_and_x4_match_round_ties_even() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f32; 4] = [-2.5, 2.5, 999.0, 999.0];
	assert_eq!(t.f32_to_i64x2(a2), [a2[0].round_ties_even() as i64, a2[1].round_ties_even() as i64]);
	let a4: [f32; 4] = [-2.5, -0.5, 0.5, 3.5];
	assert_eq!(t.f32_to_i64x4(a4), a4.map(|x| x.round_ties_even() as i64));
}

#[test]
fn f32_to_i64x2_and_x4_trunc_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f32; 4] = [-2.9, 2.9, 999.0, 999.0];
	assert_eq!(t.f32_to_i64x2_trunc(a2), [a2[0] as i64, a2[1] as i64]);
	let a4: [f32; 4] = [-2.9, -0.9, 0.9, 3.9];
	assert_eq!(t.f32_to_i64x4_trunc(a4), a4.map(|x| x as i64));
}

#[test]
fn f32_to_u64x2_and_x4_match_round_ties_even() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f32; 4] = [0.5, 2.5, 999.0, 999.0];
	assert_eq!(t.f32_to_u64x2(a2), [a2[0].round_ties_even() as u64, a2[1].round_ties_even() as u64]);
	let a4: [f32; 4] = [0.0, 0.5, 2.5, 999999.5];
	assert_eq!(t.f32_to_u64x4(a4), a4.map(|x| x.round_ties_even() as u64));
}

#[test]
fn f32_to_u64x2_and_x4_trunc_match_as_cast() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a2: [f32; 4] = [0.9, 2.9, 999.0, 999.0];
	assert_eq!(t.f32_to_u64x2_trunc(a2), [a2[0] as u64, a2[1] as u64]);
	let a4: [f32; 4] = [0.0, 0.9, 2.9, 999999.9];
	assert_eq!(t.f32_to_u64x4_trunc(a4), a4.map(|x| x as u64));
}

#[test]
fn range_f64x2_and_x4_min_and_max() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.range_f64x2::<0>([3.0; 2], [7.0; 2]), [3.0; 2]);
	assert_eq!(t.range_f64x2::<1>([3.0; 2], [7.0; 2]), [7.0; 2]);
	assert_eq!(t.range_f64x4::<0>([3.0; 4], [7.0; 4]), [3.0; 4]);
	assert_eq!(t.range_f64x4::<1>([3.0; 4], [7.0; 4]), [7.0; 4]);
}

#[test]
fn range_f32x4_and_x8_min_and_max() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.range_f32x4::<0>([3.0; 4], [7.0; 4]), [3.0; 4]);
	assert_eq!(t.range_f32x4::<1>([3.0; 4], [7.0; 4]), [7.0; 4]);
	assert_eq!(t.range_f32x8::<0>([3.0; 8], [7.0; 8]), [3.0; 8]);
	assert_eq!(t.range_f32x8::<1>([3.0; 8], [7.0; 8]), [7.0; 8]);
}

#[test]
fn reduce_f64x2_and_x4_subtract_truncated_integer_part() {
	let Some(t) = Avx512DqVl::detect() else { return };
	// IMM8=3: M=0 (no scaling), rounding mode 3 = truncate toward zero.
	assert_eq!(t.reduce_f64x2::<3>([2.5, -2.5]), [0.5, -0.5]);
	assert_eq!(t.reduce_f64x4::<3>([2.5, -2.5, 2.5, -2.5]), [0.5, -0.5, 0.5, -0.5]);
}

#[test]
fn reduce_f32x4_and_x8_subtract_truncated_integer_part() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.reduce_f32x4::<3>([2.5f32; 4]), [0.5f32; 4]);
	assert_eq!(t.reduce_f32x8::<3>([2.5f32; 8]), [0.5f32; 8]);
}

#[test]
fn fpclass_f64x2_and_x4_flag_only_nan_lane() {
	let Some(t) = Avx512DqVl::detect() else { return };
	// IMM8=1: bit0 (QNaN) only.
	assert_eq!(t.fpclass_f64x2::<1>([f64::NAN, 1.0]), 0b01);
	assert_eq!(t.fpclass_f64x4::<1>([1.0, f64::NAN, 1.0, 1.0]), 0b0010);
}

#[test]
fn fpclass_f32x4_and_x8_flag_only_nan_lane() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.fpclass_f32x4::<1>([f32::NAN, 1.0, 1.0, 1.0]), 0b0001);
	let mut a = [1.0f32; 8];
	a[5] = f32::NAN;
	assert_eq!(t.fpclass_f32x8::<1>(a), 1 << 5);
}

#[test]
fn broadcast_f32x2_to_x8_ignores_upper_input_lanes() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [1.0f32, 2.0, 999.0, 999.0];
	assert_eq!(t.broadcast_f32x2_to_x8(a), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn broadcast_i32x2_to_x8_ignores_upper_input_lanes() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [10i32, 20, 999, 999];
	assert_eq!(t.broadcast_i32x2_to_x8(a), [10, 20, 10, 20, 10, 20, 10, 20]);
}

#[test]
fn broadcast_f64x2_to_x4_repeats_pair() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.broadcast_f64x2_to_x4([1.0, 2.0]), [1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn broadcast_i64x2_to_x4_repeats_pair() {
	let Some(t) = Avx512DqVl::detect() else { return };
	assert_eq!(t.broadcast_i64x2_to_x4([10, 20]), [10, 20, 10, 20]);
}

#[test]
fn extract_f64x2_from_x4_picks_selected_half() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
	assert_eq!(t.extract_f64x2_from_x4::<0>(a), [0.0, 1.0]);
	assert_eq!(t.extract_f64x2_from_x4::<1>(a), [2.0, 3.0]);
}

#[test]
fn extract_i64x2_from_x4_picks_selected_half() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = [0, 1, 2, 3];
	assert_eq!(t.extract_i64x2_from_x4::<1>(a), [2, 3]);
}

#[test]
fn insert_f64x2_into_x4_overwrites_selected_half() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
	assert_eq!(t.insert_f64x2_into_x4::<1>(a, [99.0, 98.0]), [0.0, 1.0, 99.0, 98.0]);
}

#[test]
fn insert_i64x2_into_x4_overwrites_selected_half() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = [0, 1, 2, 3];
	assert_eq!(t.insert_i64x2_into_x4::<0>(a, [99, 98]), [99, 98, 2, 3]);
}

#[test]
fn madd52hi_u64x2_matches_scalar_reference() {
	let Some(t) = Avx512IfmaVl::detect() else { return };
	let src = [7u64, 42];
	let a = [MASK52, MASK52 / 3];
	let b = [MASK52, MASK52 / 7];
	let expect = [madd52hi_scalar(src[0], a[0], b[0]), madd52hi_scalar(src[1], a[1], b[1])];
	assert_eq!(t.madd52hi_u64x2(src, a, b), expect);
}

#[test]
fn madd52lo_u64x4_matches_scalar_reference() {
	let Some(t) = Avx512IfmaVl::detect() else { return };
	let src: [u64; 4] = [1, 2, 3, 4];
	let a: [u64; 4] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x1_0000_0007));
	let b: [u64; 4] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef);
	let expect: [u64; 4] = core::array::from_fn(|i| madd52lo_scalar(src[i], a[i], b[i]));
	assert_eq!(t.madd52lo_u64x4(src, a, b), expect);
}

#[test]
fn madd52hi_u64_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512IfmaVl::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<u64> = (0..len).map(|i| i as u64).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		t.madd52hi_u64_slice_wide(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52hi_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

macro_rules! masked_ifma_vl_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $width:literal, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512IfmaVl::detect() else { return };
			let src: [u64; $width] = core::array::from_fn(|i| i as u64 * 7 + 1);
			let a: [u64; $width] = core::array::from_fn(|i| MASK52 / (i as u64 + 1));
			let b: [u64; $width] = core::array::from_fn(|i| MASK52 / (i as u64 + 3));
			let mask: u8 = $mask_val;
			let op: fn(u64, u64, u64) -> u64 = $op;
			let merge_expect: [u64; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(src[i], a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect: [u64; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(src[i], a[i], b[i]) } else { 0 });
			assert_eq!(t.$zero_fn(mask, src, a, b), zero_expect, "zero");
		}
	};
}

masked_ifma_vl_test!(
	madd52lo_u64x2_masked_matches_scalar, madd52lo_u64x2_merge_masked, madd52lo_u64x2_zero_masked,
	2, 0b01, madd52lo_scalar
);
masked_ifma_vl_test!(
	madd52hi_u64x2_masked_matches_scalar, madd52hi_u64x2_merge_masked, madd52hi_u64x2_zero_masked,
	2, 0b01, madd52hi_scalar
);
masked_ifma_vl_test!(
	madd52lo_u64x4_masked_matches_scalar, madd52lo_u64x4_merge_masked, madd52lo_u64x4_zero_masked,
	4, 0b1010, madd52lo_scalar
);
masked_ifma_vl_test!(
	madd52hi_u64x4_masked_matches_scalar, madd52hi_u64x4_merge_masked, madd52hi_u64x4_zero_masked,
	4, 0b1010, madd52hi_scalar
);

#[test]
fn permutexvar_u8x32_matches_scalar_reference() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let idx: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let expect = permutexvar_scalar(&idx, &a);
	assert_eq!(t.permutexvar_u8x32(idx, a).to_vec(), expect);
}

#[test]
fn permutexvar_u8x16_matches_scalar_reference() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let idx: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(29) ^ 0x5A);
	let expect = permutexvar_scalar(&idx, &a);
	assert_eq!(t.permutexvar_u8x16(idx, a).to_vec(), expect);
}

#[test]
fn permutex2var_u8x32_matches_scalar_reference() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8);
	let b: [u8; 32] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let idx: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let expect = permutex2var_scalar(&a, &idx, &b);
	assert_eq!(t.permutex2var_u8x32(a, idx, b).to_vec(), expect);
}

#[test]
fn permutex2var_u8x16_select_bit_chooses_b() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let mut a = [0u8; 16];
	let mut b = [0u8; 16];
	a[3] = 1;
	b[3] = 2;
	let mut idx = [0u8; 16];
	idx[0] = 3 | 0x10; // 128-bit select bit is 0x10 (16 lanes)
	assert_eq!(t.permutex2var_u8x16(a, idx, b)[0], 2);
	idx[0] = 3;
	assert_eq!(t.permutex2var_u8x16(a, idx, b)[0], 1);
}

#[test]
fn multishift_u8x32_matches_scalar_reference() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29) ^ 0x5A);
	let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(71) ^ 0x0F);
	let expect = multishift_scalar(&a, &b);
	assert_eq!(t.multishift_u8x32(a, b).to_vec(), expect);
}

#[test]
fn multishift_u8x16_matches_scalar_reference() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17) ^ 0x2B);
	let b: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(83) ^ 0x91);
	let expect = multishift_scalar(&a, &b);
	assert_eq!(t.multishift_u8x16(a, b).to_vec(), expect);
}

#[test]
fn vbmi2_shldv_u32x8_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a = [0x1234_5678u32; 8];
	let b = [0xDEAD_BEEFu32; 8];
	let c = [0u32; 8];
	assert_eq!(t.shldv_u32x8(a, b, c), a);
}

#[test]
fn vbmi2_shrdv_u32x4_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a = [0x1234_5678u32; 4];
	let b = [0xDEAD_BEEFu32; 4];
	let c = [0u32; 4];
	assert_eq!(t.shrdv_u32x4(a, b, c), a);
}

#[test]
fn vbmi2_shldv_i64x4_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a: [i64; 4] = core::array::from_fn(|i| (i as i64).wrapping_mul(0x1_0000_0007));
	let b: [i64; 4] = core::array::from_fn(|i| (i as i64).wrapping_mul(-0x3_0000_0001));
	let c: [i64; 4] = core::array::from_fn(|i| i as i64 * 11);
	let expect: [i64; 4] = core::array::from_fn(|i| shldv_i64_scalar(a[i], b[i], c[i]));
	assert_eq!(t.shldv_i64x4(a, b, c), expect);
}

#[test]
fn vbmi2_shrdv_u16x8_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x9E37) ^ 0x1234);
	let b: [u16; 8] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x7F4A) ^ 0xABCD);
	let c: [u16; 8] = core::array::from_fn(|i| i as u16);
	let expect: [u16; 8] = core::array::from_fn(|i| shrdv_u16_scalar(a[i], b[i], c[i]));
	assert_eq!(t.shrdv_u16x8(a, b, c), expect);
}

slice_ternop_matches_scalar_test!(vbmi2_shldv_i16_slice_matches_scalar, Avx512Vbmi2Vl, shldv_i16_slice, shldv_i16_scalar, i16);
slice_ternop_matches_scalar_test!(vbmi2_shldv_u16_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldv_u16_slice_wide, shldv_u16_scalar, u16);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_i16_slice_matches_scalar, Avx512Vbmi2Vl, shrdv_i16_slice, shrdv_i16_scalar, i16);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_u16_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdv_u16_slice_wide, shrdv_u16_scalar, u16);
slice_ternop_matches_scalar_test!(vbmi2_shldv_i32_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldv_i32_slice_wide, shldv_i32_scalar, i32);
slice_ternop_matches_scalar_test!(vbmi2_shldv_u32_slice_matches_scalar, Avx512Vbmi2Vl, shldv_u32_slice, shldv_u32_scalar, u32);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_i32_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdv_i32_slice_wide, shrdv_i32_scalar, i32);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_u32_slice_matches_scalar, Avx512Vbmi2Vl, shrdv_u32_slice, shrdv_u32_scalar, u32);
slice_ternop_matches_scalar_test!(vbmi2_shldv_i64_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldv_i64_slice_wide, shldv_i64_scalar, i64);
slice_ternop_matches_scalar_test!(vbmi2_shldv_u64_slice_matches_scalar, Avx512Vbmi2Vl, shldv_u64_slice, shldv_u64_scalar, u64);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_i64_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdv_i64_slice_wide, shrdv_i64_scalar, i64);
slice_ternop_matches_scalar_test!(vbmi2_shrdv_u64_slice_matches_scalar, Avx512Vbmi2Vl, shrdv_u64_slice, shrdv_u64_scalar, u64);

#[test]
fn vbmi2_shldi_u32x8_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a = [0x1234_5678u32; 8];
	let b = [0xDEAD_BEEFu32; 8];
	assert_eq!(t.shldi_u32x8::<0>(a, b), a);
}

#[test]
fn vbmi2_shrdi_u32x4_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a = [0x1234_5678u32; 4];
	let b = [0xDEAD_BEEFu32; 4];
	assert_eq!(t.shrdi_u32x4::<0>(a, b), a);
}

#[test]
fn vbmi2_shldi_i64x4_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a: [i64; 4] = core::array::from_fn(|i| (i as i64).wrapping_mul(0x1_0000_0007));
	let b: [i64; 4] = core::array::from_fn(|i| (i as i64).wrapping_mul(-0x3_0000_0001));
	let expect: [i64; 4] = core::array::from_fn(|i| shldi_i64_scalar(a[i], b[i], 11));
	assert_eq!(t.shldi_i64x4::<11>(a, b), expect);
}

#[test]
fn vbmi2_shrdi_u16x8_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x9E37) ^ 0x1234);
	let b: [u16; 8] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x7F4A) ^ 0xABCD);
	let expect: [u16; 8] = core::array::from_fn(|i| shrdi_u16_scalar(a[i], b[i], 5));
	assert_eq!(t.shrdi_u16x8::<5>(a, b), expect);
}

#[test]
fn vbmi2_shldi_matches_shldv_with_broadcast_c() {
	let Some(t) = Avx512Vbmi2Vl::detect() else { return };
	let a: [i32; 8] = core::array::from_fn(|i| (i as i32).wrapping_mul(0x1F) - 100);
	let b: [i32; 8] = core::array::from_fn(|i| (i as i32).wrapping_mul(-7) + 50);
	let c = [11i32; 8];
	assert_eq!(t.shldi_i32x8::<11>(a, b), t.shldv_i32x8(a, b, c));
	assert_eq!(t.shrdi_i32x8::<11>(a, b), t.shrdv_i32x8(a, b, c));
}

slice_binop_imm_matches_scalar_test!(vbmi2_shldi_i16_slice_matches_scalar, Avx512Vbmi2Vl, shldi_i16_slice, 5, shldi_i16_scalar, i16);
slice_binop_imm_matches_scalar_test!(vbmi2_shldi_u16_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldi_u16_slice_wide, 5, shldi_u16_scalar, u16);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_i16_slice_matches_scalar, Avx512Vbmi2Vl, shrdi_i16_slice, 5, shrdi_i16_scalar, i16);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_u16_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdi_u16_slice_wide, 5, shrdi_u16_scalar, u16);
slice_binop_imm_matches_scalar_test!(vbmi2_shldi_i32_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldi_i32_slice_wide, 11, shldi_i32_scalar, i32);
slice_binop_imm_matches_scalar_test!(vbmi2_shldi_u32_slice_matches_scalar, Avx512Vbmi2Vl, shldi_u32_slice, 11, shldi_u32_scalar, u32);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_i32_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdi_i32_slice_wide, 11, shrdi_i32_scalar, i32);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_u32_slice_matches_scalar, Avx512Vbmi2Vl, shrdi_u32_slice, 11, shrdi_u32_scalar, u32);
slice_binop_imm_matches_scalar_test!(vbmi2_shldi_i64_slice_wide_matches_scalar, Avx512Vbmi2Vl, shldi_i64_slice_wide, 37, shldi_i64_scalar, i64);
slice_binop_imm_matches_scalar_test!(vbmi2_shldi_u64_slice_matches_scalar, Avx512Vbmi2Vl, shldi_u64_slice, 37, shldi_u64_scalar, u64);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_i64_slice_wide_matches_scalar, Avx512Vbmi2Vl, shrdi_i64_slice_wide, 37, shrdi_i64_scalar, i64);
slice_binop_imm_matches_scalar_test!(vbmi2_shrdi_u64_slice_matches_scalar, Avx512Vbmi2Vl, shrdi_u64_slice, 37, shrdi_u64_scalar, u64);

macro_rules! masked_binop_imm_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $imm:expr, $a:expr, $b:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2Vl::detect() else { return };
			const IMM8: i32 = $imm;
			let src = $a;
			let a = $a;
			let b = $b;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i], IMM8) } else { src[i] });
			assert_eq!(t.$merge_fn::<IMM8>(src, mask, a, b), merge_expect, "merge");
			let zero_expect = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 { op(a[i], b[i], IMM8) } else { Default::default() }
			});
			assert_eq!(t.$zero_fn::<IMM8>(mask, a, b), zero_expect, "zero");
		}
	};
}

masked_binop_imm_test!(
	shldi_i16x16_masked_matches_scalar, shldi_i16x16_merge_masked, shldi_i16x16_zero_masked, u16, 5,
	core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5A5Au16, |a, b, imm| shldi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u16x16_masked_matches_scalar, shldi_u16x16_merge_masked, shldi_u16x16_zero_masked, u16, 5,
	core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(7)),
	0x5A5Au16, |a, b, imm| shldi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i16x8_masked_matches_scalar, shldi_i16x8_merge_masked, shldi_i16x8_zero_masked, u8, 5,
	core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u16x8_masked_matches_scalar, shldi_u16x8_merge_masked, shldi_u16x8_zero_masked, u8, 5,
	core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i16x16_masked_matches_scalar, shrdi_i16x16_merge_masked, shrdi_i16x16_zero_masked, u16, 5,
	core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5A5Au16, |a, b, imm| shrdi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u16x16_masked_matches_scalar, shrdi_u16x16_merge_masked, shrdi_u16x16_zero_masked, u16, 5,
	core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(7)),
	0x5A5Au16, |a, b, imm| shrdi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i16x8_masked_matches_scalar, shrdi_i16x8_merge_masked, shrdi_i16x8_zero_masked, u8, 5,
	core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u16x8_masked_matches_scalar, shrdi_u16x8_merge_masked, shrdi_u16x8_zero_masked, u8, 5,
	core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i32x8_masked_matches_scalar, shldi_i32x8_merge_masked, shldi_i32x8_zero_masked, u8, 11,
	core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u32x8_masked_matches_scalar, shldi_u32x8_merge_masked, shldi_u32x8_zero_masked, u8, 11,
	core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i32x4_masked_matches_scalar, shldi_i32x4_merge_masked, shldi_i32x4_zero_masked, u8, 11,
	core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u32x4_masked_matches_scalar, shldi_u32x4_merge_masked, shldi_u32x4_zero_masked, u8, 11,
	core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i32x8_masked_matches_scalar, shrdi_i32x8_merge_masked, shrdi_i32x8_zero_masked, u8, 11,
	core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u32x8_masked_matches_scalar, shrdi_u32x8_merge_masked, shrdi_u32x8_zero_masked, u8, 11,
	core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i32x4_masked_matches_scalar, shrdi_i32x4_merge_masked, shrdi_i32x4_zero_masked, u8, 11,
	core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u32x4_masked_matches_scalar, shrdi_u32x4_merge_masked, shrdi_u32x4_zero_masked, u8, 11,
	core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i64x4_masked_matches_scalar, shldi_i64x4_merge_masked, shldi_i64x4_zero_masked, u8, 37,
	core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u64x4_masked_matches_scalar, shldi_u64x4_merge_masked, shldi_u64x4_zero_masked, u8, 37,
	core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i64x2_masked_matches_scalar, shldi_i64x2_merge_masked, shldi_i64x2_zero_masked, u8, 37,
	core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u64x2_masked_matches_scalar, shldi_u64x2_merge_masked, shldi_u64x2_zero_masked, u8, 37,
	core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i64x4_masked_matches_scalar, shrdi_i64x4_merge_masked, shrdi_i64x4_zero_masked, u8, 37,
	core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u64x4_masked_matches_scalar, shrdi_u64x4_merge_masked, shrdi_u64x4_zero_masked, u8, 37,
	core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i64x2_masked_matches_scalar, shrdi_i64x2_merge_masked, shrdi_i64x2_zero_masked, u8, 37,
	core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u64x2_masked_matches_scalar, shrdi_u64x2_merge_masked, shrdi_u64x2_zero_masked, u8, 37,
	core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u64_scalar(a, b, imm)
);

macro_rules! masked_ternop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $b:expr, $c:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2Vl::detect() else { return };
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
	shldv_i16x16_masked_matches_scalar, shldv_i16x16_merge_masked, shldv_i16x16_zero_masked, u16,
	core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(-7)), core::array::from_fn::<i16, 16, _>(|i| (i as i16) & 0xF),
	0x5A5Au16, |a, b, c| shldv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u16x16_masked_matches_scalar, shldv_u16x16_merge_masked, shldv_u16x16_zero_masked, u16,
	core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(7)), core::array::from_fn::<u16, 16, _>(|i| (i as u16) & 0xF),
	0x5A5Au16, |a, b, c| shldv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i16x8_masked_matches_scalar, shldv_i16x8_merge_masked, shldv_i16x8_zero_masked, u8,
	core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(-7)), core::array::from_fn::<i16, 8, _>(|i| (i as i16) & 0xF),
	0x5Au8, |a, b, c| shldv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u16x8_masked_matches_scalar, shldv_u16x8_merge_masked, shldv_u16x8_zero_masked, u8,
	core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(7)), core::array::from_fn::<u16, 8, _>(|i| (i as u16) & 0xF),
	0x5Au8, |a, b, c| shldv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i16x16_masked_matches_scalar, shrdv_i16x16_merge_masked, shrdv_i16x16_zero_masked, u16,
	core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 16, _>(|i| (i as i16).wrapping_mul(-7)), core::array::from_fn::<i16, 16, _>(|i| (i as i16) & 0xF),
	0x5A5Au16, |a, b, c| shrdv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u16x16_masked_matches_scalar, shrdv_u16x16_merge_masked, shrdv_u16x16_zero_masked, u16,
	core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 16, _>(|i| (i as u16).wrapping_mul(7)), core::array::from_fn::<u16, 16, _>(|i| (i as u16) & 0xF),
	0x5A5Au16, |a, b, c| shrdv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i16x8_masked_matches_scalar, shrdv_i16x8_merge_masked, shrdv_i16x8_zero_masked, u8,
	core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 8, _>(|i| (i as i16).wrapping_mul(-7)), core::array::from_fn::<i16, 8, _>(|i| (i as i16) & 0xF),
	0x5Au8, |a, b, c| shrdv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u16x8_masked_matches_scalar, shrdv_u16x8_merge_masked, shrdv_u16x8_zero_masked, u8,
	core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 8, _>(|i| (i as u16).wrapping_mul(7)), core::array::from_fn::<u16, 8, _>(|i| (i as u16) & 0xF),
	0x5Au8, |a, b, c| shrdv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i32x8_masked_matches_scalar, shldv_i32x8_merge_masked, shldv_i32x8_zero_masked, u8,
	core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(-7)), core::array::from_fn::<i32, 8, _>(|i| (i as i32) & 0xF),
	0x5Au8, |a, b, c| shldv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u32x8_masked_matches_scalar, shldv_u32x8_merge_masked, shldv_u32x8_zero_masked, u8,
	core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(7)), core::array::from_fn::<u32, 8, _>(|i| (i as u32) & 0xF),
	0x5Au8, |a, b, c| shldv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i32x4_masked_matches_scalar, shldv_i32x4_merge_masked, shldv_i32x4_zero_masked, u8,
	core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(-7)), core::array::from_fn::<i32, 4, _>(|i| (i as i32) & 0xF),
	0x5Au8, |a, b, c| shldv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u32x4_masked_matches_scalar, shldv_u32x4_merge_masked, shldv_u32x4_zero_masked, u8,
	core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(7)), core::array::from_fn::<u32, 4, _>(|i| (i as u32) & 0xF),
	0x5Au8, |a, b, c| shldv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i32x8_masked_matches_scalar, shrdv_i32x8_merge_masked, shrdv_i32x8_zero_masked, u8,
	core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 8, _>(|i| (i as i32).wrapping_mul(-7)), core::array::from_fn::<i32, 8, _>(|i| (i as i32) & 0xF),
	0x5Au8, |a, b, c| shrdv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u32x8_masked_matches_scalar, shrdv_u32x8_merge_masked, shrdv_u32x8_zero_masked, u8,
	core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 8, _>(|i| (i as u32).wrapping_mul(7)), core::array::from_fn::<u32, 8, _>(|i| (i as u32) & 0xF),
	0x5Au8, |a, b, c| shrdv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i32x4_masked_matches_scalar, shrdv_i32x4_merge_masked, shrdv_i32x4_zero_masked, u8,
	core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 4, _>(|i| (i as i32).wrapping_mul(-7)), core::array::from_fn::<i32, 4, _>(|i| (i as i32) & 0xF),
	0x5Au8, |a, b, c| shrdv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u32x4_masked_matches_scalar, shrdv_u32x4_merge_masked, shrdv_u32x4_zero_masked, u8,
	core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 4, _>(|i| (i as u32).wrapping_mul(7)), core::array::from_fn::<u32, 4, _>(|i| (i as u32) & 0xF),
	0x5Au8, |a, b, c| shrdv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i64x4_masked_matches_scalar, shldv_i64x4_merge_masked, shldv_i64x4_zero_masked, u8,
	core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(-7)), core::array::from_fn::<i64, 4, _>(|i| (i as i64) & 0xF),
	0x5Au8, |a, b, c| shldv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u64x4_masked_matches_scalar, shldv_u64x4_merge_masked, shldv_u64x4_zero_masked, u8,
	core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(7)), core::array::from_fn::<u64, 4, _>(|i| (i as u64) & 0xF),
	0x5Au8, |a, b, c| shldv_u64_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i64x2_masked_matches_scalar, shldv_i64x2_merge_masked, shldv_i64x2_zero_masked, u8,
	core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(-7)), core::array::from_fn::<i64, 2, _>(|i| (i as i64) & 0xF),
	0x5Au8, |a, b, c| shldv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u64x2_masked_matches_scalar, shldv_u64x2_merge_masked, shldv_u64x2_zero_masked, u8,
	core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(7)), core::array::from_fn::<u64, 2, _>(|i| (i as u64) & 0xF),
	0x5Au8, |a, b, c| shldv_u64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i64x4_masked_matches_scalar, shrdv_i64x4_merge_masked, shrdv_i64x4_zero_masked, u8,
	core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 4, _>(|i| (i as i64).wrapping_mul(-7)), core::array::from_fn::<i64, 4, _>(|i| (i as i64) & 0xF),
	0x5Au8, |a, b, c| shrdv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u64x4_masked_matches_scalar, shrdv_u64x4_merge_masked, shrdv_u64x4_zero_masked, u8,
	core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 4, _>(|i| (i as u64).wrapping_mul(7)), core::array::from_fn::<u64, 4, _>(|i| (i as u64) & 0xF),
	0x5Au8, |a, b, c| shrdv_u64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i64x2_masked_matches_scalar, shrdv_i64x2_merge_masked, shrdv_i64x2_zero_masked, u8,
	core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 2, _>(|i| (i as i64).wrapping_mul(-7)), core::array::from_fn::<i64, 2, _>(|i| (i as i64) & 0xF),
	0x5Au8, |a, b, c| shrdv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u64x2_masked_matches_scalar, shrdv_u64x2_merge_masked, shrdv_u64x2_zero_masked, u8,
	core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 2, _>(|i| (i as u64).wrapping_mul(7)), core::array::from_fn::<u64, 2, _>(|i| (i as u64) & 0xF),
	0x5Au8, |a, b, c| shrdv_u64_scalar(a, b, c)
);

#[test]
fn dpbusd_i32x4_sums_four_way_dot_product() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src = [1000i32, 1974, 3084, 3884];
	let a: [u8; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
	let b: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
	let expect: [i32; 4] = core::array::from_fn(|j| {
		let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
		vnni_acc_wrapping(src[j], sum)
	});
	assert_eq!(t.dpbusd_i32x4(src, a, b), expect);
}

#[test]
fn dpbusd_i32x8_matches_scalar_reference() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 5);
	let a: [u8; 32] = core::array::from_fn(|i| (i % 13) as u8 + 1);
	let b: [i8; 32] = core::array::from_fn(|i| ((i % 5) as i8) - 2);
	let expect: [i32; 8] = core::array::from_fn(|j| {
		let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
		vnni_acc_wrapping(src[j], sum)
	});
	assert_eq!(t.dpbusd_i32x8(src, a, b), expect);
}

#[test]
fn dpwssds_i32_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 17, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32 - 50).collect();
		let a: Vec<i16> = (0..len * 2).map(|i| (i as i16).wrapping_mul(37)).collect();
		let b: Vec<i16> = (0..len * 2).map(|i| (i as i16).wrapping_mul(-13)).collect();
		let mut out = vec![0i32; len];
		t.dpwssds_i32_slice_wide(&src, &a, &b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
				vnni_acc_saturating(src[j], sum)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u32x4_matches_core() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	let a: [u32; 4] = [0, u32::MAX, 0xF0F0_F0F0, 12345];
	let expect: [u32; 4] = core::array::from_fn(|i| a[i].count_ones());
	assert_eq!(t.popcnt_u32x4(a), expect);
}

#[test]
fn popcnt_u32x8_matches_core() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	let a: [u32; 8] = core::array::from_fn(|i| (i as u32).wrapping_mul(2_654_435_761));
	let expect: [u32; 8] = core::array::from_fn(|i| a[i].count_ones());
	assert_eq!(t.popcnt_u32x8(a), expect);
}

#[test]
fn popcnt_u64x2_matches_core() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	let a: [u64; 2] = [0, u64::MAX];
	let expect: [u64; 2] = core::array::from_fn(|i| a[i].count_ones() as u64);
	assert_eq!(t.popcnt_u64x2(a), expect);
}

#[test]
fn popcnt_u64x4_matches_core() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	let a: [u64; 4] = [0, u64::MAX, 0x5555_5555_5555_5555, 3];
	let expect: [u64; 4] = core::array::from_fn(|i| a[i].count_ones() as u64);
	assert_eq!(t.popcnt_u64x4(a), expect);
}

#[test]
fn popcnt_u32_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 17, 100] {
		let a: Vec<u32> = (0..len).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
		let mut out = vec![0u32; len];
		t.popcnt_u32_slice_wide(&a, &mut out);
		let expect: Vec<u32> = a.iter().map(|x| x.count_ones()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u64_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512VpopcntdqVl::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 9, 100] {
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
		let mut out = vec![0u64; len];
		t.popcnt_u64_slice_wide(&a, &mut out);
		let expect: Vec<u64> = a.iter().map(|x| x.count_ones() as u64).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn detect_bf16_vl_requires_both_bits() {
	let fs = FeatureSet::detect();
	let expect = fs.contains(Feature::Avx512bf16) && fs.contains(Feature::Avx512vl);
	assert_eq!(Avx512Bf16Vl::detect().is_some(), expect);
}

#[test]
fn dpbf16_ps_f32x8_matches_scalar_reference() {
	use super::super::avx512bf16::{bf16_to_f32_scalar, f32_to_bf16_scalar};
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let src: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5);
	let a_f32: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 0.25);
	let b_f32: [f32; 16] = core::array::from_fn(|i| (i as f32 % 5.0) - 2.0);
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_scalar(a_f32[i]));
	let b: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_scalar(b_f32[i]));

	let got = t.dpbf16_ps_f32x8(src, a, b);
	let expect: [f32; 8] = core::array::from_fn(|j| {
		let mut acc = src[j];
		acc += bf16_to_f32_scalar(a[2 * j + 1]) * bf16_to_f32_scalar(b[2 * j + 1]);
		acc += bf16_to_f32_scalar(a[2 * j]) * bf16_to_f32_scalar(b[2 * j]);
		acc
	});
	assert_eq!(got, expect);
}

#[test]
fn dpbf16_ps_f32x4_matches_scalar_reference() {
	use super::super::avx512bf16::{bf16_to_f32_scalar, f32_to_bf16_scalar};
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let src: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
	let a_f32: [f32; 8] = core::array::from_fn(|i| (i as f32 - 4.0) * 0.5);
	let b_f32: [f32; 8] = core::array::from_fn(|i| (i as f32 % 3.0) - 1.0);
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar(a_f32[i]));
	let b: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar(b_f32[i]));

	let got = t.dpbf16_ps_f32x4(src, a, b);
	let expect: [f32; 4] = core::array::from_fn(|j| {
		let mut acc = src[j];
		acc += bf16_to_f32_scalar(a[2 * j + 1]) * bf16_to_f32_scalar(b[2 * j + 1]);
		acc += bf16_to_f32_scalar(a[2 * j]) * bf16_to_f32_scalar(b[2 * j]);
		acc
	});
	assert_eq!(got, expect);
}

#[test]
fn dpbf16_ps_f32x4_zero_src_is_pure_dot_product() {
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let one = f32_to_bf16_scalar(1.0);
	let two = f32_to_bf16_scalar(2.0);
	let a = [one; 8];
	let b = [two; 8];
	let got = t.dpbf16_ps_f32x4([0.0; 4], a, b);
	assert_eq!(got, [4.0f32; 4]);
}

#[test]
fn cvtneps_pbh_u16x8_matches_scalar_rne_reference() {
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 8] =
		core::array::from_fn(|i| [1.0, -1.0, 0.0, f32::MIN_POSITIVE, 12345.678, -0.001, 9876.543, 1e30][i]);
	let got = t.cvtneps_pbh_u16x8(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtneps_pbh_u16x4_matches_scalar_rne_reference() {
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 4] = [1.0, -1.0, 12345.678, -0.001];
	let got = t.cvtneps_pbh_u16x4(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_bf16_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtne2ps_pbh_u16x16_matches_scalar_rne_reference_and_lane_order() {
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 8] = core::array::from_fn(|i| i as f32 + 0.3);
	let b: [f32; 8] = core::array::from_fn(|i| -(i as f32) - 0.7);
	let got = t.cvtne2ps_pbh_u16x16(a, b);

	let expect_low: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar(b[i]));
	let expect_high: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar(a[i]));
	assert_eq!(&got[0..8], &expect_low);
	assert_eq!(&got[8..16], &expect_high);
}

#[test]
fn cvtne2ps_pbh_u16x8_matches_scalar_rne_reference_and_lane_order() {
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 4] = [0.3, 1.3, 2.3, 3.3];
	let b: [f32; 4] = [-0.7, -1.7, -2.7, -3.7];
	let got = t.cvtne2ps_pbh_u16x8(a, b);

	let expect_low: [u16; 4] = core::array::from_fn(|i| f32_to_bf16_scalar(b[i]));
	let expect_high: [u16; 4] = core::array::from_fn(|i| f32_to_bf16_scalar(a[i]));
	assert_eq!(&got[0..4], &expect_low);
	assert_eq!(&got[4..8], &expect_high);
}

#[test]
fn cvtne2ps_pbh_u16x8_composes_with_cvtneps_pbh_u16x4_on_each_half() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 4] = [1.5, -2.5, 3.5, -4.5];
	let b: [f32; 4] = [-1.5, 2.5, -3.5, 4.5];
	let combined = t.cvtne2ps_pbh_u16x8(a, b);
	let a_alone = t.cvtneps_pbh_u16x4(a);
	let b_alone = t.cvtneps_pbh_u16x4(b);
	assert_eq!(&combined[0..4], &b_alone);
	assert_eq!(&combined[4..8], &a_alone);
}

macro_rules! masked_binop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $Elem:ty, $width:literal, $a:expr, $b:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512FVl::detect() else { return };
			let a: [$Elem; $width] = $a;
			let b: [$Elem; $width] = $b;
			let src: [$Elem; $width] = $src;
			let mask: u8 = $mask_val;
			let op = $op;
			let merge_expect: [$Elem; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect: [$Elem; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

masked_binop_test!(
	add_f32x4_masked_matches_scalar, add_f32x4_merge_masked, add_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32), [2.0f32; 4], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x5u8, |x: f32, y: f32| x + y
);
masked_binop_test!(
	sub_f32x4_masked_matches_scalar, sub_f32x4_merge_masked, sub_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32 * 10.0), [2.0f32; 4], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x5u8, |x: f32, y: f32| x - y
);
masked_binop_test!(
	mul_f32x4_masked_matches_scalar, mul_f32x4_merge_masked, mul_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32), [2.0f32; 4], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x5u8, |x: f32, y: f32| x * y
);
masked_binop_test!(
	div_f32x4_masked_matches_scalar, div_f32x4_merge_masked, div_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32 * 10.0), [2.0f32; 4], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x5u8, |x: f32, y: f32| x / y
);
masked_binop_test!(
	min_f32x4_masked_matches_scalar, min_f32x4_merge_masked, min_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32), core::array::from_fn(|i| (4 - i) as f32),
	core::array::from_fn(|i| -(i as f32) - 100.0), 0x5u8, |x: f32, y: f32| x.min(y)
);
masked_binop_test!(
	max_f32x4_masked_matches_scalar, max_f32x4_merge_masked, max_f32x4_zero_masked, f32, 4,
	core::array::from_fn(|i| (i + 1) as f32), core::array::from_fn(|i| (4 - i) as f32),
	core::array::from_fn(|i| -(i as f32) - 100.0), 0x5u8, |x: f32, y: f32| x.max(y)
);

masked_binop_test!(
	add_f32x8_masked_matches_scalar, add_f32x8_merge_masked, add_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32), [2.0f32; 8], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x55u8, |x: f32, y: f32| x + y
);
masked_binop_test!(
	sub_f32x8_masked_matches_scalar, sub_f32x8_merge_masked, sub_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32 * 10.0), [2.0f32; 8], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x55u8, |x: f32, y: f32| x - y
);
masked_binop_test!(
	mul_f32x8_masked_matches_scalar, mul_f32x8_merge_masked, mul_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32), [2.0f32; 8], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x55u8, |x: f32, y: f32| x * y
);
masked_binop_test!(
	div_f32x8_masked_matches_scalar, div_f32x8_merge_masked, div_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32 * 10.0), [2.0f32; 8], core::array::from_fn(|i| -(i as f32) - 100.0),
	0x55u8, |x: f32, y: f32| x / y
);
masked_binop_test!(
	min_f32x8_masked_matches_scalar, min_f32x8_merge_masked, min_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32), core::array::from_fn(|i| (8 - i) as f32),
	core::array::from_fn(|i| -(i as f32) - 100.0), 0x55u8, |x: f32, y: f32| x.min(y)
);
masked_binop_test!(
	max_f32x8_masked_matches_scalar, max_f32x8_merge_masked, max_f32x8_zero_masked, f32, 8,
	core::array::from_fn(|i| (i + 1) as f32), core::array::from_fn(|i| (8 - i) as f32),
	core::array::from_fn(|i| -(i as f32) - 100.0), 0x55u8, |x: f32, y: f32| x.max(y)
);

masked_binop_test!(
	add_f64x2_masked_matches_scalar, add_f64x2_merge_masked, add_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64), [2.0f64; 2], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x1u8, |x: f64, y: f64| x + y
);
masked_binop_test!(
	sub_f64x2_masked_matches_scalar, sub_f64x2_merge_masked, sub_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64 * 10.0), [2.0f64; 2], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x1u8, |x: f64, y: f64| x - y
);
masked_binop_test!(
	mul_f64x2_masked_matches_scalar, mul_f64x2_merge_masked, mul_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64), [2.0f64; 2], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x1u8, |x: f64, y: f64| x * y
);
masked_binop_test!(
	div_f64x2_masked_matches_scalar, div_f64x2_merge_masked, div_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64 * 10.0), [2.0f64; 2], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x1u8, |x: f64, y: f64| x / y
);
masked_binop_test!(
	min_f64x2_masked_matches_scalar, min_f64x2_merge_masked, min_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64), core::array::from_fn(|i| (2 - i) as f64),
	core::array::from_fn(|i| -(i as f64) - 100.0), 0x1u8, |x: f64, y: f64| x.min(y)
);
masked_binop_test!(
	max_f64x2_masked_matches_scalar, max_f64x2_merge_masked, max_f64x2_zero_masked, f64, 2,
	core::array::from_fn(|i| (i + 1) as f64), core::array::from_fn(|i| (2 - i) as f64),
	core::array::from_fn(|i| -(i as f64) - 100.0), 0x1u8, |x: f64, y: f64| x.max(y)
);

masked_binop_test!(
	add_f64x4_masked_matches_scalar, add_f64x4_merge_masked, add_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64), [2.0f64; 4], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x5u8, |x: f64, y: f64| x + y
);
masked_binop_test!(
	sub_f64x4_masked_matches_scalar, sub_f64x4_merge_masked, sub_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64 * 10.0), [2.0f64; 4], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x5u8, |x: f64, y: f64| x - y
);
masked_binop_test!(
	mul_f64x4_masked_matches_scalar, mul_f64x4_merge_masked, mul_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64), [2.0f64; 4], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x5u8, |x: f64, y: f64| x * y
);
masked_binop_test!(
	div_f64x4_masked_matches_scalar, div_f64x4_merge_masked, div_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64 * 10.0), [2.0f64; 4], core::array::from_fn(|i| -(i as f64) - 100.0),
	0x5u8, |x: f64, y: f64| x / y
);
masked_binop_test!(
	min_f64x4_masked_matches_scalar, min_f64x4_merge_masked, min_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64), core::array::from_fn(|i| (4 - i) as f64),
	core::array::from_fn(|i| -(i as f64) - 100.0), 0x5u8, |x: f64, y: f64| x.min(y)
);
masked_binop_test!(
	max_f64x4_masked_matches_scalar, max_f64x4_merge_masked, max_f64x4_zero_masked, f64, 4,
	core::array::from_fn(|i| (i + 1) as f64), core::array::from_fn(|i| (4 - i) as f64),
	core::array::from_fn(|i| -(i as f64) - 100.0), 0x5u8, |x: f64, y: f64| x.max(y)
);

masked_binop_test!(
	add_i32x4_masked_matches_scalar, add_i32x4_merge_masked, add_i32x4_zero_masked, i32, 4,
	core::array::from_fn(|i| i as i32 + 1), [7i32; 4], core::array::from_fn(|i| -(i as i32) - 100),
	0x5u8, |x: i32, y: i32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i32x4_masked_matches_scalar, sub_i32x4_merge_masked, sub_i32x4_zero_masked, i32, 4,
	core::array::from_fn(|i| i as i32 + 100), [7i32; 4], core::array::from_fn(|i| -(i as i32) - 100),
	0x5u8, |x: i32, y: i32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_i32x4_masked_matches_scalar, mul_i32x4_merge_masked, mul_i32x4_zero_masked, i32, 4,
	core::array::from_fn(|i| i as i32 + 1), [3i32; 4], core::array::from_fn(|i| -(i as i32) - 1000),
	0x5u8, |x: i32, y: i32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_i32x4_masked_matches_scalar, min_i32x4_merge_masked, min_i32x4_zero_masked, i32, 4,
	core::array::from_fn(|i| i as i32), core::array::from_fn(|i| 4 - i as i32),
	core::array::from_fn(|i| -(i as i32) - 1000), 0x5u8, |x: i32, y: i32| x.min(y)
);
masked_binop_test!(
	max_i32x4_masked_matches_scalar, max_i32x4_merge_masked, max_i32x4_zero_masked, i32, 4,
	core::array::from_fn(|i| i as i32), core::array::from_fn(|i| 4 - i as i32),
	core::array::from_fn(|i| -(i as i32) - 1000), 0x5u8, |x: i32, y: i32| x.max(y)
);

masked_binop_test!(
	add_i32x8_masked_matches_scalar, add_i32x8_merge_masked, add_i32x8_zero_masked, i32, 8,
	core::array::from_fn(|i| i as i32 + 1), [7i32; 8], core::array::from_fn(|i| -(i as i32) - 100),
	0x55u8, |x: i32, y: i32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i32x8_masked_matches_scalar, sub_i32x8_merge_masked, sub_i32x8_zero_masked, i32, 8,
	core::array::from_fn(|i| i as i32 + 100), [7i32; 8], core::array::from_fn(|i| -(i as i32) - 100),
	0x55u8, |x: i32, y: i32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_i32x8_masked_matches_scalar, mul_i32x8_merge_masked, mul_i32x8_zero_masked, i32, 8,
	core::array::from_fn(|i| i as i32 + 1), [3i32; 8], core::array::from_fn(|i| -(i as i32) - 1000),
	0x55u8, |x: i32, y: i32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_i32x8_masked_matches_scalar, min_i32x8_merge_masked, min_i32x8_zero_masked, i32, 8,
	core::array::from_fn(|i| i as i32), core::array::from_fn(|i| 8 - i as i32),
	core::array::from_fn(|i| -(i as i32) - 1000), 0x55u8, |x: i32, y: i32| x.min(y)
);
masked_binop_test!(
	max_i32x8_masked_matches_scalar, max_i32x8_merge_masked, max_i32x8_zero_masked, i32, 8,
	core::array::from_fn(|i| i as i32), core::array::from_fn(|i| 8 - i as i32),
	core::array::from_fn(|i| -(i as i32) - 1000), 0x55u8, |x: i32, y: i32| x.max(y)
);

masked_binop_test!(
	add_u32x4_masked_matches_scalar, add_u32x4_merge_masked, add_u32x4_zero_masked, u32, 4,
	core::array::from_fn(|i| i as u32), [7u32; 4], core::array::from_fn(|i| i as u32 + 1000),
	0x5u8, |x: u32, y: u32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u32x4_masked_matches_scalar, sub_u32x4_merge_masked, sub_u32x4_zero_masked, u32, 4,
	core::array::from_fn(|i| i as u32 + 1000), [7u32; 4], core::array::from_fn(|i| i as u32),
	0x5u8, |x: u32, y: u32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_u32x4_masked_matches_scalar, mul_u32x4_merge_masked, mul_u32x4_zero_masked, u32, 4,
	core::array::from_fn(|i| i as u32 + 1), [3u32; 4], core::array::from_fn(|i| i as u32 + 1000),
	0x5u8, |x: u32, y: u32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_u32x4_masked_matches_scalar, min_u32x4_merge_masked, min_u32x4_zero_masked, u32, 4,
	core::array::from_fn(|i| i as u32), core::array::from_fn(|i| 4u32.wrapping_sub(i as u32)),
	core::array::from_fn(|i| i as u32 + 1000), 0x5u8, |x: u32, y: u32| x.min(y)
);
masked_binop_test!(
	max_u32x4_masked_matches_scalar, max_u32x4_merge_masked, max_u32x4_zero_masked, u32, 4,
	core::array::from_fn(|i| i as u32), core::array::from_fn(|i| 4u32.wrapping_sub(i as u32)),
	core::array::from_fn(|i| i as u32 + 1000), 0x5u8, |x: u32, y: u32| x.max(y)
);

masked_binop_test!(
	add_u32x8_masked_matches_scalar, add_u32x8_merge_masked, add_u32x8_zero_masked, u32, 8,
	core::array::from_fn(|i| i as u32), [7u32; 8], core::array::from_fn(|i| i as u32 + 1000),
	0x55u8, |x: u32, y: u32| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u32x8_masked_matches_scalar, sub_u32x8_merge_masked, sub_u32x8_zero_masked, u32, 8,
	core::array::from_fn(|i| i as u32 + 1000), [7u32; 8], core::array::from_fn(|i| i as u32),
	0x55u8, |x: u32, y: u32| x.wrapping_sub(y)
);
masked_binop_test!(
	mul_u32x8_masked_matches_scalar, mul_u32x8_merge_masked, mul_u32x8_zero_masked, u32, 8,
	core::array::from_fn(|i| i as u32 + 1), [3u32; 8], core::array::from_fn(|i| i as u32 + 1000),
	0x55u8, |x: u32, y: u32| x.wrapping_mul(y)
);
masked_binop_test!(
	min_u32x8_masked_matches_scalar, min_u32x8_merge_masked, min_u32x8_zero_masked, u32, 8,
	core::array::from_fn(|i| i as u32), core::array::from_fn(|i| 8u32.wrapping_sub(i as u32)),
	core::array::from_fn(|i| i as u32 + 1000), 0x55u8, |x: u32, y: u32| x.min(y)
);
masked_binop_test!(
	max_u32x8_masked_matches_scalar, max_u32x8_merge_masked, max_u32x8_zero_masked, u32, 8,
	core::array::from_fn(|i| i as u32), core::array::from_fn(|i| 8u32.wrapping_sub(i as u32)),
	core::array::from_fn(|i| i as u32 + 1000), 0x55u8, |x: u32, y: u32| x.max(y)
);

masked_binop_test!(
	add_i64x2_masked_matches_scalar, add_i64x2_merge_masked, add_i64x2_zero_masked, i64, 2,
	core::array::from_fn(|i| i as i64 + 1), [7i64; 2], core::array::from_fn(|i| -(i as i64) - 100),
	0x1u8, |x: i64, y: i64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i64x2_masked_matches_scalar, sub_i64x2_merge_masked, sub_i64x2_zero_masked, i64, 2,
	core::array::from_fn(|i| i as i64 + 100), [7i64; 2], core::array::from_fn(|i| -(i as i64) - 100),
	0x1u8, |x: i64, y: i64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_i64x2_masked_matches_scalar, min_i64x2_merge_masked, min_i64x2_zero_masked, i64, 2,
	core::array::from_fn(|i| i as i64), core::array::from_fn(|i| 2 - i as i64),
	core::array::from_fn(|i| -(i as i64) - 1000), 0x1u8, |x: i64, y: i64| x.min(y)
);
masked_binop_test!(
	max_i64x2_masked_matches_scalar, max_i64x2_merge_masked, max_i64x2_zero_masked, i64, 2,
	core::array::from_fn(|i| i as i64), core::array::from_fn(|i| 2 - i as i64),
	core::array::from_fn(|i| -(i as i64) - 1000), 0x1u8, |x: i64, y: i64| x.max(y)
);

masked_binop_test!(
	add_i64x4_masked_matches_scalar, add_i64x4_merge_masked, add_i64x4_zero_masked, i64, 4,
	core::array::from_fn(|i| i as i64 + 1), [7i64; 4], core::array::from_fn(|i| -(i as i64) - 100),
	0x5u8, |x: i64, y: i64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i64x4_masked_matches_scalar, sub_i64x4_merge_masked, sub_i64x4_zero_masked, i64, 4,
	core::array::from_fn(|i| i as i64 + 100), [7i64; 4], core::array::from_fn(|i| -(i as i64) - 100),
	0x5u8, |x: i64, y: i64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_i64x4_masked_matches_scalar, min_i64x4_merge_masked, min_i64x4_zero_masked, i64, 4,
	core::array::from_fn(|i| i as i64), core::array::from_fn(|i| 4 - i as i64),
	core::array::from_fn(|i| -(i as i64) - 1000), 0x5u8, |x: i64, y: i64| x.min(y)
);
masked_binop_test!(
	max_i64x4_masked_matches_scalar, max_i64x4_merge_masked, max_i64x4_zero_masked, i64, 4,
	core::array::from_fn(|i| i as i64), core::array::from_fn(|i| 4 - i as i64),
	core::array::from_fn(|i| -(i as i64) - 1000), 0x5u8, |x: i64, y: i64| x.max(y)
);

masked_binop_test!(
	add_u64x2_masked_matches_scalar, add_u64x2_merge_masked, add_u64x2_zero_masked, u64, 2,
	core::array::from_fn(|i| i as u64), [7u64; 2], core::array::from_fn(|i| i as u64 + 1000),
	0x1u8, |x: u64, y: u64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u64x2_masked_matches_scalar, sub_u64x2_merge_masked, sub_u64x2_zero_masked, u64, 2,
	core::array::from_fn(|i| i as u64 + 1000), [7u64; 2], core::array::from_fn(|i| i as u64),
	0x1u8, |x: u64, y: u64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_u64x2_masked_matches_scalar, min_u64x2_merge_masked, min_u64x2_zero_masked, u64, 2,
	core::array::from_fn(|i| i as u64), core::array::from_fn(|i| 2u64.wrapping_sub(i as u64)),
	core::array::from_fn(|i| i as u64 + 1000), 0x1u8, |x: u64, y: u64| x.min(y)
);
masked_binop_test!(
	max_u64x2_masked_matches_scalar, max_u64x2_merge_masked, max_u64x2_zero_masked, u64, 2,
	core::array::from_fn(|i| i as u64), core::array::from_fn(|i| 2u64.wrapping_sub(i as u64)),
	core::array::from_fn(|i| i as u64 + 1000), 0x1u8, |x: u64, y: u64| x.max(y)
);

masked_binop_test!(
	add_u64x4_masked_matches_scalar, add_u64x4_merge_masked, add_u64x4_zero_masked, u64, 4,
	core::array::from_fn(|i| i as u64), [7u64; 4], core::array::from_fn(|i| i as u64 + 1000),
	0x5u8, |x: u64, y: u64| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u64x4_masked_matches_scalar, sub_u64x4_merge_masked, sub_u64x4_zero_masked, u64, 4,
	core::array::from_fn(|i| i as u64 + 1000), [7u64; 4], core::array::from_fn(|i| i as u64),
	0x5u8, |x: u64, y: u64| x.wrapping_sub(y)
);
masked_binop_test!(
	min_u64x4_masked_matches_scalar, min_u64x4_merge_masked, min_u64x4_zero_masked, u64, 4,
	core::array::from_fn(|i| i as u64), core::array::from_fn(|i| 4u64.wrapping_sub(i as u64)),
	core::array::from_fn(|i| i as u64 + 1000), 0x5u8, |x: u64, y: u64| x.min(y)
);
masked_binop_test!(
	max_u64x4_masked_matches_scalar, max_u64x4_merge_masked, max_u64x4_zero_masked, u64, 4,
	core::array::from_fn(|i| i as u64), core::array::from_fn(|i| 4u64.wrapping_sub(i as u64)),
	core::array::from_fn(|i| i as u64 + 1000), 0x5u8, |x: u64, y: u64| x.max(y)
);

#[test]
fn detect_avx512f_vl_matches_features() {
	let fs = FeatureSet::detect();
	let expect = fs.contains(Feature::Avx512f) && fs.contains(Feature::Avx512vl);
	assert_eq!(Avx512FVl::detect().is_some(), expect);
}

// Bit-exact reference for `vpternlogd`/`vpternlogq`, same shape as
// `avx512f.rs`'s 512-bit tests (per-bit 3-input lookup into `imm8`).
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

macro_rules! ternarylogic_vl_test {
	(
		$name:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $width:literal,
		$Elem:ty, $Uns:ty, $ref:ident, $a:expr, $b:expr, $c:expr, $src:expr, $mask_val:expr, $imm8:literal
	) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512FVl::detect() else { return };
			let a: [$Elem; $width] = $a;
			let b: [$Elem; $width] = $b;
			let c: [$Elem; $width] = $c;
			let src: [$Elem; $width] = $src;
			let mask: u8 = $mask_val;

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

ternarylogic_vl_test!(
	ternarylogic_i32x4_matches_bit_lookup, ternarylogic_i32x4, ternarylogic_i32x4_merge_masked,
	ternarylogic_i32x4_zero_masked, 4, i32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as i32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as i32) * -0x0789_0ABC - 3),
	core::array::from_fn(|i| (i as i32) ^ 0x5A5A_5A5A_u32 as i32), core::array::from_fn(|i| -(i as i32) - 1000),
	0x5u8, 0x96
);
ternarylogic_vl_test!(
	ternarylogic_u32x4_matches_bit_lookup, ternarylogic_u32x4, ternarylogic_u32x4_merge_masked,
	ternarylogic_u32x4_zero_masked, 4, u32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as u32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as u32) * 0x0789_0ABC + 3),
	core::array::from_fn(|i| (i as u32) ^ 0x5A5A_5A5A), core::array::from_fn(|i| (i as u32) + 9000),
	0x5u8, 0xE8
);
ternarylogic_vl_test!(
	ternarylogic_i32x8_matches_bit_lookup, ternarylogic_i32x8, ternarylogic_i32x8_merge_masked,
	ternarylogic_i32x8_zero_masked, 8, i32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as i32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as i32) * -0x0789_0ABC - 3),
	core::array::from_fn(|i| (i as i32) ^ 0x5A5A_5A5A_u32 as i32), core::array::from_fn(|i| -(i as i32) - 1000),
	0x9Au8, 0x2D
);
ternarylogic_vl_test!(
	ternarylogic_u32x8_matches_bit_lookup, ternarylogic_u32x8, ternarylogic_u32x8_merge_masked,
	ternarylogic_u32x8_zero_masked, 8, u32, u32, ternarylogic_ref_u32,
	core::array::from_fn(|i| (i as u32) * 0x0123_4567 + 7), core::array::from_fn(|i| (i as u32) * 0x0789_0ABC + 3),
	core::array::from_fn(|i| (i as u32) ^ 0x5A5A_5A5A), core::array::from_fn(|i| (i as u32) + 9000),
	0x9Au8, 0x71
);
ternarylogic_vl_test!(
	ternarylogic_i64x2_matches_bit_lookup, ternarylogic_i64x2, ternarylogic_i64x2_merge_masked,
	ternarylogic_i64x2_zero_masked, 2, i64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as i64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as i64) * -0x0000_789A_BCDE - 3),
	core::array::from_fn(|i| (i as i64) ^ 0x5A5A_5A5A_5A5A_5A5A_u64 as i64), core::array::from_fn(|i| -(i as i64) - 1000),
	0x2u8, 0xCA
);
ternarylogic_vl_test!(
	ternarylogic_u64x2_matches_bit_lookup, ternarylogic_u64x2, ternarylogic_u64x2_merge_masked,
	ternarylogic_u64x2_zero_masked, 2, u64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as u64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as u64) * 0x0000_789A_BCDE + 3),
	core::array::from_fn(|i| (i as u64) ^ 0x5A5A_5A5A_5A5A_5A5A), core::array::from_fn(|i| (i as u64) + 9000),
	0x2u8, 0x1E
);
ternarylogic_vl_test!(
	ternarylogic_i64x4_matches_bit_lookup, ternarylogic_i64x4, ternarylogic_i64x4_merge_masked,
	ternarylogic_i64x4_zero_masked, 4, i64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as i64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as i64) * -0x0000_789A_BCDE - 3),
	core::array::from_fn(|i| (i as i64) ^ 0x5A5A_5A5A_5A5A_5A5A_u64 as i64), core::array::from_fn(|i| -(i as i64) - 1000),
	0x5u8, 0x96
);
ternarylogic_vl_test!(
	ternarylogic_u64x4_matches_bit_lookup, ternarylogic_u64x4, ternarylogic_u64x4_merge_masked,
	ternarylogic_u64x4_zero_masked, 4, u64, u64, ternarylogic_ref_u64,
	core::array::from_fn(|i| (i as u64) * 0x0001_2345_6789 + 7), core::array::from_fn(|i| (i as u64) * 0x0000_789A_BCDE + 3),
	core::array::from_fn(|i| (i as u64) ^ 0x5A5A_5A5A_5A5A_5A5A), core::array::from_fn(|i| (i as u64) + 9000),
	0x5u8, 0xE8
);

macro_rules! masked_bw_binop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $Elem:ty, $width:literal, $Mask:ty, $a:expr, $b:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512BwVl::detect() else { return };
			let a: [$Elem; $width] = $a;
			let b: [$Elem; $width] = $b;
			let src: [$Elem; $width] = $src;
			let mask: $Mask = $mask_val;
			let op = $op;
			let merge_expect: [$Elem; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect: [$Elem; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

masked_bw_binop_test!(
	add_i8x16_masked_matches_scalar, add_i8x16_merge_masked, add_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (i as i8).wrapping_mul(-3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_i8x16_masked_matches_scalar, sub_i8x16_merge_masked, sub_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (i as i8).wrapping_mul(-3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_i8x16_masked_matches_scalar, adds_i8x16_merge_masked, adds_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(20)), core::array::from_fn(|i| (i as i8).wrapping_mul(-15)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_i8x16_masked_matches_scalar, subs_i8x16_merge_masked, subs_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(20)), core::array::from_fn(|i| (i as i8).wrapping_mul(-15)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.saturating_sub(y)
);
masked_bw_binop_test!(
	min_i8x16_masked_matches_scalar, min_i8x16_merge_masked, min_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (15 - i as i8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.min(y)
);
masked_bw_binop_test!(
	max_i8x16_masked_matches_scalar, max_i8x16_merge_masked, max_i8x16_zero_masked, i8, 16, u16,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (15 - i as i8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5Au16, |x: i8, y: i8| x.max(y)
);

masked_bw_binop_test!(
	add_i8x32_masked_matches_scalar, add_i8x32_merge_masked, add_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (i as i8).wrapping_mul(-3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_i8x32_masked_matches_scalar, sub_i8x32_merge_masked, sub_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (i as i8).wrapping_mul(-3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_i8x32_masked_matches_scalar, adds_i8x32_merge_masked, adds_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(20)), core::array::from_fn(|i| (i as i8).wrapping_mul(-15)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_i8x32_masked_matches_scalar, subs_i8x32_merge_masked, subs_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(20)), core::array::from_fn(|i| (i as i8).wrapping_mul(-15)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.saturating_sub(y)
);
masked_bw_binop_test!(
	min_i8x32_masked_matches_scalar, min_i8x32_merge_masked, min_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (31 - i as i8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.min(y)
);
masked_bw_binop_test!(
	max_i8x32_masked_matches_scalar, max_i8x32_merge_masked, max_i8x32_zero_masked, i8, 32, u32,
	core::array::from_fn(|i| (i as i8).wrapping_mul(7)), core::array::from_fn(|i| (31 - i as i8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)), 0x5A5A_5A5Au32, |x: i8, y: i8| x.max(y)
);

masked_bw_binop_test!(
	add_u8x16_masked_matches_scalar, add_u8x16_merge_masked, add_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_u8x16_masked_matches_scalar, sub_u8x16_merge_masked, sub_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_u8x16_masked_matches_scalar, adds_u8x16_merge_masked, adds_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(20)), core::array::from_fn(|i| (i as u8).wrapping_mul(15)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_u8x16_masked_matches_scalar, subs_u8x16_merge_masked, subs_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(20)), core::array::from_fn(|i| (i as u8).wrapping_mul(15)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.saturating_sub(y)
);
masked_bw_binop_test!(
	min_u8x16_masked_matches_scalar, min_u8x16_merge_masked, min_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (15 - i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.min(y)
);
masked_bw_binop_test!(
	max_u8x16_masked_matches_scalar, max_u8x16_merge_masked, max_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (15 - i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16, |x: u8, y: u8| x.max(y)
);

masked_bw_binop_test!(
	add_u8x32_masked_matches_scalar, add_u8x32_merge_masked, add_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_u8x32_masked_matches_scalar, sub_u8x32_merge_masked, sub_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_u8x32_masked_matches_scalar, adds_u8x32_merge_masked, adds_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(20)), core::array::from_fn(|i| (i as u8).wrapping_mul(15)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_u8x32_masked_matches_scalar, subs_u8x32_merge_masked, subs_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(20)), core::array::from_fn(|i| (i as u8).wrapping_mul(15)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.saturating_sub(y)
);
masked_bw_binop_test!(
	min_u8x32_masked_matches_scalar, min_u8x32_merge_masked, min_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (31 - i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.min(y)
);
masked_bw_binop_test!(
	max_u8x32_masked_matches_scalar, max_u8x32_merge_masked, max_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(7)), core::array::from_fn(|i| (31 - i as u8).wrapping_mul(3)),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32, |x: u8, y: u8| x.max(y)
);

masked_bw_binop_test!(
	add_i16x8_masked_matches_scalar, add_i16x8_merge_masked, add_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_i16x8_masked_matches_scalar, sub_i16x8_merge_masked, sub_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_i16x8_masked_matches_scalar, adds_i16x8_merge_masked, adds_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(9000)), core::array::from_fn(|i| (i as i16).wrapping_mul(-8000)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_i16x8_masked_matches_scalar, subs_i16x8_merge_masked, subs_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(9000)), core::array::from_fn(|i| (i as i16).wrapping_mul(-8000)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.saturating_sub(y)
);
masked_bw_binop_test!(
	mul_i16x8_masked_matches_scalar, mul_i16x8_merge_masked, mul_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.wrapping_mul(y)
);
masked_bw_binop_test!(
	min_i16x8_masked_matches_scalar, min_i16x8_merge_masked, min_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (7 - i as i16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.min(y)
);
masked_bw_binop_test!(
	max_i16x8_masked_matches_scalar, max_i16x8_merge_masked, max_i16x8_zero_masked, i16, 8, u8,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (7 - i as i16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5Au8, |x: i16, y: i16| x.max(y)
);

masked_bw_binop_test!(
	add_i16x16_masked_matches_scalar, add_i16x16_merge_masked, add_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_i16x16_masked_matches_scalar, sub_i16x16_merge_masked, sub_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_i16x16_masked_matches_scalar, adds_i16x16_merge_masked, adds_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(9000)), core::array::from_fn(|i| (i as i16).wrapping_mul(-8000)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_i16x16_masked_matches_scalar, subs_i16x16_merge_masked, subs_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(9000)), core::array::from_fn(|i| (i as i16).wrapping_mul(-8000)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.saturating_sub(y)
);
masked_bw_binop_test!(
	mul_i16x16_masked_matches_scalar, mul_i16x16_merge_masked, mul_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (i as i16).wrapping_mul(-300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.wrapping_mul(y)
);
masked_bw_binop_test!(
	min_i16x16_masked_matches_scalar, min_i16x16_merge_masked, min_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (15 - i as i16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.min(y)
);
masked_bw_binop_test!(
	max_i16x16_masked_matches_scalar, max_i16x16_merge_masked, max_i16x16_zero_masked, i16, 16, u16,
	core::array::from_fn(|i| (i as i16).wrapping_mul(700)), core::array::from_fn(|i| (15 - i as i16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as i16).wrapping_neg().wrapping_sub(5000)), 0x5A5Au16, |x: i16, y: i16| x.max(y)
);

masked_bw_binop_test!(
	add_u16x8_masked_matches_scalar, add_u16x8_merge_masked, add_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_u16x8_masked_matches_scalar, sub_u16x8_merge_masked, sub_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_u16x8_masked_matches_scalar, adds_u16x8_merge_masked, adds_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(9000)), core::array::from_fn(|i| (i as u16).wrapping_mul(8000)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_u16x8_masked_matches_scalar, subs_u16x8_merge_masked, subs_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(9000)), core::array::from_fn(|i| (i as u16).wrapping_mul(8000)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.saturating_sub(y)
);
masked_bw_binop_test!(
	mul_u16x8_masked_matches_scalar, mul_u16x8_merge_masked, mul_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.wrapping_mul(y)
);
masked_bw_binop_test!(
	min_u16x8_masked_matches_scalar, min_u16x8_merge_masked, min_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (7 - i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.min(y)
);
masked_bw_binop_test!(
	max_u16x8_masked_matches_scalar, max_u16x8_merge_masked, max_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (7 - i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8, |x: u16, y: u16| x.max(y)
);

masked_bw_binop_test!(
	add_u16x16_masked_matches_scalar, add_u16x16_merge_masked, add_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.wrapping_add(y)
);
masked_bw_binop_test!(
	sub_u16x16_masked_matches_scalar, sub_u16x16_merge_masked, sub_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.wrapping_sub(y)
);
masked_bw_binop_test!(
	adds_u16x16_masked_matches_scalar, adds_u16x16_merge_masked, adds_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(9000)), core::array::from_fn(|i| (i as u16).wrapping_mul(8000)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.saturating_add(y)
);
masked_bw_binop_test!(
	subs_u16x16_masked_matches_scalar, subs_u16x16_merge_masked, subs_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(9000)), core::array::from_fn(|i| (i as u16).wrapping_mul(8000)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.saturating_sub(y)
);
masked_bw_binop_test!(
	mul_u16x16_masked_matches_scalar, mul_u16x16_merge_masked, mul_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.wrapping_mul(y)
);
masked_bw_binop_test!(
	min_u16x16_masked_matches_scalar, min_u16x16_merge_masked, min_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (15 - i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.min(y)
);
masked_bw_binop_test!(
	max_u16x16_masked_matches_scalar, max_u16x16_merge_masked, max_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| (15 - i as u16).wrapping_mul(300)),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16, |x: u16, y: u16| x.max(y)
);

masked_bw_binop_test!(
	avg_u8x16_masked_matches_scalar, avg_u8x16_merge_masked, avg_u8x16_zero_masked, u8, 16, u16,
	core::array::from_fn(|i| (i as u8).wrapping_mul(17)), core::array::from_fn(|i| 255 - i as u8),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5Au16,
	|x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8
);
masked_bw_binop_test!(
	avg_u8x32_masked_matches_scalar, avg_u8x32_merge_masked, avg_u8x32_zero_masked, u8, 32, u32,
	core::array::from_fn(|i| (i as u8).wrapping_mul(17)), core::array::from_fn(|i| 255 - i as u8),
	core::array::from_fn(|i| (i as u8).wrapping_add(200)), 0x5A5A_5A5Au32,
	|x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8
);
masked_bw_binop_test!(
	avg_u16x8_masked_matches_scalar, avg_u16x8_merge_masked, avg_u16x8_zero_masked, u16, 8, u8,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| 65535 - i as u16),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5Au8,
	|x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16
);
masked_bw_binop_test!(
	avg_u16x16_masked_matches_scalar, avg_u16x16_merge_masked, avg_u16x16_zero_masked, u16, 16, u16,
	core::array::from_fn(|i| (i as u16).wrapping_mul(700)), core::array::from_fn(|i| 65535 - i as u16),
	core::array::from_fn(|i| (i as u16).wrapping_add(50000)), 0x5A5Au16,
	|x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16
);

#[test]
fn detect_avx512bw_vl_matches_features() {
	let fs = FeatureSet::detect();
	let expect = fs.contains(Feature::Avx512bw) && fs.contains(Feature::Avx512vl);
	assert_eq!(Avx512BwVl::detect().is_some(), expect);
}

// Tests for DqVl/VnniVl/VbmiVl/Bf16Vl's merge/zero-masked forms. Oracle
// is always the already-tested unmasked op, not a fresh scalar closure -
// isolates the one new behavior (lane selection).
fn assert_merge_zero<T: Copy + PartialEq + core::fmt::Debug + Default, const N: usize>(
	mask: u64, expect: [T; N], merged: [T; N], zeroed: [T; N], src: [T; N],
) {
	for i in 0..N {
		let selected = (mask >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { T::default() }, "zero lane {i}");
	}
}

const DQVL_MASK8: u8 = 0xA7;

#[test]
fn mullo_i64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 2] = [3i64,7];
	let b: [i64; 2] = [5i64,11];
	let src: [i64; 2] = [100i64;2];
	let expect = t.mullo_i64x2(a, b);
	let merged = t.mullo_i64x2_merge_masked(src, DQVL_MASK8, a, b);
	let zeroed = t.mullo_i64x2_zero_masked(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn mullo_i64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = [3i64,7,9,2];
	let b: [i64; 4] = [5i64,11,3,4];
	let src: [i64; 4] = [100i64;4];
	let expect = t.mullo_i64x4(a, b);
	let merged = t.mullo_i64x4_merge_masked(src, DQVL_MASK8, a, b);
	let zeroed = t.mullo_i64x4_zero_masked(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn mullo_u64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 2] = [3u64,7];
	let b: [u64; 2] = [5u64,11];
	let src: [u64; 2] = [100u64;2];
	let expect = t.mullo_u64x2(a, b);
	let merged = t.mullo_u64x2_merge_masked(src, DQVL_MASK8, a, b);
	let zeroed = t.mullo_u64x2_zero_masked(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn mullo_u64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 4] = [3u64,7,9,2];
	let b: [u64; 4] = [5u64,11,3,4];
	let src: [u64; 4] = [100u64;4];
	let expect = t.mullo_u64x4(a, b);
	let merged = t.mullo_u64x4_merge_masked(src, DQVL_MASK8, a, b);
	let zeroed = t.mullo_u64x4_zero_masked(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn i64_to_f64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 2] = [-1000i64, 42];
	let src: [f64; 2] = [999.0f64; 2];
	let expect = t.i64_to_f64x2(a);
	let merged = t.i64_to_f64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.i64_to_f64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn i64_to_f64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = [-1000i64, -1, 0, 42];
	let src: [f64; 4] = [999.0f64; 4];
	let expect = t.i64_to_f64x4(a);
	let merged = t.i64_to_f64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.i64_to_f64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn u64_to_f64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 2] = [0u64, 42];
	let src: [f64; 2] = [999.0f64; 2];
	let expect = t.u64_to_f64x2(a);
	let merged = t.u64_to_f64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.u64_to_f64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn u64_to_f64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 4] = [0u64, 1, 2, 42];
	let src: [f64; 4] = [999.0f64; 4];
	let expect = t.u64_to_f64x4(a);
	let merged = t.u64_to_f64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.u64_to_f64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 2] = [-2.5f64, 2.5];
	let src: [i64; 2] = [-1i64; 2];
	let expect = t.f64_to_i64x2(a);
	let merged = t.f64_to_i64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_i64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [-2.5f64, 2.5, 0.5, 42.0];
	let src: [i64; 4] = [-1i64; 4];
	let expect = t.f64_to_i64x4(a);
	let merged = t.f64_to_i64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_i64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x2_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 2] = [-2.9f64, 2.9];
	let src: [i64; 2] = [-1i64; 2];
	let expect = t.f64_to_i64x2_trunc(a);
	let merged = t.f64_to_i64x2_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_i64x2_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x4_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [-2.9f64, 2.9, 0.9, 42.0];
	let src: [i64; 4] = [-1i64; 4];
	let expect = t.f64_to_i64x4_trunc(a);
	let merged = t.f64_to_i64x4_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_i64x4_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 2] = [0.5f64, 2.5];
	let src: [u64; 2] = [u64::MAX; 2];
	let expect = t.f64_to_u64x2(a);
	let merged = t.f64_to_u64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_u64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [0.5f64, 2.5, 3.5, 42.0];
	let src: [u64; 4] = [u64::MAX; 4];
	let expect = t.f64_to_u64x4(a);
	let merged = t.f64_to_u64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_u64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x2_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 2] = [0.9f64, 2.9];
	let src: [u64; 2] = [u64::MAX; 2];
	let expect = t.f64_to_u64x2_trunc(a);
	let merged = t.f64_to_u64x2_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_u64x2_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x4_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = [0.9f64, 2.9, 3.9, 42.0];
	let src: [u64; 4] = [u64::MAX; 4];
	let expect = t.f64_to_u64x4_trunc(a);
	let merged = t.f64_to_u64x4_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f64_to_u64x4_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_i64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [-2.5, 2.5, 999.0, 999.0];
	let src: [i64; 2] = [-1i64; 2];
	let expect = t.f32_to_i64x2(a);
	let merged = t.f32_to_i64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_i64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_i64x2_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [-2.9, 2.9, 999.0, 999.0];
	let src: [i64; 2] = [-1i64; 2];
	let expect = t.f32_to_i64x2_trunc(a);
	let merged = t.f32_to_i64x2_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_i64x2_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [0.5, 2.5, 999.0, 999.0];
	let src: [u64; 2] = [u64::MAX; 2];
	let expect = t.f32_to_u64x2(a);
	let merged = t.f32_to_u64x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_u64x2_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x2_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [0.9, 2.9, 999.0, 999.0];
	let src: [u64; 2] = [u64::MAX; 2];
	let expect = t.f32_to_u64x2_trunc(a);
	let merged = t.f32_to_u64x2_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_u64x2_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

// i64_to_f32x2/u64_to_f32x2: only the low 2 lanes are real (mask bits 0-1);
// the upper 2 carrier lanes are hardware-zeroed unconditionally, not
// merge/zero controlled: `assert_merge_zero` assumes mask width == lane
// count, which doesn't hold for this oversized-carrier shape.

#[test]
fn i64_to_f32x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 2] = [-1000, 42];
	let src: [f32; 4] = [999.0; 4];
	let expect = t.i64_to_f32x2(a);
	let merged = t.i64_to_f32x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.i64_to_f32x2_zero_masked(DQVL_MASK8, a);
	for i in 0..2 {
		let selected = (DQVL_MASK8 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0.0 }, "zero lane {i}");
	}
	assert_eq!(&merged[2..], &[0.0, 0.0], "merge upper carrier lanes always zero");
	assert_eq!(&zeroed[2..], &[0.0, 0.0], "zero upper carrier lanes always zero");
}

#[test]
fn u64_to_f32x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 2] = [0, 42];
	let src: [f32; 4] = [999.0; 4];
	let expect = t.u64_to_f32x2(a);
	let merged = t.u64_to_f32x2_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.u64_to_f32x2_zero_masked(DQVL_MASK8, a);
	for i in 0..2 {
		let selected = (DQVL_MASK8 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0.0 }, "zero lane {i}");
	}
	assert_eq!(&merged[2..], &[0.0, 0.0], "merge upper carrier lanes always zero");
	assert_eq!(&zeroed[2..], &[0.0, 0.0], "zero upper carrier lanes always zero");
}

#[test]
fn f32_to_i64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [-2.5, -0.5, 0.5, 3.5];
	let src: [i64; 4] = [-1i64; 4];
	let expect = t.f32_to_i64x4(a);
	let merged = t.f32_to_i64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_i64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_i64x4_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [-2.9, -0.9, 0.9, 3.9];
	let src: [i64; 4] = [-1i64; 4];
	let expect = t.f32_to_i64x4_trunc(a);
	let merged = t.f32_to_i64x4_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_i64x4_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [0.0, 0.5, 2.5, 999999.5];
	let src: [u64; 4] = [u64::MAX; 4];
	let expect = t.f32_to_u64x4(a);
	let merged = t.f32_to_u64x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_u64x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x4_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f32; 4] = [0.0, 0.9, 2.9, 999999.9];
	let src: [u64; 4] = [u64::MAX; 4];
	let expect = t.f32_to_u64x4_trunc(a);
	let merged = t.f32_to_u64x4_trunc_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.f32_to_u64x4_trunc_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn i64_to_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = [-1000, -1, 42, 123456];
	let src: [f32; 4] = [999.0; 4];
	let expect = t.i64_to_f32x4(a);
	let merged = t.i64_to_f32x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.i64_to_f32x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn u64_to_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [u64; 4] = [0, 1, 42, 999999];
	let src: [f32; 4] = [999.0; 4];
	let expect = t.u64_to_f32x4(a);
	let merged = t.u64_to_f32x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.u64_to_f32x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [3.0; 2];
	let b = [7.0; 2];
	let src = [999.0; 2];
	let expect = t.range_f64x2::<1>(a, b);
	let merged = t.range_f64x2_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.range_f64x2_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [3.0; 4];
	let b = [7.0; 4];
	let src = [999.0; 4];
	let expect = t.range_f64x4::<1>(a, b);
	let merged = t.range_f64x4_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.range_f64x4_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [3.0; 4];
	let b = [7.0; 4];
	let src = [999.0; 4];
	let expect = t.range_f32x4::<1>(a, b);
	let merged = t.range_f32x4_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.range_f32x4_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [3.0; 8];
	let b = [7.0; 8];
	let src = [999.0; 8];
	let expect = t.range_f32x8::<1>(a, b);
	let merged = t.range_f32x8_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.range_f32x8_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f64x2_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [2.5f64, -2.5];
	let src = [999.0; 2];
	let expect = t.reduce_f64x2::<3>(a);
	let merged = t.reduce_f64x2_merge_masked::<3>(src, DQVL_MASK8, a);
	let zeroed = t.reduce_f64x2_zero_masked::<3>(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f64x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [2.5f64, -2.5, 2.5, -2.5];
	let src = [999.0; 4];
	let expect = t.reduce_f64x4::<3>(a);
	let merged = t.reduce_f64x4_merge_masked::<3>(src, DQVL_MASK8, a);
	let zeroed = t.reduce_f64x4_zero_masked::<3>(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [2.5f32; 4];
	let src = [999.0; 4];
	let expect = t.reduce_f32x4::<3>(a);
	let merged = t.reduce_f32x4_merge_masked::<3>(src, DQVL_MASK8, a);
	let zeroed = t.reduce_f32x4_zero_masked::<3>(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [2.5f32; 8];
	let src = [999.0; 8];
	let expect = t.reduce_f32x8::<3>(a);
	let merged = t.reduce_f32x8_merge_masked::<3>(src, DQVL_MASK8, a);
	let zeroed = t.reduce_f32x8_zero_masked::<3>(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn fpclass_f64x2_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [f64::NAN, 1.0];
	let unmasked = t.fpclass_f64x2::<1>(a);
	assert_eq!(t.fpclass_f64x2_gated::<1>(DQVL_MASK8, a), unmasked & DQVL_MASK8);
	assert_eq!(t.fpclass_f64x2_gated::<1>(0, a), 0);
}

#[test]
fn fpclass_f64x4_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [f64::NAN, 1.0, 0.0, -1.0];
	let unmasked = t.fpclass_f64x4::<1>(a);
	assert_eq!(t.fpclass_f64x4_gated::<1>(DQVL_MASK8, a), unmasked & DQVL_MASK8);
	assert_eq!(t.fpclass_f64x4_gated::<1>(0, a), 0);
}

#[test]
fn fpclass_f32x4_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [f32::NAN, 1.0, 0.0, -1.0];
	let unmasked = t.fpclass_f32x4::<1>(a);
	assert_eq!(t.fpclass_f32x4_gated::<1>(DQVL_MASK8, a), unmasked & DQVL_MASK8);
	assert_eq!(t.fpclass_f32x4_gated::<1>(0, a), 0);
}

#[test]
fn fpclass_f32x8_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [f32::NAN, 1.0, 0.0, -1.0, 2.0, 3.0, 4.0, 5.0];
	let unmasked = t.fpclass_f32x8::<1>(a);
	assert_eq!(t.fpclass_f32x8_gated::<1>(DQVL_MASK8, a), unmasked & DQVL_MASK8);
	assert_eq!(t.fpclass_f32x8_gated::<1>(0, a), 0);
}

#[test]
fn broadcast_f32x2_to_x8_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [1.0f32, 2.0, 999.0, 999.0];
	let src = [777.0f32; 8];
	let expect = t.broadcast_f32x2_to_x8(a);
	let merged = t.broadcast_f32x2_to_x8_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.broadcast_f32x2_to_x8_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_i32x2_to_x8_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [10i32, 20, 999, 999];
	let src = [777i32; 8];
	let expect = t.broadcast_i32x2_to_x8(a);
	let merged = t.broadcast_i32x2_to_x8_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.broadcast_i32x2_to_x8_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_f64x2_to_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [1.0f64, 2.0];
	let src = [777.0f64; 4];
	let expect = t.broadcast_f64x2_to_x4(a);
	let merged = t.broadcast_f64x2_to_x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.broadcast_f64x2_to_x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_i64x2_to_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a = [10i64, 20];
	let src = [777i64; 4];
	let expect = t.broadcast_i64x2_to_x4(a);
	let merged = t.broadcast_i64x2_to_x4_merge_masked(src, DQVL_MASK8, a);
	let zeroed = t.broadcast_i64x2_to_x4_zero_masked(DQVL_MASK8, a);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn extract_f64x2_from_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = core::array::from_fn(|i| i as f64);
	let src: [f64; 2] = [777.0f64; 2];
	let expect = t.extract_f64x2_from_x4::<1>(a);
	let merged = t.extract_f64x2_from_x4_merge_masked::<1>(src, 0b01, a);
	let zeroed = t.extract_f64x2_from_x4_zero_masked::<1>(0b01, a);
	assert_merge_zero(0b01u64, expect, merged, zeroed, src);
}

#[test]
fn extract_i64x2_from_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = core::array::from_fn(|i| i as i64);
	let src: [i64; 2] = [777i64; 2];
	let expect = t.extract_i64x2_from_x4::<1>(a);
	let merged = t.extract_i64x2_from_x4_merge_masked::<1>(src, 0b01, a);
	let zeroed = t.extract_i64x2_from_x4_zero_masked::<1>(0b01, a);
	assert_merge_zero(0b01u64, expect, merged, zeroed, src);
}

#[test]
fn insert_f64x2_into_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [f64; 4] = core::array::from_fn(|i| i as f64);
	let b: [f64; 2] = [99.0, 98.0];
	let src: [f64; 4] = [777.0f64; 4];
	let expect = t.insert_f64x2_into_x4::<1>(a, b);
	let merged = t.insert_f64x2_into_x4_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.insert_f64x2_into_x4_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn insert_i64x2_into_x4_masked_matches_unmasked() {
	let Some(t) = Avx512DqVl::detect() else { return };
	let a: [i64; 4] = core::array::from_fn(|i| i as i64);
	let b: [i64; 2] = [99i64, 98];
	let src: [i64; 4] = [777i64; 4];
	let expect = t.insert_i64x2_into_x4::<1>(a, b);
	let merged = t.insert_i64x2_into_x4_merge_masked::<1>(src, DQVL_MASK8, a, b);
	let zeroed = t.insert_i64x2_into_x4_zero_masked::<1>(DQVL_MASK8, a, b);
	assert_merge_zero(DQVL_MASK8 as u64, expect, merged, zeroed, src);
}


const DQVL_MASK4: u8 = 0b1011;
const DQVL_MASK8B: u8 = 0xA7;

#[test]
fn dpbusd_i32x4_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 4] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 16] = core::array::from_fn(|i| (i % 20) as u8);
	let b: [i8; 16] = core::array::from_fn(|i| (i % 7) as i8);
	let expect = t.dpbusd_i32x4(src, a, b);
	let merged = t.dpbusd_i32x4_merge_masked(src, DQVL_MASK4, a, b);
	let zeroed = t.dpbusd_i32x4_zero_masked(DQVL_MASK4, src, a, b);
	assert_merge_zero(DQVL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn dpbusd_i32x8_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 32] = core::array::from_fn(|i| (i % 20) as u8);
	let b: [i8; 32] = core::array::from_fn(|i| (i % 7) as i8);
	let expect = t.dpbusd_i32x8(src, a, b);
	let merged = t.dpbusd_i32x8_merge_masked(src, DQVL_MASK8B, a, b);
	let zeroed = t.dpbusd_i32x8_zero_masked(DQVL_MASK8B, src, a, b);
	assert_merge_zero(DQVL_MASK8B as u64, expect, merged, zeroed, src);
}

#[test]
fn dpbusds_i32x4_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 4] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 16] = core::array::from_fn(|i| (i % 20) as u8);
	let b: [i8; 16] = core::array::from_fn(|i| (i % 7) as i8);
	let expect = t.dpbusds_i32x4(src, a, b);
	let merged = t.dpbusds_i32x4_merge_masked(src, DQVL_MASK4, a, b);
	let zeroed = t.dpbusds_i32x4_zero_masked(DQVL_MASK4, src, a, b);
	assert_merge_zero(DQVL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn dpbusds_i32x8_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 32] = core::array::from_fn(|i| (i % 20) as u8);
	let b: [i8; 32] = core::array::from_fn(|i| (i % 7) as i8);
	let expect = t.dpbusds_i32x8(src, a, b);
	let merged = t.dpbusds_i32x8_merge_masked(src, DQVL_MASK8B, a, b);
	let zeroed = t.dpbusds_i32x8_zero_masked(DQVL_MASK8B, src, a, b);
	assert_merge_zero(DQVL_MASK8B as u64, expect, merged, zeroed, src);
}

#[test]
fn dpwssd_i32x4_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 4] = core::array::from_fn(|i| i as i32 * 10);
	let a: [i16; 8] = core::array::from_fn(|i| (i % 20) as i16);
	let b: [i16; 8] = core::array::from_fn(|i| (i % 7) as i16);
	let expect = t.dpwssd_i32x4(src, a, b);
	let merged = t.dpwssd_i32x4_merge_masked(src, DQVL_MASK4, a, b);
	let zeroed = t.dpwssd_i32x4_zero_masked(DQVL_MASK4, src, a, b);
	assert_merge_zero(DQVL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn dpwssd_i32x8_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 10);
	let a: [i16; 16] = core::array::from_fn(|i| (i % 20) as i16);
	let b: [i16; 16] = core::array::from_fn(|i| (i % 7) as i16);
	let expect = t.dpwssd_i32x8(src, a, b);
	let merged = t.dpwssd_i32x8_merge_masked(src, DQVL_MASK8B, a, b);
	let zeroed = t.dpwssd_i32x8_zero_masked(DQVL_MASK8B, src, a, b);
	assert_merge_zero(DQVL_MASK8B as u64, expect, merged, zeroed, src);
}

#[test]
fn dpwssds_i32x4_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 4] = core::array::from_fn(|i| i as i32 * 10);
	let a: [i16; 8] = core::array::from_fn(|i| (i % 20) as i16);
	let b: [i16; 8] = core::array::from_fn(|i| (i % 7) as i16);
	let expect = t.dpwssds_i32x4(src, a, b);
	let merged = t.dpwssds_i32x4_merge_masked(src, DQVL_MASK4, a, b);
	let zeroed = t.dpwssds_i32x4_zero_masked(DQVL_MASK4, src, a, b);
	assert_merge_zero(DQVL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn dpwssds_i32x8_masked_matches_unmasked() {
	let Some(t) = Avx512VnniVl::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 10);
	let a: [i16; 16] = core::array::from_fn(|i| (i % 20) as i16);
	let b: [i16; 16] = core::array::from_fn(|i| (i % 7) as i16);
	let expect = t.dpwssds_i32x8(src, a, b);
	let merged = t.dpwssds_i32x8_merge_masked(src, DQVL_MASK8B, a, b);
	let zeroed = t.dpwssds_i32x8_zero_masked(DQVL_MASK8B, src, a, b);
	assert_merge_zero(DQVL_MASK8B as u64, expect, merged, zeroed, src);
}

const VBMIVL_MASK16: u16 = 0x9A37;
const VBMIVL_MASK32: u32 = 0x9A37_5C81;

#[test]
fn permutexvar_u8x32_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let src: [u8; 32] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.permutexvar_u8x32(a, b);
	let merged = t.permutexvar_u8x32_merge_masked(src, VBMIVL_MASK32, a, b);
	let zeroed = t.permutexvar_u8x32_zero_masked(VBMIVL_MASK32, a, b);
	assert_merge_zero(VBMIVL_MASK32 as u64, expect, merged, zeroed, src);
}

#[test]
fn permutexvar_u8x16_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let src: [u8; 16] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.permutexvar_u8x16(a, b);
	let merged = t.permutexvar_u8x16_merge_masked(src, VBMIVL_MASK16, a, b);
	let zeroed = t.permutexvar_u8x16_zero_masked(VBMIVL_MASK16, a, b);
	assert_merge_zero(VBMIVL_MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn multishift_u8x32_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let src: [u8; 32] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.multishift_u8x32(a, b);
	let merged = t.multishift_u8x32_merge_masked(src, VBMIVL_MASK32, a, b);
	let zeroed = t.multishift_u8x32_zero_masked(VBMIVL_MASK32, a, b);
	assert_merge_zero(VBMIVL_MASK32 as u64, expect, merged, zeroed, src);
}

#[test]
fn multishift_u8x16_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let src: [u8; 16] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.multishift_u8x16(a, b);
	let merged = t.multishift_u8x16_merge_masked(src, VBMIVL_MASK16, a, b);
	let zeroed = t.multishift_u8x16_zero_masked(VBMIVL_MASK16, a, b);
	assert_merge_zero(VBMIVL_MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn permutex2var_u8x32_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8);
	let idx: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let b: [u8; 32] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.permutex2var_u8x32(a, idx, b);
	let merged = t.permutex2var_u8x32_merge_masked(a, VBMIVL_MASK32, idx, b);
	let zeroed = t.permutex2var_u8x32_zero_masked(VBMIVL_MASK32, a, idx, b);
	assert_merge_zero(VBMIVL_MASK32 as u64, expect, merged, zeroed, a);
}

#[test]
fn permutex2var_u8x16_masked_matches_unmasked() {
	let Some(t) = Avx512VbmiVl::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8);
	let idx: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let b: [u8; 16] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.permutex2var_u8x16(a, idx, b);
	let merged = t.permutex2var_u8x16_merge_masked(a, VBMIVL_MASK16, idx, b);
	let zeroed = t.permutex2var_u8x16_zero_masked(VBMIVL_MASK16, a, idx, b);
	assert_merge_zero(VBMIVL_MASK16 as u64, expect, merged, zeroed, a);
}

const BF16VL_MASK16: u16 = 0x9A37;
const BF16VL_MASK8: u8 = 0xA7;
const BF16VL_MASK4: u8 = 0b1011;

#[test]
fn dpbf16_ps_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let src: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5);
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 8.0) * 0.25));
	let b: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 % 5.0) - 2.0));
	let expect = t.dpbf16_ps_f32x8(src, a, b);
	let merged = t.dpbf16_ps_f32x8_merge_masked(src, BF16VL_MASK8, a, b);
	let zeroed = t.dpbf16_ps_f32x8_zero_masked(BF16VL_MASK8, src, a, b);
	assert_merge_zero(BF16VL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn dpbf16_ps_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let src: [f32; 4] = core::array::from_fn(|i| i as f32 * 0.5);
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 4.0) * 0.25));
	let b: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 % 5.0) - 2.0));
	let expect = t.dpbf16_ps_f32x4(src, a, b);
	let merged = t.dpbf16_ps_f32x4_merge_masked(src, BF16VL_MASK4, a, b);
	let zeroed = t.dpbf16_ps_f32x4_zero_masked(BF16VL_MASK4, src, a, b);
	assert_merge_zero(BF16VL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtneps_pbh_u16x8_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 8] = core::array::from_fn(|i| (i as f32 - 4.0) * 1.5);
	let src: [u16; 8] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtneps_pbh_u16x8(a);
	let merged = t.cvtneps_pbh_u16x8_merge_masked(src, BF16VL_MASK8, a);
	let zeroed = t.cvtneps_pbh_u16x8_zero_masked(BF16VL_MASK8, a);
	assert_merge_zero(BF16VL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtneps_pbh_u16x4_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 4] = core::array::from_fn(|i| (i as f32 - 2.0) * 1.5);
	let src: [u16; 4] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtneps_pbh_u16x4(a);
	let merged = t.cvtneps_pbh_u16x4_merge_masked(src, BF16VL_MASK4, a);
	let zeroed = t.cvtneps_pbh_u16x4_zero_masked(BF16VL_MASK4, a);
	assert_merge_zero(BF16VL_MASK4 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtne2ps_pbh_u16x16_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 8] = core::array::from_fn(|i| i as f32 + 0.3);
	let b: [f32; 8] = core::array::from_fn(|i| -(i as f32) - 0.7);
	let src: [u16; 16] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtne2ps_pbh_u16x16(a, b);
	let merged = t.cvtne2ps_pbh_u16x16_merge_masked(src, BF16VL_MASK16, a, b);
	let zeroed = t.cvtne2ps_pbh_u16x16_zero_masked(BF16VL_MASK16, a, b);
	assert_merge_zero(BF16VL_MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtne2ps_pbh_u16x8_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	let a: [f32; 4] = core::array::from_fn(|i| i as f32 + 0.3);
	let b: [f32; 4] = core::array::from_fn(|i| -(i as f32) - 0.7);
	let src: [u16; 8] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtne2ps_pbh_u16x8(a, b);
	let merged = t.cvtne2ps_pbh_u16x8_merge_masked(src, BF16VL_MASK8, a, b);
	let zeroed = t.cvtne2ps_pbh_u16x8_zero_masked(BF16VL_MASK8, a, b);
	assert_merge_zero(BF16VL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtpbh_ps_f32x8_matches_scalar_reference() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::{bf16_to_f32_scalar, f32_to_bf16_scalar};
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 4.0) * 12.375));
	let got = t.cvtpbh_ps_f32x8(a);
	let expect: [f32; 8] = core::array::from_fn(|i| bf16_to_f32_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtpbh_ps_f32x4_matches_scalar_reference() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::{bf16_to_f32_scalar, f32_to_bf16_scalar};
	let a: [u16; 4] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 2.0) * 12.375));
	let got = t.cvtpbh_ps_f32x4(a);
	let expect: [f32; 4] = core::array::from_fn(|i| bf16_to_f32_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtpbh_ps_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 4.0) * 12.375));
	let src: [f32; 8] = core::array::from_fn(|i| 1000.0 + i as f32);
	let expect = t.cvtpbh_ps_f32x8(a);
	let merged = t.cvtpbh_ps_f32x8_merge_masked(src, BF16VL_MASK8, a);
	let zeroed = t.cvtpbh_ps_f32x8_zero_masked(BF16VL_MASK8, a);
	assert_merge_zero(BF16VL_MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn cvtpbh_ps_f32x4_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16Vl::detect() else { return };
	use super::super::avx512bf16::f32_to_bf16_scalar;
	let a: [u16; 4] = core::array::from_fn(|i| f32_to_bf16_scalar((i as f32 - 2.0) * 12.375));
	let src: [f32; 4] = core::array::from_fn(|i| 1000.0 + i as f32);
	let expect = t.cvtpbh_ps_f32x4(a);
	let merged = t.cvtpbh_ps_f32x4_merge_masked(src, BF16VL_MASK4, a);
	let zeroed = t.cvtpbh_ps_f32x4_zero_masked(BF16VL_MASK4, a);
	assert_merge_zero(BF16VL_MASK4 as u64, expect, merged, zeroed, src);
}

// `vrcp14`/`vrsqrt14` are hardware approximations, max relative error <= 2^-14 per SDM.
const APPROX_TOL14: f64 = 0.00006103515625; // 2^-14

#[test]
fn rcp14_f32x4_approximates_reciprocal() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f32; 4] = core::array::from_fn(|i| (i + 1) as f32);
	let got = t.rcp14_f32x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL14 as f32 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rcp14_f32x8_approximates_reciprocal() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f32; 8] = core::array::from_fn(|i| (i + 1) as f32);
	let got = t.rcp14_f32x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL14 as f32 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt14_f32x4_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f32; 4] = core::array::from_fn(|i| (i + 1) as f32 * (i + 1) as f32);
	let got = t.rsqrt14_f32x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL14 as f32 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt14_f32x8_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f32; 8] = core::array::from_fn(|i| (i + 1) as f32 * (i + 1) as f32);
	let got = t.rsqrt14_f32x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL14 as f32 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rcp14_f64x2_approximates_reciprocal() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f64; 2] = core::array::from_fn(|i| (i + 1) as f64);
	let got = t.rcp14_f64x2(a);
	for i in 0..2 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL14 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rcp14_f64x4_approximates_reciprocal() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f64; 4] = core::array::from_fn(|i| (i + 1) as f64);
	let got = t.rcp14_f64x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL14 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt14_f64x2_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f64; 2] = core::array::from_fn(|i| (i + 1) as f64 * (i + 1) as f64);
	let got = t.rsqrt14_f64x2(a);
	for i in 0..2 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL14 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt14_f64x4_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512FVl::detect() else { return };
	let a: [f64; 4] = core::array::from_fn(|i| (i + 1) as f64 * (i + 1) as f64);
	let got = t.rsqrt14_f64x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL14 * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}
