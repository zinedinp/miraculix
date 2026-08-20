use super::super::super::macros::{slice_binop_matches_scalar_test, slice_bitop_matches_scalar_bits_test};
use super::*;

/// x86-64 psABI baseline: SSE always present.
#[test]
#[cfg(target_arch = "x86_64")]
fn detect_finds_sse_on_x86_64() {
	assert!(Sse::detect().is_some());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn assume_baseline_matches_detect() {
	let via_detect = Sse::detect().expect("x86_64 always has SSE");
	let via_baseline = Sse::assume_baseline();
	assert_eq!(
		via_detect.add_f32x4([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]),
		via_baseline.add_f32x4([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
	);
}

#[test]
fn add_f32x4_sums_lanes() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.add_f32x4([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]), [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn sub_f32x4_subtracts_lanes() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.sub_f32x4([10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]), [9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn mul_f32x4_multiplies_lanes() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.mul_f32x4([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]), [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn div_f32x4_divides_lanes() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.div_f32x4([10.0, 20.0, 30.0, 40.0], [2.0, 4.0, 5.0, 8.0]), [5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn min_f32x4_picks_smaller_lane() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.min_f32x4([1.0, 20.0, -3.0, 4.0], [10.0, 2.0, 3.0, -4.0]), [1.0, 2.0, -3.0, -4.0]);
}

#[test]
fn max_f32x4_picks_larger_lane() {
	let Some(sse) = Sse::detect() else { return };
	assert_eq!(sse.max_f32x4([1.0, 20.0, -3.0, 4.0], [10.0, 2.0, 3.0, -4.0]), [10.0, 20.0, 3.0, 4.0]);
}

/// Lanes match scalar add/sub/mul/div/min/max.
#[test]
fn matches_scalar_on_random_lanes() {
	let Some(sse) = Sse::detect() else { return };
	let a: [f32; 4] = [17.5, -3.25, 0.0, 1e6];
	let b: [f32; 4] = [-240.75, 10.0, -3.5, 255.0];

	let expect_add: [f32; 4] = core::array::from_fn(|i| a[i] + b[i]);
	let expect_sub: [f32; 4] = core::array::from_fn(|i| a[i] - b[i]);
	let expect_mul: [f32; 4] = core::array::from_fn(|i| a[i] * b[i]);
	let expect_div: [f32; 4] = core::array::from_fn(|i| a[i] / b[i]);
	let expect_min: [f32; 4] = core::array::from_fn(|i| a[i].min(b[i]));
	let expect_max: [f32; 4] = core::array::from_fn(|i| a[i].max(b[i]));

	assert_eq!(sse.add_f32x4(a, b), expect_add);
	assert_eq!(sse.sub_f32x4(a, b), expect_sub);
	assert_eq!(sse.mul_f32x4(a, b), expect_mul);
	assert_eq!(sse.div_f32x4(a, b), expect_div);
	assert_eq!(sse.min_f32x4(a, b), expect_min);
	assert_eq!(sse.max_f32x4(a, b), expect_max);
}

slice_binop_matches_scalar_test!(add_f32_slice_matches_scalar_for_various_lengths, Sse, add_f32_slice, |x, y| x + y, f32);
slice_binop_matches_scalar_test!(sub_f32_slice_matches_scalar_for_various_lengths, Sse, sub_f32_slice, |x, y| x - y, f32);
slice_binop_matches_scalar_test!(mul_f32_slice_matches_scalar_for_various_lengths, Sse, mul_f32_slice, |x, y| x * y, f32);
slice_binop_matches_scalar_test!(div_f32_slice_matches_scalar_for_various_lengths, Sse, div_f32_slice, |x, y| x / y, f32);
slice_binop_matches_scalar_test!(min_f32_slice_matches_scalar_for_various_lengths, Sse, min_f32_slice, |x, y| x.min(y), f32);
slice_binop_matches_scalar_test!(max_f32_slice_matches_scalar_for_various_lengths, Sse, max_f32_slice, |x, y| x.max(y), f32);

// Bit-exact: true mask is all-1s (a NaN bit pattern), so PartialEq is useless.
slice_bitop_matches_scalar_bits_test!(
	cmpeq_f32_slice_matches_scalar_bits, Sse, cmpeq_f32_slice,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
slice_bitop_matches_scalar_bits_test!(
	cmplt_f32_slice_matches_scalar_bits, Sse, cmplt_f32_slice,
	|x, y| if x < y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
slice_bitop_matches_scalar_bits_test!(
	cmpgt_f32_slice_matches_scalar_bits, Sse, cmpgt_f32_slice,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);

#[test]
fn cmpeq_f32x4_nan_is_false() {
	let Some(sse) = Sse::detect() else { return };
	let nan = f32::NAN;
	let r = sse.cmpeq_f32x4([nan, 1.0, nan, 2.0], [nan, 1.0, 0.0, 3.0]);
	assert_eq!(r[0].to_bits(), 0);
	assert_eq!(r[1].to_bits(), !0u32);
	assert_eq!(r[2].to_bits(), 0);
	assert_eq!(r[3].to_bits(), 0);
}

#[test]
fn and_f32x4_matches_bitwise_and() {
	let Some(sse) = Sse::detect() else { return };
	let a = [1.0f32, -1.0, 0.0, 1e6];
	let b = [1.0f32, 1.0, 1.0, 1.0];
	let expect: [f32; 4] = core::array::from_fn(|i| f32::from_bits(a[i].to_bits() & b[i].to_bits()));
	assert_eq!(sse.and_f32x4(a, b), expect);
}

#[test]
fn xor_f32x4_matches_bitwise_xor() {
	let Some(sse) = Sse::detect() else { return };
	let a = [1.0f32, -1.0, 0.0, 1e6];
	let b = [1.0f32, 1.0, 1.0, 1.0];
	let expect: [f32; 4] = core::array::from_fn(|i| f32::from_bits(a[i].to_bits() ^ b[i].to_bits()));
	assert_eq!(sse.xor_f32x4(a, b), expect);
}

// Bitwise results can land on a NaN bit pattern from non-NaN inputs; NaN != NaN
// under IEEE equality even when bit-identical, so compare `.to_bits()`, not the
// `f32` values themselves (unlike `slice_binop_matches_scalar_test!`'s plain `assert_eq!`).
slice_bitop_matches_scalar_bits_test!(
	and_f32_slice_matches_scalar_for_various_lengths, Sse, and_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	or_f32_slice_matches_scalar_for_various_lengths, Sse, or_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	xor_f32_slice_matches_scalar_for_various_lengths, Sse, xor_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f32_slice_matches_scalar_for_various_lengths, Sse, andnot_f32_slice,
	|x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()), f32
);

#[test]
fn movemask_f32x4_matches_sign_bits() {
	let Some(sse) = Sse::detect() else { return };
	let a = [-1.0f32, 1.0, -2.0, 0.0];
	assert_eq!(sse.movemask_f32x4(a), 0b0101);
}

// `sqrtps` is correctly rounded (IEEE), so an exact scalar match is expected.
#[test]
fn sqrt_f32x4_matches_scalar_sqrt() {
	let Some(sse) = Sse::detect() else { return };
	let a = [1.0f32, 4.0, 9.0, 2.0];
	let expect: [f32; 4] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(sse.sqrt_f32x4(a), expect);
}

// `rcpps`/`rsqrtps` are hardware approximations, max relative error < 1.5*2^-12 per SDM.
const APPROX_TOL: f32 = 1.5 * 0.000_244_140_63; // 1.5 * 2^-12

#[test]
fn rcp_f32x4_approximates_reciprocal() {
	let Some(sse) = Sse::detect() else { return };
	let a = [1.0f32, 2.0, 4.0, 100.0];
	let got = sse.rcp_f32x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt_f32x4_approximates_reciprocal_sqrt() {
	let Some(sse) = Sse::detect() else { return };
	let a = [1.0f32, 4.0, 16.0, 100.0];
	let got = sse.rsqrt_f32x4(a);
	for i in 0..4 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}
