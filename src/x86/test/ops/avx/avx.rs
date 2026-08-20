use super::super::super::macros::{slice_binop_matches_scalar_test, slice_bitop_matches_scalar_bits_test};
use super::*;

/// One-directional, not equality: `from_level` is a coarse `GenericLevel`
/// bucket (`V3` needs the whole AVX2+BMI1+BMI2+F16C+FMA+LZCNT+MOVBE+XSAVE
/// bundle), so it can under-detect real hardware that has raw AVX without
/// the rest of that bundle (e.g. Sandy/Ivy Bridge: confirmed via `sde64
/// -snb`, where `Avx::detect()` correctly finds `Some` but `from_level`
/// resolves to `V2` and returns `None`). `detect()` is never wrong the
/// other way: whatever `from_level` finds, raw detection must find too.
#[test]
fn from_level_agreeing_implies_detect_agrees() {
	let level = GenericLevel::detect(FeatureSet::detect());
	if Avx::from_level(level).is_some() {
		assert!(Avx::detect().is_some());
	}
}

#[test]
fn add_f32x8_sums_lanes() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
	assert_eq!(v3.add_f32x8(a, b), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
}

#[test]
fn sub_f32x8_subtracts_lanes() {
	let Some(v3) = Avx::detect() else { return };
	let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
	let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	assert_eq!(v3.sub_f32x8(a, b), [9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]);
}

#[test]
fn mul_f32x8_multiplies_lanes() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
	assert_eq!(v3.mul_f32x8(a, b), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
}

#[test]
fn div_f32x8_divides_lanes() {
	let Some(v3) = Avx::detect() else { return };
	let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
	let b = [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0];
	assert_eq!(v3.div_f32x8(a, b), [5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn min_f32x8_picks_smaller_lane() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0, 20.0, -3.0, 4.0, 5.0, -60.0, 7.0, 8.0];
	let b = [10.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 80.0];
	assert_eq!(v3.min_f32x8(a, b), [1.0, 2.0, -3.0, -4.0, 5.0, -60.0, -7.0, 8.0]);
}

#[test]
fn max_f32x8_picks_larger_lane() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0, 20.0, -3.0, 4.0, 5.0, -60.0, 7.0, 8.0];
	let b = [10.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 80.0];
	assert_eq!(v3.max_f32x8(a, b), [10.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0, 80.0]);
}

/// Lanes match scalar add/sub/mul/div/min/max.
#[test]
fn matches_scalar_on_random_lanes() {
	let Some(v3) = Avx::detect() else { return };
	let a: [f32; 8] = [17.5, -3.25, 0.0, 1e6, -1e-3, 42.0, 8.25, -8.25];
	let b: [f32; 8] = [-240.75, 10.0, -3.5, 255.0, 1.0, -42.0, 8.25, 8.25];

	let expect_add: [f32; 8] = core::array::from_fn(|i| a[i] + b[i]);
	let expect_sub: [f32; 8] = core::array::from_fn(|i| a[i] - b[i]);
	let expect_mul: [f32; 8] = core::array::from_fn(|i| a[i] * b[i]);
	let expect_div: [f32; 8] = core::array::from_fn(|i| a[i] / b[i]);
	let expect_min: [f32; 8] = core::array::from_fn(|i| a[i].min(b[i]));
	let expect_max: [f32; 8] = core::array::from_fn(|i| a[i].max(b[i]));

	assert_eq!(v3.add_f32x8(a, b), expect_add);
	assert_eq!(v3.sub_f32x8(a, b), expect_sub);
	assert_eq!(v3.mul_f32x8(a, b), expect_mul);
	assert_eq!(v3.div_f32x8(a, b), expect_div);
	assert_eq!(v3.min_f32x8(a, b), expect_min);
	assert_eq!(v3.max_f32x8(a, b), expect_max);
}

slice_binop_matches_scalar_test!(add_f32_slice_matches_scalar_for_various_lengths, Avx, add_f32_slice, |x, y| x + y, f32);
slice_binop_matches_scalar_test!(sub_f32_slice_matches_scalar_for_various_lengths, Avx, sub_f32_slice, |x, y| x - y, f32);
slice_binop_matches_scalar_test!(mul_f32_slice_matches_scalar_for_various_lengths, Avx, mul_f32_slice, |x, y| x * y, f32);
slice_binop_matches_scalar_test!(div_f32_slice_matches_scalar_for_various_lengths, Avx, div_f32_slice, |x, y| x / y, f32);
slice_binop_matches_scalar_test!(min_f32_slice_matches_scalar_for_various_lengths, Avx, min_f32_slice, |x, y| x.min(y), f32);
slice_binop_matches_scalar_test!(max_f32_slice_matches_scalar_for_various_lengths, Avx, max_f32_slice, |x, y| x.max(y), f32);
slice_bitop_matches_scalar_bits_test!(
	cmpeq_f32_slice_matches_scalar_bits, Avx, cmpeq_f32_slice,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
slice_bitop_matches_scalar_bits_test!(
	cmpgt_f32_slice_matches_scalar_bits, Avx, cmpgt_f32_slice,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
slice_bitop_matches_scalar_bits_test!(
	cmpeq_f64_slice_matches_scalar_bits, Avx, cmpeq_f64_slice,
	|x, y| if x == y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64
);

slice_binop_matches_scalar_test!(add_f64_slice_matches_scalar, Avx, add_f64_slice, |x, y| x + y, f64);
slice_binop_matches_scalar_test!(sub_f64_slice_matches_scalar, Avx, sub_f64_slice, |x, y| x - y, f64);
slice_binop_matches_scalar_test!(mul_f64_slice_matches_scalar, Avx, mul_f64_slice, |x, y| x * y, f64);
slice_binop_matches_scalar_test!(div_f64_slice_matches_scalar, Avx, div_f64_slice, |x, y| x / y, f64);
slice_binop_matches_scalar_test!(min_f64_slice_matches_scalar, Avx, min_f64_slice, |x, y| x.min(y), f64);
slice_binop_matches_scalar_test!(max_f64_slice_matches_scalar, Avx, max_f64_slice, |x, y| x.max(y), f64);

#[test]
fn and_f32x8_masks_off_sign_bit() {
	let Some(v3) = Avx::detect() else { return };
	let a = [-1.5f32; 8];
	let b = [f32::from_bits(0x7fff_ffff); 8];
	assert_eq!(v3.and_f32x8(a, b), [1.5f32; 8]);
}

slice_bitop_matches_scalar_bits_test!(
	and_f32_slice_matches_scalar, Avx, and_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	or_f32_slice_matches_scalar, Avx, or_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	xor_f32_slice_matches_scalar, Avx, xor_f32_slice,
	|x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f32_slice_matches_scalar, Avx, andnot_f32_slice,
	|x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()), f32
);
slice_bitop_matches_scalar_bits_test!(
	and_f64_slice_matches_scalar, Avx, and_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	or_f64_slice_matches_scalar, Avx, or_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	xor_f64_slice_matches_scalar, Avx, xor_f64_slice,
	|x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()), f64
);
slice_bitop_matches_scalar_bits_test!(
	andnot_f64_slice_matches_scalar, Avx, andnot_f64_slice,
	|x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()), f64
);

#[test]
fn movemask_f32x8_matches_sign_bits() {
	let Some(v3) = Avx::detect() else { return };
	let a = [-1.0f32, 1.0, -2.0, 0.0, -3.0, 3.0, 0.0, -4.0];
	assert_eq!(v3.movemask_f32x8(a), 0b1001_0101);
}

#[test]
fn movemask_f64x4_matches_sign_bits() {
	let Some(v3) = Avx::detect() else { return };
	let a = [-1.0f64, 1.0, 0.0, -2.0];
	assert_eq!(v3.movemask_f64x4(a), 0b1001);
}

#[test]
fn sqrt_f32x8_matches_scalar_sqrt() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0f32, 4.0, 9.0, 2.0, 16.0, 25.0, 0.0, 100.0];
	let expect: [f32; 8] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(v3.sqrt_f32x8(a), expect);
}

#[test]
fn sqrt_f64x4_matches_scalar_sqrt() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0f64, 4.0, 9.0, 2.0];
	let expect: [f64; 4] = core::array::from_fn(|i| a[i].sqrt());
	assert_eq!(v3.sqrt_f64x4(a), expect);
}

// `vrcpps`/`vrsqrtps` are hardware approximations, max relative error < 1.5*2^-12 per SDM.
const APPROX_TOL: f32 = 1.5 * 0.000_244_140_63; // 1.5 * 2^-12

#[test]
fn rcp_f32x8_approximates_reciprocal() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0f32, 2.0, 4.0, 100.0, 0.5, 8.0, 3.0, 7.0];
	let got = v3.rcp_f32x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= APPROX_TOL * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

#[test]
fn rsqrt_f32x8_approximates_reciprocal_sqrt() {
	let Some(v3) = Avx::detect() else { return };
	let a = [1.0f32, 4.0, 16.0, 100.0, 25.0, 9.0, 64.0, 49.0];
	let got = v3.rsqrt_f32x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= APPROX_TOL * expect.abs(), "lane {i}: got {}, expect ~{expect}", got[i]);
	}
}

// Hand-worked per Intel's per-128-bit-lane semantics (SDM `VUNPCKLPS`/
// `VUNPCKHPS`/`VSHUFPS`/`VPERM2F128`): each 256-bit op treats lanes 0-3
// and 4-7 independently, `permute2f128` being the only one that instead
// picks whole 128-bit halves.
#[test]
fn unpacklo_f32x8_interleaves_per_lane() {
	let Some(v3) = Avx::detect() else { return };
	let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
	assert_eq!(v3.unpacklo_f32x8(a, b), [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]);
}

#[test]
fn unpackhi_f32x8_interleaves_per_lane() {
	let Some(v3) = Avx::detect() else { return };
	let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
	assert_eq!(v3.unpackhi_f32x8(a, b), [2.0, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0]);
}

#[test]
fn shuffle_f32x8_selects_per_lane() {
	let Some(v3) = Avx::detect() else { return };
	let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
	// 0x44 = 0b01_00_01_00: dst = [a0, a1, b0, b1] per 128-bit lane.
	assert_eq!(v3.shuffle_f32x8::<0x44>(a, b), [0.0, 1.0, 10.0, 11.0, 4.0, 5.0, 14.0, 15.0]);
}

#[test]
fn permute2f128_f32x8_selects_whole_halves() {
	let Some(v3) = Avx::detect() else { return };
	let a = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let b = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
	// 0x20: low 128 of dst <- a's low 128, high 128 <- b's low 128.
	assert_eq!(v3.permute2f128_f32x8::<0x20>(a, b), [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
	// 0x31: low 128 of dst <- a's high 128, high 128 <- b's high 128.
	assert_eq!(v3.permute2f128_f32x8::<0x31>(a, b), [4.0, 5.0, 6.0, 7.0, 14.0, 15.0, 16.0, 17.0]);
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
fn conj_c32x8_negates_imaginary_lanes() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 8.0];
	assert_eq!(avx.conj_c32x8(a), [1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, -8.0]);
}

#[test]
fn mul_c32x8_matches_scalar_complex_multiply() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0, 2.0, -3.0, 0.5, 2.0, -2.0, 0.0, 1.0];
	let b = [5.0, -1.0, 2.0, 4.0, -1.5, 3.0, 4.0, -4.0];
	let mut expect = [0f32; 8];
	scalar_mul_c(&a, &b, false, &mut expect);
	assert_eq!(avx.mul_c32x8(a, b), expect);
}

#[test]
fn conj_mul_c32x8_matches_scalar_conjugate_multiply() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0, 2.0, -3.0, 0.5, 2.0, -2.0, 0.0, 1.0];
	let b = [5.0, -1.0, 2.0, 4.0, -1.5, 3.0, 4.0, -4.0];
	let mut expect = [0f32; 8];
	scalar_mul_c(&a, &b, true, &mut expect);
	assert_eq!(avx.conj_mul_c32x8(a, b), expect);
	assert_eq!(avx.conj_mul_c32x8(a, b), avx.mul_c32x8(avx.conj_c32x8(a), b));
}

#[test]
fn abs2_c32x8_matches_scalar_squared_magnitude() {
	let Some(avx) = Avx::detect() else { return };
	let a = [3.0, 4.0, -1.0, 2.0, 0.0, -5.0, 2.0, -2.0];
	assert_eq!(avx.abs2_c32x8(a), [25.0, 25.0, 5.0, 5.0, 25.0, 25.0, 8.0, 8.0]);
}

#[test]
fn conj_c64x4_negates_imaginary_lanes() {
	let Some(avx) = Avx::detect() else { return };
	assert_eq!(avx.conj_c64x4([1.5, -2.5, 3.0, 4.0]), [1.5, 2.5, 3.0, -4.0]);
}

#[test]
fn mul_c64x4_matches_scalar_complex_multiply() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0, 2.0, -1.0, 3.0];
	let b = [5.0, -1.0, 2.0, 0.5];
	let expect = [
		a[0] * b[0] - a[1] * b[1],
		a[0] * b[1] + a[1] * b[0],
		a[2] * b[2] - a[3] * b[3],
		a[2] * b[3] + a[3] * b[2],
	];
	assert_eq!(avx.mul_c64x4(a, b), expect);
}

#[test]
fn conj_mul_c64x4_matches_mul_with_conjugated_a() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0, 2.0, -1.0, 3.0];
	let b = [5.0, -1.0, 2.0, 0.5];
	assert_eq!(avx.conj_mul_c64x4(a, b), avx.mul_c64x4(avx.conj_c64x4(a), b));
}

#[test]
fn abs2_c64x4_matches_scalar_squared_magnitude() {
	let Some(avx) = Avx::detect() else { return };
	assert_eq!(avx.abs2_c64x4([3.0, 4.0, -1.0, 2.0]), [25.0, 25.0, 5.0, 5.0]);
}

#[test]
fn partial_load_f32x8_zero_pads_and_caps_at_width() {
	let Some(avx) = Avx::detect() else { return };
	let src: Vec<f32> = (1..=5).map(|i| i as f32).collect();
	let got = avx.partial_load_f32x8(&src);
	assert_eq!(&got[..5], src.as_slice());
	assert_eq!(&got[5..], [0.0; 3]);

	let long: Vec<f32> = (1..=20).map(|i| i as f32).collect();
	assert_eq!(avx.partial_load_f32x8(&long), core::array::from_fn::<f32, 8, _>(|i| (i + 1) as f32));
}

#[test]
fn partial_load_store_f32x8_roundtrip_various_lengths() {
	let Some(avx) = Avx::detect() else { return };
	for len in [0usize, 1, 3, 7, 8, 12] {
		let src: Vec<f32> = (0..len).map(|i| i as f32 * 1.5 - 3.0).collect();
		let v = avx.partial_load_f32x8(&src);
		let mut dst = vec![f32::NAN; len.min(8)];
		avx.partial_store_f32x8(v, &mut dst);
		assert_eq!(dst, &src[..len.min(8)], "len {len}");
	}
}

#[test]
fn partial_load_store_f64x4_roundtrip_various_lengths() {
	let Some(avx) = Avx::detect() else { return };
	for len in [0usize, 1, 3, 4, 6] {
		let src: Vec<f64> = (0..len).map(|i| i as f64 * 2.5 - 1.0).collect();
		let v = avx.partial_load_f64x4(&src);
		assert_eq!(&v[len.min(4)..], &[0.0; 4][len.min(4)..]);
		let mut dst = vec![f64::NAN; len.min(4)];
		avx.partial_store_f64x4(v, &mut dst);
		assert_eq!(dst, &src[..len.min(4)], "len {len}");
	}
}
