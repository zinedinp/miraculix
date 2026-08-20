use super::*;
use super::super::super::macros::{slice_binop_matches_scalar_test, slice_ternop_matches_scalar_test};

slice_binop_matches_scalar_test!(add_ph_u16x32_slice_matches_scalar, Avx512Fp16, add_ph_u16x32_slice, add_scalar, u16);
slice_binop_matches_scalar_test!(sub_ph_u16x32_slice_matches_scalar, Avx512Fp16, sub_ph_u16x32_slice, sub_scalar, u16);
slice_binop_matches_scalar_test!(mul_ph_u16x32_slice_matches_scalar, Avx512Fp16, mul_ph_u16x32_slice, mul_scalar, u16);
slice_binop_matches_scalar_test!(div_ph_u16x32_slice_matches_scalar, Avx512Fp16, div_ph_u16x32_slice, div_scalar, u16);
slice_binop_matches_scalar_test!(min_ph_u16x32_slice_matches_scalar, Avx512Fp16, min_ph_u16x32_slice, min_scalar, u16);
slice_binop_matches_scalar_test!(max_ph_u16x32_slice_matches_scalar, Avx512Fp16, max_ph_u16x32_slice, max_scalar, u16);

slice_binop_matches_scalar_test!(add_ph_u16x16_slice_matches_scalar, Avx512Fp16Vl, add_ph_u16x16_slice, add_scalar, u16);
slice_binop_matches_scalar_test!(add_ph_u16x8_slice_matches_scalar, Avx512Fp16Vl, add_ph_u16x8_slice, add_scalar, u16);

#[test]
fn abs_ph_u16x32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Fp16::detect() else { return };
	for len in [0usize, 1, 3, 31, 32, 33, 64, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 - (len as f32 / 2.0))).collect();
		let mut out = vec![0u16; len];
		t.abs_ph_u16x32_slice(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| abs_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn abs_ph_u16x16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	for len in [0usize, 1, 3, 15, 16, 17, 33, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 - (len as f32 / 2.0))).collect();
		let mut out = vec![0u16; len];
		t.abs_ph_u16x16_slice(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| abs_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn abs_ph_u16x8_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	for len in [0usize, 1, 3, 7, 8, 9, 17, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 - (len as f32 / 2.0))).collect();
		let mut out = vec![0u16; len];
		t.abs_ph_u16x8_slice(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| abs_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

slice_ternop_matches_scalar_test!(fmadd_ph_u16x32_slice_matches_scalar, Avx512Fp16, fmadd_ph_u16x32_slice, fmadd_scalar, u16);
slice_ternop_matches_scalar_test!(fmsub_ph_u16x32_slice_matches_scalar, Avx512Fp16, fmsub_ph_u16x32_slice, fmsub_scalar, u16);
slice_ternop_matches_scalar_test!(fnmadd_ph_u16x32_slice_matches_scalar, Avx512Fp16, fnmadd_ph_u16x32_slice, fnmadd_scalar, u16);
slice_ternop_matches_scalar_test!(fnmsub_ph_u16x32_slice_matches_scalar, Avx512Fp16, fnmsub_ph_u16x32_slice, fnmsub_scalar, u16);

#[test]
fn fmaddsub_ph_u16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Fp16::detect() else { return };
	for len in [0usize, 1, 3, 31, 32, 33, 64, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 + 1.0)).collect();
		let b: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar((len - i) as f32 + 1.0)).collect();
		let c: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 * 0.5)).collect();
		let mut out = vec![0u16; len];
		t.fmaddsub_ph_u16_slice(&a, &b, &c, &mut out);
		let expect: Vec<u16> = (0..len).map(|j| fmaddsub_scalar_at(j, a[j], b[j], c[j])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn fmsubadd_ph_u16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Fp16::detect() else { return };
	for len in [0usize, 1, 3, 31, 32, 33, 64, 100] {
		let a: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 + 1.0)).collect();
		let b: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar((len - i) as f32 + 1.0)).collect();
		let c: Vec<u16> = (0..len).map(|i| f32_to_f16_scalar(i as f32 * 0.5)).collect();
		let mut out = vec![0u16; len];
		t.fmsubadd_ph_u16_slice(&a, &b, &c, &mut out);
		let expect: Vec<u16> = (0..len).map(|j| fmsubadd_scalar_at(j, a[j], b[j], c[j])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn abs_ph_clears_sign_bit() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let neg_one = f32_to_f16_scalar(-1.0);
	let a = [neg_one; 32];
	let out = t.abs_ph_u16x32(a);
	assert_eq!(out, [f32_to_f16_scalar(1.0); 32]);
}

// FP16 has ~3 significant decimal digits; a few ULPs of headroom over
// the documented `< 1.5*2^-12` approximation bound for rsqrt/rcp.
const APPROX_TOL: f32 = 2e-3;

#[test]
fn sqrt_ph_u16x32_matches_f32_sqrt_through_round_trip() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let got = t.sqrt_ph_u16x32(a);
	for i in 0..32 {
		let expect = f16_to_f32_scalar(a[i]).sqrt();
		let got_f32 = f16_to_f32_scalar(got[i]);
		assert!((got_f32 - expect).abs() <= APPROX_TOL * expect.abs(), "i={i} got={got_f32} expect={expect}");
	}
}

#[test]
fn sqrt_ph_u16x16_matches_f32_sqrt_through_round_trip() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let got = t.sqrt_ph_u16x16(a);
	for i in 0..16 {
		let expect = f16_to_f32_scalar(a[i]).sqrt();
		let got_f32 = f16_to_f32_scalar(got[i]);
		assert!((got_f32 - expect).abs() <= APPROX_TOL * expect.abs(), "i={i} got={got_f32} expect={expect}");
	}
}

#[test]
fn sqrt_ph_u16x8_matches_f32_sqrt_through_round_trip() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let got = t.sqrt_ph_u16x8(a);
	for i in 0..8 {
		let expect = f16_to_f32_scalar(a[i]).sqrt();
		let got_f32 = f16_to_f32_scalar(got[i]);
		assert!((got_f32 - expect).abs() <= APPROX_TOL * expect.abs(), "i={i} got={got_f32} expect={expect}");
	}
}

#[test]
fn rsqrt_ph_u16x32_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let got = t.rsqrt_ph_u16x32(a);
	for i in 0..32 {
		let expect = 1.0 / f16_to_f32_scalar(a[i]).sqrt();
		let got_f32 = f16_to_f32_scalar(got[i]);
		assert!((got_f32 - expect).abs() <= APPROX_TOL * expect.abs(), "i={i} got={got_f32} expect={expect}");
	}
}

#[test]
fn rcp_ph_u16x32_approximates_reciprocal() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let got = t.rcp_ph_u16x32(a);
	for i in 0..32 {
		let expect = 1.0 / f16_to_f32_scalar(a[i]);
		let got_f32 = f16_to_f32_scalar(got[i]);
		assert!((got_f32 - expect).abs() <= APPROX_TOL * expect.abs(), "i={i} got={got_f32} expect={expect}");
	}
}

fn ph_carrier(vals: &[f32]) -> [u16; 8] {
	let mut a = [0u16; 8];
	for (i, &v) in vals.iter().enumerate() {
		a[i] = f32_to_f16_scalar(v);
	}
	a
}

#[test]
fn ph_to_f64x2_is_exact() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.5, -2.25]);
	assert_eq!(t.ph_to_f64x2(a), [1.5, -2.25]);
}

#[test]
fn ph_to_f64x4_is_exact() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.5, -2.25, 3.0, 65504.0]);
	assert_eq!(t.ph_to_f64x4(a), [1.5, -2.25, 3.0, 65504.0]);
}

#[test]
fn ph_to_f64x8_is_exact() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a = ph_carrier(&[1.5, -2.25, 3.0, 65504.0, 0.0, -0.0, 0.5, -1.0]);
	assert_eq!(t.ph_to_f64x8(a), [1.5, -2.25, 3.0, 65504.0, 0.0, -0.0, 0.5, -1.0]);
}

#[test]
fn f64x2_to_ph_rounds_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let got = t.f64x2_to_ph([1.5, -2.25]);
	assert_eq!(got[0], f32_to_f16_scalar(1.5));
	assert_eq!(got[1], f32_to_f16_scalar(-2.25));
	assert_eq!(&got[2..], &[0u16; 6]);
}

#[test]
fn f64x4_to_ph_rounds_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let got = t.f64x4_to_ph([1.5, -2.25, 3.0, 4.0]);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar([1.5, -2.25, 3.0, 4.0][i]));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn f64x8_to_ph_rounds_all_lanes() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [f64; 8] = [1.5, -2.25, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let got = t.f64x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn i16x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i16; 8] = [-8, -5, -2, 1, 4, 7, 10, 13];
	let got = t.i16x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn i16x16_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i16; 16] = [-16, -13, -10, -7, -4, -1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29];
	let got = t.i16x16_to_ph(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn i16x32_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [i16; 32] = [-32, -29, -26, -23, -20, -17, -14, -11, -8, -5, -2, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61];
	let got = t.i16x32_to_ph(a);
	let expect: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u16x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 8] = [1, 4, 7, 10, 13, 16, 19, 22];
	let got = t.u16x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u16x16_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 16] = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46];
	let got = t.u16x16_to_ph(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u16x32_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u16; 32] = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94];
	let got = t.u16x32_to_ph(a);
	let expect: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn ph_to_i16x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 8] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32];
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [i16; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as i16);
	let want_trunc: [i16; 8] = core::array::from_fn(|i| vals[i] as i16);
	assert_eq!(t.ph_to_i16x8(a), want_round);
	assert_eq!(t.ph_to_i16x8_trunc(a), want_trunc);
}

#[test]
fn ph_to_i16x16_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 16] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32, 17.5f32, -19.0f32, 21.5f32, -23.0f32, 25.5f32, -27.0f32, 29.5f32, -31.0f32];
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [i16; 16] = core::array::from_fn(|i| vals[i].round_ties_even() as i16);
	let want_trunc: [i16; 16] = core::array::from_fn(|i| vals[i] as i16);
	assert_eq!(t.ph_to_i16x16(a), want_round);
	assert_eq!(t.ph_to_i16x16_trunc(a), want_trunc);
}

#[test]
fn ph_to_i16x32_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals: [f32; 32] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32, 17.5f32, -19.0f32, 21.5f32, -23.0f32, 25.5f32, -27.0f32, 29.5f32, -31.0f32, 33.5f32, -35.0f32, 37.5f32, -39.0f32, 41.5f32, -43.0f32, 45.5f32, -47.0f32, 49.5f32, -51.0f32, 53.5f32, -55.0f32, 57.5f32, -59.0f32, 61.5f32, -63.0f32];
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [i16; 32] = core::array::from_fn(|i| vals[i].round_ties_even() as i16);
	let want_trunc: [i16; 32] = core::array::from_fn(|i| vals[i] as i16);
	assert_eq!(t.ph_to_i16x32(a), want_round);
	assert_eq!(t.ph_to_i16x32_trunc(a), want_trunc);
}

#[test]
fn ph_to_u16x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 8] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32];
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [u16; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as u16);
	let want_trunc: [u16; 8] = core::array::from_fn(|i| vals[i] as u16);
	assert_eq!(t.ph_to_u16x8(a), want_round);
	assert_eq!(t.ph_to_u16x8_trunc(a), want_trunc);
}

#[test]
fn ph_to_u16x16_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 16] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32, 17.5f32, 19.0f32, 21.5f32, 23.0f32, 25.5f32, 27.0f32, 29.5f32, 31.0f32];
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [u16; 16] = core::array::from_fn(|i| vals[i].round_ties_even() as u16);
	let want_trunc: [u16; 16] = core::array::from_fn(|i| vals[i] as u16);
	assert_eq!(t.ph_to_u16x16(a), want_round);
	assert_eq!(t.ph_to_u16x16_trunc(a), want_trunc);
}

#[test]
fn ph_to_u16x32_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals: [f32; 32] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32, 17.5f32, 19.0f32, 21.5f32, 23.0f32, 25.5f32, 27.0f32, 29.5f32, 31.0f32, 33.5f32, 35.0f32, 37.5f32, 39.0f32, 41.5f32, 43.0f32, 45.5f32, 47.0f32, 49.5f32, 51.0f32, 53.5f32, 55.0f32, 57.5f32, 59.0f32, 61.5f32, 63.0f32];
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [u16; 32] = core::array::from_fn(|i| vals[i].round_ties_even() as u16);
	let want_trunc: [u16; 32] = core::array::from_fn(|i| vals[i] as u16);
	assert_eq!(t.ph_to_u16x32(a), want_round);
	assert_eq!(t.ph_to_u16x32_trunc(a), want_trunc);
}

#[test]
fn i32x4_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i32; 4] = [-10, -5, 0, 5];
	let got = t.i32x4_to_ph(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn i32x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i32; 8] = [-20, -15, -10, -5, 0, 5, 10, 15];
	let got = t.i32x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn i32x16_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [i32; 16] = [-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35];
	let got = t.i32x16_to_ph(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u32x4_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u32; 4] = [1, 6, 11, 16];
	let got = t.u32x4_to_ph(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn u32x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u32; 8] = [1, 6, 11, 16, 21, 26, 31, 36];
	let got = t.u32x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u32x16_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u32; 16] = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76];
	let got = t.u32x16_to_ph(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn ph_to_i32x4_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 4] = [1.5f32, -3.0f32, 5.5f32, -7.0f32];
	let a = ph_carrier(&vals);
	let want_round: [i32; 4] = core::array::from_fn(|i| vals[i].round_ties_even() as i32);
	let want_trunc: [i32; 4] = core::array::from_fn(|i| vals[i] as i32);
	assert_eq!(t.ph_to_i32x4(a), want_round);
	assert_eq!(t.ph_to_i32x4_trunc(a), want_trunc);
}

#[test]
fn ph_to_i32x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 8] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32];
	let a = ph_carrier(&vals);
	let want_round: [i32; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as i32);
	let want_trunc: [i32; 8] = core::array::from_fn(|i| vals[i] as i32);
	assert_eq!(t.ph_to_i32x8(a), want_round);
	assert_eq!(t.ph_to_i32x8_trunc(a), want_trunc);
}

#[test]
fn ph_to_i32x16_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals: [f32; 16] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32, 17.5f32, -19.0f32, 21.5f32, -23.0f32, 25.5f32, -27.0f32, 29.5f32, -31.0f32];
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [i32; 16] = core::array::from_fn(|i| vals[i].round_ties_even() as i32);
	let want_trunc: [i32; 16] = core::array::from_fn(|i| vals[i] as i32);
	assert_eq!(t.ph_to_i32x16(a), want_round);
	assert_eq!(t.ph_to_i32x16_trunc(a), want_trunc);
}

#[test]
fn ph_to_u32x4_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 4] = [1.5f32, 3.0f32, 5.5f32, 7.0f32];
	let a = ph_carrier(&vals);
	let want_round: [u32; 4] = core::array::from_fn(|i| vals[i].round_ties_even() as u32);
	let want_trunc: [u32; 4] = core::array::from_fn(|i| vals[i] as u32);
	assert_eq!(t.ph_to_u32x4(a), want_round);
	assert_eq!(t.ph_to_u32x4_trunc(a), want_trunc);
}

#[test]
fn ph_to_u32x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals: [f32; 8] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32];
	let a = ph_carrier(&vals);
	let want_round: [u32; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as u32);
	let want_trunc: [u32; 8] = core::array::from_fn(|i| vals[i] as u32);
	assert_eq!(t.ph_to_u32x8(a), want_round);
	assert_eq!(t.ph_to_u32x8_trunc(a), want_trunc);
}

#[test]
fn ph_to_u32x16_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals: [f32; 16] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32, 17.5f32, 19.0f32, 21.5f32, 23.0f32, 25.5f32, 27.0f32, 29.5f32, 31.0f32];
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	let want_round: [u32; 16] = core::array::from_fn(|i| vals[i].round_ties_even() as u32);
	let want_trunc: [u32; 16] = core::array::from_fn(|i| vals[i] as u32);
	assert_eq!(t.ph_to_u32x16(a), want_round);
	assert_eq!(t.ph_to_u32x16_trunc(a), want_trunc);
}

#[test]
fn i64x2_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i64; 2] = [-7, 0];
	let got = t.i64x2_to_ph(a);
	let expect: [u16; 2] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..2], &expect);
	assert_eq!(&got[2..], &[0u16; 6]);
}

#[test]
fn i64x4_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [i64; 4] = [-14, -7, 0, 7];
	let got = t.i64x4_to_ph(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn i64x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [i64; 8] = [-28, -21, -14, -7, 0, 7, 14, 21];
	let got = t.i64x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn u64x2_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u64; 2] = [1, 8];
	let got = t.u64x2_to_ph(a);
	let expect: [u16; 2] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..2], &expect);
	assert_eq!(&got[2..], &[0u16; 6]);
}

#[test]
fn u64x4_to_ph_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u64; 4] = [1, 8, 15, 22];
	let got = t.u64x4_to_ph(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn u64x8_to_ph_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [u64; 8] = [1, 8, 15, 22, 29, 36, 43, 50];
	let got = t.u64x8_to_ph(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i] as f32));
	assert_eq!(got, expect);
}

#[test]
fn ph_to_i64x2_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, -3.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32];
	let vals: [f32; 2] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [i64; 2] = core::array::from_fn(|i| vals[i].round_ties_even() as i64);
	let want_trunc: [i64; 2] = core::array::from_fn(|i| vals[i] as i64);
	assert_eq!(t.ph_to_i64x2(a), want_round);
	assert_eq!(t.ph_to_i64x2_trunc(a), want_trunc);
}

#[test]
fn ph_to_i64x4_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32];
	let vals: [f32; 4] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [i64; 4] = core::array::from_fn(|i| vals[i].round_ties_even() as i64);
	let want_trunc: [i64; 4] = core::array::from_fn(|i| vals[i] as i64);
	assert_eq!(t.ph_to_i64x4(a), want_round);
	assert_eq!(t.ph_to_i64x4_trunc(a), want_trunc);
}

#[test]
fn ph_to_i64x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, -3.0f32, 5.5f32, -7.0f32, 9.5f32, -11.0f32, 13.5f32, -15.0f32];
	let vals: [f32; 8] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [i64; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as i64);
	let want_trunc: [i64; 8] = core::array::from_fn(|i| vals[i] as i64);
	assert_eq!(t.ph_to_i64x8(a), want_round);
	assert_eq!(t.ph_to_i64x8_trunc(a), want_trunc);
}

#[test]
fn ph_to_u64x2_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, 3.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32];
	let vals: [f32; 2] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [u64; 2] = core::array::from_fn(|i| vals[i].round_ties_even() as u64);
	let want_trunc: [u64; 2] = core::array::from_fn(|i| vals[i] as u64);
	assert_eq!(t.ph_to_u64x2(a), want_round);
	assert_eq!(t.ph_to_u64x2_trunc(a), want_trunc);
}

#[test]
fn ph_to_u64x4_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32];
	let vals: [f32; 4] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [u64; 4] = core::array::from_fn(|i| vals[i].round_ties_even() as u64);
	let want_trunc: [u64; 4] = core::array::from_fn(|i| vals[i] as u64);
	assert_eq!(t.ph_to_u64x4(a), want_round);
	assert_eq!(t.ph_to_u64x4_trunc(a), want_trunc);
}

#[test]
fn ph_to_u64x8_and_trunc_match_expected() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let carrier_vals: [f32; 8] = [1.5f32, 3.0f32, 5.5f32, 7.0f32, 9.5f32, 11.0f32, 13.5f32, 15.0f32];
	let vals: [f32; 8] = core::array::from_fn(|i| carrier_vals[i]);
	let a = ph_carrier(&carrier_vals);
	let want_round: [u64; 8] = core::array::from_fn(|i| vals[i].round_ties_even() as u64);
	let want_trunc: [u64; 8] = core::array::from_fn(|i| vals[i] as u64);
	assert_eq!(t.ph_to_u64x8(a), want_round);
	assert_eq!(t.ph_to_u64x8_trunc(a), want_trunc);
}

#[test]
fn int_to_ph_round_matches_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let i16s: [i16; 32] = core::array::from_fn(|i| i as i16 - 16);
	let u16s: [u16; 32] = core::array::from_fn(|i| i as u16 + 1);
	let i32s: [i32; 16] = core::array::from_fn(|i| i as i32 - 8);
	let u32s: [u32; 16] = core::array::from_fn(|i| i as u32 + 1);
	let i64s: [i64; 8] = core::array::from_fn(|i| i as i64 - 4);
	let u64s: [u64; 8] = core::array::from_fn(|i| i as u64 + 1);
	assert_eq!(t.i16x32_to_ph_round::<RN>(i16s), t.i16x32_to_ph(i16s));
	assert_eq!(t.u16x32_to_ph_round::<RN>(u16s), t.u16x32_to_ph(u16s));
	assert_eq!(t.i32x16_to_ph_round::<RN>(i32s), t.i32x16_to_ph(i32s));
	assert_eq!(t.u32x16_to_ph_round::<RN>(u32s), t.u32x16_to_ph(u32s));
	assert_eq!(t.i64x8_to_ph_round::<RN>(i64s), t.i64x8_to_ph(i64s));
	assert_eq!(t.u64x8_to_ph_round::<RN>(u64s), t.u64x8_to_ph(u64s));
}

#[test]
fn ph_to_int_round_matches_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 - 16.0));
	let a16: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 - 8.0));
	let a8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 - 4.0));
	assert_eq!(t.ph_to_i16x32_round::<RN>(a32), t.ph_to_i16x32(a32));
	assert_eq!(t.ph_to_u16x32_round::<RN>(a32), t.ph_to_u16x32(a32));
	assert_eq!(t.ph_to_i32x16_round::<RN>(a16), t.ph_to_i32x16(a16));
	assert_eq!(t.ph_to_u32x16_round::<RN>(a16), t.ph_to_u32x16(a16));
	assert_eq!(t.ph_to_i64x8_round::<RN>(a8), t.ph_to_i64x8(a8));
	assert_eq!(t.ph_to_u64x8_round::<RN>(a8), t.ph_to_u64x8(a8));
}

#[test]
fn f32x4_to_ph_x_matches_f32_to_f16_scalar_and_zeros_upper_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [f32; 4] = [-2.5, -1.0, 1.0, 2.5];
	let got = t.f32x4_to_ph_x(a);
	let expect: [u16; 4] = core::array::from_fn(|i| f32_to_f16_scalar(a[i]));
	assert_eq!(&got[..4], &expect);
	assert_eq!(&got[4..], &[0u16; 4]);
}

#[test]
fn f32x8_to_ph_x_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [f32; 8] = [-3.5, -2.5, -1.0, -0.5, 0.5, 1.0, 2.5, 3.5];
	let got = t.f32x8_to_ph_x(a);
	let expect: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn f32x16_to_ph_x_matches_f32_to_f16_scalar() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 - 8.0 + 0.5);
	let got = t.f32x16_to_ph_x(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn ph_to_f32x4_x_is_exact() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.5, -2.25, 3.0, 4.0]);
	assert_eq!(t.ph_to_f32x4_x(a), [1.5f32, -2.25, 3.0, 4.0]);
}

#[test]
fn ph_to_f32x8_x_is_exact() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let vals = [1.5f32, -2.25, 3.0, 4.0, -5.5, 6.0, 7.25, -8.0];
	let a = ph_carrier(&vals);
	assert_eq!(t.ph_to_f32x8_x(a), vals);
}

#[test]
fn ph_to_f32x16_x_is_exact() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals: [f32; 16] = core::array::from_fn(|i| i as f32 - 8.0 + 0.5);
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(vals[i]));
	assert_eq!(t.ph_to_f32x16_x(a), vals);
}

#[test]
fn f32_to_ph_x_round_matches_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 - 8.0);
	assert_eq!(t.f32x16_to_ph_x_round::<RN>(a), t.f32x16_to_ph_x(a));
	let ph: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(a[i]));
	assert_eq!(t.ph_to_f32x16_x_round::<RN>(ph), t.ph_to_f32x16_x(ph));
}

#[test]
fn cmp_ph_mask_u16x8_predicates_match_expected_bits() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	// lane0: a<b, lane1: a==b, lane2: a>b, lane3: NaN (unordered, all false).
	// lanes 4..8 default to 0.0==0.0 in both `a`/`b` (`ph_carrier` zero-pads) -
	// mask those bits off rather than special-casing them per predicate.
	let a = ph_carrier(&[1.0, 2.0, 3.0, f32::NAN]);
	let b = ph_carrier(&[2.0, 2.0, 1.0, 1.0]);
	const LOW4: u8 = 0b1111;
	assert_eq!(t.cmpeq_ph_mask_u16x8(a, b) & LOW4, 0b0010);
	assert_eq!(t.cmplt_ph_mask_u16x8(a, b) & LOW4, 0b0001);
	assert_eq!(t.cmple_ph_mask_u16x8(a, b) & LOW4, 0b0011);
	assert_eq!(t.cmpgt_ph_mask_u16x8(a, b) & LOW4, 0b0100);
	assert_eq!(t.cmpge_ph_mask_u16x8(a, b) & LOW4, 0b0110);
}

#[test]
fn cmp_ph_mask_u16x16_predicates_match_expected_bits() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let mut av = [0.0f32; 16];
	let mut bv = [0.0f32; 16];
	av[0] = 1.0;
	bv[0] = 2.0;
	av[1] = 2.0;
	bv[1] = 2.0;
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(av[i]));
	let b: [u16; 16] = core::array::from_fn(|i| f32_to_f16_scalar(bv[i]));
	assert_eq!(t.cmplt_ph_mask_u16x16(a, b), 0b01);
	assert_eq!(t.cmpeq_ph_mask_u16x16(a, b) & 0b11, 0b10);
}

#[test]
fn cmp_ph_mask_u16x32_predicates_match_expected_bits() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let mut av = [0.0f32; 32];
	let mut bv = [0.0f32; 32];
	av[0] = 1.0;
	bv[0] = 2.0;
	av[1] = 2.0;
	bv[1] = 2.0;
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(av[i]));
	let b: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(bv[i]));
	assert_eq!(t.cmplt_ph_mask_u16x32(a, b) & 0b11, 0b01);
	assert_eq!(t.cmpeq_ph_mask_u16x32(a, b) & 0b11, 0b10);
}

#[test]
fn mul_pch_u16x8_matches_complex_multiply_formula() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	// (2+3i)(4+5i) = (2*4-3*5) + (2*5+3*4)i = -7 + 22i; other 3 pairs = (1+1i)*(1+1i)=(0+2i).
	let a = ph_carrier(&[2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
	let b = ph_carrier(&[4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
	let got = t.mul_pch_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), -7.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 22.0);
	assert_eq!(f16_to_f32_scalar(got[2]), 0.0);
	assert_eq!(f16_to_f32_scalar(got[3]), 2.0);
}

#[test]
fn cmul_pch_u16x8_matches_conjugate_multiply_formula() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	// (2+3i)*conj(4+5i) = (2*4+3*5) + (3*4-2*5)i = 23 + 2i.
	let a = ph_carrier(&[2.0, 3.0, 0.0, 0.0]);
	let b = ph_carrier(&[4.0, 5.0, 0.0, 0.0]);
	let got = t.cmul_pch_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 23.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 2.0);
}

#[test]
fn fmadd_pch_u16x8_matches_complex_fma_formula() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	// (2+3i)(4+5i) + (1+1i) = (-7+22i) + (1+1i) = -6 + 23i.
	let a = ph_carrier(&[2.0, 3.0, 0.0, 0.0]);
	let b = ph_carrier(&[4.0, 5.0, 0.0, 0.0]);
	let c = ph_carrier(&[1.0, 1.0, 0.0, 0.0]);
	let got = t.fmadd_pch_u16x8(a, b, c);
	assert_eq!(f16_to_f32_scalar(got[0]), -6.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 23.0);
}

#[test]
fn fcmadd_pch_u16x8_matches_conjugate_fma_formula() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	// (2+3i)*conj(4+5i) + (1+1i) = (23+2i) + (1+1i) = 24 + 3i.
	let a = ph_carrier(&[2.0, 3.0, 0.0, 0.0]);
	let b = ph_carrier(&[4.0, 5.0, 0.0, 0.0]);
	let c = ph_carrier(&[1.0, 1.0, 0.0, 0.0]);
	let got = t.fcmadd_pch_u16x8(a, b, c);
	assert_eq!(f16_to_f32_scalar(got[0]), 24.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 3.0);
}

#[test]
fn mul_pch_widths_agree_on_shared_low_pairs() {
	let Some(t128) = Avx512Fp16Vl::detect() else { return };
	let Some(t512) = Avx512Fp16::detect() else { return };
	let a8 = ph_carrier(&[2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
	let b8 = ph_carrier(&[4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
	let got128 = t128.mul_pch_u16x8(a8, b8);
	let mut a32 = [0u16; 32];
	let mut b32 = [0u16; 32];
	a32[..8].copy_from_slice(&a8);
	b32[..8].copy_from_slice(&b8);
	let got512 = t512.mul_pch_u16x32(a32, b32);
	assert_eq!(&got512[..8], &got128);
}

#[test]
fn mul_sch_computes_lane0_and_passes_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[2.0, 3.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.mul_sch_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), -7.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 22.0);
	assert_eq!(&got[2..], &a[2..]);
}

#[test]
fn fmadd_sch_computes_lane0_and_passes_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[2.0, 3.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let c = ph_carrier(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.fmadd_sch_u16x8(a, b, c);
	assert_eq!(f16_to_f32_scalar(got[0]), -6.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 23.0);
	assert_eq!(&got[2..], &a[2..]);
}

#[test]
fn min_sh_computes_lane0_and_passes_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[5.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.min_sh_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 3.0);
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn max_sh_computes_lane0_and_passes_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[5.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.max_sh_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 5.0);
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn sqrt_sh_matches_f32_sqrt_and_passes_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.sqrt_sh_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 2.0);
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn getexp_sh_extracts_unbiased_exponent() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.getexp_sh_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 3.0); // 8.0 = 1.0 * 2^3
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn scalef_sh_scales_by_power_of_two() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let mut base = a;
	base[0] = f32_to_f16_scalar(1.5);
	let got = t.scalef_sh_u16x8(base, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 12.0); // 1.5 * 2^3
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn reduce_sh_reduces_to_fractional_part() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[3.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	// IMM8 = 3: round toward zero, 0 fraction bits kept below the integer -> subtracts truncated integer part.
	let got = t.reduce_sh_u16x8::<3>(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 0.75);
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn roundscale_sh_rounds_to_integer() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[3.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	// IMM8 = 3: round toward zero, 0 fraction bits.
	let got = t.roundscale_sh_u16x8::<3>(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 3.0);
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn getmant_sh_normalizes_mantissa() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	use core::arch::x86_64::{_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC};
	let a = ph_carrier(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let got = t.getmant_sh_u16x8::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 1.5); // 12.0 = 1.5 * 2^3, normalized to [1,2)
	assert_eq!(&got[1..], &a[1..]);
}

#[test]
fn cmp_sh_mask_predicates_only_touch_bit0() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	assert_eq!(t.cmplt_sh_mask_u16x8(a, b), 1);
	assert_eq!(t.cmpgt_sh_mask_u16x8(a, b), 0);
	assert_eq!(t.cmple_sh_mask_u16x8(a, b), 1);
	assert_eq!(t.cmpge_sh_mask_u16x8(a, b), 0);
	assert_eq!(t.cmpeq_sh_mask_u16x8(a, a), 1);
}

#[test]
fn comi_sh_predicates_return_boolean_ints() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	assert_eq!(t.comilt_sh(a, b), 1);
	assert_eq!(t.comigt_sh(a, b), 0);
	assert_eq!(t.comile_sh(a, b), 1);
	assert_eq!(t.comige_sh(a, b), 0);
	assert_eq!(t.comieq_sh(a, a), 1);
	assert_eq!(t.comineq_sh(a, b), 1);
}

#[test]
fn ucomi_sh_predicates_return_boolean_ints() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	assert_eq!(t.ucomilt_sh(a, b), 1);
	assert_eq!(t.ucomigt_sh(a, b), 0);
	assert_eq!(t.ucomile_sh(a, b), 1);
	assert_eq!(t.ucomige_sh(a, b), 0);
	assert_eq!(t.ucomieq_sh(a, a), 1);
	assert_eq!(t.ucomineq_sh(a, b), 1);
}

#[test]
fn cvti32_sh_and_cvtu32_sh_write_lane0_and_pass_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
	let got_i = t.cvti32_sh(a, -5);
	assert_eq!(f16_to_f32_scalar(got_i[0]), -5.0);
	assert_eq!(&got_i[1..], &a[1..]);
	let got_u = t.cvtu32_sh(a, 5);
	assert_eq!(f16_to_f32_scalar(got_u[0]), 5.0);
	assert_eq!(&got_u[1..], &a[1..]);
}

#[test]
fn cvtss_sh_and_cvtsd_sh_write_lane0_and_pass_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
	let got_ss = t.cvtss_sh(a, [3.5, 0.0, 0.0, 0.0]);
	assert_eq!(f16_to_f32_scalar(got_ss[0]), 3.5);
	assert_eq!(&got_ss[1..], &a[1..]);
	let got_sd = t.cvtsd_sh(a, [3.5, 0.0]);
	assert_eq!(f16_to_f32_scalar(got_sd[0]), 3.5);
	assert_eq!(&got_sd[1..], &a[1..]);
}

#[test]
fn cvtsh_i32_and_cvtsh_u32_round_to_nearest_even() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	assert_eq!(t.cvtsh_i32(a), -5);
	let b = ph_carrier(&[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	assert_eq!(t.cvtsh_u32(b), 5);
}

#[test]
fn cvtsh_ss_and_cvtsh_sd_write_lane0_and_pass_through_rest() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let b = ph_carrier(&[3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let a_ss = [9.0f32, 1.0, 2.0, 3.0];
	let got_ss = t.cvtsh_ss(a_ss, b);
	assert_eq!(got_ss[0], 3.5);
	assert_eq!(&got_ss[1..], &a_ss[1..]);
	let a_sd = [9.0f64, 1.0];
	let got_sd = t.cvtsh_sd(a_sd, b);
	assert_eq!(got_sd[0], 3.5);
	assert_eq!(&got_sd[1..], &a_sd[1..]);
}

#[test]
fn add_round_ph_respects_explicit_rounding_mode() {
	let Some(t) = Avx512Fp16::detect() else { return };
	// a=1.0, b=2^-11: exact sum 1.00048828125 sits exactly halfway between the two adjacent
	// f16 values 1.0 (0x3c00) and 1.0+2^-10 (0x3c01, the next representable f16 up): both
	// operands are exactly representable in f16, so no input rounding muddies the result.
	let a = [0x3c00u16; 32]; // 1.0
	let b = [0x1000u16; 32]; // 2^-11
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_ZERO};
	let toward_zero = t.add_round_ph_u16x32::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a, b);
	let toward_pos_inf = t.add_round_ph_u16x32::<{ _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC }>(a, b);
	assert_eq!(toward_zero[0], 0x3c00, "toward-zero picks the smaller-magnitude tie");
	assert_eq!(toward_pos_inf[0], 0x3c01, "toward-pos-inf picks the larger tie");
}

#[test]
fn sub_mul_div_round_ph_match_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a = [f32_to_f16_scalar(6.0); 32];
	let b = [f32_to_f16_scalar(2.0); 32];
	assert_eq!(t.sub_round_ph_u16x32::<RN>(a, b), t.sub_ph_u16x32(a, b));
	assert_eq!(t.mul_round_ph_u16x32::<RN>(a, b), t.mul_ph_u16x32(a, b));
	assert_eq!(t.div_round_ph_u16x32::<RN>(a, b), t.div_ph_u16x32(a, b));
}

#[test]
fn sqrt_round_ph_matches_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	let a = [f32_to_f16_scalar(9.0); 32];
	assert_eq!(t.sqrt_round_ph_u16x32::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a), t.sqrt_ph_u16x32(a));
}

#[test]
fn fma_family_round_ph_match_default_rounding_for_exact_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a = [f32_to_f16_scalar(2.0); 32];
	let b = [f32_to_f16_scalar(3.0); 32];
	let c = [f32_to_f16_scalar(1.0); 32];
	assert_eq!(t.fmadd_round_ph_u16x32::<RN>(a, b, c), t.fmadd_ph_u16x32(a, b, c));
	assert_eq!(t.fmsub_round_ph_u16x32::<RN>(a, b, c), t.fmsub_ph_u16x32(a, b, c));
	assert_eq!(t.fnmadd_round_ph_u16x32::<RN>(a, b, c), t.fnmadd_ph_u16x32(a, b, c));
	assert_eq!(t.fnmsub_round_ph_u16x32::<RN>(a, b, c), t.fnmsub_ph_u16x32(a, b, c));
	assert_eq!(t.fmaddsub_round_ph_u16x32::<RN>(a, b, c), t.fmaddsub_ph_u16x32(a, b, c));
	assert_eq!(t.fmsubadd_round_ph_u16x32::<RN>(a, b, c), t.fmsubadd_ph_u16x32(a, b, c));
}

#[test]
fn getexp_ph_u16x8_extracts_unbiased_exponent() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[8.0, 1.0, 0.5, 4.0]);
	let got = t.getexp_ph_u16x8(a);
	assert_eq!(f16_to_f32_scalar(got[0]), 3.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 0.0);
	assert_eq!(f16_to_f32_scalar(got[2]), -1.0);
	assert_eq!(f16_to_f32_scalar(got[3]), 2.0);
}

#[test]
fn getexp_ph_widths_agree_on_shared_low_lanes() {
	let Some(t128) = Avx512Fp16Vl::detect() else { return };
	let Some(t512) = Avx512Fp16::detect() else { return };
	let a8 = ph_carrier(&[8.0, 1.0, 0.5, 4.0]);
	let got128 = t128.getexp_ph_u16x8(a8);
	let mut a32 = [f32_to_f16_scalar(1.0); 32];
	a32[..8].copy_from_slice(&a8);
	let got512 = t512.getexp_ph_u16x32(a32);
	assert_eq!(&got512[..8], &got128);
}

#[test]
fn scalef_ph_u16x8_scales_by_power_of_two() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.5, 3.0, -2.0, 1.0]);
	let b = ph_carrier(&[3.0, 0.0, -1.0, 2.0]);
	let got = t.scalef_ph_u16x8(a, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 12.0); // 1.5 * 2^3
	assert_eq!(f16_to_f32_scalar(got[1]), 3.0); // 3.0 * 2^0
	assert_eq!(f16_to_f32_scalar(got[2]), -1.0); // -2.0 * 2^-1
	assert_eq!(f16_to_f32_scalar(got[3]), 4.0); // 1.0 * 2^2
}

#[test]
fn reduce_ph_u16x8_reduces_to_fractional_part() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[3.75, -1.25, 0.0, 0.0]);
	// IMM8 = 3: round toward zero, 0 fraction bits kept -> subtracts the truncated integer part.
	let got = t.reduce_ph_u16x8::<3>(a);
	assert_eq!(f16_to_f32_scalar(got[0]), 0.75);
	assert_eq!(f16_to_f32_scalar(got[1]), -0.25);
}

#[test]
fn roundscale_ph_u16x8_rounds_to_integer() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[3.75, -1.25, 0.0, 0.0]);
	// IMM8 = 3: round toward zero, 0 fraction bits.
	let got = t.roundscale_ph_u16x8::<3>(a);
	assert_eq!(f16_to_f32_scalar(got[0]), 3.0);
	assert_eq!(f16_to_f32_scalar(got[1]), -1.0);
}

#[test]
fn getmant_ph_u16x8_normalizes_mantissa() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	use core::arch::x86_64::{_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC};
	let a = ph_carrier(&[12.0, -12.0, 0.0, 0.0]);
	let got = t.getmant_ph_u16x8::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a);
	assert_eq!(f16_to_f32_scalar(got[0]), 1.5); // 12.0 = 1.5 * 2^3
	assert_eq!(f16_to_f32_scalar(got[1]), -1.5); // sign preserved (SIGN_SRC)
}

#[test]
fn reduce_roundscale_getmant_widths_agree_on_shared_low_lanes() {
	let Some(t128) = Avx512Fp16Vl::detect() else { return };
	let Some(t512) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC};
	let a8 = ph_carrier(&[3.75, -1.25, 12.0, -12.0]);
	let mut a32 = [f32_to_f16_scalar(1.0); 32];
	a32[..8].copy_from_slice(&a8);

	assert_eq!(&t512.reduce_ph_u16x32::<3>(a32)[..8], &t128.reduce_ph_u16x8::<3>(a8));
	assert_eq!(&t512.roundscale_ph_u16x32::<3>(a32)[..8], &t128.roundscale_ph_u16x8::<3>(a8));
	assert_eq!(
		&t512.getmant_ph_u16x32::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a32)[..8],
		&t128.getmant_ph_u16x8::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a8)
	);
}

#[test]
fn permutexvar_ph_u16x8_reverses_lanes() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 2.0, 3.0, 4.0]);
	let mut a_full = a;
	a_full[4..].copy_from_slice(&ph_carrier(&[5.0, 6.0, 7.0, 8.0])[4..]);
	let idx: [u16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
	let got = t.permutexvar_ph_u16x8(idx, a_full);
	let expect: [u16; 8] = core::array::from_fn(|i| a_full[7 - i]);
	assert_eq!(got, expect);
}

#[test]
fn permutex2var_ph_u16x8_selects_across_a_and_b() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 2.0, 3.0, 4.0]);
	let b = ph_carrier(&[10.0, 20.0, 30.0, 40.0]);
	// idx bit3 (0x8) selects b over a for 8-lane 128-bit form.
	let idx: [u16; 8] = [0, 0x8, 1, 0x8 | 1, 0, 0, 0, 0];
	let got = t.permutex2var_ph_u16x8(a, idx, b);
	assert_eq!(f16_to_f32_scalar(got[0]), 1.0); // a[0]
	assert_eq!(f16_to_f32_scalar(got[1]), 10.0); // b[0]
	assert_eq!(f16_to_f32_scalar(got[2]), 2.0); // a[1]
	assert_eq!(f16_to_f32_scalar(got[3]), 20.0); // b[1]
}

#[test]
fn permutexvar_widths_agree_on_shared_low_lanes() {
	let Some(t128) = Avx512Fp16Vl::detect() else { return };
	let Some(t512) = Avx512Fp16::detect() else { return };
	let a8 = ph_carrier(&[1.0, 2.0, 3.0, 4.0]);
	let idx8: [u16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
	let got128 = t128.permutexvar_ph_u16x8(idx8, a8);

	let mut a32 = [f32_to_f16_scalar(9.0); 32];
	a32[..8].copy_from_slice(&a8);
	let mut idx32 = [0u16; 32];
	idx32[..8].copy_from_slice(&idx8);
	let got512 = t512.permutexvar_ph_u16x32(idx32, a32);
	assert_eq!(&got512[..8], &got128);
}

#[test]
fn fmaddsub_ph_u16x8_matches_alternating_lane_semantics() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[2.0, 2.0, 2.0, 2.0]);
	let b = ph_carrier(&[3.0, 3.0, 3.0, 3.0]);
	let c = ph_carrier(&[1.0, 1.0, 1.0, 1.0]);
	let got = t.fmaddsub_ph_u16x8(a, b, c);
	// even lanes: a*b: c = 5.0; odd lanes: a*b + c = 7.0.
	assert_eq!(f16_to_f32_scalar(got[0]), 5.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 7.0);
}

#[test]
fn fmsubadd_ph_u16x8_matches_alternating_lane_semantics() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[2.0, 2.0, 2.0, 2.0]);
	let b = ph_carrier(&[3.0, 3.0, 3.0, 3.0]);
	let c = ph_carrier(&[1.0, 1.0, 1.0, 1.0]);
	let got = t.fmsubadd_ph_u16x8(a, b, c);
	// even lanes: a*b + c = 7.0; odd lanes: a*b: c = 5.0.
	assert_eq!(f16_to_f32_scalar(got[0]), 7.0);
	assert_eq!(f16_to_f32_scalar(got[1]), 5.0);
}

#[test]
fn fmaddsub_fmsubadd_ph_u16x16_match_alternating_lane_semantics() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = [f32_to_f16_scalar(2.0); 16];
	let b = [f32_to_f16_scalar(3.0); 16];
	let c = [f32_to_f16_scalar(1.0); 16];
	let addsub = t.fmaddsub_ph_u16x16(a, b, c);
	let subadd = t.fmsubadd_ph_u16x16(a, b, c);
	assert_eq!(f16_to_f32_scalar(addsub[0]), 5.0);
	assert_eq!(f16_to_f32_scalar(addsub[1]), 7.0);
	assert_eq!(f16_to_f32_scalar(subadd[0]), 7.0);
	assert_eq!(f16_to_f32_scalar(subadd[1]), 5.0);
}

// The oracle is the already-tested unmasked op, not a fresh scalar closure:
// that isolates the one new behavior (lane selection) and is the only
// workable reference for `rcp`/`rsqrt`, which are hardware approximations.

macro_rules! fp16_masked_binop_test {
	($name:ident, $Token:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $width:literal, $Mask:ty, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = $Token::detect() else { return };
			let a: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
			let b: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
			let src: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
			let mask: $Mask = $mask_val;
			let unmasked = t.$fixed_fn(a, b);
			let merge_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { 0 });
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

macro_rules! fp16_masked_unop_test {
	($name:ident, $Token:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $width:literal, $Mask:ty, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = $Token::detect() else { return };
			let a: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
			let src: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
			let mask: $Mask = $mask_val;
			let unmasked = t.$fixed_fn(a);
			let merge_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			let zero_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { 0 });
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

macro_rules! fp16_masked_ternop_test {
	($name:ident, $Token:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $width:literal, $Mask:ty, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = $Token::detect() else { return };
			let a: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
			let b: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
			let c: [u16; $width] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.25 + 3.0));
			let mask: $Mask = $mask_val;
			let unmasked = t.$fixed_fn(a, b, c);
			let merge_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { a[i] });
			assert_eq!(t.$merge_fn(a, mask, b, c), merge_expect, "merge");
			let zero_expect: [u16; $width] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { 0 });
			assert_eq!(t.$zero_fn(mask, a, b, c), zero_expect, "zero");
		}
	};
}

fp16_masked_binop_test!(
	add_ph_u16x32_masked_matches_unmasked, Avx512Fp16, add_ph_u16x32,
	add_ph_u16x32_merge_masked, add_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	add_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, add_ph_u16x16,
	add_ph_u16x16_merge_masked, add_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	add_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, add_ph_u16x8,
	add_ph_u16x8_merge_masked, add_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_binop_test!(
	sub_ph_u16x32_masked_matches_unmasked, Avx512Fp16, sub_ph_u16x32,
	sub_ph_u16x32_merge_masked, sub_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	sub_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, sub_ph_u16x16,
	sub_ph_u16x16_merge_masked, sub_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	sub_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, sub_ph_u16x8,
	sub_ph_u16x8_merge_masked, sub_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_binop_test!(
	mul_ph_u16x32_masked_matches_unmasked, Avx512Fp16, mul_ph_u16x32,
	mul_ph_u16x32_merge_masked, mul_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	mul_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, mul_ph_u16x16,
	mul_ph_u16x16_merge_masked, mul_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	mul_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, mul_ph_u16x8,
	mul_ph_u16x8_merge_masked, mul_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_binop_test!(
	div_ph_u16x32_masked_matches_unmasked, Avx512Fp16, div_ph_u16x32,
	div_ph_u16x32_merge_masked, div_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	div_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, div_ph_u16x16,
	div_ph_u16x16_merge_masked, div_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	div_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, div_ph_u16x8,
	div_ph_u16x8_merge_masked, div_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_binop_test!(
	min_ph_u16x32_masked_matches_unmasked, Avx512Fp16, min_ph_u16x32,
	min_ph_u16x32_merge_masked, min_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	min_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, min_ph_u16x16,
	min_ph_u16x16_merge_masked, min_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	min_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, min_ph_u16x8,
	min_ph_u16x8_merge_masked, min_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_binop_test!(
	max_ph_u16x32_masked_matches_unmasked, Avx512Fp16, max_ph_u16x32,
	max_ph_u16x32_merge_masked, max_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	max_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, max_ph_u16x16,
	max_ph_u16x16_merge_masked, max_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	max_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, max_ph_u16x8,
	max_ph_u16x8_merge_masked, max_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);

fp16_masked_unop_test!(
	sqrt_ph_u16x32_masked_matches_unmasked, Avx512Fp16, sqrt_ph_u16x32,
	sqrt_ph_u16x32_merge_masked, sqrt_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_unop_test!(
	sqrt_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, sqrt_ph_u16x16,
	sqrt_ph_u16x16_merge_masked, sqrt_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_unop_test!(
	sqrt_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, sqrt_ph_u16x8,
	sqrt_ph_u16x8_merge_masked, sqrt_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_unop_test!(
	rsqrt_ph_u16x32_masked_matches_unmasked, Avx512Fp16, rsqrt_ph_u16x32,
	rsqrt_ph_u16x32_merge_masked, rsqrt_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_unop_test!(
	rsqrt_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, rsqrt_ph_u16x16,
	rsqrt_ph_u16x16_merge_masked, rsqrt_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_unop_test!(
	rsqrt_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, rsqrt_ph_u16x8,
	rsqrt_ph_u16x8_merge_masked, rsqrt_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_unop_test!(
	rcp_ph_u16x32_masked_matches_unmasked, Avx512Fp16, rcp_ph_u16x32,
	rcp_ph_u16x32_merge_masked, rcp_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_unop_test!(
	rcp_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, rcp_ph_u16x16,
	rcp_ph_u16x16_merge_masked, rcp_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_unop_test!(
	rcp_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, rcp_ph_u16x8,
	rcp_ph_u16x8_merge_masked, rcp_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);

fp16_masked_ternop_test!(
	fmadd_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fmadd_ph_u16x32,
	fmadd_ph_u16x32_merge_masked, fmadd_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fmadd_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fmadd_ph_u16x16,
	fmadd_ph_u16x16_merge_masked, fmadd_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fmadd_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fmadd_ph_u16x8,
	fmadd_ph_u16x8_merge_masked, fmadd_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_ternop_test!(
	fmsub_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fmsub_ph_u16x32,
	fmsub_ph_u16x32_merge_masked, fmsub_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fmsub_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fmsub_ph_u16x16,
	fmsub_ph_u16x16_merge_masked, fmsub_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fmsub_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fmsub_ph_u16x8,
	fmsub_ph_u16x8_merge_masked, fmsub_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_ternop_test!(
	fnmadd_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fnmadd_ph_u16x32,
	fnmadd_ph_u16x32_merge_masked, fnmadd_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fnmadd_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fnmadd_ph_u16x16,
	fnmadd_ph_u16x16_merge_masked, fnmadd_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fnmadd_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fnmadd_ph_u16x8,
	fnmadd_ph_u16x8_merge_masked, fnmadd_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_ternop_test!(
	fnmsub_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fnmsub_ph_u16x32,
	fnmsub_ph_u16x32_merge_masked, fnmsub_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fnmsub_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fnmsub_ph_u16x16,
	fnmsub_ph_u16x16_merge_masked, fnmsub_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fnmsub_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fnmsub_ph_u16x8,
	fnmsub_ph_u16x8_merge_masked, fnmsub_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_ternop_test!(
	fmaddsub_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fmaddsub_ph_u16x32,
	fmaddsub_ph_u16x32_merge_masked, fmaddsub_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fmaddsub_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fmaddsub_ph_u16x16,
	fmaddsub_ph_u16x16_merge_masked, fmaddsub_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fmaddsub_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fmaddsub_ph_u16x8,
	fmaddsub_ph_u16x8_merge_masked, fmaddsub_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);
fp16_masked_ternop_test!(
	fmsubadd_ph_u16x32_masked_matches_unmasked, Avx512Fp16, fmsubadd_ph_u16x32,
	fmsubadd_ph_u16x32_merge_masked, fmsubadd_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_ternop_test!(
	fmsubadd_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, fmsubadd_ph_u16x16,
	fmsubadd_ph_u16x16_merge_masked, fmsubadd_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_ternop_test!(
	fmsubadd_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, fmsubadd_ph_u16x8,
	fmsubadd_ph_u16x8_merge_masked, fmsubadd_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);

fp16_masked_unop_test!(
	getexp_ph_u16x32_masked_matches_unmasked, Avx512Fp16, getexp_ph_u16x32,
	getexp_ph_u16x32_merge_masked, getexp_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_unop_test!(
	getexp_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, getexp_ph_u16x16,
	getexp_ph_u16x16_merge_masked, getexp_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_unop_test!(
	getexp_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, getexp_ph_u16x8,
	getexp_ph_u16x8_merge_masked, getexp_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);

fp16_masked_binop_test!(
	scalef_ph_u16x32_masked_matches_unmasked, Avx512Fp16, scalef_ph_u16x32,
	scalef_ph_u16x32_merge_masked, scalef_ph_u16x32_zero_masked, 32, u32, 0x5A5A_5A5Au32
);
fp16_masked_binop_test!(
	scalef_ph_u16x16_masked_matches_unmasked, Avx512Fp16Vl, scalef_ph_u16x16,
	scalef_ph_u16x16_merge_masked, scalef_ph_u16x16_zero_masked, 16, u16, 0x5A5Au16
);
fp16_masked_binop_test!(
	scalef_ph_u16x8_masked_matches_unmasked, Avx512Fp16Vl, scalef_ph_u16x8,
	scalef_ph_u16x8_merge_masked, scalef_ph_u16x8_zero_masked, 8, u8, 0x5Au8
);

// Scalar `_sh` masked ops only apply merge/zero semantics to lane 0;
// lanes 1..8 are unconditionally passed through from `a` (matching the
// unmasked scalar `_sh` ops' own "passed through from a" behavior),
// never `src`-controlled: confirmed via `sde64` execution after the
// generic packed-binop test macro's full-8-lane assumption failed here.
macro_rules! fp16_masked_sh_binop_test {
	($name:ident, $fixed_fn:ident, $merge_fn:ident, $zero_fn:ident, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Fp16Vl::detect() else { return };
			let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
			let b: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
			let src: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
			let mask: u8 = $mask_val;
			let unmasked = t.$fixed_fn(a, b);
			let mut merge_expect = a;
			merge_expect[0] = if mask & 1 == 1 { unmasked[0] } else { src[0] };
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let mut zero_expect = a;
			zero_expect[0] = if mask & 1 == 1 { unmasked[0] } else { 0 };
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

fp16_masked_sh_binop_test!(rcp_sh_masked_matches_unmasked, rcp_sh_u16x8, rcp_sh_merge_masked, rcp_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(rsqrt_sh_masked_matches_unmasked, rsqrt_sh_u16x8, rsqrt_sh_merge_masked, rsqrt_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(sqrt_sh_masked_matches_unmasked, sqrt_sh_u16x8, sqrt_sh_merge_masked, sqrt_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(min_sh_masked_matches_unmasked, min_sh_u16x8, min_sh_merge_masked, min_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(max_sh_masked_matches_unmasked, max_sh_u16x8, max_sh_merge_masked, max_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(getexp_sh_masked_matches_unmasked, getexp_sh_u16x8, getexp_sh_merge_masked, getexp_sh_zero_masked, 0x5Au8);
fp16_masked_sh_binop_test!(scalef_sh_masked_matches_unmasked, scalef_sh_u16x8, scalef_sh_merge_masked, scalef_sh_zero_masked, 0x5Au8);

#[test]
fn reduce_ph_masked_matches_unmasked() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	const IMM: i32 = 3;
        let mask32: u32 = 0x5A5A_5A5A;
	let unmasked32 = t512.reduce_ph_u16x32::<IMM>(a32);
	let merge32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { src32[i] });
	assert_eq!(t512.reduce_ph_u16x32_merge_masked::<IMM>(src32, mask32, a32), merge32);
	let zero32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { 0 });
	assert_eq!(t512.reduce_ph_u16x32_zero_masked::<IMM>(mask32, a32), zero32);

	let a8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask8: u8 = 0x5A;
	let unmasked8 = t.reduce_ph_u16x8::<IMM>(a8);
	let merge8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { src8[i] });
	assert_eq!(t.reduce_ph_u16x8_merge_masked::<IMM>(src8, mask8, a8), merge8);
	let zero8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { 0 });
	assert_eq!(t.reduce_ph_u16x8_zero_masked::<IMM>(mask8, a8), zero8);
}

#[test]
fn roundscale_ph_masked_matches_unmasked() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	const IMM: i32 = 0;
	let mask32: u32 = 0x5A5A_5A5A;
	let unmasked32 = t512.roundscale_ph_u16x32::<IMM>(a32);
	let merge32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { src32[i] });
	assert_eq!(t512.roundscale_ph_u16x32_merge_masked::<IMM>(src32, mask32, a32), merge32);
	let zero32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { 0 });
	assert_eq!(t512.roundscale_ph_u16x32_zero_masked::<IMM>(mask32, a32), zero32);

	let a8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask8: u8 = 0x5A;
	let unmasked8 = t.roundscale_ph_u16x8::<IMM>(a8);
	let merge8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { src8[i] });
	assert_eq!(t.roundscale_ph_u16x8_merge_masked::<IMM>(src8, mask8, a8), merge8);
	let zero8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { 0 });
	assert_eq!(t.roundscale_ph_u16x8_zero_masked::<IMM>(mask8, a8), zero8);
}

// Same lane0-only masking as the `fp16_masked_sh_binop_test!` scalar ops
// above; `reduce_sh`/`roundscale_sh` add an `IMM8`, no other change.
#[test]
fn reduce_sh_masked_matches_unmasked() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let b: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
	let src: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	const IMM: i32 = 3;
	let mask: u8 = 0x5A;
	let unmasked = t.reduce_sh_u16x8::<IMM>(a, b);
	let mut merge = a;
	merge[0] = if mask & 1 == 1 { unmasked[0] } else { src[0] };
	assert_eq!(t.reduce_sh_u16x8_merge_masked::<IMM>(src, mask, a, b), merge);
	let mut zero = a;
	zero[0] = if mask & 1 == 1 { unmasked[0] } else { 0 };
	assert_eq!(t.reduce_sh_u16x8_zero_masked::<IMM>(mask, a, b), zero);
}

#[test]
fn roundscale_sh_masked_matches_unmasked() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let b: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
	let src: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	const IMM: i32 = 0;
	let mask: u8 = 0x5A;
	let unmasked = t.roundscale_sh_u16x8::<IMM>(a, b);
	let mut merge = a;
	merge[0] = if mask & 1 == 1 { unmasked[0] } else { src[0] };
	assert_eq!(t.roundscale_sh_u16x8_merge_masked::<IMM>(src, mask, a, b), merge);
	let mut zero = a;
	zero[0] = if mask & 1 == 1 { unmasked[0] } else { 0 };
	assert_eq!(t.roundscale_sh_u16x8_zero_masked::<IMM>(mask, a, b), zero);
}

#[test]
fn getmant_ph_masked_matches_unmasked() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	use core::arch::x86_64::{_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC};
	let a32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src32: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask32: u32 = 0x5A5A_5A5A;
	let unmasked32 = t512.getmant_ph_u16x32::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a32);
	let merge32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { src32[i] });
	assert_eq!(t512.getmant_ph_u16x32_merge_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(src32, mask32, a32), merge32);
	let zero32: [u16; 32] = core::array::from_fn(|i| if (mask32 >> i) & 1 == 1 { unmasked32[i] } else { 0 });
	assert_eq!(t512.getmant_ph_u16x32_zero_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(mask32, a32), zero32);

	let a8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 1.0));
	let src8: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask8: u8 = 0x5A;
	let unmasked8 = t.getmant_ph_u16x8::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a8);
	let merge8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { src8[i] });
	assert_eq!(t.getmant_ph_u16x8_merge_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(src8, mask8, a8), merge8);
	let zero8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { 0 });
	assert_eq!(t.getmant_ph_u16x8_zero_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(mask8, a8), zero8);
}

#[test]
fn getmant_sh_masked_matches_unmasked() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	use core::arch::x86_64::{_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC};
	let a: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let b: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
	let src: [u16; 8] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask: u8 = 0x5A;
	let unmasked = t.getmant_sh_u16x8::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(a, b);
	let mut merge = a;
	merge[0] = if mask & 1 == 1 { unmasked[0] } else { src[0] };
	assert_eq!(t.getmant_sh_merge_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(src, mask, a, b), merge);
	let mut zero = a;
	zero[0] = if mask & 1 == 1 { unmasked[0] } else { 0 };
	assert_eq!(t.getmant_sh_zero_masked::<_MM_MANT_NORM_1_2, _MM_MANT_SIGN_SRC>(mask, a, b), zero);
}

#[test]
fn cvtph_pd_masked_matches_unmasked() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.5, -2.25, 3.0, 65504.0, 0.5, -0.5, 1.0, -1.0]);

	let src8 = [999.0f64; 8];
	let mask8: u8 = 0x5A;
	let unmasked8 = t512.ph_to_f64x8(a);
	let merge8: [f64; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { src8[i] });
	assert_eq!(t512.ph_to_f64x8_merge_masked(src8, mask8, a), merge8);
	let zero8: [f64; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { 0.0 });
	assert_eq!(t512.ph_to_f64x8_zero_masked(mask8, a), zero8);

	let src4 = [999.0f64; 4];
	let mask4: u8 = 0x0A;
	let unmasked4 = t.ph_to_f64x4(a);
	let merge4: [f64; 4] = core::array::from_fn(|i| if (mask4 >> i) & 1 == 1 { unmasked4[i] } else { src4[i] });
	assert_eq!(t.ph_to_f64x4_merge_masked(src4, mask4, a), merge4);

	let src2 = [999.0f64; 2];
	let mask2: u8 = 0x02;
	let unmasked2 = t.ph_to_f64x2(a);
	let merge2: [f64; 2] = core::array::from_fn(|i| if (mask2 >> i) & 1 == 1 { unmasked2[i] } else { src2[i] });
	assert_eq!(t.ph_to_f64x2_merge_masked(src2, mask2, a), merge2);
}

#[test]
fn cvtpd_ph_masked_matches_unmasked_and_zeros_upper_lanes() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };

	let a8: [f64; 8] = [1.5, -2.25, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let src8 = [999u16; 8];
	let mask8: u8 = 0x5A;
	let unmasked8 = t512.f64x8_to_ph(a8);
	let merge8: [u16; 8] = core::array::from_fn(|i| if (mask8 >> i) & 1 == 1 { unmasked8[i] } else { src8[i] });
	assert_eq!(t512.f64x8_to_ph_merge_masked(src8, mask8, a8), merge8);

	// f64x2_to_ph: only lanes 0-1 are real (mask bits 0-1); lanes 2-7 are
	// hardware-zeroed unconditionally, same "oversized carrier" rule as
	// i64_to_f32x2 in avx512vl.rs (confirmed via `sde64` execution here
	// after the naive full-8-lane formula failed).
	let a2: [f64; 2] = [1.5, -2.25];
	let src2 = [999u16; 8];
	let mask2: u8 = 0x02;
	let unmasked2 = t.f64x2_to_ph(a2);
	let mut merge2 = [0u16; 8];
	for i in 0..2 {
		merge2[i] = if (mask2 >> i) & 1 == 1 { unmasked2[i] } else { src2[i] };
	}
	assert_eq!(t.f64x2_to_ph_merge_masked(src2, mask2, a2), merge2);
}

#[test]
fn round_ph_masked_matches_unmasked() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let b: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
	let src: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(-(i as f32) - 100.0));
	let mask: u32 = 0x5A5A_5A5A;

	let unmasked_add = t.add_round_ph_u16x32::<RN>(a, b);
	let merge_add: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked_add[i] } else { src[i] });
	assert_eq!(t.add_round_ph_u16x32_merge_masked::<RN>(src, mask, a, b), merge_add);

	let unmasked_sub = t.sub_round_ph_u16x32::<RN>(a, b);
	let merge_sub: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked_sub[i] } else { src[i] });
	assert_eq!(t.sub_round_ph_u16x32_merge_masked::<RN>(src, mask, a, b), merge_sub);

	let unmasked_mul = t.mul_round_ph_u16x32::<RN>(a, b);
	let merge_mul: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked_mul[i] } else { src[i] });
	assert_eq!(t.mul_round_ph_u16x32_merge_masked::<RN>(src, mask, a, b), merge_mul);

	let unmasked_div = t.div_round_ph_u16x32::<RN>(a, b);
	let merge_div: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked_div[i] } else { src[i] });
	assert_eq!(t.div_round_ph_u16x32_merge_masked::<RN>(src, mask, a, b), merge_div);

	let unmasked_sqrt = t.sqrt_round_ph_u16x32::<RN>(a);
	let merge_sqrt: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked_sqrt[i] } else { src[i] });
	assert_eq!(t.sqrt_round_ph_u16x32_merge_masked::<RN>(src, mask, a), merge_sqrt);
}

#[test]
fn fma_round_ph_masked_matches_unmasked() {
	let Some(t) = Avx512Fp16::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 + 1.0));
	let b: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.5 + 2.0));
	let c: [u16; 32] = core::array::from_fn(|i| f32_to_f16_scalar(i as f32 * 0.25 - 3.0));
	let mask: u32 = 0x5A5A_5A5A;
	let zero: [u16; 32] = [0; 32];

	// FMA merge has no separate `src`: unmasked lanes keep `a`, not a 4th
	// operand: same shape as the unmasked `fmadd_round_ph_u16x32` etc.
	macro_rules! check {
		($unmasked:ident, $merge:ident, $zero:ident) => {
			let unmasked = t.$unmasked::<RN>(a, b, c);
			let merge: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { a[i] });
			assert_eq!(t.$merge::<RN>(a, mask, b, c), merge);
			let zeroed: [u16; 32] = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { unmasked[i] } else { zero[i] });
			assert_eq!(t.$zero::<RN>(mask, a, b, c), zeroed);
		};
	}
	check!(fmadd_round_ph_u16x32, fmadd_round_ph_u16x32_merge_masked, fmadd_round_ph_u16x32_zero_masked);
	check!(fmsub_round_ph_u16x32, fmsub_round_ph_u16x32_merge_masked, fmsub_round_ph_u16x32_zero_masked);
	check!(fnmadd_round_ph_u16x32, fnmadd_round_ph_u16x32_merge_masked, fnmadd_round_ph_u16x32_zero_masked);
	check!(fnmsub_round_ph_u16x32, fnmsub_round_ph_u16x32_merge_masked, fnmsub_round_ph_u16x32_zero_masked);
	check!(fmaddsub_round_ph_u16x32, fmaddsub_round_ph_u16x32_merge_masked, fmaddsub_round_ph_u16x32_zero_masked);
	check!(fmsubadd_round_ph_u16x32, fmsubadd_round_ph_u16x32_merge_masked, fmsubadd_round_ph_u16x32_zero_masked);
}

#[test]
fn cmp_ph_mask_gated_matches_unmasked_and_k1() {
	let Some(t512) = Avx512Fp16::detect() else { return };
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a32 = ph_carrier(&[1.0, 2.0, 3.0, f32::NAN, 0.0, 0.0, 0.0, 0.0]);
	let mut a32full = [0u16; 32];
	a32full[..8].copy_from_slice(&a32);
	let mut b32full = [0u16; 32];
	b32full[..8].copy_from_slice(&ph_carrier(&[2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
	let unmasked32 = t512.cmpeq_ph_mask_u16x32(a32full, b32full);
	assert_eq!(t512.cmpeq_ph_mask_u16x32_gated(u32::MAX, a32full, b32full), unmasked32);
	assert_eq!(t512.cmpeq_ph_mask_u16x32_gated(0, a32full, b32full), 0);

	let a16 = ph_carrier(&[1.0, 2.0, 3.0, f32::NAN, 0.0, 0.0, 0.0, 0.0]);
	let mut a16full = [0u16; 16];
	a16full[..8].copy_from_slice(&a16);
	let mut b16full = [0u16; 16];
	b16full[..8].copy_from_slice(&ph_carrier(&[2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
	let unmasked16 = t.cmpeq_ph_mask_u16x16(a16full, b16full);
	assert_eq!(t.cmpeq_ph_mask_u16x16_gated(u16::MAX, a16full, b16full), unmasked16);
	assert_eq!(t.cmpeq_ph_mask_u16x16_gated(0, a16full, b16full), 0);

	let a8 = ph_carrier(&[1.0, 2.0, 3.0, f32::NAN, 0.0, 0.0, 0.0, 0.0]);
	let b8 = ph_carrier(&[2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
	let unmasked8 = t.cmpeq_ph_mask_u16x8(a8, b8);
	assert_eq!(t.cmpeq_ph_mask_u16x8_gated(u8::MAX, a8, b8), unmasked8);
	assert_eq!(t.cmpeq_ph_mask_u16x8_gated(0, a8, b8), 0);
}

#[test]
fn cmp_sh_mask_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512Fp16Vl::detect() else { return };
	let a = ph_carrier(&[1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
	let b = ph_carrier(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
	let unmasked = t.cmplt_sh_mask_u16x8(a, b);
	assert_eq!(t.cmplt_sh_mask_gated(0xFF, a, b), unmasked);
	assert_eq!(t.cmplt_sh_mask_gated(0x00, a, b), 0);
}

#[test]
fn ph_f64_roundtrip_is_exact_for_representable_values() {
	let Some(t) = Avx512Fp16::detect() else { return };
	let vals = [0.0f32, -0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1023.5];
	let a = ph_carrier(&vals);
	let as_f64 = t.ph_to_f64x8(a);
	let back = t.f64x8_to_ph(as_f64);
	assert_eq!(back, a);
}
