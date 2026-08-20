use super::super::super::macros::slice_ternop_matches_scalar_test;
use super::super::super::super::{detect_level, GenericLevel};
use super::*;

#[test]
fn from_level_matches_v3() {
	let level = detect_level();
	assert_eq!(Fma::from_level(level).is_some(), level >= GenericLevel::V3);
}

#[test]
fn fmadd_f32x4_matches_mul_add() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	let b = [2.0f32, 2.0, 2.0, 2.0];
	let c = [1.0f32, 1.0, 1.0, 1.0];
	assert_eq!(fma.fmadd_f32x4(a, b, c), [3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn fmadd_f32x8_matches_mul_add() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let b = [1.0f32; 8];
	let c = [0.5f32; 8];
	let expect = [1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
	assert_eq!(fma.fmadd_f32x8(a, b, c), expect);
}

slice_ternop_matches_scalar_test!(
	fmadd_f32x4_slice_matches_scalar, Fma, fmadd_f32x4_slice, |a, b, c| a * b + c, f32
);
slice_ternop_matches_scalar_test!(
	fmadd_f32x8_slice_matches_scalar, Fma, fmadd_f32x8_slice, |a, b, c| a * b + c, f32
);
slice_ternop_matches_scalar_test!(
	fmadd_f64x2_slice_matches_scalar, Fma, fmadd_f64x2_slice, |a, b, c| a * b + c, f64
);
slice_ternop_matches_scalar_test!(
	fmadd_f64x4_slice_matches_scalar, Fma, fmadd_f64x4_slice, |a, b, c| a * b + c, f64
);

#[test]
fn fmsub_f32x4_matches_mul_sub() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	let b = [2.0f32, 2.0, 2.0, 2.0];
	let c = [1.0f32, 1.0, 1.0, 1.0];
	assert_eq!(fma.fmsub_f32x4(a, b, c), [1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn fnmadd_f32x4_negates_the_product() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	let b = [2.0f32, 2.0, 2.0, 2.0];
	let c = [10.0f32, 10.0, 10.0, 10.0];
	assert_eq!(fma.fnmadd_f32x4(a, b, c), [8.0, 6.0, 4.0, 2.0]);
}

#[test]
fn fnmsub_f32x4_negates_the_product_and_subtracts() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	let b = [2.0f32, 2.0, 2.0, 2.0];
	let c = [10.0f32, 10.0, 10.0, 10.0];
	assert_eq!(fma.fnmsub_f32x4(a, b, c), [-12.0, -14.0, -16.0, -18.0]);
}

slice_ternop_matches_scalar_test!(
	fmsub_f32x8_slice_matches_scalar, Fma, fmsub_f32x8_slice, |a, b, c| a * b - c, f32
);
slice_ternop_matches_scalar_test!(
	fmsub_f64x4_slice_matches_scalar, Fma, fmsub_f64x4_slice, |a, b, c| a * b - c, f64
);
slice_ternop_matches_scalar_test!(
	fnmadd_f32x8_slice_matches_scalar, Fma, fnmadd_f32x8_slice, |a: f32, b: f32, c: f32| -(a * b) + c, f32
);
slice_ternop_matches_scalar_test!(
	fnmadd_f64x4_slice_matches_scalar, Fma, fnmadd_f64x4_slice, |a: f64, b: f64, c: f64| -(a * b) + c, f64
);
slice_ternop_matches_scalar_test!(
	fnmsub_f32x8_slice_matches_scalar, Fma, fnmsub_f32x8_slice, |a: f32, b: f32, c: f32| -(a * b) - c, f32
);
slice_ternop_matches_scalar_test!(
	fnmsub_f64x4_slice_matches_scalar, Fma, fnmsub_f64x4_slice, |a: f64, b: f64, c: f64| -(a * b) - c, f64
);
