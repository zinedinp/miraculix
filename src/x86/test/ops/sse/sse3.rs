use super::*;

#[test]
fn addsub_f32x4_alternates_sub_and_add() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	assert_eq!(sse3.addsub_f32x4(a, b), [1.0 - 10.0, 2.0 + 20.0, 3.0 - 30.0, 4.0 + 40.0]);
}

/// Lanes match scalar even-sub/odd-add.
#[test]
fn matches_scalar_on_random_lanes() {
	let Some(sse3) = Sse3::detect() else { return };
	let a: [f32; 4] = [17.5, -3.25, 0.0, 1e6];
	let b: [f32; 4] = [-240.75, 10.0, -3.5, 255.0];

	let mut expect = [0f32; 4];
	for i in 0..4 {
		expect[i] = if i % 2 == 0 { a[i] - b[i] } else { a[i] + b[i] };
	}
	assert_eq!(sse3.addsub_f32x4(a, b), expect);
}

#[test]
fn addsub_f64x2_alternates_sub_and_add() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0];
	let b = [10.0, 20.0];
	assert_eq!(sse3.addsub_f64x2(a, b), [1.0 - 10.0, 2.0 + 20.0]);
}

#[test]
fn hadd_f32x4_sums_pairs_within_each_input() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	assert_eq!(sse3.hadd_f32x4(a, b), [1.0 + 2.0, 3.0 + 4.0, 10.0 + 20.0, 30.0 + 40.0]);
}

#[test]
fn hsub_f32x4_subtracts_pairs_within_each_input() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	assert_eq!(sse3.hsub_f32x4(a, b), [1.0 - 2.0, 3.0 - 4.0, 10.0 - 20.0, 30.0 - 40.0]);
}

#[test]
fn hadd_f64x2_sums_pair_from_each_input() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0];
	let b = [10.0, 20.0];
	assert_eq!(sse3.hadd_f64x2(a, b), [1.0 + 2.0, 10.0 + 20.0]);
}

#[test]
fn hsub_f64x2_subtracts_pair_from_each_input() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0];
	let b = [10.0, 20.0];
	assert_eq!(sse3.hsub_f64x2(a, b), [1.0 - 2.0, 10.0 - 20.0]);
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
fn conj_c32x4_negates_imaginary_lanes() {
	let Some(sse3) = Sse3::detect() else { return };
	assert_eq!(sse3.conj_c32x4([1.0, 2.0, -3.0, 4.0]), [1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn mul_c32x4_matches_scalar_complex_multiply() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0, -3.0, 0.5];
	let b = [5.0, -1.0, 2.0, 4.0];
	let mut expect = [0f32; 4];
	scalar_mul_c(&a, &b, false, &mut expect);
	assert_eq!(sse3.mul_c32x4(a, b), expect);
}

#[test]
fn conj_mul_c32x4_matches_scalar_conjugate_multiply() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0, -3.0, 0.5];
	let b = [5.0, -1.0, 2.0, 4.0];
	let mut expect = [0f32; 4];
	scalar_mul_c(&a, &b, true, &mut expect);
	assert_eq!(sse3.conj_mul_c32x4(a, b), expect);
	// Also matches `mul(conj(a), b)` computed via the two separate ops.
	assert_eq!(sse3.conj_mul_c32x4(a, b), sse3.mul_c32x4(sse3.conj_c32x4(a), b));
}

#[test]
fn abs2_c32x4_matches_scalar_squared_magnitude() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [3.0, 4.0, -1.0, 2.0];
	assert_eq!(sse3.abs2_c32x4(a), [25.0, 25.0, 5.0, 5.0]);
}

#[test]
fn conj_c64x2_negates_imaginary_lane() {
	let Some(sse3) = Sse3::detect() else { return };
	assert_eq!(sse3.conj_c64x2([1.5, -2.5]), [1.5, 2.5]);
}

#[test]
fn mul_c64x2_matches_scalar_complex_multiply() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0];
	let b = [5.0, -1.0];
	// re = 1*5 - 2*-1 = 7, im = 1*-1 + 2*5 = 9.
	assert_eq!(sse3.mul_c64x2(a, b), [7.0, 9.0]);
}

#[test]
fn conj_mul_c64x2_matches_mul_with_conjugated_a() {
	let Some(sse3) = Sse3::detect() else { return };
	let a = [1.0, 2.0];
	let b = [5.0, -1.0];
	assert_eq!(sse3.conj_mul_c64x2(a, b), sse3.mul_c64x2(sse3.conj_c64x2(a), b));
}

#[test]
fn abs2_c64x2_matches_scalar_squared_magnitude() {
	let Some(sse3) = Sse3::detect() else { return };
	assert_eq!(sse3.abs2_c64x2([3.0, 4.0]), [25.0, 25.0]);
}
