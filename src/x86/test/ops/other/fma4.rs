use super::*;

/// Every real target of this crate (Intel, or AMD since ~2015) lacks FMA4;
/// this asserts `detect` fails closed rather than false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Fma4::detect().is_none(), "host has real FMA4, review this test");
}

#[test]
fn fmadd_f32x4_matches_scalar() {
	let Some(fma4) = Fma4::detect() else { return };
	let a = [1.0, 2.0, 3.0, 4.0];
	let b = [10.0, 20.0, 30.0, 40.0];
	let c = [100.0, 200.0, 300.0, 400.0];
	let mut expect = [0f32; 4];
	for i in 0..4 {
		expect[i] = a[i] * b[i] + c[i];
	}
	assert_eq!(fma4.fmadd_f32x4(a, b, c), expect);
}
