use super::*;

/// Knights Landing is EOL and outside this crate's reachable test
/// matrix; this asserts `detect` fails closed rather than
/// false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Avx512Er::detect().is_none(), "host has real AVX512ER, review this test");
}

/// Relative-error tolerance for the 28-bit approximations, matching the
/// SDM's documented `<= 2^-28` bound with headroom.
const RCP_TOL: f32 = 1e-7;
const EXP_TOL: f32 = 1e-6;

#[test]
fn rcp28_f32x16_approximates_reciprocal() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	let got = t.rcp28_f32x16(a);
	for i in 0..16 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= RCP_TOL * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}

#[test]
fn rsqrt28_f32x16_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0);
	let got = t.rsqrt28_f32x16(a);
	for i in 0..16 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= RCP_TOL * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}

#[test]
fn exp2_f32x16_approximates_base2_exponential() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.25 - 2.0);
	let got = t.exp2_f32x16(a);
	for i in 0..16 {
		let expect = a[i].exp2();
		assert!((got[i] - expect).abs() <= EXP_TOL * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}

#[test]
fn rcp28_f64x8_approximates_reciprocal() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.0);
	let got = t.rcp28_f64x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i];
		assert!((got[i] - expect).abs() <= RCP_TOL as f64 * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}

#[test]
fn rsqrt28_f64x8_approximates_reciprocal_sqrt() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.0);
	let got = t.rsqrt28_f64x8(a);
	for i in 0..8 {
		let expect = 1.0 / a[i].sqrt();
		assert!((got[i] - expect).abs() <= RCP_TOL as f64 * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}

#[test]
fn exp2_f64x8_approximates_base2_exponential() {
	let Some(t) = Avx512Er::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64 * 0.25 - 2.0);
	let got = t.exp2_f64x8(a);
	for i in 0..8 {
		let expect = a[i].exp2();
		assert!((got[i] - expect).abs() <= EXP_TOL as f64 * expect.abs(), "i={i} got={} expect={expect}", got[i]);
	}
}
