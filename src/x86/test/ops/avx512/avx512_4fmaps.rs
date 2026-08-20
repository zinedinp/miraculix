use super::*;

/// Knights Mill is EOL and outside this crate's reachable test matrix;
/// this asserts `detect` fails closed rather than false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Avx5124fmaps::detect().is_none(), "host has real AVX512_4FMAPS, review this test");
}

#[test]
fn p4fmadd_f32x16_matches_manual_four_way_accumulate() {
	let Some(t) = Avx5124fmaps::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [[f32; 16]; 4] = core::array::from_fn(|n| core::array::from_fn(|i| (n * 16 + i) as f32 - 20.0));
	let c = [3.0f32, -1.5, 0.0, 2.0];
	let expect: [f32; 16] =
		core::array::from_fn(|i| a[i] + b[0][i] * c[0] + b[1][i] * c[1] + b[2][i] * c[2] + b[3][i] * c[3]);
	assert_eq!(t.p4fmadd_f32x16(a, b, c), expect);
}

#[test]
fn p4fnmadd_f32x16_matches_manual_four_way_subtract() {
	let Some(t) = Avx5124fmaps::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [[f32; 16]; 4] = core::array::from_fn(|n| core::array::from_fn(|i| (n * 16 + i) as f32 - 20.0));
	let c = [3.0f32, -1.5, 0.0, 2.0];
	let expect: [f32; 16] =
		core::array::from_fn(|i| a[i] - b[0][i] * c[0] - b[1][i] * c[1] - b[2][i] * c[2] - b[3][i] * c[3]);
	assert_eq!(t.p4fnmadd_f32x16(a, b, c), expect);
}

#[test]
fn p4fmadd_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx5124fmaps::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let b: [Vec<f32>; 4] =
			core::array::from_fn(|n| (0..len).map(|i| (n * len + i) as f32 * 0.25 - 10.0).collect());
		let c = [1.0f32, -2.0, 0.5, 4.0];
		let mut out = vec![0f32; len];
		t.p4fmadd_f32_slice(&a, [&b[0], &b[1], &b[2], &b[3]], c, &mut out);
		let expect: Vec<f32> = (0..len)
			.map(|i| a[i] + b[0][i] * c[0] + b[1][i] * c[1] + b[2][i] * c[2] + b[3][i] * c[3])
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4fmadd_f32_slice_panics_on_length_mismatch() {
	let Some(t) = Avx5124fmaps::detect() else { return };
	let a = [0f32; 4];
	let b0 = [0f32; 4];
	let b1 = [0f32; 3];
	let b2 = [0f32; 4];
	let b3 = [0f32; 4];
	let mut out = [0f32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.p4fmadd_f32_slice(&a, [&b0, &b1, &b2, &b3], [0.0; 4], &mut out);
	}));
	assert!(result.is_err());
}
