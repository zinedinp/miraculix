use super::*;

/// Knights Mill is EOL and outside this crate's reachable test matrix;
/// this asserts `detect` fails closed rather than false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Avx5124vnniw::detect().is_none(), "host has real AVX512_4VNNIW, review this test");
}

#[test]
fn p4dpwssd_i32x16_matches_manual_four_way_dot_product() {
	let Some(t) = Avx5124vnniw::detect() else { return };
	let src: [i32; 16] = core::array::from_fn(|i| i as i32 * 10);
	let a: [[i16; 32]; 4] = core::array::from_fn(|n| core::array::from_fn(|i| (n * 32 + i) as i16 - 60));
	let b: [i16; 8] = [3, -2, 1, 0, -1, 5, 2, -4];
	let expect: [i32; 16] = core::array::from_fn(|j| {
		let mut acc: i64 = src[j] as i64;
		for n in 0..4 {
			acc += a[n][2 * j] as i64 * b[2 * n] as i64 + a[n][2 * j + 1] as i64 * b[2 * n + 1] as i64;
		}
		acc as i32
	});
	assert_eq!(t.p4dpwssd_i32x16(src, a, b), expect);
}

#[test]
fn p4dpwssds_i32x16_saturates_at_i32_max() {
	let Some(t) = Avx5124vnniw::detect() else { return };
	let src = [i32::MAX; 16];
	let a = [[i16::MAX; 32]; 4];
	let b = [i16::MAX; 8];
	let out = t.p4dpwssds_i32x16(src, a, b);
	assert!(out.iter().all(|&x| x == i32::MAX));
}

#[test]
fn p4dpwssd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx5124vnniw::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: [Vec<i16>; 4] =
			core::array::from_fn(|n| (0..len * 2).map(|i| ((n * len * 2 + i) as i16) - 50).collect());
		let b: [i16; 8] = [3, -2, 1, 0, -1, 5, 2, -4];
		let mut out = vec![0i32; len];
		t.p4dpwssd_i32_slice(&src, [&a[0], &a[1], &a[2], &a[3]], b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let mut acc: i64 = src[j] as i64;
				for n in 0..4 {
					acc += a[n][2 * j] as i64 * b[2 * n] as i64 + a[n][2 * j + 1] as i64 * b[2 * n + 1] as i64;
				}
				acc as i32
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4dpwssd_i32_slice_panics_on_length_mismatch() {
	let Some(t) = Avx5124vnniw::detect() else { return };
	let src = [0i32; 4];
	let a0 = [0i16; 8];
	let a1 = [0i16; 7];
	let a2 = [0i16; 8];
	let a3 = [0i16; 8];
	let mut out = [0i32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.p4dpwssd_i32_slice(&src, [&a0, &a1, &a2, &a3], [0i16; 8], &mut out);
	}));
	assert!(result.is_err());
}
