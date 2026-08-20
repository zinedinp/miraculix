use super::*;

#[test]
fn dpbusd_i32x16_sums_four_way_dot_product() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src: [i32; 16] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 64] = core::array::from_fn(|i| (i % 20) as u8 + 1);
	let b: [i8; 64] = core::array::from_fn(|i| ((i % 7) as i8) - 3);
	let expect: [i32; 16] = core::array::from_fn(|j| {
		let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
		vnni_acc_wrapping(src[j], sum)
	});
	assert_eq!(t.dpbusd_i32x16(src, a, b), expect);
}

#[test]
fn dpbusds_i32x16_saturates_at_i32_max() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MAX; 16];
	let a = [u8::MAX; 64];
	let b = [i8::MAX; 64];
	let out = t.dpbusds_i32x16(src, a, b);
	assert!(out.iter().all(|&x| x == i32::MAX));
}

#[test]
fn dpbusd_i32x16_wraps_on_overflow() {
	let Some(t) = Avx512Vnni::detect() else { return };
	// Non-`s` form wraps, does not sat. MAX + positive product -> wrap.
	let src = [i32::MAX; 16];
	let a = [u8::MAX; 64];
	let b = [i8::MAX; 64];
	let out = t.dpbusd_i32x16(src, a, b);
	let expect: [i32; 16] = core::array::from_fn(|_| {
		let sum: i64 = (0..4).map(|_| u8::MAX as i64 * i8::MAX as i64).sum();
		vnni_acc_wrapping(i32::MAX, sum)
	});
	assert_eq!(out, expect);
	assert!(out.iter().all(|&x| x != i32::MAX));
}

#[test]
fn dpwssd_i32x16_sums_two_way_dot_product() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src: [i32; 16] = [0; 16];
	let a: [i16; 32] = core::array::from_fn(|i| (i as i16) - 16);
	let b: [i16; 32] = core::array::from_fn(|i| (i as i16) - 16);
	let expect: [i32; 16] = core::array::from_fn(|j| {
		let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
		vnni_acc_wrapping(src[j], sum)
	});
	assert_eq!(t.dpwssd_i32x16(src, a, b), expect);
}

#[test]
fn dpwssds_i32x16_saturates_at_i32_min() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MIN; 16];
	let a = [i16::MIN; 32];
	let b = [i16::MAX; 32];
	let out = t.dpwssds_i32x16(src, a, b);
	assert!(out.iter().all(|&x| x == i32::MIN));
}

// Oracle is the already-tested unmasked op, same as the FP16/IFMA masked
// batch: isolates the one new behavior (lane selection) rather than
// re-deriving the dot-product scalar reference a second time.
const MASK16: u16 = 0x9A37;

#[test]
fn dpbusd_i32x16_masked_matches_unmasked() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src: [i32; 16] = core::array::from_fn(|i| i as i32 * 10);
	let a: [u8; 64] = core::array::from_fn(|i| (i % 20) as u8 + 1);
	let b: [i8; 64] = core::array::from_fn(|i| ((i % 7) as i8) - 3);
	let expect = t.dpbusd_i32x16(src, a, b);
	let merged = t.dpbusd_i32x16_merge_masked(src, MASK16, a, b);
	let zeroed = t.dpbusd_i32x16_zero_masked(MASK16, src, a, b);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "zero lane {i}");
	}
}

#[test]
fn dpbusds_i32x16_masked_saturates_selected_lanes_only() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MAX; 16];
	let a = [u8::MAX; 64];
	let b = [i8::MAX; 64];
	let merged = t.dpbusds_i32x16_merge_masked(src, MASK16, a, b);
	let expect = t.dpbusds_i32x16(src, a, b);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "lane {i}");
	}
}

#[test]
fn dpwssd_i32x16_masked_matches_unmasked() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src: [i32; 16] = [0; 16];
	let a: [i16; 32] = core::array::from_fn(|i| (i as i16) - 16);
	let b: [i16; 32] = core::array::from_fn(|i| (i as i16) - 16);
	let expect = t.dpwssd_i32x16(src, a, b);
	let zeroed = t.dpwssd_i32x16_zero_masked(MASK16, src, a, b);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "lane {i}");
	}
}

#[test]
fn dpwssds_i32x16_masked_matches_unmasked() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MIN; 16];
	let a = [i16::MIN; 32];
	let b = [i16::MAX; 32];
	let expect = t.dpwssds_i32x16(src, a, b);
	let merged = t.dpwssds_i32x16_merge_masked(src, MASK16, a, b);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "lane {i}");
	}
}

#[test]
fn dpbusd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Vnni::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<u8> = (0..len * 4).map(|i| (i % 20) as u8 + 1).collect();
		let b: Vec<i8> = (0..len * 4).map(|i| ((i % 7) as i8) - 3).collect();
		let mut out = vec![0i32; len];
		t.dpbusd_i32_slice(&src, &a, &b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
				vnni_acc_wrapping(src[j], sum)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn dpwssd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Vnni::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<i16> = (0..len * 2).map(|i| (i as i16) - 50).collect();
		let b: Vec<i16> = (0..len * 2).map(|i| (i as i16) - 25).collect();
		let mut out = vec![0i32; len];
		t.dpwssd_i32_slice(&src, &a, &b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
				vnni_acc_wrapping(src[j], sum)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn dpbusd_i32_slice_panics_on_length_mismatch() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [0i32; 4];
	let a = [0u8; 16];
	let b = [0i8; 15];
	let mut out = [0i32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.dpbusd_i32_slice(&src, &a, &b, &mut out);
	}));
	assert!(result.is_err());
}

#[test]
fn p4dpwssd_i32x16_matches_manual_four_way_dot_product() {
	let Some(t) = Avx512Vnni::detect() else { return };
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
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MAX; 16];
	let a = [[i16::MAX; 32]; 4];
	let b = [i16::MAX; 8];
	let out = t.p4dpwssds_i32x16(src, a, b);
	assert!(out.iter().all(|&x| x == i32::MAX));
}

#[test]
fn p4dpwssd_i32x16_wraps_on_overflow() {
	let Some(t) = Avx512Vnni::detect() else { return };
	let src = [i32::MAX; 16];
	let a = [[i16::MAX; 32]; 4];
	let b = [i16::MAX; 8];
	let out = t.p4dpwssd_i32x16(src, a, b);
	assert!(out.iter().all(|&x| x != i32::MAX));
}

#[test]
fn p4dpwssd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Vnni::detect() else { return };
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
	let Some(t) = Avx512Vnni::detect() else { return };
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
