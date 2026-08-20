use super::*;

#[test]
fn dpbusd_i32x4_sums_four_way_dot_product() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [1000, 2000, 3000, 4000];
	let a: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
	let b: [i8; 16] = [1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -2, -2, -2, -2];
	// groups: 10, -26, 84, -116 -> +src
	assert_eq!(t.dpbusd_i32x4(src, a, b), [1010, 1974, 3084, 3884]);
}

#[test]
fn dpbusds_i32x4_saturates_at_i32_max() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [i32::MAX - 5, 0, 0, 0];
	let a: [u8; 16] = [255; 16];
	let b: [i8; 16] = [127; 16];
	let out = t.dpbusds_i32x4(src, a, b);
	assert_eq!(out[0], i32::MAX);
}

#[test]
fn dpwssd_i32x4_sums_two_way_dot_product() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [0, 0, 0, 0];
	let a: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
	let b: [i16; 8] = [10, -10, 20, -20, 30, -30, 40, -40];
	assert_eq!(t.dpwssd_i32x4(src, a, b), [-10, -20, -30, -40]);
}

#[test]
fn dpwssds_i32x4_saturates_at_i32_min() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [i32::MIN + 100, 0, 0, 0];
	let a: [i16; 8] = [i16::MAX, i16::MAX, 0, 0, 0, 0, 0, 0];
	let b: [i16; 8] = [i16::MIN, i16::MIN, 0, 0, 0, 0, 0, 0];
	let out = t.dpwssds_i32x4(src, a, b);
	assert_eq!(out[0], i32::MIN);
}

#[test]
fn dpbusd_i32x8_matches_scalar_reference() {
	let Some(t) = AvxVnni::detect() else { return };
	let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 100);
	let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
	let b: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_sub(16));
	let mut expect = [0i32; 8];
	for j in 0..8 {
		let dot: i32 = (0..4).map(|k| a[4 * j + k] as i32 * b[4 * j + k] as i32).sum();
		expect[j] = src[j] + dot;
	}
	assert_eq!(t.dpbusd_i32x8(src, a, b), expect);
}

#[test]
fn dpbusd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnni::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<u8> = (0..len * 4).map(|i| i as u8).collect();
		let b: Vec<i8> = (0..len * 4).map(|i| (i as i8).wrapping_sub(5)).collect();
		let mut out = vec![0i32; len];
		t.dpbusd_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let dot: i32 = (0..4).map(|k| a[4 * j + k] as i32 * b[4 * j + k] as i32).sum();
			expect[j] = src[j].wrapping_add(dot);
		}
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn dpwssd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnni::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<i16> = (0..len * 2).map(|i| i as i16 - 10).collect();
		let b: Vec<i16> = (0..len * 2).map(|i| (i as i16) * 3 - 4).collect();
		let mut out = vec![0i32; len];
		t.dpwssd_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
			expect[j] = vnni_acc_wrapping(src[j], sum);
		}
		assert_eq!(out, expect, "len={len}");
	}
}

/// Always runs: helpers, not HW. Two `i16::MIN*MIN` = 2^31.
#[test]
fn vnni_acc_helpers_handle_i16_min_product_sum() {
	let sum = (i16::MIN as i64) * (i16::MIN as i64) * 2; // 2^31
	assert_eq!(sum, 1i64 << 31);
	assert_eq!(vnni_acc_wrapping(0, sum), i32::MIN); // wrap 2^31
	assert_eq!(vnni_acc_saturating(0, sum), i32::MAX); // sat 2^31
	// (MIN+100) + 2^31 = 100 exactly (no sat).
	assert_eq!(vnni_acc_saturating(i32::MIN + 100, sum), 100);
	assert_eq!(vnni_acc_saturating(1, sum), i32::MAX);
	assert_eq!(vnni_acc_wrapping(1, sum), i32::MIN.wrapping_add(1));
}

/// Rem-only length (1): forces scalar path on extreme products.
#[test]
fn dpwssds_i32_slice_rem_saturates_two_min_min_products() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [0i32];
	let a = [i16::MIN, i16::MIN];
	let b = [i16::MIN, i16::MIN];
	let mut out = [0i32];
	t.dpwssds_i32_slice(&src, &a, &b, &mut out);
	assert_eq!(out[0], i32::MAX);
}

#[test]
fn dpwssd_i32_slice_rem_wraps_two_min_min_products() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [0i32];
	let a = [i16::MIN, i16::MIN];
	let b = [i16::MIN, i16::MIN];
	let mut out = [0i32];
	t.dpwssd_i32_slice(&src, &a, &b, &mut out);
	assert_eq!(out[0], i32::MIN);
}

#[test]
fn dpbusd_i32_slice_panics_on_length_mismatch() {
	let Some(t) = AvxVnni::detect() else { return };
	let src = [0i32; 4];
	let a = [0u8; 16];
	let b = [0i8; 15];
	let mut out = [0i32; 4];
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		t.dpbusd_i32_slice(&src, &a, &b, &mut out);
	}));
	assert!(r.is_err(), "expected length-mismatch panic");
}
