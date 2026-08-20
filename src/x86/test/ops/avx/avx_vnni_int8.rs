use super::*;

#[test]
fn dpbssd_i32x4_sums_signed_signed_dot_product() {
	let Some(t) = AvxVnniInt8::detect() else { return };
	let src = [0, 0, 0, 0];
	let a: [i8; 16] = [-1, -2, -3, -4, 1, 2, 3, 4, -5, 5, -5, 5, 10, -10, 10, -10];
	let b: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, -1, -1, -1, -1];
	let mut expect = [0i32; 4];
	for j in 0..4 {
		expect[j] = (0..4).map(|k| a[4 * j + k] as i32 * b[4 * j + k] as i32).sum();
	}
	assert_eq!(t.dpbssd_i32x4(src, a, b), expect);
}

#[test]
fn dpbuud_i32x4_sums_unsigned_unsigned_dot_product() {
	let Some(t) = AvxVnniInt8::detect() else { return };
	let src = [100, 200, 300, 400];
	let a: [u8; 16] = [255; 16];
	let b: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
	let mut expect = [0i32; 4];
	for j in 0..4 {
		let dot: i32 = (0..4).map(|k| a[4 * j + k] as i32 * b[4 * j + k] as i32).sum();
		expect[j] = src[j] + dot;
	}
	assert_eq!(t.dpbuud_i32x4(src, a, b), expect);
}

#[test]
fn dpbsuds_i32x4_saturates_at_i32_max() {
	let Some(t) = AvxVnniInt8::detect() else { return };
	let src = [i32::MAX - 1, 0, 0, 0];
	let a: [i8; 16] = [127; 16];
	let b: [u8; 16] = [255; 16];
	let out = t.dpbsuds_i32x4(src, a, b);
	assert_eq!(out[0], i32::MAX);
}

#[test]
fn dpbssd_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnniInt8::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<i8> = (0..len * 4).map(|i| (i as i8).wrapping_sub(20)).collect();
		let b: Vec<i8> = (0..len * 4).map(|i| (i as i8).wrapping_mul(3)).collect();
		let mut out = vec![0i32; len];
		t.dpbssd_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
			expect[j] = vnni_acc_wrapping(src[j], sum);
		}
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn dpbuud_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnniInt8::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<u8> = (0..len * 4).map(|i| i as u8).collect();
		let b: Vec<u8> = (0..len * 4).map(|i| (i as u8).wrapping_mul(5)).collect();
		let mut out = vec![0i32; len];
		t.dpbuud_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
			expect[j] = vnni_acc_wrapping(src[j], sum);
		}
		assert_eq!(out, expect, "len={len}");
	}
}
