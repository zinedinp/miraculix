use super::*;

#[test]
fn dpwsud_i32x4_sums_signed_unsigned_dot_product() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	let src = [0, 0, 0, 0];
	let a: [i16; 8] = [-1, -2, 3, 4, -5, 6, 7, -8];
	let b: [u16; 8] = [1, 1, 2, 2, 3, 3, 4, 4];
	let mut expect = [0i32; 4];
	for j in 0..4 {
		expect[j] = (0..2).map(|k| a[2 * j + k] as i32 * b[2 * j + k] as i32).sum();
	}
	assert_eq!(t.dpwsud_i32x4(src, a, b), expect);
}

#[test]
fn dpwuud_i32x4_sums_unsigned_unsigned_dot_product() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	let src = [1000, 2000, 3000, 4000];
	let a: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
	let b: [u16; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
	let mut expect = [0i32; 4];
	for j in 0..4 {
		let dot: i32 = (0..2).map(|k| a[2 * j + k] as i32 * b[2 * j + k] as i32).sum();
		expect[j] = src[j] + dot;
	}
	assert_eq!(t.dpwuud_i32x4(src, a, b), expect);
}

#[test]
fn dpwusds_i32x4_saturates_at_i32_max() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	let src = [i32::MAX - 1, 0, 0, 0];
	let a: [u16; 8] = [u16::MAX, u16::MAX, 0, 0, 0, 0, 0, 0];
	let b: [i16; 8] = [i16::MAX, i16::MAX, 0, 0, 0, 0, 0, 0];
	let out = t.dpwusds_i32x4(src, a, b);
	assert_eq!(out[0], i32::MAX);
}

#[test]
fn dpwsud_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<i16> = (0..len * 2).map(|i| (i as i16).wrapping_sub(10)).collect();
		let b: Vec<u16> = (0..len * 2).map(|i| i as u16 * 3).collect();
		let mut out = vec![0i32; len];
		t.dpwsud_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
			expect[j] = vnni_acc_wrapping(src[j], sum);
		}
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn dpwuud_i32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<u16> = (0..len * 2).map(|i| i as u16).collect();
		let b: Vec<u16> = (0..len * 2).map(|i| i as u16 * 5).collect();
		let mut out = vec![0i32; len];
		t.dpwuud_i32_slice(&src, &a, &b, &mut out);

		let mut expect = vec![0i32; len];
		for j in 0..len {
			let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
			expect[j] = vnni_acc_wrapping(src[j], sum);
		}
		assert_eq!(out, expect, "len={len}");
	}
}

/// Rem-only: two `u16::MAX` products overflow i32 mid-sum without i64.
#[test]
fn dpwuud_i32_slice_rem_wraps_u16_max_products() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	let src = [0i32];
	let a = [u16::MAX, u16::MAX];
	let b = [u16::MAX, u16::MAX];
	let mut out = [0i32];
	t.dpwuud_i32_slice(&src, &a, &b, &mut out);
	let sum = (u16::MAX as i64) * (u16::MAX as i64) * 2;
	assert_eq!(out[0], vnni_acc_wrapping(0, sum));
}

#[test]
fn dpwuuds_i32_slice_rem_saturates_u16_max_products() {
	let Some(t) = AvxVnniInt16::detect() else { return };
	let src = [0i32];
	let a = [u16::MAX, u16::MAX];
	let b = [u16::MAX, u16::MAX];
	let mut out = [0i32];
	t.dpwuuds_i32_slice(&src, &a, &b, &mut out);
	assert_eq!(out[0], i32::MAX);
}
