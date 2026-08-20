use super::*;

#[test]
fn madd52lo_u64x2_adds_low_52_bits_of_product() {
	let Some(t) = AvxIfma::detect() else { return };
	let src = [1000u64, 2000];
	let a = [3, 5];
	let b = [4, 6];
	assert_eq!(t.madd52lo_u64x2(src, a, b), [1012, 2030]);
}

#[test]
fn madd52lo_u64x2_ignores_bits_above_52_in_operands() {
	let Some(t) = AvxIfma::detect() else { return };
	// Top 12 operand bits masked before multiply.
	let a = [(1u64 << 52) | 3, 0];
	let b = [4, 0];
	let src = [0u64, 0];
	assert_eq!(t.madd52lo_u64x2(src, a, b), [12, 0]);
}

#[test]
fn madd52hi_u64x2_matches_scalar_reference() {
	let Some(t) = AvxIfma::detect() else { return };
	let src = [7u64, 42];
	let a = [MASK52, MASK52 / 3];
	let b = [MASK52, MASK52 / 7];
	let expect = [madd52hi_scalar(src[0], a[0], b[0]), madd52hi_scalar(src[1], a[1], b[1])];
	assert_eq!(t.madd52hi_u64x2(src, a, b), expect);
}

#[test]
fn madd52lo_u64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxIfma::detect() else { return };
	for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<u64> = (0..len).map(|i| i as u64 * 1000).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		t.madd52lo_u64_slice(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52lo_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn madd52hi_u64_slice_wide_matches_scalar_for_various_lengths() {
	let Some(t) = AvxIfma::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let src: Vec<u64> = (0..len).map(|i| i as u64).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		t.madd52hi_u64_slice_wide(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52hi_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}
