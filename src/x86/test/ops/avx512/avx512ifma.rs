use super::*;

#[test]
fn madd52lo_u64x8_adds_low_52_bits_of_product() {
	let Some(t) = Avx512Ifma::detect() else { return };
	let src = [1000u64, 2000, 0, 0, 0, 0, 0, 0];
	let a = [3, 5, 0, 0, 0, 0, 0, 0];
	let b = [4, 6, 0, 0, 0, 0, 0, 0];
	let out = t.madd52lo_u64x8(src, a, b);
	assert_eq!(out[0], 1012);
	assert_eq!(out[1], 2030);
}

#[test]
fn madd52lo_u64x8_ignores_bits_above_52_in_operands() {
	let Some(t) = Avx512Ifma::detect() else { return };
	// Operand high 12 bits are stripped before multiply; acc is not.
	let mut a = [0u64; 8];
	let mut b = [0u64; 8];
	a[0] = (1u64 << 52) | 3;
	b[0] = 4;
	let src = [0u64; 8];
	assert_eq!(t.madd52lo_u64x8(src, a, b)[0], 12);
}

#[test]
fn madd52hi_u64x8_matches_scalar_reference() {
	let Some(t) = Avx512Ifma::detect() else { return };
	let src: [u64; 8] = core::array::from_fn(|i| i as u64 * 7);
	let a: [u64; 8] = core::array::from_fn(|i| MASK52 / (i as u64 + 1));
	let b: [u64; 8] = core::array::from_fn(|i| MASK52 / (i as u64 + 3));
	let expect: [u64; 8] = core::array::from_fn(|i| madd52hi_scalar(src[i], a[i], b[i]));
	assert_eq!(t.madd52hi_u64x8(src, a, b), expect);
}

#[test]
fn madd52lo_u64x8_wrapping_add_on_accumulator() {
	let Some(t) = Avx512Ifma::detect() else { return };
	// 1*1 -> low52 = 1; u64::MAX + 1 wraps to 0.
	let src = [u64::MAX; 8];
	let a = [1u64; 8];
	let b = [1u64; 8];
	assert_eq!(t.madd52lo_u64x8(src, a, b), [0u64; 8]);
}

macro_rules! masked_ternop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Ifma::detect() else { return };
			let src: [u64; 8] = core::array::from_fn(|i| i as u64 * 7 + 1);
			let a: [u64; 8] = core::array::from_fn(|i| MASK52 / (i as u64 + 1));
			let b: [u64; 8] = core::array::from_fn(|i| MASK52 / (i as u64 + 3));
			let mask: u8 = 0x5A;
			let op: fn(u64, u64, u64) -> u64 = $op;
			let merge_expect: [u64; 8] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(src[i], a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect: [u64; 8] =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(src[i], a[i], b[i]) } else { 0 });
			assert_eq!(t.$zero_fn(mask, src, a, b), zero_expect, "zero");
		}
	};
}

masked_ternop_test!(
	madd52lo_u64x8_masked_matches_scalar, madd52lo_u64x8_merge_masked, madd52lo_u64x8_zero_masked, madd52lo_scalar
);
masked_ternop_test!(
	madd52hi_u64x8_masked_matches_scalar, madd52hi_u64x8_merge_masked, madd52hi_u64x8_zero_masked, madd52hi_scalar
);

#[test]
fn madd52lo_u64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Ifma::detect() else { return };
	for len in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
		let src: Vec<u64> = (0..len).map(|i| i as u64 * 1000).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		t.madd52lo_u64_slice(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52lo_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}
