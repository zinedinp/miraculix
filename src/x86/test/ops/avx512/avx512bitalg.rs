use super::*;

#[test]
fn popcnt_u8x64_matches_core() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let expect: [u8; 64] = core::array::from_fn(|i| a[i].count_ones() as u8);
	assert_eq!(t.popcnt_u8x64(a), expect);
}

#[test]
fn popcnt_u16x32_matches_core() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(6151) ^ 0x5A5A);
	let expect: [u16; 32] = core::array::from_fn(|i| a[i].count_ones() as u16);
	assert_eq!(t.popcnt_u16x32(a), expect);
}

#[test]
fn popcnt_u8_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	for len in [0usize, 1, 63, 64, 65, 100, 200] {
		let a: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(53)).collect();
		let mut out = vec![0u8; len];
		t.popcnt_u8_slice(&a, &mut out);
		let expect: Vec<u8> = a.iter().map(|x| x.count_ones() as u8).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	for len in [0usize, 1, 31, 32, 33, 70, 200] {
		let a: Vec<u16> = (0..len).map(|i| (i as u16).wrapping_mul(6151)).collect();
		let mut out = vec![0u16; len];
		t.popcnt_u16_slice(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|x| x.count_ones() as u16).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

macro_rules! masked_unop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Bitalg::detect() else { return };
			let a = $a;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

masked_unop_test!(
	popcnt_u8x64_masked_matches_core, popcnt_u8x64_merge_masked, popcnt_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| (i as u8).wrapping_mul(37) ^ 0xA5),
	core::array::from_fn::<u8, 64, _>(|i| (i as u8).wrapping_add(200)), 0x9A37_5C81_2468_ACE0u64,
	|x: u8| x.count_ones() as u8
);
masked_unop_test!(
	popcnt_u16x32_masked_matches_core, popcnt_u16x32_merge_masked, popcnt_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(6151) ^ 0x5A5A),
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_add(9000)), 0x9A37_5C81u32,
	|x: u16| x.count_ones() as u16
);

#[test]
fn bitshuffle_mask_u64x8_matches_scalar_reference() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	let b: [u64; 8] = [
		0x0123_4567_89AB_CDEF,
		u64::MAX,
		0,
		1,
		0x8000_0000_0000_0000,
		0x5555_5555_5555_5555,
		0xF0F0_F0F0_F0F0_F0F0,
		42,
	];
	let c: [u64; 8] = core::array::from_fn(|i| (i as u64).wrapping_mul(0x0102_0304_0506_0708) ^ 0xDEAD_BEEF);
	let expect = bitshuffle_scalar(&b, &c);
	assert_eq!(t.bitshuffle_mask_u64x8(b, c), expect);
}

#[test]
fn bitshuffle_mask_u64x8_high_index_bits_are_masked_to_6_bits() {
	let Some(t) = Avx512Bitalg::detect() else { return };
	// 0xFF & 0x3F == 63 -> top bit of the lane.
	let b: [u64; 8] = [1 << 63, 0, 0, 0, 0, 0, 0, 0];
	let c: [u64; 8] = [0xFF, 0, 0, 0, 0, 0, 0, 0];
	assert_eq!(t.bitshuffle_mask_u64x8(b, c), 1);
}
