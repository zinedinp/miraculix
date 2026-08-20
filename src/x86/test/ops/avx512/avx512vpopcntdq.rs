use super::*;

macro_rules! masked_unop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vpopcntdq::detect() else { return };
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
	popcnt_u32x16_masked_matches_core, popcnt_u32x16_merge_masked, popcnt_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ 0xABCD_1234),
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 9000), 0x5555u16, u32::count_ones
);
masked_unop_test!(
	popcnt_u64x8_masked_matches_core, popcnt_u64x8_merge_masked, popcnt_u64x8_zero_masked, u8,
	[0u64, u64::MAX, 1, 1 << 63, 0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA, 3, 7],
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 9000), 0x55u8, |x: u64| x.count_ones() as u64
);

#[test]
fn popcnt_u32x16_matches_core() {
	let Some(t) = Avx512Vpopcntdq::detect() else { return };
	let a: [u32; 16] = core::array::from_fn(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ 0xABCD_1234);
	let expect: [u32; 16] = core::array::from_fn(|i| a[i].count_ones());
	assert_eq!(t.popcnt_u32x16(a), expect);
}

#[test]
fn popcnt_u64x8_matches_core() {
	let Some(t) = Avx512Vpopcntdq::detect() else { return };
	let a: [u64; 8] = [0, u64::MAX, 1, 1 << 63, 0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA, 3, 7];
	let expect: [u64; 8] = core::array::from_fn(|i| a[i].count_ones() as u64);
	assert_eq!(t.popcnt_u64x8(a), expect);
}

#[test]
fn popcnt_u32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Vpopcntdq::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 200] {
		let a: Vec<u32> = (0..len).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
		let mut out = vec![0u32; len];
		t.popcnt_u32_slice(&a, &mut out);
		let expect: Vec<u32> = a.iter().map(|x| x.count_ones()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u64_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Vpopcntdq::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 17, 100] {
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
		let mut out = vec![0u64; len];
		t.popcnt_u64_slice(&a, &mut out);
		let expect: Vec<u64> = a.iter().map(|x| x.count_ones() as u64).collect();
		assert_eq!(out, expect, "len={len}");
	}
}
