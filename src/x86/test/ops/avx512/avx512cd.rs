use super::*;

#[test]
fn detect_matches_v4_from_level() {
	let Some(cd) = Avx512cd::detect() else { return };
	let level = GenericLevel::detect(FeatureSet::detect());
	assert!(Avx512cd::from_level(level).is_some());
	let _ = cd;
}

#[test]
fn leading_zeros_u32x16_matches_core() {
	let Some(cd) = Avx512cd::detect() else { return };
	let a: [u32; 16] = [0, 1, 2, 3, 4, 8, 16, 255, 256, 65535, 65536, u32::MAX, 1 << 30, 1 << 31, 7, 9];
	let expect: [u32; 16] = core::array::from_fn(|i| a[i].leading_zeros());
	assert_eq!(cd.leading_zeros_u32x16(a), expect);
}

#[test]
fn leading_zeros_u64x8_matches_core() {
	let Some(cd) = Avx512cd::detect() else { return };
	let a: [u64; 8] = [0, 1, 2, u32::MAX as u64, u64::MAX, 1 << 62, 1 << 63, 9];
	let expect: [u64; 8] = core::array::from_fn(|i| a[i].leading_zeros() as u64);
	assert_eq!(cd.leading_zeros_u64x8(a), expect);
}

#[test]
fn leading_zeros_u32_slice_matches_scalar_for_various_lengths() {
	let Some(cd) = Avx512cd::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 200] {
		let a: Vec<u32> = (0..len).map(|i| i as u32).collect();
		let mut out = vec![0u32; len];
		cd.leading_zeros_u32_slice(&a, &mut out);
		let expect: Vec<u32> = a.iter().map(|x| x.leading_zeros()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn leading_zeros_u64_slice_matches_scalar_for_various_lengths() {
	let Some(cd) = Avx512cd::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 17, 100] {
		let a: Vec<u64> = (0..len).map(|i| i as u64).collect();
		let mut out = vec![0u64; len];
		cd.leading_zeros_u64_slice(&a, &mut out);
		let expect: Vec<u64> = a.iter().map(|x| x.leading_zeros() as u64).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

macro_rules! masked_unop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(cd) = Avx512cd::detect() else { return };
			let a = $a;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { src[i] });
			assert_eq!(cd.$merge_fn(src, mask, a), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { Default::default() });
			assert_eq!(cd.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

masked_unop_test!(
	leading_zeros_u32x16_masked_matches_core, leading_zeros_u32x16_merge_masked, leading_zeros_u32x16_zero_masked,
	u16, [0u32, 1, 2, 3, 4, 8, 16, 255, 256, 65535, 65536, u32::MAX, 1 << 30, 1 << 31, 7, 9],
	core::array::from_fn::<u32, 16, _>(|i| i as u32 + 9000), 0x5555u16, u32::leading_zeros
);
masked_unop_test!(
	leading_zeros_u64x8_masked_matches_core, leading_zeros_u64x8_merge_masked, leading_zeros_u64x8_zero_masked,
	u8, [0u64, 1, 2, u32::MAX as u64, u64::MAX, 1 << 62, 1 << 63, 9],
	core::array::from_fn::<u64, 8, _>(|i| i as u64 + 9000), 0x55u8, |x: u64| x.leading_zeros() as u64
);

fn conflict_reference(window: &[u32]) -> Vec<u32> {
	(0..window.len())
		.map(|i| {
			let mut mask = 0u32;
			for j in 0..i {
				if window[j] == window[i] {
					mask |= 1 << j;
				}
			}
			mask
		})
		.collect()
}

fn conflict_reference_u64(window: &[u64]) -> Vec<u64> {
	(0..window.len())
		.map(|i| {
			let mut mask = 0u64;
			for j in 0..i {
				if window[j] == window[i] {
					mask |= 1 << j;
				}
			}
			mask
		})
		.collect()
}

#[test]
fn conflict_u32x16_finds_earlier_duplicates() {
	let Some(cd) = Avx512cd::detect() else { return };
	let a: [u32; 16] = [7, 3, 7, 7, 1, 3, 2, 7, 0, 0, 0, 5, 6, 7, 8, 9];
	let expect: [u32; 16] = conflict_reference(&a).try_into().unwrap();
	assert_eq!(cd.conflict_u32x16(a), expect);
}

#[test]
fn conflict_u64x8_finds_earlier_duplicates() {
	let Some(cd) = Avx512cd::detect() else { return };
	let a: [u64; 8] = [7, 3, 7, 7, 1, 3, 2, 7];
	let expect: [u64; 8] = conflict_reference_u64(&a).try_into().unwrap();
	assert_eq!(cd.conflict_u64x8(a), expect);
}

#[test]
fn conflict_u32_slice_matches_windowed_scalar_reference() {
	let Some(cd) = Avx512cd::detect() else { return };
	for len in [0usize, 1, 15, 16, 17, 33, 200] {
		let a: Vec<u32> = (0..len).map(|i| (i % 5) as u32).collect();
		let mut out = vec![0u32; len];
		cd.conflict_u32_slice(&a, &mut out);

		let expect: Vec<u32> = a.chunks(16).flat_map(conflict_reference).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn conflict_u64_slice_matches_windowed_scalar_reference() {
	let Some(cd) = Avx512cd::detect() else { return };
	for len in [0usize, 1, 7, 8, 9, 17, 100] {
		let a: Vec<u64> = (0..len).map(|i| (i % 3) as u64).collect();
		let mut out = vec![0u64; len];
		cd.conflict_u64_slice(&a, &mut out);

		let expect: Vec<u64> = a.chunks(8).flat_map(conflict_reference_u64).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn broadcast_mask_u32x16_replicates_mask_into_every_lane() {
	let Some(cd) = Avx512cd::detect() else { return };
	assert_eq!(cd.broadcast_mask_u32x16(0), [0u32; 16]);
	assert_eq!(cd.broadcast_mask_u32x16(0xFFFF), [0xFFFFu32; 16]);
	assert_eq!(cd.broadcast_mask_u32x16((1 << 3) | (1 << 10)), [(1u32 << 3) | (1 << 10); 16]);
}

#[test]
fn broadcast_mask_u64x8_replicates_mask_into_every_lane() {
	let Some(cd) = Avx512cd::detect() else { return };
	assert_eq!(cd.broadcast_mask_u64x8(0), [0u64; 8]);
	assert_eq!(cd.broadcast_mask_u64x8(0xFF), [0xFFu64; 8]);
	assert_eq!(cd.broadcast_mask_u64x8((1 << 0) | (1 << 7)), [(1u64 << 0) | (1 << 7); 8]);
}
