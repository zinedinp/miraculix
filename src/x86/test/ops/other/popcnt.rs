use super::*;

#[test]
fn popcnt_u32_matches_count_ones() {
	let Some(popcnt) = Popcnt::detect() else { return };
	for x in [0u32, 1, 0xff, 0xffff_ffff, 0x8000_0000, 0x1234_5678] {
		assert_eq!(popcnt.popcnt_u32(x), x.count_ones());
	}
}

#[test]
fn popcnt_u64_matches_count_ones() {
	let Some(popcnt) = Popcnt::detect() else { return };
	for x in [0u64, 1, 0xff, 0xffff_ffff_ffff_ffff, 0x8000_0000_0000_0000, 0x1234_5678_9abc_def0] {
		assert_eq!(popcnt.popcnt_u64(x), x.count_ones() as u64);
	}
}

#[test]
fn popcnt_u32_slice_matches_scalar_for_various_lengths() {
	let Some(popcnt) = Popcnt::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 9, 100] {
		let a: Vec<u32> = (0..len).map(|i| (i as u32).wrapping_mul(0x9E37_79B9)).collect();
		let mut out = vec![0u32; len];
		popcnt.popcnt_u32_slice(&a, &mut out);
		let expect: Vec<u32> = a.iter().map(|x| x.count_ones()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn popcnt_u64_slice_matches_scalar_for_various_lengths() {
	let Some(popcnt) = Popcnt::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 9, 100] {
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
		let mut out = vec![0u64; len];
		popcnt.popcnt_u64_slice(&a, &mut out);
		let expect: Vec<u64> = a.iter().map(|x| x.count_ones() as u64).collect();
		assert_eq!(out, expect, "len={len}");
	}
}
