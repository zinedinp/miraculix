use super::*;

#[test]
fn extract_bits_matches_shift_and_mask() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let x = 0x0123_4567_89ab_cdefu64;
	// bits [15:8] of x are 0xcd.
	assert_eq!(sse4a.extract_bits::<8, 8>(x), 0xcd);
}

#[test]
fn extract_bits_zero_len_returns_full_64_bits() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let x = 0x0123_4567_89ab_cdefu64;
	assert_eq!(sse4a.extract_bits::<0, 0>(x), x);
}

#[test]
fn insert_bits_overwrites_selected_field() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let dst = 0x0000_0000_0000_0000u64;
	let src = 0xffu64;
	// Insert 8 bits of `src` at bit offset 8: bits [15:8] become 0xff.
	assert_eq!(sse4a.insert_bits::<8, 8>(dst, src), 0x0000_0000_0000_ff00);
}

#[test]
fn insert_bits_leaves_other_bits_of_dst_unchanged() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let dst = 0xffff_ffff_ffff_ffffu64;
	let src = 0x0u64;
	// Insert 8 zero bits at offset 8: only bits [15:8] clear, rest of dst stays all-1s.
	assert_eq!(sse4a.insert_bits::<8, 8>(dst, src), 0xffff_ffff_ffff_00ff);
}

#[test]
fn insert_then_extract_roundtrips() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let dst = 0x0123_4567_89ab_cdefu64;
	let src = 0xab;
	let inserted = sse4a.insert_bits::<8, 16>(dst, src);
	assert_eq!(sse4a.extract_bits::<8, 16>(inserted), 0xab);
}

#[test]
fn insert_bits_zero_len_replaces_full_64_bits() {
	let Some(sse4a) = Sse4a::detect() else { return };
	let dst = 0x0123_4567_89ab_cdefu64;
	let src = 0xfedc_ba98_7654_3210u64;
	assert_eq!(sse4a.insert_bits::<0, 0>(dst, src), src);
}
