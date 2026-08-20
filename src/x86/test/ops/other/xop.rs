use super::*;

/// Every real target of this crate (Intel, or AMD since ~2015) lacks XOP;
/// this asserts `detect` fails closed rather than false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Xop::detect().is_none(), "host has real XOP, review this test");
}

#[test]
fn rotl_u32x4_matches_scalar_rotate_left() {
	let Some(xop) = Xop::detect() else { return };
	let a = [0x0000_0001u32, 0xF000_0000, 0x8000_0000, 0x1234_5678];
	let counts = [1u32, 4, 1, 8];
	let mut expect = [0u32; 4];
	for i in 0..4 {
		expect[i] = a[i].rotate_left(counts[i]);
	}
	assert_eq!(xop.rotl_u32x4(a, counts), expect);
}
