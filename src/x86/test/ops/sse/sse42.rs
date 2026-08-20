use super::*;

/// CRC-32C/iSCSI standard check value: byte-at-a-time over ASCII
/// "123456789", seeded and finalized with the usual CRC convention
/// (invert in, invert out), must equal `0xE3069283`.
#[test]
fn crc32_u8_matches_iscsi_check_value() {
	let Some(sse42) = Sse42::detect() else { return };
	let mut crc = 0xFFFF_FFFFu32;
	for &b in b"123456789" {
		crc = sse42.crc32_u8(crc, b);
	}
	assert_eq!(crc ^ 0xFFFF_FFFF, 0xE306_9283);
}

#[test]
fn crc32_u8_is_deterministic() {
	let Some(sse42) = Sse42::detect() else { return };
	assert_eq!(sse42.crc32_u8(0, 42), sse42.crc32_u8(0, 42));
}

#[test]
fn crc32_u16_matches_two_bytewise_folds() {
	let Some(sse42) = Sse42::detect() else { return };
	let v = 0xBEEFu16;
	let byte_folded = sse42.crc32_u8(sse42.crc32_u8(0x1234_5678, v.to_le_bytes()[0]), v.to_le_bytes()[1]);
	assert_eq!(sse42.crc32_u16(0x1234_5678, v), byte_folded);
}

#[test]
fn crc32_u32_matches_iscsi_check_value_via_le_word() {
	// "1234" as a little-endian u32 must fold identically to 4 byte-at-a-time steps.
	let Some(sse42) = Sse42::detect() else { return };
	let v = u32::from_le_bytes(*b"1234");
	let mut byte_folded = 0xFFFF_FFFFu32;
	for &b in b"1234" {
		byte_folded = sse42.crc32_u8(byte_folded, b);
	}
	assert_eq!(sse42.crc32_u32(0xFFFF_FFFF, v), byte_folded);
}

#[test]
fn crc32_u64_matches_eight_bytewise_folds() {
	let Some(sse42) = Sse42::detect() else { return };
	let v = u64::from_le_bytes(*b"12345678");
	let mut byte_folded = 0xFFFF_FFFFu32;
	for &b in b"12345678" {
		byte_folded = sse42.crc32_u8(byte_folded, b);
	}
	assert_eq!(sse42.crc32_u64(0xFFFF_FFFF, v), byte_folded as u64);
}

#[test]
fn crc32_u64_is_deterministic() {
	let Some(sse42) = Sse42::detect() else { return };
	assert_eq!(sse42.crc32_u64(0, 0x1122_3344_5566_7788), sse42.crc32_u64(0, 0x1122_3344_5566_7788));
}

#[test]
fn cmpgt_i64x2_matches_scalar() {
	let Some(t) = Sse42::detect() else { return };
	assert_eq!(t.cmpgt_i64x2([5, -1], [3, 2]), [-1, 0]);
}

#[test]
fn cmplt_cmple_cmpge_i64x2_match_scalar() {
	let Some(t) = Sse42::detect() else { return };
	let a = [5i64, -1];
	let b = [3i64, 2];
	assert_eq!(t.cmplt_i64x2(a, b), [0, -1]);
	assert_eq!(t.cmple_i64x2(a, b), [0, -1]);
	assert_eq!(t.cmpge_i64x2(a, b), [-1, 0]);
}

#[test]
fn cmpgt_u64x2_matches_scalar() {
	let Some(t) = Sse42::detect() else { return };
	assert_eq!(t.cmpgt_u64x2([u64::MAX, 3], [1, 5]), [!0, 0]);
}

#[test]
fn min_max_i64x2_match_scalar() {
	let Some(t) = Sse42::detect() else { return };
	let a = [i64::MIN, 7];
	let b = [3, -7];
	assert_eq!(t.min_i64x2(a, b), [i64::MIN, -7]);
	assert_eq!(t.max_i64x2(a, b), [3, 7]);
}

#[test]
fn min_max_u64x2_match_scalar() {
	let Some(t) = Sse42::detect() else { return };
	let a = [u64::MAX, 7];
	let b = [3, 20];
	assert_eq!(t.min_u64x2(a, b), [3, 7]);
	assert_eq!(t.max_u64x2(a, b), [u64::MAX, 20]);
}

#[test]
fn cmpgt_i64_slice_matches_scalar() {
	let Some(t) = Sse42::detect() else { return };
	for len in [0usize, 1, 2, 3, 5, 9] {
		let a: Vec<i64> = (0..len).map(|i| i as i64 - 2).collect();
		let b: Vec<i64> = (0..len).map(|i| (len - i) as i64 - 3).collect();
		let mut out = vec![0i64; len];
		t.cmpgt_i64_slice(&a, &b, &mut out);
		let expect: Vec<i64> = a.iter().zip(&b).map(|(&x, &y)| if x > y { -1 } else { 0 }).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn min_i64_slice_matches_scalar() {
	let Some(t) = Sse42::detect() else { return };
	for len in [0usize, 1, 2, 3, 5, 9] {
		let a: Vec<i64> = (0..len).map(|i| i as i64 - 2).collect();
		let b: Vec<i64> = (0..len).map(|i| (len - i) as i64 - 3).collect();
		let mut out = vec![0i64; len];
		t.min_i64_slice(&a, &b, &mut out);
		let expect: Vec<i64> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn min_u64_slice_matches_scalar() {
	let Some(t) = Sse42::detect() else { return };
	for len in [0usize, 1, 2, 3, 5, 9] {
		let a: Vec<u64> = (0..len).map(|i| i as u64).collect();
		let b: Vec<u64> = (0..len).map(|i| (len - i) as u64).collect();
		let mut out = vec![0u64; len];
		t.min_u64_slice(&a, &b, &mut out);
		let expect: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}
