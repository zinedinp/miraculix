use super::I8mm;

fn require() -> Option<I8mm> {
	I8mm::detect()
}

/// `USDOT` shape: same 4-consecutive-bytes-per-lane dot product as
/// `Dotprod::dot_s32`/`dot_u32`, just mixed sign (`a` unsigned, `b` signed).
fn oracle_dot_us32(acc: [i32; 4], a: [u8; 16], b: [i8; 16]) -> [i32; 4] {
	core::array::from_fn(|i| {
		acc[i] + (0..4).map(|k| a[4 * i + k] as i32 * b[4 * i + k] as i32).sum::<i32>()
	})
}

/// `SMMLA`/`UMMLA`/`USMMLA` 2x2-block layout (see `ops/i8mm.rs`'s module
/// doc): `a`/`b` each split into two 8-byte "rows"; lane `i` = `acc[i] +
/// dot(a_row[i/2], b_row[i%2])`.
fn mmla_lane<T: Copy, U: Copy>(acc: i64, a_row: &[T], b_row: &[U], mul: impl Fn(T, U) -> i64) -> i64 {
	acc + a_row.iter().zip(b_row).map(|(&x, &y)| mul(x, y)).sum::<i64>()
}

fn oracle_mmla_s32(acc: [i32; 4], a: [i8; 16], b: [i8; 16]) -> [i32; 4] {
	let mul = |x: i8, y: i8| x as i64 * y as i64;
	core::array::from_fn(|i| {
		mmla_lane(acc[i] as i64, &a[8 * (i / 2)..8 * (i / 2) + 8], &b[8 * (i % 2)..8 * (i % 2) + 8], mul) as i32
	})
}

fn oracle_mmla_u32(acc: [u32; 4], a: [u8; 16], b: [u8; 16]) -> [u32; 4] {
	let mul = |x: u8, y: u8| x as i64 * y as i64;
	core::array::from_fn(|i| {
		mmla_lane(acc[i] as i64, &a[8 * (i / 2)..8 * (i / 2) + 8], &b[8 * (i % 2)..8 * (i % 2) + 8], mul) as u32
	})
}

fn oracle_mmla_us32(acc: [i32; 4], a: [u8; 16], b: [i8; 16]) -> [i32; 4] {
	let mul = |x: u8, y: i8| x as i64 * y as i64;
	core::array::from_fn(|i| {
		mmla_lane(acc[i] as i64, &a[8 * (i / 2)..8 * (i / 2) + 8], &b[8 * (i % 2)..8 * (i % 2) + 8], mul) as i32
	})
}

#[test]
fn dot_us32_matches_scalar() {
	let Some(t) = require() else { return };
	let acc = [1, -2, 3, 4];
	let a: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
	let b: [i8; 16] = core::array::from_fn(|i| (i as i32 * -3 + 20) as i8);
	assert_eq!(t.dot_us32(acc, a, b), oracle_dot_us32(acc, a, b));
}

#[test]
fn mmla_s32_matches_2x2_block_reference() {
	let Some(t) = require() else { return };
	let acc = [1, -2, 3, -4];
	let a: [i8; 16] = core::array::from_fn(|i| (i as i32 * 7 - 50) as i8);
	let b: [i8; 16] = core::array::from_fn(|i| (i as i32 * -3 + 20) as i8);
	assert_eq!(t.mmla_s32(acc, a, b), oracle_mmla_s32(acc, a, b));
}

#[test]
fn mmla_u32_matches_2x2_block_reference() {
	let Some(t) = require() else { return };
	let acc = [1u32, 2, 3, 4];
	let a: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
	let b: [u8; 16] = core::array::from_fn(|i| (i * 23 + 5) as u8);
	assert_eq!(t.mmla_u32(acc, a, b), oracle_mmla_u32(acc, a, b));
}

#[test]
fn mmla_us32_matches_2x2_block_reference() {
	let Some(t) = require() else { return };
	let acc = [1, -2, 3, -4];
	let a: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
	let b: [i8; 16] = core::array::from_fn(|i| (i as i32 * -3 + 20) as i8);
	assert_eq!(t.mmla_us32(acc, a, b), oracle_mmla_us32(acc, a, b));
}
