use super::Dotprod;

fn require() -> Option<Dotprod> {
	Dotprod::detect()
}

/// ARM ARM `DOT` pseudocode: lane `i` = `acc[i] + sum(a[4i+k] * b[4i+k])`
/// for `k` in `0..4`: 4 consecutive bytes per lane, no rounding.
fn oracle_s32(acc: [i32; 4], a: [i8; 16], b: [i8; 16]) -> [i32; 4] {
	core::array::from_fn(|i| {
		acc[i].wrapping_add((0..4).map(|k| a[4 * i + k] as i32 * b[4 * i + k] as i32).sum::<i32>())
	})
}

fn oracle_u32(acc: [u32; 4], a: [u8; 16], b: [u8; 16]) -> [u32; 4] {
	core::array::from_fn(|i| {
		acc[i].wrapping_add((0..4).map(|k| a[4 * i + k] as u32 * b[4 * i + k] as u32).sum::<u32>())
	})
}

#[test]
fn dot_s32_matches_scalar() {
	let Some(t) = require() else { return };
	let acc = [1, -2, 3, i32::MIN + 1000];
	let a: [i8; 16] = core::array::from_fn(|i| (i as i32 * 7 - 50) as i8);
	let b: [i8; 16] = core::array::from_fn(|i| (i as i32 * -3 + 20) as i8);
	assert_eq!(t.dot_s32(acc, a, b), oracle_s32(acc, a, b));
}

#[test]
fn dot_u32_matches_scalar() {
	let Some(t) = require() else { return };
	let acc = [1u32, 2, 3, 4];
	let a: [u8; 16] = core::array::from_fn(|i| (i * 17) as u8);
	let b: [u8; 16] = core::array::from_fn(|i| (i * 23 + 5) as u8);
	assert_eq!(t.dot_u32(acc, a, b), oracle_u32(acc, a, b));
}
