use super::Neon;

fn require() -> Option<Neon> {
	Neon::detect()
}

#[test]
fn add_i32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1, -2, i32::MAX, i32::MIN];
	let b = [10, -20, 1, -1];
	let expect = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
	assert_eq!(t.add_i32x4(a, b), expect);
}

#[test]
fn sub_i32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1, -2, i32::MIN, i32::MAX];
	let b = [10, -20, 1, -1];
	let expect = core::array::from_fn(|i| a[i].wrapping_sub(b[i]));
	assert_eq!(t.sub_i32x4(a, b), expect);
}

#[test]
fn mul_i32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1, -2, 70000, i32::MIN];
	let b = [10, -20, 70000, -1];
	let expect = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(t.mul_i32x4(a, b), expect);
}

#[test]
fn add_f32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1.5, -2.25, f32::MAX, f32::MIN];
	let b = [10.25, -20.5, 1.0, -1.0];
	let expect = core::array::from_fn(|i| a[i] + b[i]);
	assert_eq!(t.add_f32x4(a, b), expect);
}

#[test]
fn sub_f32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1.5, -2.25, 0.0, f32::MAX];
	let b = [10.25, -20.5, 1.0, -1.0];
	let expect = core::array::from_fn(|i| a[i] - b[i]);
	assert_eq!(t.sub_f32x4(a, b), expect);
}

#[test]
fn mul_f32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [1.5, -2.25, 3.0, -4.0];
	let b = [10.25, -20.5, 0.5, -0.5];
	let expect = core::array::from_fn(|i| a[i] * b[i]);
	assert_eq!(t.mul_f32x4(a, b), expect);
}

#[test]
fn and_u32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [0xFFFF_0000, 0x0F0F_0F0F, u32::MAX, 0];
	let b = [0x0000_FFFF, 0xF0F0_F0F0, 0, u32::MAX];
	let expect = core::array::from_fn(|i| a[i] & b[i]);
	assert_eq!(t.and_u32x4(a, b), expect);
}

#[test]
fn or_u32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [0xFFFF_0000, 0x0F0F_0F0F, u32::MAX, 0];
	let b = [0x0000_FFFF, 0xF0F0_F0F0, 0, u32::MAX];
	let expect = core::array::from_fn(|i| a[i] | b[i]);
	assert_eq!(t.or_u32x4(a, b), expect);
}

#[test]
fn xor_u32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [0xFFFF_0000, 0x0F0F_0F0F, u32::MAX, 0];
	let b = [0x0000_FFFF, 0xF0F0_F0F0, 0, u32::MAX];
	let expect = core::array::from_fn(|i| a[i] ^ b[i]);
	assert_eq!(t.xor_u32x4(a, b), expect);
}

#[test]
fn andnot_u32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [0xFFFF_0000, 0x0F0F_0F0F, u32::MAX, 0];
	let b = [0x0000_FFFF, 0xF0F0_F0F0, 0, u32::MAX];
	// Native `vbicq_u32` order: `a & !b` (mirror image of x86 `andnot`, see
	// `Neon::andnot_u32x4`'s doc comment).
	let expect = core::array::from_fn(|i| a[i] & !b[i]);
	assert_eq!(t.andnot_u32x4(a, b), expect);
}

const MASK_TRUE: u32 = u32::MAX;
const MASK_FALSE: u32 = 0;

#[test]
fn cmp_i32x4_family_matches_scalar_lane_masks() {
	let Some(t) = require() else { return };
	let a = [1, -2, 5, i32::MIN];
	let b = [1, 2, 3, i32::MAX];
	let mask = |f: fn(i32, i32) -> bool| -> [u32; 4] {
		core::array::from_fn(|i| if f(a[i], b[i]) { MASK_TRUE } else { MASK_FALSE })
	};
	assert_eq!(t.cmpeq_i32x4(a, b), mask(|x, y| x == y));
	assert_eq!(t.cmpgt_i32x4(a, b), mask(|x, y| x > y));
	assert_eq!(t.cmpge_i32x4(a, b), mask(|x, y| x >= y));
	assert_eq!(t.cmplt_i32x4(a, b), mask(|x, y| x < y));
	assert_eq!(t.cmple_i32x4(a, b), mask(|x, y| x <= y));
}

#[test]
fn cmp_f32x4_family_matches_scalar_lane_masks() {
	let Some(t) = require() else { return };
	let a = [1.0, -2.5, 5.0, f32::NAN];
	let b = [1.0, 2.5, 3.0, 1.0];
	let mask = |f: fn(f32, f32) -> bool| -> [u32; 4] {
		core::array::from_fn(|i| if f(a[i], b[i]) { MASK_TRUE } else { MASK_FALSE })
	};
	assert_eq!(t.cmpeq_f32x4(a, b), mask(|x, y| x == y));
	assert_eq!(t.cmpgt_f32x4(a, b), mask(|x, y| x > y));
	assert_eq!(t.cmpge_f32x4(a, b), mask(|x, y| x >= y));
	assert_eq!(t.cmplt_f32x4(a, b), mask(|x, y| x < y));
	assert_eq!(t.cmple_f32x4(a, b), mask(|x, y| x <= y));
}

#[test]
fn max_min_i32x4_match_scalar() {
	let Some(t) = require() else { return };
	let a = [1, -2, 5, i32::MIN];
	let b = [1, 2, 3, i32::MAX];
	let expect_max: [i32; 4] = core::array::from_fn(|i| a[i].max(b[i]));
	let expect_min: [i32; 4] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(t.max_i32x4(a, b), expect_max);
	assert_eq!(t.min_i32x4(a, b), expect_min);
}

#[test]
fn max_min_f32x4_match_scalar() {
	let Some(t) = require() else { return };
	let a: [f32; 4] = [1.0, -2.5, 5.0, -1.0];
	let b: [f32; 4] = [1.0, 2.5, 3.0, 1.0];
	let expect_max: [f32; 4] = core::array::from_fn(|i| a[i].max(b[i]));
	let expect_min: [f32; 4] = core::array::from_fn(|i| a[i].min(b[i]));
	assert_eq!(t.max_f32x4(a, b), expect_max);
	assert_eq!(t.min_f32x4(a, b), expect_min);
}

#[test]
fn abs_neg_i32x4_match_scalar() {
	let Some(t) = require() else { return };
	let a = [1, -2, 5, i32::MIN + 1];
	let expect_abs: [i32; 4] = core::array::from_fn(|i| a[i].wrapping_abs());
	let expect_neg: [i32; 4] = core::array::from_fn(|i| a[i].wrapping_neg());
	assert_eq!(t.abs_i32x4(a), expect_abs);
	assert_eq!(t.neg_i32x4(a), expect_neg);
}

#[test]
fn abs_neg_f32x4_match_scalar() {
	let Some(t) = require() else { return };
	let a: [f32; 4] = [1.0, -2.5, 5.0, -0.0];
	let expect_abs: [f32; 4] = core::array::from_fn(|i| a[i].abs());
	let expect_neg: [f32; 4] = core::array::from_fn(|i| -a[i]);
	assert_eq!(t.abs_f32x4(a), expect_abs);
	assert_eq!(t.neg_f32x4(a), expect_neg);
}

#[test]
fn not_u32x4_matches_scalar() {
	let Some(t) = require() else { return };
	let a = [0xFFFF_0000, 0x0F0F_0F0F, u32::MAX, 0];
	let expect: [u32; 4] = core::array::from_fn(|i| !a[i]);
	assert_eq!(t.not_u32x4(a), expect);
}

#[test]
fn shl_i32x4_matches_scalar_variable_shift() {
	let Some(t) = require() else { return };
	let a: [i32; 4] = [1, -8, 100, -100];
	let shift: [i32; 4] = [3, -2, 0, 4];
	let expect: [i32; 4] = core::array::from_fn(|i| {
		if shift[i] >= 0 { a[i].wrapping_shl(shift[i] as u32) } else { a[i].wrapping_shr((-shift[i]) as u32) }
	});
	assert_eq!(t.shl_i32x4(a, shift), expect);
}

#[test]
fn select_u32x4_matches_scalar_bit_select() {
	let Some(t) = require() else { return };
	let mask = [MASK_TRUE, MASK_FALSE, 0xFFFF_0000, 0x0000_FFFF];
	let b = [1, 2, 0xAAAA_AAAA, 0xBBBB_BBBB];
	let c = [3, 4, 0x5555_5555, 0x1234_5678];
	let expect: [u32; 4] = core::array::from_fn(|i| (mask[i] & b[i]) | (!mask[i] & c[i]));
	assert_eq!(t.select_u32x4(mask, b, c), expect);
}
