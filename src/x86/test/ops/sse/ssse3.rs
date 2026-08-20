use super::*;

#[test]
fn abs_i8x16_wrapping_abs_of_min() {
	let Some(t) = Ssse3::detect() else { return };
	let mut a = [0i8; 16];
	a[0] = i8::MIN;
	a[1] = -5;
	let mut expect = [0i8; 16];
	expect[0] = i8::MIN; // wrapping_abs(MIN) == MIN
	expect[1] = 5;
	assert_eq!(t.abs_i8x16(a), expect);
}

#[test]
fn abs_i16x8_wrapping_abs_of_min() {
	let Some(t) = Ssse3::detect() else { return };
	let mut a = [0i16; 8];
	a[0] = i16::MIN;
	a[1] = -5;
	let mut expect = [0i16; 8];
	expect[0] = i16::MIN;
	expect[1] = 5;
	assert_eq!(t.abs_i16x8(a), expect);
}

#[test]
fn abs_i32x4_wrapping_abs_of_min() {
	let Some(t) = Ssse3::detect() else { return };
	let mut a = [0i32; 4];
	a[0] = i32::MIN;
	a[1] = -5;
	let mut expect = [0i32; 4];
	expect[0] = i32::MIN;
	expect[1] = 5;
	assert_eq!(t.abs_i32x4(a), expect);
}

#[test]
fn identity_indices_return_input_unchanged() {
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8 * 5);
	let identity: [i8; 16] = core::array::from_fn(|i| i as i8);
	assert_eq!(ssse3.shuffle_i8x16(a, identity), a);
}

#[test]
fn reversed_indices_reverse_lanes() {
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8);
	let reversed: [i8; 16] = core::array::from_fn(|i| (15 - i) as i8);
	let expect: [i8; 16] = core::array::from_fn(|i| (15 - i) as i8);
	assert_eq!(ssse3.shuffle_i8x16(a, reversed), expect);
}

#[test]
fn negative_index_zeroes_the_lane() {
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [i8; 16] = core::array::from_fn(|i| i as i8 + 1);
	let mut indices: [i8; 16] = core::array::from_fn(|i| i as i8);
	indices[3] = -1;
	let mut expect = a;
	expect[3] = 0;
	assert_eq!(ssse3.shuffle_i8x16(a, indices), expect);
}

#[test]
fn alignr_u8x16_imm0_returns_b_unchanged() {
	let Some(t) = Ssse3::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 16] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x16::<0>(a, b), b);
}

#[test]
fn alignr_u8x16_imm16_returns_a_unchanged() {
	let Some(t) = Ssse3::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8 + 100);
	let b: [u8; 16] = core::array::from_fn(|i| i as u8 + 1);
	assert_eq!(t.alignr_u8x16::<16>(a, b), a);
}

#[test]
fn alignr_u8x16_imm32_or_more_is_all_zero() {
	let Some(t) = Ssse3::detect() else { return };
	let a = [0xFFu8; 16];
	let b = [0xFFu8; 16];
	assert_eq!(t.alignr_u8x16::<32>(a, b), [0u8; 16]);
	assert_eq!(t.alignr_u8x16::<200>(a, b), [0u8; 16]);
}

#[test]
fn alignr_u8x16_matches_scalar_reference() {
	let Some(t) = Ssse3::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let b: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	for imm in [1, 5, 15, 17, 24, 31] {
		let expect = alignr_scalar(&a, &b, imm);
		let out = match imm {
			1 => t.alignr_u8x16::<1>(a, b),
			5 => t.alignr_u8x16::<5>(a, b),
			15 => t.alignr_u8x16::<15>(a, b),
			17 => t.alignr_u8x16::<17>(a, b),
			24 => t.alignr_u8x16::<24>(a, b),
			31 => t.alignr_u8x16::<31>(a, b),
			_ => unreachable!(),
		};
		assert_eq!(out.to_vec(), expect, "imm={imm}");
	}
}
