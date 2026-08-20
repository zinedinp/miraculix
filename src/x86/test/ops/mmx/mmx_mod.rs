use super::*;

/// x86-64 psABI baseline: MMX always present.
#[test]
#[cfg(target_arch = "x86_64")]
fn detect_finds_mmx_on_x86_64() {
	assert!(Mmx::detect().is_some());
}

#[test]
fn add_u8x8_wraps_mod_256() {
	let Some(mmx) = Mmx::detect() else { return };
	let a = [250, 1, 2, 3, 4, 5, 6, 7];
	let b = [10, 1, 2, 3, 4, 5, 6, 7];
	assert_eq!(mmx.add_u8x8(a, b), [4, 2, 4, 6, 8, 10, 12, 14]);
}

#[test]
fn sub_u8x8_wraps_mod_256() {
	let Some(mmx) = Mmx::detect() else { return };
	let a = [5, 1, 2, 3, 4, 5, 6, 7];
	let b = [10, 1, 2, 3, 4, 5, 6, 7];
	assert_eq!(mmx.sub_u8x8(a, b), [251, 0, 0, 0, 0, 0, 0, 0]);
}

/// Lanes match scalar wrapping add/sub.
#[test]
fn matches_scalar_wrapping_on_random_lanes() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [u8; 8] = [17, 250, 3, 200, 91, 0, 255, 128];
	let b: [u8; 8] = [240, 10, 3, 200, 5, 255, 1, 128];

	let mut expect_add = [0u8; 8];
	let mut expect_sub = [0u8; 8];
	for i in 0..8 {
		expect_add[i] = a[i].wrapping_add(b[i]);
		expect_sub[i] = a[i].wrapping_sub(b[i]);
	}

	assert_eq!(mmx.add_u8x8(a, b), expect_add);
	assert_eq!(mmx.sub_u8x8(a, b), expect_sub);
}

/// `emms` leaves x87 usable after MMX.
#[test]
fn emms_leaves_x87_float_math_sane() {
	let Some(mmx) = Mmx::detect() else { return };
	let a = [1u8; 8];
	let b = [1u8; 8];
	let _ = mmx.add_u8x8(a, b);

	let x: f64 = 3.5;
	let y: f64 = 2.25;
	assert_eq!(x + y, 5.75);
}

#[test]
fn add_i8x8_matches_scalar_wrapping() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i8; 8] = [120, -120, 1, -1, 0, 100, -100, 5];
	let b: [i8; 8] = [10, -10, 1, -1, 0, 30, -30, -5];
	let expect: [i8; 8] = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
	assert_eq!(mmx.add_i8x8(a, b), expect);
}

#[test]
fn adds_i8x8_saturates_at_bounds() {
	let Some(mmx) = Mmx::detect() else { return };
	let mut a = [0i8; 8];
	let mut b = [0i8; 8];
	a[0] = i8::MAX;
	b[0] = 1;
	let mut expect = [0i8; 8];
	expect[0] = i8::MAX;
	assert_eq!(mmx.adds_i8x8(a, b), expect);
}

#[test]
fn adds_u8x8_saturates_at_bounds() {
	let Some(mmx) = Mmx::detect() else { return };
	let mut a = [0u8; 8];
	let mut b = [0u8; 8];
	a[0] = u8::MAX;
	b[0] = 1;
	let mut expect = [0u8; 8];
	expect[0] = u8::MAX;
	assert_eq!(mmx.adds_u8x8(a, b), expect);
}

#[test]
fn subs_u8x8_saturates_at_zero() {
	let Some(mmx) = Mmx::detect() else { return };
	let a = [0u8, 5, 255, 0, 0, 0, 0, 0];
	let b = [1u8, 10, 1, 0, 0, 0, 0, 0];
	assert_eq!(mmx.subs_u8x8(a, b), [0, 0, 254, 0, 0, 0, 0, 0]);
}

#[test]
fn cmpeq_i8x8_matches_scalar_equality() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
	let b: [i8; 8] = [1, 0, 3, 0, 5, 0, 7, 0];
	let expect: [i8; 8] = core::array::from_fn(|i| if a[i] == b[i] { -1 } else { 0 });
	assert_eq!(mmx.cmpeq_i8x8(a, b), expect);
}

#[test]
fn and_or_xor_andnot_u8x8_match_scalar_bitwise() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [u8; 8] = [0b1100_1100; 8];
	let b: [u8; 8] = [0b1010_1010; 8];
	assert_eq!(mmx.and_u8x8(a, b), [0b1000_1000; 8]);
	assert_eq!(mmx.or_u8x8(a, b), [0b1110_1110; 8]);
	assert_eq!(mmx.xor_u8x8(a, b), [0b0110_0110; 8]);
	assert_eq!(mmx.andnot_u8x8(a, b), [!0b1100_1100 & 0b1010_1010; 8]);
}

#[test]
fn add_i16x4_matches_scalar_wrapping() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [30000, -30000, 1, -1];
	let b: [i16; 4] = [10000, -10000, 1, -1];
	let expect: [i16; 4] = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
	assert_eq!(mmx.add_i16x4(a, b), expect);
}

#[test]
fn adds_i16x4_saturates_at_bounds() {
	let Some(mmx) = Mmx::detect() else { return };
	let mut a = [0i16; 4];
	let mut b = [0i16; 4];
	a[0] = i16::MAX;
	b[0] = 1;
	let mut expect = [0i16; 4];
	expect[0] = i16::MAX;
	assert_eq!(mmx.adds_i16x4(a, b), expect);
}

#[test]
fn mullo_i16x4_matches_scalar_wrapping_mul() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [1000, -1000, 3, 4];
	let b: [i16; 4] = [1000, 1000, 3, 4];
	let expect: [i16; 4] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(mmx.mullo_i16x4(a, b), expect);
}

#[test]
fn mulhi_i16x4_matches_scalar_high_bits() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [1000, -1000, 20000, -20000];
	let b: [i16; 4] = [1000, 1000, 20000, 20000];
	let expect: [i16; 4] = core::array::from_fn(|i| ((a[i] as i32 * b[i] as i32) >> 16) as i16);
	assert_eq!(mmx.mulhi_i16x4(a, b), expect);
}

#[test]
fn maddwd_i16x4_sums_products_of_pairs() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [1, 2, 3, 4];
	let b: [i16; 4] = [10, 20, 30, 40];
	// out[0] = 1*10 + 2*20 = 50; out[1] = 3*30 + 4*40 = 250.
	assert_eq!(mmx.maddwd_i16x4(a, b), [50, 250]);
}

#[test]
fn maddwd_i16x4_matches_scalar_reference() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [-100, 200, -300, 400];
	let b: [i16; 4] = [50, -60, 70, -80];
	let expect: [i32; 2] =
		[a[0] as i32 * b[0] as i32 + a[1] as i32 * b[1] as i32, a[2] as i32 * b[2] as i32 + a[3] as i32 * b[3] as i32];
	assert_eq!(mmx.maddwd_i16x4(a, b), expect);
}

#[test]
fn add_i32x2_matches_scalar_wrapping() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i32; 2] = [i32::MAX, -1];
	let b: [i32; 2] = [1, -1];
	let expect: [i32; 2] = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
	assert_eq!(mmx.add_i32x2(a, b), expect);
}

#[test]
fn cmpeq_u32x2_matches_scalar_equality() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [u32; 2] = [7, 8];
	let b: [u32; 2] = [7, 9];
	assert_eq!(mmx.cmpeq_u32x2(a, b), [!0u32, 0]);
}

#[test]
fn shl_i16x4_matches_scalar_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [1, 2, 3, 4];
	let expect: [i16; 4] = core::array::from_fn(|i| a[i].wrapping_shl(3));
	assert_eq!(mmx.shl_i16x4::<3>(a), expect);
}

#[test]
fn shr_u16x4_matches_scalar_logical_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [u16; 4] = [0xffff, 0x8000, 0x0002, 0x0001];
	let expect: [u16; 4] = core::array::from_fn(|i| a[i].wrapping_shr(2));
	assert_eq!(mmx.shr_u16x4::<2>(a), expect);
}

#[test]
fn sra_i16x4_matches_scalar_arithmetic_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i16; 4] = [-8, -1, 16, -16];
	let expect: [i16; 4] = core::array::from_fn(|i| a[i].wrapping_shr(2));
	assert_eq!(mmx.sra_i16x4::<2>(a), expect);
}

#[test]
fn shl_i32x2_matches_scalar_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i32; 2] = [1, -1];
	let expect: [i32; 2] = core::array::from_fn(|i| a[i].wrapping_shl(4));
	assert_eq!(mmx.shl_i32x2::<4>(a), expect);
}

#[test]
fn sra_i32x2_matches_scalar_arithmetic_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [i32; 2] = [-1000, 1000];
	let expect: [i32; 2] = core::array::from_fn(|i| a[i].wrapping_shr(3));
	assert_eq!(mmx.sra_i32x2::<3>(a), expect);
}

#[test]
fn shr_u32x2_matches_scalar_logical_shift() {
	let Some(mmx) = Mmx::detect() else { return };
	let a: [u32; 2] = [0xffff_ffff, 0x8000_0000];
	let expect: [u32; 2] = core::array::from_fn(|i| a[i].wrapping_shr(5));
	assert_eq!(mmx.shr_u32x2::<5>(a), expect);
}
