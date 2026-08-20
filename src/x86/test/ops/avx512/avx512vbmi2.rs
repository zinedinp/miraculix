use super::super::super::macros::{slice_binop_imm_matches_scalar_test, slice_ternop_matches_scalar_test};
use super::*;

#[test]
fn shldv_u32x16_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a = [0x1234_5678u32; 16];
	let b = [0xDEAD_BEEFu32; 16];
	let c = [0u32; 16];
	assert_eq!(t.shldv_u32x16(a, b, c), a);
}

#[test]
fn shrdv_u32x16_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a = [0x1234_5678u32; 16];
	let b = [0xDEAD_BEEFu32; 16];
	let c = [0u32; 16];
	assert_eq!(t.shrdv_u32x16(a, b, c), a);
}

#[test]
fn shldv_u16x32_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x9E37) ^ 0x1234);
	let b: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x7F4A) ^ 0xABCD);
	let c: [u16; 32] = core::array::from_fn(|i| i as u16);
	let expect: [u16; 32] = core::array::from_fn(|i| shldv_u16_scalar(a[i], b[i], c[i]));
	assert_eq!(t.shldv_u16x32(a, b, c), expect);
}

#[test]
fn shrdv_i64x8_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| (i as i64).wrapping_mul(0x1_0000_0007));
	let b: [i64; 8] = core::array::from_fn(|i| (i as i64).wrapping_mul(-0x3_0000_0001));
	let c: [i64; 8] = core::array::from_fn(|i| i as i64 * 7);
	let expect: [i64; 8] = core::array::from_fn(|i| shrdv_i64_scalar(a[i], b[i], c[i]));
	assert_eq!(t.shrdv_i64x8(a, b, c), expect);
}

slice_ternop_matches_scalar_test!(shldv_i16_slice_matches_scalar, Avx512Vbmi2, shldv_i16_slice, shldv_i16_scalar, i16);
slice_ternop_matches_scalar_test!(shldv_u16_slice_matches_scalar, Avx512Vbmi2, shldv_u16_slice, shldv_u16_scalar, u16);
slice_ternop_matches_scalar_test!(shrdv_i16_slice_matches_scalar, Avx512Vbmi2, shrdv_i16_slice, shrdv_i16_scalar, i16);
slice_ternop_matches_scalar_test!(shrdv_u16_slice_matches_scalar, Avx512Vbmi2, shrdv_u16_slice, shrdv_u16_scalar, u16);
slice_ternop_matches_scalar_test!(shldv_i32_slice_matches_scalar, Avx512Vbmi2, shldv_i32_slice, shldv_i32_scalar, i32);
slice_ternop_matches_scalar_test!(shldv_u32_slice_matches_scalar, Avx512Vbmi2, shldv_u32_slice, shldv_u32_scalar, u32);
slice_ternop_matches_scalar_test!(shrdv_i32_slice_matches_scalar, Avx512Vbmi2, shrdv_i32_slice, shrdv_i32_scalar, i32);
slice_ternop_matches_scalar_test!(shrdv_u32_slice_matches_scalar, Avx512Vbmi2, shrdv_u32_slice, shrdv_u32_scalar, u32);
slice_ternop_matches_scalar_test!(shldv_i64_slice_matches_scalar, Avx512Vbmi2, shldv_i64_slice, shldv_i64_scalar, i64);
slice_ternop_matches_scalar_test!(shldv_u64_slice_matches_scalar, Avx512Vbmi2, shldv_u64_slice, shldv_u64_scalar, u64);
slice_ternop_matches_scalar_test!(shrdv_i64_slice_matches_scalar, Avx512Vbmi2, shrdv_i64_slice, shrdv_i64_scalar, i64);
slice_ternop_matches_scalar_test!(shrdv_u64_slice_matches_scalar, Avx512Vbmi2, shrdv_u64_slice, shrdv_u64_scalar, u64);

#[test]
fn shldi_u32x16_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a = [0x1234_5678u32; 16];
	let b = [0xDEAD_BEEFu32; 16];
	assert_eq!(t.shldi_u32x16::<0>(a, b), a);
}

#[test]
fn shrdi_u32x16_shift_by_zero_returns_a() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a = [0x1234_5678u32; 16];
	let b = [0xDEAD_BEEFu32; 16];
	assert_eq!(t.shrdi_u32x16::<0>(a, b), a);
}

#[test]
fn shldi_u16x32_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x9E37) ^ 0x1234);
	let b: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x7F4A) ^ 0xABCD);
	let expect: [u16; 32] = core::array::from_fn(|i| shldi_u16_scalar(a[i], b[i], 5));
	assert_eq!(t.shldi_u16x32::<5>(a, b), expect);
}

#[test]
fn shrdi_i64x8_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| (i as i64).wrapping_mul(0x1_0000_0007));
	let b: [i64; 8] = core::array::from_fn(|i| (i as i64).wrapping_mul(-0x3_0000_0001));
	let expect: [i64; 8] = core::array::from_fn(|i| shrdi_i64_scalar(a[i], b[i], 37));
	assert_eq!(t.shrdi_i64x8::<37>(a, b), expect);
}

#[test]
fn shldi_matches_shldv_with_broadcast_c() {
	let Some(t) = Avx512Vbmi2::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| (i as i32).wrapping_mul(0x1F) - 100);
	let b: [i32; 16] = core::array::from_fn(|i| (i as i32).wrapping_mul(-7) + 50);
	let c = [11i32; 16];
	assert_eq!(t.shldi_i32x16::<11>(a, b), t.shldv_i32x16(a, b, c));
	assert_eq!(t.shrdi_i32x16::<11>(a, b), t.shrdv_i32x16(a, b, c));
}

slice_binop_imm_matches_scalar_test!(shldi_i16_slice_matches_scalar, Avx512Vbmi2, shldi_i16_slice, 5, shldi_i16_scalar, i16);
slice_binop_imm_matches_scalar_test!(shldi_u16_slice_matches_scalar, Avx512Vbmi2, shldi_u16_slice, 5, shldi_u16_scalar, u16);
slice_binop_imm_matches_scalar_test!(shrdi_i16_slice_matches_scalar, Avx512Vbmi2, shrdi_i16_slice, 5, shrdi_i16_scalar, i16);
slice_binop_imm_matches_scalar_test!(shrdi_u16_slice_matches_scalar, Avx512Vbmi2, shrdi_u16_slice, 5, shrdi_u16_scalar, u16);
slice_binop_imm_matches_scalar_test!(shldi_i32_slice_matches_scalar, Avx512Vbmi2, shldi_i32_slice, 11, shldi_i32_scalar, i32);
slice_binop_imm_matches_scalar_test!(shldi_u32_slice_matches_scalar, Avx512Vbmi2, shldi_u32_slice, 11, shldi_u32_scalar, u32);
slice_binop_imm_matches_scalar_test!(shrdi_i32_slice_matches_scalar, Avx512Vbmi2, shrdi_i32_slice, 11, shrdi_i32_scalar, i32);
slice_binop_imm_matches_scalar_test!(shrdi_u32_slice_matches_scalar, Avx512Vbmi2, shrdi_u32_slice, 11, shrdi_u32_scalar, u32);
slice_binop_imm_matches_scalar_test!(shldi_i64_slice_matches_scalar, Avx512Vbmi2, shldi_i64_slice, 37, shldi_i64_scalar, i64);
slice_binop_imm_matches_scalar_test!(shldi_u64_slice_matches_scalar, Avx512Vbmi2, shldi_u64_slice, 37, shldi_u64_scalar, u64);
slice_binop_imm_matches_scalar_test!(shrdi_i64_slice_matches_scalar, Avx512Vbmi2, shrdi_i64_slice, 37, shrdi_i64_scalar, i64);
slice_binop_imm_matches_scalar_test!(shrdi_u64_slice_matches_scalar, Avx512Vbmi2, shrdi_u64_slice, 37, shrdi_u64_scalar, u64);

macro_rules! masked_binop_imm_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $imm:expr, $a:expr, $b:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			const IMM8: i32 = $imm;
			let src = $a;
			let a = $a;
			let b = $b;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i], IMM8) } else { src[i] });
			assert_eq!(t.$merge_fn::<IMM8>(src, mask, a, b), merge_expect, "merge");
			let zero_expect = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 { op(a[i], b[i], IMM8) } else { Default::default() }
			});
			assert_eq!(t.$zero_fn::<IMM8>(mask, a, b), zero_expect, "zero");
		}
	};
}

masked_binop_imm_test!(
	shldi_i16x32_masked_matches_scalar, shldi_i16x32_merge_masked, shldi_i16x32_zero_masked, u32, 5,
	core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5A5A_5A5Au32, |a, b, imm| shldi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u16x32_masked_matches_scalar, shldi_u16x32_merge_masked, shldi_u16x32_zero_masked, u32, 5,
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(7)),
	0x5A5A_5A5Au32, |a, b, imm| shldi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i16x32_masked_matches_scalar, shrdi_i16x32_merge_masked, shrdi_i16x32_zero_masked, u32, 5,
	core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(-7)),
	0x5A5A_5A5Au32, |a, b, imm| shrdi_i16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u16x32_masked_matches_scalar, shrdi_u16x32_merge_masked, shrdi_u16x32_zero_masked, u32, 5,
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(7)),
	0x5A5A_5A5Au32, |a, b, imm| shrdi_u16_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i32x16_masked_matches_scalar, shldi_i32x16_merge_masked, shldi_i32x16_zero_masked, u16, 11,
	core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5A5Au16, |a, b, imm| shldi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u32x16_masked_matches_scalar, shldi_u32x16_merge_masked, shldi_u32x16_zero_masked, u16, 11,
	core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(7)),
	0x5A5Au16, |a, b, imm| shldi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i32x16_masked_matches_scalar, shrdi_i32x16_merge_masked, shrdi_i32x16_zero_masked, u16, 11,
	core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(-7)),
	0x5A5Au16, |a, b, imm| shrdi_i32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u32x16_masked_matches_scalar, shrdi_u32x16_merge_masked, shrdi_u32x16_zero_masked, u16, 11,
	core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(7)),
	0x5A5Au16, |a, b, imm| shrdi_u32_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_i64x8_masked_matches_scalar, shldi_i64x8_merge_masked, shldi_i64x8_zero_masked, u8, 37,
	core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shldi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shldi_u64x8_masked_matches_scalar, shldi_u64x8_merge_masked, shldi_u64x8_zero_masked, u8, 37,
	core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shldi_u64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_i64x8_masked_matches_scalar, shrdi_i64x8_merge_masked, shrdi_i64x8_zero_masked, u8, 37,
	core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(-7)),
	0x5Au8, |a, b, imm| shrdi_i64_scalar(a, b, imm)
);
masked_binop_imm_test!(
	shrdi_u64x8_masked_matches_scalar, shrdi_u64x8_merge_masked, shrdi_u64x8_zero_masked, u8, 37,
	core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(7)),
	0x5Au8, |a, b, imm| shrdi_u64_scalar(a, b, imm)
);

macro_rules! masked_ternop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $b:expr, $c:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			let a = $a;
			let b = $b;
			let c = $c;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i], c[i]) } else { a[i] });
			assert_eq!(t.$merge_fn(a, mask, b, c), merge_expect, "merge");
			let zero_expect = core::array::from_fn(|i| {
				if (mask >> i) & 1 == 1 { op(a[i], b[i], c[i]) } else { Default::default() }
			});
			assert_eq!(t.$zero_fn(mask, a, b, c), zero_expect, "zero");
		}
	};
}

masked_ternop_test!(
	shldv_i16x32_masked_matches_scalar, shldv_i16x32_merge_masked, shldv_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(-7)),
	core::array::from_fn::<i16, 32, _>(|i| i as i16), 0x5A5A_5A5Au32, |a, b, c| shldv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u16x32_masked_matches_scalar, shldv_u16x32_merge_masked, shldv_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(7)),
	core::array::from_fn::<u16, 32, _>(|i| i as u16), 0x5A5A_5A5Au32, |a, b, c| shldv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i16x32_masked_matches_scalar, shrdv_i16x32_merge_masked, shrdv_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(0x1F)), core::array::from_fn::<i16, 32, _>(|i| (i as i16).wrapping_mul(-7)),
	core::array::from_fn::<i16, 32, _>(|i| i as i16), 0x5A5A_5A5Au32, |a, b, c| shrdv_i16_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u16x32_masked_matches_scalar, shrdv_u16x32_merge_masked, shrdv_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(0x1F)), core::array::from_fn::<u16, 32, _>(|i| (i as u16).wrapping_mul(7)),
	core::array::from_fn::<u16, 32, _>(|i| i as u16), 0x5A5A_5A5Au32, |a, b, c| shrdv_u16_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i32x16_masked_matches_scalar, shldv_i32x16_merge_masked, shldv_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(-7)),
	core::array::from_fn::<i32, 16, _>(|i| i as i32), 0x5A5Au16, |a, b, c| shldv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u32x16_masked_matches_scalar, shldv_u32x16_merge_masked, shldv_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(7)),
	core::array::from_fn::<u32, 16, _>(|i| i as u32), 0x5A5Au16, |a, b, c| shldv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i32x16_masked_matches_scalar, shrdv_i32x16_merge_masked, shrdv_i32x16_zero_masked, u16,
	core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(0x1F)), core::array::from_fn::<i32, 16, _>(|i| (i as i32).wrapping_mul(-7)),
	core::array::from_fn::<i32, 16, _>(|i| i as i32), 0x5A5Au16, |a, b, c| shrdv_i32_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u32x16_masked_matches_scalar, shrdv_u32x16_merge_masked, shrdv_u32x16_zero_masked, u16,
	core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(0x1F)), core::array::from_fn::<u32, 16, _>(|i| (i as u32).wrapping_mul(7)),
	core::array::from_fn::<u32, 16, _>(|i| i as u32), 0x5A5Au16, |a, b, c| shrdv_u32_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_i64x8_masked_matches_scalar, shldv_i64x8_merge_masked, shldv_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(-7)),
	core::array::from_fn::<i64, 8, _>(|i| i as i64), 0x5Au8, |a, b, c| shldv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shldv_u64x8_masked_matches_scalar, shldv_u64x8_merge_masked, shldv_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(7)),
	core::array::from_fn::<u64, 8, _>(|i| i as u64), 0x5Au8, |a, b, c| shldv_u64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_i64x8_masked_matches_scalar, shrdv_i64x8_merge_masked, shrdv_i64x8_zero_masked, u8,
	core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(0x1F)), core::array::from_fn::<i64, 8, _>(|i| (i as i64).wrapping_mul(-7)),
	core::array::from_fn::<i64, 8, _>(|i| i as i64), 0x5Au8, |a, b, c| shrdv_i64_scalar(a, b, c)
);
masked_ternop_test!(
	shrdv_u64x8_masked_matches_scalar, shrdv_u64x8_merge_masked, shrdv_u64x8_zero_masked, u8,
	core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(0x1F)), core::array::from_fn::<u64, 8, _>(|i| (i as u64).wrapping_mul(7)),
	core::array::from_fn::<u64, 8, _>(|i| i as u64), 0x5Au8, |a, b, c| shrdv_u64_scalar(a, b, c)
);

// compress/expand pack/unpack by output position, not per-lane.
macro_rules! masked_compress_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $width:literal, $Elem:ty, $a:expr, $src:expr, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $mask = $mask_val;
			let mut merge_expect = src;
			let mut zero_expect: [$Elem; $width] = [Default::default(); $width];
			let mut j = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					merge_expect[j] = a[i];
					zero_expect[j] = a[i];
					j += 1;
				}
			}
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

macro_rules! masked_expand_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $width:literal, $Elem:ty, $a:expr, $src:expr, $mask_val:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $mask = $mask_val;
			let mut merge_expect = src;
			let mut zero_expect: [$Elem; $width] = [Default::default(); $width];
			let mut k = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					merge_expect[i] = a[k];
					zero_expect[i] = a[k];
					k += 1;
				}
			}
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

masked_compress_test!(
	compress_i8x64_masked_packs_selected, compress_i8x64_merge_masked, compress_i8x64_zero_masked, u64, 64, i8,
	core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(20)), core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)),
	0x9A37_5C81_2468_ACE0u64
);
masked_compress_test!(
	compress_u8x64_masked_packs_selected, compress_u8x64_merge_masked, compress_u8x64_zero_masked, u64, 64, u8,
	core::array::from_fn(|i| (i as u8).wrapping_mul(3)), core::array::from_fn(|i| (i as u8).wrapping_add(200)),
	0x9A37_5C81_2468_ACE0u64
);
masked_compress_test!(
	compress_i16x32_masked_packs_selected, compress_i16x32_merge_masked, compress_i16x32_zero_masked, u32, 32, i16,
	core::array::from_fn(|i| (i as i16) * 3 - 20), core::array::from_fn(|i| -(i as i16) - 1000), 0x9A37_5C81u32
);
masked_compress_test!(
	compress_u16x32_masked_packs_selected, compress_u16x32_merge_masked, compress_u16x32_zero_masked, u32, 32, u16,
	core::array::from_fn(|i| (i as u16) * 3 + 1), core::array::from_fn(|i| (i as u16) + 9000), 0x9A37_5C81u32
);

masked_expand_test!(
	expand_i8x64_masked_unpacks_selected, expand_i8x64_merge_masked, expand_i8x64_zero_masked, u64, 64, i8,
	core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(20)), core::array::from_fn(|i| (i as i8).wrapping_neg().wrapping_sub(50)),
	0x9A37_5C81_2468_ACE0u64
);
masked_expand_test!(
	expand_u8x64_masked_unpacks_selected, expand_u8x64_merge_masked, expand_u8x64_zero_masked, u64, 64, u8,
	core::array::from_fn(|i| (i as u8).wrapping_mul(3)), core::array::from_fn(|i| (i as u8).wrapping_add(200)),
	0x9A37_5C81_2468_ACE0u64
);
masked_expand_test!(
	expand_i16x32_masked_unpacks_selected, expand_i16x32_merge_masked, expand_i16x32_zero_masked, u32, 32, i16,
	core::array::from_fn(|i| (i as i16) * 3 - 20), core::array::from_fn(|i| -(i as i16) - 1000), 0x9A37_5C81u32
);
masked_expand_test!(
	expand_u16x32_masked_unpacks_selected, expand_u16x32_merge_masked, expand_u16x32_zero_masked, u32, 32, u16,
	core::array::from_fn(|i| (i as u16) * 3 + 1), core::array::from_fn(|i| (i as u16) + 9000), 0x9A37_5C81u32
);

macro_rules! compressstoreu_test {
	($name:ident, $fixed_fn:ident, $Elem:ty, $width:literal, $Mask:ty, $mask_val:expr, $a:expr, $sentinel:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			let a: [$Elem; $width] = $a;
			let mask: $Mask = $mask_val;
			let popcount = mask.count_ones() as usize;
			let mut out = [$sentinel; $width];
			assert_eq!(t.$fixed_fn(&mut out, mask, a), popcount, "written count");
			let mut j = 0usize;
			for i in 0..$width {
				if (mask >> i) & 1 == 1 {
					assert_eq!(out[j], a[i], "packed lane {j}");
					j += 1;
				}
			}
			for (k, o) in out[popcount..].iter().enumerate() {
				assert_eq!(*o, $sentinel, "tail lane {k} must be untouched");
			}
		}
	};
}

// Cross-checked against the register forms above rather than a hand-written
// expectation: same lane selection, only the operand source differs.
macro_rules! expandloadu_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $reg_merge_fn:ident, $reg_zero_fn:ident,
	 $Elem:ty, $width:literal, $Mask:ty, $mask_val:expr, $a:expr, $src:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Vbmi2::detect() else { return };
			let a: [$Elem; $width] = $a;
			let src: [$Elem; $width] = $src;
			let mask: $Mask = $mask_val;
			assert_eq!(t.$merge_fn(src, mask, &a), t.$reg_merge_fn(src, mask, a), "merge");
			assert_eq!(t.$zero_fn(mask, &a), t.$reg_zero_fn(mask, a), "zero");
			let n = mask.count_ones() as usize;
			assert_eq!(t.$merge_fn(src, mask, &a[..n]), t.$reg_merge_fn(src, mask, a), "exact-length mem");
		}
	};
}

compressstoreu_test!(
	compressstoreu_i8x64_packs_selected_lanes, compressstoreu_i8x64, i8, 64, u64,
	0x9A37_5C81_0F2E_47B3u64, core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_add(1)), -1i8
);
compressstoreu_test!(
	compressstoreu_u8x64_packs_selected_lanes, compressstoreu_u8x64, u8, 64, u64,
	0x9A37_5C81_0F2E_47B3u64, core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(1)), u8::MAX
);
compressstoreu_test!(
	compressstoreu_i16x32_packs_selected_lanes, compressstoreu_i16x32, i16, 32, u32,
	0x9A37_5C81u32, core::array::from_fn(|i| (i as i16).wrapping_mul(7).wrapping_add(1)), -1i16
);
compressstoreu_test!(
	compressstoreu_u16x32_packs_selected_lanes, compressstoreu_u16x32, u16, 32, u32,
	0x9A37_5C81u32, core::array::from_fn(|i| (i as u16).wrapping_mul(7).wrapping_add(1)), u16::MAX
);

expandloadu_test!(
	expandloadu_i8x64_matches_register_form,
	expandloadu_i8x64_merge_masked, expandloadu_i8x64_zero_masked,
	expand_i8x64_merge_masked, expand_i8x64_zero_masked,
	i8, 64, u64, 0x9A37_5C81_0F2E_47B3u64, core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_add(1)), core::array::from_fn(|i| (i as i8).wrapping_neg())
);
expandloadu_test!(
	expandloadu_u8x64_matches_register_form,
	expandloadu_u8x64_merge_masked, expandloadu_u8x64_zero_masked,
	expand_u8x64_merge_masked, expand_u8x64_zero_masked,
	u8, 64, u64, 0x9A37_5C81_0F2E_47B3u64, core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(1)), core::array::from_fn(|i| (i as u8).wrapping_add(200))
);
expandloadu_test!(
	expandloadu_i16x32_matches_register_form,
	expandloadu_i16x32_merge_masked, expandloadu_i16x32_zero_masked,
	expand_i16x32_merge_masked, expand_i16x32_zero_masked,
	i16, 32, u32, 0x9A37_5C81u32, core::array::from_fn(|i| (i as i16).wrapping_mul(7).wrapping_add(1)), core::array::from_fn(|i| -(i as i16) - 100)
);
expandloadu_test!(
	expandloadu_u16x32_matches_register_form,
	expandloadu_u16x32_merge_masked, expandloadu_u16x32_zero_masked,
	expand_u16x32_merge_masked, expand_u16x32_zero_masked,
	u16, 32, u32, 0x9A37_5C81u32, core::array::from_fn(|i| (i as u16).wrapping_mul(7).wrapping_add(1)), core::array::from_fn(|i| (i as u16).wrapping_add(1000))
);
