use super::super::super::macros::{slice_binop_matches_scalar_test, slice_shift_imm_matches_scalar_test};
use super::*;

/// One-directional, not equality: `from_level` under-detects real hardware
/// outside its bucket (see the identical fix + rationale on
/// `Avx::from_level_agreeing_implies_detect_agrees`, `avx.rs`).
#[test]
fn from_level_agreeing_implies_detect_agrees() {
	let level = GenericLevel::detect(FeatureSet::detect());
	if Avx512Bw::from_level(level).is_some() {
		assert!(Avx512Bw::detect().is_some());
	}
}

#[test]
fn add_i8x64_wraps_on_overflow() {
	let Some(t) = Avx512Bw::detect() else { return };
	let mut a = [0i8; 64];
	let mut b = [0i8; 64];
	a[0] = i8::MAX;
	b[0] = 1;
	let mut expect = [0i8; 64];
	expect[0] = i8::MIN;
	assert_eq!(t.add_i8x64(a, b), expect);
}

#[test]
fn adds_i8x64_saturates_on_overflow() {
	let Some(t) = Avx512Bw::detect() else { return };
	let mut a = [0i8; 64];
	let mut b = [0i8; 64];
	a[0] = i8::MAX;
	b[0] = 1;
	let mut expect = [0i8; 64];
	expect[0] = i8::MAX;
	assert_eq!(t.adds_i8x64(a, b), expect);
}

#[test]
fn mul_i16x32_wraps_on_overflow() {
	let Some(t) = Avx512Bw::detect() else { return };
	let mut a = [0i16; 32];
	let mut b = [0i16; 32];
	a[0] = i16::MAX;
	b[0] = 4;
	let expect: [i16; 32] = core::array::from_fn(|i| if i == 0 { i16::MAX.wrapping_mul(4) } else { 0 });
	assert_eq!(t.mul_i16x32(a, b), expect);
}

#[test]
fn avg_u8x64_rounds_up() {
	let Some(t) = Avx512Bw::detect() else { return };
	let mut a = [0u8; 64];
	let mut b = [0u8; 64];
	a[0] = 3;
	b[0] = 4;
	let mut expect = [0u8; 64];
	expect[0] = 4; // (3+4+1)/2 = 4
	assert_eq!(t.avg_u8x64(a, b), expect);
}

slice_binop_matches_scalar_test!(
	cmpgt_i8_slice_matches_scalar, Avx512Bw, cmpgt_i8_slice,
	|x, y| if x > y { -1 } else { 0 }, i8
);
slice_binop_matches_scalar_test!(
	cmple_u8_slice_matches_scalar, Avx512Bw, cmple_u8_slice,
	|x, y| if x <= y { !0 } else { 0 }, u8
);
slice_binop_matches_scalar_test!(
	cmpgt_i16_slice_matches_scalar, Avx512Bw, cmpgt_i16_slice,
	|x, y| if x > y { -1 } else { 0 }, i16
);
slice_binop_matches_scalar_test!(
	cmpgt_u16_slice_matches_scalar, Avx512Bw, cmpgt_u16_slice,
	|x, y| if x > y { !0 } else { 0 }, u16
);

#[test]
fn select_i8x64_picks_b_where_mask_set() {
	let Some(t) = Avx512Bw::detect() else { return };
	let a: [i8; 64] = core::array::from_fn(|i| i as i8);
	let b: [i8; 64] = core::array::from_fn(|i| 100 - i as i8);
	let mask: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(t.select_i8x64(a, b, mask), expect);
}

#[test]
fn select_i16x32_picks_b_where_mask_set() {
	let Some(t) = Avx512Bw::detect() else { return };
	let a: [i16; 32] = core::array::from_fn(|i| i as i16);
	let b: [i16; 32] = core::array::from_fn(|i| 100 - i as i16);
	let mask: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1 } else { 0 });
	let expect: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { b[i] } else { a[i] });
	assert_eq!(t.select_i16x32(a, b, mask), expect);
}

// select_i8/u8/i16/u16: no shared slice_ternop test (out-of-domain mask, same reason as select_i32).

slice_binop_matches_scalar_test!(add_i8_slice_matches_scalar, Avx512Bw, add_i8_slice, |x: i8, y: i8| x.wrapping_add(y), i8);
slice_binop_matches_scalar_test!(sub_i8_slice_matches_scalar, Avx512Bw, sub_i8_slice, |x: i8, y: i8| x.wrapping_sub(y), i8);
slice_binop_matches_scalar_test!(min_i8_slice_matches_scalar, Avx512Bw, min_i8_slice, |x, y| x.min(y), i8);
slice_binop_matches_scalar_test!(max_i8_slice_matches_scalar, Avx512Bw, max_i8_slice, |x, y| x.max(y), i8);
slice_binop_matches_scalar_test!(
	cmpeq_i8_slice_matches_scalar, Avx512Bw, cmpeq_i8_slice,
	|x, y| if x == y { -1 } else { 0 }, i8
);

slice_binop_matches_scalar_test!(add_u8_slice_matches_scalar, Avx512Bw, add_u8_slice, |x: u8, y: u8| x.wrapping_add(y), u8);
slice_binop_matches_scalar_test!(min_u8_slice_matches_scalar, Avx512Bw, min_u8_slice, |x, y| x.min(y), u8);
slice_binop_matches_scalar_test!(max_u8_slice_matches_scalar, Avx512Bw, max_u8_slice, |x, y| x.max(y), u8);
slice_binop_matches_scalar_test!(
	cmpeq_u8_slice_matches_scalar, Avx512Bw, cmpeq_u8_slice,
	|x, y| if x == y { !0 } else { 0 }, u8
);

slice_binop_matches_scalar_test!(add_i16_slice_matches_scalar, Avx512Bw, add_i16_slice, |x: i16, y: i16| x.wrapping_add(y), i16);
slice_binop_matches_scalar_test!(sub_i16_slice_matches_scalar, Avx512Bw, sub_i16_slice, |x: i16, y: i16| x.wrapping_sub(y), i16);
slice_binop_matches_scalar_test!(mul_i16_slice_matches_scalar, Avx512Bw, mul_i16_slice, |x: i16, y: i16| x.wrapping_mul(y), i16);
slice_binop_matches_scalar_test!(min_i16_slice_matches_scalar, Avx512Bw, min_i16_slice, |x, y| x.min(y), i16);
slice_binop_matches_scalar_test!(max_i16_slice_matches_scalar, Avx512Bw, max_i16_slice, |x, y| x.max(y), i16);
slice_binop_matches_scalar_test!(
	cmpeq_i16_slice_matches_scalar, Avx512Bw, cmpeq_i16_slice,
	|x, y| if x == y { -1 } else { 0 }, i16
);
slice_shift_imm_matches_scalar_test!(
	shl_i16_slice_matches_scalar, Avx512Bw, shl_i16_slice, 3,
	|x: i16, imm| x.wrapping_shl(imm), i16
);
slice_shift_imm_matches_scalar_test!(
	sra_i16_slice_matches_scalar, Avx512Bw, sra_i16_slice, 1,
	|x: i16, imm| x.wrapping_shr(imm), i16
);

slice_binop_matches_scalar_test!(add_u16_slice_matches_scalar, Avx512Bw, add_u16_slice, |x: u16, y: u16| x.wrapping_add(y), u16);
slice_binop_matches_scalar_test!(min_u16_slice_matches_scalar, Avx512Bw, min_u16_slice, |x, y| x.min(y), u16);
slice_binop_matches_scalar_test!(max_u16_slice_matches_scalar, Avx512Bw, max_u16_slice, |x, y| x.max(y), u16);
slice_binop_matches_scalar_test!(
	cmpeq_u16_slice_matches_scalar, Avx512Bw, cmpeq_u16_slice,
	|x, y| if x == y { !0 } else { 0 }, u16
);
slice_shift_imm_matches_scalar_test!(
	shl_u16_slice_matches_scalar, Avx512Bw, shl_u16_slice, 3,
	|x: u16, imm| x.wrapping_shl(imm), u16
);
slice_shift_imm_matches_scalar_test!(
	shr_u16_slice_matches_scalar, Avx512Bw, shr_u16_slice, 1,
	|x: u16, imm| x.wrapping_shr(imm), u16
);

macro_rules! masked_binop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $b:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Bw::detect() else { return };
			let a = $a;
			let b = $b;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a, b), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i], b[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a, b), zero_expect, "zero");
		}
	};
}

macro_rules! masked_unop_test {
	($name:ident, $merge_fn:ident, $zero_fn:ident, $mask:ty, $a:expr, $src:expr, $mask_val:expr, $op:expr) => {
		#[test]
		fn $name() {
			let Some(t) = Avx512Bw::detect() else { return };
			let a = $a;
			let src = $src;
			let mask: $mask = $mask_val;
			let op = $op;
			let merge_expect = core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { src[i] });
			assert_eq!(t.$merge_fn(src, mask, a), merge_expect, "merge");
			let zero_expect =
				core::array::from_fn(|i| if (mask >> i) & 1 == 1 { op(a[i]) } else { Default::default() });
			assert_eq!(t.$zero_fn(mask, a), zero_expect, "zero");
		}
	};
}

const MASK64: u64 = 0x5555_5555_5555_5555u64;
const MASK32: u32 = 0x5555_5555u32;

masked_binop_test!(
	add_i8x64_masked_matches_scalar, add_i8x64_merge_masked, add_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|_| 3i8),
	core::array::from_fn::<i8, 64, _>(|i| -(i as i8)), MASK64, |x: i8, y: i8| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i8x64_masked_matches_scalar, sub_i8x64_merge_masked, sub_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|_| 3i8),
	core::array::from_fn::<i8, 64, _>(|i| -(i as i8)), MASK64, |x: i8, y: i8| x.wrapping_sub(y)
);
masked_binop_test!(
	adds_i8x64_masked_matches_scalar, adds_i8x64_merge_masked, adds_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|_| 100i8),
	core::array::from_fn::<i8, 64, _>(|i| -(i as i8)), MASK64, |x: i8, y: i8| x.saturating_add(y)
);
masked_binop_test!(
	subs_i8x64_masked_matches_scalar, subs_i8x64_merge_masked, subs_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|_| 100i8),
	core::array::from_fn::<i8, 64, _>(|i| -(i as i8)), MASK64, |x: i8, y: i8| x.saturating_sub(y)
);
masked_binop_test!(
	min_i8x64_masked_matches_scalar, min_i8x64_merge_masked, min_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|i| 32i8.wrapping_sub(i as i8)),
	core::array::from_fn::<i8, 64, _>(|i| (i as i8).wrapping_neg().wrapping_sub(50)), MASK64, |x: i8, y: i8| x.min(y)
);
masked_binop_test!(
	max_i8x64_masked_matches_scalar, max_i8x64_merge_masked, max_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| i as i8), core::array::from_fn::<i8, 64, _>(|i| 32i8.wrapping_sub(i as i8)),
	core::array::from_fn::<i8, 64, _>(|i| (i as i8).wrapping_neg().wrapping_sub(50)), MASK64, |x: i8, y: i8| x.max(y)
);

masked_binop_test!(
	add_u8x64_masked_matches_scalar, add_u8x64_merge_masked, add_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|_| 3u8),
	core::array::from_fn::<u8, 64, _>(|i| i as u8 + 100), MASK64, |x: u8, y: u8| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u8x64_masked_matches_scalar, sub_u8x64_merge_masked, sub_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8 + 100), core::array::from_fn::<u8, 64, _>(|_| 3u8),
	core::array::from_fn::<u8, 64, _>(|i| i as u8), MASK64, |x: u8, y: u8| x.wrapping_sub(y)
);
masked_binop_test!(
	adds_u8x64_masked_matches_scalar, adds_u8x64_merge_masked, adds_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|_| 200u8),
	core::array::from_fn::<u8, 64, _>(|i| i as u8), MASK64, |x: u8, y: u8| x.saturating_add(y)
);
masked_binop_test!(
	subs_u8x64_masked_matches_scalar, subs_u8x64_merge_masked, subs_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|_| 200u8),
	core::array::from_fn::<u8, 64, _>(|i| i as u8), MASK64, |x: u8, y: u8| x.saturating_sub(y)
);
masked_binop_test!(
	min_u8x64_masked_matches_scalar, min_u8x64_merge_masked, min_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|i| 32u8.wrapping_sub(i as u8)),
	core::array::from_fn::<u8, 64, _>(|i| (i as u8).wrapping_add(100)), MASK64, |x: u8, y: u8| x.min(y)
);
masked_binop_test!(
	max_u8x64_masked_matches_scalar, max_u8x64_merge_masked, max_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|i| 32u8.wrapping_sub(i as u8)),
	core::array::from_fn::<u8, 64, _>(|i| (i as u8).wrapping_add(100)), MASK64, |x: u8, y: u8| x.max(y)
);

masked_binop_test!(
	add_i16x32_masked_matches_scalar, add_i16x32_merge_masked, add_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|_| 3i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16)), MASK32, |x: i16, y: i16| x.wrapping_add(y)
);
masked_binop_test!(
	sub_i16x32_masked_matches_scalar, sub_i16x32_merge_masked, sub_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|_| 3i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16)), MASK32, |x: i16, y: i16| x.wrapping_sub(y)
);
masked_binop_test!(
	adds_i16x32_masked_matches_scalar, adds_i16x32_merge_masked, adds_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|_| 30000i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16)), MASK32, |x: i16, y: i16| x.saturating_add(y)
);
masked_binop_test!(
	subs_i16x32_masked_matches_scalar, subs_i16x32_merge_masked, subs_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|_| 30000i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16)), MASK32, |x: i16, y: i16| x.saturating_sub(y)
);
masked_binop_test!(
	mul_i16x32_masked_matches_scalar, mul_i16x32_merge_masked, mul_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16 + 1), core::array::from_fn::<i16, 32, _>(|_| 3i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16) - 1000), MASK32, |x: i16, y: i16| x.wrapping_mul(y)
);
masked_binop_test!(
	min_i16x32_masked_matches_scalar, min_i16x32_merge_masked, min_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|i| 16 - i as i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16) - 1000), MASK32, |x: i16, y: i16| x.min(y)
);
masked_binop_test!(
	max_i16x32_masked_matches_scalar, max_i16x32_merge_masked, max_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| i as i16), core::array::from_fn::<i16, 32, _>(|i| 16 - i as i16),
	core::array::from_fn::<i16, 32, _>(|i| -(i as i16) - 1000), MASK32, |x: i16, y: i16| x.max(y)
);

masked_binop_test!(
	add_u16x32_masked_matches_scalar, add_u16x32_merge_masked, add_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16), core::array::from_fn::<u16, 32, _>(|_| 3u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), MASK32, |x: u16, y: u16| x.wrapping_add(y)
);
masked_binop_test!(
	sub_u16x32_masked_matches_scalar, sub_u16x32_merge_masked, sub_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), core::array::from_fn::<u16, 32, _>(|_| 3u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16), MASK32, |x: u16, y: u16| x.wrapping_sub(y)
);
masked_binop_test!(
	adds_u16x32_masked_matches_scalar, adds_u16x32_merge_masked, adds_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16), core::array::from_fn::<u16, 32, _>(|_| 60000u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16), MASK32, |x: u16, y: u16| x.saturating_add(y)
);
masked_binop_test!(
	subs_u16x32_masked_matches_scalar, subs_u16x32_merge_masked, subs_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16), core::array::from_fn::<u16, 32, _>(|_| 60000u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16), MASK32, |x: u16, y: u16| x.saturating_sub(y)
);
masked_binop_test!(
	mul_u16x32_masked_matches_scalar, mul_u16x32_merge_masked, mul_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1), core::array::from_fn::<u16, 32, _>(|_| 3u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), MASK32, |x: u16, y: u16| x.wrapping_mul(y)
);
masked_binop_test!(
	min_u16x32_masked_matches_scalar, min_u16x32_merge_masked, min_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16), core::array::from_fn::<u16, 32, _>(|i| 16u16.wrapping_sub(i as u16)),
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), MASK32, |x: u16, y: u16| x.min(y)
);
masked_binop_test!(
	max_u16x32_masked_matches_scalar, max_u16x32_merge_masked, max_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16), core::array::from_fn::<u16, 32, _>(|i| 16u16.wrapping_sub(i as u16)),
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), MASK32, |x: u16, y: u16| x.max(y)
);
masked_binop_test!(
	avg_u8x64_masked_matches_scalar, avg_u8x64_merge_masked, avg_u8x64_zero_masked, u64,
	core::array::from_fn::<u8, 64, _>(|i| i as u8), core::array::from_fn::<u8, 64, _>(|i| 255u8 - i as u8),
	core::array::from_fn::<u8, 64, _>(|i| i as u8 + 100), MASK64,
	|x: u8, y: u8| ((x as u16) + (y as u16)).div_ceil(2) as u8
);
masked_binop_test!(
	avg_u16x32_masked_matches_scalar, avg_u16x32_merge_masked, avg_u16x32_zero_masked, u32,
	core::array::from_fn::<u16, 32, _>(|i| i as u16 * 3), core::array::from_fn::<u16, 32, _>(|i| 65535u16 - i as u16),
	core::array::from_fn::<u16, 32, _>(|i| i as u16 + 1000), MASK32,
	|x: u16, y: u16| ((x as u32) + (y as u32)).div_ceil(2) as u16
);

masked_unop_test!(
	abs_i8x64_masked_matches_scalar, abs_i8x64_merge_masked, abs_i8x64_zero_masked, u64,
	core::array::from_fn::<i8, 64, _>(|i| (i as i8).wrapping_sub(32).wrapping_mul(2)),
	core::array::from_fn::<i8, 64, _>(|i| (i as i8).wrapping_neg().wrapping_sub(50)),
	MASK64, |x: i8| x.wrapping_abs()
);
masked_unop_test!(
	abs_i16x32_masked_matches_scalar, abs_i16x32_merge_masked, abs_i16x32_zero_masked, u32,
	core::array::from_fn::<i16, 32, _>(|i| (i as i16 - 16) * 3), core::array::from_fn::<i16, 32, _>(|i| -(i as i16) - 1000),
	MASK32, |x: i16| x.wrapping_abs()
);

#[test]
fn bslli_u8x64_shifts_each_128_bit_lane_independently() {
	let Some(t) = Avx512Bw::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| (i + 1) as u8);
	let got = t.bslli_u8x64::<3>(a);
	let mut expect = [0u8; 64];
	for lane in 0..4 {
		let base = lane * 16;
		expect[base + 3..base + 16].copy_from_slice(&a[base..base + 13]);
	}
	assert_eq!(got, expect);
	assert_eq!(t.bslli_u8x64::<0>(a), a);
	assert_eq!(t.bslli_u8x64::<16>(a), [0u8; 64]);
}

#[test]
fn broadcast_u8x64_replicates_byte_across_all_lanes() {
	let Some(t) = Avx512Bw::detect() else { return };
	assert_eq!(t.broadcast_u8x64(0x7A), [0x7Au8; 64]);
	assert_eq!(t.broadcast_u8x64(0), [0u8; 64]);
}
