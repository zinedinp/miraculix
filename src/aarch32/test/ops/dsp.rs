use super::Dsp;

fn require() -> Option<Dsp> {
	Dsp::detect()
}

#[test]
fn qadd_matches_scalar_saturating_add() {
	let Some(t) = require() else { return };
	for (a, b) in [(0, 0), (i32::MAX, 1), (i32::MIN, -1), (100, -50), (-100, 50)] {
		let expect = (a as i64 + b as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
		assert_eq!(t.qadd(a, b), expect, "qadd({a}, {b})");
	}
}

#[test]
fn qsub_matches_scalar_saturating_sub() {
	let Some(t) = require() else { return };
	for (a, b) in [(0, 0), (i32::MIN, 1), (i32::MAX, -1), (100, -50), (-100, 50)] {
		let expect = (a as i64 - b as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
		assert_eq!(t.qsub(a, b), expect, "qsub({a}, {b})");
	}
}

#[test]
fn qadd8_matches_per_lane_saturating_add() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4]); 3] = [
		([1, 2, 3, 4], [5, 6, 7, 8]),
		([i8::MAX, i8::MIN, 0, -1], [1, -1, i8::MAX, i8::MIN]),
		([-100, 100, 50, -50], [50, -50, 100, -100]),
	];
	for (a, b) in cases {
		let expect = core::array::from_fn(|i| a[i].saturating_add(b[i]));
		assert_eq!(t.qadd8(a, b), expect, "qadd8({a:?}, {b:?})");
	}
}

#[test]
fn qsub8_matches_per_lane_saturating_sub() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4]); 3] = [
		([1, 2, 3, 4], [5, 6, 7, 8]),
		([i8::MIN, i8::MAX, 0, -1], [1, -1, i8::MIN, i8::MAX]),
		([-100, 100, 50, -50], [50, -50, 100, -100]),
	];
	for (a, b) in cases {
		let expect = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
		assert_eq!(t.qsub8(a, b), expect, "qsub8({a:?}, {b:?})");
	}
}

#[test]
fn sadd8_matches_per_lane_wrapping_add() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4]); 3] = [
		([1, 2, 3, 4], [5, 6, 7, 8]),
		([i8::MAX, i8::MIN, 0, -1], [1, -1, i8::MAX, i8::MIN]),
		([-100, 100, 50, -50], [50, -50, 100, -100]),
	];
	for (a, b) in cases {
		let expect = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
		assert_eq!(t.sadd8(a, b), expect, "sadd8({a:?}, {b:?})");
	}
}

#[test]
fn ssub8_matches_per_lane_wrapping_sub() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4]); 3] = [
		([1, 2, 3, 4], [5, 6, 7, 8]),
		([i8::MIN, i8::MAX, 0, -1], [1, -1, i8::MIN, i8::MAX]),
		([-100, 100, 50, -50], [50, -50, 100, -100]),
	];
	for (a, b) in cases {
		let expect = core::array::from_fn(|i| a[i].wrapping_sub(b[i]));
		assert_eq!(t.ssub8(a, b), expect, "ssub8({a:?}, {b:?})");
	}
}

#[test]
fn smulbb_matches_scalar_low_halfword_multiply() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([3, 999], [7, 999]), ([-5, 1], [6, 1]), ([i16::MAX, 0], [i16::MIN, 0])];
	for (a, b) in cases {
		let expect = a[0] as i32 * b[0] as i32;
		assert_eq!(t.smulbb(a, b), expect, "smulbb({a:?}, {b:?})");
	}
}

#[test]
fn smlabb_matches_scalar_low_halfword_multiply_accumulate() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2], i32); 3] =
		[([3, 999], [7, 999], 100), ([-5, 1], [6, 1], -1), ([i16::MAX, 0], [i16::MIN, 0], i32::MAX)];
	for (a, b, c) in cases {
		let expect = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_add(c);
		assert_eq!(t.smlabb(a, b, c), expect, "smlabb({a:?}, {b:?}, {c})");
	}
}

#[test]
fn shadd8_shsub8_match_per_lane_halving_signed() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4]); 3] = [
		([1, 2, 3, 4], [5, 6, 7, 8]),
		([i8::MAX, i8::MIN, 0, -1], [1, -1, i8::MAX, i8::MIN]),
		([-100, 100, 50, -50], [50, -50, 100, -100]),
	];
	for (a, b) in cases {
		let expect_add: [i8; 4] = core::array::from_fn(|i| ((a[i] as i32 + b[i] as i32) >> 1) as i8);
		let expect_sub: [i8; 4] = core::array::from_fn(|i| ((a[i] as i32 - b[i] as i32) >> 1) as i8);
		assert_eq!(t.shadd8(a, b), expect_add, "shadd8({a:?}, {b:?})");
		assert_eq!(t.shsub8(a, b), expect_sub, "shsub8({a:?}, {b:?})");
	}
}

#[test]
fn usub8_matches_per_lane_wrapping_unsigned_sub() {
	let Some(t) = require() else { return };
	let cases: [([u8; 4], [u8; 4]); 3] =
		[([1, 2, 3, 4], [5, 6, 7, 8]), ([255, 0, 128, 1], [1, 255, 128, 0]), ([100, 200, 50, 10], [50, 100, 200, 10])];
	for (a, b) in cases {
		let expect: [u8; 4] = core::array::from_fn(|i| a[i].wrapping_sub(b[i]));
		assert_eq!(t.usub8(a, b), expect, "usub8({a:?}, {b:?})");
	}
}

#[test]
fn usad8_usada8_match_sum_of_absolute_differences() {
	let Some(t) = require() else { return };
	let cases: [([u8; 4], [u8; 4], u32); 3] =
		[([1, 2, 3, 4], [5, 6, 7, 8], 0), ([255, 0, 128, 1], [1, 255, 128, 0], 1000), ([10, 20, 30, 40], [5, 5, 5, 5], 7)];
	for (a, b, c) in cases {
		let sad: u32 = (0..4).map(|i| (a[i] as i32 - b[i] as i32).unsigned_abs()).sum();
		assert_eq!(t.usad8(a, b), sad, "usad8({a:?}, {b:?})");
		assert_eq!(t.usada8(a, b, c), sad + c, "usada8({a:?}, {b:?}, {c})");
	}
}

#[test]
fn sel_after_sadd8_matches_ge_flag_select() {
	let Some(t) = require() else { return };
	let cases: [([i8; 4], [i8; 4], [i8; 4], [i8; 4]); 2] = [
		([1, 2, 3, 4], [5, 6, 7, 8], [10, 20, 30, 40], [-10, -20, -30, -40]),
		([i8::MAX, i8::MIN, 0, -1], [1, -1, i8::MAX, i8::MIN], [1, 2, 3, 4], [5, 6, 7, 8]),
	];
	for (add_a, add_b, sel_a, sel_b) in cases {
		let expect_sum: [i8; 4] = core::array::from_fn(|i| add_a[i].wrapping_add(add_b[i]));
		// ARM ARM pseudocode for SADD8's GE bits: computed from the exact
		// (unwrapped) per-lane signed sum, not the truncated result.
		let ge: [bool; 4] = core::array::from_fn(|i| add_a[i] as i32 + add_b[i] as i32 >= 0);
		let expect_sel: [i8; 4] = core::array::from_fn(|i| if ge[i] { sel_a[i] } else { sel_b[i] });
		assert_eq!(
			t.sel_after_sadd8(add_a, add_b, sel_a, sel_b),
			(expect_sum, expect_sel),
			"sel_after_sadd8({add_a:?}, {add_b:?}, {sel_a:?}, {sel_b:?})"
		);
	}
}

#[test]
fn qadd16_qsub16_match_per_lane_saturating() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([1, 2], [5, 6]), ([i16::MAX, i16::MIN], [1, -1]), ([-100, 100], [50, -50])];
	for (a, b) in cases {
		let expect_add: [i16; 2] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
		let expect_sub: [i16; 2] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
		assert_eq!(t.qadd16(a, b), expect_add, "qadd16({a:?}, {b:?})");
		assert_eq!(t.qsub16(a, b), expect_sub, "qsub16({a:?}, {b:?})");
	}
}

#[test]
fn qasx_qsax_match_cross_lane_saturating() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([1, 2], [5, 6]), ([i16::MAX, i16::MIN], [1, -1]), ([-100, 100], [50, -50])];
	for (a, b) in cases {
		let expect_asx = [a[0].saturating_sub(b[1]), a[1].saturating_add(b[0])];
		let expect_sax = [a[0].saturating_add(b[1]), a[1].saturating_sub(b[0])];
		assert_eq!(t.qasx(a, b), expect_asx, "qasx({a:?}, {b:?})");
		assert_eq!(t.qsax(a, b), expect_sax, "qsax({a:?}, {b:?})");
	}
}

#[test]
fn sadd16_matches_per_lane_wrapping_add() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([1, 2], [5, 6]), ([i16::MAX, i16::MIN], [1, -1]), ([-100, 100], [50, -50])];
	for (a, b) in cases {
		let expect: [i16; 2] = core::array::from_fn(|i| a[i].wrapping_add(b[i]));
		assert_eq!(t.sadd16(a, b), expect, "sadd16({a:?}, {b:?})");
	}
}

#[test]
fn sasx_matches_cross_lane_wrapping() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([1, 2], [5, 6]), ([i16::MAX, i16::MIN], [1, -1]), ([-100, 100], [50, -50])];
	for (a, b) in cases {
		let expect = [a[0].wrapping_sub(b[1]), a[1].wrapping_add(b[0])];
		assert_eq!(t.sasx(a, b), expect, "sasx({a:?}, {b:?})");
	}
}

#[test]
fn shadd16_shsub16_match_per_lane_halving_signed() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([1, 2], [5, 6]), ([i16::MAX, i16::MIN], [1, -1]), ([-100, 100], [50, -50])];
	for (a, b) in cases {
		let expect_add: [i16; 2] = core::array::from_fn(|i| ((a[i] as i32 + b[i] as i32) >> 1) as i16);
		let expect_sub: [i16; 2] = core::array::from_fn(|i| ((a[i] as i32 - b[i] as i32) >> 1) as i16);
		assert_eq!(t.shadd16(a, b), expect_add, "shadd16({a:?}, {b:?})");
		assert_eq!(t.shsub16(a, b), expect_sub, "shsub16({a:?}, {b:?})");
	}
}

#[test]
fn smul_family_matches_scalar_halfword_products() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([3, 999], [7, 999]), ([-5, 1], [6, 1]), ([i16::MAX, i16::MIN], [i16::MIN, i16::MAX])];
	for (a, b) in cases {
		assert_eq!(t.smulbb(a, b), a[0] as i32 * b[0] as i32, "smulbb({a:?}, {b:?})");
		assert_eq!(t.smultb(a, b), a[1] as i32 * b[0] as i32, "smultb({a:?}, {b:?})");
		assert_eq!(t.smulbt(a, b), a[0] as i32 * b[1] as i32, "smulbt({a:?}, {b:?})");
		assert_eq!(t.smultt(a, b), a[1] as i32 * b[1] as i32, "smultt({a:?}, {b:?})");
	}
}

#[test]
fn smulw_family_matches_scalar_wide_by_halfword_product() {
	let Some(t) = require() else { return };
	let cases: [(i32, [i16; 2]); 3] = [(30, [10, 20]), (-1000, [5, -5]), (i32::MAX, [1, -1])];
	for (a, b) in cases {
		let expect_wb = ((a as i64 * b[0] as i64) >> 16) as i32;
		let expect_wt = ((a as i64 * b[1] as i64) >> 16) as i32;
		assert_eq!(t.smulwb(a, b), expect_wb, "smulwb({a}, {b:?})");
		assert_eq!(t.smulwt(a, b), expect_wt, "smulwt({a}, {b:?})");
	}
}

#[test]
fn smla_family_matches_scalar_halfword_products_plus_accumulator() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2], i32); 3] =
		[([3, 999], [7, 999], 100), ([-5, 1], [6, 1], -1), ([i16::MAX, i16::MIN], [i16::MIN, i16::MAX], 5)];
	for (a, b, c) in cases {
		let bb = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_add(c);
		let bt = (a[0] as i32).wrapping_mul(b[1] as i32).wrapping_add(c);
		let tb = (a[1] as i32).wrapping_mul(b[0] as i32).wrapping_add(c);
		let tt = (a[1] as i32).wrapping_mul(b[1] as i32).wrapping_add(c);
		assert_eq!(t.smlabb(a, b, c), bb, "smlabb({a:?}, {b:?}, {c})");
		assert_eq!(t.smlabt(a, b, c), bt, "smlabt({a:?}, {b:?}, {c})");
		assert_eq!(t.smlatb(a, b, c), tb, "smlatb({a:?}, {b:?}, {c})");
		assert_eq!(t.smlatt(a, b, c), tt, "smlatt({a:?}, {b:?}, {c})");
	}
}

#[test]
fn smlaw_family_matches_scalar_wide_by_halfword_product_plus_accumulator() {
	let Some(t) = require() else { return };
	let cases: [(i32, [i16; 2], i32); 3] = [(30, [10, 20], 100), (-1000, [5, -5], -1), (i32::MAX, [1, -1], 5)];
	for (a, b, c) in cases {
		let expect_wb = (((a as i64 * b[0] as i64) + ((c as i64) << 16)) >> 16) as i32;
		let expect_wt = (((a as i64 * b[1] as i64) + ((c as i64) << 16)) >> 16) as i32;
		assert_eq!(t.smlawb(a, b, c), expect_wb, "smlawb({a}, {b:?}, {c})");
		assert_eq!(t.smlawt(a, b, c), expect_wt, "smlawt({a}, {b:?}, {c})");
	}
}

#[test]
fn smuad_family_matches_scalar_dual_multiply() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2]); 3] =
		[([3, 999], [7, 999]), ([-5, 1], [6, 1]), ([i16::MAX, i16::MIN], [i16::MIN, i16::MAX])];
	for (a, b) in cases {
		let ad = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_add((a[1] as i32).wrapping_mul(b[1] as i32));
		let adx = (a[0] as i32).wrapping_mul(b[1] as i32).wrapping_add((a[1] as i32).wrapping_mul(b[0] as i32));
		let sd = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_sub((a[1] as i32).wrapping_mul(b[1] as i32));
		let sdx = (a[0] as i32).wrapping_mul(b[1] as i32).wrapping_sub((a[1] as i32).wrapping_mul(b[0] as i32));
		assert_eq!(t.smuad(a, b), ad, "smuad({a:?}, {b:?})");
		assert_eq!(t.smuadx(a, b), adx, "smuadx({a:?}, {b:?})");
		assert_eq!(t.smusd(a, b), sd, "smusd({a:?}, {b:?})");
		assert_eq!(t.smusdx(a, b), sdx, "smusdx({a:?}, {b:?})");
	}
}

#[test]
fn smlad_smlsd_match_scalar_dual_multiply_plus_accumulator() {
	let Some(t) = require() else { return };
	let cases: [([i16; 2], [i16; 2], i32); 3] =
		[([3, 999], [7, 999], 100), ([-5, 1], [6, 1], -1), ([i16::MAX, i16::MIN], [i16::MIN, i16::MAX], 5)];
	for (a, b, c) in cases {
		let ad = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_add((a[1] as i32).wrapping_mul(b[1] as i32));
		let sd = (a[0] as i32).wrapping_mul(b[0] as i32).wrapping_sub((a[1] as i32).wrapping_mul(b[1] as i32));
		assert_eq!(t.smlad(a, b, c), ad.wrapping_add(c), "smlad({a:?}, {b:?}, {c})");
		assert_eq!(t.smlsd(a, b, c), sd.wrapping_add(c), "smlsd({a:?}, {b:?}, {c})");
	}
}

#[test]
fn qdbl_matches_scalar_saturating_double() {
	let Some(t) = require() else { return };
	for a in [0i32, 1, -1, 1000, -1000, i32::MAX, i32::MIN, i32::MAX / 2 + 1, i32::MIN / 2 - 1] {
		let expect = (a as i64 * 2).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
		assert_eq!(t.qdbl(a), expect, "qdbl({a})");
	}
}
