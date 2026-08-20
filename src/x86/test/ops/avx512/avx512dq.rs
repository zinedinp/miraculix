use super::super::super::macros::slice_binop_matches_scalar_test;
use super::*;

/// One-directional, not equality: `from_level` under-detects real hardware
/// outside its bucket (see the identical fix + rationale on
/// `Avx::from_level_agreeing_implies_detect_agrees`, `avx.rs`).
#[test]
fn from_level_agreeing_implies_detect_agrees() {
	let level = GenericLevel::detect(FeatureSet::detect());
	if Avx512Dq::from_level(level).is_some() {
		assert!(Avx512Dq::detect().is_some());
	}
}

#[test]
fn mullo_i64x8_wraps_on_overflow() {
	let Some(t) = Avx512Dq::detect() else { return };
	let mut a = [0i64; 8];
	let mut b = [0i64; 8];
	a[0] = i64::MAX;
	b[0] = 2;
	let mut expect = [0i64; 8];
	expect[0] = i64::MAX.wrapping_mul(2);
	assert_eq!(t.mullo_i64x8(a, b), expect);
}

slice_binop_matches_scalar_test!(
	mullo_i64_slice_matches_scalar, Avx512Dq, mullo_i64_slice, |x: i64, y: i64| x.wrapping_mul(y), i64
);
slice_binop_matches_scalar_test!(
	mullo_u64_slice_matches_scalar, Avx512Dq, mullo_u64_slice, |x: u64, y: u64| x.wrapping_mul(y), u64
);

#[test]
fn i64_to_f64x8_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = [-1000, -1, 0, 1, 2, 42, 1000, 123456];
	assert_eq!(t.i64_to_f64x8(a), a.map(|x| x as f64));
}

#[test]
fn u64_to_f64x8_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [u64; 8] = [0, 1, 2, 42, 1000, 123456, 999999, 5];
	assert_eq!(t.u64_to_f64x8(a), a.map(|x| x as f64));
}

#[test]
fn f64_to_i64x8_matches_round_ties_even() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [-1000.5, -2.5, -0.5, 0.5, 2.5, 3.5, 1000.5, 42.0];
	assert_eq!(t.f64_to_i64x8(a), a.map(|x| x.round_ties_even() as i64));
}

#[test]
fn f64_to_i64x8_trunc_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [-1000.9, -2.5, -0.9, 0.9, 2.5, 3.9, 1000.9, 42.0];
	assert_eq!(t.f64_to_i64x8_trunc(a), a.map(|x| x as i64));
}

#[test]
fn f64_to_u64x8_matches_round_ties_even() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [0.0, 0.5, 2.5, 3.5, 42.0, 1000.5, 123456.5, 5.0];
	assert_eq!(t.f64_to_u64x8(a), a.map(|x| x.round_ties_even() as u64));
}

#[test]
fn f64_to_u64x8_trunc_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [0.0, 0.9, 2.5, 3.9, 42.0, 1000.9, 123456.9, 5.0];
	assert_eq!(t.f64_to_u64x8_trunc(a), a.map(|x| x as u64));
}

#[test]
fn i64_to_f32x8_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = [-1000, -1, 0, 1, 2, 42, 1000, 123456];
	assert_eq!(t.i64_to_f32x8(a), a.map(|x| x as f32));
}

#[test]
fn u64_to_f32x8_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [u64; 8] = [0, 1, 2, 42, 1000, 123456, 999999, 5];
	assert_eq!(t.u64_to_f32x8(a), a.map(|x| x as f32));
}

#[test]
fn f32_to_i64x8_matches_round_ties_even() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [-1000.5, -2.5, -0.5, 0.5, 2.5, 3.5, 1000.5, 42.0];
	assert_eq!(t.f32_to_i64x8(a), a.map(|x| x.round_ties_even() as i64));
}

#[test]
fn f32_to_i64x8_trunc_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [-1000.9, -2.5, -0.9, 0.9, 2.5, 3.9, 1000.9, 42.0];
	assert_eq!(t.f32_to_i64x8_trunc(a), a.map(|x| x as i64));
}

#[test]
fn f32_to_u64x8_matches_round_ties_even() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [0.0, 0.5, 2.5, 3.5, 42.0, 1000.5, 123456.5, 5.0];
	assert_eq!(t.f32_to_u64x8(a), a.map(|x| x.round_ties_even() as u64));
}

#[test]
fn f32_to_u64x8_trunc_matches_as_cast() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [0.0, 0.9, 2.5, 3.9, 42.0, 1000.9, 123456.9, 5.0];
	assert_eq!(t.f32_to_u64x8_trunc(a), a.map(|x| x as u64));
}

#[test]
fn round_cvt_matches_default_rounding_for_exact_values() {
	let Some(t) = Avx512Dq::detect() else { return };
	use core::arch::x86_64::{_MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
	const RN: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	let f64s: [f64; 8] = [-1000.0, -1.0, 0.0, 1.0, 2.0, 42.0, 1000.0, 123456.0];
	let i64s: [i64; 8] = [-1000, -1, 0, 1, 2, 42, 1000, 123456];
	let u64s: [u64; 8] = [0, 1, 2, 42, 1000, 123456, 999999, 5];
	let f32s: [f32; 8] = [-1000.0, -1.0, 0.0, 1.0, 2.0, 42.0, 1000.0, 123456.0];
	assert_eq!(t.f64_to_i64x8_round::<RN>(f64s), t.f64_to_i64x8(f64s));
	assert_eq!(t.f64_to_u64x8_round::<RN>(f64s.map(f64::abs)), t.f64_to_u64x8(f64s.map(f64::abs)));
	assert_eq!(t.i64_to_f64x8_round::<RN>(i64s), t.i64_to_f64x8(i64s));
	assert_eq!(t.u64_to_f64x8_round::<RN>(u64s), t.u64_to_f64x8(u64s));
	assert_eq!(t.f32_to_i64x8_round::<RN>(f32s), t.f32_to_i64x8(f32s));
	assert_eq!(t.f32_to_u64x8_round::<RN>(f32s.map(f32::abs)), t.f32_to_u64x8(f32s.map(f32::abs)));
	assert_eq!(t.i64_to_f32x8_round::<RN>(i64s), t.i64_to_f32x8(i64s));
	assert_eq!(t.u64_to_f32x8_round::<RN>(u64s), t.u64_to_f32x8(u64s));
}

#[test]
fn range_f64x8_min_and_max() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [3.0; 8];
	let b = [7.0; 8];
	assert_eq!(t.range_f64x8::<0>(a, b), [3.0; 8]); // bits[1:0]=00: min
	assert_eq!(t.range_f64x8::<1>(a, b), [7.0; 8]); // bits[1:0]=01: max
}

#[test]
fn range_f32x16_min_and_max() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [3.0; 16];
	let b = [7.0; 16];
	assert_eq!(t.range_f32x16::<0>(a, b), [3.0; 16]);
	assert_eq!(t.range_f32x16::<1>(a, b), [7.0; 16]);
}

#[test]
fn reduce_f64x8_subtracts_truncated_integer_part() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5];
	// IMM8=3: M=0 (no scaling), rounding mode 3 = truncate toward zero.
	assert_eq!(t.reduce_f64x8::<3>(a), [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);
}

#[test]
fn reduce_f32x16_subtracts_truncated_integer_part() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [2.5f32; 16];
	assert_eq!(t.reduce_f32x16::<3>(a), [0.5f32; 16]);
}

#[test]
fn fpclass_f64x8_flags_only_nan_lane() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [f64::NAN, 1.0, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0];
	// IMM8=1: bit0 (QNaN) only.
	assert_eq!(t.fpclass_f64x8::<1>(a), 0b0000_0001);
}

#[test]
fn fpclass_f32x16_flags_only_nan_lane() {
	let Some(t) = Avx512Dq::detect() else { return };
	let mut a = [1.0f32; 16];
	a[3] = f32::NAN;
	assert_eq!(t.fpclass_f32x16::<1>(a), 1 << 3);
}

#[test]
fn fpclass_sd_flags_lane0_nan() {
	let Some(t) = Avx512Dq::detect() else { return };
	assert_eq!(t.fpclass_sd::<1>([f64::NAN, 0.0]), 0b0000_0001);
	assert_eq!(t.fpclass_sd::<1>([1.0, f64::NAN]), 0);
}

#[test]
fn fpclass_ss_flags_lane0_nan() {
	let Some(t) = Avx512Dq::detect() else { return };
	assert_eq!(t.fpclass_ss::<1>([f32::NAN, 0.0, 0.0, 0.0]), 0b0000_0001);
	assert_eq!(t.fpclass_ss::<1>([1.0, f32::NAN, 0.0, 0.0]), 0);
}

#[test]
fn broadcast_f32x2_to_x16_ignores_upper_input_lanes() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [1.0f32, 2.0, 999.0, 999.0];
	let expect: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { 2.0 });
	assert_eq!(t.broadcast_f32x2_to_x16(a), expect);
}

#[test]
fn broadcast_i32x2_to_x16_ignores_upper_input_lanes() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [10i32, 20, 999, 999];
	let expect: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { 10 } else { 20 });
	assert_eq!(t.broadcast_i32x2_to_x16(a), expect);
}

#[test]
fn broadcast_f64x2_to_x8_repeats_pair() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [1.0f64, 2.0];
	assert_eq!(t.broadcast_f64x2_to_x8(a), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn broadcast_i64x2_to_x8_repeats_pair() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [10i64, 20];
	assert_eq!(t.broadcast_i64x2_to_x8(a), [10, 20, 10, 20, 10, 20, 10, 20]);
}

#[test]
fn extract_f32x8_from_x16_picks_selected_half() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	assert_eq!(t.extract_f32x8_from_x16::<0>(a), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
	assert_eq!(t.extract_f32x8_from_x16::<1>(a), [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn extract_i32x8_from_x16_picks_selected_half() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32);
	assert_eq!(t.extract_i32x8_from_x16::<0>(a), [0, 1, 2, 3, 4, 5, 6, 7]);
	assert_eq!(t.extract_i32x8_from_x16::<1>(a), [8, 9, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn extract_f64x2_from_x8_picks_selected_quarter() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64);
	assert_eq!(t.extract_f64x2_from_x8::<1>(a), [2.0, 3.0]);
	assert_eq!(t.extract_f64x2_from_x8::<3>(a), [6.0, 7.0]);
}

#[test]
fn extract_i64x2_from_x8_picks_selected_quarter() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| i as i64);
	assert_eq!(t.extract_i64x2_from_x8::<2>(a), [4, 5]);
}

#[test]
fn insert_f32x8_into_x16_overwrites_selected_half() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [f32; 8] = core::array::from_fn(|i| 100.0 + i as f32);
	let expect: [f32; 16] =
		[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
	assert_eq!(t.insert_f32x8_into_x16::<1>(a, b), expect);
}

#[test]
fn insert_i32x8_into_x16_overwrites_selected_half() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32);
	let b: [i32; 8] = core::array::from_fn(|i| 100 + i as i32);
	let expect: [i32; 16] = [100, 101, 102, 103, 104, 105, 106, 107, 8, 9, 10, 11, 12, 13, 14, 15];
	assert_eq!(t.insert_i32x8_into_x16::<0>(a, b), expect);
}

#[test]
fn insert_f64x2_into_x8_overwrites_selected_quarter() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64);
	let b = [99.0, 98.0];
	assert_eq!(t.insert_f64x2_into_x8::<1>(a, b), [0.0, 1.0, 99.0, 98.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn insert_i64x2_into_x8_overwrites_selected_quarter() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| i as i64);
	let b = [99i64, 98];
	assert_eq!(t.insert_i64x2_into_x8::<3>(a, b), [0, 1, 2, 3, 4, 5, 99, 98]);
}

// Oracle for every masked test below is the already-tested unmasked op,
// not a fresh scalar closure: isolates the one new behavior (lane
// selection), same approach as the FP16/IFMA/VNNI/VBMI masked batches.
// `mask` is `u64` so one helper covers every width/mask type here via
// `as u64` at the call site.
fn assert_merge_zero<T: Copy + PartialEq + core::fmt::Debug + Default, const N: usize>(
	mask: u64, expect: [T; N], merged: [T; N], zeroed: [T; N], src: [T; N],
) {
	for i in 0..N {
		let selected = (mask >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { T::default() }, "zero lane {i}");
	}
}

const MASK8: u8 = 0xA7;
const MASK16: u16 = 0x9A37;

#[test]
fn mullo_i64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let mut a = [3i64; 8];
	let mut b = [5i64; 8];
	a[0] = i64::MAX;
	b[0] = 2;
	let src = [100i64; 8];
	let expect = t.mullo_i64x8(a, b);
	let merged = t.mullo_i64x8_merge_masked(src, MASK8, a, b);
	let zeroed = t.mullo_i64x8_zero_masked(MASK8, a, b);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn mullo_u64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [7u64; 8];
	let b = [11u64; 8];
	let src = [100u64; 8];
	let expect = t.mullo_u64x8(a, b);
	let merged = t.mullo_u64x8_merge_masked(src, MASK8, a, b);
	let zeroed = t.mullo_u64x8_zero_masked(MASK8, a, b);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn i64_to_f64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = [-1000, -1, 0, 1, 2, 42, 1000, 123456];
	let src = [999.0f64; 8];
	let expect = t.i64_to_f64x8(a);
	let merged = t.i64_to_f64x8_merge_masked(src, MASK8, a);
	let zeroed = t.i64_to_f64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn u64_to_f64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [u64; 8] = [0, 1, 2, 42, 1000, 123456, 999999, 5];
	let src = [999.0f64; 8];
	let expect = t.u64_to_f64x8(a);
	let merged = t.u64_to_f64x8_merge_masked(src, MASK8, a);
	let zeroed = t.u64_to_f64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [-1000.5, -2.5, -0.5, 0.5, 2.5, 3.5, 1000.5, 42.0];
	let src = [-1i64; 8];
	let expect = t.f64_to_i64x8(a);
	let merged = t.f64_to_i64x8_merge_masked(src, MASK8, a);
	let zeroed = t.f64_to_i64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_i64x8_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [-1000.9, -2.5, -0.9, 0.9, 2.5, 3.9, 1000.9, 42.0];
	let src = [-1i64; 8];
	let expect = t.f64_to_i64x8_trunc(a);
	let merged = t.f64_to_i64x8_trunc_merge_masked(src, MASK8, a);
	let zeroed = t.f64_to_i64x8_trunc_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [0.0, 0.5, 2.5, 3.5, 42.0, 1000.5, 123456.5, 5.0];
	let src = [u64::MAX; 8];
	let expect = t.f64_to_u64x8(a);
	let merged = t.f64_to_u64x8_merge_masked(src, MASK8, a);
	let zeroed = t.f64_to_u64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f64_to_u64x8_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = [0.0, 0.9, 2.5, 3.9, 42.0, 1000.9, 123456.9, 5.0];
	let src = [u64::MAX; 8];
	let expect = t.f64_to_u64x8_trunc(a);
	let merged = t.f64_to_u64x8_trunc_merge_masked(src, MASK8, a);
	let zeroed = t.f64_to_u64x8_trunc_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn i64_to_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = [-1000, -1, 0, 1, 2, 42, 1000, 123456];
	let src = [999.0f32; 8];
	let expect = t.i64_to_f32x8(a);
	let merged = t.i64_to_f32x8_merge_masked(src, MASK8, a);
	let zeroed = t.i64_to_f32x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn u64_to_f32x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [u64; 8] = [0, 1, 2, 42, 1000, 123456, 999999, 5];
	let src = [999.0f32; 8];
	let expect = t.u64_to_f32x8(a);
	let merged = t.u64_to_f32x8_merge_masked(src, MASK8, a);
	let zeroed = t.u64_to_f32x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_i64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [-1000.5, -2.5, -0.5, 0.5, 2.5, 3.5, 1000.5, 42.0];
	let src = [-1i64; 8];
	let expect = t.f32_to_i64x8(a);
	let merged = t.f32_to_i64x8_merge_masked(src, MASK8, a);
	let zeroed = t.f32_to_i64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_i64x8_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [-1000.9, -2.5, -0.9, 0.9, 2.5, 3.9, 1000.9, 42.0];
	let src = [-1i64; 8];
	let expect = t.f32_to_i64x8_trunc(a);
	let merged = t.f32_to_i64x8_trunc_merge_masked(src, MASK8, a);
	let zeroed = t.f32_to_i64x8_trunc_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [0.0, 0.5, 2.5, 3.5, 42.0, 1000.5, 123456.5, 5.0];
	let src = [u64::MAX; 8];
	let expect = t.f32_to_u64x8(a);
	let merged = t.f32_to_u64x8_merge_masked(src, MASK8, a);
	let zeroed = t.f32_to_u64x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn f32_to_u64x8_trunc_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 8] = [0.0, 0.9, 2.5, 3.9, 42.0, 1000.9, 123456.9, 5.0];
	let src = [u64::MAX; 8];
	let expect = t.f32_to_u64x8_trunc(a);
	let merged = t.f32_to_u64x8_trunc_merge_masked(src, MASK8, a);
	let zeroed = t.f32_to_u64x8_trunc_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [3.0; 8];
	let b = [7.0; 8];
	let src = [999.0; 8];
	let expect = t.range_f64x8::<1>(a, b); // bits[1:0]=01: max
	let merged = t.range_f64x8_merge_masked::<1>(src, MASK8, a, b);
	let zeroed = t.range_f64x8_zero_masked::<1>(MASK8, a, b);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn range_f32x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [3.0; 16];
	let b = [7.0; 16];
	let src = [999.0; 16];
	let expect = t.range_f32x16::<0>(a, b); // bits[1:0]=00: min
	let merged = t.range_f32x16_merge_masked::<0>(src, MASK16, a, b);
	let zeroed = t.range_f32x16_zero_masked::<0>(MASK16, a, b);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f64x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5];
	let src = [999.0; 8];
	let expect = t.reduce_f64x8::<3>(a);
	let merged = t.reduce_f64x8_merge_masked::<3>(src, MASK8, a);
	let zeroed = t.reduce_f64x8_zero_masked::<3>(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn reduce_f32x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [2.5f32; 16];
	let src = [999.0f32; 16];
	let expect = t.reduce_f32x16::<3>(a);
	let merged = t.reduce_f32x16_merge_masked::<3>(src, MASK16, a);
	let zeroed = t.reduce_f32x16_zero_masked::<3>(MASK16, a);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

// `fpclass` gets one gated form, not a merge/zero pair (see the doc
// comment above `fpclass_f64x8_gated`'s definition): the assert is the
// literal ISA semantics, `unmasked(a) & k1`, not just a convenient oracle.
#[test]
fn fpclass_f64x8_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [f64::NAN, 1.0, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0];
	let unmasked = t.fpclass_f64x8::<1>(a); // bit0 QNaN
	assert_eq!(t.fpclass_f64x8_gated::<1>(u8::MAX, a), unmasked);
	assert_eq!(t.fpclass_f64x8_gated::<1>(0x00, a), 0);
	assert_eq!(t.fpclass_f64x8_gated::<1>(MASK8, a), unmasked & MASK8);
}

#[test]
fn fpclass_f32x16_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512Dq::detect() else { return };
	let mut a = [1.0f32; 16];
	a[3] = f32::NAN;
	let unmasked = t.fpclass_f32x16::<1>(a);
	assert_eq!(t.fpclass_f32x16_gated::<1>(u16::MAX, a), unmasked);
	assert_eq!(t.fpclass_f32x16_gated::<1>(MASK16, a), unmasked & MASK16);
}

#[test]
fn fpclass_sd_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [f64::NAN, 0.0];
	let unmasked = t.fpclass_sd::<1>(a);
	assert_eq!(t.fpclass_sd_gated::<1>(0xFF, a), unmasked);
	assert_eq!(t.fpclass_sd_gated::<1>(0x00, a), 0);
}

#[test]
fn fpclass_ss_gated_matches_unmasked_and_k1() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [f32::NAN, 0.0, 0.0, 0.0];
	let unmasked = t.fpclass_ss::<1>(a);
	assert_eq!(t.fpclass_ss_gated::<1>(0xFF, a), unmasked);
	assert_eq!(t.fpclass_ss_gated::<1>(0x00, a), 0);
}

#[test]
fn broadcast_f32x2_to_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [1.0f32, 2.0, 999.0, 999.0];
	let src = [777.0f32; 16];
	let expect = t.broadcast_f32x2_to_x16(a);
	let merged = t.broadcast_f32x2_to_x16_merge_masked(src, MASK16, a);
	let zeroed = t.broadcast_f32x2_to_x16_zero_masked(MASK16, a);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_i32x2_to_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [10i32, 20, 999, 999];
	let src = [777i32; 16];
	let expect = t.broadcast_i32x2_to_x16(a);
	let merged = t.broadcast_i32x2_to_x16_merge_masked(src, MASK16, a);
	let zeroed = t.broadcast_i32x2_to_x16_zero_masked(MASK16, a);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_f64x2_to_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [1.0f64, 2.0];
	let src = [777.0f64; 8];
	let expect = t.broadcast_f64x2_to_x8(a);
	let merged = t.broadcast_f64x2_to_x8_merge_masked(src, MASK8, a);
	let zeroed = t.broadcast_f64x2_to_x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn broadcast_i64x2_to_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a = [10i64, 20];
	let src = [777i64; 8];
	let expect = t.broadcast_i64x2_to_x8(a);
	let merged = t.broadcast_i64x2_to_x8_merge_masked(src, MASK8, a);
	let zeroed = t.broadcast_i64x2_to_x8_zero_masked(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn extract_f32x8_from_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let src = [777.0f32; 8];
	let expect = t.extract_f32x8_from_x16::<1>(a);
	let merged = t.extract_f32x8_from_x16_merge_masked::<1>(src, MASK8, a);
	let zeroed = t.extract_f32x8_from_x16_zero_masked::<1>(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn extract_i32x8_from_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32);
	let src = [777i32; 8];
	let expect = t.extract_i32x8_from_x16::<1>(a);
	let merged = t.extract_i32x8_from_x16_merge_masked::<1>(src, MASK8, a);
	let zeroed = t.extract_i32x8_from_x16_zero_masked::<1>(MASK8, a);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn extract_f64x2_from_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64);
	let src = [777.0f64; 2];
	let expect = t.extract_f64x2_from_x8::<1>(a);
	let merged = t.extract_f64x2_from_x8_merge_masked::<1>(src, 0b01, a);
	let zeroed = t.extract_f64x2_from_x8_zero_masked::<1>(0b01, a);
	assert_merge_zero(0b01u64, expect, merged, zeroed, src);
}

#[test]
fn extract_i64x2_from_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| i as i64);
	let src = [777i64; 2];
	let expect = t.extract_i64x2_from_x8::<2>(a);
	let merged = t.extract_i64x2_from_x8_merge_masked::<2>(src, 0b10, a);
	let zeroed = t.extract_i64x2_from_x8_zero_masked::<2>(0b10, a);
	assert_merge_zero(0b10u64, expect, merged, zeroed, src);
}

#[test]
fn insert_f32x8_into_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let b: [f32; 8] = core::array::from_fn(|i| 100.0 + i as f32);
	let src = [777.0f32; 16];
	let expect = t.insert_f32x8_into_x16::<1>(a, b);
	let merged = t.insert_f32x8_into_x16_merge_masked::<1>(src, MASK16, a, b);
	let zeroed = t.insert_f32x8_into_x16_zero_masked::<1>(MASK16, a, b);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn insert_i32x8_into_x16_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i32; 16] = core::array::from_fn(|i| i as i32);
	let b: [i32; 8] = core::array::from_fn(|i| 100 + i as i32);
	let src = [777i32; 16];
	let expect = t.insert_i32x8_into_x16::<0>(a, b);
	let merged = t.insert_i32x8_into_x16_merge_masked::<0>(src, MASK16, a, b);
	let zeroed = t.insert_i32x8_into_x16_zero_masked::<0>(MASK16, a, b);
	assert_merge_zero(MASK16 as u64, expect, merged, zeroed, src);
}

#[test]
fn insert_f64x2_into_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [f64; 8] = core::array::from_fn(|i| i as f64);
	let b = [99.0, 98.0];
	let src = [777.0f64; 8];
	let expect = t.insert_f64x2_into_x8::<1>(a, b);
	let merged = t.insert_f64x2_into_x8_merge_masked::<1>(src, MASK8, a, b);
	let zeroed = t.insert_f64x2_into_x8_zero_masked::<1>(MASK8, a, b);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}

#[test]
fn insert_i64x2_into_x8_masked_matches_unmasked() {
	let Some(t) = Avx512Dq::detect() else { return };
	let a: [i64; 8] = core::array::from_fn(|i| i as i64);
	let b = [99i64, 98];
	let src = [777i64; 8];
	let expect = t.insert_i64x2_into_x8::<3>(a, b);
	let merged = t.insert_i64x2_into_x8_merge_masked::<3>(src, MASK8, a, b);
	let zeroed = t.insert_i64x2_into_x8_zero_masked::<3>(MASK8, a, b);
	assert_merge_zero(MASK8 as u64, expect, merged, zeroed, src);
}
