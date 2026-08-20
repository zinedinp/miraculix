use super::*;

/// Every real target of this crate (Intel, or AMD since 2011) lacks 3DNow!;
/// this asserts `detect` fails closed rather than false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(ThreeDNow::detect().is_none(), "host has real 3DNow!, review this test");
}

#[test]
fn add_f32x2_sums_lanes() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.add_f32x2([1.0, 2.0], [10.0, 20.0]), [11.0, 22.0]);
}

#[test]
fn sub_f32x2_subtracts_lanes() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.sub_f32x2([10.0, 20.0], [1.0, 2.0]), [9.0, 18.0]);
}

#[test]
fn subr_f32x2_reverse_subtracts_lanes() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.subr_f32x2([1.0, 2.0], [10.0, 20.0]), [9.0, 18.0]);
}

#[test]
fn mul_f32x2_multiplies_lanes() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.mul_f32x2([2.0, 3.0], [10.0, 20.0]), [20.0, 60.0]);
}

#[test]
fn min_max_f32x2_pick_extremes() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.min_f32x2([1.0, -1.0], [-2.0, 2.0]), [-2.0, -1.0]);
	assert_eq!(td.max_f32x2([1.0, -1.0], [-2.0, 2.0]), [1.0, 2.0]);
}

#[test]
fn cmpeq_cmpgt_cmpge_f32x2_match_scalar_comparisons() {
	let Some(td) = ThreeDNow::detect() else { return };
	let a = [1.0f32, 2.0];
	let b = [1.0f32, 1.0];
	let expect_eq: [u32; 2] = core::array::from_fn(|i| if a[i] == b[i] { !0 } else { 0 });
	let expect_gt: [u32; 2] = core::array::from_fn(|i| if a[i] > b[i] { !0 } else { 0 });
	let expect_ge: [u32; 2] = core::array::from_fn(|i| if a[i] >= b[i] { !0 } else { 0 });
	let to_bits = |v: [f32; 2]| -> [u32; 2] { core::array::from_fn(|i| v[i].to_bits()) };
	assert_eq!(to_bits(td.cmpeq_f32x2(a, b)), expect_eq);
	assert_eq!(to_bits(td.cmpgt_f32x2(a, b)), expect_gt);
	assert_eq!(to_bits(td.cmpge_f32x2(a, b)), expect_ge);
}

#[test]
fn pfacc_f32x2_sums_pair_from_each_input() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.pfacc_f32x2([1.0, 2.0], [10.0, 20.0]), [1.0 + 2.0, 10.0 + 20.0]);
}

#[test]
fn to_i32x2_truncates_toward_zero() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.to_i32x2([1.9, -1.9]), [1, -1]);
}

#[test]
fn from_i32x2_converts_to_float() {
	let Some(td) = ThreeDNow::detect() else { return };
	assert_eq!(td.from_i32x2([1, -1]), [1.0, -1.0]);
}

#[test]
fn to_i32x2_then_from_i32x2_roundtrips_for_integral_values() {
	let Some(td) = ThreeDNow::detect() else { return };
	let a = [42.0f32, -17.0];
	assert_eq!(td.from_i32x2(td.to_i32x2(a)), a);
}
