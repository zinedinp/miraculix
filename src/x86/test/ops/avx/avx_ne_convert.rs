use super::*;

const BF16_ONE: u16 = 0b0011_1111_1000_0000;
const BF16_TWO: u16 = 0b0100_0000_0000_0000;
const BF16_THREE: u16 = 0b0100_0000_0100_0000;
const BF16_FOUR: u16 = 0b0100_0000_1000_0000;
const BF16_FIVE: u16 = 0b0100_0000_1010_0000;
const BF16_SIX: u16 = 0b0100_0000_1100_0000;
const BF16_SEVEN: u16 = 0b0100_0000_1110_0000;
const BF16_EIGHT: u16 = 0b0100_0001_0000_0000;

const F16_ONE: u16 = 0x3c00;
const F16_TWO: u16 = 0x4000;
const F16_THREE: u16 = 0x4200;
const F16_FOUR: u16 = 0x4400;
const F16_FIVE: u16 = 0x4500;
const F16_SIX: u16 = 0x4600;
const F16_SEVEN: u16 = 0x4700;
const F16_EIGHT: u16 = 0x4800;

#[test]
fn cvtneebf16_ps_x4_takes_even_indexed_elements() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT];
	assert_eq!(t.cvtneebf16_ps_x4(&a), [1., 3., 5., 7.]);
}

#[test]
fn cvtneobf16_ps_x4_takes_odd_indexed_elements() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT];
	assert_eq!(t.cvtneobf16_ps_x4(&a), [2., 4., 6., 8.]);
}

#[test]
fn cvtneeph_ps_x4_takes_even_indexed_elements() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_SEVEN, F16_EIGHT];
	assert_eq!(t.cvtneeph_ps_x4(&a), [1., 3., 5., 7.]);
}

#[test]
fn cvtneoph_ps_x4_takes_odd_indexed_elements() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_SEVEN, F16_EIGHT];
	assert_eq!(t.cvtneoph_ps_x4(&a), [2., 4., 6., 8.]);
}

#[test]
fn cvtneebf16_ps_x8_matches_scalar_reference() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| (i as u16) << 7 | 0x3f00);
	let hw = t.cvtneebf16_ps_x8(&a);
	let expect: [f32; 8] = core::array::from_fn(|j| bf16_to_f32_scalar(a[2 * j]));
	assert_eq!(hw, expect);
}

#[test]
fn cvtneoph_ps_x8_matches_scalar_reference() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| F16_ONE.wrapping_add(i as u16));
	let hw = t.cvtneoph_ps_x8(&a);
	for (j, &v) in hw.iter().enumerate() {
		let sw = f16_to_f32_scalar(a[2 * j + 1]);
		assert!(v.to_bits() == sw.to_bits() || (v.is_nan() && sw.is_nan()), "j={j} hw={v} sw={sw}");
	}
}

#[test]
fn cvtneebf16_ps_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxNeConvert::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let a: Vec<u16> = (0..len * 2).map(|i| (i as u16) << 4 | 0x3f00).collect();
		let mut out = vec![0f32; len];
		t.cvtneebf16_ps_slice(&a, &mut out);
		let expect: Vec<f32> = (0..len).map(|j| bf16_to_f32_scalar(a[2 * j])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn cvtneps_avx_pbh_x4_matches_known_values() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	assert_eq!(t.cvtneps_avx_pbh_x4(a), [BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR]);
}

#[test]
fn cvtneps_avx_pbh_x8_matches_known_values() {
	let Some(t) = AvxNeConvert::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	assert_eq!(t.cvtneps_avx_pbh_x8(a), [BF16_ONE, BF16_TWO, BF16_THREE, BF16_FOUR, BF16_FIVE, BF16_SIX, BF16_SEVEN, BF16_EIGHT]);
}

#[test]
fn cvtneps_avx_pbh_slice_matches_scalar_for_various_lengths() {
	let Some(t) = AvxNeConvert::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 1.5 - 3.0).collect();
		let mut out = vec![0u16; len];
		t.cvtneps_avx_pbh_slice(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_bf16_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn bf16_to_f32_scalar_roundtrips_representable_values() {
	assert_eq!(bf16_to_f32_scalar(BF16_ONE), 1.0);
	assert_eq!(bf16_to_f32_scalar(BF16_TWO), 2.0);
}

#[test]
fn f32_to_bf16_scalar_rounds_to_nearest_even() {
	assert_eq!(f32_to_bf16_scalar(1.0), BF16_ONE);
	assert!(f32_to_bf16_scalar(f32::NAN) & 0x7f80 == 0x7f80);
}
