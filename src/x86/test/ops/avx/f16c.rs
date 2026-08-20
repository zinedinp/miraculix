use super::*;

const F16_ONE: u16 = 0x3c00;
const F16_TWO: u16 = 0x4000;
const F16_THREE: u16 = 0x4200;
const F16_FOUR: u16 = 0x4400;
const F16_FIVE: u16 = 0x4500;
const F16_SIX: u16 = 0x4600;
const F16_SEVEN: u16 = 0x4700;
const F16_EIGHT: u16 = 0x4800;

/// One-directional, not equality: `from_level` under-detects real hardware
/// outside its bucket (see the identical fix + rationale on
/// `Avx::from_level_agreeing_implies_detect_agrees`, `avx.rs`).
#[test]
fn from_level_agreeing_implies_detect_agrees() {
	let level = GenericLevel::detect(FeatureSet::detect());
	if F16c::from_level(level).is_some() {
		assert!(F16c::detect().is_some());
	}
}

#[test]
fn f16_to_f32x4_converts_known_values() {
	let Some(t) = F16c::detect() else { return };
	let a = [F16_ONE, F16_TWO, F16_THREE, F16_FOUR];
	assert_eq!(t.f16_to_f32x4(a), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f16_to_f32x8_converts_known_values() {
	let Some(t) = F16c::detect() else { return };
	let a = [F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_SEVEN, F16_EIGHT];
	assert_eq!(t.f16_to_f32x8(a), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn f32_to_f16x4_converts_known_values() {
	let Some(t) = F16c::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0];
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	assert_eq!(t.f32_to_f16x4::<ROUNDING>(a), [F16_ONE, F16_TWO, F16_THREE, F16_FOUR]);
}

#[test]
fn f32_to_f16x8_converts_known_values() {
	let Some(t) = F16c::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	assert_eq!(
		t.f32_to_f16x8::<ROUNDING>(a),
		[F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_SEVEN, F16_EIGHT]
	);
}

#[test]
fn roundtrip_is_exact_for_representable_values() {
	let Some(t) = F16c::detect() else { return };
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	// 1023.5 not 1024.5: half ULP is 0.5 in [512,1024), 1.0 in [1024,2048).
	let f32s = [0.0f32, -0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1023.5];
	let halves = t.f32_to_f16x8::<ROUNDING>(f32s);
	let back = t.f16_to_f32x8(halves);
	assert_eq!(back, f32s);
}

#[test]
fn scalar_matches_hardware_for_various_bit_patterns() {
	let Some(t) = F16c::detect() else { return };
	for bits in (0u16..=0xffff).step_by(37) {
		let hw = t.f16_to_f32x8([bits, 0, 0, 0, 0, 0, 0, 0])[0];
		let sw = f16_to_f32_scalar(bits);
		assert!(hw.to_bits() == sw.to_bits() || (hw.is_nan() && sw.is_nan()), "bits={bits:#06x} hw={hw} sw={sw}");
	}
}

#[test]
fn f16_to_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = F16c::detect() else { return };
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<u16> = (0..len).map(|i| (i as u16 % 100) | F16_ONE).collect();
		let mut out = vec![0f32; len];
		t.f16_to_f32_slice(&a, &mut out);
		let expect: Vec<f32> = a.iter().map(|&x| f16_to_f32_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn f32_to_f16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = F16c::detect() else { return };
	const ROUNDING: i32 = core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
		let mut out = vec![0u16; len];
		t.f32_to_f16_slice::<ROUNDING>(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_f16_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn f32_to_f16_scalar_handles_inf_nan_and_overflow() {
	assert_eq!(f32_to_f16_scalar(f32::INFINITY), 0x7c00);
	assert_eq!(f32_to_f16_scalar(f32::NEG_INFINITY), 0xfc00);
	assert!(f16_to_f32_scalar(f32_to_f16_scalar(f32::NAN)).is_nan());
	assert_eq!(f32_to_f16_scalar(1.0e10), 0x7c00);
	assert_eq!(f32_to_f16_scalar(-1.0e10), 0xfc00);
}
