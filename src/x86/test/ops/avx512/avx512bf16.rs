use super::*;

use f32_to_bf16_scalar as f32_to_bf16_rne;
use bf16_to_f32_scalar as bf16_to_f32;

#[test]
fn dpbf16_ps_matches_scalar_reference() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5);
	let a_f32: [f32; 32] = core::array::from_fn(|i| (i as f32 - 16.0) * 0.25);
	let b_f32: [f32; 32] = core::array::from_fn(|i| (i as f32 % 5.0) - 2.0);
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_bf16_rne(a_f32[i]));
	let b: [u16; 32] = core::array::from_fn(|i| f32_to_bf16_rne(b_f32[i]));

	let got = t.dpbf16_ps_f32x16(src, a, b);
	let expect: [f32; 16] = core::array::from_fn(|j| {
		let mut acc = src[j];
		acc += bf16_to_f32(a[2 * j + 1]) * bf16_to_f32(b[2 * j + 1]);
		acc += bf16_to_f32(a[2 * j]) * bf16_to_f32(b[2 * j]);
		acc
	});
	assert_eq!(got, expect);
}

#[test]
fn dpbf16_ps_zero_src_is_pure_dot_product() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let one = f32_to_bf16_rne(1.0);
	let two = f32_to_bf16_rne(2.0);
	let a = [one; 32];
	let b = [two; 32];
	let got = t.dpbf16_ps_f32x16([0.0; 16], a, b);
	// Each lane: 1*2 + 1*2 = 4.0 exactly (both operands exact in BF16).
	assert_eq!(got, [4.0f32; 16]);
}

#[test]
fn cvtneps_pbh_matches_scalar_rne_reference() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] =
		core::array::from_fn(|i| [1.0, -1.0, 0.0, f32::MIN_POSITIVE, 12345.678, -0.001, 9876.543, 1e30][i % 8] * (i as f32 + 1.0));
	let got = t.cvtneps_pbh_u16x16(a);
	let expect: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_rne(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtneps_pbh_exact_values_round_trip() {
	let Some(t) = Avx512Bf16::detect() else { return };
	// Values exactly representable in BF16 (top 16 bits, zero mantissa tail).
	let a: [f32; 16] = core::array::from_fn(|i| f32::from_bits((i as u32 + 1) << 23));
	let got = t.cvtneps_pbh_u16x16(a);
	for i in 0..16 {
		assert_eq!(bf16_to_f32(got[i]), a[i], "lane {i}");
	}
}

#[test]
fn cvtne2ps_pbh_matches_scalar_rne_reference_and_lane_order() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 0.3);
	let b: [f32; 16] = core::array::from_fn(|i| -(i as f32) - 0.7);
	let got = t.cvtne2ps_pbh_u16x32(a, b);

	// Per Guide pseudocode: dst.word[j] = b[j] for j<16, a[j-16] for j>=16.
	let expect_low: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_rne(b[i]));
	let expect_high: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_rne(a[i]));
	assert_eq!(&got[0..16], &expect_low);
	assert_eq!(&got[16..32], &expect_high);
}

#[test]
fn cvtne2ps_pbh_composes_with_cvtneps_pbh_on_each_half() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 1.5);
	let b: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * -2.5);
	let combined = t.cvtne2ps_pbh_u16x32(a, b);
	let a_alone = t.cvtneps_pbh_u16x16(a);
	let b_alone = t.cvtneps_pbh_u16x16(b);
	assert_eq!(&combined[0..16], &b_alone);
	assert_eq!(&combined[16..32], &a_alone);
}

#[test]
fn dpbf16_ps_f32_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Bf16::detect() else { return };
	for len in [0usize, 1, 3, 15, 16, 17, 31, 32, 33, 100] {
		let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let a: Vec<u16> = (0..2 * len).map(|i| f32_to_bf16_rne((i as f32 - 40.0) * 0.25)).collect();
		let b: Vec<u16> = (0..2 * len).map(|i| f32_to_bf16_rne((i as f32 % 5.0) - 2.0)).collect();
		let mut got = vec![0f32; len];
		t.dpbf16_ps_f32_slice(&src, &a, &b, &mut got);

		let expect: Vec<f32> = (0..len)
			.map(|j| {
				let mut acc = src[j];
				acc += bf16_to_f32(a[2 * j + 1]) * bf16_to_f32(b[2 * j + 1]);
				acc += bf16_to_f32(a[2 * j]) * bf16_to_f32(b[2 * j]);
				acc
			})
			.collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
fn cvtneps_pbh_u16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Bf16::detect() else { return };
	for len in [0usize, 1, 3, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| (i as f32 - 50.0) * 12.375).collect();
		let mut got = vec![0u16; len];
		t.cvtneps_pbh_u16_slice(&a, &mut got);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_bf16_rne(x)).collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
fn cvtne2ps_pbh_u16_slice_matches_scalar_for_various_lengths() {
	let Some(t) = Avx512Bf16::detect() else { return };
	for len in [0usize, 1, 3, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 + 0.3).collect();
		let b: Vec<f32> = (0..len).map(|i| -(i as f32) - 0.7).collect();
		let mut got = vec![0u16; 2 * len];
		t.cvtne2ps_pbh_u16_slice(&a, &b, &mut got);
		let expect: Vec<u16> = b.iter().map(|&x| f32_to_bf16_rne(x)).chain(a.iter().map(|&x| f32_to_bf16_rne(x))).collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
fn cvtpbh_ps_f32x16_matches_scalar_reference() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_rne((i as f32 - 8.0) * 12.375));
	let got = t.cvtpbh_ps_f32x16(a);
	let expect: [f32; 16] = core::array::from_fn(|i| bf16_to_f32(a[i]));
	assert_eq!(got, expect);
}

#[test]
fn cvtpbh_ps_f32x16_round_trips_with_cvtneps_pbh() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| f32::from_bits((i as u32 + 1) << 23));
	let bf16 = t.cvtneps_pbh_u16x16(a);
	assert_eq!(t.cvtpbh_ps_f32x16(bf16), a);
}

// Oracle for every masked test below is the already-tested unmasked op,
// not a fresh scalar closure: isolates the one new behavior (lane
// selection), same approach as the FP16/IFMA/VNNI masked batches.
const MASK16: u16 = 0x9A37;
const MASK32: u32 = 0x9A37_5C81;

#[test]
fn dpbf16_ps_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5);
	let a: [u16; 32] = core::array::from_fn(|i| f32_to_bf16_rne((i as f32 - 16.0) * 0.25));
	let b: [u16; 32] = core::array::from_fn(|i| f32_to_bf16_rne((i as f32 % 5.0) - 2.0));
	let expect = t.dpbf16_ps_f32x16(src, a, b);
	let merged = t.dpbf16_ps_f32x16_merge_masked(src, MASK16, a, b);
	let zeroed = t.dpbf16_ps_f32x16_zero_masked(MASK16, src, a, b);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0.0 }, "zero lane {i}");
	}
}

#[test]
fn cvtneps_pbh_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| (i as f32 - 8.0) * 1.5);
	let src: [u16; 16] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtneps_pbh_u16x16(a);
	let merged = t.cvtneps_pbh_u16x16_merge_masked(src, MASK16, a);
	let zeroed = t.cvtneps_pbh_u16x16_zero_masked(MASK16, a);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "zero lane {i}");
	}
}

#[test]
fn cvtne2ps_pbh_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32 + 0.3);
	let b: [f32; 16] = core::array::from_fn(|i| -(i as f32) - 0.7);
	let src: [u16; 32] = core::array::from_fn(|i| 999u16.wrapping_add(i as u16));
	let expect = t.cvtne2ps_pbh_u16x32(a, b);
	let merged = t.cvtne2ps_pbh_u16x32_merge_masked(src, MASK32, a, b);
	let zeroed = t.cvtne2ps_pbh_u16x32_zero_masked(MASK32, a, b);
	for i in 0..32 {
		let selected = (MASK32 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "zero lane {i}");
	}
}

#[test]
fn cvtpbh_ps_masked_matches_unmasked() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let a: [u16; 16] = core::array::from_fn(|i| f32_to_bf16_rne((i as f32 - 8.0) * 12.375));
	let src: [f32; 16] = core::array::from_fn(|i| 1000.0 + i as f32);
	let expect = t.cvtpbh_ps_f32x16(a);
	let merged = t.cvtpbh_ps_f32x16_merge_masked(src, MASK16, a);
	let zeroed = t.cvtpbh_ps_f32x16_zero_masked(MASK16, a);
	for i in 0..16 {
		let selected = (MASK16 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0.0 }, "zero lane {i}");
	}
}
