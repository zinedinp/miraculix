use super::*;

#[test]
fn permutexvar_u8x64_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let idx: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let expect = permutexvar_scalar(&idx, &a);
	assert_eq!(t.permutexvar_u8x64(idx, a).to_vec(), expect);
}

#[test]
fn permutexvar_u8x64_index_wraps_mod_64() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let mut a = [0u8; 64];
	a[5] = 0xAB;
	let mut idx = [0u8; 64];
	// 5 + 64 wraps to 5 via the low 6 bits (bit 6 has no meaning here).
	idx[0] = 5 + 64;
	assert_eq!(t.permutexvar_u8x64(idx, a)[0], 0xAB);
}

#[test]
fn permutex2var_u8x64_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	let b: [u8; 64] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let idx: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let expect = permutex2var_scalar(&a, &idx, &b);
	assert_eq!(t.permutex2var_u8x64(a, idx, b).to_vec(), expect);
}

#[test]
fn permutex2var_u8x64_select_bit_chooses_b() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let mut a = [0u8; 64];
	let mut b = [0u8; 64];
	a[3] = 1;
	b[3] = 2;
	let mut idx = [0u8; 64];
	idx[0] = 3 | 0x40; // select b, offset 3
	assert_eq!(t.permutex2var_u8x64(a, idx, b)[0], 2);
	idx[0] = 3; // select a, offset 3
	assert_eq!(t.permutex2var_u8x64(a, idx, b)[0], 1);
}

#[test]
fn multishift_u8x64_matches_scalar_reference() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29) ^ 0x5A);
	let b: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(71) ^ 0x0F);
	let expect = multishift_scalar(&a, &b);
	assert_eq!(t.multishift_u8x64(a, b).to_vec(), expect);
}

#[test]
fn multishift_u8x64_zero_control_takes_low_byte() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a = [0u8; 64]; // every control byte is 0
	let mut b = [0u8; 64];
	b[0] = 0xAB; // low byte of the first qword
	let out = t.multishift_u8x64(a, b);
	assert_eq!(out[0], 0xAB);
}

// Oracle is the already-tested unmasked op, not the scalar reference: it
// isolates the one new behavior (lane selection) the same way the FP16
// masked batch did.
const MASK64: u64 = 0x9A37_5C81_0F2E_47B3;

fn assert_masked_binop_matches<F: Fn([u8; 64], [u8; 64]) -> [u8; 64]>(
	unmasked: F, merged: [u8; 64], zeroed: [u8; 64], src: [u8; 64], a: [u8; 64], b: [u8; 64],
) {
	let expect = unmasked(a, b);
	for i in 0..64 {
		let selected = (MASK64 >> i) & 1 == 1;
		assert_eq!(merged[i], if selected { expect[i] } else { src[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "zero lane {i}");
	}
}

#[test]
fn permutexvar_u8x64_masked_matches_unmasked() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let idx: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(53) ^ 0x3C);
	let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
	let src: [u8; 64] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let merged = t.permutexvar_u8x64_merge_masked(src, MASK64, idx, a);
	let zeroed = t.permutexvar_u8x64_zero_masked(MASK64, idx, a);
	assert_masked_binop_matches(|idx, a| t.permutexvar_u8x64(idx, a), merged, zeroed, src, idx, a);
}

#[test]
fn multishift_u8x64_masked_matches_unmasked() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29) ^ 0x5A);
	let b: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(71) ^ 0x0F);
	let src: [u8; 64] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let merged = t.multishift_u8x64_merge_masked(src, MASK64, a, b);
	let zeroed = t.multishift_u8x64_zero_masked(MASK64, a, b);
	assert_masked_binop_matches(|a, b| t.multishift_u8x64(a, b), merged, zeroed, src, a, b);
}

#[test]
fn permutex2var_u8x64_masked_matches_unmasked() {
	let Some(t) = Avx512Vbmi::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	let idx: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(41) ^ 0x11);
	let b: [u8; 64] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
	let expect = t.permutex2var_u8x64(a, idx, b);
	let merged = t.permutex2var_u8x64_merge_masked(a, MASK64, idx, b);
	let zeroed = t.permutex2var_u8x64_zero_masked(MASK64, a, idx, b);
	for i in 0..64 {
		let selected = (MASK64 >> i) & 1 == 1;
		// Merge fallback is `a`, the first operand, not a separate `src` -
		// the encoding has no room for one (see the doc comment above).
		assert_eq!(merged[i], if selected { expect[i] } else { a[i] }, "merge lane {i}");
		assert_eq!(zeroed[i], if selected { expect[i] } else { 0 }, "zero lane {i}");
	}
}
