use super::*;

/// XOR-shift carry-less multiply: `out = sum_i (b bit i set) ? a << i : 0`,
/// XORed instead of added (no carry). Reference for [`Pclmulqdq::clmul`].
fn clmul_scalar(a: u64, b: u64) -> u128 {
	let mut result: u128 = 0;
	for i in 0..64 {
		if (b >> i) & 1 == 1 {
			result ^= (a as u128) << i;
		}
	}
	result
}

#[test]
fn clmul_matches_hand_verified_small_cases() {
	let Some(pclmul) = Pclmulqdq::detect() else { return };
	// 0b11 CLMUL 0b11: shift0 (0b11) xor shift1 (0b110) = 0b101 = 5.
	let a = [0b11u64, 0];
	let b = [0b11u64, 0];
	assert_eq!(pclmul.clmul::<0x00>(a, b), [5, 0]);

	// 1 CLMUL 1 = 1 (identity).
	let a = [1u64, 0];
	let b = [1u64, 0];
	assert_eq!(pclmul.clmul::<0x00>(a, b), [1, 0]);
}

#[test]
fn clmul_matches_scalar_reference_for_all_half_selections() {
	let Some(pclmul) = Pclmulqdq::detect() else { return };
	let a: [u64; 2] = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
	let b: [u64; 2] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];

	let cases: [(i32, u64, u64); 4] =
		[(0x00, a[0], b[0]), (0x01, a[1], b[0]), (0x10, a[0], b[1]), (0x11, a[1], b[1])];

	for (imm, av, bv) in cases {
		let expect = clmul_scalar(av, bv);
		let expect = [expect as u64, (expect >> 64) as u64];
		let out = match imm {
			0x00 => pclmul.clmul::<0x00>(a, b),
			0x01 => pclmul.clmul::<0x01>(a, b),
			0x10 => pclmul.clmul::<0x10>(a, b),
			0x11 => pclmul.clmul::<0x11>(a, b),
			_ => unreachable!(),
		};
		assert_eq!(out, expect, "imm=0x{imm:02x}");
	}
}

#[test]
fn clmul_matches_scalar_reference_for_random_inputs() {
	let Some(pclmul) = Pclmulqdq::detect() else { return };
	let mut state = 0x9E37_79B9_7F4A_7C15u64;
	let mut next = || {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		state
	};
	for _ in 0..20 {
		let av = next();
		let bv = next();
		let a = [av, 0];
		let b = [bv, 0];
		let expect = clmul_scalar(av, bv);
		let expect = [expect as u64, (expect >> 64) as u64];
		assert_eq!(pclmul.clmul::<0x00>(a, b), expect);
	}
}
