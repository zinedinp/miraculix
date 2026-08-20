use super::*;

/// XOR-shift carry-less multiply, same reference as `pclmulqdq.rs`'s
/// `clmul_scalar` (duplicated, not shared: this crate's convention:
/// each crypto file keeps its own scalar reference, see `sha512.rs`/
/// `sm3.rs`/`sm4.rs`).
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
fn clmul_256_matches_scalar_reference_per_lane() {
	let Some(t) = Vpclmulqdq::detect() else { return };
	let a: [u64; 4] = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
	let b: [u64; 4] = [0xaaaa_bbbb_cccc_dddd, 0x0f0f_0f0f_0f0f_0f0f, 0x9999_8888_7777_6666, 0x1234_5678_9abc_def0];

	let got = t.clmul_u64x4::<0x00>(a, b);
	let lane0 = clmul_scalar(a[0], b[0]);
	let lane1 = clmul_scalar(a[2], b[2]);
	assert_eq!(got, [lane0 as u64, (lane0 >> 64) as u64, lane1 as u64, (lane1 >> 64) as u64]);
}

#[test]
fn clmul_256_all_half_selections_match_scalar_reference() {
	let Some(t) = Vpclmulqdq::detect() else { return };
	let a: [u64; 4] = [1, 2, 3, 4];
	let b: [u64; 4] = [5, 6, 7, 8];

	let cases: [(i32, u64, u64, u64, u64); 4] =
		[(0x00, a[0], b[0], a[2], b[2]), (0x01, a[1], b[0], a[3], b[2]), (0x10, a[0], b[1], a[2], b[3]), (0x11, a[1], b[1], a[3], b[3])];
	for (imm, av0, bv0, av1, bv1) in cases {
		let expect0 = clmul_scalar(av0, bv0);
		let expect1 = clmul_scalar(av1, bv1);
		let expect = [expect0 as u64, (expect0 >> 64) as u64, expect1 as u64, (expect1 >> 64) as u64];
		let out = match imm {
			0x00 => t.clmul_u64x4::<0x00>(a, b),
			0x01 => t.clmul_u64x4::<0x01>(a, b),
			0x10 => t.clmul_u64x4::<0x10>(a, b),
			0x11 => t.clmul_u64x4::<0x11>(a, b),
			_ => unreachable!(),
		};
		assert_eq!(out, expect, "imm=0x{imm:02x}");
	}
}

#[test]
fn clmul_512_matches_scalar_reference_per_lane() {
	let Some(t) = Vpclmulqdq512::detect() else { return };
	let a: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
	let b: [u64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

	let got = t.clmul_u64x8::<0x00>(a, b);
	for lane in 0..4 {
		let expect = clmul_scalar(a[lane * 2], b[lane * 2]);
		assert_eq!(got[lane * 2], expect as u64, "lane {lane} low");
		assert_eq!(got[lane * 2 + 1], (expect >> 64) as u64, "lane {lane} high");
	}
}

#[test]
fn clmul_matches_hand_verified_small_case() {
	let Some(t) = Vpclmulqdq::detect() else { return };
	// Same hand-verified case as pclmulqdq.rs: 0b11 CLMUL 0b11 = 0b101 = 5.
	let a = [0b11u64, 0, 0b11, 0];
	let b = [0b11u64, 0, 0b11, 0];
	assert_eq!(t.clmul_u64x4::<0x00>(a, b), [5, 0, 5, 0]);
}
