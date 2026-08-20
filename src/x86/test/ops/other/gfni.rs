use super::*;

fn affine_reference<const W: usize>(x: [u8; W], matrix: [u8; W], imm8: u8, inv: bool) -> [u8; W] {
	let mut out = [0u8; W];
	for lane in (0..W).step_by(8) {
		let mut mword = 0u64;
		for i in 0..8 {
			mword |= (matrix[lane + i] as u64) << (8 * i);
		}
		let xlane: [u8; 8] = core::array::from_fn(|i| x[lane + i]);
		let rlane = affine_lane(mword, xlane, imm8, inv);
		out[lane..lane + 8].copy_from_slice(&rlane);
	}
	out
}

#[test]
fn gf2p8mul_matches_scalar_reference_for_random_bytes() {
	let Some(t) = Gfni::detect() else { return };
	let mut state = 0x9E37_79B9u32;
	let mut next = || {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		state as u8
	};
	let a: [u8; 16] = core::array::from_fn(|_| next());
	let b: [u8; 16] = core::array::from_fn(|_| next());
	let expect: [u8; 16] = core::array::from_fn(|i| gf2p8mul_byte(a[i], b[i]));
	assert_eq!(t.gf2p8mul_epi8_u8x16(a, b), expect);

	let a32: [u8; 32] = core::array::from_fn(|_| next());
	let b32: [u8; 32] = core::array::from_fn(|_| next());
	let expect32: [u8; 32] = core::array::from_fn(|i| gf2p8mul_byte(a32[i], b32[i]));
	assert_eq!(t.gf2p8mul_epi8_u8x32(a32, b32), expect32);
}

#[test]
fn gf2p8mul_512_matches_scalar_reference() {
	let Some(t) = Gfni512::detect() else { return };
	let mut state = 0xD1B5_4A32u32;
	let mut next = || {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		state as u8
	};
	let a: [u8; 64] = core::array::from_fn(|_| next());
	let b: [u8; 64] = core::array::from_fn(|_| next());
	let expect: [u8; 64] = core::array::from_fn(|i| gf2p8mul_byte(a[i], b[i]));
	assert_eq!(t.gf2p8mul_epi8_u8x64(a, b), expect);
}

#[test]
fn gf2p8mul_identity_and_zero() {
	let Some(t) = Gfni::detect() else { return };
	let a = [0x53u8; 16];
	let one = [1u8; 16];
	let zero = [0u8; 16];
	assert_eq!(t.gf2p8mul_epi8_u8x16(a, one), a);
	assert_eq!(t.gf2p8mul_epi8_u8x16(a, zero), zero);
}

#[test]
fn gf2p8affine_matches_scalar_reference() {
	let Some(t) = Gfni::detect() else { return };
	let x: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(53).wrapping_add(7));
	let matrix: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
	let expect = affine_reference::<16>(x, matrix, 0x63, false);
	assert_eq!(t.gf2p8affine_epi64_epi8_u8x16::<0x63>(x, matrix), expect);
}

#[test]
fn gf2p8affine_512_matches_scalar_reference() {
	let Some(t) = Gfni512::detect() else { return };
	let x: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(53).wrapping_add(7));
	let matrix: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
	let expect = affine_reference::<64>(x, matrix, 0x1D, false);
	assert_eq!(t.gf2p8affine_epi64_epi8_u8x64::<0x1D>(x, matrix), expect);
}

#[test]
fn gf2p8affineinv_matches_scalar_reference() {
	let Some(t) = Gfni::detect() else { return };
	let x: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
	let matrix: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(1));
	let expect = affine_reference::<16>(x, matrix, 0x00, true);
	assert_eq!(t.gf2p8affineinv_epi64_epi8_u8x16::<0x00>(x, matrix), expect);
}

#[test]
fn gf2p8affineinv_identity_matrix_and_zero_imm_is_inverse() {
	// The identity matrix (each qword = 0x8040201008040201, one set bit
	// per output row) with imm8=0 makes affineinv exactly the GF(2^8)
	// inverse (0 -> 0).
	let Some(t) = Gfni::detect() else { return };
	// byte[j] = 1 << (7-j): affine_byte reads `matrix.byte[7-i]`, so this
	// puts bit i of the matrix's "row 7-i" at the position that isolates
	// src.bit[i] under `parity(matrix.byte[7-i] AND src)`.
	let identity_qword: u64 = 0x0102_0408_1020_4080;
	let matrix: [u8; 16] = core::array::from_fn(|i| ((identity_qword >> (8 * (i % 8))) & 0xff) as u8);
	let x = [0u8, 1, 2, 5, 0x53, 0xff, 0x11, 0x80, 3, 4, 6, 7, 9, 0xAA, 0x55, 0x01];
	let got = t.gf2p8affineinv_epi64_epi8_u8x16::<0x00>(x, matrix);
	for i in 0..16 {
		let inv = if x[i] == 0 { 0 } else { (1..=255u16).map(|v| v as u8).find(|&v| gf2p8mul_byte(x[i], v) == 1).unwrap() };
		assert_eq!(got[i], inv, "byte {i}");
	}
}

#[test]
fn gf2p8mul_matches_aes_sbox_style_reference_property() {
	// GF(2^8) multiplication is commutative and distributes over the same
	// field AES itself uses; sanity check a hand-computed small case from
	// the Guide's own pseudocode shape (0x02 CLMUL-like doubling: x*2 in
	// AES's field is `(x<<1) ^ (0x1B if x&0x80 else 0)`, the standard
	// `xtime` used by MixColumns).
	let Some(t) = Gfni::detect() else { return };
	fn xtime(x: u8) -> u8 {
		if x & 0x80 != 0 { (x << 1) ^ 0x1B } else { x << 1 }
	}
	let a = [0x57u8; 16];
	let two = [2u8; 16];
	let expect = [xtime(0x57); 16];
	assert_eq!(t.gf2p8mul_epi8_u8x16(a, two), expect);
}
