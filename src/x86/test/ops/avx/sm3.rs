use super::*;

/// GB/T 32905-2016 IV.
const IV: [u32; 8] =
	[0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600, 0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e];

fn rol32(x: u32, n: u32) -> u32 {
	x.rotate_left(n)
}
fn p0(x: u32) -> u32 {
	x ^ rol32(x, 9) ^ rol32(x, 17)
}
fn p1(x: u32) -> u32 {
	x ^ rol32(x, 15) ^ rol32(x, 23)
}
fn ff(x: u32, y: u32, z: u32, j: usize) -> u32 {
	if j < 16 { x ^ y ^ z } else { (x & y) | (x & z) | (y & z) }
}
fn gg(x: u32, y: u32, z: u32, j: usize) -> u32 {
	if j < 16 { x ^ y ^ z } else { (x & y) | (!x & z) }
}
fn t_const(j: usize) -> u32 {
	if j < 16 { 0x79cc4519 } else { 0x7a879d8a }
}

/// Test-only: runtime even `round` (0..=62) -> `sm3rnds2::<IMM8>` match.
fn sm3rnds2_dispatch(t: Sm3, cdgh: [u32; 4], abef: [u32; 4], wp: [u32; 4], round: usize) -> [u32; 4] {
	macro_rules! dispatch {
		($($r:literal),*) => {
			match round {
				$($r => t.sm3rnds2::<$r>(cdgh, abef, wp),)*
				_ => unreachable!("round must be even, 0..=62"),
			}
		};
	}
	dispatch!(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62)
}

fn compress_block_hardware(t: Sm3, state: &mut [u32; 8], block: &[u8; 64]) {
	let mut w = [0u32; 68];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
	}
	for j in (16..68).step_by(4) {
		let msg1_a: [u32; 4] = [w[j - 9], w[j - 8], w[j - 7], w[j - 6]];
		let msg1_b: [u32; 4] = [w[j - 3], w[j - 2], w[j - 1], 0];
		let msg1_c: [u32; 4] = [w[j - 16], w[j - 15], w[j - 14], w[j - 13]];
		let msg1_out = t.sm3msg1(msg1_a, msg1_b, msg1_c);
		let msg2_b: [u32; 4] = [w[j - 13], w[j - 12], w[j - 11], w[j - 10]];
		let msg2_c: [u32; 4] = [w[j - 6], w[j - 5], w[j - 4], w[j - 3]];
		let new_w = t.sm3msg2(msg1_out, msg2_b, msg2_c);
		w[j..j + 4].copy_from_slice(&new_w);
	}
	let wp: Vec<u32> = (0..64).map(|j| w[j] ^ w[j + 4]).collect();

	let mut cdgh: [u32; 4] = [state[7], state[6], state[3], state[2]];
	let mut abef: [u32; 4] = [state[5], state[4], state[1], state[0]];

	for j in (0..64).step_by(2) {
		let wp4: [u32; 4] = [w[j], w[j + 1], wp[j], wp[j + 1]];
		let new_abef = sm3rnds2_dispatch(t, cdgh, abef, wp4, j);
		cdgh = abef;
		abef = new_abef;
	}

	let final_abcdefgh = [abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0]];
	for i in 0..8 {
		state[i] ^= final_abcdefgh[i];
	}
}

fn sm3_hardware(t: Sm3, message: &[u8]) -> [u8; 32] {
	let bit_len = (message.len() as u64) * 8;
	let mut padded = message.to_vec();
	padded.push(0x80);
	while padded.len() % 64 != 56 {
		padded.push(0);
	}
	padded.extend_from_slice(&bit_len.to_be_bytes());

	let mut state = IV;
	for block in padded.chunks_exact(64) {
		let block: [u8; 64] = block.try_into().expect("64 bytes");
		compress_block_hardware(t, &mut state, &block);
	}
	let mut out = [0u8; 32];
	for (i, word) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
	}
	out
}

// Software ref always runs: most machines lack SM3 silicon (detect skips HW).
fn compress_block_software(state: &mut [u32; 8], block: &[u8; 64]) {
	let mut w = [0u32; 68];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
	}
	for j in 16..68 {
		w[j] = p1(w[j - 16] ^ w[j - 9] ^ rol32(w[j - 3], 15)) ^ rol32(w[j - 13], 7) ^ w[j - 6];
	}
	let wp: Vec<u32> = (0..64).map(|j| w[j] ^ w[j + 4]).collect();

	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
	for j in 0..64 {
		let ss1 = rol32(rol32(a, 12).wrapping_add(e).wrapping_add(rol32(t_const(j), (j as u32) % 32)), 7);
		let ss2 = ss1 ^ rol32(a, 12);
		let tt1 = ff(a, b, c, j).wrapping_add(d).wrapping_add(ss2).wrapping_add(wp[j]);
		let tt2 = gg(e, f, g, j).wrapping_add(h).wrapping_add(ss1).wrapping_add(w[j]);
		d = c;
		c = rol32(b, 9);
		b = a;
		a = tt1;
		h = g;
		g = rol32(f, 19);
		f = e;
		e = p0(tt2);
	}
	let new_state = [a, b, c, d, e, f, g, h];
	for i in 0..8 {
		state[i] ^= new_state[i];
	}
}

fn sm3_software(message: &[u8]) -> [u8; 32] {
	let bit_len = (message.len() as u64) * 8;
	let mut padded = message.to_vec();
	padded.push(0x80);
	while padded.len() % 64 != 56 {
		padded.push(0);
	}
	padded.extend_from_slice(&bit_len.to_be_bytes());

	let mut state = IV;
	for block in padded.chunks_exact(64) {
		let block: [u8; 64] = block.try_into().expect("64 bytes");
		compress_block_software(&mut state, &block);
	}
	let mut out = [0u8; 32];
	for (i, word) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
	}
	out
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// Vectors via `openssl dgst -sm3` (OpenSSL 3.6.3).
#[test]
fn software_reference_matches_openssl_for_abc() {
	assert_eq!(hex(&sm3_software(b"abc")), "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0");
}

#[test]
fn software_reference_matches_openssl_for_empty_message() {
	assert_eq!(hex(&sm3_software(b"")), "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b");
}

#[test]
fn software_reference_matches_openssl_for_long_message() {
	assert_eq!(
		hex(&sm3_software(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
		"639b6cc5e64d9e37a390b192df4fa1ea0720ab747ff692b9f38c4e66ad7b8c05"
	);
}

#[test]
fn hardware_path_matches_software_reference_for_random_messages() {
	let Some(t) = Sm3::detect() else { return };
	for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 128, 200] {
		let message: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
		assert_eq!(sm3_hardware(t, &message), sm3_software(&message), "len={len}");
	}
}

#[test]
fn sm3msg1_matches_pseudocode_shape() {
	let Some(t) = Sm3::detect() else { return };
	let a = [1u32, 2, 3, 4];
	let b = [5u32, 6, 7, 8];
	let c = [9u32, 10, 11, 12];
	let w4 = [a[0], a[1], a[2], a[3]];
	let expect: [u32; 4] = core::array::from_fn(|i| {
		if i < 3 { p1(c[i] ^ w4[i] ^ rol32(b[i], 15)) } else { p1(c[3] ^ w4[3]) }
	});
	assert_eq!(t.sm3msg1(a, b, c), expect);
}
