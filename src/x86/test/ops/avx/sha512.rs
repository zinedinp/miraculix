use super::*;

/// FIPS 180-4 K: frac(cbrt) of first 80 primes, high 64 bits.
#[rustfmt::skip]
const K: [u64; 80] = [
	0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
	0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
	0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
	0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
	0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
	0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
	0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
	0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
	0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
	0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
	0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
	0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
	0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
	0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
	0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
	0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
	0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
	0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
	0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
	0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// FIPS 180-4 IV: frac(sqrt) of first 8 primes, high 64 bits.
const H0: [u64; 8] = [
	0x6a09e667f3bcc908,
	0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b,
	0xa54ff53a5f1d36f1,
	0x510e527fade682d1,
	0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b,
	0x5be0cd19137e2179,
];

/// One 128-byte block via the 3 hardware primitives (in-place Davies-Meyer).
fn compress_block(t: Sha512, state: &mut [u64; 8], block: &[u8; 128]) {
	let mut w = [0u64; 80];
	for i in 0..16 {
		w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
	}
	for t4 in (16..80).step_by(4) {
		let msg1_a: [u64; 4] = [w[t4 - 16], w[t4 - 15], w[t4 - 14], w[t4 - 13]];
		let msg1_b: [u64; 2] = [w[t4 - 12], w[t4 - 11]];
		let msg1_out = t.sha512msg1(msg1_a, msg1_b);
		let aligned: [u64; 4] = [w[t4 - 7], w[t4 - 6], w[t4 - 5], w[t4 - 4]];
		let p: [u64; 4] = core::array::from_fn(|i| msg1_out[i].wrapping_add(aligned[i]));
		let msg2_b: [u64; 4] = [w[t4 - 4], w[t4 - 3], w[t4 - 2], w[t4 - 1]];
		let new_w = t.sha512msg2(p, msg2_b);
		w[t4..t4 + 4].copy_from_slice(&new_w);
	}

	let mut cdgh: [u64; 4] = [state[7], state[6], state[3], state[2]];
	let mut abef: [u64; 4] = [state[5], state[4], state[1], state[0]];

	for r in (0..80).step_by(2) {
		let wk: [u64; 2] = [w[r].wrapping_add(K[r]), w[r + 1].wrapping_add(K[r + 1])];
		let new_abef = t.sha512rnds2(cdgh, abef, wk);
		cdgh = abef;
		abef = new_abef;
	}

	let final_abcdefgh = [abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0]];
	for i in 0..8 {
		state[i] = state[i].wrapping_add(final_abcdefgh[i]);
	}
}

/// Full SHA-512 (standard padding) via [`compress_block`].
fn sha512(t: Sha512, message: &[u8]) -> [u8; 64] {
	let bit_len = (message.len() as u128) * 8;
	let mut padded = message.to_vec();
	padded.push(0x80);
	while padded.len() % 128 != 112 {
		padded.push(0);
	}
	padded.extend_from_slice(&bit_len.to_be_bytes());
	debug_assert_eq!(padded.len() % 128, 0);

	let mut state = H0;
	for block in padded.chunks_exact(128) {
		let block: [u8; 128] = block.try_into().expect("128 bytes");
		compress_block(t, &mut state, &block);
	}

	let mut out = [0u8; 64];
	for (i, word) in state.iter().enumerate() {
		out[i * 8..i * 8 + 8].copy_from_slice(&word.to_be_bytes());
	}
	out
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn sha512_matches_nist_vector_for_empty_message() {
	let Some(t) = Sha512::detect() else { return };
	let digest = sha512(t, b"");
	assert_eq!(
		hex(&digest),
		"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
	);
}

#[test]
fn sha512_matches_nist_vector_for_abc() {
	let Some(t) = Sha512::detect() else { return };
	let digest = sha512(t, b"abc");
	assert_eq!(
		hex(&digest),
		"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
	);
}

#[test]
fn sha512_matches_nist_vector_for_two_block_message() {
	let Some(t) = Sha512::detect() else { return };
	// FIPS 180-4 56-byte example (full schedule, non-trivial words).
	let digest = sha512(t, b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
	assert_eq!(
		hex(&digest),
		"204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445"
	);
}

#[test]
fn sha512_matches_nist_vector_for_message_spanning_two_blocks() {
	let Some(t) = Sha512::detect() else { return };
	// 130 'a': padding spans a second block.
	let digest = sha512(t, &[b'a'; 130]);
	assert_eq!(
		hex(&digest),
		"b2fc6acdce83feb0b9439433915fe5dc1c73af6f17e962d7badd7ad5dd7c5032bc1744855d0ba09da5e4ab1bb1caca3aad8e4a947faa19c4769e128bacfe6b85"
	);
}

#[test]
fn sha512msg1_matches_pseudocode_shape() {
	let Some(t) = Sha512::detect() else { return };
	fn ror64(x: u64, n: u32) -> u64 {
		x.rotate_right(n)
	}
	fn s0(x: u64) -> u64 {
		ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7)
	}
	let a = [1u64, 2, 3, 4];
	let b = [5u64, 6];
	let w4 = [a[1], a[2], a[3], b[0]];
	let expect: [u64; 4] = core::array::from_fn(|i| a[i].wrapping_add(s0(w4[i])));
	assert_eq!(t.sha512msg1(a, b), expect);
}

// Software ref always runs: most machines lack SHA512 silicon (detect skips HW).
// Validates IV/K/schedule/rounds vs NIST; HW register mapping still needs real silicon.
fn ror64(x: u64, n: u32) -> u64 {
	x.rotate_right(n)
}
fn big_sigma0(x: u64) -> u64 {
	ror64(x, 28) ^ ror64(x, 34) ^ ror64(x, 39)
}
fn big_sigma1(x: u64) -> u64 {
	ror64(x, 14) ^ ror64(x, 18) ^ ror64(x, 41)
}
fn small_sigma0(x: u64) -> u64 {
	ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7)
}
fn small_sigma1(x: u64) -> u64 {
	ror64(x, 19) ^ ror64(x, 61) ^ (x >> 6)
}
fn ch(x: u64, y: u64, z: u64) -> u64 {
	(x & y) ^ (!x & z)
}
fn maj(x: u64, y: u64, z: u64) -> u64 {
	(x & y) ^ (x & z) ^ (y & z)
}

fn compress_block_software(state: &mut [u64; 8], block: &[u8; 128]) {
	let mut w = [0u64; 80];
	for i in 0..16 {
		w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
	}
	for t in 16..80 {
		w[t] = small_sigma1(w[t - 2])
			.wrapping_add(w[t - 7])
			.wrapping_add(small_sigma0(w[t - 15]))
			.wrapping_add(w[t - 16]);
	}

	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
	for t in 0..80 {
		let t1 = h.wrapping_add(big_sigma1(e)).wrapping_add(ch(e, f, g)).wrapping_add(K[t]).wrapping_add(w[t]);
		let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
		h = g;
		g = f;
		f = e;
		e = d.wrapping_add(t1);
		d = c;
		c = b;
		b = a;
		a = t1.wrapping_add(t2);
	}

	let new_state = [a, b, c, d, e, f, g, h];
	for i in 0..8 {
		state[i] = state[i].wrapping_add(new_state[i]);
	}
}

fn sha512_software(message: &[u8]) -> [u8; 64] {
	let bit_len = (message.len() as u128) * 8;
	let mut padded = message.to_vec();
	padded.push(0x80);
	while padded.len() % 128 != 112 {
		padded.push(0);
	}
	padded.extend_from_slice(&bit_len.to_be_bytes());

	let mut state = H0;
	for block in padded.chunks_exact(128) {
		let block: [u8; 128] = block.try_into().expect("128 bytes");
		compress_block_software(&mut state, &block);
	}

	let mut out = [0u8; 64];
	for (i, word) in state.iter().enumerate() {
		out[i * 8..i * 8 + 8].copy_from_slice(&word.to_be_bytes());
	}
	out
}

#[test]
fn software_reference_matches_nist_vector_for_abc() {
	assert_eq!(
		hex(&sha512_software(b"abc")),
		"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
	);
}

#[test]
fn software_reference_matches_nist_vector_for_empty_message() {
	assert_eq!(
		hex(&sha512_software(b"")),
		"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
	);
}

#[test]
fn software_reference_matches_nist_vector_for_two_block_message() {
	assert_eq!(
		hex(&sha512_software(b"a".repeat(130).as_slice())),
		"b2fc6acdce83feb0b9439433915fe5dc1c73af6f17e962d7badd7ad5dd7c5032bc1744855d0ba09da5e4ab1bb1caca3aad8e4a947faa19c4769e128bacfe6b85"
	);
}

#[test]
fn hardware_path_matches_software_reference_for_random_messages() {
	let Some(t) = Sha512::detect() else { return };
	// Real silicon only: HW assembly vs software ref.
	for len in [0usize, 1, 55, 56, 111, 112, 113, 200, 256] {
		let message: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
		assert_eq!(sha512(t, &message), sha512_software(&message), "len={len}");
	}
}
