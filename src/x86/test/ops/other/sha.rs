use super::*;

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// SHA-1

const SHA1_K: [u32; 4] = [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6];
const SHA1_H0: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];

fn rotl32(x: u32, n: u32) -> u32 {
	x.rotate_left(n)
}

fn sha1_pad(message: &[u8], out: &mut [u8; 256], scratch_blocks: usize) -> usize {
	let bit_len = (message.len() as u64) * 8;
	let mut len = message.len();
	out[..len].copy_from_slice(message);
	out[len] = 0x80;
	len += 1;
	while len % 64 != 56 {
		out[len] = 0;
		len += 1;
	}
	out[len..len + 8].copy_from_slice(&bit_len.to_be_bytes());
	len += 8;
	debug_assert_eq!(len % 64, 0);
	debug_assert!(len <= scratch_blocks * 64);
	len
}

fn sha1_compress_block(t: Sha, state: &mut [u32; 5], block: &[u8; 64]) {
	let mut w = [0u32; 80];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
	}
	for i in (16..80).step_by(4) {
		let a4 = [w[i - 13], w[i - 14], w[i - 15], w[i - 16]];
		let b4 = [w[i - 9], w[i - 10], w[i - 11], w[i - 12]];
		let m1 = t.sha1msg1(a4, b4);
		let c4 = [w[i - 5], w[i - 6], w[i - 7], w[i - 8]];
		let xored: [u32; 4] = core::array::from_fn(|j| m1[j] ^ c4[j]);
		let d4 = [w[i - 1], w[i - 2], w[i - 3], w[i - 4]];
		let m2 = t.sha1msg2(xored, d4);
		w[i] = m2[3];
		w[i + 1] = m2[2];
		w[i + 2] = m2[1];
		w[i + 3] = m2[0];
	}

	let [h0, h1, h2, h3, h4] = *state;
	// state array = [D, C, B, A] (index 0 = bottom lane).
	let mut abcd = [h3, h2, h1, h0];
	let mut saved_input: Option<[u32; 4]> = None;
	for g in 0..20 {
		let func_idx = g / 5;
		let wgrp = [w[4 * g + 3], w[4 * g + 2], w[4 * g + 1], w[4 * g]];
		let wadj = if g == 0 {
			[wgrp[0], wgrp[1], wgrp[2], wgrp[3].wrapping_add(h4)]
		} else {
			t.sha1nexte(saved_input.expect("set on g>0"), wgrp)
		};
		let this_input = abcd;
		abcd = match func_idx {
			0 => t.sha1rnds4::<0>(abcd, wadj),
			1 => t.sha1rnds4::<1>(abcd, wadj),
			2 => t.sha1rnds4::<2>(abcd, wadj),
			_ => t.sha1rnds4::<3>(abcd, wadj),
		};
		saved_input = Some(this_input);
	}
	let e2 = rotl32(saved_input.expect("20 groups ran")[3], 30);
	let [d2, c2, b2, a2] = abcd;
	state[0] = state[0].wrapping_add(a2);
	state[1] = state[1].wrapping_add(b2);
	state[2] = state[2].wrapping_add(c2);
	state[3] = state[3].wrapping_add(d2);
	state[4] = state[4].wrapping_add(e2);
}

fn sha1(t: Sha, message: &[u8]) -> [u8; 20] {
	let mut buf = [0u8; 256];
	let padded_len = sha1_pad(message, &mut buf, 4);
	let mut state = SHA1_H0;
	for block in buf[..padded_len].chunks_exact(64) {
		sha1_compress_block(t, &mut state, block.try_into().unwrap());
	}
	let mut out = [0u8; 20];
	for (i, w) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
	}
	out
}

fn f_ch(b: u32, c: u32, d: u32) -> u32 {
	(b & c) | (!b & d)
}
fn f_parity(b: u32, c: u32, d: u32) -> u32 {
	b ^ c ^ d
}
fn f_maj(b: u32, c: u32, d: u32) -> u32 {
	(b & c) | (b & d) | (c & d)
}

fn sha1_compress_block_software(state: &mut [u32; 5], block: &[u8; 64]) {
	let mut w = [0u32; 80];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
	}
	for i in 16..80 {
		w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
	}
	let [mut a, mut b, mut c, mut d, mut e] = *state;
	for (i, &wi) in w.iter().enumerate() {
		let (f, k) = match i / 20 {
			0 => (f_ch(b, c, d), SHA1_K[0]),
			1 => (f_parity(b, c, d), SHA1_K[1]),
			2 => (f_maj(b, c, d), SHA1_K[2]),
			_ => (f_parity(b, c, d), SHA1_K[3]),
		};
		let temp = rotl32(a, 5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
		e = d;
		d = c;
		c = rotl32(b, 30);
		b = a;
		a = temp;
	}
	let new_state = [a, b, c, d, e];
	for i in 0..5 {
		state[i] = state[i].wrapping_add(new_state[i]);
	}
}

fn sha1_software(message: &[u8]) -> [u8; 20] {
	let mut buf = [0u8; 256];
	let padded_len = sha1_pad(message, &mut buf, 4);
	let mut state = SHA1_H0;
	for block in buf[..padded_len].chunks_exact(64) {
		sha1_compress_block_software(&mut state, block.try_into().unwrap());
	}
	let mut out = [0u8; 20];
	for (i, w) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
	}
	out
}

#[test]
fn sha1_software_matches_nist_vector_for_abc() {
	assert_eq!(hex(&sha1_software(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
}

#[test]
fn sha1_software_matches_nist_vector_for_empty_message() {
	assert_eq!(hex(&sha1_software(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
}

#[test]
fn sha1_hardware_matches_nist_vector_for_abc() {
	let Some(t) = Sha::detect() else { return };
	assert_eq!(hex(&sha1(t, b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
}

#[test]
fn sha1_hardware_matches_software_reference_for_random_messages() {
	let Some(t) = Sha::detect() else { return };
	for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 100, 119, 120] {
		let mut message = [0u8; 120];
		for (i, byte) in message.iter_mut().enumerate().take(len) {
			*byte = (i * 37 + 11) as u8;
		}
		assert_eq!(sha1(t, &message[..len]), sha1_software(&message[..len]), "len={len}");
	}
}

// SHA-256

#[rustfmt::skip]
const SHA256_K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const SHA256_H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

fn rotr32(x: u32, n: u32) -> u32 {
	x.rotate_right(n)
}

fn sha256_compress_block(t: Sha, state: &mut [u32; 8], block: &[u8; 64]) {
	let mut w = [0u32; 64];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
	}
	for t4 in (16..64).step_by(4) {
		let msg1_a = [w[t4 - 16], w[t4 - 15], w[t4 - 14], w[t4 - 13]];
		let msg1_b = [w[t4 - 12], 0, 0, 0];
		let m1 = t.sha256msg1(msg1_a, msg1_b);
		let aligned = [w[t4 - 7], w[t4 - 6], w[t4 - 5], w[t4 - 4]];
		let p: [u32; 4] = core::array::from_fn(|i| m1[i].wrapping_add(aligned[i]));
		let msg2_b = [w[t4 - 4], w[t4 - 3], w[t4 - 2], w[t4 - 1]];
		let new_w = t.sha256msg2(p, msg2_b);
		w[t4..t4 + 4].copy_from_slice(&new_w);
	}

	let [a, b, c, d, e, f, g, h] = *state;
	let mut cdgh = [h, g, d, c];
	let mut abef = [f, e, b, a];

	for r in (0..64).step_by(2) {
		let wk = [w[r].wrapping_add(SHA256_K[r]), w[r + 1].wrapping_add(SHA256_K[r + 1]), 0, 0];
		let new_abef = t.sha256rnds2(cdgh, abef, wk);
		cdgh = abef;
		abef = new_abef;
	}

	let final_state = [abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0]];
	for i in 0..8 {
		state[i] = state[i].wrapping_add(final_state[i]);
	}
}

fn sha256(t: Sha, message: &[u8]) -> [u8; 32] {
	let mut buf = [0u8; 256];
	let padded_len = sha1_pad(message, &mut buf, 4);
	let mut state = SHA256_H0;
	for block in buf[..padded_len].chunks_exact(64) {
		sha256_compress_block(t, &mut state, block.try_into().unwrap());
	}
	let mut out = [0u8; 32];
	for (i, w) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
	}
	out
}

fn big_sigma0(x: u32) -> u32 {
	rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)
}
fn big_sigma1(x: u32) -> u32 {
	rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)
}
fn small_sigma0(x: u32) -> u32 {
	rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)
}
fn small_sigma1(x: u32) -> u32 {
	rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)
}
fn ch32(x: u32, y: u32, z: u32) -> u32 {
	(x & y) ^ (!x & z)
}
fn maj32(x: u32, y: u32, z: u32) -> u32 {
	(x & y) ^ (x & z) ^ (y & z)
}

fn sha256_compress_block_software(state: &mut [u32; 8], block: &[u8; 64]) {
	let mut w = [0u32; 64];
	for i in 0..16 {
		w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
	}
	for t in 16..64 {
		w[t] = small_sigma1(w[t - 2]).wrapping_add(w[t - 7]).wrapping_add(small_sigma0(w[t - 15])).wrapping_add(w[t - 16]);
	}
	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
	for t in 0..64 {
		let t1 = h.wrapping_add(big_sigma1(e)).wrapping_add(ch32(e, f, g)).wrapping_add(SHA256_K[t]).wrapping_add(w[t]);
		let t2 = big_sigma0(a).wrapping_add(maj32(a, b, c));
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

fn sha256_software(message: &[u8]) -> [u8; 32] {
	let mut buf = [0u8; 256];
	let padded_len = sha1_pad(message, &mut buf, 4);
	let mut state = SHA256_H0;
	for block in buf[..padded_len].chunks_exact(64) {
		sha256_compress_block_software(&mut state, block.try_into().unwrap());
	}
	let mut out = [0u8; 32];
	for (i, w) in state.iter().enumerate() {
		out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
	}
	out
}

#[test]
fn sha256_software_matches_nist_vector_for_abc() {
	assert_eq!(hex(&sha256_software(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[test]
fn sha256_software_matches_nist_vector_for_empty_message() {
	assert_eq!(hex(&sha256_software(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

#[test]
fn sha256_hardware_matches_nist_vector_for_abc() {
	let Some(t) = Sha::detect() else { return };
	assert_eq!(hex(&sha256(t, b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[test]
fn sha256_hardware_matches_software_reference_for_random_messages() {
	let Some(t) = Sha::detect() else { return };
	for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 100, 119, 120] {
		let mut message = [0u8; 120];
		for (i, byte) in message.iter_mut().enumerate().take(len) {
			*byte = (i * 37 + 11) as u8;
		}
		assert_eq!(sha256(t, &message[..len]), sha256_software(&message[..len]), "len={len}");
	}
}
