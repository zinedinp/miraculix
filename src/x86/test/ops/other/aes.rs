use core::arch::x86_64::{_mm_shuffle_epi32, _mm_slli_si128, _mm_xor_si128};

use super::*;

fn xor16(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
	core::array::from_fn(|i| a[i] ^ b[i])
}

/// Test helper: one AES-NI 128-bit key-expansion step using
/// `aeskeygenassist` and small SSE2 lane-shuffles. This is test-only
/// infrastructure, not a public wrapped op.
fn expand_key<const RCON: i32>(aes: Aes, prev: [u8; 16]) -> [u8; 16] {
	let assist = aes.aeskeygenassist::<RCON>(prev);
	unsafe {
		let mut key: __m128i = _mm_loadu_si128(prev.as_ptr().cast());
		let keygened: __m128i = _mm_loadu_si128(assist.as_ptr().cast());
		// 0xff = _MM_SHUFFLE(3, 3, 3, 3): broadcast lane 3 into all 4 lanes.
		let keygened = _mm_shuffle_epi32::<0xff>(keygened);
		key = _mm_xor_si128(key, _mm_slli_si128::<4>(key));
		key = _mm_xor_si128(key, _mm_slli_si128::<4>(key));
		key = _mm_xor_si128(key, _mm_slli_si128::<4>(key));
		key = _mm_xor_si128(key, keygened);
		let mut out = [0u8; 16];
		_mm_storeu_si128(out.as_mut_ptr().cast(), key);
		out
	}
}

/// Full AES-128 key schedule: 11 round keys (the original key plus 10
/// derived rounds), the standard Intel AES-NI construction.
fn key_schedule_128(aes: Aes, key: [u8; 16]) -> [[u8; 16]; 11] {
	let k0 = key;
	let k1 = expand_key::<0x01>(aes, k0);
	let k2 = expand_key::<0x02>(aes, k1);
	let k3 = expand_key::<0x04>(aes, k2);
	let k4 = expand_key::<0x08>(aes, k3);
	let k5 = expand_key::<0x10>(aes, k4);
	let k6 = expand_key::<0x20>(aes, k5);
	let k7 = expand_key::<0x40>(aes, k6);
	let k8 = expand_key::<0x80>(aes, k7);
	let k9 = expand_key::<0x1B>(aes, k8);
	let k10 = expand_key::<0x36>(aes, k9);
	[k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]
}

fn aes128_encrypt(aes: Aes, key: [u8; 16], plaintext: [u8; 16]) -> [u8; 16] {
	let rk = key_schedule_128(aes, key);
	let mut state = xor16(plaintext, rk[0]);
	for round_key in &rk[1..10] {
		state = aes.aesenc(state, *round_key);
	}
	aes.aesenclast(state, rk[10])
}

fn aes128_decrypt(aes: Aes, key: [u8; 16], ciphertext: [u8; 16]) -> [u8; 16] {
	let rk = key_schedule_128(aes, key);
	let mut state = xor16(ciphertext, rk[10]);
	for round_key in rk[1..10].iter().rev() {
		state = aes.aesdec(state, aes.aesimc(*round_key));
	}
	aes.aesdeclast(state, rk[0])
}

/// FIPS-197 Appendix B: the canonical AES-128 encrypt test vector.
#[test]
fn aes128_encrypt_matches_fips197_appendix_b() {
	let Some(aes) = Aes::detect() else { return };
	let key: [u8; 16] = [
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	];
	let plaintext: [u8; 16] = [
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
	];
	let expect_ciphertext: [u8; 16] = [
		0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
	];
	assert_eq!(aes128_encrypt(aes, key, plaintext), expect_ciphertext);
}

/// Decrypt with `aesdec`/`aesdeclast`/`aesimc` inverts the encrypt
/// pipeline above, recovering the same FIPS-197 plaintext.
#[test]
fn aes128_decrypt_matches_fips197_appendix_b() {
	let Some(aes) = Aes::detect() else { return };
	let key: [u8; 16] = [
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	];
	let ciphertext: [u8; 16] = [
		0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
	];
	let expect_plaintext: [u8; 16] = [
		0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
	];
	assert_eq!(aes128_decrypt(aes, key, ciphertext), expect_plaintext);
}

#[test]
fn encrypt_then_decrypt_roundtrips_for_arbitrary_input() {
	let Some(aes) = Aes::detect() else { return };
	let key: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(3));
	let plaintext: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(101));
	let ciphertext = aes128_encrypt(aes, key, plaintext);
	assert_eq!(aes128_decrypt(aes, key, ciphertext), plaintext);
}

#[test]
fn aesimc_is_its_own_use_documented_transform() {
	// aesimc is InvMixColumns; applying it twice must NOT be identity
	// (MixColumns is not an involution), sanity-checking the wrapper
	// actually calls through to real hardware behavior, not a no-op.
	let Some(aes) = Aes::detect() else { return };
	let block: [u8; 16] = core::array::from_fn(|i| i as u8 + 1);
	let once = aes.aesimc(block);
	let twice = aes.aesimc(once);
	assert_ne!(once, block);
	assert_ne!(twice, once);
}
