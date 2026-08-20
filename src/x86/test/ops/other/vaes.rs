use super::super::aes::Aes;
use super::*;

fn repeat_lanes<const W: usize>(lane: [u8; 16]) -> [u8; W] {
	core::array::from_fn(|i| lane[i % 16])
}

#[test]
fn aesenc_256_matches_aes_ni_per_lane() {
	let (Some(vaes), Some(aes)) = (Vaes::detect(), Aes::detect()) else { return };
	let state: [u8; 16] = core::array::from_fn(|i| i as u8);
	let round_key: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(3));
	let wide_state = repeat_lanes::<32>(state);
	let wide_key = repeat_lanes::<32>(round_key);
	let expect_lane = aes.aesenc(state, round_key);
	let got = vaes.aesenc_u8x32(wide_state, wide_key);
	assert_eq!(&got[0..16], &expect_lane[..]);
	assert_eq!(&got[16..32], &expect_lane[..]);
}

#[test]
fn aesenc_256_two_independent_lanes_match_aes_ni_each() {
	let (Some(vaes), Some(aes)) = (Vaes::detect(), Aes::detect()) else { return };
	let state0: [u8; 16] = core::array::from_fn(|i| i as u8);
	let state1: [u8; 16] = core::array::from_fn(|i| 255 - i as u8);
	let key0: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(11));
	let key1: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(13).wrapping_add(1));
	let mut wide_state = [0u8; 32];
	wide_state[0..16].copy_from_slice(&state0);
	wide_state[16..32].copy_from_slice(&state1);
	let mut wide_key = [0u8; 32];
	wide_key[0..16].copy_from_slice(&key0);
	wide_key[16..32].copy_from_slice(&key1);

	let got = vaes.aesenc_u8x32(wide_state, wide_key);
	assert_eq!(&got[0..16], &aes.aesenc(state0, key0)[..]);
	assert_eq!(&got[16..32], &aes.aesenc(state1, key1)[..]);
}

#[test]
fn aesdec_256_matches_aes_ni_per_lane() {
	let (Some(vaes), Some(aes)) = (Vaes::detect(), Aes::detect()) else { return };
	let state: [u8; 16] = core::array::from_fn(|i| i as u8 + 1);
	let round_key: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17));
	let wide_state = repeat_lanes::<32>(state);
	let wide_key = repeat_lanes::<32>(round_key);
	let expect_lane = aes.aesdec(state, round_key);
	let got = vaes.aesdec_u8x32(wide_state, wide_key);
	assert_eq!(&got[0..16], &expect_lane[..]);
	assert_eq!(&got[16..32], &expect_lane[..]);
}

#[test]
fn aesenclast_and_aesdeclast_256_match_aes_ni_per_lane() {
	let (Some(vaes), Some(aes)) = (Vaes::detect(), Aes::detect()) else { return };
	let state: [u8; 16] = core::array::from_fn(|i| i as u8);
	let round_key: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(5).wrapping_add(9));
	let wide_state = repeat_lanes::<32>(state);
	let wide_key = repeat_lanes::<32>(round_key);
	assert_eq!(&vaes.aesenclast_u8x32(wide_state, wide_key)[0..16], &aes.aesenclast(state, round_key)[..]);
	assert_eq!(&vaes.aesdeclast_u8x32(wide_state, wide_key)[0..16], &aes.aesdeclast(state, round_key)[..]);
}

#[test]
fn aes_512_four_lanes_match_aes_ni_each() {
	let (Some(vaes512), Some(aes)) = (Vaes512::detect(), Aes::detect()) else { return };
	let states: [[u8; 16]; 4] = core::array::from_fn(|lane| core::array::from_fn(|i| (i as u8).wrapping_add(lane as u8 * 17)));
	let keys: [[u8; 16]; 4] = core::array::from_fn(|lane| core::array::from_fn(|i| (i as u8).wrapping_mul(3).wrapping_add(lane as u8)));
	let mut wide_state = [0u8; 64];
	let mut wide_key = [0u8; 64];
	for lane in 0..4 {
		wide_state[lane * 16..lane * 16 + 16].copy_from_slice(&states[lane]);
		wide_key[lane * 16..lane * 16 + 16].copy_from_slice(&keys[lane]);
	}
	let got_enc = vaes512.aesenc_u8x64(wide_state, wide_key);
	let got_dec = vaes512.aesdec_u8x64(wide_state, wide_key);
	for lane in 0..4 {
		assert_eq!(&got_enc[lane * 16..lane * 16 + 16], &aes.aesenc(states[lane], keys[lane])[..], "enc lane {lane}");
		assert_eq!(&got_dec[lane * 16..lane * 16 + 16], &aes.aesdec(states[lane], keys[lane])[..], "dec lane {lane}");
	}
}
