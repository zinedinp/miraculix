use super::{Aes, Sha};

fn require_aes() -> Option<Aes> {
	Aes::detect()
}

fn require_sha() -> Option<Sha> {
	Sha::detect()
}

#[test]
fn aesmc_aesimc_are_exact_inverses() {
	let Some(t) = require_aes() else { return };
	// `MixColumns`/`InvMixColumns` are exact inverse linear maps over
	// GF(2^8): this holds for *any* 16-byte input, not just a real AES
	// state, so no S-box table is needed to check it.
	for data in [[0u8; 16], [0xFFu8; 16], core::array::from_fn(|i| (i * 17) as u8)] {
		assert_eq!(t.aesimc(t.aesmc(data)), data, "aesimc(aesmc({data:?}))");
	}
}

#[test]
fn aesd_undoes_aese_with_a_zero_key() {
	let Some(t) = require_aes() else { return };
	// AESE = SubBytes(ShiftRows(data ^ key)); AESD = InvSubBytes(
	// InvShiftRows(data ^ key)). With `key = 0` both AddRoundKey steps
	// vanish, and ShiftRows/InvShiftRows + SubBytes/InvSubBytes are exact
	// inverse pairs by definition, so `aesd(aese(x, 0), 0) == x` for any
	// `x`; again no S-box table needed to check it.
	let zero = [0u8; 16];
	for data in [[0u8; 16], [0xFFu8; 16], core::array::from_fn(|i| (i * 17) as u8)] {
		assert_eq!(t.aesd(t.aese(data, zero), zero), data, "aesd(aese({data:?}, 0), 0)");
	}
}

#[test]
fn sha1h_matches_fixed_rotate() {
	let Some(t) = require_sha() else { return };
	// `SHA1H`: fixed rotate-left-by-30 (equivalently rotate-right-by-2) -
	// simple enough to check exactly without a full SHA1 reference.
	for x in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
		assert_eq!(t.sha1h(x), x.rotate_left(30), "sha1h({x:#010x})");
	}
}

/// The `sha1{c,p,m}`/`sha1su{0,1}`/`sha256h{,2}`/`sha256su{0,1}` intrinsics
/// implement multi-round SHA1/SHA256 compression steps with no simple
/// algebraic identity like AES's linear `MixColumns`: a full scalar
/// reference would need the complete SHA1/SHA256 round-constant tables
/// transcribed by hand, which is a real transcription-error risk on its
/// own. Scoped down to a smoke test instead (per the plan's explicit
/// "single-round smoke test, not a full reference" allowance): every
/// method must be deterministic (same input twice -> same output) and
/// input-sensitive (changing one operand changes the output); catches a
/// no-op/miswired/wrong-intrinsic-name bug without needing the exact
/// compression math.
macro_rules! smoke_ternop_u32x4x3 {
	($name:ident, $method:ident) => {
		#[test]
		fn $name() {
			let Some(t) = require_sha() else { return };
			let a = [1u32, 2, 3, 4];
			let b = [5u32, 6, 7, 8];
			let c = [9u32, 10, 11, 12];
			let r1 = t.$method(a, b, c);
			let r2 = t.$method(a, b, c);
			assert_eq!(r1, r2, "{} is deterministic", stringify!($method));
			assert_ne!(r1, t.$method([0, 0, 0, 0], b, c), "{} is sensitive to its first operand", stringify!($method));
		}
	};
}

macro_rules! smoke_binop_u32x4x2 {
	($name:ident, $method:ident) => {
		#[test]
		fn $name() {
			let Some(t) = require_sha() else { return };
			let a = [1u32, 2, 3, 4];
			let b = [5u32, 6, 7, 8];
			let r1 = t.$method(a, b);
			let r2 = t.$method(a, b);
			assert_eq!(r1, r2, "{} is deterministic", stringify!($method));
			assert_ne!(r1, t.$method([0, 0, 0, 0], b), "{} is sensitive to its first operand", stringify!($method));
		}
	};
}

smoke_ternop_u32x4x3!(sha1su0_smoke, sha1su0);
smoke_binop_u32x4x2!(sha1su1_smoke, sha1su1);
smoke_ternop_u32x4x3!(sha256h_smoke, sha256h);
smoke_ternop_u32x4x3!(sha256h2_smoke, sha256h2);
smoke_binop_u32x4x2!(sha256su0_smoke, sha256su0);
smoke_ternop_u32x4x3!(sha256su1_smoke, sha256su1);

#[test]
fn sha1_round_functions_are_deterministic_and_input_sensitive() {
	let Some(t) = require_sha() else { return };
	let abcd = [1u32, 2, 3, 4];
	let e = 5u32;
	let wk = [6u32, 7, 8, 9];
	for (r1, r2, r3) in [
		(t.sha1c(abcd, e, wk), t.sha1c(abcd, e, wk), t.sha1c([0, 0, 0, 0], e, wk)),
		(t.sha1p(abcd, e, wk), t.sha1p(abcd, e, wk), t.sha1p([0, 0, 0, 0], e, wk)),
		(t.sha1m(abcd, e, wk), t.sha1m(abcd, e, wk), t.sha1m([0, 0, 0, 0], e, wk)),
	] {
		assert_eq!(r1, r2, "deterministic");
		assert_ne!(r1, r3, "sensitive to hash_abcd");
	}
}
