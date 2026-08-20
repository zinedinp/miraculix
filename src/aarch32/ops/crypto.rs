//! ARMv8-A32 Crypto: AES + SHA1/SHA256 (alongside [`super::crc32`]).
//! Tokens: [`Aes`], [`Sha`]. Same `stdarch_arm_neon_intrinsics` as Neon
//! (no separate CRC-style feature). One [`Sha`] token: LLVM uses a single
//! `"sha2"` feature for SHA1 and SHA256; [`Sha::from_features`] requires
//! both HWCAP bits. Stacked features: neon(+v7) for load/store, aes/sha2(+v8)
//! for the crypto op.

use super::super::{Feature, FeatureSet};

/// Proof that the ARMv8-A32 AES extension is available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Aes(());

/// Proof that the ARMv8-A32 SHA1+SHA256 extension is available (see the
/// module doc for why this is one token). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sha(());

macro_rules! crypto_binop_u8x16 {
	($(#[$doc:meta])* $name:ident, $extra:literal, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
			#[target_feature(enable = "neon")]
			#[target_feature(enable = $extra)]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
				let av = unsafe { core::arch::arm::vld1q_u8(a.as_ptr()) };
				let bv = unsafe { core::arch::arm::vld1q_u8(b.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av, bv);
				let mut out = [0u8; 16];
				unsafe { core::arch::arm::vst1q_u8(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b) }
		}
	};
}

macro_rules! crypto_unop_u8x16 {
	($(#[$doc:meta])* $name:ident, $extra:literal, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u8; 16]) -> [u8; 16] {
			#[target_feature(enable = "neon")]
			#[target_feature(enable = $extra)]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(a: [u8; 16]) -> [u8; 16] {
				let av = unsafe { core::arch::arm::vld1q_u8(a.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av);
				let mut out = [0u8; 16];
				unsafe { core::arch::arm::vst1q_u8(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a) }
		}
	};
}

impl Aes {
	/// Probe once: `Some(token)` if the AES extension is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from a set you already have (e.g.
	/// [`crate::aarch32::FeatureSet::detect`]).
	///
	/// Returns `None` if `Feature::Aes` is missing.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Aes).then_some(Aes(()))
	}

	crypto_binop_u8x16!(
		/// `AESE`: one AES single-round encryption step (`data` XOR round
		/// key, then `SubBytes`+`ShiftRows`: no `MixColumns`, chain with
		/// [`Aes::aesmc`] for a full round).
		aese,
		"aes",
		vaeseq_u8
	);
	crypto_binop_u8x16!(
		/// `AESD`: one AES single-round decryption step (inverse of
		/// [`Aes::aese`]: `InvShiftRows`+`InvSubBytes`, then `data` XOR
		/// round key).
		aesd,
		"aes",
		vaesdq_u8
	);
	crypto_unop_u8x16!(
		/// `AESMC`: AES `MixColumns` step.
		aesmc,
		"aes",
		vaesmcq_u8
	);
	crypto_unop_u8x16!(
		/// `AESIMC`: AES `InvMixColumns` step.
		aesimc,
		"aes",
		vaesimcq_u8
	);
}

impl Sha {
	/// Probe once: `Some(token)` if SHA1+SHA256 are available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from a set you already have (e.g.
	/// [`crate::aarch32::FeatureSet::detect`]).
	///
	/// Returns `None` unless both `Feature::Sha1` and `Feature::Sha2` are present.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		(set.contains(Feature::Sha1) && set.contains(Feature::Sha2)).then_some(Sha(()))
	}

	/// `SHA1H`: SHA1 fixed rotate (rotate left by 30, expressed upstream as
	/// a right-rotate-by-2; takes/returns a plain scalar `u32`, no vector
	/// load/store needed).
	#[inline]
	pub fn sha1h(self, hash_e: u32) -> u32 {
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_e: u32) -> u32 {
			core::arch::arm::vsha1h_u32(hash_e)
		}
		unsafe { imp(hash_e) }
	}

	/// One SHA1 "choose" round (`f = ch`): `hash_abcd`, `hash_e` (scalar),
	/// `wk` (message schedule word + round constant) -> next `hash_abcd`.
	#[inline]
	pub fn sha1c(self, hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
			let abcd = unsafe { core::arch::arm::vld1q_u32(hash_abcd.as_ptr()) };
			let wkv = unsafe { core::arch::arm::vld1q_u32(wk.as_ptr()) };
			let rv = core::arch::arm::vsha1cq_u32(abcd, hash_e, wkv);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(hash_abcd, hash_e, wk) }
	}
	/// One SHA1 "parity" round (`f = parity`), same shape as [`Sha::sha1c`].
	#[inline]
	pub fn sha1p(self, hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
			let abcd = unsafe { core::arch::arm::vld1q_u32(hash_abcd.as_ptr()) };
			let wkv = unsafe { core::arch::arm::vld1q_u32(wk.as_ptr()) };
			let rv = core::arch::arm::vsha1pq_u32(abcd, hash_e, wkv);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(hash_abcd, hash_e, wk) }
	}
	/// One SHA1 "majority" round (`f = maj`), same shape as [`Sha::sha1c`].
	#[inline]
	pub fn sha1m(self, hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_abcd: [u32; 4], hash_e: u32, wk: [u32; 4]) -> [u32; 4] {
			let abcd = unsafe { core::arch::arm::vld1q_u32(hash_abcd.as_ptr()) };
			let wkv = unsafe { core::arch::arm::vld1q_u32(wk.as_ptr()) };
			let rv = core::arch::arm::vsha1mq_u32(abcd, hash_e, wkv);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(hash_abcd, hash_e, wk) }
	}

	/// `SHA1SU0`: SHA1 message schedule update, part 1 of 2 (three 128-bit
	/// message-word groups in, next schedule state out).
	#[inline]
	pub fn sha1su0(self, w0_3: [u32; 4], w4_7: [u32; 4], w8_11: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(w0_3: [u32; 4], w4_7: [u32; 4], w8_11: [u32; 4]) -> [u32; 4] {
			let a = unsafe { core::arch::arm::vld1q_u32(w0_3.as_ptr()) };
			let b = unsafe { core::arch::arm::vld1q_u32(w4_7.as_ptr()) };
			let c = unsafe { core::arch::arm::vld1q_u32(w8_11.as_ptr()) };
			let rv = core::arch::arm::vsha1su0q_u32(a, b, c);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(w0_3, w4_7, w8_11) }
	}
	/// `SHA1SU1`: SHA1 message schedule update, part 2 of 2.
	#[inline]
	pub fn sha1su1(self, tw0_3: [u32; 4], w12_15: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(tw0_3: [u32; 4], w12_15: [u32; 4]) -> [u32; 4] {
			let a = unsafe { core::arch::arm::vld1q_u32(tw0_3.as_ptr()) };
			let b = unsafe { core::arch::arm::vld1q_u32(w12_15.as_ptr()) };
			let rv = core::arch::arm::vsha1su1q_u32(a, b);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(tw0_3, w12_15) }
	}

	/// `SHA256H`: SHA256 hash round, first half (`abcd`).
	#[inline]
	pub fn sha256h(self, hash_abcd: [u32; 4], hash_efgh: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_abcd: [u32; 4], hash_efgh: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
			let abcd = unsafe { core::arch::arm::vld1q_u32(hash_abcd.as_ptr()) };
			let efgh = unsafe { core::arch::arm::vld1q_u32(hash_efgh.as_ptr()) };
			let wkv = unsafe { core::arch::arm::vld1q_u32(wk.as_ptr()) };
			let rv = core::arch::arm::vsha256hq_u32(abcd, efgh, wkv);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(hash_abcd, hash_efgh, wk) }
	}
	/// `SHA256H2`: SHA256 hash round, second half (`efgh`). Same parameter
	/// order as [`Sha::sha256h`] (`hash_abcd, hash_efgh, wk`; verified
	/// against stdarch source, the two intrinsics are *not* mirrored).
	#[inline]
	pub fn sha256h2(self, hash_abcd: [u32; 4], hash_efgh: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(hash_abcd: [u32; 4], hash_efgh: [u32; 4], wk: [u32; 4]) -> [u32; 4] {
			let abcd = unsafe { core::arch::arm::vld1q_u32(hash_abcd.as_ptr()) };
			let efgh = unsafe { core::arch::arm::vld1q_u32(hash_efgh.as_ptr()) };
			let wkv = unsafe { core::arch::arm::vld1q_u32(wk.as_ptr()) };
			let rv = core::arch::arm::vsha256h2q_u32(abcd, efgh, wkv);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(hash_abcd, hash_efgh, wk) }
	}

	/// `SHA256SU0`: SHA256 message schedule update, part 1 of 2.
	#[inline]
	pub fn sha256su0(self, w0_3: [u32; 4], w4_7: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(w0_3: [u32; 4], w4_7: [u32; 4]) -> [u32; 4] {
			let a = unsafe { core::arch::arm::vld1q_u32(w0_3.as_ptr()) };
			let b = unsafe { core::arch::arm::vld1q_u32(w4_7.as_ptr()) };
			let rv = core::arch::arm::vsha256su0q_u32(a, b);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(w0_3, w4_7) }
	}
	/// `SHA256SU1`: SHA256 message schedule update, part 2 of 2.
	#[inline]
	pub fn sha256su1(self, tw0_3: [u32; 4], w8_11: [u32; 4], w12_15: [u32; 4]) -> [u32; 4] {
		#[target_feature(enable = "neon")]
		#[target_feature(enable = "sha2")]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
		#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
		unsafe fn imp(tw0_3: [u32; 4], w8_11: [u32; 4], w12_15: [u32; 4]) -> [u32; 4] {
			let a = unsafe { core::arch::arm::vld1q_u32(tw0_3.as_ptr()) };
			let b = unsafe { core::arch::arm::vld1q_u32(w8_11.as_ptr()) };
			let c = unsafe { core::arch::arm::vld1q_u32(w12_15.as_ptr()) };
			let rv = core::arch::arm::vsha256su1q_u32(a, b, c);
			let mut out = [0u32; 4];
			unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
			out
		}
		unsafe { imp(tw0_3, w8_11, w12_15) }
	}
}

#[cfg(test)]
#[path = "../test/ops/crypto.rs"]
mod tests;
