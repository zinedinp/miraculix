//! ARMv8-A32 CRC32. Token: [`Crc32`]. Upstream:
//! `core::arch::arm::{__crc32b,h,w,d,__crc32cb,ch,cw,cd}`
//! (`stdarch_aarch32_crc32`, separate from Neon's library feature).
//! On arm: `#[target_feature(enable = "crc")]` + `enable = "v8"`. Detect:
//! [`Feature::Crc32`]. `crc32*` = IEEE 802.3; `crc32c*` = Castagnoli.

use super::super::{Feature, FeatureSet};

/// Proof that the ARMv8-A32 CRC32 extension is available. Zero-sized, `Copy`.
///
/// Obtain via [`Crc32::detect`] or [`Crc32::from_features`], then call
/// methods on the token.
#[derive(Debug, Clone, Copy)]
pub struct Crc32(());

/// One `core::arch::arm` `__crc32*` intrinsic: fold `data` into the running
/// `crc` accumulator.
macro_rules! crc32_step {
	($(#[$doc:meta])* $name:ident, $data:ty, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, crc: u32, data: $data) -> u32 {
			#[target_feature(enable = "crc")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(crc: u32, data: $data) -> u32 {
				core::arch::arm::$intrinsic(crc, data)
			}
			unsafe { imp(crc, data) }
		}
	};
}

impl Crc32 {
	/// Probe once: `Some(token)` if the CRC32 extension is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` if `Feature::Crc32` is missing.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Crc32).then_some(Crc32(()))
	}

	crc32_step!(
		/// `CRC32B`: fold one byte into `crc` (CRC-32/IEEE).
		crc32b,
		u8,
		__crc32b
	);
	crc32_step!(
		/// `CRC32H`: fold one halfword into `crc` (CRC-32/IEEE).
		crc32h,
		u16,
		__crc32h
	);
	crc32_step!(
		/// `CRC32W`: fold one word into `crc` (CRC-32/IEEE).
		crc32w,
		u32,
		__crc32w
	);
	crc32_step!(
		/// `CRC32D`: fold one doubleword into `crc` (CRC-32/IEEE).
		crc32d,
		u64,
		__crc32d
	);
	crc32_step!(
		/// `CRC32CB`: fold one byte into `crc` (CRC-32C/Castagnoli).
		crc32cb,
		u8,
		__crc32cb
	);
	crc32_step!(
		/// `CRC32CH`: fold one halfword into `crc` (CRC-32C/Castagnoli).
		crc32ch,
		u16,
		__crc32ch
	);
	crc32_step!(
		/// `CRC32CW`: fold one word into `crc` (CRC-32C/Castagnoli).
		crc32cw,
		u32,
		__crc32cw
	);
	crc32_step!(
		/// `CRC32CD`: fold one doubleword into `crc` (CRC-32C/Castagnoli).
		crc32cd,
		u64,
		__crc32cd
	);
}

#[cfg(test)]
#[path = "../test/ops/crc32.rs"]
mod tests;
