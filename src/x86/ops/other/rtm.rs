//! RTM (TSX, 2013): `_xbegin`/`_xend`/`_xtest` control flow, not lane ops.
//! Nightly (`nightly-rtm`). HLE has no `core::arch`. Token: [`Rtm::detect`].

use core::arch::x86_64::{_XBEGIN_STARTED, _xbegin, _xend, _xtest};

use super::super::super::{Feature, FeatureSet};

/// Proof token: RTM available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Rtm(());

impl Rtm {
	/// `None` if RTM missing or fused off (2021 erratum).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Rtm).then_some(Rtm(()))
	}

	/// Begin transaction. `Ok(())` or `Err(abort_status)` (`_XABORT_*`).
	#[inline]
	pub fn xbegin(self) -> Result<(), u32> {
		let status = unsafe { xbegin_raw() };
		if status == _XBEGIN_STARTED { Ok(()) } else { Err(status) }
	}

	/// Commit open transaction. Unmatched `xend` is a logic fault (`#GP`), not UB.
	#[inline]
	pub fn xend(self) {
		unsafe { xend_raw() }
	}

	/// `true` inside an RTM/HLE region.
	#[inline]
	pub fn xtest(self) -> bool {
		unsafe { xtest_raw() }
	}
}

/// # Safety
/// Caller proved RTM via [`Rtm`].
#[inline]
#[target_feature(enable = "rtm")]
unsafe fn xbegin_raw() -> u32 {
	unsafe { _xbegin() }
}

/// # Safety
/// Caller proved RTM via [`Rtm`] and has an open transaction from [`Rtm::xbegin`].
#[inline]
#[target_feature(enable = "rtm")]
unsafe fn xend_raw() {
	unsafe { _xend() }
}

/// # Safety
/// Caller proved RTM via [`Rtm`].
#[inline]
#[target_feature(enable = "rtm")]
unsafe fn xtest_raw() -> bool {
	unsafe { _xtest() != 0 }
}

#[cfg(test)]
#[path = "../../test/ops/other/rtm.rs"]
mod tests;
