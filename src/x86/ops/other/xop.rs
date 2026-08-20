//! XOP (AMD Bulldozer, 2011): packed rotate `vprotd`. Same `asm!` gap as
//! [`super::fma4`]. Dead post-Excavator. Token: [`Xop::detect`].

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: XOP available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Xop(());

impl Xop {
	/// `None` on any CPU without XOP (every Intel CPU; AMD since ~2015).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Xop).then_some(Xop(()))
	}

	/// Per-lane left rotate of `a` by `counts` (mod 32) (`vprotd`, variable form).
	#[inline]
	pub fn rotl_u32x4(self, a: [u32; 4], counts: [u32; 4]) -> [u32; 4] {
		unsafe { vprotd(&a, &counts) }
	}
}

/// `vprotd` via unaligned `movups`.
///
/// # Safety
/// Caller proved XOP via [`Xop`].
#[inline]
unsafe fn vprotd(a: &[u32; 4], counts: &[u32; 4]) -> [u32; 4] {
	let mut out = [0u32; 4];
	unsafe {
		asm!(
			"movups xmm1, [{a}]",
			"movups xmm2, [{counts}]",
			"vprotd xmm0, xmm1, xmm2",
			"movups [{out}], xmm0",
			a = in(reg) a.as_ptr(),
			counts = in(reg) counts.as_ptr(),
			out = in(reg) out.as_mut_ptr(),
			out("xmm0") _,
			out("xmm1") _,
			out("xmm2") _,
		);
	}
	out
}

#[cfg(test)]
#[path = "../../test/ops/other/xop.rs"]
mod tests;
