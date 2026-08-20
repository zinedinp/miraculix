//! FMA4 (AMD Bulldozer, 2011): four-operand FMA (`vfmaddps`, dest free).
//! No `core::arch` intrinsics; `asm!` exception. Dead post-Excavator.
//! Token: [`Fma4::detect`].

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: FMA4 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Fma4(());

impl Fma4 {
	/// `None` on any CPU without FMA4 (every Intel CPU; AMD since ~2015).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Fma4).then_some(Fma4(()))
	}

	/// `a * b + c` per lane (`vfmaddps`).
	#[inline]
	pub fn fmadd_f32x4(self, a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
		unsafe { vfmaddps(&a, &b, &c) }
	}
}

/// `vfmaddps` via unaligned `movups`.
///
/// # Safety
/// Caller proved FMA4 via [`Fma4`].
#[inline]
unsafe fn vfmaddps(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> [f32; 4] {
	let mut out = [0f32; 4];
	unsafe {
		asm!(
			"movups xmm1, [{a}]",
			"movups xmm2, [{b}]",
			"movups xmm3, [{c}]",
			"vfmaddps xmm0, xmm1, xmm2, xmm3",
			"movups [{out}], xmm0",
			a = in(reg) a.as_ptr(),
			b = in(reg) b.as_ptr(),
			c = in(reg) c.as_ptr(),
			out = in(reg) out.as_mut_ptr(),
			out("xmm0") _,
			out("xmm1") _,
			out("xmm2") _,
			out("xmm3") _,
		);
	}
	out
}

#[cfg(test)]
#[path = "../../test/ops/other/fma4.rs"]
mod tests;
