//! 3DNow! (AMD K6-2, 1998): two f32 in MMX regs via `asm!` + `femms`.
//! Long deprecated; [`ThreeDNow::detect`] is `None` on modern hosts
//! (compile-checked only). Provides basic MMX-style f32 ops.

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: 3DNow! available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ThreeDNow(());

impl ThreeDNow {
	/// `None` on any CPU without 3DNow! (every Intel CPU; AMD since Bulldozer, 2011).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::ThreeDNow).then_some(ThreeDNow(()))
	}
}

/// Binop over one 3DNow! mnemonic: `impl ThreeDNow` method plus its `asm!`
/// helper (`movq` load x2, op, `movq` store, `femms`).
macro_rules! td_binop_asm {
	($fixed_fn:ident, $asm_fn:ident, $mnemonic:literal, $fixed_doc:literal) => {
		impl ThreeDNow {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
				unsafe { $asm_fn(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved 3DNow! via [`ThreeDNow`].
		#[inline]
		unsafe fn $asm_fn(a: &[f32; 2], b: &[f32; 2]) -> [f32; 2] {
			let mut out = [0f32; 2];
			unsafe {
				asm!(
					"movq mm0, [{a}]",
					concat!($mnemonic, " mm0, [{b}]"),
					"movq [{out}], mm0",
					"femms",
					a = in(reg) a.as_ptr(),
					b = in(reg) b.as_ptr(),
					out = in(reg) out.as_mut_ptr(),
					out("mm0") _,
				);
			}
			out
		}
	};
}

td_binop_asm!(add_f32x2, pfadd, "pfadd", "`a + b` per lane (`pfadd`).");
td_binop_asm!(sub_f32x2, pfsub, "pfsub", "`a - b` per lane (`pfsub`).");
td_binop_asm!(subr_f32x2, pfsubr, "pfsubr", "`b - a` per lane, reverse subtract (`pfsubr`).");
td_binop_asm!(mul_f32x2, pfmul, "pfmul", "`a * b` per lane (`pfmul`).");
td_binop_asm!(min_f32x2, pfmin, "pfmin", "Per-lane min (`pfmin`).");
td_binop_asm!(max_f32x2, pfmax, "pfmax", "Per-lane max (`pfmax`).");
td_binop_asm!(cmpeq_f32x2, pfcmpeq, "pfcmpeq", "Lane equality mask (`pfcmpeq`): all-1s if equal, else 0.");
td_binop_asm!(cmpgt_f32x2, pfcmpgt, "pfcmpgt", "Lane greater-than mask (`pfcmpgt`): all-1s if `a[i] > b[i]`, else 0.");
td_binop_asm!(cmpge_f32x2, pfcmpge, "pfcmpge", "Lane greater-or-equal mask (`pfcmpge`): all-1s if `a[i] >= b[i]`, else 0.");
td_binop_asm!(pfacc_f32x2, pfacc, "pfacc", "Horizontal accumulate (`pfacc`): `[a[0]+a[1], b[0]+b[1]]`.");

impl ThreeDNow {
	/// Per-lane float-to-int32, truncating toward zero (`pf2id`).
	#[inline]
	pub fn to_i32x2(self, a: [f32; 2]) -> [i32; 2] {
		unsafe { pf2id(&a) }
	}

	/// Per-lane int32-to-float (`pi2fd`).
	#[inline]
	pub fn from_i32x2(self, a: [i32; 2]) -> [f32; 2] {
		unsafe { pi2fd(&a) }
	}
}

/// # Safety
/// Caller proved 3DNow! via [`ThreeDNow`].
#[inline]
unsafe fn pf2id(a: &[f32; 2]) -> [i32; 2] {
	let mut out = [0i32; 2];
	unsafe {
		asm!(
			"movq mm0, [{a}]",
			"pf2id mm0, mm0",
			"movq [{out}], mm0",
			"femms",
			a = in(reg) a.as_ptr(),
			out = in(reg) out.as_mut_ptr(),
			out("mm0") _,
		);
	}
	out
}

/// # Safety
/// Caller proved 3DNow! via [`ThreeDNow`].
#[inline]
unsafe fn pi2fd(a: &[i32; 2]) -> [f32; 2] {
	let mut out = [0f32; 2];
	unsafe {
		asm!(
			"movq mm0, [{a}]",
			"pi2fd mm0, mm0",
			"movq [{out}], mm0",
			"femms",
			a = in(reg) a.as_ptr(),
			out = in(reg) out.as_mut_ptr(),
			out("mm0") _,
		);
	}
	out
}

#[cfg(test)]
#[path = "../../test/ops/other/amd3dnow.rs"]
mod tests;
