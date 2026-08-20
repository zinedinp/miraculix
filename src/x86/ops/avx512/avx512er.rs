//! AVX512ER: Phi-only reciprocal/rsqrt/exp2 approximations encoded via `asm!`.
//! No Rust `target_feature` exists for this CPUID bit; detection is
//! compile-checked only and `detect()` is `None` on current hosts.

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: AVX512ER available. Zero-sized, `Copy`.
///
/// `detect` can never return `Some` on any CPU reachable by this crate's
/// test matrix (Xeon Phi Knights Landing/Knights Mill only, discontinued
/// 2020): same caveat class as [`super::super::other::amd3dnow::ThreeDNow`].
#[derive(Debug, Clone, Copy)]
pub struct Avx512Er(());

impl Avx512Er {
	/// `None` on every CPU this crate can detect on (Xeon Phi only; platform
	/// EOL).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512er).then_some(Avx512Er(()))
	}

	/// Approximate `1.0 / a[i]`, relative error `<= 2^-28`, one native
	/// `VRCP28PS`.
	#[inline]
	pub fn rcp28_f32x16(self, a: [f32; 16]) -> [f32; 16] {
		unsafe { rcp28_ps_native(&a) }
	}

	/// Approximate `1.0 / sqrt(a[i])`, relative error `<= 2^-28`, one native
	/// `VRSQRT28PS`.
	#[inline]
	pub fn rsqrt28_f32x16(self, a: [f32; 16]) -> [f32; 16] {
		unsafe { rsqrt28_ps_native(&a) }
	}

	/// Approximate `2^a[i]`, relative error `<= 2^-23`, one native `VEXP2PS`.
	#[inline]
	pub fn exp2_f32x16(self, a: [f32; 16]) -> [f32; 16] {
		unsafe { exp2_ps_native(&a) }
	}

	/// [`Avx512Er::rcp28_f32x16`] in `f64`, 8-wide, one native `VRCP28PD`.
	#[inline]
	pub fn rcp28_f64x8(self, a: [f64; 8]) -> [f64; 8] {
		unsafe { rcp28_pd_native(&a) }
	}

	/// [`Avx512Er::rsqrt28_f32x16`] in `f64`, 8-wide, one native `VRSQRT28PD`.
	#[inline]
	pub fn rsqrt28_f64x8(self, a: [f64; 8]) -> [f64; 8] {
		unsafe { rsqrt28_pd_native(&a) }
	}

	/// [`Avx512Er::exp2_f32x16`] in `f64`, 8-wide, one native `VEXP2PD`.
	#[inline]
	pub fn exp2_f64x8(self, a: [f64; 8]) -> [f64; 8] {
		unsafe { exp2_pd_native(&a) }
	}
}

/// # Safety
/// Caller proved AVX512ER via [`Avx512Er`]. `target_feature = "avx512f"`
/// only covers the `zmm` register class the `asm!` block uses: there is no
/// Rust-recognized feature string for the `avx512er` CPUID bit itself, so
/// [`Avx512Er::detect`] is the sole real gate.
#[target_feature(enable = "avx512f")]
unsafe fn rcp28_ps_native(a: &[f32; 16]) -> [f32; 16] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovups zmm0, [{acc}]",
			"vrcp28ps zmm0, zmm0",
			"vmovups [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`rcp28_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn rsqrt28_ps_native(a: &[f32; 16]) -> [f32; 16] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovups zmm0, [{acc}]",
			"vrsqrt28ps zmm0, zmm0",
			"vmovups [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`rcp28_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn exp2_ps_native(a: &[f32; 16]) -> [f32; 16] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovups zmm0, [{acc}]",
			"vexp2ps zmm0, zmm0",
			"vmovups [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`rcp28_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn rcp28_pd_native(a: &[f64; 8]) -> [f64; 8] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovupd zmm0, [{acc}]",
			"vrcp28pd zmm0, zmm0",
			"vmovupd [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`rcp28_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn rsqrt28_pd_native(a: &[f64; 8]) -> [f64; 8] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovupd zmm0, [{acc}]",
			"vrsqrt28pd zmm0, zmm0",
			"vmovupd [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`rcp28_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn exp2_pd_native(a: &[f64; 8]) -> [f64; 8] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovupd zmm0, [{acc}]",
			"vexp2pd zmm0, zmm0",
			"vmovupd [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			out("zmm0") _,
		);
	}
	acc
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512er.rs"]
mod tests;
