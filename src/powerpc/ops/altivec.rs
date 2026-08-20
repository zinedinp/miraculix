//! AltiVec (1998): 128-bit vector, packed f32 (`vec_add`/`vec_sub`). Real
//! `core::arch` but nightly-only (`nightly-altivec`). Token: [`Altivec::detect`].

use core::arch::powerpc64::{vec_add, vec_sub, vec_xl, vec_xst};

use super::super::{Feature, FeatureSet};

/// Proof token: AltiVec available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Altivec(());

impl Altivec {
	/// `None` if the CPU (or the compile-time target) lacks AltiVec.
	pub fn detect() -> Option<Self> {
		FeatureSet::detect().contains(Feature::Altivec).then_some(Altivec(()))
	}

	/// `a + b` per lane (`vec_add` / `vaddfp`).
	#[inline]
	pub fn add_f32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { vaddfp(&a, &b) }
	}

	/// `a - b` per lane (`vec_sub` / `vsubfp`).
	#[inline]
	pub fn sub_f32x4(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
		unsafe { vsubfp(&a, &b) }
	}
}

/// `vec_xl` / `vec_add` / `vec_xst` (unaligned).
///
/// # Safety
/// Caller proved AltiVec via [`Altivec`].
#[inline]
#[target_feature(enable = "altivec")]
unsafe fn vaddfp(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va = vec_xl(0, a.as_ptr());
		let vb = vec_xl(0, b.as_ptr());
		let vr = vec_add(va, vb);
		let mut out = [0f32; 4];
		vec_xst(vr, 0, out.as_mut_ptr());
		out
	}
}

/// `vec_xl` / `vec_sub` / `vec_xst` (unaligned).
///
/// # Safety
/// Caller proved AltiVec via [`Altivec`].
#[inline]
#[target_feature(enable = "altivec")]
unsafe fn vsubfp(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
	unsafe {
		let va = vec_xl(0, a.as_ptr());
		let vb = vec_xl(0, b.as_ptr());
		let vr = vec_sub(va, vb);
		let mut out = [0f32; 4];
		vec_xst(vr, 0, out.as_mut_ptr());
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn add_f32x4_sums_lanes() {
		let Some(av) = Altivec::detect() else { return };
		let a = [1.0, 2.0, 3.0, 4.0];
		let b = [10.0, 20.0, 30.0, 40.0];
		assert_eq!(av.add_f32x4(a, b), [11.0, 22.0, 33.0, 44.0]);
	}

	#[test]
	fn sub_f32x4_subtracts_lanes() {
		let Some(av) = Altivec::detect() else { return };
		let a = [10.0, 20.0, 30.0, 40.0];
		let b = [1.0, 2.0, 3.0, 4.0];
		assert_eq!(av.sub_f32x4(a, b), [9.0, 18.0, 27.0, 36.0]);
	}

	/// Lanes match scalar add/sub.
	#[test]
	fn matches_scalar_on_random_lanes() {
		let Some(av) = Altivec::detect() else { return };
		let a: [f32; 4] = [17.5, -3.25, 0.0, 1e6];
		let b: [f32; 4] = [-240.75, 10.0, -3.5, 255.0];

		let mut expect_add = [0f32; 4];
		let mut expect_sub = [0f32; 4];
		for i in 0..4 {
			expect_add[i] = a[i] + b[i];
			expect_sub[i] = a[i] - b[i];
		}

		assert_eq!(av.add_f32x4(a, b), expect_add);
		assert_eq!(av.sub_f32x4(a, b), expect_sub);
	}
}
