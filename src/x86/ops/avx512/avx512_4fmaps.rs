//! AVX512_4FMAPS (Knights Mill): Xeon Phi `V4FMADDPS`/`V4FNMADDPS` encoded
//! via `asm!`. No runtime `target_feature` exists for this CPUID bit; the
//! token's `detect()` is `None` on hosts reachable by this crate. This file
//! is compile-checked and verified by assembled-bytes vs disassembly round-trip.

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: AVX512_4FMAPS available. Zero-sized, `Copy`.
///
/// `detect` is always `None` on hosts reachable by this crate's test matrix,
/// same caveat class as `ThreeDNow`.
#[derive(Debug, Clone, Copy)]
pub struct Avx5124fmaps(());

impl Avx5124fmaps {
	/// `None` on every CPU this crate can detect on (Xeon Phi Knights Mill
	/// only; platform EOL).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx5124fmaps).then_some(Avx5124fmaps(()))
	}

	/// `a + b[0]*c[0] + b[1]*c[1] + b[2]*c[2] + b[3]*c[3]`, one native `V4FMADDPS`.
	/// Mainstream equivalent: [`super::avx512f::Avx512f::p4fmadd_f32x16`].
	#[inline]
	pub fn p4fmadd_f32x16(self, a: [f32; 16], b: [[f32; 16]; 4], c: [f32; 4]) -> [f32; 16] {
		unsafe { p4fmadd_ps_native(&a, &b, &c) }
	}

	/// `a - b[0]*c[0] - b[1]*c[1] - b[2]*c[2] - b[3]*c[3]`, one native `V4FNMADDPS`.
	#[inline]
	pub fn p4fnmadd_f32x16(self, a: [f32; 16], b: [[f32; 16]; 4], c: [f32; 4]) -> [f32; 16] {
		unsafe { p4fnmadd_ps_native(&a, &b, &c) }
	}

	/// [`Avx5124fmaps::p4fmadd_f32x16`] over slices: `out[i] = a[i] +
	/// sum_n(b[n][i] * c[n])`. 16-wide chunks (one `V4FMADDPS` each),
	/// scalar remainder.
	///
	/// # Panics
	/// `a`/`out`/every `b[n]` length mismatch.
	pub fn p4fmadd_f32_slice(self, a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
		p4fmaps_slice(a, b, c, out, |x, y, z| self.p4fmadd_f32x16(x, y, z), |x, y, z| x * y + z);
	}

	/// [`Avx5124fmaps::p4fnmadd_f32x16`] over slices: `out[i] = a[i] -
	/// sum_n(b[n][i] * c[n])`. 16-wide chunks, scalar remainder.
	///
	/// # Panics
	/// `a`/`out`/every `b[n]` length mismatch.
	pub fn p4fnmadd_f32_slice(self, a: &[f32], b: [&[f32]; 4], c: [f32; 4], out: &mut [f32]) {
		p4fmaps_slice(a, b, c, out, |x, y, z| self.p4fnmadd_f32x16(x, y, z), |x, y, z| -(x * y) + z);
	}
}

fn p4fmaps_slice(
	a: &[f32],
	b: [&[f32]; 4],
	c: [f32; 4],
	out: &mut [f32],
	fixed: impl Fn([f32; 16], [[f32; 16]; 4], [f32; 4]) -> [f32; 16],
	scalar: impl Fn(f32, f32, f32) -> f32,
) {
	assert_eq!(out.len(), a.len());
	for bn in &b {
		assert_eq!(bn.len(), a.len());
	}

	let mut a_chunks = a.chunks_exact(16);
	let mut b_chunks: [_; 4] = core::array::from_fn(|n| b[n].chunks_exact(16));
	let mut out_chunks = out.chunks_exact_mut(16);

	for (ac, oc) in (&mut a_chunks).zip(&mut out_chunks) {
		let av: [f32; 16] = ac.try_into().expect("chunks_exact width");
		let bv: [[f32; 16]; 4] = core::array::from_fn(|n| {
			b_chunks[n].next().expect("chunks_exact len").try_into().expect("chunks_exact width")
		});
		oc.copy_from_slice(&fixed(av, bv, c));
	}

	let a_rem = a_chunks.remainder();
	let b_rem: [&[f32]; 4] = core::array::from_fn(|n| b_chunks[n].remainder());
	for (i, (&av, o)) in a_rem.iter().zip(out_chunks.into_remainder()).enumerate() {
		let mut acc = av;
		for n in 0..4 {
			acc = scalar(b_rem[n][i], c[n], acc);
		}
		*o = acc;
	}
}

/// # Safety
/// Caller proved AVX512_4FMAPS via [`Avx5124fmaps`]. `target_feature =
/// "avx512f"` only covers the `zmm` register class the `asm!` block uses -
/// there is no Rust-recognized feature string for the `avx512_4fmaps`
/// CPUID bit itself, so [`Avx5124fmaps::detect`] is the sole real gate.
#[target_feature(enable = "avx512f")]
unsafe fn p4fmadd_ps_native(a: &[f32; 16], b: &[[f32; 16]; 4], c: &[f32; 4]) -> [f32; 16] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovups zmm0, [{acc}]",
			"vmovups zmm4, [{b0}]",
			"vmovups zmm5, [{b1}]",
			"vmovups zmm6, [{b2}]",
			"vmovups zmm7, [{b3}]",
			"v4fmaddps zmm0, zmm4, [{c}]",
			"vmovups [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			b0 = in(reg) b[0].as_ptr(),
			b1 = in(reg) b[1].as_ptr(),
			b2 = in(reg) b[2].as_ptr(),
			b3 = in(reg) b[3].as_ptr(),
			c = in(reg) c.as_ptr(),
			out("zmm0") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`p4fmadd_ps_native`].
#[target_feature(enable = "avx512f")]
unsafe fn p4fnmadd_ps_native(a: &[f32; 16], b: &[[f32; 16]; 4], c: &[f32; 4]) -> [f32; 16] {
	let mut acc = *a;
	unsafe {
		asm!(
			"vmovups zmm0, [{acc}]",
			"vmovups zmm4, [{b0}]",
			"vmovups zmm5, [{b1}]",
			"vmovups zmm6, [{b2}]",
			"vmovups zmm7, [{b3}]",
			"v4fnmaddps zmm0, zmm4, [{c}]",
			"vmovups [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			b0 = in(reg) b[0].as_ptr(),
			b1 = in(reg) b[1].as_ptr(),
			b2 = in(reg) b[2].as_ptr(),
			b3 = in(reg) b[3].as_ptr(),
			c = in(reg) c.as_ptr(),
			out("zmm0") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
		);
	}
	acc
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512_4fmaps.rs"]
mod tests;
