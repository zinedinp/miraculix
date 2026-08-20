//! AVX512_4VNNIW (Knights Mill): Xeon Phi `VP4DPWSSD`/`VP4DPWSSDS` encoded
//! via `asm!`. No `target_feature` exists for this CPUID bit; detection is
//! compile-checked only and `detect()` is `None` on hosts reachable by this
//! crate. Verification is encoding-only: assembled bytes vs disassembly round-trip.

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};
use super::super::avx::avx_vnni::{vnni_acc_saturating, vnni_acc_wrapping};

/// Proof token: AVX512_4VNNIW available. Zero-sized, `Copy`.
///
/// `detect` can never return `Some` on any CPU reachable by this crate's
/// test matrix (Xeon Phi Knights Mill only, discontinued 2020): same
/// caveat class as [`super::super::other::amd3dnow::ThreeDNow`].
#[derive(Debug, Clone, Copy)]
pub struct Avx5124vnniw(());

impl Avx5124vnniw {
	/// `None` on every CPU this crate can detect on (Xeon Phi Knights Mill
	/// only; platform EOL).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx5124vnniw).then_some(Avx5124vnniw(()))
	}

	/// `src[j] + sum_n(a[n][2j]*b[2n] + a[n][2j+1]*b[2n+1])`, one native
	/// `VP4DPWSSD` (Knights Mill hardware-pipelined 4-source dot-product -
	/// the mainstream-hardware equivalent is
	/// [`super::avx512vnni::Avx512Vnni::p4dpwssd_i32x16`]).
	#[inline]
	pub fn p4dpwssd_i32x16(self, src: [i32; 16], a: [[i16; 32]; 4], b: [i16; 8]) -> [i32; 16] {
		unsafe { p4dpwssd_native(&src, &a, &b) }
	}

	/// Saturating [`Avx5124vnniw::p4dpwssd_i32x16`], one native `VP4DPWSSDS`.
	#[inline]
	pub fn p4dpwssds_i32x16(self, src: [i32; 16], a: [[i16; 32]; 4], b: [i16; 8]) -> [i32; 16] {
		unsafe { p4dpwssds_native(&src, &a, &b) }
	}

	/// [`Avx5124vnniw::p4dpwssd_i32x16`] over slices. 16-wide chunks (one
	/// `VP4DPWSSD` each), scalar remainder.
	///
	/// # Panics
	/// `out.len() != src.len()`, or any `a[n].len() != src.len() * 2`.
	pub fn p4dpwssd_i32_slice(self, src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
		p4vnniw_slice(self, src, a, b, out, false);
	}

	/// Saturating [`Avx5124vnniw::p4dpwssd_i32_slice`].
	///
	/// # Panics
	/// Same as [`Avx5124vnniw::p4dpwssd_i32_slice`].
	pub fn p4dpwssds_i32_slice(self, src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
		p4vnniw_slice(self, src, a, b, out, true);
	}
}

fn p4vnniw_slice(t: Avx5124vnniw, src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32], saturating: bool) {
	assert_eq!(out.len(), src.len());
	for an in &a {
		assert_eq!(an.len(), src.len() * 2);
	}

	let mut src_chunks = src.chunks_exact(16);
	let mut a_chunks: [_; 4] = core::array::from_fn(|n| a[n].chunks_exact(32));
	let mut out_chunks = out.chunks_exact_mut(16);

	for (sc, oc) in (&mut src_chunks).zip(&mut out_chunks) {
		let sv: [i32; 16] = sc.try_into().expect("chunks_exact width");
		let av: [[i16; 32]; 4] = core::array::from_fn(|n| {
			a_chunks[n].next().expect("chunks_exact len").try_into().expect("chunks_exact width")
		});
		let rv = if saturating { t.p4dpwssds_i32x16(sv, av, b) } else { t.p4dpwssd_i32x16(sv, av, b) };
		oc.copy_from_slice(&rv);
	}

	let src_rem = src_chunks.remainder();
	let a_rem: [&[i16]; 4] = core::array::from_fn(|n| a_chunks[n].remainder());
	let acc_fn: fn(i32, i64) -> i32 = if saturating { vnni_acc_saturating } else { vnni_acc_wrapping };
	for (i, (&sv, o)) in src_rem.iter().zip(out_chunks.into_remainder()).enumerate() {
		let mut acc = sv;
		for n in 0..4 {
			let sum: i64 = a_rem[n][2 * i] as i64 * b[2 * n] as i64 + a_rem[n][2 * i + 1] as i64 * b[2 * n + 1] as i64;
			acc = acc_fn(acc, sum);
		}
		*o = acc;
	}
}

/// # Safety
/// Caller proved AVX512_4VNNIW via [`Avx5124vnniw`]. `target_feature =
/// "avx512f"` only covers the `zmm` register class the `asm!` block uses -
/// there is no Rust-recognized feature string for the `avx512_4vnniw`
/// CPUID bit itself, so [`Avx5124vnniw::detect`] is the sole real gate.
#[target_feature(enable = "avx512f")]
unsafe fn p4dpwssd_native(src: &[i32; 16], a: &[[i16; 32]; 4], b: &[i16; 8]) -> [i32; 16] {
	let mut acc = *src;
	unsafe {
		asm!(
			"vmovdqu32 zmm0, [{acc}]",
			"vmovdqu32 zmm4, [{a0}]",
			"vmovdqu32 zmm5, [{a1}]",
			"vmovdqu32 zmm6, [{a2}]",
			"vmovdqu32 zmm7, [{a3}]",
			"vp4dpwssd zmm0, zmm4, [{b}]",
			"vmovdqu32 [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			a0 = in(reg) a[0].as_ptr(),
			a1 = in(reg) a[1].as_ptr(),
			a2 = in(reg) a[2].as_ptr(),
			a3 = in(reg) a[3].as_ptr(),
			b = in(reg) b.as_ptr(),
			out("zmm0") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
		);
	}
	acc
}

/// # Safety
/// Same as [`p4dpwssd_native`].
#[target_feature(enable = "avx512f")]
unsafe fn p4dpwssds_native(src: &[i32; 16], a: &[[i16; 32]; 4], b: &[i16; 8]) -> [i32; 16] {
	let mut acc = *src;
	unsafe {
		asm!(
			"vmovdqu32 zmm0, [{acc}]",
			"vmovdqu32 zmm4, [{a0}]",
			"vmovdqu32 zmm5, [{a1}]",
			"vmovdqu32 zmm6, [{a2}]",
			"vmovdqu32 zmm7, [{a3}]",
			"vp4dpwssds zmm0, zmm4, [{b}]",
			"vmovdqu32 [{acc}], zmm0",
			acc = in(reg) acc.as_mut_ptr(),
			a0 = in(reg) a[0].as_ptr(),
			a1 = in(reg) a[1].as_ptr(),
			a2 = in(reg) a[2].as_ptr(),
			a3 = in(reg) a[3].as_ptr(),
			b = in(reg) b.as_ptr(),
			out("zmm0") _, out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
		);
	}
	acc
}

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512_4vnniw.rs"]
mod tests;
