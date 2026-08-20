//! AVX512VNNI: 512-bit narrow dot-product into `i32` (`avx512vnni`).
//! Generated via `simd_vnni_dot`. 128/256-bit companions live in `super::avx512vl`.
//! Auto dispatch cascades to VEX `AvxVnni` or scalar.

use core::arch::x86_64::{
	__m512i, _mm512_dpbusd_epi32, _mm512_dpbusds_epi32, _mm512_dpwssd_epi32, _mm512_dpwssds_epi32,
	_mm512_loadu_si512, _mm512_mask_dpbusd_epi32, _mm512_mask_dpbusds_epi32, _mm512_mask_dpwssd_epi32,
	_mm512_mask_dpwssds_epi32, _mm512_maskz_dpbusd_epi32, _mm512_maskz_dpbusds_epi32, _mm512_maskz_dpwssd_epi32,
	_mm512_maskz_dpwssds_epi32, _mm512_storeu_si512,
};

use super::super::super::{Feature, FeatureSet};
use super::super::avx::avx_vnni::{vnni_acc_saturating, vnni_acc_wrapping};
use super::super::macros::{simd_vnni_dot, simd_vnni_dot_masked};

/// Proof token: AVX512VNNI, 512-bit forms. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Vnni(());

impl Avx512Vnni {
	/// `None` if the CPU (or the compile-time target) lacks AVX512VNNI.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512vnni).then_some(Avx512Vnni(()))
	}
}

// Plain EVEX names only (not `_avx` / INT8 / INT16). Intel order: (src, a, b).

simd_vnni_dot! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	fixed_fn = dpbusd_i32x16, slice_fn = dpbusd_i32_slice, intrinsic_fn = dpbusd_i32x16_intrinsic,
	width = 16, group = 4, a_elem = u8, b_elem = i8,
	vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_dpbusd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[4j+k] as i32 * b[4j+k] as i32)` (`vpdpbusd`, 512-bit, `u8`x`i8`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[4j+k]*b[4j+k])`. 16-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	fixed_fn = dpbusds_i32x16, slice_fn = dpbusds_i32_slice, intrinsic_fn = dpbusds_i32x16_intrinsic,
	width = 16, group = 4, a_elem = u8, b_elem = i8,
	vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_dpbusds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512Vnni::dpbusd_i32x16`] (`vpdpbusds`, 512-bit).",
	slice_doc = "Saturating [`Avx512Vnni::dpbusd_i32_slice`]. 16-wide chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	fixed_fn = dpwssd_i32x16, slice_fn = dpwssd_i32_slice, intrinsic_fn = dpwssd_i32x16_intrinsic,
	width = 16, group = 2, a_elem = i16, b_elem = i16,
	vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_dpwssd_epi32, acc = vnni_acc_wrapping,
	fixed_doc = "`dst[j] = src[j] + sum_k(a[2j+k] as i32 * b[2j+k] as i32)` (`vpdpwssd`, 512-bit, `i16`x`i16`).",
	slice_doc = "`out[j] = src[j] + sum_k(a[2j+k]*b[2j+k])`. 16-wide `src`/`out` chunks, software scalar rem.",
}

simd_vnni_dot! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	fixed_fn = dpwssds_i32x16, slice_fn = dpwssds_i32_slice, intrinsic_fn = dpwssds_i32x16_intrinsic,
	width = 16, group = 2, a_elem = i16, b_elem = i16,
	vec = __m512i, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	intrinsic = _mm512_dpwssds_epi32, acc = vnni_acc_saturating,
	fixed_doc = "Saturating [`Avx512Vnni::dpwssd_i32x16`] (`vpdpwssds`, 512-bit).",
	slice_doc = "Saturating [`Avx512Vnni::dpwssd_i32_slice`]. 16-wide chunks, software scalar rem.",
}

// Merge/zero-masked forms. `p4dpwssd`/`p4dpwssds` get none: they're this
// crate's own software 4-iteration composition over `dpwssd`/`dpwssds`, not a
// hardware instruction, so there's no `_mm512_mask_*` to wrap.
simd_vnni_dot_masked! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	merge_fn = dpbusd_i32x16_merge_masked, zero_fn = dpbusd_i32x16_zero_masked,
	merge_intrinsic_fn = mask_dpbusd_i32x16_intrinsic, zero_intrinsic_fn = maskz_dpbusd_i32x16_intrinsic,
	width = 16, group = 4, a_elem = u8, b_elem = i8,
	vec = __m512i, mask = u16, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_dpbusd_epi32, zero_intrinsic = _mm512_maskz_dpbusd_epi32,
	merge_doc = "[`Avx512Vnni::dpbusd_i32x16`] where `mask` bit is set, else copied from `src` (`vpdpbusd`, merge-masked).",
	zero_doc = "[`Avx512Vnni::dpbusd_i32x16`] where `mask` bit is set, else zero (`vpdpbusd`, zero-masked). `src` is still a real input here, not just a merge fallback - the dot-product-accumulate is computed for every lane before masking zeroes the unselected ones.",
}

simd_vnni_dot_masked! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	merge_fn = dpbusds_i32x16_merge_masked, zero_fn = dpbusds_i32x16_zero_masked,
	merge_intrinsic_fn = mask_dpbusds_i32x16_intrinsic, zero_intrinsic_fn = maskz_dpbusds_i32x16_intrinsic,
	width = 16, group = 4, a_elem = u8, b_elem = i8,
	vec = __m512i, mask = u16, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_dpbusds_epi32, zero_intrinsic = _mm512_maskz_dpbusds_epi32,
	merge_doc = "[`Avx512Vnni::dpbusds_i32x16`] where `mask` bit is set, else copied from `src` (`vpdpbusds`, merge-masked).",
	zero_doc = "[`Avx512Vnni::dpbusds_i32x16`] where `mask` bit is set, else zero (`vpdpbusds`, zero-masked).",
}

simd_vnni_dot_masked! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	merge_fn = dpwssd_i32x16_merge_masked, zero_fn = dpwssd_i32x16_zero_masked,
	merge_intrinsic_fn = mask_dpwssd_i32x16_intrinsic, zero_intrinsic_fn = maskz_dpwssd_i32x16_intrinsic,
	width = 16, group = 2, a_elem = i16, b_elem = i16,
	vec = __m512i, mask = u16, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_dpwssd_epi32, zero_intrinsic = _mm512_maskz_dpwssd_epi32,
	merge_doc = "[`Avx512Vnni::dpwssd_i32x16`] where `mask` bit is set, else copied from `src` (`vpdpwssd`, merge-masked).",
	zero_doc = "[`Avx512Vnni::dpwssd_i32x16`] where `mask` bit is set, else zero (`vpdpwssd`, zero-masked).",
}

simd_vnni_dot_masked! {
	token = Avx512Vnni, target_feature = "avx512vnni",
	merge_fn = dpwssds_i32x16_merge_masked, zero_fn = dpwssds_i32x16_zero_masked,
	merge_intrinsic_fn = mask_dpwssds_i32x16_intrinsic, zero_intrinsic_fn = maskz_dpwssds_i32x16_intrinsic,
	width = 16, group = 2, a_elem = i16, b_elem = i16,
	vec = __m512i, mask = u16, loadu = _mm512_loadu_si512, storeu = _mm512_storeu_si512,
	merge_intrinsic = _mm512_mask_dpwssds_epi32, zero_intrinsic = _mm512_maskz_dpwssds_epi32,
	merge_doc = "[`Avx512Vnni::dpwssds_i32x16`] where `mask` bit is set, else copied from `src` (`vpdpwssds`, merge-masked).",
	zero_doc = "[`Avx512Vnni::dpwssds_i32x16`] where `mask` bit is set, else zero (`vpdpwssds`, zero-masked).",
}

impl Avx512Vnni {
	/// `VP4DPWSSD`: 4-way `i16`x`i16` dot-product-accumulate, folded in
	/// sequence. `b[2n]`/`b[2n+1]` are the weight pair broadcast against
	/// `a[n]` on iteration `n`; wrapping accumulate (bit-exact vs a real
	/// 4VNNIW part: wrapping add is associative mod 2^32, so per-step vs.
	/// single-shot truncation cannot differ).
	#[inline]
	pub fn p4dpwssd_i32x16(self, src: [i32; 16], a: [[i16; 32]; 4], b: [i16; 8]) -> [i32; 16] {
		let mut acc = src;
		for n in 0..4 {
			let bvec: [i16; 32] = core::array::from_fn(|i| b[2 * n + i % 2]);
			acc = self.dpwssd_i32x16(acc, a[n], bvec);
		}
		acc
	}

	/// Saturating [`Avx512Vnni::p4dpwssd_i32x16`] (`VP4DPWSSDS`): each of the
	/// 4 folded steps saturates independently, matching the per-iteration
	/// saturation the instruction's own "4 iterations" name describes.
	#[inline]
	pub fn p4dpwssds_i32x16(self, src: [i32; 16], a: [[i16; 32]; 4], b: [i16; 8]) -> [i32; 16] {
		let mut acc = src;
		for n in 0..4 {
			let bvec: [i16; 32] = core::array::from_fn(|i| b[2 * n + i % 2]);
			acc = self.dpwssds_i32x16(acc, a[n], bvec);
		}
		acc
	}

	/// [`Avx512Vnni::p4dpwssd_i32x16`] over slices: `out[j] = src[j] +
	/// sum_n(a[n][2j]*b[2n] + a[n][2j+1]*b[2n+1])`. 16-wide chunks, scalar
	/// remainder.
	///
	/// # Panics
	/// `out.len() != src.len()`, or any `a[n].len() != src.len() * 2`.
	pub fn p4dpwssd_i32_slice(self, src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
		p4dpwssd_slice_impl(self, src, a, b, out, false);
	}

	/// Saturating [`Avx512Vnni::p4dpwssd_i32_slice`].
	///
	/// # Panics
	/// `out.len() != src.len()`, or any `a[n].len() != src.len() * 2`.
	pub fn p4dpwssds_i32_slice(self, src: &[i32], a: [&[i16]; 4], b: [i16; 8], out: &mut [i32]) {
		p4dpwssd_slice_impl(self, src, a, b, out, true);
	}
}

fn p4dpwssd_slice_impl(
	t: Avx512Vnni,
	src: &[i32],
	a: [&[i16]; 4],
	b: [i16; 8],
	out: &mut [i32],
	saturating: bool,
) {
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

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512vnni.rs"]
mod tests;
