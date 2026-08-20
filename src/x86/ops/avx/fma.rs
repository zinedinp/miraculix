//! FMA3: fused f32/f64 fmadd/fmsub/fnmadd/fnmsub at 128/256-bit (`"fma"`). Token: [`Fma`].
//! AVX-512 FMA lives under `super::super::avx512`.

use core::arch::x86_64::{
	__m128, __m128d, __m256, __m256d, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_fmsub_pd, _mm256_fmsub_ps,
	_mm256_fnmadd_pd, _mm256_fnmadd_ps, _mm256_fnmsub_pd, _mm256_fnmsub_ps, _mm256_loadu_pd, _mm256_loadu_ps,
	_mm256_storeu_pd, _mm256_storeu_ps, _mm_fmadd_pd, _mm_fmadd_ps, _mm_fmsub_pd, _mm_fmsub_ps, _mm_fnmadd_pd,
	_mm_fnmadd_ps, _mm_fnmsub_pd, _mm_fnmsub_ps, _mm_loadu_pd, _mm_loadu_ps, _mm_storeu_pd, _mm_storeu_ps,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::macros::simd_ternop;

/// Proof token: FMA3 available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Fma(());

impl Fma {
	/// `None` if the CPU (or the compile-time target) lacks FMA3.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Fma).then_some(Fma(()))
	}

	/// From resolved tier (`V3`/`V4` list `Feature::Fma`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Fma).then_some(Fma(()))
	}
}

// 128-bit: `vis = pub` (auto uses only 256-bit forms).
macro_rules! fma_f32x4_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Fma, vis = pub, target_feature = "fma",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! fma_f32x8_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Fma, target_feature = "fma",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = f32, vec = __m256, loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! fma_f64x2_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Fma, vis = pub, target_feature = "fma",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 2, elem = f64, vec = __m128d, loadu = _mm_loadu_pd, storeu = _mm_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! fma_f64x4_ternop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_ternop! {
			token = Fma, target_feature = "fma",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = f64, vec = __m256d, loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

fma_f32x4_ternop!(
	fmadd_f32x4, fmadd_f32x4_slice, vfmaddps128, _mm_fmadd_ps, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddps`, 128-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 4-wide chunks, scalar `a*b+c` remainder."
);
fma_f32x8_ternop!(
	fmadd_f32x8, fmadd_f32x8_slice, vfmaddps256, _mm256_fmadd_ps, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddps`, 256-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 8-wide chunks, scalar `a*b+c` remainder."
);
fma_f64x2_ternop!(
	fmadd_f64x2, fmadd_f64x2_slice, vfmaddpd128, _mm_fmadd_pd, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddpd`, 128-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 2-wide chunks, scalar `a*b+c` remainder."
);
fma_f64x4_ternop!(
	fmadd_f64x4, fmadd_f64x4_slice, vfmaddpd256, _mm256_fmadd_pd, |a, b, c| a * b + c,
	"`a * b + c` per lane, fused (`vfmaddpd`, 256-bit).",
	"`out[i] = a[i] * b[i] + c[i]` (HW fused). 4-wide chunks, scalar `a*b+c` remainder."
);

fma_f32x4_ternop!(
	fmsub_f32x4, fmsub_f32x4_slice, vfmsubps128, _mm_fmsub_ps, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubps`, 128-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 4-wide chunks, scalar `a*b-c` remainder."
);
fma_f32x8_ternop!(
	fmsub_f32x8, fmsub_f32x8_slice, vfmsubps256, _mm256_fmsub_ps, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubps`, 256-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 8-wide chunks, scalar `a*b-c` remainder."
);
fma_f64x2_ternop!(
	fmsub_f64x2, fmsub_f64x2_slice, vfmsubpd128, _mm_fmsub_pd, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubpd`, 128-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 2-wide chunks, scalar `a*b-c` remainder."
);
fma_f64x4_ternop!(
	fmsub_f64x4, fmsub_f64x4_slice, vfmsubpd256, _mm256_fmsub_pd, |a, b, c| a * b - c,
	"`a * b - c` per lane, fused (`vfmsubpd`, 256-bit).",
	"`out[i] = a[i] * b[i] - c[i]` (HW fused). 4-wide chunks, scalar `a*b-c` remainder."
);

fma_f32x4_ternop!(
	fnmadd_f32x4, fnmadd_f32x4_slice, vfnmaddps128, _mm_fnmadd_ps, |a: f32, b: f32, c: f32| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddps`, 128-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 4-wide chunks, scalar remainder."
);
fma_f32x8_ternop!(
	fnmadd_f32x8, fnmadd_f32x8_slice, vfnmaddps256, _mm256_fnmadd_ps, |a: f32, b: f32, c: f32| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddps`, 256-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
fma_f64x2_ternop!(
	fnmadd_f64x2, fnmadd_f64x2_slice, vfnmaddpd128, _mm_fnmadd_pd, |a: f64, b: f64, c: f64| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddpd`, 128-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 2-wide chunks, scalar remainder."
);
fma_f64x4_ternop!(
	fnmadd_f64x4, fnmadd_f64x4_slice, vfnmaddpd256, _mm256_fnmadd_pd, |a: f64, b: f64, c: f64| -(a * b) + c,
	"`-(a * b) + c` per lane, fused (`vfnmaddpd`, 256-bit).",
	"`out[i] = -(a[i] * b[i]) + c[i]` (HW fused). 4-wide chunks, scalar remainder."
);

fma_f32x4_ternop!(
	fnmsub_f32x4, fnmsub_f32x4_slice, vfnmsubps128, _mm_fnmsub_ps, |a: f32, b: f32, c: f32| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubps`, 128-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 4-wide chunks, scalar remainder."
);
fma_f32x8_ternop!(
	fnmsub_f32x8, fnmsub_f32x8_slice, vfnmsubps256, _mm256_fnmsub_ps, |a: f32, b: f32, c: f32| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubps`, 256-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 8-wide chunks, scalar remainder."
);
fma_f64x2_ternop!(
	fnmsub_f64x2, fnmsub_f64x2_slice, vfnmsubpd128, _mm_fnmsub_pd, |a: f64, b: f64, c: f64| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubpd`, 128-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 2-wide chunks, scalar remainder."
);
fma_f64x4_ternop!(
	fnmsub_f64x4, fnmsub_f64x4_slice, vfnmsubpd256, _mm256_fnmsub_pd, |a: f64, b: f64, c: f64| -(a * b) - c,
	"`-(a * b) - c` per lane, fused (`vfnmsubpd`, 256-bit).",
	"`out[i] = -(a[i] * b[i]) - c[i]` (HW fused). 4-wide chunks, scalar remainder."
);

#[cfg(test)]
#[path = "../../test/ops/avx/fma.rs"]
mod tests;
