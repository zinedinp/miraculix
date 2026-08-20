//! SSE4a (AMD, 2007): immediate `extrq`/`insertq` bitfield ops. Stable
//! `core::arch`. Token: [`Sse4a::detect`]. No reg-control forms; no
//! non-temporal stores (side-effect, not compute-return).

use core::arch::x86_64::{_mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_extracti_si64, _mm_inserti_si64};

use super::super::super::{Feature, FeatureSet};

/// Proof token: SSE4a available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sse4a(());

impl Sse4a {
	/// `None` if the CPU (every Intel CPU; AMD without SSE4a) lacks it.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse4a).then_some(Sse4a(()))
	}

	/// Extract `LEN` bits at `IDX` from `x` (`extrq`). `LEN==0` means 64.
	#[inline]
	pub fn extract_bits<const LEN: i32, const IDX: i32>(self, x: u64) -> u64 {
		unsafe { extrqi::<LEN, IDX>(x) }
	}

	/// Insert the low `LEN` bits of `src` into `dst` at `IDX`, other bits of
	/// `dst` unchanged (`insertq`). `LEN==0` means 64.
	#[inline]
	pub fn insert_bits<const LEN: i32, const IDX: i32>(self, dst: u64, src: u64) -> u64 {
		unsafe { insertqi::<LEN, IDX>(dst, src) }
	}
}

/// # Safety
/// Caller proved SSE4a via [`Sse4a`].
#[inline]
#[target_feature(enable = "sse4a")]
unsafe fn extrqi<const LEN: i32, const IDX: i32>(x: u64) -> u64 {
	let v = _mm_cvtsi64_si128(x as i64);
	let r = _mm_extracti_si64::<LEN, IDX>(v);
	_mm_cvtsi128_si64(r) as u64
}

/// # Safety
/// Caller proved SSE4a via [`Sse4a`].
#[inline]
#[target_feature(enable = "sse4a")]
unsafe fn insertqi<const LEN: i32, const IDX: i32>(dst: u64, src: u64) -> u64 {
	let vd = _mm_cvtsi64_si128(dst as i64);
	let vs = _mm_cvtsi64_si128(src as i64);
	let r = _mm_inserti_si64::<LEN, IDX>(vd, vs);
	_mm_cvtsi128_si64(r) as u64
}

#[cfg(test)]
#[path = "../../test/ops/other/sse4a.rs"]
mod tests;
