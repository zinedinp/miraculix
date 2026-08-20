//! SSE: 128-bit XMM packed f32. Stable `core::arch`. Token: [`Sse::detect`].
//! Uses `simd_binop` for arithmetic, bitwise, and compare masks. `sqrtps`/
//! `rcpps`/`rsqrtps` are fixed-width only (see `simd_unop_fixed` doc): the
//! HW op needs no libm, but a `_slice` remainder loop would.

use core::arch::x86_64::{
	__m128, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_cmpeq_ps, _mm_cmpge_ps, _mm_cmpgt_ps, _mm_cmple_ps,
	_mm_cmplt_ps, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_movemask_ps, _mm_mul_ps, _mm_or_ps,
	_mm_rcp_ps, _mm_rsqrt_ps, _mm_sqrt_ps, _mm_storeu_ps, _mm_sub_ps, _mm_xor_ps,
};

use super::super::super::{Feature, FeatureSet};
use super::super::macros::{simd_binop, simd_movemask, simd_unop_fixed};

/// Proof token: SSE available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Sse(());

impl Sse {
	/// `None` if the CPU (or the compile-time target) lacks SSE.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Sse).then_some(Sse(()))
	}

	/// x86_64 ABI always has SSE/SSE2; skip CPUID. 32-bit x86 must use [`detect`].
	#[cfg(target_arch = "x86_64")]
	pub(crate) fn assume_baseline() -> Self {
		Sse(())
	}
}

macro_rules! sse_f32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Sse, target_feature = "sse",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

sse_f32_binop!(
	add_f32x4, add_f32_slice, addps, _mm_add_ps, |x, y| x + y,
	"`a + b` per lane (`addps`).",
	"`out[i] = a[i] + b[i]`. 4-wide `add_f32x4` chunks, scalar remainder."
);
sse_f32_binop!(
	sub_f32x4, sub_f32_slice, subps, _mm_sub_ps, |x, y| x - y,
	"`a - b` per lane (`subps`).",
	"`out[i] = a[i] - b[i]`. 4-wide `sub_f32x4` chunks, scalar remainder."
);
sse_f32_binop!(
	mul_f32x4, mul_f32_slice, mulps, _mm_mul_ps, |x, y| x * y,
	"`a * b` per lane (`mulps`).",
	"`out[i] = a[i] * b[i]`. 4-wide `mul_f32x4` chunks, scalar remainder."
);
sse_f32_binop!(
	div_f32x4, div_f32_slice, divps, _mm_div_ps, |x, y| x / y,
	"`a / b` per lane (`divps`).",
	"`out[i] = a[i] / b[i]`. 4-wide `div_f32x4` chunks, scalar remainder."
);
sse_f32_binop!(
	min_f32x4, min_f32_slice, minps, _mm_min_ps, |x, y| x.min(y),
	"Per-lane min (`minps`). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])`. 4-wide `min_f32x4` chunks, scalar remainder."
);
sse_f32_binop!(
	max_f32x4, max_f32_slice, maxps, _mm_max_ps, |x, y| x.max(y),
	"Per-lane max (`maxps`). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])`. 4-wide `max_f32x4` chunks, scalar remainder."
);
// `vis = pub`: no auto_up cascade calls these (float bitwise has no cross-tier
// dispatch entry), so the slice methods are their own public entry point, same
// reasoning as `avx2.rs`'s i8/u8/i16/u16 family.
simd_binop! {
	token = Sse, vis = pub, target_feature = "sse",
	fixed_fn = and_f32x4, slice_fn = and_f32_slice, intrinsic_fn = andps,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_and_ps, scalar = |x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()),
	fixed_doc = "`a & b` per lane, bitwise (`andps`).",
	slice_doc = "`out[i] = a[i] & b[i]` (bitwise). 4-wide chunks, scalar remainder.",
}
simd_binop! {
	token = Sse, vis = pub, target_feature = "sse",
	fixed_fn = or_f32x4, slice_fn = or_f32_slice, intrinsic_fn = orps,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_or_ps, scalar = |x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()),
	fixed_doc = "`a | b` per lane, bitwise (`orps`).",
	slice_doc = "`out[i] = a[i] | b[i]` (bitwise). 4-wide chunks, scalar remainder.",
}
simd_binop! {
	token = Sse, vis = pub, target_feature = "sse",
	fixed_fn = xor_f32x4, slice_fn = xor_f32_slice, intrinsic_fn = xorps,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_xor_ps, scalar = |x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()),
	fixed_doc = "`a ^ b` per lane, bitwise (`xorps`).",
	slice_doc = "`out[i] = a[i] ^ b[i]` (bitwise). 4-wide chunks, scalar remainder.",
}
simd_binop! {
	token = Sse, vis = pub, target_feature = "sse",
	fixed_fn = andnot_f32x4, slice_fn = andnot_f32_slice, intrinsic_fn = andnps,
	width = 4, elem = f32, vec = __m128, loadu = _mm_loadu_ps, storeu = _mm_storeu_ps,
	intrinsic = _mm_andnot_ps, scalar = |x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()),
	fixed_doc = "`!a & b` per lane, bitwise (`andnps`).",
	slice_doc = "`out[i] = !a[i] & b[i]` (bitwise). 4-wide chunks, scalar remainder.",
}

// Lane compare masks as all-1s/0s bit patterns (not bool). True lanes have MSB set.
sse_f32_binop!(
	cmpeq_f32x4, cmpeq_f32_slice, cmpeqps, _mm_cmpeq_ps,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane equality mask (`cmpeqps`): all-1s bits if equal, else 0. NaN never equals.",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder."
);
sse_f32_binop!(
	cmplt_f32x4, cmplt_f32_slice, cmpltps, _mm_cmplt_ps,
	|x, y| if x < y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-than mask (`cmpltps`): all-1s bits if `a<b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 4-wide chunks, scalar remainder."
);
sse_f32_binop!(
	cmple_f32x4, cmple_f32_slice, cmpleps, _mm_cmple_ps,
	|x, y| if x <= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-equal mask (`cmpleps`): all-1s bits if `a<=b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 4-wide chunks, scalar remainder."
);
sse_f32_binop!(
	cmpgt_f32x4, cmpgt_f32_slice, cmpgtps, _mm_cmpgt_ps,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-than mask (`cmpgtps`): all-1s bits if `a>b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 4-wide chunks, scalar remainder."
);
sse_f32_binop!(
	cmpge_f32x4, cmpge_f32_slice, cmpgeps, _mm_cmpge_ps,
	|x, y| if x >= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-equal mask (`cmpgeps`): all-1s bits if `a>=b`, else 0. False if either NaN.",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 4-wide chunks, scalar remainder."
);

simd_movemask! {
	token = Sse, target_feature = "sse",
	fixed_fn = movemask_f32x4, intrinsic_fn = movemask_ps,
	width = 4, elem = f32, vec = __m128, mask = u8,
	loadu = _mm_loadu_ps, intrinsic = _mm_movemask_ps,
	doc = "Sign-bit mask, one bit per lane (`movmskps`). Low 4 bits meaningful, rest 0.",
}

// Fixed-width only (no `_slice`/`auto`): the HW op needs no libm, but a
// `_slice` remainder closure would need `f32::sqrt`, unavailable under
// `no_std` without an external libm dependency.
simd_unop_fixed! {
	token = Sse, target_feature = "sse",
	fixed_fn = sqrt_f32x4, intrinsic_fn = sqrtps,
	width = 4, elem = f32, vec = __m128,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps, intrinsic = _mm_sqrt_ps,
	fixed_doc = "Correctly-rounded per-lane sqrt (`sqrtps`). Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Sse, target_feature = "sse",
	fixed_fn = rcp_f32x4, intrinsic_fn = rcpps,
	width = 4, elem = f32, vec = __m128,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps, intrinsic = _mm_rcp_ps,
	fixed_doc = "Approximate per-lane reciprocal (`rcpps`), max relative error < 1.5*2^-12. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Sse, target_feature = "sse",
	fixed_fn = rsqrt_f32x4, intrinsic_fn = rsqrtps,
	width = 4, elem = f32, vec = __m128,
	loadu = _mm_loadu_ps, storeu = _mm_storeu_ps, intrinsic = _mm_rsqrt_ps,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`rsqrtps`), max relative error < 1.5*2^-12. Fixed-width only, see module doc.",
}

#[cfg(test)]
#[path = "../../test/ops/sse/sse.rs"]
mod tests;
