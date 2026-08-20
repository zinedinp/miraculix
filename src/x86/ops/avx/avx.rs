//! AVX (2011): 256-bit YMM float ops (`"avx"`). Token: [`Avx::detect`] / [`Avx::from_level`].
//! Supports f32/f64 arithmetic, bitwise, and compare masks; integer 256-bit is in [`super::avx2`].

use core::arch::x86_64::{
	__m256, __m256d, __m256i, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _mm256_add_pd,
	_mm256_add_ps, _mm256_addsub_pd, _mm256_addsub_ps, _mm256_and_pd, _mm256_and_ps, _mm256_andnot_pd,
	_mm256_andnot_ps, _mm256_castpd_si256, _mm256_castps_si256, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_div_pd,
	_mm256_div_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_maskload_pd, _mm256_maskload_ps,
	_mm256_maskstore_pd, _mm256_maskstore_ps, _mm256_max_pd, _mm256_max_ps, _mm256_min_pd, _mm256_min_ps,
	_mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_movemask_pd, _mm256_movemask_ps, _mm256_mul_pd,
	_mm256_mul_ps, _mm256_or_pd, _mm256_or_ps, _mm256_permute2f128_ps, _mm256_permute_pd, _mm256_permute_ps,
	_mm256_rcp_ps, _mm256_rsqrt_ps, _mm256_shuffle_ps, _mm256_sqrt_pd, _mm256_sqrt_ps, _mm256_storeu_pd,
	_mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps, _mm256_unpackhi_pd, _mm256_unpackhi_ps, _mm256_unpacklo_pd,
	_mm256_unpacklo_ps, _mm256_xor_pd, _mm256_xor_ps,
};

use super::super::super::{Feature, FeatureSet, GenericLevel};
use super::super::macros::{simd_binop, simd_binop_fixed, simd_binop_imm_fixed, simd_movemask, simd_unop_fixed};

/// Proof token: AVX available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx(());

impl Avx {
	/// `None` if the CPU (or the compile-time target) lacks AVX.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From resolved tier (`V3`/`V4` list `Feature::Avx`); no new CPUID.
	pub fn from_level(level: GenericLevel) -> Option<Self> {
		level.required_features().contains(&Feature::Avx).then_some(Avx(()))
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx).then_some(Avx(()))
	}
}

macro_rules! avx_f32_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx, target_feature = "avx",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 8, elem = f32, vec = __m256, loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

macro_rules! avx_f64_binop {
	($fixed_fn:ident, $slice_fn:ident, $intrinsic_fn:ident, $intrinsic:path, $scalar:expr, $fixed_doc:literal, $slice_doc:literal) => {
		simd_binop! {
			token = Avx, target_feature = "avx",
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = 4, elem = f64, vec = __m256d, loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd,
			intrinsic = $intrinsic, scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

avx_f32_binop!(
	add_f32x8, add_f32_slice, addps, _mm256_add_ps, |x, y| x + y,
	"`a + b` per lane (`vaddps`, 256-bit).",
	"`out[i] = a[i] + b[i]`. 8-wide `add_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	sub_f32x8, sub_f32_slice, subps, _mm256_sub_ps, |x, y| x - y,
	"`a - b` per lane (`vsubps`, 256-bit).",
	"`out[i] = a[i] - b[i]`. 8-wide `sub_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	mul_f32x8, mul_f32_slice, mulps, _mm256_mul_ps, |x, y| x * y,
	"`a * b` per lane (`vmulps`, 256-bit).",
	"`out[i] = a[i] * b[i]`. 8-wide `mul_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	div_f32x8, div_f32_slice, divps, _mm256_div_ps, |x, y| x / y,
	"`a / b` per lane (`vdivps`, 256-bit).",
	"`out[i] = a[i] / b[i]`. 8-wide `div_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	min_f32x8, min_f32_slice, minps, _mm256_min_ps, |x, y| x.min(y),
	"Per-lane min (`vminps`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f32::min`.",
	"`out[i] = min(a[i], b[i])`. 8-wide `min_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	max_f32x8, max_f32_slice, maxps, _mm256_max_ps, |x, y| x.max(y),
	"Per-lane max (`vmaxps`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f32::max`.",
	"`out[i] = max(a[i], b[i])`. 8-wide `max_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	and_f32x8, and_f32_slice, andps, _mm256_and_ps, |x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()),
	"Bitwise AND per lane (`vandps`, 256-bit).",
	"`out[i] = a[i] & b[i]` bitwise. 8-wide `and_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	or_f32x8, or_f32_slice, orps, _mm256_or_ps, |x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()),
	"Bitwise OR per lane (`vorps`, 256-bit).",
	"`out[i] = a[i] | b[i]` bitwise. 8-wide `or_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	xor_f32x8, xor_f32_slice, xorps, _mm256_xor_ps, |x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()),
	"Bitwise XOR per lane (`vxorps`, 256-bit).",
	"`out[i] = a[i] ^ b[i]` bitwise. 8-wide `xor_f32x8` chunks, scalar remainder."
);
avx_f32_binop!(
	andnot_f32x8, andnot_f32_slice, andnps, _mm256_andnot_ps, |x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()),
	"Bitwise `!a & b` per lane (`vandnps`, 256-bit).",
	"`out[i] = !a[i] & b[i]` bitwise. 8-wide `andnot_f32x8` chunks, scalar remainder."
);

avx_f64_binop!(
	add_f64x4, add_f64_slice, addpd, _mm256_add_pd, |x, y| x + y,
	"`a + b` per lane (`vaddpd`, 256-bit).",
	"`out[i] = a[i] + b[i]`. 4-wide `add_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	sub_f64x4, sub_f64_slice, subpd, _mm256_sub_pd, |x, y| x - y,
	"`a - b` per lane (`vsubpd`, 256-bit).",
	"`out[i] = a[i] - b[i]`. 4-wide `sub_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	mul_f64x4, mul_f64_slice, mulpd, _mm256_mul_pd, |x, y| x * y,
	"`a * b` per lane (`vmulpd`, 256-bit).",
	"`out[i] = a[i] * b[i]`. 4-wide `mul_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	div_f64x4, div_f64_slice, divpd, _mm256_div_pd, |x, y| x / y,
	"`a / b` per lane (`vdivpd`, 256-bit).",
	"`out[i] = a[i] / b[i]`. 4-wide `div_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	min_f64x4, min_f64_slice, minpd, _mm256_min_pd, |x, y| x.min(y),
	"Per-lane min (`vminpd`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f64::min`.",
	"`out[i] = min(a[i], b[i])`. 4-wide `min_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	max_f64x4, max_f64_slice, maxpd, _mm256_max_pd, |x, y| x.max(y),
	"Per-lane max (`vmaxpd`, 256-bit). NaN: second-operand-on-NaN, not IEEE `f64::max`.",
	"`out[i] = max(a[i], b[i])`. 4-wide `max_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	and_f64x4, and_f64_slice, andpd, _mm256_and_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()),
	"Bitwise AND per lane (`vandpd`, 256-bit).",
	"`out[i] = a[i] & b[i]` bitwise. 4-wide `and_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	or_f64x4, or_f64_slice, orpd, _mm256_or_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()),
	"Bitwise OR per lane (`vorpd`, 256-bit).",
	"`out[i] = a[i] | b[i]` bitwise. 4-wide `or_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	xor_f64x4, xor_f64_slice, xorpd, _mm256_xor_pd, |x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()),
	"Bitwise XOR per lane (`vxorpd`, 256-bit).",
	"`out[i] = a[i] ^ b[i]` bitwise. 4-wide `xor_f64x4` chunks, scalar remainder."
);
avx_f64_binop!(
	andnot_f64x4, andnot_f64_slice, andnpd, _mm256_andnot_pd, |x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()),
	"Bitwise `!a & b` per lane (`vandnpd`, 256-bit).",
	"`out[i] = !a[i] & b[i]` bitwise. 4-wide `andnot_f64x4` chunks, scalar remainder."
);

// Compare wrappers: `_mm256_cmp_*` needs a const IMM (OQ = ordered quiet).
#[inline]
unsafe fn vcmpeq_ps(a: __m256, b: __m256) -> __m256 {
	unsafe { _mm256_cmp_ps::<{ _CMP_EQ_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmplt_ps(a: __m256, b: __m256) -> __m256 {
	unsafe { _mm256_cmp_ps::<{ _CMP_LT_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmple_ps(a: __m256, b: __m256) -> __m256 {
	unsafe { _mm256_cmp_ps::<{ _CMP_LE_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmpgt_ps(a: __m256, b: __m256) -> __m256 {
	unsafe { _mm256_cmp_ps::<{ _CMP_GT_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmpge_ps(a: __m256, b: __m256) -> __m256 {
	unsafe { _mm256_cmp_ps::<{ _CMP_GE_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmpeq_pd(a: __m256d, b: __m256d) -> __m256d {
	unsafe { _mm256_cmp_pd::<{ _CMP_EQ_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmplt_pd(a: __m256d, b: __m256d) -> __m256d {
	unsafe { _mm256_cmp_pd::<{ _CMP_LT_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmple_pd(a: __m256d, b: __m256d) -> __m256d {
	unsafe { _mm256_cmp_pd::<{ _CMP_LE_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmpgt_pd(a: __m256d, b: __m256d) -> __m256d {
	unsafe { _mm256_cmp_pd::<{ _CMP_GT_OQ }>(a, b) }
}
#[inline]
unsafe fn vcmpge_pd(a: __m256d, b: __m256d) -> __m256d {
	unsafe { _mm256_cmp_pd::<{ _CMP_GE_OQ }>(a, b) }
}

avx_f32_binop!(
	cmpeq_f32x8, cmpeq_f32_slice, vcmpeqps, vcmpeq_ps,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane equality mask (`vcmpps` EQ_OQ, 256-bit): all-1s bits if equal, else 0.",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx_f32_binop!(
	cmplt_f32x8, cmplt_f32_slice, vcmpltps, vcmplt_ps,
	|x, y| if x < y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-than mask (`vcmpps` LT_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx_f32_binop!(
	cmple_f32x8, cmple_f32_slice, vcmpleps, vcmple_ps,
	|x, y| if x <= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane less-equal mask (`vcmpps` LE_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx_f32_binop!(
	cmpgt_f32x8, cmpgt_f32_slice, vcmpgtps, vcmpgt_ps,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-than mask (`vcmpps` GT_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx_f32_binop!(
	cmpge_f32x8, cmpge_f32_slice, vcmpgeps, vcmpge_ps,
	|x, y| if x >= y { f32::from_bits(!0) } else { f32::from_bits(0) },
	"Lane greater-equal mask (`vcmpps` GE_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 8-wide chunks, scalar remainder."
);
avx_f64_binop!(
	cmpeq_f64x4, cmpeq_f64_slice, vcmpeqpd, vcmpeq_pd,
	|x, y| if x == y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane equality mask (`vcmppd` EQ_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]==b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx_f64_binop!(
	cmplt_f64x4, cmplt_f64_slice, vcmpltpd, vcmplt_pd,
	|x, y| if x < y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-than mask (`vcmppd` LT_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]<b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx_f64_binop!(
	cmple_f64x4, cmple_f64_slice, vcmplepd, vcmple_pd,
	|x, y| if x <= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane less-equal mask (`vcmppd` LE_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]<=b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx_f64_binop!(
	cmpgt_f64x4, cmpgt_f64_slice, vcmpgtpd, vcmpgt_pd,
	|x, y| if x > y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-than mask (`vcmppd` GT_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]>b[i] else 0`. 4-wide chunks, scalar remainder."
);
avx_f64_binop!(
	cmpge_f64x4, cmpge_f64_slice, vcmpgepd, vcmpge_pd,
	|x, y| if x >= y { f64::from_bits(!0) } else { f64::from_bits(0) },
	"Lane greater-equal mask (`vcmppd` GE_OQ, 256-bit).",
	"`out[i] = all-1s bits if a[i]>=b[i] else 0`. 4-wide chunks, scalar remainder."
);

simd_movemask! {
	token = Avx, target_feature = "avx",
	fixed_fn = movemask_f32x8, intrinsic_fn = movemask_ps,
	width = 8, elem = f32, vec = __m256, mask = u8,
	loadu = _mm256_loadu_ps, intrinsic = _mm256_movemask_ps,
	doc = "Sign-bit mask, one bit per lane (`vmovmskps`).",
}
simd_movemask! {
	token = Avx, target_feature = "avx",
	fixed_fn = movemask_f64x4, intrinsic_fn = movemask_pd,
	width = 4, elem = f64, vec = __m256d, mask = u8,
	loadu = _mm256_loadu_pd, intrinsic = _mm256_movemask_pd,
	doc = "Sign-bit mask, one bit per lane (`vmovmskpd`). Low 4 bits meaningful, rest 0.",
}

simd_unop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = sqrt_f32x8, intrinsic_fn = sqrtps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_sqrt_ps,
	fixed_doc = "Correctly-rounded per-lane sqrt (`vsqrtps`, 256-bit). Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = rcp_f32x8, intrinsic_fn = rcpps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_rcp_ps,
	fixed_doc = "Approximate per-lane reciprocal (`vrcpps`, 256-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = rsqrt_f32x8, intrinsic_fn = rsqrtps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_rsqrt_ps,
	fixed_doc = "Approximate per-lane reciprocal sqrt (`vrsqrtps`, 256-bit), max relative error < 1.5*2^-12. Fixed-width only, see module doc.",
}
simd_unop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = sqrt_f64x4, intrinsic_fn = sqrtpd256,
	width = 4, elem = f64, vec = __m256d,
	loadu = _mm256_loadu_pd, storeu = _mm256_storeu_pd, intrinsic = _mm256_sqrt_pd,
	fixed_doc = "Correctly-rounded per-lane sqrt (`vsqrtpd`, 256-bit). Fixed-width only, see module doc.",
}

// Structural (not elementwise) 256-bit float ops: no honest per-lane scalar
// reference (result lane depends on lane *position*, not just `a[i]`/`b[i]`),
// so fixed-width only, same reasoning as the DQ extract/insert family.
simd_binop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = unpacklo_f32x8, intrinsic_fn = unpacklo_ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_unpacklo_ps,
	fixed_doc = "Interleaves the low half of each 128-bit lane of `a`/`b` (`vunpcklps`, 256-bit).",
}
simd_binop_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = unpackhi_f32x8, intrinsic_fn = unpackhi_ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_unpackhi_ps,
	fixed_doc = "Interleaves the high half of each 128-bit lane of `a`/`b` (`vunpckhps`, 256-bit).",
}
simd_binop_imm_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = shuffle_f32x8, intrinsic_fn = shuffle_ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_shuffle_ps,
	fixed_doc = "Per-128-bit-lane 4-way shuffle of `a`/`b` by `IMM8` (`vshufps`, 256-bit).",
}
simd_binop_imm_fixed! {
	token = Avx, target_feature = "avx",
	fixed_fn = permute2f128_f32x8, intrinsic_fn = permute2f128_ps256,
	width = 8, elem = f32, vec = __m256,
	loadu = _mm256_loadu_ps, storeu = _mm256_storeu_ps, intrinsic = _mm256_permute2f128_ps,
	fixed_doc = "Selects one 128-bit half from `a` and one from `b` by `IMM8` (`vperm2f128`, 256-bit).",
}

// Complex f32/f64, interleaved `[re0, im0, re1, im1, ...]` layout, 2 pairs
// per 128-bit lane. Same AP-15 network as `Sse3`'s complex ops (see that
// module's doc), widened to 256-bit: `permute_ps`/`permute_pd` apply the
// swap-immediate per 128-bit lane so the 2-complex-per-lane shape carries
// over unchanged.
const COMPLEX_SWAP_PAIRS_F32: i32 = 0b10_11_00_01;
const COMPLEX_SWAP_PAIRS_F64: i32 = 0b0101;
const COMPLEX_CONJ_SIGN_F32X8: [f32; 8] = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
const COMPLEX_CONJ_SIGN_F64X4: [f64; 4] = [0.0, -0.0, 0.0, -0.0];
/// `mul_c*_intrinsic(conj=true)` negates `moveldup`/`unpacklo`'s broadcast of
/// `a.im` (present in *both* lanes of a pair), so it needs an all-lanes
/// negation, not the alternating `COMPLEX_CONJ_SIGN_*` pattern.
const COMPLEX_NEGATE_ALL_F32X8: [f32; 8] = [-0.0; 8];
const COMPLEX_NEGATE_ALL_F64X4: [f64; 4] = [-0.0; 4];

impl Avx {
	/// Negate the imaginary lane of each complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c32x8(self, a: [f32; 8]) -> [f32; 8] {
		unsafe { conj_c32x8_intrinsic(&a) }
	}

	/// Complex multiply per pair: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c32x8(self, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
		unsafe { mul_c32x8_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b` per pair, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c32x8(self, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
		unsafe { mul_c32x8_intrinsic(&a, &b, true) }
	}

	/// `|a|^2` per pair, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c32x8(self, a: [f32; 8]) -> [f32; 8] {
		unsafe { abs2_c32x8_intrinsic(&a) }
	}

	/// Negate the imaginary lane of each complex pair (`a.re + i*a.im -> a.re - i*a.im`).
	#[inline]
	pub fn conj_c64x4(self, a: [f64; 4]) -> [f64; 4] {
		unsafe { conj_c64x4_intrinsic(&a) }
	}

	/// Complex multiply per pair: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
	#[inline]
	pub fn mul_c64x4(self, a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
		unsafe { mul_c64x4_intrinsic(&a, &b, false) }
	}

	/// `conj(a) * b` per pair, fused (no separate conjugate pass).
	#[inline]
	pub fn conj_mul_c64x4(self, a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
		unsafe { mul_c64x4_intrinsic(&a, &b, true) }
	}

	/// `|a|^2` per pair, broadcast to both re and im lanes: `a.re*a.re + a.im*a.im`.
	#[inline]
	pub fn abs2_c64x4(self, a: [f64; 4]) -> [f64; 4] {
		unsafe { abs2_c64x4_intrinsic(&a) }
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn conj_c32x8_intrinsic(a: &[f32; 8]) -> [f32; 8] {
	unsafe {
		let va = _mm256_loadu_ps(a.as_ptr());
		let sign = _mm256_loadu_ps(COMPLEX_CONJ_SIGN_F32X8.as_ptr());
		let vr = _mm256_xor_ps(va, sign);
		let mut out = [0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// `conj` selects the negated-`b` conjugate-multiply variant instead of a separate pre-pass.
///
/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn mul_c32x8_intrinsic(a: &[f32; 8], b: &[f32; 8], conj: bool) -> [f32; 8] {
	unsafe {
		let ab = _mm256_loadu_ps(a.as_ptr());
		let xy = _mm256_loadu_ps(b.as_ptr());
		let yx = _mm256_permute_ps::<COMPLEX_SWAP_PAIRS_F32>(xy);
		let aa = _mm256_moveldup_ps(ab);
		let mut bb = _mm256_movehdup_ps(ab);
		if conj {
			let sign = _mm256_loadu_ps(COMPLEX_NEGATE_ALL_F32X8.as_ptr());
			bb = _mm256_xor_ps(bb, sign);
		}
		let t1 = _mm256_mul_ps(aa, xy);
		let t2 = _mm256_mul_ps(bb, yx);
		let vr = _mm256_addsub_ps(t1, t2);
		let mut out = [0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn abs2_c32x8_intrinsic(a: &[f32; 8]) -> [f32; 8] {
	unsafe {
		let va = _mm256_loadu_ps(a.as_ptr());
		let sqr = _mm256_mul_ps(va, va);
		let sqr_rev = _mm256_permute_ps::<COMPLEX_SWAP_PAIRS_F32>(sqr);
		let vr = _mm256_add_ps(sqr, sqr_rev);
		let mut out = [0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn conj_c64x4_intrinsic(a: &[f64; 4]) -> [f64; 4] {
	unsafe {
		let va = _mm256_loadu_pd(a.as_ptr());
		let sign = _mm256_loadu_pd(COMPLEX_CONJ_SIGN_F64X4.as_ptr());
		let vr = _mm256_xor_pd(va, sign);
		let mut out = [0f64; 4];
		_mm256_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn mul_c64x4_intrinsic(a: &[f64; 4], b: &[f64; 4], conj: bool) -> [f64; 4] {
	unsafe {
		let ab = _mm256_loadu_pd(a.as_ptr());
		let xy = _mm256_loadu_pd(b.as_ptr());
		let yx = _mm256_permute_pd::<COMPLEX_SWAP_PAIRS_F64>(xy);
		let aa = _mm256_unpacklo_pd(ab, ab);
		let mut bb = _mm256_unpackhi_pd(ab, ab);
		if conj {
			let sign = _mm256_loadu_pd(COMPLEX_NEGATE_ALL_F64X4.as_ptr());
			bb = _mm256_xor_pd(bb, sign);
		}
		let t1 = _mm256_mul_pd(aa, xy);
		let t2 = _mm256_mul_pd(bb, yx);
		let vr = _mm256_addsub_pd(t1, t2);
		let mut out = [0f64; 4];
		_mm256_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn abs2_c64x4_intrinsic(a: &[f64; 4]) -> [f64; 4] {
	unsafe {
		let va = _mm256_loadu_pd(a.as_ptr());
		let sqr = _mm256_mul_pd(va, va);
		let sqr_rev = _mm256_permute_pd::<COMPLEX_SWAP_PAIRS_F64>(sqr);
		let vr = _mm256_add_pd(sqr, sqr_rev);
		let mut out = [0f64; 4];
		_mm256_storeu_pd(out.as_mut_ptr(), vr);
		out
	}
}

// Partial (ragged-tail) load/store, AVX's `VMASKMOVPS`/`VMASKMOVPD`. Unlike
// AVX-512's k-mask (a compact bitfield, one bit per lane), `VMASKMOV`'s mask
// is a full-width *vector*: only the MSB of each lane's 32/64-bit element is
// read, and there's no scalar "first n bits" trick to build it, so "first
// n lanes active" is built here as a real per-lane compare (lane-index
// constant `<` broadcast `n`), matching how every other constant-vector in
// this file is loaded (array + `loadu`, no `_mm256_set1_*` import needed).
// Fault-suppression on masked-off lanes is architecturally the same
// guarantee AVX-512's masked load/store gives (see `Avx512f`'s
// `partial_load_f32x16` doc): safe to read/write a `slice` shorter than
// the width.
const PARTIAL_LANE_IDX_F32X8: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
const PARTIAL_LANE_IDX_F64X4: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

impl Avx {
	/// Loads `slice.len().min(8)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_f32x8(self, slice: &[f32]) -> [f32; 8] {
		unsafe { partial_load_f32x8_intrinsic(slice.as_ptr(), slice.len().min(8) as u32) }
	}

	/// Writes `slice.len().min(8)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_f32x8(self, v: [f32; 8], slice: &mut [f32]) {
		unsafe { partial_store_f32x8_intrinsic(slice.as_mut_ptr(), slice.len().min(8) as u32, &v) }
	}

	/// Loads `slice.len().min(4)` elements from the front of `slice`, zero-padding the rest.
	#[inline]
	pub fn partial_load_f64x4(self, slice: &[f64]) -> [f64; 4] {
		unsafe { partial_load_f64x4_intrinsic(slice.as_ptr(), slice.len().min(4) as u32) }
	}

	/// Writes `slice.len().min(4)` elements of `v` to the front of `slice`; `v`'s remaining lanes are ignored.
	#[inline]
	pub fn partial_store_f64x4(self, v: [f64; 4], slice: &mut [f64]) {
		unsafe { partial_store_f64x4_intrinsic(slice.as_mut_ptr(), slice.len().min(4) as u32, &v) }
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_mask_f32x8(n: u32) -> __m256i {
	unsafe {
		let idx = _mm256_loadu_ps(PARTIAL_LANE_IDX_F32X8.as_ptr());
		let n_bcast = [n as f32; 8];
		let n_bcast = _mm256_loadu_ps(n_bcast.as_ptr());
		_mm256_castps_si256(_mm256_cmp_ps::<{ _CMP_LT_OQ }>(idx, n_bcast))
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_load_f32x8_intrinsic(ptr: *const f32, n: u32) -> [f32; 8] {
	unsafe {
		let mask = partial_mask_f32x8(n);
		let v = _mm256_maskload_ps(ptr, mask);
		let mut out = [0f32; 8];
		_mm256_storeu_ps(out.as_mut_ptr(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_store_f32x8_intrinsic(ptr: *mut f32, n: u32, v: &[f32; 8]) {
	unsafe {
		let mask = partial_mask_f32x8(n);
		let vv = _mm256_loadu_ps(v.as_ptr());
		_mm256_maskstore_ps(ptr, mask, vv);
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`].
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_mask_f64x4(n: u32) -> __m256i {
	unsafe {
		let idx = _mm256_loadu_pd(PARTIAL_LANE_IDX_F64X4.as_ptr());
		let n_bcast = [n as f64; 4];
		let n_bcast = _mm256_loadu_pd(n_bcast.as_ptr());
		_mm256_castpd_si256(_mm256_cmp_pd::<{ _CMP_LT_OQ }>(idx, n_bcast))
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_load_f64x4_intrinsic(ptr: *const f64, n: u32) -> [f64; 4] {
	unsafe {
		let mask = partial_mask_f64x4(n);
		let v = _mm256_maskload_pd(ptr, mask);
		let mut out = [0f64; 4];
		_mm256_storeu_pd(out.as_mut_ptr(), v);
		out
	}
}

/// # Safety
/// Caller proved AVX via [`Avx`]. `n`'s set-mask lanes must not exceed `ptr`'s valid element count.
#[inline]
#[target_feature(enable = "avx")]
unsafe fn partial_store_f64x4_intrinsic(ptr: *mut f64, n: u32, v: &[f64; 4]) {
	unsafe {
		let mask = partial_mask_f64x4(n);
		let vv = _mm256_loadu_pd(v.as_ptr());
		_mm256_maskstore_pd(ptr, mask, vv);
	}
}

#[cfg(test)]
#[path = "../../test/ops/avx/avx.rs"]
mod tests;
