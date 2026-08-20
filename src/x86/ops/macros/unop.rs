//! 1-operand HW ops: `simd_unop`, its merge/zero-masked and
//! immediate-taking variants, plus the sign-bit `movemask` family.

/// 1-op HW unop ([`simd_binop`] minus second input). Per-lane scalar rem
/// only (`fn(elem) -> elem`); not for window-relative ops. Slice always
/// `pub`: some paths feed `auto` (popcnt via BITALG/VPOPCNTDQ), others are
/// tier-unique (e.g. `avx512cd` leading zeros) with no cascade.
macro_rules! simd_unop {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != a.len()`.
			pub fn $slice_fn(self, a: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, out) }
			}
		}

		// See simd_binop!'s comment: whole loop in one #[target_feature] fn.
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], out: &mut [$Elem]) {
			let a_chunks = a.chunks_exact($width);
			let a_rem = a_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let vr = $intrinsic(va);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}
			let op: fn($Elem) -> $Elem = $scalar;
			for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
				*o = op(x);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $intrinsic(va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked 1-op HW unop: same op as [`simd_unop`] plus a k-mask
/// operand (`_mm512_mask_*`/`_mm512_maskz_*` shape). Same `src`-vs-zero
/// split as [`simd_binop_masked`]: unop has room for a distinct `src`
/// register (only one data input), unlike the fused-op case in
/// [`simd_ternop_masked`]. Fixed-width only, no slice form.
macro_rules! simd_unop_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(self, src: [$Elem; $width], mask: $Mask, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(src: &[$Elem; $width], mask: $Mask, a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_unop`] minus the scalar closure/slice/auto: fixed-width only,
/// for unops with no honest per-lane no_std scalar reference (`sqrt`/`rcp`/
/// `rsqrt` on plain f32/f64: the HW op itself needs no libm, but a
/// `_slice` remainder loop would need `f32::sqrt`, unavailable under
/// `no_std` without an external libm dependency: same reasoning already
/// used for AVX512FP16's `sqrt_ph`/`rsqrt_ph`/`rcp_ph`).
macro_rules! simd_unop_fixed {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $intrinsic(va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Vector-to-scalar sign-bit mask (`movemask`-family: `pmovmskb`/`movmskps`/
/// `movmskpd` and their `v`-prefixed 256-bit forms). One bit per lane, set iff
/// the lane's MSB is set; unlike AVX-512's `movepi*_mask`/`movm_epi*`
/// (`k_vec_to_mask!`/`k_mask_to_vec!` in `avx512/kmask/bridge.rs`), there is no HW
/// inverse below AVX-512: these intrinsics only ever extract, never broadcast
/// a mask back into a vector. The intrinsic itself always returns `i32`; `$Mask`
/// truncates to the lane count (upper bits are architecturally zero, so `as`
/// truncation loses nothing).
macro_rules! simd_movemask {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		doc = $doc:literal,
	) => {
		impl $Token {
			#[doc = $doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width]) -> $Mask {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width]) -> $Mask {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				$intrinsic(va) as $Mask
			}
		}
	};
}

/// [`simd_unop`] plus a `const IMM8`. Fixed-width only, no `slice_fn`: see
/// [`simd_cvt`]'s doc for why this family of macros skips it (no honest
/// scalar-Rust closure for ops like RANGE/REDUCE's IMM8-selected behavior).
macro_rules! simd_unop_imm {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM8>(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $intrinsic::<IMM8>(va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_unop_imm`]: adds `src`/`mask`, same `const IMM8`.
macro_rules! simd_unop_imm_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:ident, zero_intrinsic = $zero_intrinsic:ident,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn<const IMM8: i32>(self, src: [$Elem; $width], mask: $Mask, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn::<IMM8>(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM8: i32>(self, mask: $Mask, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn::<IMM8>(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn<const IMM8: i32>(
			src: &[$Elem; $width], mask: $Mask, a: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic::<IMM8>(vsrc, mask, va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn<const IMM8: i32>(mask: $Mask, a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM8>(mask, va);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_movemask`] plus a `const IMM8` (FPCLASS-shaped: vector + immediate
/// category bitmask -> mask register, no `loadu`-of-second-operand). Fixed-
/// width only, same reasoning as [`simd_cvt`].
macro_rules! simd_unop_imm_mask {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$Elem; $width]) -> $Mask {
				unsafe { $intrinsic_fn::<IMM8>(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$Elem; $width]) -> $Mask {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				$intrinsic::<IMM8>(va) as $Mask
			}
		}
	};
}

/// Gated [`simd_unop_imm_mask`]: an extra `k1` input mask ANDed into the
/// result (`_mm512_mask_fpclass_pd_mask(k1, a) == fpclass(a) & k1`). Not a
/// merge/zero pair: the output is already a mask, so there's nothing for a
/// separate "zero" form to do that plain `&` doesn't already do, and there is
/// no `_maskz_fpclass_*` intrinsic in stdarch to wrap.
macro_rules! simd_unop_imm_mask_gated {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, k1: $Mask, a: [$Elem; $width]) -> $Mask {
				unsafe { $intrinsic_fn::<IMM8>(k1, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(k1: $Mask, a: &[$Elem; $width]) -> $Mask {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				$intrinsic::<IMM8>(k1, va) as $Mask
			}
		}
	};
}

pub(crate) use simd_movemask;
pub(crate) use simd_unop;
pub(crate) use simd_unop_fixed;
pub(crate) use simd_unop_imm;
pub(crate) use simd_unop_imm_mask;
pub(crate) use simd_unop_imm_mask_gated;
pub(crate) use simd_unop_imm_masked;
pub(crate) use simd_unop_masked;
