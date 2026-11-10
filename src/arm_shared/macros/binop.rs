//! 2-operand Neon ops: fixed-width rule, slice rule, and the combined rule
//! that emits both.
//!
//! The two rules are deliberately **separate** (x86 fuses them into one
//! `simd_binop!`): a fixed-width form is inherently width-bound, while the
//! `_slice` form is a contract about whole buffers. SVE fills exactly the same
//! `_slice` signature with a `whilelo`-predicated loop instead of chunk +
//! scalar remainder, so keeping the rules apart is what lets both back-ends
//! serve one public API. See `plans/AArch64-A SIMD-Plan.md` stage 1.1.

/// Fixed-width 2-op Neon binop: `[$Elem; $lanes]` in, same out.
///
/// `$lanes` is a parameter, not hard-coded to 4 as on aarch32: AArch64 has
/// both the 128-bit Q forms (`vaddq_s32`, 4 lanes) and the 64-bit D forms
/// (`vadd_s32`, 2 lanes), and the same rule has to serve `i8x16` down to
/// `f64x1`.
///
/// Neon `vld1*`/`vst1*` take an already-typed pointer (unlike x86's untyped
/// `_mm_loadu_si128`), so there is no `.cast()` here.
macro_rules! neon_binop_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $lanes], b: [$Elem; $lanes]) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn(&a, &b) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$Elem; $lanes], b: &[$Elem; $lanes]) -> [$Elem; $lanes] {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                let vb: $Vec = $loadu(b.as_ptr());
                let vr = $intrinsic(va, vb);
                let mut out: [$Elem; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice 2-op Neon binop: chunk by `$lanes`, scalar remainder.
///
/// Default slice visibility is `pub(crate)` (the `auto_up` cascade re-exports
/// it); pass `vis = pub` for a tier-unique family with no cascade rung.
macro_rules! neon_binop_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, b, out) }
			}
		}

		// The whole loop lives in one `#[target_feature]` fn rather than
		// calling `$fixed_fn` per chunk: rustc will not inline a
		// `#[target_feature]` callee into a caller without the same
		// attribute, so a per-chunk call leaves a real `call` plus a stack
		// round-trip in the hot loop. Same finding as x86's `simd_binop!`.
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (b_chunks, b_rem) = b.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					let vb: $Vec = $loadu(bc.as_ptr());
					let vr = $intrinsic(va, vb);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_rem) {
				*o = op(x, y);
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_binop_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one binop in a single call: the ordinary way to declare an
/// op that has a full-buffer meaning. Use the two rules above directly only
/// when an op has a fixed-width form but no sensible slice form (or the
/// reverse).
macro_rules! neon_binop {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_binop_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_binop_slice! {
            token = $Token, vis = $vis, target_feature = $tf,
            slice_fn = $slice_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            slice_doc = $slice_doc,
        }
    };
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_binop! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

/// 2-op with no Neon path at all (`i32`/`i64` division, and the 64-bit
/// integer multiply Neon has no `vmulq_s64` for). No `unsafe`, no
/// `target_feature`, no chunking - but the same method shape as
/// [`neon_binop`], so an `auto_up` cascade rung cannot tell the difference.
macro_rules! neon_scalar_only_binop {
	(
		token = $Token:ty, vis = $vis:vis,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $lanes], b: [$Elem; $lanes]) -> [$Elem; $lanes] {
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				core::array::from_fn(|i| op(a[i], b[i]))
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
					*o = op(x, y);
				}
			}
		}
	};
	(
		token = $Token:ty,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_scalar_only_binop! {
			token = $Token, vis = pub(crate),
			fixed_fn = $fixed_fn, slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

pub(crate) use {neon_binop, neon_binop_fixed, neon_binop_slice, neon_scalar_only_binop};
