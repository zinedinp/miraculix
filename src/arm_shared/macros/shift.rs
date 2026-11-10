//! Immediate-count Neon shifts (`vshlq_n_*`, `vshrq_n_*`, `vsraq_n_*`).
//!
//! The count is a `const` generic, not a runtime argument: the A64 encoding
//! puts it in the instruction word, and stdarch enforces the legal range with
//! a `static_assert`. Variable-count shifts (`vshlq_s32`, where the second
//! vector holds per-lane counts) are ordinary binops and use
//! [`super::binop::neon_binop`] instead.

/// Fixed-width immediate shift: `[$Elem; $lanes]` in, same out, count in
/// `IMM`.
macro_rules! neon_shift_imm_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn<const IMM: i32>(self, a: [$Elem; $lanes]) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn::<IMM>(&a) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn<const IMM: i32>(a: &[$Elem; $lanes]) -> [$Elem; $lanes] {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                let vr = $intrinsic::<IMM>(va);
                let mut out: [$Elem; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice immediate shift. `scalar` takes the element and the count, so
/// the tail shifts by the same amount the vector body did.
macro_rules! neon_shift_imm_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != a.len()`.
			$vis fn $slice_fn<const IMM: i32>(self, a: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn::<IMM>(a, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn<const IMM: i32>(a: &[$Elem], out: &mut [$Elem]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for (ac, oc) in a_chunks.iter().zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					let vr = $intrinsic::<IMM>(va);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Elem, i32) -> $Elem = $scalar;
			for (&x, o) in a_rem.iter().zip(out_rem) {
				*o = op(x, IMM);
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_shift_imm_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one immediate shift in a single call.
macro_rules! neon_shift_imm {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_shift_imm_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_shift_imm_slice! {
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
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_shift_imm! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

pub(crate) use {neon_shift_imm, neon_shift_imm_fixed, neon_shift_imm_slice};
