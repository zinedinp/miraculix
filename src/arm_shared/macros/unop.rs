//! 1-operand Neon ops (`abs`, `neg`, `not`, `sqrt`, `rnd*`). Same
//! fixed/slice split as [`super::binop`], one operand.

/// Fixed-width 1-op Neon unop: `[$Elem; $lanes]` in, same out.
macro_rules! neon_unop_fixed {
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
            pub fn $fixed_fn(self, a: [$Elem; $lanes]) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn(&a) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$Elem; $lanes]) -> [$Elem; $lanes] {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                let vr = $intrinsic(va);
                let mut out: [$Elem; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice 1-op Neon unop: chunk by `$lanes`, scalar remainder.
macro_rules! neon_unop_slice {
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
			/// `out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], out: &mut [$Elem]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for (ac, oc) in a_chunks.iter().zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					let vr = $intrinsic(va);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Elem) -> $Elem = $scalar;
			for (&x, o) in a_rem.iter().zip(out_rem) {
				*o = op(x);
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
		$crate::arm_shared::macros::neon_unop_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one unop in a single call.
macro_rules! neon_unop {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_unop_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_unop_slice! {
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
        $crate::arm_shared::macros::neon_unop! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

pub(crate) use {neon_unop, neon_unop_fixed, neon_unop_slice};
