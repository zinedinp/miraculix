//! Across-lane Neon reductions (`vaddvq_s32`, `vmaxvq_f32`, ...): a vector in,
//! one scalar out.
//!
//! A64 has real across-vector instructions here; A32 Neon does not (it needs a
//! `vpadd` tree), which is why this rule has no aarch32 counterpart to align
//! with.
//!
//! **Floating-point sum is not associative**, so `sum_f32_slice` is not
//! bit-identical to a left-to-right scalar sum over the same data. That is
//! inherent to any vector reduction and matches x86's `reduce` family; the
//! tests compare against a chunked oracle with the same grouping, never
//! against a plain scalar fold.

/// Fixed-width across-lane reduction: `[$Elem; $lanes]` in, one `$Elem` out.
macro_rules! neon_reduce_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $lanes]) -> $Elem {
                unsafe { $intrinsic_fn(&a) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$Elem; $lanes]) -> $Elem {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                $intrinsic(va)
            }
        }
    };
}

/// Whole-slice reduction: reduce each `$lanes`-wide chunk in hardware, fold
/// the per-chunk scalars and the tail with `combine`, starting from `init`.
///
/// `init` must be the identity of `combine` (0 for sum, `$Elem::MIN` for max);
/// an empty slice returns it unchanged.
macro_rules! neon_reduce_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		init = $init:expr, combine = $combine:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			$vis fn $slice_fn(self, a: &[$Elem]) -> $Elem {
				unsafe { $slice_fn(a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem]) -> $Elem {
			let (chunks, rem) = a.as_chunks::<$lanes>();
			let fold: fn($Elem, $Elem) -> $Elem = $combine;
			let mut acc: $Elem = $init;

			for ac in chunks {
				let part = unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					$intrinsic(va)
				};
				acc = fold(acc, part);
			}
			for &x in rem {
				acc = fold(acc, x);
			}
			acc
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		init = $init:expr, combine = $combine:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_reduce_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, vec = $Vec,
			loadu = $loadu, intrinsic = $intrinsic,
			init = $init, combine = $combine,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one reduction in a single call.
macro_rules! neon_reduce {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		init = $init:expr, combine = $combine:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_reduce_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_reduce_slice! {
            token = $Token, vis = $vis, target_feature = $tf,
            slice_fn = $slice_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, intrinsic = $intrinsic,
            init = $init, combine = $combine,
            slice_doc = $slice_doc,
        }
    };
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		init = $init:expr, combine = $combine:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_reduce! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, intrinsic = $intrinsic,
            init = $init, combine = $combine,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

pub(crate) use {neon_reduce, neon_reduce_fixed, neon_reduce_slice};
