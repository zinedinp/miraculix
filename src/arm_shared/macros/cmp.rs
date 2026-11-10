//! Neon lane compares. Two things differ from x86 and are baked into this
//! rule rather than left to call sites:
//!
//! 1. The result is **always unsigned**, whatever the input is signed,
//!    unsigned or floating (`vceqq_s32 -> uint32x4_t`, `vceqq_f64 ->
//!    uint64x2_t`). x86 types the mask like the input.
//! 2. The mask's lane width **follows the input's** lane width, so the mask
//!    element type is a parameter (`$Mask`): `cmpeq_i8` yields `u8` lanes,
//!    `cmpeq_f64` yields `u64` lanes. aarch32 only ever produced `u32` and
//!    could hard-code it.
//!
//! Lanes are all-ones or all-zero, not `bool`. Same as every other SIMD
//! compare in this crate.

/// Fixed-width Neon compare: `[$Elem; $lanes]` in, `[$Mask; $lanes]` out.
macro_rules! neon_cmp_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_storeu = $mask_storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $lanes], b: [$Elem; $lanes]) -> [$Mask; $lanes] {
                unsafe { $intrinsic_fn(&a, &b) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$Elem; $lanes], b: &[$Elem; $lanes]) -> [$Mask; $lanes] {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                let vb: $Vec = $loadu(b.as_ptr());
                let vr: $MaskVec = $intrinsic(va, vb);
                let mut out: [$Mask; $lanes] = [0; $lanes];
                $mask_storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice Neon compare: chunk by `$lanes`, scalar remainder.
///
/// `scalar` returns the mask element, so it carries the all-ones convention
/// into the tail: `|x, y| if x == y { u32::MAX } else { 0 }`.
macro_rules! neon_cmp_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_storeu = $mask_storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Mask]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, b, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], b: &[$Elem], out: &mut [$Mask]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (b_chunks, b_rem) = b.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					let vb: $Vec = $loadu(bc.as_ptr());
					let vr: $MaskVec = $intrinsic(va, vb);
					$mask_storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Elem, $Elem) -> $Mask = $scalar;
			for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_rem) {
				*o = op(x, y);
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_storeu = $mask_storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_cmp_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, mask = $Mask,
			vec = $Vec, mask_vec = $MaskVec,
			loadu = $loadu, mask_storeu = $mask_storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one compare in a single call.
macro_rules! neon_cmp {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_storeu = $mask_storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_cmp_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_storeu = $mask_storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_cmp_slice! {
            token = $Token, vis = $vis, target_feature = $tf,
            slice_fn = $slice_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_storeu = $mask_storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            slice_doc = $slice_doc,
        }
    };
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_storeu = $mask_storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_cmp! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_storeu = $mask_storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

pub(crate) use {neon_cmp, neon_cmp_fixed, neon_cmp_slice};
