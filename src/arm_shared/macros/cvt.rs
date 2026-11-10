//! Lane-preserving Neon conversions (`vcvtq_s32_f32`, `vcvtq_f32_s32`, ...):
//! the element type changes, the lane count does not.
//!
//! Width-changing conversions (`f32` -> `f64` halves the lane count,
//! `vmovl_s16` widens) do **not** fit here: their chunking story is different
//! on both ends. They are written out by hand in the op file that needs them.

/// Fixed-width Neon convert: `[$Src; $lanes]` in, `[$Dst; $lanes]` out.
macro_rules! neon_cvt_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, src = $Src:ty, dst = $Dst:ty,
		src_vec = $SrcVec:ty, dst_vec = $DstVec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Src; $lanes]) -> [$Dst; $lanes] {
                unsafe { $intrinsic_fn(&a) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(a: &[$Src; $lanes]) -> [$Dst; $lanes] {
            unsafe {
                let va: $SrcVec = $loadu(a.as_ptr());
                let vr: $DstVec = $intrinsic(va);
                let mut out: [$Dst; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice Neon convert: chunk by `$lanes`, scalar remainder.
macro_rules! neon_cvt_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, src = $Src:ty, dst = $Dst:ty,
		src_vec = $SrcVec:ty, dst_vec = $DstVec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Src], out: &mut [$Dst]) {
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Src], out: &mut [$Dst]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for (ac, oc) in a_chunks.iter().zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $SrcVec = $loadu(ac.as_ptr());
					let vr: $DstVec = $intrinsic(va);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Src) -> $Dst = $scalar;
			for (&x, o) in a_rem.iter().zip(out_rem) {
				*o = op(x);
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, src = $Src:ty, dst = $Dst:ty,
		src_vec = $SrcVec:ty, dst_vec = $DstVec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_cvt_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, src = $Src, dst = $Dst,
			src_vec = $SrcVec, dst_vec = $DstVec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one conversion in a single call.
macro_rules! neon_cvt {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, src = $Src:ty, dst = $Dst:ty,
		src_vec = $SrcVec:ty, dst_vec = $DstVec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_cvt_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, src = $Src, dst = $Dst,
            src_vec = $SrcVec, dst_vec = $DstVec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_cvt_slice! {
            token = $Token, vis = $vis, target_feature = $tf,
            slice_fn = $slice_fn,
            lanes = $lanes, src = $Src, dst = $Dst,
            src_vec = $SrcVec, dst_vec = $DstVec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            slice_doc = $slice_doc,
        }
    };
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, src = $Src:ty, dst = $Dst:ty,
		src_vec = $SrcVec:ty, dst_vec = $DstVec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_cvt! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, src = $Src, dst = $Dst,
            src_vec = $SrcVec, dst_vec = $DstVec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

pub(crate) use {neon_cvt, neon_cvt_fixed, neon_cvt_slice};
