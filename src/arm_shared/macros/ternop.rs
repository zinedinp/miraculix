//! 3-operand Neon ops. Three shapes, not one, because their operands are
//! not interchangeable:
//!
//! - [`neon_fma`] - `a * b + c`. Split out from a generic rule because
//!   ACLE's `vfmaq_f32(a, b, c)` computes `a + b * c`, i.e. the **addend
//!   comes first**. Writing that shuffle once here is the point; leaving it
//!   to call sites is how an operand-order bug gets in.
//! - [`neon_select`] - `vbslq_*`, whose first operand is a lane **mask**
//!   with its own element type (`u32` selecting between two `f32` vectors).
//! - [`neon_dot_acc`] - accumulate-shaped DotProd/I8MM ops: a narrow input
//!   pair folded into a wider accumulator. Fixed-width only; these have no
//!   independent-lane slice meaning, so they stay off the `auto` layer by
//!   the same policy x86 and aarch32 use.
//!
//! The first two follow the same fixed / slice / combined split as
//! [`super::binop`]; the 64-bit D-register forms only ever take the `_fixed`
//! rule, because the `_slice` name for an element type belongs to the widest
//! register shape that has it.

/// Fixed-width fused multiply-add: `a * b + c`, one rounding.
///
/// Emits `$intrinsic(vc, va, vb)` - see this module's doc for why.
macro_rules! neon_fma_fixed {
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
            pub fn $fixed_fn(
                self,
                a: [$Elem; $lanes],
                b: [$Elem; $lanes],
                c: [$Elem; $lanes],
            ) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn(&a, &b, &c) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(
            a: &[$Elem; $lanes],
            b: &[$Elem; $lanes],
            c: &[$Elem; $lanes],
        ) -> [$Elem; $lanes] {
            unsafe {
                let va: $Vec = $loadu(a.as_ptr());
                let vb: $Vec = $loadu(b.as_ptr());
                let vc: $Vec = $loadu(c.as_ptr());
                let vr = $intrinsic(vc, va, vb);
                let mut out: [$Elem; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice fused multiply-add.
///
/// The scalar tail is **not** fused (it evaluates `a * b + c` as written),
/// matching x86's `fmadd_f32`. Callers who need a fused tail pad to a whole
/// number of lanes.
macro_rules! neon_fma_slice {
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
			/// `a`, `b`, `c` and `out` do not all have the same length.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(a.len(), c.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, b, c, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (b_chunks, b_rem) = b.as_chunks::<$lanes>();
			let (c_chunks, c_rem) = c.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for (((ac, bc), cc), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(out_chunks.iter_mut()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr());
					let vb: $Vec = $loadu(bc.as_ptr());
					let vc: $Vec = $loadu(cc.as_ptr());
					let vr = $intrinsic(vc, va, vb);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Elem, $Elem, $Elem) -> $Elem = $scalar;
			for (((&x, &y), &z), o) in a_rem.iter().zip(b_rem).zip(c_rem).zip(out_rem) {
				*o = op(x, y, z);
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
		$crate::arm_shared::macros::neon_fma_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one fused multiply-add in a single call.
macro_rules! neon_fma {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_fma_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_fma_slice! {
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
        $crate::arm_shared::macros::neon_fma! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, vec = $Vec,
            loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

/// Fixed-width `vbslq_*` bit-select: `(mask & a) | (!mask & b)`, per bit.
///
/// The mask has its own element type (`$Mask`/`$MaskVec`): selecting between
/// two `f32` vectors takes a `u32` mask, exactly what [`super::cmp`] produces.
/// Unlike x86's `blendv`, this tests **every bit**, not just the sign bit.
macro_rules! neon_select_fixed {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_loadu = $mask_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(
                self,
                mask: [$Mask; $lanes],
                a: [$Elem; $lanes],
                b: [$Elem; $lanes],
            ) -> [$Elem; $lanes] {
                unsafe { $intrinsic_fn(&mask, &a, &b) }
            }
        }

        /// # Safety
        /// Caller proved the feature via the token.
        #[inline]
        #[target_feature(enable = $tf)]
        unsafe fn $intrinsic_fn(
            mask: &[$Mask; $lanes],
            a: &[$Elem; $lanes],
            b: &[$Elem; $lanes],
        ) -> [$Elem; $lanes] {
            unsafe {
                let vm: $MaskVec = $mask_loadu(mask.as_ptr());
                let va: $Vec = $loadu(a.as_ptr());
                let vb: $Vec = $loadu(b.as_ptr());
                let vr = $intrinsic(vm, va, vb);
                let mut out: [$Elem; $lanes] = [Default::default(); $lanes];
                $storeu(out.as_mut_ptr(), vr);
                out
            }
        }
    };
}

/// Whole-slice bit-select.
macro_rules! neon_select_slice {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_loadu = $mask_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `mask`, `a`, `b` and `out` do not all have the same length.
			$vis fn $slice_fn(self, mask: &[$Mask], a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(mask.len(), a.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(mask, a, b, out) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(mask: &[$Mask], a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let (m_chunks, m_rem) = mask.as_chunks::<$lanes>();
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (b_chunks, b_rem) = b.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();

			for (((mc, ac), bc), oc) in m_chunks.iter().zip(a_chunks).zip(b_chunks).zip(out_chunks.iter_mut()) {
				unsafe {
					let vm: $MaskVec = $mask_loadu(mc.as_ptr());
					let va: $Vec = $loadu(ac.as_ptr());
					let vb: $Vec = $loadu(bc.as_ptr());
					let vr = $intrinsic(vm, va, vb);
					$storeu(oc.as_mut_ptr(), vr);
				}
			}
			let op: fn($Mask, $Elem, $Elem) -> $Elem = $scalar;
			for (((&m, &x), &y), o) in m_rem.iter().zip(a_rem).zip(b_rem).zip(out_rem) {
				*o = op(m, x, y);
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_loadu = $mask_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		slice_doc = $slice_doc:literal,
	) => {
		$crate::arm_shared::macros::neon_select_slice! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			slice_fn = $slice_fn,
			lanes = $lanes, elem = $Elem, mask = $Mask,
			vec = $Vec, mask_vec = $MaskVec,
			loadu = $loadu, mask_loadu = $mask_loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			slice_doc = $slice_doc,
		}
	};
}

/// Both forms of one bit-select in a single call.
macro_rules! neon_select {
    (
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_loadu = $mask_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_select_fixed! {
            token = $Token, target_feature = $tf,
            fixed_fn = $fixed_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_loadu = $mask_loadu, storeu = $storeu, intrinsic = $intrinsic,
            fixed_doc = $fixed_doc,
        }
        $crate::arm_shared::macros::neon_select_slice! {
            token = $Token, vis = $vis, target_feature = $tf,
            slice_fn = $slice_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_loadu = $mask_loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            slice_doc = $slice_doc,
        }
    };
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		lanes = $lanes:literal, elem = $Elem:ty, mask = $Mask:ty,
		vec = $Vec:ty, mask_vec = $MaskVec:ty,
		loadu = $loadu:path, mask_loadu = $mask_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        $crate::arm_shared::macros::neon_select! {
            token = $Token, vis = pub(crate), target_feature = $tf,
            fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
            lanes = $lanes, elem = $Elem, mask = $Mask,
            vec = $Vec, mask_vec = $MaskVec,
            loadu = $loadu, mask_loadu = $mask_loadu, storeu = $storeu, intrinsic = $intrinsic,
            scalar = $scalar,
            fixed_doc = $fixed_doc, slice_doc = $slice_doc,
        }
    };
}

/// Accumulate-shaped dot / matrix-multiply op: `acc` plus two 128-bit
/// operands of `in_lanes` narrow elements, folded into a 4-lane accumulator.
///
/// Two back-ends, one method shape:
///
/// - `intrinsic = path,` - a stable `core::arch` intrinsic taking
///   `(acc, a, b)` (DotProd's `vdotq_s32`/`vdotq_u32`, stable since 1.98).
/// - `asm = "…",` - an **`asm!` template**, for the users whose intrinsics
///   are still unstable or not exported (I8MM `SMMLA`/`UMMLA`/`USMMLA`/
///   `USDOT`, BF16 `BFDOT`/`BFMMLA`/`BFMLALB`/`BFMLALT`) while their target
///   features are stable. `$asm` names its operands `{acc}`, `{a}` and `{b}`
///   (`"smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b"`). When such an intrinsic is
///   stabilised, the call site swaps `asm = ...` for `intrinsic = ...` and
///   the public API does not move.
///
/// Fixed-width only, by policy: these reduce across lanes, so there is no
/// elementwise slice contract to offer and no `auto` rung.
macro_rules! neon_dot_acc {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident, in_lanes = $n:literal,
		acc = $Acc:ty, acc_vec = $AccVec:ty, a_elem = $A:ty, a_vec = $AVec:ty, b_elem = $B:ty, b_vec = $BVec:ty,
		acc_loadu = $acc_loadu:path, acc_storeu = $acc_storeu:path,
		a_loadu = $a_loadu:path, b_loadu = $b_loadu:path,
		intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, acc: [$Acc; 4], a: [$A; $n], b: [$B; $n]) -> [$Acc; 4] {
				unsafe { $intrinsic_fn(&acc, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(acc: &[$Acc; 4], a: &[$A; $n], b: &[$B; $n]) -> [$Acc; 4] {
			unsafe {
				let vacc: $AccVec = $acc_loadu(acc.as_ptr());
				let va: $AVec = $a_loadu(a.as_ptr());
				let vb: $BVec = $b_loadu(b.as_ptr());
				let vr = $intrinsic(vacc, va, vb);
				let mut out: [$Acc; 4] = [Default::default(); 4];
				$acc_storeu(out.as_mut_ptr(), vr);
				out
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident, in_lanes = $n:literal,
		acc = $Acc:ty, acc_vec = $AccVec:ty, a_elem = $A:ty, a_vec = $AVec:ty, b_elem = $B:ty, b_vec = $BVec:ty,
		acc_loadu = $acc_loadu:path, acc_storeu = $acc_storeu:path,
		a_loadu = $a_loadu:path, b_loadu = $b_loadu:path,
		asm = $asm:literal,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, acc: [$Acc; 4], a: [$A; $n], b: [$B; $n]) -> [$Acc; 4] {
				unsafe { $intrinsic_fn(&acc, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(acc: &[$Acc; 4], a: &[$A; $n], b: &[$B; $n]) -> [$Acc; 4] {
			unsafe {
				let mut vacc: $AccVec = $acc_loadu(acc.as_ptr());
				let va: $AVec = $a_loadu(a.as_ptr());
				let vb: $BVec = $b_loadu(b.as_ptr());
				core::arch::asm!(
					$asm,
					acc = inout(vreg) vacc,
					a = in(vreg) va,
					b = in(vreg) vb,
					options(pure, nomem, nostack, preserves_flags),
				);
				let mut out: [$Acc; 4] = [Default::default(); 4];
				$acc_storeu(out.as_mut_ptr(), vacc);
				out
			}
		}
	};
}

pub(crate) use {
    neon_dot_acc, neon_fma, neon_fma_fixed, neon_fma_slice, neon_select, neon_select_fixed,
    neon_select_slice,
};
