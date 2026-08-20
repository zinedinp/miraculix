//! Register-count `const IMM` shifts (`psll`/`psrl`/`psra` + `_mm_cvtsi32_si128`).

/// Uniform shift by `const IMM: u32`. Register-count form (`psll`/`psrl`/
/// `psra` + `_mm_cvtsi32_si128`), not imm variants, so one IMM type across
/// SSE/AVX2/AVX-512. Scalar rem: `fn(elem, u32) -> elem`. Slice vis: binop.
macro_rules! simd_shift_imm {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, shift = $shift:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM: u32>(self, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM>(&a) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != a.len()`.
			$vis fn $slice_fn<const IMM: u32>(self, a: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn::<IMM>(a, out) }
			}
		}

		// whole loop in one #[target_feature] fn.
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn<const IMM: u32>(a: &[$Elem], out: &mut [$Elem]) {
			let a_chunks = a.chunks_exact($width);
			let a_rem = a_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for (ac, oc) in a_chunks.zip(out_chunks.by_ref()) {
				unsafe {
					use core::arch::x86_64::_mm_cvtsi32_si128;
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let count = _mm_cvtsi32_si128(IMM as i32);
					let vr = $shift(va, count);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}
			let op: fn($Elem, u32) -> $Elem = $scalar;
			for (&x, o) in a_rem.iter().zip(out_chunks.into_remainder()) {
				*o = op(x, IMM);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM: u32>(a: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				use core::arch::x86_64::_mm_cvtsi32_si128;
				let va: $Vec = $loadu(a.as_ptr().cast());
				let count = _mm_cvtsi32_si128(IMM as i32);
				let vr = $shift(va, count);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, shift = $shift:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		simd_shift_imm! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = $width, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, shift = $shift,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

pub(crate) use simd_shift_imm;
