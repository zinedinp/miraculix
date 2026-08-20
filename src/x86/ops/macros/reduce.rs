//! Cross-lane horizontal reduction: whole register -> one scalar.

/// Whole-register reduce (`_mm512_reduce_*`: compiler-synthesized
/// shuffle+op chain, no single HW instruction, but gated on the feature
/// like any other op here since it still needs a live `__m512*` value).
/// Fixed-width only: folding results across slice chunks needs a
/// caller-chosen combining op (`+`/`min`/`max`/`*`), which isn't this
/// macro's business: callers fold per-chunk reduce results themselves.
macro_rules! simd_reduce {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, out = $Out:ty, vec = $Vec:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width]) -> $Out {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width]) -> $Out {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				// same-width `as`: identity for the native return type, bit-preserving
				// int<->int reinterpret for the typed-unsigned sibling (e.g. `epi32`'s
				// `i32` result read as `u32`: same trick `shl_u32x16` uses to share
				// `shl_i32x16`'s intrinsic).
				$intrinsic(va) as $Out
			}
		}
	};
}

pub(crate) use simd_reduce;
