//! Structural data movement: broadcast (narrow->wide), extract/insert (IMM8-selected),
//! and AVX-512 compress/expand-from-memory, each with merge/zero-masked siblings where present.

/// Broadcast a narrower vector across a wider one (`narrow_width` ->
/// `wide_width`, same `$Elem`, no immediate). Fixed-width only, same
/// reasoning as [`simd_cvt`]: pure data movement has no per-lane scalar
/// closure at all.
macro_rules! simd_broadcast {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		narrow_width = $narrow_width:literal, wide_width = $wide_width:literal, elem = $Elem:ty,
		narrow_vec = $NarrowVec:ty, wide_vec = $WideVec:ty,
		narrow_loadu = $narrow_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $narrow_width]) -> [$Elem; $wide_width] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $narrow_width]) -> [$Elem; $wide_width] {
			unsafe {
				let va: $NarrowVec = $narrow_loadu(a.as_ptr().cast());
				let vr: $WideVec = $intrinsic(va);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_broadcast`]: `src`/mask/zero-fill are all
/// `wide_width` (the broadcast result's shape), plus a `wide_loadu` to load
/// `src`: the plain macro never needed one since it has no wide-shaped input.
macro_rules! simd_broadcast_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		narrow_width = $narrow_width:literal, wide_width = $wide_width:literal, elem = $Elem:ty,
		narrow_vec = $NarrowVec:ty, wide_vec = $WideVec:ty, mask = $Mask:ty,
		narrow_loadu = $narrow_loadu:path, wide_loadu = $wide_loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(self, src: [$Elem; $wide_width], mask: $Mask, a: [$Elem; $narrow_width]) -> [$Elem; $wide_width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, a: [$Elem; $narrow_width]) -> [$Elem; $wide_width] {
				unsafe { $zero_intrinsic_fn(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			src: &[$Elem; $wide_width], mask: $Mask, a: &[$Elem; $narrow_width],
		) -> [$Elem; $wide_width] {
			unsafe {
				let vsrc: $WideVec = $wide_loadu(src.as_ptr().cast());
				let va: $NarrowVec = $narrow_loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, a: &[$Elem; $narrow_width]) -> [$Elem; $wide_width] {
			unsafe {
				let va: $NarrowVec = $narrow_loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Extract a narrower sub-vector out of a wider one, IMM8-selected
/// (`wide_width` -> `narrow_width`, same `$Elem`). Fixed-width only, same
/// reasoning as [`simd_broadcast`].
macro_rules! simd_extract_imm {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		wide_width = $wide_width:literal, narrow_width = $narrow_width:literal, elem = $Elem:ty,
		wide_vec = $WideVec:ty, narrow_vec = $NarrowVec:ty,
		wide_loadu = $wide_loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$Elem; $wide_width]) -> [$Elem; $narrow_width] {
				unsafe { $intrinsic_fn::<IMM8>(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$Elem; $wide_width]) -> [$Elem; $narrow_width] {
			unsafe {
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vr: $NarrowVec = $intrinsic::<IMM8>(va);
				let mut out: [$Elem; $narrow_width] = [Default::default(); $narrow_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_extract_imm`]: `src`/mask/zero-fill are all
/// `narrow_width` (the extracted result's shape), plus a `narrow_loadu` for
/// `src`, same reasoning as [`simd_broadcast_masked`].
macro_rules! simd_extract_imm_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		wide_width = $wide_width:literal, narrow_width = $narrow_width:literal, elem = $Elem:ty,
		wide_vec = $WideVec:ty, narrow_vec = $NarrowVec:ty, mask = $Mask:ty,
		wide_loadu = $wide_loadu:path, narrow_loadu = $narrow_loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:ident, zero_intrinsic = $zero_intrinsic:ident,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn<const IMM8: i32>(
				self, src: [$Elem; $narrow_width], mask: $Mask, a: [$Elem; $wide_width],
			) -> [$Elem; $narrow_width] {
				unsafe { $merge_intrinsic_fn::<IMM8>(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM8: i32>(self, mask: $Mask, a: [$Elem; $wide_width]) -> [$Elem; $narrow_width] {
				unsafe { $zero_intrinsic_fn::<IMM8>(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn<const IMM8: i32>(
			src: &[$Elem; $narrow_width], mask: $Mask, a: &[$Elem; $wide_width],
		) -> [$Elem; $narrow_width] {
			unsafe {
				let vsrc: $NarrowVec = $narrow_loadu(src.as_ptr().cast());
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic::<IMM8>(vsrc, mask, va);
				let mut out: [$Elem; $narrow_width] = [Default::default(); $narrow_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn<const IMM8: i32>(mask: $Mask, a: &[$Elem; $wide_width]) -> [$Elem; $narrow_width] {
			unsafe {
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM8>(mask, va);
				let mut out: [$Elem; $narrow_width] = [Default::default(); $narrow_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Insert a narrower sub-vector into a wider one, IMM8-selected
/// (`a: [Elem; wide]` + `b: [Elem; narrow]` -> `[Elem; wide]`). Fixed-width
/// only, same reasoning as [`simd_broadcast`].
macro_rules! simd_insert_imm {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		wide_width = $wide_width:literal, narrow_width = $narrow_width:literal, elem = $Elem:ty,
		wide_vec = $WideVec:ty, narrow_vec = $NarrowVec:ty,
		wide_loadu = $wide_loadu:path, narrow_loadu = $narrow_loadu:path, storeu = $storeu:path,
		intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(
				self,
				a: [$Elem; $wide_width],
				b: [$Elem; $narrow_width],
			) -> [$Elem; $wide_width] {
				unsafe { $intrinsic_fn::<IMM8>(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(
			a: &[$Elem; $wide_width],
			b: &[$Elem; $narrow_width],
		) -> [$Elem; $wide_width] {
			unsafe {
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vb: $NarrowVec = $narrow_loadu(b.as_ptr().cast());
				let vr = $intrinsic::<IMM8>(va, vb);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_insert_imm`]: `src` is `wide_width`-shaped, same
/// as `a`: the intrinsic itself takes `(src, k, a, b)`/`(k, a, b)`, so `src`
/// gets its own `wide_loadu` call rather than reusing `a`'s.
macro_rules! simd_insert_imm_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		wide_width = $wide_width:literal, narrow_width = $narrow_width:literal, elem = $Elem:ty,
		wide_vec = $WideVec:ty, narrow_vec = $NarrowVec:ty, mask = $Mask:ty,
		wide_loadu = $wide_loadu:path, narrow_loadu = $narrow_loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:ident, zero_intrinsic = $zero_intrinsic:ident,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn<const IMM8: i32>(
				self, src: [$Elem; $wide_width], mask: $Mask, a: [$Elem; $wide_width], b: [$Elem; $narrow_width],
			) -> [$Elem; $wide_width] {
				unsafe { $merge_intrinsic_fn::<IMM8>(&src, mask, &a, &b) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM8: i32>(
				self, mask: $Mask, a: [$Elem; $wide_width], b: [$Elem; $narrow_width],
			) -> [$Elem; $wide_width] {
				unsafe { $zero_intrinsic_fn::<IMM8>(mask, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn<const IMM8: i32>(
			src: &[$Elem; $wide_width], mask: $Mask, a: &[$Elem; $wide_width], b: &[$Elem; $narrow_width],
		) -> [$Elem; $wide_width] {
			unsafe {
				let vsrc: $WideVec = $wide_loadu(src.as_ptr().cast());
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vb: $NarrowVec = $narrow_loadu(b.as_ptr().cast());
				let vr = $merge_intrinsic::<IMM8>(vsrc, mask, va, vb);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn<const IMM8: i32>(
			mask: $Mask, a: &[$Elem; $wide_width], b: &[$Elem; $narrow_width],
		) -> [$Elem; $wide_width] {
			unsafe {
				let va: $WideVec = $wide_loadu(a.as_ptr().cast());
				let vb: $NarrowVec = $narrow_loadu(b.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM8>(mask, va, vb);
				let mut out: [$Elem; $wide_width] = [Default::default(); $wide_width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Masked compress-to-memory (`vpcompress*`/`vcompress*` store form): the
/// selected lanes are left-packed and written to `out[..mask.count_ones()]`,
/// which is exactly how much memory the instruction touches: unselected lanes
/// produce no store at all, so there is no merge/zero split and no `maskz`
/// intrinsic. The length assert against the popcount is what makes the raw
/// pointer form safe to expose; the written count comes back so a caller
/// walking a buffer does not have to recompute it.
macro_rules! simd_compressstoreu {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, ptr_elem = $PtrElem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, intrinsic = $intrinsic:path,
		doc = $doc:literal,
	) => {
		impl $Token {
			#[doc = $doc]
			///
			/// Returns the number of elements written (`mask.count_ones()`).
			///
			/// # Panics
			/// `out.len() < mask.count_ones()`.
			#[inline]
			pub fn $fixed_fn(self, out: &mut [$Elem], mask: $Mask, a: [$Elem; $width]) -> usize {
				let written = mask.count_ones() as usize;
				assert!(out.len() >= written, "out too short for mask popcount");
				unsafe { $intrinsic_fn(out.as_mut_ptr(), mask, &a) };
				written
			}
		}

		/// # Safety
		/// Caller proved the feature via the token and checked that at least
		/// `mask.count_ones()` elements are writable at `out`.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(out: *mut $Elem, mask: $Mask, a: &[$Elem; $width]) {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				$intrinsic(out.cast::<$PtrElem>(), mask, va);
			}
		}
	};
}

/// Masked expand-from-memory (`vpexpand*`/`vexpand*` load form): the inverse of
/// [`simd_compressstoreu`]: each set mask bit consumes the next element from
/// `mem` in increasing-index order, so the instruction reads exactly
/// `mask.count_ones()` elements. Unlike the store form this one does have both
/// a merge and a zero variant, since unselected *lanes* (not memory) still need
/// a value.
macro_rules! simd_expandloadu {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, ptr_elem = $PtrElem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			///
			/// # Panics
			/// `mem.len() < mask.count_ones()`.
			#[inline]
			pub fn $merge_fn(self, src: [$Elem; $width], mask: $Mask, mem: &[$Elem]) -> [$Elem; $width] {
				assert!(mem.len() >= mask.count_ones() as usize, "mem too short for mask popcount");
				unsafe { $merge_intrinsic_fn(&src, mask, mem.as_ptr()) }
			}

			#[doc = $zero_doc]
			///
			/// # Panics
			/// `mem.len() < mask.count_ones()`.
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, mem: &[$Elem]) -> [$Elem; $width] {
				assert!(mem.len() >= mask.count_ones() as usize, "mem too short for mask popcount");
				unsafe { $zero_intrinsic_fn(mask, mem.as_ptr()) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token and checked that at least
		/// `mask.count_ones()` elements are readable at `mem`.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(src: &[$Elem; $width], mask: $Mask, mem: *const $Elem) -> [$Elem; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, mem.cast::<$PtrElem>());
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token and checked that at least
		/// `mask.count_ones()` elements are readable at `mem`.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, mem: *const $Elem) -> [$Elem; $width] {
			unsafe {
				let vr = $zero_intrinsic(mask, mem.cast::<$PtrElem>());
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

pub(crate) use simd_broadcast;
pub(crate) use simd_broadcast_masked;
pub(crate) use simd_compressstoreu;
pub(crate) use simd_expandloadu;
pub(crate) use simd_extract_imm;
pub(crate) use simd_extract_imm_masked;
pub(crate) use simd_insert_imm;
pub(crate) use simd_insert_imm_masked;
