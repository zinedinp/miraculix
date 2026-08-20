//! Element-type conversions: same-width (`simd_cvt`, plus its
//! merge/zero-masked form) and width-changing (`simd_cvt_widen`/
//! `simd_cvt_narrow`).

/// Convert between element types, same vector bit-width both sides (e.g.
/// `__m512d` <-> `__m512i`, never a width change in any current caller).
/// Fixed-width only, no `slice_fn`: HW "convert" intrinsics return an
/// integer-indefinite sentinel on overflow/NaN while Rust `as` casts
/// saturate instead: the two only agree in-range, which a chunked slice API
/// can't guarantee up front, so there is no honest scalar closure to check a
/// remainder against.
macro_rules! simd_cvt {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_storeu = $out_storeu:path,
		intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$InElem; $width]) -> [$OutElem; $width] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$InElem; $width]) -> [$OutElem; $width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr: $OutVec = $intrinsic(va);
				let mut out: [$OutElem; $width] = [Default::default(); $width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_cvt`]: same differing-in/out-element shape, plus
/// a `src`/`mask` pair. `src` and the zero-fill are always `$OutElem`: the
/// masked lanes never see `$InElem`, only the converted result does.
macro_rules! simd_cvt_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_loadu = $out_loadu:path, out_storeu = $out_storeu:path,
		mask = $Mask:ty,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(self, src: [$OutElem; $width], mask: $Mask, a: [$InElem; $width]) -> [$OutElem; $width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, a: [$InElem; $width]) -> [$OutElem; $width] {
				unsafe { $zero_intrinsic_fn(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			src: &[$OutElem; $width], mask: $Mask, a: &[$InElem; $width],
		) -> [$OutElem; $width] {
			unsafe {
				let vsrc: $OutVec = $out_loadu(src.as_ptr().cast());
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va);
				let mut out: [$OutElem; $width] = [Default::default(); $width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, a: &[$InElem; $width]) -> [$OutElem; $width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va);
				let mut out: [$OutElem; $width] = [Default::default(); $width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_cvt`] generalized to a fixed `in_width` (the source register's
/// full lane count) widening out to a variable `out_width`: needed for
/// `cvtph_pd`, which always reads a full 8-lane `__m128h` (only the low
/// `out_width` lanes are meaningful) but writes 2/4/8 `f64` lanes depending
/// on the target width. Fixed-width only, same shape as [`simd_broadcast`]
/// plus a type change.
macro_rules! simd_cvt_widen {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		in_width = $in_width:literal, out_width = $out_width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_storeu = $out_storeu:path,
		intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$InElem; $in_width]) -> [$OutElem; $out_width] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$InElem; $in_width]) -> [$OutElem; $out_width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr: $OutVec = $intrinsic(va);
				let mut out: [$OutElem; $out_width] = [Default::default(); $out_width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_cvt`] generalized to a variable `in_width` narrowing down to a
/// fixed `out_width` (the destination register's full lane count, upper
/// lanes zeroed by hardware): the mirror image of [`simd_cvt_widen`],
/// needed for `cvtpd_ph`.
macro_rules! simd_cvt_narrow {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		in_width = $in_width:literal, out_width = $out_width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_storeu = $out_storeu:path,
		intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$InElem; $in_width]) -> [$OutElem; $out_width] {
				unsafe { $intrinsic_fn(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$InElem; $in_width]) -> [$OutElem; $out_width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr: $OutVec = $intrinsic(va);
				let mut out: [$OutElem; $out_width] = [Default::default(); $out_width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_cvt_widen`]
macro_rules! simd_cvt_widen_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		in_width = $in_width:literal, out_width = $out_width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_loadu = $out_loadu:path, out_storeu = $out_storeu:path,
		mask = $Mask:ty,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(
				self, src: [$OutElem; $out_width], mask: $Mask, a: [$InElem; $in_width],
			) -> [$OutElem; $out_width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, a: [$InElem; $in_width]) -> [$OutElem; $out_width] {
				unsafe { $zero_intrinsic_fn(mask, &a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			src: &[$OutElem; $out_width], mask: $Mask, a: &[$InElem; $in_width],
		) -> [$OutElem; $out_width] {
			unsafe {
				let vsrc: $OutVec = $out_loadu(src.as_ptr().cast());
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va);
				let mut out: [$OutElem; $out_width] = [Default::default(); $out_width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, a: &[$InElem; $in_width]) -> [$OutElem; $out_width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va);
				let mut out: [$OutElem; $out_width] = [Default::default(); $out_width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_cvt_narrow`]. Identical body to
/// [`simd_cvt_widen_masked`] (the widen/narrow split is naming only, both
/// sides just take independent `in_width`/`out_width` literals): kept as a
/// separate macro to match the unmasked [`simd_cvt_narrow`]/[`simd_cvt_widen`]
/// pair.
macro_rules! simd_cvt_narrow_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		in_width = $in_width:literal, out_width = $out_width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_loadu = $out_loadu:path, out_storeu = $out_storeu:path,
		mask = $Mask:ty,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		simd_cvt_widen_masked! {
			token = $Token, target_feature = $tf,
			merge_fn = $merge_fn, zero_fn = $zero_fn,
			merge_intrinsic_fn = $merge_intrinsic_fn, zero_intrinsic_fn = $zero_intrinsic_fn,
			in_width = $in_width, out_width = $out_width,
			in_elem = $InElem, in_vec = $InVec, in_loadu = $in_loadu,
			out_elem = $OutElem, out_vec = $OutVec, out_loadu = $out_loadu, out_storeu = $out_storeu,
			mask = $Mask,
			merge_intrinsic = $merge_intrinsic, zero_intrinsic = $zero_intrinsic,
			merge_doc = $merge_doc, zero_doc = $zero_doc,
		}
	};
}

/// [`simd_cvt`] plus an embedded-rounding `<const IMM8: i32>` passthrough
/// (`_MM_FROUND_TO_*` OR `_MM_FROUND_NO_EXC`/`_MM_FROUND_CUR_DIRECTION` -
/// stdarch's own `static_assert_rounding!` rejects other combinations at
/// compile time). SAE/embedded rounding is EVEX-512-bit-only in this ISA, so
/// unlike `simd_cvt` this has no widen/narrow sibling: every caller so far
/// has matching in/out lane counts at 512-bit.
macro_rules! simd_cvt_imm {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal,
		in_elem = $InElem:ty, in_vec = $InVec:ty, in_loadu = $in_loadu:path,
		out_elem = $OutElem:ty, out_vec = $OutVec:ty, out_storeu = $out_storeu:path,
		intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$InElem; $width]) -> [$OutElem; $width] {
				unsafe { $intrinsic_fn::<IMM8>(&a) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$InElem; $width]) -> [$OutElem; $width] {
			unsafe {
				let va: $InVec = $in_loadu(a.as_ptr().cast());
				let vr: $OutVec = $intrinsic::<IMM8>(va);
				let mut out: [$OutElem; $width] = [Default::default(); $width];
				$out_storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

pub(crate) use simd_cvt;
pub(crate) use simd_cvt_imm;
pub(crate) use simd_cvt_masked;
pub(crate) use simd_cvt_narrow;
pub(crate) use simd_cvt_narrow_masked;
pub(crate) use simd_cvt_widen;
pub(crate) use simd_cvt_widen_masked;
