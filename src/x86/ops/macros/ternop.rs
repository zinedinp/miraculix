//! 3-operand HW ops (FMA-shaped `a*b+c`): `simd_ternop` and its
//! merge/zero-masked, fixed-width-only, and immediate-taking variants,
//! plus ternary-logic (`vpternlog*`) and the VNNI dot-product family
//! (mixed-element-type, so it doesn't fit `simd_ternop`'s single `$Elem`).

/// 3-op HW ternop (FMA `a*b+c`). Load/store cast as [`simd_binop`]. Scalar
/// rem is non-fused `a*b+c` (HW may round differently). Slice vis: same as
/// binop (`fma` 128-bit uses `vis = pub`; 256-bit is auto-cascaded).
macro_rules! simd_ternop {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(
				self,
				a: [$Elem; $width],
				b: [$Elem; $width],
				c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a, &b, &c) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// Length mismatch among `a`, `b`, `c`, `out`.
			$vis fn $slice_fn(
				self,
				a: &[$Elem],
				b: &[$Elem],
				c: &[$Elem],
				out: &mut [$Elem],
			) {
				assert_eq!(a.len(), b.len());
				assert_eq!(a.len(), c.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, b, c, out) }
			}
		}

		// whole loop in one #[target_feature] fn,
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
			let a_chunks = a.chunks_exact($width);
			let b_chunks = b.chunks_exact($width);
			let c_chunks = c.chunks_exact($width);
			let a_rem = a_chunks.remainder();
			let b_rem = b_chunks.remainder();
			let c_rem = c_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for (((ac, bc), cc), oc) in a_chunks
				.zip(b_chunks)
				.zip(c_chunks)
				.zip(out_chunks.by_ref())
			{
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let vb: $Vec = $loadu(bc.as_ptr().cast());
					let vc: $Vec = $loadu(cc.as_ptr().cast());
					let vr = $intrinsic(va, vb, vc);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}
			let op: fn($Elem, $Elem, $Elem) -> $Elem = $scalar;
			for (((&x, &y), &z), o) in a_rem
				.iter()
				.zip(b_rem)
				.zip(c_rem)
				.zip(out_chunks.into_remainder())
			{
				*o = op(x, y, z);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(
			a: &[$Elem; $width],
			b: &[$Elem; $width],
			c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $intrinsic(va, vb, vc);
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
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		simd_ternop! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = $width, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

/// Merge/zero-masked 3-op HW ternop (FMA `a*b+c`, masked). Unlike
/// [`simd_binop_masked`]/[`simd_unop_masked`], the merge form has no distinct
/// `src` register: hardware's FMA encoding is already 3 operands (`a`/`b`/`c`)
/// plus the mask, so `a` doubles as both an input and the merge fallback
/// (`_mm512_mask_fmadd_ps(a, k, b, c)`: unmasked lanes keep `a`, matching
/// Intel's own intrinsic signature: not a `simd_binop_masked`-style 4th
/// operand, that would need a register the instruction doesn't have).
macro_rules! simd_ternop_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(
				self, a: [$Elem; $width], mask: $Mask, b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn(&a, mask, &b, &c) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(
				self, mask: $Mask, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn(mask, &a, &b, &c) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			a: &[$Elem; $width], mask: $Mask, b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $merge_intrinsic(va, mask, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(
			mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_ternop_masked`] plus a compile-time rounding immediate
/// (`<const IMM: i32>`), for FMA `_round_ph`-style ops where hardware only
/// offers embedded rounding at 512-bit (no 128/256-bit form exists)
/// [`simd_ternop_masked`].
macro_rules! simd_ternop_imm_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:ident, zero_intrinsic = $zero_intrinsic:ident,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn<const IMM: i32>(
				self, a: [$Elem; $width], mask: $Mask, b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn::<IMM>(&a, mask, &b, &c) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM: i32>(
				self, mask: $Mask, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn::<IMM>(mask, &a, &b, &c) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn<const IMM: i32>(
			a: &[$Elem; $width], mask: $Mask, b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $merge_intrinsic::<IMM>(va, mask, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn<const IMM: i32>(
			mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM>(mask, va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Bitwise ternary logic (`vpternlogd`/`vpternlogq`): three vector inputs
/// plus a compile-time truth-table immediate (`IMM8`) selecting, per input
/// bit combination, what the output bit is. Unmasked form takes 3
/// independent inputs (`a`, `b`, `c`), same shape as [`simd_ternop`] plus
/// the immediate. The merge-masked intrinsic reuses its first operand as
/// both a logic input and the merge fallback
/// (`_mm512_mask_ternarylogic_epi32(src, k, a, b, IMM8)` computes
/// `ternarylogic(src, a, b)` then selects `src` for unmasked lanes): same
/// `src`/`mask`/`a`/`b` shape as [`simd_binop_masked`], not a plain ternop
/// merge. The zero-masked form keeps 3 independent inputs like the
/// unmasked op, just with mask-driven zero-fill
macro_rules! simd_ternarylogic {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		intrinsic_fn = $intrinsic_fn:ident, merge_intrinsic_fn = $merge_intrinsic_fn:ident,
		zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty, mask = $Mask:ty,
		loadu = $loadu:path, storeu = $storeu:path,
		intrinsic = $intrinsic:ident, merge_intrinsic = $merge_intrinsic:ident, zero_intrinsic = $zero_intrinsic:ident,
		fixed_doc = $fixed_doc:literal, merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(
				self, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM8>(&a, &b, &c) }
			}

			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn<const IMM8: i32>(
				self, src: [$Elem; $width], mask: $Mask, a: [$Elem; $width], b: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn::<IMM8>(&src, mask, &a, &b) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM8: i32>(
				self, mask: $Mask, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn::<IMM8>(mask, &a, &b, &c) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(
			a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $intrinsic::<IMM8>(va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn<const IMM8: i32>(
			src: &[$Elem; $width], mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $merge_intrinsic::<IMM8>(vsrc, mask, va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn<const IMM8: i32>(
			mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM8>(mask, va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Cross-type VNNI-style dot-product into `i32`:
/// `dst[j] = acc(src[j], sum_k(a[group*j+k] as i64 * b[group*j+k] as i64))`.
/// `a`/`b` are narrower and denser than `src`/`out`. Slice API is `pub`.
macro_rules! simd_vnni_dot {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, group = $group:literal, a_elem = $AElem:ty, b_elem = $BElem:ty,
		vec = $Vec:ty, loadu = $loadu:path, storeu = $storeu:path,
		intrinsic = $intrinsic:path,
		acc = $acc:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(
				self,
				src: [i32; $width],
				a: [$AElem; $width * $group],
				b: [$BElem; $width * $group],
			) -> [i32; $width] {
				unsafe { $intrinsic_fn(&src, &a, &b) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != src.len()`, or `a.len() != src.len() * $group`, or `b.len() != a.len()`.
			pub fn $slice_fn(self, src: &[i32], a: &[$AElem], b: &[$BElem], out: &mut [i32]) {
				assert_eq!(a.len(), src.len() * $group);
				assert_eq!(b.len(), a.len());
				assert_eq!(out.len(), src.len());
				unsafe { $slice_fn(src, a, b, out) }
			}
		}

		// whole loop in one #[target_feature] fn,
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(src: &[i32], a: &[$AElem], b: &[$BElem], out: &mut [i32]) {
			let src_chunks = src.chunks_exact($width);
			let src_rem = src_chunks.remainder();
			let a_chunks = a.chunks_exact($width * $group);
			let a_rem = a_chunks.remainder();
			let b_chunks = b.chunks_exact($width * $group);
			let b_rem = b_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for (((sc, ac), bc), oc) in src_chunks.zip(a_chunks).zip(b_chunks).zip(out_chunks.by_ref()) {
				unsafe {
					let vsrc: $Vec = $loadu(sc.as_ptr().cast());
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let vb: $Vec = $loadu(bc.as_ptr().cast());
					let vr = $intrinsic(vsrc, va, vb);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}

			let acc: fn(i32, i64) -> i32 = $acc;
			for (((&s, ag), bg), o) in src_rem
				.iter()
				.zip(a_rem.chunks_exact($group))
				.zip(b_rem.chunks_exact($group))
				.zip(out_chunks.into_remainder())
			{
				let sum: i64 = ag.iter().zip(bg).map(|(&x, &y)| (x as i64) * (y as i64)).sum();
				*o = acc(s, sum);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(
			src: &[i32; $width],
			a: &[$AElem; $width * $group],
			b: &[$BElem; $width * $group],
		) -> [i32; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $intrinsic(vsrc, va, vb);
				let mut out: [i32; $width] = [0; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_vnni_dot`]: same mixed-element-type shape (`a`/`b`
/// narrower than `src`/output, loaded through `$loadu` via `.cast()`
/// regardless of their own element type: see [`simd_vnni_dot`]'s comment).
/// `src` is not just a merge fallback here: the dot-product-accumulate is
/// `src + dot(a, b)` for every lane before masking is applied, so both the
/// merge *and* zero forms need it as a real input, unlike
/// [`simd_binop_masked`]'s zero form which drops `src` entirely. Confirmed
/// against stdarch: `_mm512_mask_dpbusd_epi32(src, k, a, b)` /
/// `_mm512_maskz_dpbusd_epi32(k, src, a, b)`.
macro_rules! simd_vnni_dot_masked {
	(
		token = $Token:ty, target_feature = $tf:literal,
		merge_fn = $merge_fn:ident, zero_fn = $zero_fn:ident,
		merge_intrinsic_fn = $merge_intrinsic_fn:ident, zero_intrinsic_fn = $zero_intrinsic_fn:ident,
		width = $width:literal, group = $group:literal, a_elem = $AElem:ty, b_elem = $BElem:ty,
		vec = $Vec:ty, mask = $Mask:ty, loadu = $loadu:path, storeu = $storeu:path,
		merge_intrinsic = $merge_intrinsic:path, zero_intrinsic = $zero_intrinsic:path,
		merge_doc = $merge_doc:literal, zero_doc = $zero_doc:literal,
	) => {
		impl $Token {
			#[doc = $merge_doc]
			#[inline]
			pub fn $merge_fn(
				self, src: [i32; $width], mask: $Mask, a: [$AElem; $width * $group], b: [$BElem; $width * $group],
			) -> [i32; $width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a, &b) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(
				self, mask: $Mask, src: [i32; $width], a: [$AElem; $width * $group], b: [$BElem; $width * $group],
			) -> [i32; $width] {
				unsafe { $zero_intrinsic_fn(mask, &src, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			src: &[i32; $width], mask: $Mask, a: &[$AElem; $width * $group], b: &[$BElem; $width * $group],
		) -> [i32; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va, vb);
				let mut out: [i32; $width] = [0; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(
			mask: $Mask, src: &[i32; $width], a: &[$AElem; $width * $group], b: &[$BElem; $width * $group],
		) -> [i32; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $zero_intrinsic(mask, vsrc, va, vb);
				let mut out: [i32; $width] = [0; $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_ternop`] minus the scalar closure/slice/auto: fixed-width only,
/// same reasoning as [`simd_binop_fixed`] (packed-complex FMA, and
/// alternating-lane `fmaddsub`/`fmsubadd` whose "scalar op" isn't a single
/// per-lane function of three scalars).
macro_rules! simd_ternop_fixed {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a, &b, &c) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $intrinsic(va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_ternop_fixed`] plus a `const IMM`: embedded-rounding FMA-family
/// ops (`fmadd_round_ph` and siblings). Fixed-width only; unlike
/// [`simd_ternop_masked`] there's no merge/zero-mask operand here, just the
/// extra immediate.
macro_rules! simd_ternop_imm {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM: i32>(
				self, a: [$Elem; $width], b: [$Elem; $width], c: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM>(&a, &b, &c) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM: i32>(
			a: &[$Elem; $width], b: &[$Elem; $width], c: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vc: $Vec = $loadu(c.as_ptr().cast());
				let vr = $intrinsic::<IMM>(va, vb, vc);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

pub(crate) use simd_ternarylogic;
pub(crate) use simd_ternop;
pub(crate) use simd_ternop_fixed;
pub(crate) use simd_ternop_imm;
pub(crate) use simd_ternop_imm_masked;
pub(crate) use simd_ternop_masked;
pub(crate) use simd_vnni_dot;
pub(crate) use simd_vnni_dot_masked;
