//! 2-operand HW ops: full-fat (`simd_binop`) through scalar-only,
//! immediate-taking, and fixed-width-only variants, plus their
//! merge/zero-masked siblings.

/// 2-op HW binop. `loadu`/`storeu` get `.cast()` so typed (`_mm_loadu_ps`)
/// and untyped (`_mm_loadu_si128`) paths share one form. Slice default
/// `pub(crate)` (auto cascade re-exports); `vis = pub` for tier-unique
/// families (e.g. avx2 i8/u16) with no cascade.
macro_rules! simd_binop {
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
			pub fn $fixed_fn(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a, &b) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn(a, b, out) }
			}
		}

		// Whole loop lives in one `#[target_feature]` fn (not a per-chunk call
		// into `$fixed_fn`): rustc won't inline a `#[target_feature]` callee
		// into a caller that doesn't share the same attribute, so a per-chunk
		// call left a real `call`+stack-round-trip in the hot loop.
		// This keeps the load/op/store loop register-resident.
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let a_chunks = a.chunks_exact($width);
			let b_chunks = b.chunks_exact($width);
			let a_rem = a_chunks.remainder();
			let b_rem = b_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let vb: $Vec = $loadu(bc.as_ptr().cast());
					let vr = $intrinsic(va, vb);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
				*o = op(x, y);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $intrinsic(va, vb);
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
		simd_binop! {
			token = $Token, vis = pub(crate), target_feature = $tf,
			fixed_fn = $fixed_fn, slice_fn = $slice_fn, intrinsic_fn = $intrinsic_fn,
			width = $width, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

/// Merge/zero-masked 2-op HW binop: same op as [`simd_binop`] plus a k-mask
/// operand (`_mm512_mask_*`/`_mm512_maskz_*` shape). Merge form takes `src`
/// (copied into lanes whose mask bit is unset); zero form zeroes those
/// lanes instead. Fixed-width only, no slice form: a slice mask would need
/// its own chunking story, not attempted here.
macro_rules! simd_binop_masked {
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
			pub fn $merge_fn(self, src: [$Elem; $width], mask: $Mask, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn(&src, mask, &a, &b) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn(self, mask: $Mask, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn(mask, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $merge_intrinsic_fn(
			src: &[$Elem; $width], mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width],
		) -> [$Elem; $width] {
			unsafe {
				let vsrc: $Vec = $loadu(src.as_ptr().cast());
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $merge_intrinsic(vsrc, mask, va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $zero_intrinsic_fn(mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $zero_intrinsic(mask, va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// 2-op with no HW SIMD path (integer div, 8-bit mul): no `unsafe`, no
/// `target_feature`, no load/store/chunking. Same method shape as
/// [`simd_binop`] so auto cascades are agnostic. Slice vis: same as binop.
macro_rules! scalar_only_binop {
	(
		token = $Token:ty, vis = $vis:vis,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		width = $width:literal, elem = $Elem:ty,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				core::array::from_fn(|i| op(a[i], b[i]))
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				for ((&x, &y), o) in a.iter().zip(b).zip(out.iter_mut()) {
					*o = op(x, y);
				}
			}
		}
	};
	(
		token = $Token:ty,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		width = $width:literal, elem = $Elem:ty,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		scalar_only_binop! {
			token = $Token, vis = pub(crate),
			fixed_fn = $fixed_fn, slice_fn = $slice_fn,
			width = $width, elem = $Elem,
			scalar = $scalar,
			fixed_doc = $fixed_doc, slice_doc = $slice_doc,
		}
	};
}

/// 2-op HW binop with a compile-time immediate (`const IMM8`), e.g. VBMI2's
/// `shldi`/`shrdi` (`_mm512_shldi_epi64::<IMM8>(a, b)`). Unlike
/// [`simd_shift_imm`] (one data input, immediate converted to a runtime
/// shift-count register via `_mm_cvtsi32_si128`), this has two independent
/// data inputs and the immediate goes straight into the intrinsic's own
/// turbofish: no runtime conversion needed, since the HW instruction itself
/// takes an 8-bit immediate operand. stdarch defines `shldi` as `shldv(a, b,
/// set1(IMM8))` internally, but wraps a real immediate-form instruction, so
/// this macro calls the intrinsic directly rather than composing through
/// [`simd_ternop`]'s runtime-vector third operand.
macro_rules! simd_binop_imm {
	(
		token = $Token:ty, vis = $vis:vis, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident, intrinsic_fn = $intrinsic_fn:ident,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:ident,
		scalar = $scalar:expr,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM8>(&a, &b) }
			}

			#[doc = $slice_doc]
			///
			/// # Panics
			/// `a.len() != b.len() || out.len() != a.len()`.
			$vis fn $slice_fn<const IMM8: i32>(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				assert_eq!(a.len(), b.len());
				assert_eq!(out.len(), a.len());
				unsafe { $slice_fn::<IMM8>(a, b, out) }
			}
		}

		// whole loop in one #[target_feature] fn
		/// # Safety
		/// Caller proved the feature via the token.
		#[target_feature(enable = $tf)]
		unsafe fn $slice_fn<const IMM8: i32>(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			let a_chunks = a.chunks_exact($width);
			let b_chunks = b.chunks_exact($width);
			let a_rem = a_chunks.remainder();
			let b_rem = b_chunks.remainder();
			let mut out_chunks = out.chunks_exact_mut($width);

			for ((ac, bc), oc) in a_chunks.zip(b_chunks).zip(out_chunks.by_ref()) {
				unsafe {
					let va: $Vec = $loadu(ac.as_ptr().cast());
					let vb: $Vec = $loadu(bc.as_ptr().cast());
					let vr = $intrinsic::<IMM8>(va, vb);
					$storeu(oc.as_mut_ptr().cast(), vr);
				}
			}
			let op: fn($Elem, $Elem, i32) -> $Elem = $scalar;
			for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_chunks.into_remainder()) {
				*o = op(x, y, IMM8);
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $intrinsic::<IMM8>(va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Merge/zero-masked [`simd_binop_imm`]. Unlike [`simd_ternop_masked`]'s
/// fused-op reuse (no distinct `src`, the first data operand doubles as the
/// merge fallback), `shldi`/`shrdi`'s merge intrinsic has a genuinely
/// separate `src` register: `_mm512_mask_shldi_epi64(src, k, a, b)` takes 3
/// independent vector params (confirmed via stdarch), the same
/// `src`/`mask`/`a`/`b` shape as [`simd_binop_masked`], just with the added
/// immediate. Fixed-width only, no slice form (mask has no chunking story
/// yet, same reasoning as [`simd_binop_masked`]).
macro_rules! simd_binop_imm_masked {
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
			pub fn $merge_fn<const IMM8: i32>(
				self, src: [$Elem; $width], mask: $Mask, a: [$Elem; $width], b: [$Elem; $width],
			) -> [$Elem; $width] {
				unsafe { $merge_intrinsic_fn::<IMM8>(&src, mask, &a, &b) }
			}

			#[doc = $zero_doc]
			#[inline]
			pub fn $zero_fn<const IMM8: i32>(self, mask: $Mask, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $zero_intrinsic_fn::<IMM8>(mask, &a, &b) }
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
		unsafe fn $zero_intrinsic_fn<const IMM8: i32>(mask: $Mask, a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $zero_intrinsic::<IMM8>(mask, va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_binop_imm`] minus the `slice_fn` (RANGE-shaped: no honest scalar
/// closure for its 16 IMM8-selected min/max/sign combos: see [`simd_cvt`]'s
/// doc for the general reasoning this macro family shares).
macro_rules! simd_binop_imm_fixed {
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
			pub fn $fixed_fn<const IMM8: i32>(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn::<IMM8>(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn<const IMM8: i32>(a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $intrinsic::<IMM8>(va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// [`simd_binop`] minus the scalar closure/slice/auto: fixed-width only,
/// for binops with no honest per-lane scalar reference (packed-complex
/// multiply, `scalef`'s libm-shaped `x * 2^floor(y)`, and the scalar `_sh`
/// family's lane-0-only + passthrough semantics).
macro_rules! simd_binop_fixed {
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
			pub fn $fixed_fn(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $intrinsic_fn(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $intrinsic_fn(a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			unsafe {
				let va: $Vec = $loadu(a.as_ptr().cast());
				let vb: $Vec = $loadu(b.as_ptr().cast());
				let vr = $intrinsic(va, vb);
				let mut out: [$Elem; $width] = [Default::default(); $width];
				$storeu(out.as_mut_ptr().cast(), vr);
				out
			}
		}
	};
}

/// Experimental "wider-bus lift" companion to [`simd_binop`] (`notes/
/// miraculix/TODO.md` section D): same elementwise op, same `$width` per
/// chain, but the slice loop processes two independent chains per
/// iteration (interleaved, not a straight single stream) compiled with
/// `target_feature = $lift_tf`: meant to be invoked *alongside*
/// `simd_binop!` with a `$lift_tf` naming a wider tier than the token's own
/// base feature (e.g. `Avx2`'s `"avx2"` base -> `"avx2,avx512f,avx512vl"`
/// lift; a SSE-family token's own base -> `",avx"`-suffixed lift). Callers
/// supply `lifted_fn` explicitly (stable `macro_rules!` can't paste
/// `$slice_fn` + `"_lifted"` into a new ident, same as `simd_binop!`
/// already requires an explicit `slice_fn` rather than deriving it from
/// `fixed_fn`); the internal loop fn is hidden inside an anonymous `const
/// _: () = { ... };` scope instead, so it needs no distinct name per
/// invocation despite every invocation reusing the same internal identifier
/// (`impl` blocks aren't namespaced by their surrounding scope for method
/// resolution, so `$lifted_fn` stays reachable as `Token::$lifted_fn`).
/// `#[cfg(feature = "wider-bus-lift")]`-gated; the feature-off arm expands
/// to nothing so call sites need no `#[cfg]` of their own.
#[cfg(feature = "wider-bus-lift")]
macro_rules! simd_binop_lifted {
	(
		token = $Token:ty, vis = $vis:vis, lift_target_feature = $lift_tf:literal,
		lifted_fn = $lifted_fn:ident, lift_proof = $LiftProof:ty,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		lifted_doc = $lifted_doc:literal,
	) => {
		const _: () = {
			// Two independent $width-wide chains per iteration (not one
			// $width*2 chain): the point is issuing two independent-
			// dependency-chain HW ops of the token's *native* width, not a
			// wider op that doesn't exist. See benches/wider_bus_lift.rs's
			// investigation.
			/// # Safety
			/// Caller proved `$lift_tf` (the wider tier being lifted
			/// toward), not just the token's own base feature: see
			/// `x86::auto_up`/`x86::auto_down`.
			#[target_feature(enable = $lift_tf)]
			unsafe fn lifted_impl(a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
				const PAIR: usize = $width * 2;
				let n = a.len();
				let pairs = n / PAIR;
				unsafe {
					for i in 0..pairs {
						let base0 = i * PAIR;
						let base1 = base0 + $width;
						let va0: $Vec = $loadu(a.as_ptr().add(base0).cast());
						let va1: $Vec = $loadu(a.as_ptr().add(base1).cast());
						let vb0: $Vec = $loadu(b.as_ptr().add(base0).cast());
						let vb1: $Vec = $loadu(b.as_ptr().add(base1).cast());
						let vr0 = $intrinsic(va0, vb0);
						let vr1 = $intrinsic(va1, vb1);
						$storeu(out.as_mut_ptr().add(base0).cast(), vr0);
						$storeu(out.as_mut_ptr().add(base1).cast(), vr1);
					}
				}
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let done = pairs * PAIR;
				for j in done..n {
					out[j] = op(a[j], b[j]);
				}
			}

			impl $Token {
				#[doc = $lifted_doc]
				///
				/// The `_proof` parameter is otherwise-unused: it exists
				/// purely so the *caller* has to hold a `$LiftProof` to call
				/// this at all. `self: Token` alone doesn't prove the lift
				/// target's features, unlike every other op in this crate
				/// where the receiver token is sufficient proof: see
				/// `x86::auto_up`/`x86::auto_down`, the only intended
				/// callers.
				///
				/// # Panics
				/// `a.len() != b.len() || out.len() != a.len()`.
				$vis fn $lifted_fn(self, _proof: $LiftProof, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
					assert_eq!(a.len(), b.len());
					assert_eq!(out.len(), a.len());
					unsafe { lifted_impl(a, b, out) }
				}
			}
		};
	};
	(
		token = $Token:ty, lift_target_feature = $lift_tf:literal,
		lifted_fn = $lifted_fn:ident, lift_proof = $LiftProof:ty,
		width = $width:literal, elem = $Elem:ty, vec = $Vec:ty,
		loadu = $loadu:path, storeu = $storeu:path, intrinsic = $intrinsic:path,
		scalar = $scalar:expr,
		lifted_doc = $lifted_doc:literal,
	) => {
		simd_binop_lifted! {
			token = $Token, vis = pub(crate), lift_target_feature = $lift_tf,
			lifted_fn = $lifted_fn, lift_proof = $LiftProof,
			width = $width, elem = $Elem, vec = $Vec,
			loadu = $loadu, storeu = $storeu, intrinsic = $intrinsic,
			scalar = $scalar,
			lifted_doc = $lifted_doc,
		}
	};
}
#[cfg(not(feature = "wider-bus-lift"))]
macro_rules! simd_binop_lifted {
	($($tt:tt)*) => {};
}

pub(crate) use scalar_only_binop;
pub(crate) use simd_binop;
pub(crate) use simd_binop_fixed;
pub(crate) use simd_binop_imm;
pub(crate) use simd_binop_imm_fixed;
pub(crate) use simd_binop_imm_masked;
pub(crate) use simd_binop_lifted;
pub(crate) use simd_binop_masked;
