//! SVE wrappers: a `whilelo` loop in `asm!`, and the safe token methods in
//! front of it.
//!
//! No stable SVE intrinsics (`stdarch#1509`). The `sve`/`sve2` target
//! features are stable, so `asm!` inside `#[target_feature(enable = "sve")]`
//! assembles SVE. Same pattern as MMX, XOP, and Neon ops whose intrinsics
//! are still unstable.
//!
//! ## The loop
//!
//! Every streaming op is one shape, whatever the element type:
//!
//! ```text
//!     mov     i, #0
//!     whilelo p0.T, i, n        // lanes i..min(i + VL, n) active
//!     b.none  3f                // n == 0
//! 2:  ld1X    { z0.T }, p0/z, [a, i, lsl #k]   // inactive lanes: no access
//!     <op>                      // result in z<result>
//!     st1X    { z<result>.T }, p0, [out, i, lsl #k]
//!     incX    i                 // i += lanes per vector
//!     whilelo p0.T, i, n
//!     b.first 2b
//! 3:
//! ```
//!
//! **Predicate.** Last partial vector is a shorter predicate. **No scalar
//! tail**, unlike Neon's chunk + remainder. Same `_slice` signature.
//! Inactive lanes are not loaded or stored, so a short slice never goes past
//! its end.
//!
//! **Clobber.** `z0`..`z3`, `p0`, `p1`. This crate has no SVE types in Rust,
//! so LLVM will not hold a scalable value across the `asm!`. Clobbering the
//! overlapping `v0`..`v3` NEON views is enough.
//!
//! ## Vector length
//!
//! The loop advances by whatever VL is. [`sve_fixed`] on `SveVl*` runs the
//! same loop over an array, so it is exact on any VL. The token's VL
//! guarantee only means the loop runs once.

/// The `#[target_feature]` inner fn: one `whilelo` loop over `n` elements.
///
/// `size = [T, ld, st, shift, inc]` pins the lane size (`"s"`, `"ld1w"`,
/// `"st1w"`, `", lsl #2"`, `"incw"`); every stream of one op has the same lane
/// size (compare masks and same-width conversions included). `ins` maps each
/// input pointer to the `z` register it is loaded into; `result` names the
/// register that is stored. `imm = IMM,` adds a `const IMM: i32` generic,
/// reachable in `op` as `{imm}`.
macro_rules! sve_stream {
	(
		target_feature = $tf:literal, inner_fn = $inner:ident,
		size = [$t:literal, $ld:literal, $st:literal, $sh:literal, $inc:literal],
		ins = [$($x:ident: $X:ty => $z:literal),+ $(,)?], out = $O:ty, result = $zr:literal,
		op = [$($op:literal),+ $(,)?],
	) => {
		/// # Safety
		/// Caller proved the feature via the token; every pointer is valid for
		/// `n` elements.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $inner($($x: *const $X,)+ out: *mut $O, n: usize) {
			unsafe {
				$crate::aarch64::ops::macros::sve_stream!(@asm
					[$t, $ld, $st, $sh, $inc], [$($x => $z),+], $zr, [$($op),+],
					[$($x = in(reg) $x,)+ o = in(reg) out, n = in(reg) n,]
				);
			}
		}
	};
	(
		target_feature = $tf:literal, inner_fn = $inner:ident,
		size = [$t:literal, $ld:literal, $st:literal, $sh:literal, $inc:literal],
		ins = [$($x:ident: $X:ty => $z:literal),+ $(,)?], out = $O:ty, result = $zr:literal,
		op = [$($op:literal),+ $(,)?],
		imm = IMM,
	) => {
		/// # Safety
		/// Caller proved the feature via the token; every pointer is valid for
		/// `n` elements. `IMM` is in range for the instruction (checked by the
		/// caller's `const` assert, else the assembler rejects it).
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $inner<const IMM: i32>($($x: *const $X,)+ out: *mut $O, n: usize) {
			unsafe {
				$crate::aarch64::ops::macros::sve_stream!(@asm
					[$t, $ld, $st, $sh, $inc], [$($x => $z),+], $zr, [$($op),+],
					[$($x = in(reg) $x,)+ o = in(reg) out, n = in(reg) n, imm = const IMM,]
				);
			}
		}
	};
	(@asm
		[$t:literal, $ld:literal, $st:literal, $sh:literal, $inc:literal],
		[$($x:ident => $z:literal),+], $zr:literal, [$($op:literal),+], [$($operands:tt)*]
	) => {
		core::arch::asm!(
			"mov {i}, #0",
			concat!("whilelo p0.", $t, ", {i}, {n}"),
			"b.none 3f",
			"2:",
			$(concat!($ld, " {{ z", $z, ".", $t, " }}, p0/z, [{", stringify!($x), "}, {i}", $sh, "]"),)+
			$($op,)+
			concat!($st, " {{ z", $zr, ".", $t, " }}, p0, [{o}, {i}", $sh, "]"),
			concat!($inc, " {i}"),
			concat!("whilelo p0.", $t, ", {i}, {n}"),
			"b.first 2b",
			"3:",
			i = out(reg) _,
			$($operands)*
			out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("p0") _, out("p1") _,
			options(nostack),
		)
	};
}

/// Across-slice reduction: `init` seeds the accumulator `z3` (or `s3`/`d3`)
/// from the `{init}` bit pattern, `step` folds each predicated vector `z0`
/// into it, `fini` moves the answer into the GPR `{r}`, and `ret` turns those
/// 64 bits back into `$E`.
macro_rules! sve_fold {
	(
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident, inner_fn = $inner:ident,
		size = [$t:literal, $ld:literal, $sh:literal, $inc:literal],
		elem = $E:ty, init_bits = $bits:expr,
		init = [$($init:literal),+ $(,)?], step = [$($step:literal),+ $(,)?], fini = [$($fini:literal),+ $(,)?],
		ret = $ret:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			#[inline]
			pub fn $slice_fn(self, a: &[$E]) -> $E {
				let r = unsafe { $inner(a.as_ptr(), a.len()) };
				let conv: fn(u64) -> $E = $ret;
				conv(r)
			}
		}

		/// # Safety
		/// Caller proved the feature via the token; `a` is valid for `n`
		/// elements.
		#[inline]
		#[target_feature(enable = $tf)]
		unsafe fn $inner(a: *const $E, n: usize) -> u64 {
			let r: u64;
			unsafe {
				core::arch::asm!(
					$($init,)+
					"mov {i}, #0",
					concat!("whilelo p0.", $t, ", {i}, {n}"),
					"b.none 3f",
					"2:",
					concat!($ld, " {{ z0.", $t, " }}, p0/z, [{a}, {i}", $sh, "]"),
					$($step,)+
					concat!($inc, " {i}"),
					concat!("whilelo p0.", $t, ", {i}, {n}"),
					"b.first 2b",
					"3:",
					$($fini,)+
					i = out(reg) _, n = in(reg) n, a = in(reg) a,
					init = in(reg) $bits, r = lateout(reg) r,
					out("v0") _, out("v3") _, out("p0") _, out("p1") _,
					options(nostack, readonly),
				);
			}
			r
		}
	};
}

/// Token-side `_slice` method over an [`sve_stream`] inner fn: length checks,
/// then one call. `ins` lists the input slices in call order; the inner fn
/// takes them in the same order, then `out`.
macro_rules! sve_slice {
	(
		token = $Token:ty, slice_fn = $slice_fn:ident, inner_fn = $inner:ident,
		ins = [$($x:ident: $X:ty),+ $(,)?], out = $O:ty,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// If any input slice's length differs from `out.len()`.
			#[inline]
			pub fn $slice_fn(self, $($x: &[$X],)+ out: &mut [$O]) {
				$(assert_eq!($x.len(), out.len());)+
				unsafe { $inner($($x.as_ptr(),)+ out.as_mut_ptr(), out.len()) }
			}
		}
	};
	(
		token = $Token:ty, slice_fn = $slice_fn:ident, inner_fn = $inner:ident,
		ins = [$($x:ident: $X:ty),+ $(,)?], out = $O:ty,
		imm = IMM, imm_check = $check:expr,
		slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $slice_doc]
			///
			/// # Panics
			/// `out.len() != a.len()`. An out-of-range `IMM` is a compile error.
			#[inline]
			pub fn $slice_fn<const IMM: i32>(self, $($x: &[$X],)+ out: &mut [$O]) {
				const { assert!($check, "IMM out of range") };
				$(assert_eq!($x.len(), out.len());)+
				unsafe { $inner::<IMM>($($x.as_ptr(),)+ out.as_mut_ptr(), out.len()) }
			}
		}
	};
}

/// Fixed-width method on a VL token (`add_i32x8` on `SveVl256`): the same
/// inner loop over an array. Exact on any hardware VL; the token's VL floor
/// only guarantees it is a single iteration.
macro_rules! sve_fixed {
	(
		token = $Token:ty, fixed_fn = $fixed_fn:ident, inner_fn = $inner:ident, lanes = $N:literal,
		ins = [$($x:ident: $X:ty),+ $(,)?], out = $O:ty,
		fixed_doc = $fixed_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, $($x: [$X; $N],)+) -> [$O; $N] {
				let mut out: [$O; $N] = [Default::default(); $N];
				unsafe { $inner($($x.as_ptr(),)+ out.as_mut_ptr(), $N) };
				out
			}
		}
	};
}

pub(crate) use {sve_fixed, sve_fold, sve_slice, sve_stream};
