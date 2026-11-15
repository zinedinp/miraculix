//! The wasm wrapper DSL: token declaration plus one rule per op shape, each
//! emitting a fixed-width method (`add_i32x4`) and a `_slice` method
//! (`add_i32_slice`). Same key=value grammar and field names as the Arm DSL
//! in `crate::arm_shared::macros`, so the two can merge later.
//!
//! Differences from the Arm rules:
//!
//! - **One vector type.** Every lane shape lives in `v128`, so there is no
//!   `vec`/`loadu`/`storeu` key: [`load`] and [`store`] handle every
//!   `[T; N]` that is 16 bytes.
//! - **Tail through the vector op** (as pulp's `partial_load`/
//!   `partial_store`): the last `len % lanes` elements are copied into a
//!   zero-padded array, run through the same intrinsic, and copied back. The
//!   tail is therefore bit-identical to the body, needs no scalar formula
//!   (no closure inside a `#[target_feature]` body), and `sqrt` or the
//!   rounding ops work although `core` has no `f32::sqrt`.
//! - **No `unsafe` at the call.** rustc lets safe code call a
//!   `#[target_feature]` fn on wasm: running an opcode the engine lacks is
//!   impossible, because such a module fails validation before any code
//!   runs. The token is still the gate that picks the SIMD path; only
//!   `v128_load`/`v128_store` need `unsafe` (raw pointers).
//!
//! Each rule keeps the whole slice loop inside one `#[target_feature]` fn:
//! rustc will not inline a `#[target_feature]` callee into a caller without
//! the attribute, so a per-chunk call would leave a real `call` in the loop.

use core::arch::wasm32::{v128, v128_load, v128_store};

/// Plain numeric lane types that fill a `v128` evenly. Every bit pattern is
/// a valid value, which is what makes [`store`] sound.
pub(crate) trait Lane: Copy + Default {
    /// Identity of `+` for [`wasm_sum`]: `0`, or `-0.0` for floats
    /// (`-0.0 + x == x` for every `x`, including `x == -0.0`).
    const SUM_ID: Self;
    /// Wrapping `+` for integers, IEEE `+` for floats.
    fn lane_add(self, other: Self) -> Self;
}

macro_rules! lane_int {
	($($T:ty),*) => {$(
		impl Lane for $T {
			const SUM_ID: Self = 0;
			#[inline]
			fn lane_add(self, other: Self) -> Self {
				self.wrapping_add(other)
			}
		}
	)*};
}
lane_int!(i8, u8, i16, u16, i32, u32, i64, u64);

macro_rules! lane_float {
	($($T:ty),*) => {$(
		impl Lane for $T {
			const SUM_ID: Self = -0.0;
			#[inline]
			fn lane_add(self, other: Self) -> Self {
				self + other
			}
		}
	)*};
}
lane_float!(f32, f64);

/// `[T; N]` (16 bytes) into a `v128`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn load<T: Lane, const N: usize>(x: &[T; N]) -> v128 {
    const { assert!(size_of::<[T; N]>() == 16, "a v128 holds exactly 16 bytes") };
    // SAFETY: `x` is 16 readable bytes (asserted above). `v128_load` has no
    // alignment requirement.
    unsafe { v128_load(x.as_ptr().cast()) }
}

/// `v128` into `[T; N]` (16 bytes).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn store<T: Lane, const N: usize>(x: &mut [T; N], v: v128) {
    const { assert!(size_of::<[T; N]>() == 16, "a v128 holds exactly 16 bytes") };
    // SAFETY: `x` is 16 writable bytes (asserted above), and every bit
    // pattern is a valid `T` (`Lane` is only implemented for plain integers
    // and floats). No alignment requirement.
    unsafe { v128_store(x.as_mut_ptr().cast(), v) }
}

/// A tail shorter than `N`, padded with `fill` to a full vector.
#[inline]
pub(crate) fn pad<T: Lane, const N: usize>(rem: &[T], fill: T) -> [T; N] {
    let mut p = [fill; N];
    p[..rem.len()].copy_from_slice(rem);
    p
}

/// Token: zero-sized proof type plus `detect` / `from_features`. Same shape
/// as `arm_token!`; `Feature` / `FeatureSet` are `crate::wasm`'s.
macro_rules! wasm_token {
	(
		$(#[$doc:meta])*
		token = $Token:ident, requires = [$($F:ident),+ $(,)?],
	) => {
		$(#[$doc])*
		///
		/// Zero-sized, `Copy`. Obtain via `detect()` or `from_features()`, then
		/// call methods on the token.
		#[derive(Debug, Clone, Copy)]
		pub struct $Token(());

		impl $Token {
			/// `Some(token)` if the feature is compiled in (`cfg`) or switched on
			/// at run time (`crate::wasm::enable_simd128` and friends).
			pub fn detect() -> Option<Self> {
				Self::from_features($crate::wasm::detect_features())
			}

			/// Build a token from a set you already have (e.g. from
			/// `crate::wasm::detect_features()`).
			#[doc = concat!("Returns `None` unless every one of ", $("`Feature::", stringify!($F), "` "),+, "is present.")]
			pub fn from_features(set: $crate::wasm::FeatureSet) -> Option<Self> {
				set.contains_all(&[$($crate::wasm::Feature::$F),+]).then_some($Token(()))
			}
		}
	};
}

/// 2-op: `[$Elem; $lanes]` x 2 in, `[$Out; $lanes]` out. `$Out` is `$Elem`
/// for arithmetic and the unsigned mask type for compares.
macro_rules! wasm_binop {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, out = $Out:ty, intrinsic = $intr:path,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $L], b: [$Elem; $L]) -> [$Out; $L] {
                #[inline]
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem; $L], b: &[$Elem; $L]) -> [$Out; $L] {
                    use $crate::wasm::ops::macros::{load, store};
                    let mut o = [<$Out>::default(); $L];
                    store(&mut o, $intr(load(a), load(b)));
                    o
                }
                imp(&a, &b)
            }

            #[doc = $slice_doc]
            ///
            /// # Panics
            /// `a.len() != b.len() || out.len() != a.len()`.
            pub fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Out]) {
                assert_eq!(a.len(), b.len());
                assert_eq!(out.len(), a.len());
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem], b: &[$Elem], out: &mut [$Out]) {
                    use $crate::wasm::ops::macros::{load, pad, store};
                    let (ac, ar) = a.as_chunks::<$L>();
                    let (bc, br) = b.as_chunks::<$L>();
                    let (oc, or) = out.as_chunks_mut::<$L>();
                    for ((x, y), o) in ac.iter().zip(bc).zip(oc.iter_mut()) {
                        store(o, $intr(load(x), load(y)));
                    }
                    if !ar.is_empty() {
                        let mut o = [<$Out>::default(); $L];
                        let x: [$Elem; $L] = pad(ar, Default::default());
                        let y: [$Elem; $L] = pad(br, Default::default());
                        store(&mut o, $intr(load(&x), load(&y)));
                        or.copy_from_slice(&o[..ar.len()]);
                    }
                }
                imp(a, b, out)
            }
        }
    };
}

/// 1-op: `[$Elem; $lanes]` in, `[$Out; $lanes]` out.
macro_rules! wasm_unop {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, out = $Out:ty, intrinsic = $intr:path,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $L]) -> [$Out; $L] {
                #[inline]
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem; $L]) -> [$Out; $L] {
                    use $crate::wasm::ops::macros::{load, store};
                    let mut o = [<$Out>::default(); $L];
                    store(&mut o, $intr(load(a)));
                    o
                }
                imp(&a)
            }

            #[doc = $slice_doc]
            ///
            /// # Panics
            /// `out.len() != a.len()`.
            pub fn $slice_fn(self, a: &[$Elem], out: &mut [$Out]) {
                assert_eq!(out.len(), a.len());
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem], out: &mut [$Out]) {
                    use $crate::wasm::ops::macros::{load, pad, store};
                    let (ac, ar) = a.as_chunks::<$L>();
                    let (oc, or) = out.as_chunks_mut::<$L>();
                    for (x, o) in ac.iter().zip(oc.iter_mut()) {
                        store(o, $intr(load(x)));
                    }
                    if !ar.is_empty() {
                        let mut o = [<$Out>::default(); $L];
                        let x: [$Elem; $L] = pad(ar, Default::default());
                        store(&mut o, $intr(load(&x)));
                        or.copy_from_slice(&o[..ar.len()]);
                    }
                }
                imp(a, out)
            }
        }
    };
}

/// Shift every lane by the constant `IMM`. `IMM >= $bits` is a compile
/// error: wasm would take it modulo the lane width, x86 would give 0, so the
/// crate refuses both readings instead of picking one.
macro_rules! wasm_shift {
	(
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, bits = $bits:literal, intrinsic = $intr:path,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
		impl $Token {
			#[doc = $fixed_doc]
			///
			#[doc = concat!("`IMM` must be below ", stringify!($bits), " (compile error otherwise).")]
			#[inline]
			pub fn $fixed_fn<const IMM: u32>(self, a: [$Elem; $L]) -> [$Elem; $L] {
				const { assert!(IMM < $bits, "shift count must be below the lane width") };
				#[inline]
				#[target_feature(enable = $tf)]
				fn imp(a: &[$Elem; $L], n: u32) -> [$Elem; $L] {
					use $crate::wasm::ops::macros::{load, store};
					let mut o = [<$Elem>::default(); $L];
					store(&mut o, $intr(load(a), n));
					o
				}
				imp(&a, IMM)
			}

			#[doc = $slice_doc]
			///
			#[doc = concat!("`IMM` must be below ", stringify!($bits), " (compile error otherwise).")]
			///
			/// # Panics
			/// `out.len() != a.len()`.
			pub fn $slice_fn<const IMM: u32>(self, a: &[$Elem], out: &mut [$Elem]) {
				const { assert!(IMM < $bits, "shift count must be below the lane width") };
				assert_eq!(out.len(), a.len());
				#[target_feature(enable = $tf)]
				fn imp(a: &[$Elem], out: &mut [$Elem], n: u32) {
					use $crate::wasm::ops::macros::{load, pad, store};
					let (ac, ar) = a.as_chunks::<$L>();
					let (oc, or) = out.as_chunks_mut::<$L>();
					for (x, o) in ac.iter().zip(oc.iter_mut()) {
						store(o, $intr(load(x), n));
					}
					if !ar.is_empty() {
						let mut o = [<$Elem>::default(); $L];
						let x: [$Elem; $L] = pad(ar, Default::default());
						store(&mut o, $intr(load(&x), n));
						or.copy_from_slice(&o[..ar.len()]);
					}
				}
				imp(a, out, IMM)
			}
		}
	};
}

/// Per-bit select (`v128.bitselect`): `b` where the mask bit is set, `a`
/// where it is clear. Argument order `(a, b, mask)`, as in `auto`.
macro_rules! wasm_select {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, mask = $Mask:ty,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(
                self,
                a: [$Elem; $L],
                b: [$Elem; $L],
                mask: [$Mask; $L],
            ) -> [$Elem; $L] {
                #[inline]
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem; $L], b: &[$Elem; $L], m: &[$Mask; $L]) -> [$Elem; $L] {
                    use $crate::wasm::ops::macros::{load, store};
                    let mut o = [<$Elem>::default(); $L];
                    store(
                        &mut o,
                        core::arch::wasm32::v128_bitselect(load(b), load(a), load(m)),
                    );
                    o
                }
                imp(&a, &b, &mask)
            }

            #[doc = $slice_doc]
            ///
            /// # Panics
            /// Length mismatch among `a`, `b`, `mask`, `out`.
            pub fn $slice_fn(self, a: &[$Elem], b: &[$Elem], mask: &[$Mask], out: &mut [$Elem]) {
                assert_eq!(a.len(), b.len());
                assert_eq!(mask.len(), a.len());
                assert_eq!(out.len(), a.len());
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem], b: &[$Elem], m: &[$Mask], out: &mut [$Elem]) {
                    use core::arch::wasm32::v128_bitselect;
                    use $crate::wasm::ops::macros::{load, pad, store};
                    let (ac, ar) = a.as_chunks::<$L>();
                    let (bc, br) = b.as_chunks::<$L>();
                    let (mc, mr) = m.as_chunks::<$L>();
                    let (oc, or) = out.as_chunks_mut::<$L>();
                    for (((x, y), k), o) in ac.iter().zip(bc).zip(mc).zip(oc.iter_mut()) {
                        store(o, v128_bitselect(load(y), load(x), load(k)));
                    }
                    if !ar.is_empty() {
                        let mut o = [<$Elem>::default(); $L];
                        let x: [$Elem; $L] = pad(ar, Default::default());
                        let y: [$Elem; $L] = pad(br, Default::default());
                        let k: [$Mask; $L] = pad(mr, Default::default());
                        store(&mut o, v128_bitselect(load(&y), load(&x), load(&k)));
                        or.copy_from_slice(&o[..ar.len()]);
                    }
                }
                imp(a, b, mask, out)
            }
        }
    };
}

/// 3-op (the FMA family): `$intr(a, b, c)` with `a * b` first, `c` last.
macro_rules! wasm_ternop {
    (
		token = $Token:ty, target_feature = $tf:literal,
		fixed_fn = $fixed_fn:ident, slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, intrinsic = $intr:path,
		fixed_doc = $fixed_doc:literal, slice_doc = $slice_doc:literal,
	) => {
        impl $Token {
            #[doc = $fixed_doc]
            #[inline]
            pub fn $fixed_fn(self, a: [$Elem; $L], b: [$Elem; $L], c: [$Elem; $L]) -> [$Elem; $L] {
                #[inline]
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem; $L], b: &[$Elem; $L], c: &[$Elem; $L]) -> [$Elem; $L] {
                    use $crate::wasm::ops::macros::{load, store};
                    let mut o = [<$Elem>::default(); $L];
                    store(&mut o, $intr(load(a), load(b), load(c)));
                    o
                }
                imp(&a, &b, &c)
            }

            #[doc = $slice_doc]
            ///
            /// # Panics
            /// `a`, `b`, `c` and `out` do not all have the same length.
            pub fn $slice_fn(self, a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
                assert_eq!(a.len(), b.len());
                assert_eq!(c.len(), a.len());
                assert_eq!(out.len(), a.len());
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem], b: &[$Elem], c: &[$Elem], out: &mut [$Elem]) {
                    use $crate::wasm::ops::macros::{load, pad, store};
                    let (ac, ar) = a.as_chunks::<$L>();
                    let (bc, br) = b.as_chunks::<$L>();
                    let (cc, cr) = c.as_chunks::<$L>();
                    let (oc, or) = out.as_chunks_mut::<$L>();
                    for (((x, y), z), o) in ac.iter().zip(bc).zip(cc).zip(oc.iter_mut()) {
                        store(o, $intr(load(x), load(y), load(z)));
                    }
                    if !ar.is_empty() {
                        let mut o = [<$Elem>::default(); $L];
                        let x: [$Elem; $L] = pad(ar, Default::default());
                        let y: [$Elem; $L] = pad(br, Default::default());
                        let z: [$Elem; $L] = pad(cr, Default::default());
                        store(&mut o, $intr(load(&x), load(&y), load(&z)));
                        or.copy_from_slice(&o[..ar.len()]);
                    }
                }
                imp(a, b, c, out)
            }
        }
    };
}

/// Sum of a whole slice: one vector accumulator (lane `j` sums elements
/// `j, j + lanes, ...`), then the lanes left to right. The tail is padded
/// with [`Lane::SUM_ID`], so it changes nothing but the elements it holds.
/// Integers wrap. Floats are grouped by lane, so not bit-identical to a
/// plain left-to-right fold; `auto_down` uses the same grouping.
macro_rules! wasm_sum {
    (
		token = $Token:ty, target_feature = $tf:literal,
		slice_fn = $slice_fn:ident,
		lanes = $L:literal, elem = $Elem:ty, intrinsic = $intr:path,
		slice_doc = $slice_doc:literal,
	) => {
        impl $Token {
            #[doc = $slice_doc]
            pub fn $slice_fn(self, a: &[$Elem]) -> $Elem {
                #[target_feature(enable = $tf)]
                fn imp(a: &[$Elem]) -> $Elem {
                    use $crate::wasm::ops::macros::{Lane, load, pad, store};
                    let id = <$Elem as Lane>::SUM_ID;
                    let (ac, ar) = a.as_chunks::<$L>();
                    let mut acc = load(&[id; $L]);
                    for x in ac {
                        acc = $intr(acc, load(x));
                    }
                    if !ar.is_empty() {
                        let x: [$Elem; $L] = pad(ar, id);
                        acc = $intr(acc, load(&x));
                    }
                    let mut lanes = [id; $L];
                    store(&mut lanes, acc);
                    lanes.iter().fold(id, |s, &x| s.lane_add(x))
                }
                imp(a)
            }
        }
    };
}

pub(crate) use {
    wasm_binop, wasm_select, wasm_shift, wasm_sum, wasm_ternop, wasm_token, wasm_unop,
};
