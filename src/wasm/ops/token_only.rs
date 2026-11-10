//! Fixed-width ops with no elementwise slice meaning (lanes interact, or
//! the result width differs), so no `_slice` and no `auto` fn. Same
//! "tier-unique stays off `auto`" policy as x86 and aarch64.

use core::arch::wasm32::{i8x16_swizzle, i16x8_q15mulr_sat, i32x4_dot_i16x8};

use super::macros::{load, store};
use super::simd128::Simd128;

impl Simd128 {
    /// `i8x16.swizzle`: `out[i] = a[idx[i]]`, or 0 where `idx[i] >= 16`.
    #[inline]
    pub fn swizzle_u8x16(self, a: [u8; 16], idx: [u8; 16]) -> [u8; 16] {
        #[inline]
        #[target_feature(enable = "simd128")]
        fn imp(a: &[u8; 16], idx: &[u8; 16]) -> [u8; 16] {
            let mut o = [0u8; 16];
            store(&mut o, i8x16_swizzle(load(a), load(idx)));
            o
        }
        imp(&a, &idx)
    }

    /// `i32x4.dot_i16x8_s`: `out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]`, products
    /// widened to `i32`, the sum wrapping (only `-32768 * -32768` twice can
    /// overflow).
    #[inline]
    pub fn dot_i16x8(self, a: [i16; 8], b: [i16; 8]) -> [i32; 4] {
        #[inline]
        #[target_feature(enable = "simd128")]
        fn imp(a: &[i16; 8], b: &[i16; 8]) -> [i32; 4] {
            let mut o = [0i32; 4];
            store(&mut o, i32x4_dot_i16x8(load(a), load(b)));
            o
        }
        imp(&a, &b)
    }

    /// `i16x8.q15mulr_sat_s`: Q15 fixed-point multiply,
    /// `sat((a * b + 0x4000) >> 15)`.
    #[inline]
    pub fn q15mulr_sat_i16x8(self, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        #[inline]
        #[target_feature(enable = "simd128")]
        fn imp(a: &[i16; 8], b: &[i16; 8]) -> [i16; 8] {
            let mut o = [0i16; 8];
            store(&mut o, i16x8_q15mulr_sat(load(a), load(b)));
            o
        }
        imp(&a, &b)
    }
}

#[cfg(any(feature = "wasm-relaxed-simd", target_feature = "relaxed-simd"))]
mod relaxed_only {
    use core::arch::wasm32::{
        f32x4_relaxed_max, f32x4_relaxed_min, i8x16_relaxed_laneselect, i8x16_relaxed_swizzle,
        i16x8_relaxed_dot_i8x16_i7x16, i16x8_relaxed_q15mulr, i32x4_relaxed_trunc_f32x4,
    };

    use super::super::macros::{load, store};
    use super::super::relaxed::RelaxedSimd;

    /// One relaxed op on whole fixed-width arrays.
    macro_rules! relaxed_fixed {
		($(#[$doc:meta])* $name:ident($($arg:ident: $In:ty),+) -> $Out:ty = $intr:ident) => {
			$(#[$doc])*
			#[inline]
			pub fn $name(self, $($arg: $In),+) -> $Out {
				#[inline]
				#[target_feature(enable = "simd128,relaxed-simd")]
				fn imp($($arg: &$In),+) -> $Out {
					let mut o: $Out = Default::default();
					store(&mut o, $intr($(load($arg)),+));
					o
				}
				imp($(&$arg),+)
			}
		};
	}

    impl RelaxedSimd {
        relaxed_fixed! {
            /// `f32x4.relaxed_min`: like `f32x4.min` where neither input is NaN
            /// and they are not `±0.0` of opposite sign; otherwise the engine
            /// picks either input (or NaN).
            relaxed_min_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] = f32x4_relaxed_min
        }
        relaxed_fixed! {
            /// `f32x4.relaxed_max`: the `max` counterpart of
            /// [`Self::relaxed_min_f32x4`].
            relaxed_max_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] = f32x4_relaxed_max
        }
        relaxed_fixed! {
            /// `i8x16.relaxed_laneselect`: `a` where the mask byte is all-ones,
            /// `b` where it is zero. wasm's operand order, the reverse of
            /// `select_*`. Any other mask byte: engine-defined (per bit or by
            /// the top bit).
            relaxed_laneselect_u8x16(a: [u8; 16], b: [u8; 16], mask: [u8; 16]) -> [u8; 16] = i8x16_relaxed_laneselect
        }
        relaxed_fixed! {
            /// `i8x16.relaxed_swizzle`: `a[idx[i]]` for `idx[i] < 16`; any other
            /// index gives 0 or `a[idx[i] % 16]`, the engine picks.
            relaxed_swizzle_u8x16(a: [u8; 16], idx: [u8; 16]) -> [u8; 16] = i8x16_relaxed_swizzle
        }
        relaxed_fixed! {
            /// `i32x4.relaxed_trunc_f32x4_s`: truncate to `i32`. NaN and
            /// out-of-range lanes are engine-defined (`trunc_sat` result or
            /// `i32::MIN`).
            relaxed_trunc_f32x4(a: [f32; 4]) -> [i32; 4] = i32x4_relaxed_trunc_f32x4
        }
        relaxed_fixed! {
            /// `i16x8.relaxed_q15mulr_s`: Q15 multiply; `-32768 * -32768` is
            /// engine-defined (saturated or wrapped).
            relaxed_q15mulr_i16x8(a: [i16; 8], b: [i16; 8]) -> [i16; 8] = i16x8_relaxed_q15mulr
        }
        relaxed_fixed! {
            /// `i16x8.relaxed_dot_i8x16_i7x16_s`: pairwise `a[2i]*b[2i] +
            /// a[2i+1]*b[2i+1]` to `i16`. Exact when every `b` byte is in
            /// `0..=127`; a set top bit in `b` is engine-defined (signed or
            /// unsigned).
            relaxed_dot_i8x16_i7x16(a: [i8; 16], b: [i8; 16]) -> [i16; 8] = i16x8_relaxed_dot_i8x16_i7x16
        }
    }
}
