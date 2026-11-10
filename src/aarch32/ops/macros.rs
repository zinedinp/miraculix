//! DSP/SIMD32 macros, local to aarch32: the ARMv6 packed ops live in plain
//! GPRs and have no A64 encoding, so they are not part of the Neon DSL in
//! `crate::arm_shared` (which the Neon op files here use). Written fresh,
//! not ported from `x86::ops::macros`; the **pattern** is the same as every
//! x86 token method: private `#[target_feature(enable = "...")]` impl fn,
//! public safe wrapper.

/// One `core::arch::arm` DSP/SIMD32 intrinsic taking two plain `i32`
/// registers and returning one, wrapped as a safe fixed-width method.
macro_rules! dsp_binop_i32 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: i32, b: i32) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			unsafe { imp(a, b) }
		}
	};
}

/// DSP/SIMD32 intrinsic on 4 packed `i8` lanes in a 32-bit GPR (upstream
/// `int8x4_t` = `i32` alias). Public API is `[i8; 4]`. Packing is
/// little-endian (lane 0 = low byte, ACLE); independent of host byte order.
macro_rules! dsp_binop_i8x4 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [i8; 4], b: [i8; 4]) -> [i8; 4] {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			let pack = |v: [i8; 4]| i32::from_le_bytes(v.map(|x| x as u8));
			let unpack = |v: i32| v.to_le_bytes().map(|x| x as i8);
			unpack(unsafe { imp(pack(a), pack(b)) })
		}
	};
}

/// One `core::arch::arm` DSP/SIMD32 intrinsic taking a single plain `i32`
/// register and returning one.
macro_rules! dsp_unop_i32 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: i32) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a) }
			}
			unsafe { imp(a) }
		}
	};
}

/// DSP/SIMD32 intrinsic on 2 packed `i16` lanes (`int16x2_t` = `i32`).
/// `[i16; 2]` in/out; lane 0 = low halfword (ACLE), same packing as
/// [`dsp_binop_i8x4`]. Cross-lane ops (`__qasx`/etc.) still use this shape.
macro_rules! dsp_binop_i16x2 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [i16; 2], b: [i16; 2]) -> [i16; 2] {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			let pack = |v: [i16; 2]| ((v[1] as u16 as i32) << 16) | (v[0] as u16 as i32);
			let unpack = |v: i32| [v as i16, (v >> 16) as i16];
			unpack(unsafe { imp(pack(a), pack(b)) })
		}
	};
}

/// As [`dsp_binop_i8x4`] for unsigned packed `u8` (`uint8x4_t` = `u32`),
/// e.g. `__usub8`.
macro_rules! dsp_binop_u8x4 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: u32, b: u32) -> u32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			unsafe { imp(u32::from_le_bytes(a), u32::from_le_bytes(b)) }.to_le_bytes()
		}
	};
}

/// Two packed `[i16; 2]` operands folded to one `i32` scalar (16-bit or dual
/// multiply; halfword choice is in `$intrinsic`, e.g. `__smultb`/`__smuad`).
macro_rules! dsp_mul16x2 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [i16; 2], b: [i16; 2]) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			let pack = |v: [i16; 2]| ((v[1] as u16 as i32) << 16) | (v[0] as u16 as i32);
			unsafe { imp(pack(a), pack(b)) }
		}
	};
}

/// As [`dsp_mul16x2`], plus a 32-bit accumulator `c` folded into the result
/// (`__smlabt`-family multiply-accumulate, `__smlad`/`__smlsd` dual-multiply
/// with accumulate).
macro_rules! dsp_mla16x2 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [i16; 2], b: [i16; 2], c: i32) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32, c: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b, c) }
			}
			let pack = |v: [i16; 2]| ((v[1] as u16 as i32) << 16) | (v[0] as u16 as i32);
			unsafe { imp(pack(a), pack(b), c) }
		}
	};
}

/// Full `i32` `a` times one halfword of packed `[i16; 2]` `b` (`__smulwb`/
/// `__smulwt`), unlike [`dsp_mul16x2`] where both operands are packed.
macro_rules! dsp_mulw {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: i32, b: [i16; 2]) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			let pack = |v: [i16; 2]| ((v[1] as u16 as i32) << 16) | (v[0] as u16 as i32);
			unsafe { imp(a, pack(b)) }
		}
	};
}

/// As [`dsp_mulw`], plus a 32-bit accumulator `c` folded into the result
/// (`__smlawb`/`__smlawt`).
macro_rules! dsp_mlaw {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: i32, b: [i16; 2], c: i32) -> i32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32, c: i32) -> i32 {
				unsafe { core::arch::arm::$intrinsic(a, b, c) }
			}
			let pack = |v: [i16; 2]| ((v[1] as u16 as i32) << 16) | (v[0] as u16 as i32);
			unsafe { imp(a, pack(b), c) }
		}
	};
}

/// `__usad8`/`__usada8` shape: 4 packed `u8` in, one `u32` SAD out.
macro_rules! dsp_sad_u8x4 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u8; 4], b: [u8; 4]) -> u32 {
			#[target_feature(enable = "dsp")]
			unsafe fn imp(a: i32, b: i32) -> u32 {
				unsafe { core::arch::arm::$intrinsic(a, b) }
			}
			unsafe { imp(i32::from_le_bytes(a), i32::from_le_bytes(b)) }
		}
	};
}

/// Whole-slice form of a packed DSP binop: chunk by `$lanes`, call the
/// fixed-width method per chunk, finish the tail with `$scalar`. The fixed
/// method is a GPR op (no vector state to keep live), so calling it per chunk
/// costs nothing that one big `#[target_feature]` body would save.
macro_rules! dsp_binop_slice {
	($(#[$doc:meta])* $slice_fn:ident, $fixed_fn:ident, $Elem:ty, $lanes:literal, $scalar:expr) => {
		$(#[$doc])*
		///
		#[doc = concat!("A tail shorter than ", stringify!($lanes), " lanes is finished scalar-wise.")]
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`.
		pub fn $slice_fn(self, a: &[$Elem], b: &[$Elem], out: &mut [$Elem]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			let (a_chunks, a_rem) = a.as_chunks::<$lanes>();
			let (b_chunks, b_rem) = b.as_chunks::<$lanes>();
			let (out_chunks, out_rem) = out.as_chunks_mut::<$lanes>();
			for ((x, y), o) in a_chunks.iter().zip(b_chunks).zip(out_chunks.iter_mut()) {
				*o = self.$fixed_fn(*x, *y);
			}
			let op: fn($Elem, $Elem) -> $Elem = $scalar;
			for ((&x, &y), o) in a_rem.iter().zip(b_rem).zip(out_rem) {
				*o = op(x, y);
			}
		}
	};
}

/// As [`dsp_binop_slice`] for the cross add-subtract ops (`qasx`/`qsax`/
/// `sasx`): they work on whole `[i16; 2]` pairs, so there is no tail. An odd
/// length panics instead of dropping or misreading the last element.
macro_rules! dsp_cross_slice {
	($(#[$doc:meta])* $slice_fn:ident, $fixed_fn:ident) => {
		$(#[$doc])*
		///
		/// # Panics
		/// `a.len() != b.len() || out.len() != a.len()`, or an odd length.
		pub fn $slice_fn(self, a: &[i16], b: &[i16], out: &mut [i16]) {
			assert_eq!(a.len(), b.len());
			assert_eq!(out.len(), a.len());
			assert_eq!(a.len() % 2, 0, "cross add-subtract needs pairs; odd length has no defined remainder");
			let (a_chunks, _) = a.as_chunks::<2>();
			let (b_chunks, _) = b.as_chunks::<2>();
			let (out_chunks, _) = out.as_chunks_mut::<2>();
			for ((x, y), o) in a_chunks.iter().zip(b_chunks).zip(out_chunks.iter_mut()) {
				*o = self.$fixed_fn(*x, *y);
			}
		}
	};
}

pub(crate) use {
    dsp_binop_i8x4, dsp_binop_i16x2, dsp_binop_i32, dsp_binop_slice, dsp_binop_u8x4,
    dsp_cross_slice, dsp_mla16x2, dsp_mlaw, dsp_mul16x2, dsp_mulw, dsp_sad_u8x4, dsp_unop_i32,
};
