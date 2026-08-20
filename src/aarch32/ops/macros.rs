//! Local macros for AArch32 token-gated ops. Written fresh, not ported from
//! `x86::ops::macros`: DSP/SIMD32's plain-GPR shape and Neon's
//! vector-register shape don't fit x86's `_mm_*`-oriented macros. The
//! **pattern** stays the same as every x86 token method: private
//! `#[target_feature(enable = "...")]` impl fn, public safe wrapper.

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

/// Neon binop on a 4-lane 128-bit vector: safe `[T; 4]` via loadu -> op ->
/// storeu. On arm the impl enables both `"neon"` and `"v7"` (aarch64 only
/// needs `"neon"`).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_binop_x4 {
	($(#[$doc:meta])* $name:ident, $elem:ty, $load:ident, $store:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [$elem; 4], b: [$elem; 4]) -> [$elem; 4] {
			#[target_feature(enable = "neon")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			unsafe fn imp(a: [$elem; 4], b: [$elem; 4]) -> [$elem; 4] {
				let av = unsafe { core::arch::arm::$load(a.as_ptr()) };
				let bv = unsafe { core::arch::arm::$load(b.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av, bv);
				let mut out = [0 as $elem; 4];
				unsafe { core::arch::arm::$store(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b) }
		}
	};
}

/// One `core::arch::arm` Neon unop on a 4-lane 128-bit vector (same
/// `loadu -> intrinsic -> storeu` bridge as [`neon_binop_x4`], one operand).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_unop_x4 {
	($(#[$doc:meta])* $name:ident, $elem:ty, $load:ident, $store:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [$elem; 4]) -> [$elem; 4] {
			#[target_feature(enable = "neon")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			unsafe fn imp(a: [$elem; 4]) -> [$elem; 4] {
				let av = unsafe { core::arch::arm::$load(a.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av);
				let mut out = [0 as $elem; 4];
				unsafe { core::arch::arm::$store(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a) }
		}
	};
}

/// Neon compare on 4-lane vectors: `$elem` in, `[u32; 4]` lane mask out
/// (all-1s or 0, not `bool`; same as x86 compares).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_cmp_x4 {
	($(#[$doc:meta])* $name:ident, $elem:ty, $load:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [$elem; 4], b: [$elem; 4]) -> [u32; 4] {
			#[target_feature(enable = "neon")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			unsafe fn imp(a: [$elem; 4], b: [$elem; 4]) -> [u32; 4] {
				let av = unsafe { core::arch::arm::$load(a.as_ptr()) };
				let bv = unsafe { core::arch::arm::$load(b.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av, bv);
				let mut out = [0u32; 4];
				unsafe { core::arch::arm::vst1q_u32(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b) }
		}
	};
}

/// Neon ternary on 4-lane vectors (same bridge as [`neon_binop_x4`]);
/// covers `vbslq_*` and `vfmaq_*` (`(a,b,c) -> result`).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_ternop_x4 {
	($(#[$doc:meta])* $name:ident, $elem:ty, $load:ident, $store:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [$elem; 4], b: [$elem; 4], c: [$elem; 4]) -> [$elem; 4] {
			#[target_feature(enable = "neon")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			unsafe fn imp(a: [$elem; 4], b: [$elem; 4], c: [$elem; 4]) -> [$elem; 4] {
				let av = unsafe { core::arch::arm::$load(a.as_ptr()) };
				let bv = unsafe { core::arch::arm::$load(b.as_ptr()) };
				let cv = unsafe { core::arch::arm::$load(c.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(av, bv, cv);
				let mut out = [0 as $elem; 4];
				unsafe { core::arch::arm::$store(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b, c) }
		}
	};
}

/// FullFP16 Neon binop on `float16x8_t` as safe `[u16; 8]` bit patterns
/// (see [`super::fp16`]). loadu -> reinterpret -> op -> reinterpret -> storeu.
/// Enables `"neon,fp16"` plus arm `"v7"`/`"v8"`; reinterprets need only neon+v7.
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_binop_f16x8 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
			#[target_feature(enable = "neon,fp16")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
				let av = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(a.as_ptr()) });
				let bv = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(b.as_ptr()) });
				let rv = core::arch::arm::vreinterpretq_u16_f16(core::arch::arm::$intrinsic(av, bv));
				let mut out = [0u16; 8];
				unsafe { core::arch::arm::vst1q_u16(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b) }
		}
	};
}

/// As [`neon_binop_f16x8`], one operand (abs/neg-shaped).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_unop_f16x8 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u16; 8]) -> [u16; 8] {
			#[target_feature(enable = "neon,fp16")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(a: [u16; 8]) -> [u16; 8] {
				let av = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(a.as_ptr()) });
				let rv = core::arch::arm::vreinterpretq_u16_f16(core::arch::arm::$intrinsic(av));
				let mut out = [0u16; 8];
				unsafe { core::arch::arm::vst1q_u16(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a) }
		}
	};
}

/// As [`neon_binop_f16x8`], but the intrinsic returns `uint16x8_t` lane
/// masks (16-bit wide, like [`neon_cmp_x4`]; no output reinterpret).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_cmp_f16x8 {
	($(#[$doc:meta])* $name:ident, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
			#[target_feature(enable = "neon,fp16")]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
				let av = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(a.as_ptr()) });
				let bv = core::arch::arm::vreinterpretq_f16_u16(unsafe { core::arch::arm::vld1q_u16(b.as_ptr()) });
				let rv = core::arch::arm::$intrinsic(av, bv);
				let mut out = [0u16; 8];
				unsafe { core::arch::arm::vst1q_u16(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(a, b) }
		}
	};
}

/// 8-bit dot / I8mm matmul: `[$acc;4]` + two 16-lane byte vectors -> new
/// `[$acc;4]`. `$extra` is `"dotprod"` or `"i8mm"` stacked on `"neon"`
/// (same `$extra:literal` style as crypto's `crypto_binop_u8x16!`).
#[cfg(any(target_feature = "v7", doc))]
macro_rules! neon_dot_acc_x4 {
	($(#[$doc:meta])* $name:ident, $acc:ty, $a:ty, $b:ty, $acc_load:ident, $acc_store:ident, $a_load:ident, $b_load:ident, $extra:literal, $intrinsic:ident) => {
		$(#[$doc])*
		#[inline]
		pub fn $name(self, acc: [$acc; 4], a: [$a; 16], b: [$b; 16]) -> [$acc; 4] {
			#[target_feature(enable = "neon")]
			#[target_feature(enable = $extra)]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
			#[cfg_attr(target_arch = "arm", target_feature(enable = "v8"))]
			unsafe fn imp(acc: [$acc; 4], a: [$a; 16], b: [$b; 16]) -> [$acc; 4] {
				let accv = unsafe { core::arch::arm::$acc_load(acc.as_ptr()) };
				let av = unsafe { core::arch::arm::$a_load(a.as_ptr()) };
				let bv = unsafe { core::arch::arm::$b_load(b.as_ptr()) };
				let rv = core::arch::arm::$intrinsic(accv, av, bv);
				let mut out = [0 as $acc; 4];
				unsafe { core::arch::arm::$acc_store(out.as_mut_ptr(), rv) };
				out
			}
			unsafe { imp(acc, a, b) }
		}
	};
}

pub(crate) use {
	dsp_binop_i16x2, dsp_binop_i32, dsp_binop_i8x4, dsp_binop_u8x4, dsp_mla16x2, dsp_mlaw, dsp_mul16x2, dsp_mulw,
	dsp_sad_u8x4, dsp_unop_i32,
};
#[cfg(any(target_feature = "v7", doc))]
pub(crate) use {
	neon_binop_f16x8, neon_binop_x4, neon_cmp_f16x8, neon_cmp_x4, neon_dot_acc_x4, neon_ternop_x4, neon_unop_f16x8,
	neon_unop_x4,
};
