//! ARMv6 DSP/SIMD32: 16-bit multiply(-accumulate) and packed 8x4 integer
//! arithmetic in a plain 32-bit GPR (no vector regs). Token: [`Dsp`].
//! Upstream: `core::arch::arm::{dsp,simd32}` (`stdarch_arm_dsp`). Detect:
//! [`Feature::Edsp`] (HWCAP `EDSP`; one bit covers both stdarch files on real
//! hardware). ACLE: `Q` = saturating, `S` = signed wrapping.

use super::super::{Feature, FeatureSet};
use super::macros::{
	dsp_binop_i16x2, dsp_binop_i32, dsp_binop_i8x4, dsp_binop_u8x4, dsp_mla16x2, dsp_mlaw, dsp_mul16x2, dsp_mulw,
	dsp_sad_u8x4, dsp_unop_i32,
};

/// Proof that the ARMv6 DSP/SIMD32 extension is available. Zero-sized, `Copy`.
///
/// Obtain via [`Dsp::detect`] or [`Dsp::from_features`], then call methods
/// on the token.
#[derive(Debug, Clone, Copy)]
pub struct Dsp(());

impl Dsp {
	/// Probe once: `Some(token)` if the DSP extension is available, else `None`.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// Build a token from an existing [`crate::aarch32::FeatureSet`].
	///
	/// Returns `None` if `Feature::Edsp` is missing.
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Edsp).then_some(Dsp(()))
	}

	dsp_binop_i32!(
		/// `QADD`: 32-bit saturating signed addition.
		qadd,
		__qadd
	);
	dsp_binop_i32!(
		/// `QSUB`: 32-bit saturating signed subtraction.
		qsub,
		__qsub
	);

	dsp_binop_i8x4!(
		/// `QADD8`: 4-lane saturating signed `i8` addition, packed in one GPR.
		qadd8,
		__qadd8
	);
	dsp_binop_i8x4!(
		/// `QSUB8`: 4-lane saturating signed `i8` subtraction, packed in one GPR.
		qsub8,
		__qsub8
	);
	dsp_binop_i8x4!(
		/// `SADD8`: 4-lane wrapping signed `i8` addition, packed in one GPR.
		sadd8,
		__sadd8
	);
	dsp_binop_i8x4!(
		/// `SSUB8`: 4-lane wrapping signed `i8` subtraction, packed in one GPR.
		ssub8,
		__ssub8
	);
	dsp_binop_i8x4!(
		/// `SHADD8`: 4-lane halving signed `i8` addition (`(a+b)/2`, no
		/// saturation), packed in one GPR.
		shadd8,
		__shadd8
	);
	dsp_binop_i8x4!(
		/// `SHSUB8`: 4-lane halving signed `i8` subtraction (`(a-b)/2`, no
		/// saturation), packed in one GPR.
		shsub8,
		__shsub8
	);
	dsp_binop_u8x4!(
		/// `USUB8`: 4-lane wrapping unsigned `u8` subtraction, packed in one GPR.
		usub8,
		__usub8
	);
	dsp_sad_u8x4!(
		/// `USAD8`: sum of absolute differences of 4 unsigned `u8` lanes.
		usad8,
		__usad8
	);

	/// `USADA8`: sum of absolute differences of 4 unsigned `u8` lanes, plus
	/// accumulator `c`.
	#[inline]
	pub fn usada8(self, a: [u8; 4], b: [u8; 4], c: u32) -> u32 {
		#[target_feature(enable = "dsp")]
		unsafe fn imp(a: i32, b: i32, c: u32) -> u32 {
			unsafe { core::arch::arm::__usada8(a, b, c) }
		}
		unsafe { imp(i32::from_le_bytes(a), i32::from_le_bytes(b), c) }
	}

	/// `SADD8` then `SEL`, reading the `APSR.GE` flags `SADD8` just set.
	/// Fused into one `unsafe fn` because `__sel` has no explicit flags
	/// operand: LLVM models `APSR.GE` as an implicit physical-register
	/// dependency that only holds within one compiled function body.
	/// Returns `(sadd8(add_a, add_b), sel(sel_a, sel_b))`.
	#[inline]
	pub fn sel_after_sadd8(
		self,
		add_a: [i8; 4],
		add_b: [i8; 4],
		sel_a: [i8; 4],
		sel_b: [i8; 4],
	) -> ([i8; 4], [i8; 4]) {
		#[target_feature(enable = "dsp")]
		unsafe fn imp(add_a: i32, add_b: i32, sel_a: i32, sel_b: i32) -> (i32, i32) {
			let sum = unsafe { core::arch::arm::__sadd8(add_a, add_b) };
			let sel = unsafe { core::arch::arm::__sel(sel_a, sel_b) };
			(sum, sel)
		}
		let pack = |v: [i8; 4]| i32::from_le_bytes(v.map(|x| x as u8));
		let unpack = |v: i32| v.to_le_bytes().map(|x| x as i8);
		let (sum, sel) = unsafe { imp(pack(add_a), pack(add_b), pack(sel_a), pack(sel_b)) };
		(unpack(sum), unpack(sel))
	}

	dsp_binop_i16x2!(
		/// `QADD16`: 2-lane saturating signed `i16` addition, packed in one GPR.
		qadd16,
		__qadd16
	);
	dsp_binop_i16x2!(
		/// `QSUB16`: 2-lane saturating signed `i16` subtraction, packed in one GPR.
		qsub16,
		__qsub16
	);
	dsp_binop_i16x2!(
		/// `QASX`: saturating cross add-subtract (`res[0]=a[0]-b[1],
		/// res[1]=a[1]+b[0]`), packed in one GPR.
		qasx,
		__qasx
	);
	dsp_binop_i16x2!(
		/// `QSAX`: saturating cross subtract-add (`res[0]=a[0]+b[1],
		/// res[1]=a[1]-b[0]`), packed in one GPR.
		qsax,
		__qsax
	);
	dsp_binop_i16x2!(
		/// `SADD16`: 2-lane wrapping signed `i16` addition, packed in one GPR.
		sadd16,
		__sadd16
	);
	dsp_binop_i16x2!(
		/// `SASX`: wrapping cross add-subtract (`res[0]=a[0]-b[1],
		/// res[1]=a[1]+b[0]`), packed in one GPR.
		sasx,
		__sasx
	);
	dsp_binop_i16x2!(
		/// `SHADD16`: 2-lane halving signed `i16` addition (no saturation),
		/// packed in one GPR.
		shadd16,
		__shadd16
	);
	dsp_binop_i16x2!(
		/// `SHSUB16`: 2-lane halving signed `i16` subtraction (no
		/// saturation), packed in one GPR.
		shsub16,
		__shsub16
	);

	dsp_mul16x2!(
		/// `SMULBB`: signed 16-bit multiply of `a`'s and `b`'s low halfwords
		/// (`a[0] * b[0]`).
		smulbb,
		__smulbb
	);
	dsp_mul16x2!(
		/// `SMULTB`: signed 16-bit multiply of `a`'s high and `b`'s low
		/// halfword (`a[1] * b[0]`).
		smultb,
		__smultb
	);
	dsp_mul16x2!(
		/// `SMULBT`: signed 16-bit multiply of `a`'s low and `b`'s high
		/// halfword (`a[0] * b[1]`).
		smulbt,
		__smulbt
	);
	dsp_mul16x2!(
		/// `SMULTT`: signed 16-bit multiply of `a`'s and `b`'s high halfwords
		/// (`a[1] * b[1]`).
		smultt,
		__smultt
	);
	dsp_mulw!(
		/// `SMULWB`: signed multiply of `a` (full 32-bit) by `b`'s low
		/// halfword, top 32 bits of the 48-bit product.
		smulwb,
		__smulwb
	);
	dsp_mulw!(
		/// `SMULWT`: signed multiply of `a` (full 32-bit) by `b`'s high
		/// halfword, top 32 bits of the 48-bit product.
		smulwt,
		__smulwt
	);
	dsp_mul16x2!(
		/// `SMUAD`: dual signed 16-bit multiply with addition of products
		/// (`a[0]*b[0] + a[1]*b[1]`).
		smuad,
		__smuad
	);
	dsp_mul16x2!(
		/// `SMUADX`: dual signed 16-bit multiply (`b` exchanged) with
		/// addition of products (`a[0]*b[1] + a[1]*b[0]`).
		smuadx,
		__smuadx
	);
	dsp_mul16x2!(
		/// `SMUSD`: dual signed 16-bit multiply with subtraction of products
		/// (`a[0]*b[0] - a[1]*b[1]`).
		smusd,
		__smusd
	);
	dsp_mul16x2!(
		/// `SMUSDX`: dual signed 16-bit multiply (`b` exchanged) with
		/// subtraction of products (`a[0]*b[1] - a[1]*b[0]`).
		smusdx,
		__smusdx
	);

	dsp_mla16x2!(
		/// `SMLABB`: signed 16-bit multiply-accumulate of `a`'s and `b`'s low
		/// halfwords plus `c` (`a[0] * b[0] + c`). Sets the CPU `Q` sticky
		/// saturation flag on overflow; this wrapper only returns the wrapping
		/// numeric result.
		smlabb,
		__smlabb
	);
	dsp_mla16x2!(
		/// `SMLABT`: as [`Dsp::smlabb`], `a`'s low x `b`'s high halfword.
		smlabt,
		__smlabt
	);
	dsp_mla16x2!(
		/// `SMLATB`: as [`Dsp::smlabb`], `a`'s high x `b`'s low halfword.
		smlatb,
		__smlatb
	);
	dsp_mla16x2!(
		/// `SMLATT`: as [`Dsp::smlabb`], `a`'s and `b`'s high halfwords.
		smlatt,
		__smlatt
	);
	dsp_mlaw!(
		/// `SMLAWB`: `(a * b[0] + (c << 16)) >> 16`, `a` full 32-bit, `b[0]`
		/// the low halfword. Sets the `Q` flag on overflow (not exposed).
		smlawb,
		__smlawb
	);
	dsp_mlaw!(
		/// `SMLAWT`: `(a * b[1] + (c << 16)) >> 16`, `a` full 32-bit, `b[1]`
		/// the high halfword. Sets the `Q` flag on overflow (not exposed).
		smlawt,
		__smlawt
	);
	dsp_mla16x2!(
		/// `SMLAD`: dual signed 16-bit multiply with addition of products
		/// plus accumulator `c` (`a[0]*b[0] + a[1]*b[1] + c`).
		smlad,
		__smlad
	);
	dsp_mla16x2!(
		/// `SMLSD`: dual signed 16-bit multiply with subtraction of products
		/// plus accumulator `c` (`a[0]*b[0] - a[1]*b[1] + c`).
		smlsd,
		__smlsd
	);

	dsp_unop_i32!(
		/// `QDBL`: 32-bit saturating signed doubling (`a + a`, saturated).
		qdbl,
		__qdbl
	);
}

#[cfg(test)]
#[path = "../test/ops/dsp.rs"]
mod tests;
