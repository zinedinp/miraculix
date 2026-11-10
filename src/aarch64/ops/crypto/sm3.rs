//! ARMv8.2-A `FEAT_SM3`: Chinese national-standard SM3 hash steps. Token:
//! [`Sm3`].
//!
//! **Why `asm!`:** the SM3/SM4 intrinsics are still unstable
//! (`stdarch_neon_sm4`). LLVM's `sm4` target feature (stable) covers both
//! SM3 and SM4, while Linux reports two separate HWCAP bits - so this token
//! checks `Feature::Sm3` and enables `"sm4"`.
//!
//! The `sm3tt*` ops take a lane index `IMM` (`0..4`) selecting one 32-bit
//! word of `b`; it is dispatched to one of four fixed encodings because an
//! `asm!` template cannot splice a const generic.

use core::arch::asm;

use super::super::macros::arm_token;
use super::super::simd_arch::{uint32x4_t, vld1q_u32, vst1q_u32};

arm_token! {
    /// Proof that `FEAT_SM3` is available.
    token = Sm3, requires = [Sm3],
}

#[inline(always)]
unsafe fn ld(a: &[u32; 4]) -> uint32x4_t {
    unsafe { vld1q_u32(a.as_ptr()) }
}

#[inline(always)]
unsafe fn st(v: uint32x4_t) -> [u32; 4] {
    let mut out = [0u32; 4];
    unsafe { vst1q_u32(out.as_mut_ptr(), v) };
    out
}

impl Sm3 {
    /// `SS1` term of an SM3 round: lane 3 = `((a3 <<< 12) + b3 + c3) <<< 7`,
    /// lanes 0-2 = 0 (`sm3ss1 v.4s, v.4s, v.4s, v.4s`).
    #[inline]
    pub fn sm3ss1(self, a: [u32; 4], b: [u32; 4], c: [u32; 4]) -> [u32; 4] {
        unsafe { sm3ss1_fixed(&a, &b, &c) }
    }

    /// SM3 message expansion, part 1 (`sm3partw1 v.4s, v.4s, v.4s`).
    #[inline]
    pub fn sm3partw1(self, acc: [u32; 4], a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        unsafe { sm3partw1_fixed(&acc, &a, &b) }
    }

    /// SM3 message expansion, part 2 (`sm3partw2 v.4s, v.4s, v.4s`).
    #[inline]
    pub fn sm3partw2(self, acc: [u32; 4], a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        unsafe { sm3partw2_fixed(&acc, &a, &b) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "neon,sm4")]
unsafe fn sm3ss1_fixed(a: &[u32; 4], b: &[u32; 4], c: &[u32; 4]) -> [u32; 4] {
    unsafe {
        let r: uint32x4_t;
        asm!(
            "sm3ss1 {r:v}.4s, {a:v}.4s, {b:v}.4s, {c:v}.4s",
            r = lateout(vreg) r,
            a = in(vreg) ld(a),
            b = in(vreg) ld(b),
            c = in(vreg) ld(c),
            options(pure, nomem, nostack, preserves_flags),
        );
        st(r)
    }
}

macro_rules! sm3_acc3 {
	($intrinsic_fn:ident, $asm:literal) => {
		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,sm4")]
		unsafe fn $intrinsic_fn(acc: &[u32; 4], a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
			unsafe {
				let mut d = ld(acc);
				asm!($asm, d = inout(vreg) d, a = in(vreg) ld(a), b = in(vreg) ld(b), options(pure, nomem, nostack, preserves_flags));
				st(d)
			}
		}
	};
}

sm3_acc3!(sm3partw1_fixed, "sm3partw1 {d:v}.4s, {a:v}.4s, {b:v}.4s");
sm3_acc3!(sm3partw2_fixed, "sm3partw2 {d:v}.4s, {a:v}.4s, {b:v}.4s");

/// One `SM3TT*` op with its lane immediate dispatched to four encodings.
macro_rules! sm3tt {
	($fixed_fn:ident, $intrinsic_fn:ident, $mn:literal, $doc:literal) => {
		impl Sm3 {
			#[doc = $doc]
			///
			/// `IMM` selects the word of `b` (`0..4`); anything else fails to
			/// compile.
			#[inline]
			pub fn $fixed_fn<const IMM: i32>(self, acc: [u32; 4], a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
				const { assert!(IMM >= 0 && IMM < 4, "lane index must be 0..4") };
				unsafe { $intrinsic_fn::<IMM>(&acc, &a, &b) }
			}
		}

		/// # Safety
		/// Caller proved the feature via the token.
		#[inline]
		#[target_feature(enable = "neon,sm4")]
		unsafe fn $intrinsic_fn<const IMM: i32>(acc: &[u32; 4], a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
			unsafe {
				let mut d = ld(acc);
				let (va, vb) = (ld(a), ld(b));
				match IMM {
					0 => asm!(concat!($mn, " {d:v}.4s, {a:v}.4s, {b:v}.s[0]"), d = inout(vreg) d, a = in(vreg) va, b = in(vreg) vb, options(pure, nomem, nostack, preserves_flags)),
					1 => asm!(concat!($mn, " {d:v}.4s, {a:v}.4s, {b:v}.s[1]"), d = inout(vreg) d, a = in(vreg) va, b = in(vreg) vb, options(pure, nomem, nostack, preserves_flags)),
					2 => asm!(concat!($mn, " {d:v}.4s, {a:v}.4s, {b:v}.s[2]"), d = inout(vreg) d, a = in(vreg) va, b = in(vreg) vb, options(pure, nomem, nostack, preserves_flags)),
					_ => asm!(concat!($mn, " {d:v}.4s, {a:v}.4s, {b:v}.s[3]"), d = inout(vreg) d, a = in(vreg) va, b = in(vreg) vb, options(pure, nomem, nostack, preserves_flags)),
				}
				st(d)
			}
		}
	};
}

sm3tt!(
    sm3tt1a,
    sm3tt1a_fixed,
    "sm3tt1a",
    "SM3 `TT1` compression step, rounds 0-15 variant (`sm3tt1a v.4s, v.4s, v.s[IMM]`)."
);
sm3tt!(
    sm3tt1b,
    sm3tt1b_fixed,
    "sm3tt1b",
    "SM3 `TT1` compression step, rounds 16-63 variant (`sm3tt1b v.4s, v.4s, v.s[IMM]`)."
);
sm3tt!(
    sm3tt2a,
    sm3tt2a_fixed,
    "sm3tt2a",
    "SM3 `TT2` compression step, rounds 0-15 variant (`sm3tt2a v.4s, v.4s, v.s[IMM]`)."
);
sm3tt!(
    sm3tt2b,
    sm3tt2b_fixed,
    "sm3tt2b",
    "SM3 `TT2` compression step, rounds 16-63 variant (`sm3tt2b v.4s, v.4s, v.s[IMM]`)."
);

#[cfg(test)]
#[path = "../../test/ops/crypto/sm3.rs"]
mod tests;
