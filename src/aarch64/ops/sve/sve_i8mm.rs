//! SVE `FEAT_I8MM` (v8.6): 8-bit integer matrix multiply-accumulate and the
//! mixed-sign dot product on `z` registers. Token: [`SveI8mm`].
//!
//! Same maths as the Neon [`I8mm`](super::super::neon::i8mm::I8mm) token,
//! over whole slices: every 128-bit segment is one independent 2×8 · 8×2
//! product. Segment `k` reads `a[16k..16k+16]`, `b[16k..16k+16]`,
//! `acc[4k..4k+4]` and writes `out[4k..4k+4]`, with Neon's row-major layout
//! (`out[2i+j] = acc[2i+j] + Σₖ a[8i+k]·b[8j+k]`). Accumulation wraps modulo
//! 2³², as the instructions do.
//!
//! The byte operands are loaded as 32-bit words (`ld1w`), which puts the same
//! bytes in the same register positions as `ld1b` and keeps every stream on
//! one `.s` predicate.

use super::super::macros::{arm_token, sve_stream};
use super::check_acc_shape;

arm_token! {
    /// Proof that SVE with `FEAT_I8MM` (`SMMLA`/`UMMLA`/`USMMLA`/`USDOT` on
    /// `z` registers) is available.
    token = SveI8mm, requires = [Sve, Sve2I8mm],
}

macro_rules! i8mm_op {
	($(($name:ident, $inner:ident, $Acc:ty, $A:ty, $B:ty, $asm:literal, $group:literal, $doc:literal)),+ $(,)?) => {$(
		sve_stream! {
			target_feature = "sve,i8mm", inner_fn = $inner,
			size = ["s", "ld1w", "st1w", ", lsl #2", "incw"],
			ins = [acc: $Acc => "0", a: u32 => "1", b: u32 => "2"], out = $Acc, result = "0",
			op = [$asm],
		}

		impl SveI8mm {
			#[doc = $doc]
			///
			/// # Panics
			/// `acc.len() != out.len()`, `a` or `b` not `4 * out.len()` long, or
			/// (for the `mmla` forms) `out.len()` not a multiple of 4.
			#[inline]
			pub fn $name(self, acc: &[$Acc], a: &[$A], b: &[$B], out: &mut [$Acc]) {
				check_acc_shape(acc.len(), out.len(), &[a.len(), b.len()], 4, $group);
				unsafe {
					$inner(acc.as_ptr(), a.as_ptr().cast(), b.as_ptr().cast(), out.as_mut_ptr(), out.len())
				}
			}
		}
	)+};
}

i8mm_op!(
    (
        mmla_s32_slice,
        sve_smmla,
        i32,
        i8,
        i8,
        "smmla z0.s, z1.b, z2.b",
        4,
        "Signed 2×8 · 8×2 matrix multiply-accumulate per 128-bit segment (`smmla z.s`); slice form of `I8mm::mmla_s32`."
    ),
    (
        mmla_u32_slice,
        sve_ummla,
        u32,
        u8,
        u8,
        "ummla z0.s, z1.b, z2.b",
        4,
        "Unsigned 2×8 · 8×2 matrix multiply-accumulate per segment (`ummla z.s`); slice form of `I8mm::mmla_u32`."
    ),
    (
        mmla_us32_slice,
        sve_usmmla,
        i32,
        u8,
        i8,
        "usmmla z0.s, z1.b, z2.b",
        4,
        "Mixed-sign matrix multiply-accumulate: unsigned `a`, signed `b` (`usmmla z.s`); slice form of `I8mm::mmla_us32`."
    ),
    (
        dot_us32_slice,
        sve_usdot,
        i32,
        u8,
        i8,
        "usdot z0.s, z1.b, z2.b",
        1,
        "Mixed-sign 4-way dot product per lane: `out[l] = acc[l] + Σₖ a[4l+k]·b[4l+k]`, unsigned `a`, signed `b` (`usdot z.s`); slice form of `I8mm::dot_us32`."
    ),
);

#[cfg(test)]
#[path = "../../test/ops/sve/sve_i8mm.rs"]
mod tests;
