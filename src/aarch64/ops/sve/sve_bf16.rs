//! SVE `FEAT_BF16` (v8.6): bfloat16 dot product, matrix multiply, widening
//! multiply-add and `f32` → `bf16` narrowing on `z` registers. Token:
//! [`SveBf16`].
//!
//! Slice forms of the Neon [`Bf16`](super::super::neon::bf16::Bf16) token's
//! ops, with the same two rounding regimes (see that module): `bfmlalb` /
//! `bfmlalt` / `cvt` are IEEE, `bfdot` / `bfmmla` round intermediates to odd
//! and flush denormals unless `FEAT_EBF16` is present.
//!
//! `bf16` data crosses the API as `u16` bit patterns. Each 32-bit lane takes
//! one **pair** of `bf16`s, loaded as a word, so `a` and `b` are twice as long
//! as `acc`/`out`.

use super::super::macros::{arm_token, sve_stream};
use super::check_acc_shape;

arm_token! {
    /// Proof that SVE with `FEAT_BF16` (`BFDOT`/`BFMMLA`/`BFMLALB`/`BFMLALT`/
    /// `BFCVT` on `z` registers) is available.
    token = SveBf16, requires = [Sve, Sve2Bf16],
}

macro_rules! bf16_op {
	($(($name:ident, $inner:ident, $asm:literal, $group:literal, $doc:literal)),+ $(,)?) => {$(
		sve_stream! {
			target_feature = "sve,bf16", inner_fn = $inner,
			size = ["s", "ld1w", "st1w", ", lsl #2", "incw"],
			ins = [acc: f32 => "0", a: u32 => "1", b: u32 => "2"], out = f32, result = "0",
			op = [$asm],
		}

		impl SveBf16 {
			#[doc = $doc]
			///
			/// # Panics
			/// `acc.len() != out.len()`, `a` or `b` not `2 * out.len()` long,
			/// or (for `bfmmla`) `out.len()` not a multiple of 4.
			#[inline]
			pub fn $name(self, acc: &[f32], a: &[u16], b: &[u16], out: &mut [f32]) {
				check_acc_shape(acc.len(), out.len(), &[a.len(), b.len()], 2, $group);
				unsafe {
					$inner(acc.as_ptr(), a.as_ptr().cast(), b.as_ptr().cast(), out.as_mut_ptr(), out.len())
				}
			}
		}
	)+};
}

bf16_op!(
    (
        bfdot_f32_slice,
        sve_bfdot,
        "bfdot z0.s, z1.h, z2.h",
        1,
        "2-way `bf16` dot product accumulate: `out[i] = acc[i] + a[2i]·b[2i] + a[2i+1]·b[2i+1]` (`bfdot z.s`). Non-IEEE intermediate rounding."
    ),
    (
        bfmmla_f32_slice,
        sve_bfmmla,
        "bfmmla z0.s, z1.h, z2.h",
        4,
        "`bf16` 2×4 · 4×2 matrix multiply-accumulate per 128-bit segment, row-major: `out[2i+j] = acc[2i+j] + Σₖ a[4i+k]·b[4j+k]` within the segment (`bfmmla z.s`). Non-IEEE intermediate rounding."
    ),
    (
        bfmlalb_f32_slice,
        sve_bfmlalb,
        "bfmlalb z0.s, z1.h, z2.h",
        1,
        "Widening fused multiply-add on the **even** `bf16` lanes: `out[i] = acc[i] + a[2i]·b[2i]`, one IEEE rounding (`bfmlalb z.s`)."
    ),
    (
        bfmlalt_f32_slice,
        sve_bfmlalt,
        "bfmlalt z0.s, z1.h, z2.h",
        1,
        "Widening fused multiply-add on the **odd** `bf16` lanes: `out[i] = acc[i] + a[2i+1]·b[2i+1]`, one IEEE rounding (`bfmlalt z.s`)."
    ),
);

impl SveBf16 {
    /// Narrow `f32` to `bf16` bit patterns, round-to-nearest-even
    /// (`bfcvt z.h, z.s`, then a truncating `st1h` of each 32-bit lane).
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    #[inline]
    pub fn cvt_f32_to_bf16_slice(self, a: &[f32], out: &mut [u16]) {
        assert_eq!(a.len(), out.len());
        unsafe { sve_bfcvt(a.as_ptr(), out.as_mut_ptr(), out.len()) }
    }
}

/// # Safety
/// Caller proved the feature via the token; `a` and `out` are valid for `n`
/// elements.
#[inline]
#[target_feature(enable = "sve,bf16")]
unsafe fn sve_bfcvt(a: *const f32, out: *mut u16, n: usize) {
    unsafe {
        // Not `sve_stream!`: the load is 32-bit and the store 16-bit per lane.
        // `BFCVT` leaves the `bf16` in the bottom half of each `.s` container,
        // and `st1h { z.s }` stores exactly that half.
        core::arch::asm!(
            "mov {i}, #0",
            "whilelo p0.s, {i}, {n}",
            "b.none 3f",
            "2:",
            "ld1w {{ z0.s }}, p0/z, [{a}, {i}, lsl #2]",
            "bfcvt z0.h, p0/m, z0.s",
            "st1h {{ z0.s }}, p0, [{o}, {i}, lsl #1]",
            "incw {i}",
            "whilelo p0.s, {i}, {n}",
            "b.first 2b",
            "3:",
            i = out(reg) _, n = in(reg) n, a = in(reg) a, o = in(reg) out,
            out("v0") _, out("p0") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
#[path = "../../test/ops/sve/sve_bf16.rs"]
mod tests;
