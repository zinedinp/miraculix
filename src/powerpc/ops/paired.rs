//! Paired-single kernels for Gekko (GameCube), Broadway (Wii) and Espresso
//! (Wii U): every FPR holds two `f32` (`ps0`, `ps1`) and the quantized
//! load/store unit converts `u8`/`s8`/`u16`/`s16` to and from `f32` for free.
//!
//! - **One `asm!` block per kernel.** LLVM treats FPRs as plain `f64` and
//!   would spill them with `stfd`, which drops `ps1`, so no value may live in
//!   an FPR between blocks. Load, compute and store all happen in one block.
//! - **GQR7.** The quantization registers are global supervisor state. Each
//!   kernel saves GQR7, sets the conversion it needs, and restores it; GQR0-6
//!   are never touched.
//! - **Registers.** 32 FPRs x 2 lanes: the reductions keep
//!   [`ACCUMULATORS`] pair sums and load [`UNROLL`] pairs per trip.
//! - **Encodings.** No assembler accepts these instructions, so every one is a
//!   `.long` from `encode` (checked against binutils on the host). GPRs and
//!   FPRs are fixed (`r5`-`r8`, `f0`-`f31`) because the numbers are baked into
//!   the words.
//! - **Alignment.** `out` is aligned to 8 bytes by a scalar head; inputs only
//!   need their natural element alignment. A pair that straddles a 32-byte
//!   line is slower, but the 750 handles it in hardware for word-aligned
//!   addresses.
//!
//! Only reported on bare metal (`target_os = "none"`); see
//! [`crate::powerpc::features`].

use super::encode::{
    gqr, ps_add, ps_madd, ps_merge00, ps_mr, ps_mul, ps_sub, ps_sum0, psq_l, psq_st,
};
use crate::powerpc::{Feature, FeatureSet};

/// Pairs loaded per loop trip.
pub const UNROLL: usize = 8;
/// Independent pair accumulators in the reductions.
pub const ACCUMULATORS: usize = 8;

/// GQR used by every kernel (saved and restored).
const GQR: u32 = 7;

/// Proof of paired singles with `HID2[PSE]` and `HID2[LSQE]` set.
#[derive(Debug, Clone, Copy)]
pub struct PairedSingle(());

/// Scalar elements before `p` is 8-byte aligned (0 or 1 for `f32`).
#[inline]
fn head8(p: *const f32, n: usize) -> usize {
    ((p as usize & 7 != 0) as usize).min(n)
}

/// `out = op(a, b)` over whole groups of four pairs (32 bytes).
macro_rules! pair_map2 {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// `a`, `b` readable and `o` writable for `trips * 8` floats.
        unsafe fn $name(a: *const f32, b: *const f32, o: *mut f32, trips: usize) {
            // SAFETY: caller's contract; GQR7 is restored before the block ends.
            unsafe {
                asm_mixed!(
                    s("mfspr {save}, 919"),
                    s("li {t}, 0"),
                    s("mtspr 919, {t}"),
                    s("isync"),
                    s("mtctr {n}"),
                    s("2:"),
                    w(psq_l(1, 5, 0, false, GQR)), w(psq_l(2, 5, 8, false, GQR)),
                    w(psq_l(3, 5, 16, false, GQR)), w(psq_l(4, 5, 24, false, GQR)),
                    w(psq_l(5, 6, 0, false, GQR)), w(psq_l(6, 6, 8, false, GQR)),
                    w(psq_l(7, 6, 16, false, GQR)), w(psq_l(8, 6, 24, false, GQR)),
                    w($op(1, 1, 5)), w($op(2, 2, 6)), w($op(3, 3, 7)), w($op(4, 4, 8)),
                    w(psq_st(1, 7, 0, false, GQR)), w(psq_st(2, 7, 8, false, GQR)),
                    w(psq_st(3, 7, 16, false, GQR)), w(psq_st(4, 7, 24, false, GQR)),
                    s("addi 5, 5, 32"),
                    s("addi 6, 6, 32"),
                    s("addi 7, 7, 32"),
                    s("bdnz 2b"),
                    s("mtspr 919, {save}"),
                    s("isync"),
                    ;
                    save = out(reg) _, t = out(reg) _, n = in(reg) trips,
                    inout("r5") a => _, inout("r6") b => _, inout("r7") o => _,
                    out("f1") _, out("f2") _, out("f3") _, out("f4") _,
                    out("f5") _, out("f6") _, out("f7") _, out("f8") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
    };
}

pair_map2!(add_body, ps_add);
pair_map2!(sub_body, ps_sub);
pair_map2!(mul_body, ps_mul);

impl PairedSingle {
    /// Token if the cached feature set has paired singles.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has paired singles.
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains(Feature::PairedSingle)
            .then_some(PairedSingle(()))
    }

    fn map2(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        scalar: impl Fn(f32, f32) -> f32,
        body: unsafe fn(*const f32, *const f32, *mut f32, usize),
    ) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let n = out.len();
        let h = head8(out.as_ptr(), n);
        let trips = (n - h) / 8;
        for i in 0..h {
            out[i] = scalar(a[i], b[i]);
        }
        if trips > 0 {
            // SAFETY: the token proves paired singles; `trips * 8` floats of
            // each slice from `h` are in bounds.
            unsafe {
                body(
                    a[h..].as_ptr(),
                    b[h..].as_ptr(),
                    out[h..].as_mut_ptr(),
                    trips,
                )
            };
        }
        for i in h + trips * 8..n {
            out[i] = scalar(a[i], b[i]);
        }
    }

    /// `out[i] = a[i] + b[i]`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn add_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x + y, add_body);
    }

    /// `out[i] = a[i] - b[i]`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn sub_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x - y, sub_body);
    }

    /// `out[i] = a[i] * b[i]`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn mul_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x * y, mul_body);
    }

    /// `out[i] = a[i] * b[i] + c[i]` (`ps_madd` in the paired part).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
        let n = out.len();
        let h = head8(out.as_ptr(), n);
        let trips = (n - h) / 8;
        for i in 0..h {
            out[i] = a[i] * b[i] + c[i];
        }
        if trips > 0 {
            // SAFETY: token proves paired singles; `trips * 8` floats of each
            // slice from `h` are in bounds; GQR7 is restored.
            unsafe {
                asm_mixed!(
                    s("mfspr {save}, 919"),
                    s("li {t}, 0"),
                    s("mtspr 919, {t}"),
                    s("isync"),
                    s("mtctr {n}"),
                    s("2:"),
                    w(psq_l(1, 5, 0, false, GQR)), w(psq_l(2, 5, 8, false, GQR)),
                    w(psq_l(3, 5, 16, false, GQR)), w(psq_l(4, 5, 24, false, GQR)),
                    w(psq_l(5, 6, 0, false, GQR)), w(psq_l(6, 6, 8, false, GQR)),
                    w(psq_l(7, 6, 16, false, GQR)), w(psq_l(8, 6, 24, false, GQR)),
                    w(psq_l(9, 7, 0, false, GQR)), w(psq_l(10, 7, 8, false, GQR)),
                    w(psq_l(11, 7, 16, false, GQR)), w(psq_l(12, 7, 24, false, GQR)),
                    // a * b + c
                    w(ps_madd(1, 1, 5, 9)), w(ps_madd(2, 2, 6, 10)),
                    w(ps_madd(3, 3, 7, 11)), w(ps_madd(4, 4, 8, 12)),
                    w(psq_st(1, 8, 0, false, GQR)), w(psq_st(2, 8, 8, false, GQR)),
                    w(psq_st(3, 8, 16, false, GQR)), w(psq_st(4, 8, 24, false, GQR)),
                    s("addi 5, 5, 32"),
                    s("addi 6, 6, 32"),
                    s("addi 7, 7, 32"),
                    s("addi 8, 8, 32"),
                    s("bdnz 2b"),
                    s("mtspr 919, {save}"),
                    s("isync"),
                    ;
                    save = out(reg) _, t = out(reg) _, n = in(reg) trips,
                    inout("r5") a[h..].as_ptr() => _, inout("r6") b[h..].as_ptr() => _,
                    inout("r7") c[h..].as_ptr() => _, inout("r8") out[h..].as_mut_ptr() => _,
                    out("f1") _, out("f2") _, out("f3") _, out("f4") _, out("f5") _,
                    out("f6") _, out("f7") _, out("f8") _, out("f9") _, out("f10") _,
                    out("f11") _, out("f12") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        for i in h + trips * 8..n {
            out[i] = a[i] * b[i] + c[i];
        }
    }

    /// Sum of all elements: [`ACCUMULATORS`] pair sums over 16 floats per
    /// trip, then a scalar tail. NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 16;
        if trips == 0 {
            return crate::reduce_common::sum4(a);
        }
        let zero = 0f32;
        let mut sum = 0f32;
        // SAFETY: token proves paired singles; `trips * 16` floats of `a` are
        // in bounds; `stfs` writes `sum`; GQR7 is restored.
        unsafe {
            asm_mixed!(
                s("mfspr {save}, 919"),
                s("li {t}, 0"),
                s("mtspr 919, {t}"),
                s("isync"),
                s("lfs 0, 0({z})"),
                w(ps_merge00(0, 0, 0)),
                w(ps_mr(1, 0)), w(ps_mr(2, 0)), w(ps_mr(3, 0)), w(ps_mr(4, 0)),
                w(ps_mr(5, 0)), w(ps_mr(6, 0)), w(ps_mr(7, 0)), w(ps_mr(8, 0)),
                s("mtctr {n}"),
                s("2:"),
                w(psq_l(9, 5, 0, false, GQR)), w(psq_l(10, 5, 8, false, GQR)),
                w(psq_l(11, 5, 16, false, GQR)), w(psq_l(12, 5, 24, false, GQR)),
                w(psq_l(13, 5, 32, false, GQR)), w(psq_l(14, 5, 40, false, GQR)),
                w(psq_l(15, 5, 48, false, GQR)), w(psq_l(16, 5, 56, false, GQR)),
                w(ps_add(1, 1, 9)), w(ps_add(2, 2, 10)), w(ps_add(3, 3, 11)), w(ps_add(4, 4, 12)),
                w(ps_add(5, 5, 13)), w(ps_add(6, 6, 14)), w(ps_add(7, 7, 15)), w(ps_add(8, 8, 16)),
                s("addi 5, 5, 64"),
                s("bdnz 2b"),
                w(ps_add(1, 1, 2)), w(ps_add(3, 3, 4)), w(ps_add(5, 5, 6)), w(ps_add(7, 7, 8)),
                w(ps_add(1, 1, 3)), w(ps_add(5, 5, 7)), w(ps_add(1, 1, 5)),
                // ps0 = ps0 + ps1
                w(ps_sum0(1, 1, 1, 1)),
                s("stfs 1, 0({s})"),
                s("mtspr 919, {save}"),
                s("isync"),
                ;
                save = out(reg) _, t = out(reg) _, n = in(reg) trips,
                z = in(reg_nonzero) &zero as *const f32,
                s = in(reg_nonzero) &mut sum as *mut f32,
                inout("r5") a.as_ptr() => _,
                out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _,
                out("f5") _, out("f6") _, out("f7") _, out("f8") _, out("f9") _,
                out("f10") _, out("f11") _, out("f12") _, out("f13") _, out("f14") _,
                out("f15") _, out("f16") _, out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::sum4(&a[trips * 16..])
    }

    /// Dot product: [`ACCUMULATORS`] `ps_madd` pair sums over 16 floats per
    /// trip, then a scalar tail. NaN and inf propagate.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let trips = a.len() / 16;
        if trips == 0 {
            return crate::reduce_common::dot4(a, b, 0);
        }
        let zero = 0f32;
        let mut sum = 0f32;
        // SAFETY: token proves paired singles; `trips * 16` floats of `a` and
        // `b` are in bounds; `stfs` writes `sum`; GQR7 is restored.
        unsafe {
            asm_mixed!(
                s("mfspr {save}, 919"),
                s("li {t}, 0"),
                s("mtspr 919, {t}"),
                s("isync"),
                s("lfs 0, 0({z})"),
                w(ps_merge00(0, 0, 0)),
                w(ps_mr(1, 0)), w(ps_mr(2, 0)), w(ps_mr(3, 0)), w(ps_mr(4, 0)),
                w(ps_mr(5, 0)), w(ps_mr(6, 0)), w(ps_mr(7, 0)), w(ps_mr(8, 0)),
                s("mtctr {n}"),
                s("2:"),
                w(psq_l(9, 5, 0, false, GQR)), w(psq_l(10, 5, 8, false, GQR)),
                w(psq_l(11, 5, 16, false, GQR)), w(psq_l(12, 5, 24, false, GQR)),
                w(psq_l(13, 5, 32, false, GQR)), w(psq_l(14, 5, 40, false, GQR)),
                w(psq_l(15, 5, 48, false, GQR)), w(psq_l(16, 5, 56, false, GQR)),
                w(psq_l(17, 6, 0, false, GQR)), w(psq_l(18, 6, 8, false, GQR)),
                w(psq_l(19, 6, 16, false, GQR)), w(psq_l(20, 6, 24, false, GQR)),
                w(psq_l(21, 6, 32, false, GQR)), w(psq_l(22, 6, 40, false, GQR)),
                w(psq_l(23, 6, 48, false, GQR)), w(psq_l(24, 6, 56, false, GQR)),
                // acc = a * b + acc
                w(ps_madd(1, 9, 17, 1)), w(ps_madd(2, 10, 18, 2)),
                w(ps_madd(3, 11, 19, 3)), w(ps_madd(4, 12, 20, 4)),
                w(ps_madd(5, 13, 21, 5)), w(ps_madd(6, 14, 22, 6)),
                w(ps_madd(7, 15, 23, 7)), w(ps_madd(8, 16, 24, 8)),
                s("addi 5, 5, 64"),
                s("addi 6, 6, 64"),
                s("bdnz 2b"),
                w(ps_add(1, 1, 2)), w(ps_add(3, 3, 4)), w(ps_add(5, 5, 6)), w(ps_add(7, 7, 8)),
                w(ps_add(1, 1, 3)), w(ps_add(5, 5, 7)), w(ps_add(1, 1, 5)),
                w(ps_sum0(1, 1, 1, 1)),
                s("stfs 1, 0({s})"),
                s("mtspr 919, {save}"),
                s("isync"),
                ;
                save = out(reg) _, t = out(reg) _, n = in(reg) trips,
                z = in(reg_nonzero) &zero as *const f32,
                s = in(reg_nonzero) &mut sum as *mut f32,
                inout("r5") a.as_ptr() => _, inout("r6") b.as_ptr() => _,
                out("f0") _, out("f1") _, out("f2") _, out("f3") _, out("f4") _,
                out("f5") _, out("f6") _, out("f7") _, out("f8") _, out("f9") _,
                out("f10") _, out("f11") _, out("f12") _, out("f13") _, out("f14") _,
                out("f15") _, out("f16") _, out("f17") _, out("f18") _, out("f19") _,
                out("f20") _, out("f21") _, out("f22") _, out("f23") _, out("f24") _,
                out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::dot4(a, b, trips * 16)
    }

    /// `out[i] = a * q[i] + b`, two roundings like the portable code. The
    /// quantized load turns each `u16` pair into two `f32` in hardware
    /// (GQR7 load type `u16`, scale 0).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn affine_u16_f32(self, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
        assert_eq!(q.len(), out.len());
        let n = out.len();
        let h = head8(out.as_ptr(), n);
        crate::reduce_common::affine_u16_f32_scalar(&q[..h], a, b, &mut out[..h], 0);
        let trips = (n - h) / 8;
        if trips > 0 {
            // `lis` loads the upper half: GQR value `U16 << 16`, f32 stores.
            const GQR_HI: u32 = gqr::value(gqr::U16, gqr::F32) >> 16;
            const _: () = assert!(gqr::value(gqr::U16, gqr::F32) & 0xffff == 0);
            // SAFETY: token proves paired singles; `trips * 8` elements of `q`
            // and `out` from `h` are in bounds; GQR7 is restored.
            unsafe {
                asm_mixed!(
                    s("mfspr {save}, 919"),
                    s("lis {t}, {hi}"),
                    s("mtspr 919, {t}"),
                    s("isync"),
                    s("lfs 30, 0({ka})"),
                    s("lfs 31, 0({kb})"),
                    w(ps_merge00(30, 30, 30)),
                    w(ps_merge00(31, 31, 31)),
                    s("mtctr {n}"),
                    s("2:"),
                    w(psq_l(1, 5, 0, false, GQR)), w(psq_l(2, 5, 4, false, GQR)),
                    w(psq_l(3, 5, 8, false, GQR)), w(psq_l(4, 5, 12, false, GQR)),
                    w(ps_mul(1, 1, 30)), w(ps_mul(2, 2, 30)), w(ps_mul(3, 3, 30)), w(ps_mul(4, 4, 30)),
                    w(ps_add(1, 1, 31)), w(ps_add(2, 2, 31)), w(ps_add(3, 3, 31)), w(ps_add(4, 4, 31)),
                    w(psq_st(1, 7, 0, false, GQR)), w(psq_st(2, 7, 8, false, GQR)),
                    w(psq_st(3, 7, 16, false, GQR)), w(psq_st(4, 7, 24, false, GQR)),
                    s("addi 5, 5, 16"),
                    s("addi 7, 7, 32"),
                    s("bdnz 2b"),
                    s("mtspr 919, {save}"),
                    s("isync"),
                    ;
                    save = out(reg) _, t = out(reg) _, n = in(reg) trips, hi = const GQR_HI,
                    ka = in(reg_nonzero) &a as *const f32, kb = in(reg_nonzero) &b as *const f32,
                    inout("r5") q[h..].as_ptr() => _, inout("r7") out[h..].as_mut_ptr() => _,
                    out("f1") _, out("f2") _, out("f3") _, out("f4") _,
                    out("f30") _, out("f31") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        crate::reduce_common::affine_u16_f32_scalar(q, a, b, out, h + trips * 8);
    }
}
