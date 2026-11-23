//! `Zvfbfmin` / `Zvfbfwma`: vector BF16. Tokens: [`Zvfbfmin`], [`Zvfbfwma`].
//!
//! BF16 bit patterns (`u16`). `Zvfbfmin` converts BF16 <-> `f32`
//! (`vfwcvtbf16.f.f.v`, `vfncvtbf16.f.f.w`; the narrowing one uses the
//! dynamic rounding mode, nearest even by default, overflow gives infinity).
//! `Zvfbfwma` multiply-accumulates BF16 pairs into `f32` (`vfwmaccbf16.vv`).
//! All loops are vector-length-agnostic.

use core::arch::asm;

use super::super::{Feature, FeatureSet};

/// Proof that `V` and `Zvfbfmin` are available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Zvfbfmin(());

/// Proof that `V`, `Zvfbfmin` and `Zvfbfwma` are available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Zvfbfwma(());

impl Zvfbfmin {
    /// `None` if the CPU lacks `Zvfbfmin`.
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::riscv::detect_features())
    }

    /// From a raw feature bitset.
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        (set.contains(Feature::V) && set.contains(Feature::Zvfbfmin)).then_some(Zvfbfmin(()))
    }

    /// BF16 bits to `f32`, whole slices.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn cvt_bf16_to_f32_slice(self, a: &[u16], out: &mut [f32]) {
        assert_eq!(out.len(), a.len());
        let (mut s, mut d, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: token proves V+Zvfbfmin; reads `n` u16, writes `n` f32.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v, +zvfbfmin",
                "2:",
                "vsetvli {vl}, {n}, e16, mf2, ta, ma",
                "vle16.v v8, ({s})",
                "vfwcvtbf16.f.f.v v9, v8",
                "vsetvli zero, {vl}, e32, m1, ta, ma",
                "vse32.v v9, ({d})",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 1",
                "add {s}, {s}, {t}",
                "slli {t}, {vl}, 2",
                "add {d}, {d}, {t}",
                "bnez {n}, 2b",
                ".option pop",
                s = inout(reg) s, d = inout(reg) d, n = inout(reg) n,
                vl = out(reg) _, t = out(reg) _,
                out("v8") _, out("v9") _,
                options(nostack),
            );
        }
        let _ = (s, d, n);
    }

    /// `f32` to BF16 bits, whole slices.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn cvt_f32_to_bf16_slice(self, a: &[f32], out: &mut [u16]) {
        assert_eq!(out.len(), a.len());
        let (mut s, mut d, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: as above, reversed.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v, +zvfbfmin",
                "2:",
                "vsetvli {vl}, {n}, e32, m1, ta, ma",
                "vle32.v v8, ({s})",
                "vsetvli zero, {vl}, e16, mf2, ta, ma",
                "vfncvtbf16.f.f.w v9, v8",
                "vse16.v v9, ({d})",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {s}, {s}, {t}",
                "slli {t}, {vl}, 1",
                "add {d}, {d}, {t}",
                "bnez {n}, 2b",
                ".option pop",
                s = inout(reg) s, d = inout(reg) d, n = inout(reg) n,
                vl = out(reg) _, t = out(reg) _,
                out("v8") _, out("v9") _,
                options(nostack),
            );
        }
        let _ = (s, d, n);
    }
}

impl Zvfbfwma {
    /// `None` if the CPU lacks `Zvfbfwma`.
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::riscv::detect_features())
    }

    /// From a raw feature bitset.
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        (set.contains(Feature::V)
            && set.contains(Feature::Zvfbfmin)
            && set.contains(Feature::Zvfbfwma))
        .then_some(Zvfbfwma(()))
    }

    /// `out[i] = acc[i] + a[i] * b[i]` with BF16 `a`, `b` and `f32` `acc`/`out`
    /// (`vfwmaccbf16.vv`). The product of two BF16 values is exact in `f32`.
    ///
    /// # Panics
    /// If the four slices differ in length.
    pub fn fmacc_bf16_f32_slice(self, acc: &[f32], a: &[u16], b: &[u16], out: &mut [f32]) {
        assert!(acc.len() == a.len() && a.len() == b.len() && b.len() == out.len());
        let (mut pa, mut pb, mut pc, mut po, mut n) = (
            a.as_ptr(),
            b.as_ptr(),
            acc.as_ptr(),
            out.as_mut_ptr(),
            a.len(),
        );
        if n == 0 {
            return;
        }
        // SAFETY: token proves the extension; each slice has `n` elements.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v, +zvfbfmin, +zvfbfwma",
                "2:",
                "vsetvli {vl}, {n}, e16, mf2, ta, ma",
                "vle16.v v8, ({pa})",
                "vle16.v v9, ({pb})",
                "vle32.v v10, ({pc})",
                "vfwmaccbf16.vv v10, v8, v9",
                "vse32.v v10, ({po})",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 1",
                "add {pa}, {pa}, {t}",
                "add {pb}, {pb}, {t}",
                "slli {t}, {vl}, 2",
                "add {pc}, {pc}, {t}",
                "add {po}, {po}, {t}",
                "bnez {n}, 2b",
                ".option pop",
                pa = inout(reg) pa, pb = inout(reg) pb, pc = inout(reg) pc,
                po = inout(reg) po, n = inout(reg) n,
                vl = out(reg) _, t = out(reg) _,
                out("v8") _, out("v9") _, out("v10") _,
                options(nostack),
            );
        }
        let _ = (pa, pb, pc, po, n);
    }
}
