//! `Zvfh`: vector half-precision FP. Token: [`Zvfh`].
//!
//! FP16 bit patterns (`u16`) <-> `f32` with `vfwcvt.f.f.v` / `vfncvt.f.f.w`.
//! The narrowing convert uses the dynamic rounding mode, round to nearest
//! even unless you changed `frm`. Overflow gives infinity.

use core::arch::asm;

use super::super::{Feature, FeatureSet};

/// Proof that `V` and `Zvfh` are available (hwprobe on Linux). Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Zvfh(());

impl Zvfh {
    /// `None` if the CPU lacks `Zvfh`.
    pub fn detect() -> Option<Self> {
        Self::from_features(FeatureSet::detect())
    }

    /// From a raw feature bitset.
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        (set.contains(Feature::V) && set.contains(Feature::Zvfh)).then_some(Zvfh(()))
    }

    /// FP16 bits to `f32`, whole slices (strip-mined, any vector length).
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn cvt_f16_to_f32_slice(self, a: &[u16], out: &mut [f32]) {
        assert_eq!(out.len(), a.len());
        let (mut s, mut d, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: token proves V+Zvfh; the loop reads `n` u16 and writes `n` f32.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v, +zvfh",
                "2:",
                "vsetvli {vl}, {n}, e16, mf2, ta, ma",
                "vle16.v v8, ({s})",
                "vfwcvt.f.f.v v9, v8",
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

    /// `f32` to FP16 bits, whole slices (strip-mined, any vector length).
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn cvt_f32_to_f16_slice(self, a: &[f32], out: &mut [u16]) {
        assert_eq!(out.len(), a.len());
        let (mut s, mut d, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: as above, reversed.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v, +zvfh",
                "2:",
                "vsetvli {vl}, {n}, e32, m1, ta, ma",
                "vle32.v v8, ({s})",
                "vsetvli zero, {vl}, e16, mf2, ta, ma",
                "vfncvt.f.f.w v9, v8",
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
