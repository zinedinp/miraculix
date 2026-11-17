//! Base RVV 1.0 slice ops. Token: [`Rvv`].
//!
//! `core::arch::riscv64` has no stable vector intrinsics, so every body is an
//! `asm!` loop strip-mined with `vsetvli` (LMUL 4), correct for any VLEN.
//! All vector registers are declared clobbered.
//!
//! These cover what a kernel needs beyond conversions: elementwise
//! add/mul/FMA, reductions, compare + select-by-index, and a widening
//! `i8` dot product. Names and NaN rules match `x86::reduce` where both exist.

use core::arch::asm;

use super::super::{Feature, FeatureSet};

/// Proof that the `V` extension is available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Rvv(());

macro_rules! rvv_asm {
    ($($tt:tt)*) => {
        asm!($($tt)*,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            options(nostack))
    };
}

impl Rvv {
    /// `None` if the CPU lacks `V`.
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::riscv::detect_features())
    }

    /// From a raw feature bitset.
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains(Feature::V).then_some(Rvv(()))
    }
}

macro_rules! binop {
    ($(#[$m:meta])* $name:ident, $t:ty, $ew:literal, $shift:literal, $vop:literal) => {
        $(#[$m])*
        ///
        /// # Panics
        /// If the three slices differ in length.
        pub fn $name(self, a: &[$t], b: &[$t], out: &mut [$t]) {
            assert!(a.len() == b.len() && b.len() == out.len());
            let (mut pa, mut pb, mut po, mut n) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
            if n == 0 {
                return;
            }
            // SAFETY: token proves V; each slice has `n` elements.
            unsafe {
                rvv_asm!(
                    ".option push",
                    ".option arch, +v",
                    "2:",
                    concat!("vsetvli {vl}, {n}, ", $ew, ", m4, ta, ma"),
                    concat!("vle", $shift, ".v v8, ({a})"),
                    concat!("vle", $shift, ".v v12, ({b})"),
                    concat!($vop, " v8, v8, v12"),
                    concat!("vse", $shift, ".v v8, ({o})"),
                    "sub {n}, {n}, {vl}",
                    "slli {t}, {vl}, 2",
                    "add {a}, {a}, {t}",
                    "add {b}, {b}, {t}",
                    "add {o}, {o}, {t}",
                    "bnez {n}, 2b",
                    ".option pop",
                    a = inout(reg) pa, b = inout(reg) pb, o = inout(reg) po, n = inout(reg) n,
                    vl = out(reg) _, t = out(reg) _
                );
            }
            let _ = (pa, pb, po, n);
        }
    };
}

impl Rvv {
    binop!(/// `out[i] = a[i] + b[i]` (`vfadd.vv`).
        add_f32, f32, "e32", "32", "vfadd.vv");
    binop!(/// `out[i] = a[i] - b[i]` (`vfsub.vv`).
        sub_f32, f32, "e32", "32", "vfsub.vv");
    binop!(/// `out[i] = a[i] * b[i]` (`vfmul.vv`).
        mul_f32, f32, "e32", "32", "vfmul.vv");
    binop!(/// `out[i] = a[i] + b[i]`, wrapping (`vadd.vv`).
        add_i32, i32, "e32", "32", "vadd.vv");

    /// `out[i] = a[i] * b[i] + c[i]`, fused (`vfmacc.vv`).
    ///
    /// # Panics
    /// If the slices differ in length.
    pub fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
        let (mut pa, mut pb, mut pc, mut po, mut n) =
            (a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: token proves V; each slice has `n` elements.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({a})",
                "vle32.v v12, ({b})",
                "vle32.v v16, ({c})",
                "vfmacc.vv v16, v8, v12",
                "vse32.v v16, ({o})",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {a}, {a}, {t}",
                "add {b}, {b}, {t}",
                "add {c}, {c}, {t}",
                "add {o}, {o}, {t}",
                "bnez {n}, 2b",
                ".option pop",
                a = inout(reg) pa, b = inout(reg) pb, c = inout(reg) pc, o = inout(reg) po,
                n = inout(reg) n, vl = out(reg) _, t = out(reg) _
            );
        }
        let _ = (pa, pb, pc, po, n);
    }

    /// Sum of all elements (`vfredusum`, unordered: tier-dependent order). NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let (mut p, mut n) = (a.as_ptr(), a.len());
        if n == 0 {
            return 0.0;
        }
        let r: f32;
        // SAFETY: token proves V; reads `n` f32.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {t}, zero, e32, m1, ta, ma",
                "vmv.v.i v24, 0",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({p})",
                "vfredusum.vs v24, v8, v24",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {p}, {p}, {t}",
                "bnez {n}, 2b",
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vfmv.f.s {r}, v24",
                ".option pop",
                p = inout(reg) p, n = inout(reg) n, vl = out(reg) _, t = out(reg) _, r = out(freg) r
            );
        }
        let _ = (p, n);
        r
    }

    /// Largest element, NaN ignored (`vfredmax`); `-inf` if there is none.
    pub fn reduce_max_f32(self, a: &[f32]) -> f32 {
        let (mut p, mut n) = (a.as_ptr(), a.len());
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let r: f32;
        // SAFETY: as above. `vfredmax` follows IEEE maximumNumber: a NaN lane
        // loses to a number; it returns NaN only if the accumulator is NaN.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {t}, zero, e32, m1, ta, ma",
                "vfmv.v.f v24, {init}",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({p})",
                "vfredmax.vs v24, v8, v24",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {p}, {p}, {t}",
                "bnez {n}, 2b",
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vfmv.f.s {r}, v24",
                ".option pop",
                p = inout(reg) p, n = inout(reg) n, vl = out(reg) _, t = out(reg) _,
                init = in(freg) f32::NEG_INFINITY, r = out(freg) r
            );
        }
        let _ = (p, n);
        if r.is_nan() { f32::NEG_INFINITY } else { r }
    }

    /// Smallest element, NaN ignored (`vfredmin`); `+inf` if there is none.
    pub fn reduce_min_f32(self, a: &[f32]) -> f32 {
        let (mut p, mut n) = (a.as_ptr(), a.len());
        if n == 0 {
            return f32::INFINITY;
        }
        let r: f32;
        // SAFETY: as above.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {t}, zero, e32, m1, ta, ma",
                "vfmv.v.f v24, {init}",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({p})",
                "vfredmin.vs v24, v8, v24",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {p}, {p}, {t}",
                "bnez {n}, 2b",
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vfmv.f.s {r}, v24",
                ".option pop",
                p = inout(reg) p, n = inout(reg) n, vl = out(reg) _, t = out(reg) _,
                init = in(freg) f32::INFINITY, r = out(freg) r
            );
        }
        let _ = (p, n);
        if r.is_nan() { f32::INFINITY } else { r }
    }

    /// First index of the largest element and its value, NaN ignored.
    /// `None` if the slice is empty or all NaN.
    pub fn argmax_f32(self, a: &[f32]) -> Option<(usize, f32)> {
        let m = self.reduce_max_f32(a);
        let (mut p, mut n) = (a.as_ptr(), a.len());
        if n == 0 {
            return None;
        }
        let found: isize;
        // SAFETY: token proves V; reads at most `n` f32. `base` counts consumed elements.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "li {base}, 0",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({p})",
                "vmfeq.vf v0, v8, {m}",
                "vfirst.m {t}, v0",
                "bgez {t}, 3f",
                "sub {n}, {n}, {vl}",
                "add {base}, {base}, {vl}",
                "slli {t}, {vl}, 2",
                "add {p}, {p}, {t}",
                "bnez {n}, 2b",
                "li {t}, -1",
                "3:",
                "mv {r}, {t}",
                "bltz {t}, 4f",
                "add {r}, {t}, {base}",
                "4:",
                ".option pop",
                p = inout(reg) p, n = inout(reg) n, vl = out(reg) _, t = out(reg) _,
                base = out(reg) _, m = in(freg) m, r = out(reg) found
            );
        }
        let _ = (p, n);
        (found >= 0).then_some((found as usize, m))
    }

    /// Writes the indices of all `a[i] > threshold` to `out`, ascending, and
    /// returns how many (`vmfgt` + `vcompress`). NaN never matches.
    ///
    /// # Panics
    /// If `out.len() < a.len()`.
    pub fn compress_gt_f32(self, a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
        assert!(out.len() >= a.len(), "out must hold a.len() indices");
        let (mut p, mut o, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return 0;
        }
        let total: usize;
        // SAFETY: token proves V; writes at most `a.len()` u32 into `out`.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "li {base}, 0",
                "li {total}, 0",
                "2:",
                "vsetvli {vl}, {n}, e32, m4, ta, ma",
                "vle32.v v8, ({p})",
                "vmfgt.vf v0, v8, {thr}",
                "vid.v v12",
                "vadd.vx v12, v12, {base}",
                "vcompress.vm v16, v12, v0",
                "vcpop.m {c}, v0",
                "vsetvli zero, {c}, e32, m4, ta, ma",
                "vse32.v v16, ({o})",
                "add {total}, {total}, {c}",
                "slli {c}, {c}, 2",
                "add {o}, {o}, {c}",
                "add {base}, {base}, {vl}",
                "slli {t}, {vl}, 2",
                "add {p}, {p}, {t}",
                "sub {n}, {n}, {vl}",
                "bnez {n}, 2b",
                ".option pop",
                p = inout(reg) p, o = inout(reg) o, n = inout(reg) n, vl = out(reg) _,
                t = out(reg) _, c = out(reg) _, base = out(reg) _, total = out(reg) total,
                thr = in(freg) threshold
            );
        }
        let _ = (p, o, n);
        total
    }

    /// The `k = idx.len()` largest elements, NaN ignored, descending, ties to
    /// the lower index. See `x86::reduce::topk_f32`.
    ///
    /// # Panics
    /// If `idx.len() != val.len()`.
    pub fn topk_f32(self, a: &[f32], idx: &mut [u32], val: &mut [f32]) -> usize {
        crate::reduce_common::topk_with(a, idx, val, |c, t, b| self.compress_gt_f32(c, t, b))
    }

    /// Dot product of two `i8` slices into `i32` (`vwmul` + `vwredsum`). Wraps on overflow.
    ///
    /// # Panics
    /// If the slices differ in length.
    pub fn dot_i8_i32(self, a: &[i8], b: &[i8]) -> i32 {
        assert_eq!(a.len(), b.len());
        let (mut pa, mut pb, mut n) = (a.as_ptr(), b.as_ptr(), a.len());
        if n == 0 {
            return 0;
        }
        let r: i64;
        // SAFETY: token proves V; reads `n` i8 from each.
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "vsetvli {t}, zero, e32, m1, ta, ma",
                "vmv.v.i v24, 0",
                "2:",
                "vsetvli {vl}, {n}, e8, m2, ta, ma",
                "vle8.v v8, ({a})",
                "vle8.v v10, ({b})",
                "vwmul.vv v16, v8, v10",
                "vsetvli zero, {vl}, e16, m4, ta, ma",
                "vwredsum.vs v24, v16, v24",
                "sub {n}, {n}, {vl}",
                "add {a}, {a}, {vl}",
                "add {b}, {b}, {vl}",
                "bnez {n}, 2b",
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vmv.x.s {r}, v24",
                ".option pop",
                a = inout(reg) pa, b = inout(reg) pb, n = inout(reg) n, vl = out(reg) _,
                t = out(reg) _, r = out(reg) r
            );
        }
        let _ = (pa, pb, n);
        r as i32
    }
}
