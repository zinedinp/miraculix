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
use crate::formats::bfp::{BFP_ROW, BfpElem};
use crate::reduce_common::{EXP_HI, EXP_LO, LN2_HI, LN2_LO, LOG2E, P};

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

impl Rvv {
    /// `out[i] = exp(a[i])`, an approximation. Same algorithm, constants and
    /// contract as `x86::reduce::exp_f32`: relative error <= 3e-7 for inputs in
    /// `[-87.3, 88.7]`, `0` below `-87.33655`, `+inf` above `88.72283`, NaN
    /// stays NaN. The polynomial is fused (`vfmadd`/`vfnmsac`), so the last bit
    /// can differ from the portable scalar path.
    ///
    /// # Panics
    /// If `a.len() != out.len()`.
    pub fn exp_f32(self, a: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), out.len());
        let (mut pa, mut po, mut n) = (a.as_ptr(), out.as_mut_ptr(), a.len());
        if n == 0 {
            return;
        }
        // SAFETY: token proves V; reads and writes `n` f32. Note the rounding:
        // `vfcvt.x.f.v` uses `frm` (nearest even by default, as `asm!` code
        // must assume the ABI default).
        unsafe {
            rvv_asm!(
                ".option push",
                ".option arch, +v",
                "2:",
                "vsetvli {vl}, {n}, e32, m2, ta, ma",
                "vle32.v v8, ({a})",
                "vfmax.vf v10, v8, {lo}",
                "vfmin.vf v10, v10, {hi}",
                "vfmul.vf v12, v10, {log2e}",
                "vfcvt.x.f.v v12, v12",
                "vfcvt.f.x.v v14, v12",
                "vmv.v.v v16, v10",
                "vfnmsac.vf v16, {ln2hi}, v14",
                "vfnmsac.vf v16, {ln2lo}, v14",
                "vfmv.v.f v18, {p0}",
                "vfmv.v.f v20, {p1}",
                "vfmadd.vv v18, v16, v20",
                "vfmv.v.f v20, {p2}",
                "vfmadd.vv v18, v16, v20",
                "vfmv.v.f v20, {p3}",
                "vfmadd.vv v18, v16, v20",
                "vfmv.v.f v20, {p4}",
                "vfmadd.vv v18, v16, v20",
                "vfmv.v.f v20, {p5}",
                "vfmadd.vv v18, v16, v20",
                "vfmul.vv v22, v16, v16",
                "vfadd.vf v20, v16, {one}",
                "vfmacc.vv v20, v18, v22",
                "vsra.vi v24, v12, 1",
                "vsub.vv v26, v12, v24",
                "li {t}, 127",
                "vadd.vx v24, v24, {t}",
                "vadd.vx v26, v26, {t}",
                "vsll.vi v24, v24, 23",
                "vsll.vi v26, v26, 23",
                "vfmul.vv v20, v20, v24",
                "vfmul.vv v20, v20, v26",
                "vmflt.vf v0, v8, {lo}",
                "vmerge.vim v20, v20, 0, v0",
                "vmfgt.vf v0, v8, {hi}",
                "vfmerge.vfm v20, v20, {inf}, v0",
                "vmfne.vv v0, v8, v8",
                "vmerge.vvm v20, v20, v8, v0",
                "vse32.v v20, ({o})",
                "sub {n}, {n}, {vl}",
                "slli {t}, {vl}, 2",
                "add {a}, {a}, {t}",
                "add {o}, {o}, {t}",
                "bnez {n}, 2b",
                ".option pop",
                a = inout(reg) pa, o = inout(reg) po, n = inout(reg) n, vl = out(reg) _, t = out(reg) _,
                lo = in(freg) EXP_LO, hi = in(freg) EXP_HI, log2e = in(freg) LOG2E,
                ln2hi = in(freg) LN2_HI, ln2lo = in(freg) LN2_LO, one = in(freg) 1.0f32,
                inf = in(freg) f32::INFINITY,
                p0 = in(freg) P[0], p1 = in(freg) P[1], p2 = in(freg) P[2],
                p3 = in(freg) P[3], p4 = in(freg) P[4], p5 = in(freg) P[5]
            );
        }
        let _ = (pa, po, n);
    }
}

/// One BFP_B row (16 values) per call. `LOAD` leaves the 16 `f32` bit patterns
/// in `v8` (e32, m4, vl 16). The rest follows `bfp_simd::row_scalar` step for
/// step; `PACK` narrows the codes in `v24` and stores `row_bytes` bytes.
macro_rules! bfp_row {
    ($name:ident, $load:expr, $bits:literal, $shift:literal, $rnd:literal, $limit:literal, $pack:expr) => {
        /// # Safety
        /// `V` available; `s` has 16 source values, `e` one byte, `o` the row's data bytes.
        unsafe fn $name(s: *const u8, e: *mut u8, o: *mut u8) {
            unsafe {
                rvv_asm!(
                    ".option push",
                    ".option arch, +v",
                    $load,
                    "vsrl.vi v12, v8, 23",
                    "li {t}, 255",
                    "vand.vx v12, v12, {t}",
                    "vmv.s.x v24, zero",
                    "vredmaxu.vs v24, v12, v24",
                    "vmv.x.s {sh}, v24",
                    "sb {sh}, 0({e})",
                    "vrsub.vx v16, v12, {sh}",
                    "li {t}, 31",
                    "vminu.vx v16, v16, {t}",
                    "li {t}, 0x7fffff",
                    "vand.vx v20, v8, {t}",
                    "li {t}, 0x800000",
                    "vor.vx v20, v20, {t}",
                    "vsrl.vv v20, v20, v16",
                    concat!("li {t}, ", $shift),
                    "vsrl.vx v28, v20, {t}",
                    "vand.vi v28, v28, 1",
                    "vadd.vv v20, v20, v28",
                    concat!("li {t}, ", $rnd),
                    "vadd.vx v20, v20, {t}",
                    concat!("li {t}, ", $shift),
                    "vsrl.vx v20, v20, {t}",
                    concat!("li {t}, ", $limit),
                    "vminu.vx v20, v20, {t}",
                    "vmsne.vi v0, v12, 0",
                    "vmsne.vi v1, v20, 0",
                    "vmand.mm v0, v0, v1",
                    "vsrl.vi v28, v8, 31",
                    concat!("li {t}, ", $bits),
                    "vsll.vx v28, v28, {t}",
                    "vor.vv v20, v20, v28",
                    "vmv.v.i v24, 0",
                    "vmerge.vvm v24, v24, v20, v0",
                    $pack,
                    ".option pop",
                    s = in(reg) s, e = in(reg) e, o = in(reg) o, t = out(reg) _, sh = out(reg) _
                );
            }
        }
    };
}

macro_rules! load_f32 {
    () => {
        "vsetivli zero, 16, e32, m4, ta, ma\nvle32.v v8, ({s})"
    };
}
macro_rules! load_bf16 {
    () => {
        "vsetivli zero, 16, e16, m2, ta, ma\nvle16.v v16, ({s})\nvsetivli zero, 16, e32, m4, ta, ma\nvzext.vf2 v8, v16\nvsll.vi v8, v8, 16"
    };
}
// Narrow the 16 codes in v24 (e32) to 16 bytes in v12.
macro_rules! narrow_to_bytes {
    () => {
        "vsetivli zero, 16, e16, m2, ta, ma\nvnsrl.wi v16, v24, 0\nvsetivli zero, 16, e8, m1, ta, ma\nvnsrl.wi v12, v16, 0\n"
    };
}
macro_rules! pack7 {
    () => {
        concat!(narrow_to_bytes!(), "vse8.v v12, ({o})")
    };
}
macro_rules! pack3 {
    () => {
        concat!(
            narrow_to_bytes!(),
            "vsetivli zero, 8, e16, m1, ta, ma\nli {t}, 15\nvand.vx v14, v12, {t}\nvsrl.vi v15, v12, 4\n",
            "li {t}, 240\nvand.vx v15, v15, {t}\nvor.vv v14, v14, v15\n",
            "vsetivli zero, 8, e8, mf2, ta, ma\nvnsrl.wi v16, v14, 0\nvse8.v v16, ({o})"
        )
    };
}
macro_rules! pack1 {
    () => {
        concat!(
            narrow_to_bytes!(),
            "vsetivli zero, 4, e32, m1, ta, ma\nvand.vi v14, v12, 3\n",
            "li {t}, 12\nvsrl.vi v15, v12, 6\nvand.vx v15, v15, {t}\n",
            "li {t}, 48\nvsrl.vi v16, v12, 12\nvand.vx v16, v16, {t}\n",
            "li {t}, 192\nvsrl.vi v17, v12, 18\nvand.vx v17, v17, {t}\n",
            "vor.vv v14, v14, v15\nvor.vv v16, v16, v17\nvor.vv v14, v14, v16\n",
            "vsetivli zero, 4, e16, mf2, ta, ma\nvnsrl.wi v18, v14, 0\n",
            "vsetivli zero, 4, e8, mf4, ta, ma\nvnsrl.wi v19, v18, 0\nvse8.v v19, ({o})"
        )
    };
}

bfp_row!(row_m7_f32, load_f32!(), "7", "17", "65535", "127", pack7!());
bfp_row!(row_m3_f32, load_f32!(), "3", "21", "1048575", "7", pack3!());
bfp_row!(row_m1_f32, load_f32!(), "1", "23", "4194303", "1", pack1!());
bfp_row!(row_m7_bf16, load_bf16!(), "7", "17", "65535", "127", pack7!());
bfp_row!(row_m3_bf16, load_bf16!(), "3", "21", "1048575", "7", pack3!());
bfp_row!(row_m1_bf16, load_bf16!(), "1", "23", "4194303", "1", pack1!());

impl Rvv {
    /// Encodes `f32` rows as Tenstorrent `BFP_B` (see [`crate::formats::bfp`]);
    /// bit-exact with the portable encoder. One row (16 values) per loop trip.
    ///
    /// # Panics
    /// If `src.len()` is not a multiple of 16, or `exps` / `data` have the wrong length.
    pub fn bfp_encode_rows(self, elem: BfpElem, src: &[f32], exps: &mut [u8], data: &mut [u8]) {
        let rb = elem.row_bytes();
        assert_eq!(src.len() % BFP_ROW, 0);
        assert!(exps.len() == src.len() / BFP_ROW && data.len() == exps.len() * rb);
        for (r, e) in exps.iter_mut().enumerate() {
            let s = src[r * BFP_ROW..][..BFP_ROW].as_ptr() as *const u8;
            let d = data[r * rb..][..rb].as_mut_ptr();
            // SAFETY: token proves V; `s` has 16 `f32`, `d` has `rb` bytes.
            unsafe {
                match elem {
                    BfpElem::M7 => row_m7_f32(s, e, d),
                    BfpElem::M3 => row_m3_f32(s, e, d),
                    BfpElem::M1 => row_m1_f32(s, e, d),
                }
            }
        }
    }

    /// Token-taking [`crate::formats::bfp::bfp_decode_rows`] (portable decoder).
    pub fn bfp_decode_rows(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [f32]) {
        crate::formats::bfp::bfp_decode_rows(elem, exps, data, out);
    }

    /// Token-taking [`crate::formats::bfp::bfp_decode_rows_bf16`] (portable decoder).
    pub fn bfp_decode_rows_bf16(self, elem: BfpElem, exps: &[u8], data: &[u8], out: &mut [u16]) {
        crate::formats::bfp::bfp_decode_rows_bf16(elem, exps, data, out);
    }

    /// [`Self::bfp_encode_rows`] for bfloat16 input (`u16` bit patterns).
    ///
    /// # Panics
    /// If `src.len()` is not a multiple of 16, or `exps` / `data` have the wrong length.
    pub fn bfp_encode_rows_bf16(self, elem: BfpElem, src: &[u16], exps: &mut [u8], data: &mut [u8]) {
        let rb = elem.row_bytes();
        assert_eq!(src.len() % BFP_ROW, 0);
        assert!(exps.len() == src.len() / BFP_ROW && data.len() == exps.len() * rb);
        for (r, e) in exps.iter_mut().enumerate() {
            let s = src[r * BFP_ROW..][..BFP_ROW].as_ptr() as *const u8;
            let d = data[r * rb..][..rb].as_mut_ptr();
            // SAFETY: token proves V; `s` has 16 `u16`, `d` has `rb` bytes.
            unsafe {
                match elem {
                    BfpElem::M7 => row_m7_bf16(s, e, d),
                    BfpElem::M3 => row_m3_bf16(s, e, d),
                    BfpElem::M1 => row_m1_bf16(s, e, d),
                }
            }
        }
    }
}
