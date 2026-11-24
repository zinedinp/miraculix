//! AltiVec / VMX slice kernels in stable `asm!` (big-endian: G4, G5, Cell PPE,
//! Xenon), the `riscv::ops::rvv` pattern on PowerPC.
//!
//! **Loads.** Classic VMX has only aligned `lvx`, so an unaligned vector at `p`
//! is `vperm(lvx(p), lvx(p + 15), lvsl(p))`. Both loads stay inside the blocks
//! holding `p..p + 16`, so nothing outside the slice's own 16-byte blocks is
//! touched, and for an aligned `p` the permute is the identity.
//!
//! **Stores.** `stvx` is aligned only: the first `0..=3` elements are written
//! in scalar code until `out` is 16-byte aligned, then whole vectors, then a
//! scalar tail. Inputs keep their own alignment (one `lvsl` each).
//!
//! **Registers.** VMX has 32 vector registers, twice the SSE/AVX2 file, so the
//! reductions keep [`ACCUMULATORS`] independent sums (8 vectors = 32 floats per
//! loop trip) and the element-wise loops have [`UNROLL`] vectors in flight. That
//! hides the 4-12 cycle FP latency of the in-order Cell and Xenon cores.
//!
//! **Non-Java mode.** Linux and Mac OS start threads with `VSCR[NJ] = 1`, so
//! VMX float ops flush denormal inputs and results to zero. Results can then
//! differ from the scalar path for values below `f32::MIN_POSITIVE`.

use crate::powerpc::{Feature, FeatureSet};

/// Vectors per trip in the element-wise loops.
pub const UNROLL: usize = 4;
/// Independent vector accumulators in the reductions.
pub const ACCUMULATORS: usize = 8;

/// Proof that AltiVec/VMX is available and switched on.
#[derive(Debug, Clone, Copy)]
pub struct Vmx(());

/// 16-byte aligned splat for `lvx`.
#[repr(C, align(16))]
struct Splat([u32; 4]);

impl Splat {
    fn f32(x: f32) -> Self {
        Splat([x.to_bits(); 4])
    }
}

/// Scalar elements to write before `p` is 16-byte aligned (at most `n`).
#[inline]
fn head<T>(p: *const T, n: usize) -> usize {
    let mis = p as usize & 15;
    let to_align = if mis == 0 {
        0
    } else {
        (16 - mis) / core::mem::size_of::<T>()
    };
    to_align.min(n)
}

/// One unaligned vector from `{p} + {lo}` into `v$d`, `v$t` scratch, control `v$c`.
macro_rules! ld {
    ($p:literal, $lo:literal, $hi:literal, $d:literal, $t:literal, $c:literal) => {
        concat!(
            "lvx ", $d, ", {", $p, "}, {", $lo, "}\n", "lvx ", $t, ", {", $p, "}, {", $hi, "}\n",
            "vperm ", $d, ", ", $d, ", ", $t, ", ", $c
        )
    };
}

/// Element-wise `out = op(a, b)` over whole vectors: [`UNROLL`] per trip, then one by one.
/// `$op` gets `(dest, a, b)` register numbers and returns the instruction.
macro_rules! map2 {
    ($name:ident, $t:ty, $op:ident) => {
        /// # Safety
        /// `a`, `b` readable and `o` writable for `n` elements; `o` 16-byte aligned.
        unsafe fn $name(mut a: *const $t, mut b: *const $t, mut o: *mut $t, n: usize) {
            let lanes = 16 / core::mem::size_of::<$t>();
            let (n4, n1) = (n / (lanes * UNROLL), n / lanes % UNROLL);
            if n4 > 0 {
                // SAFETY: caller's contract; `n4` trips of 64 bytes stay in bounds.
                unsafe {
                    core::arch::asm!(
                        "lvsl 20, 0, {a}",
                        "lvsl 21, 0, {b}",
                        "vspltisw 19, -1",
                        "vslw 19, 19, 19",
                        "mtctr {n}",
                        "2:",
                        ld!("a", "c0", "c15", "0", "1", "20"),
                        ld!("b", "c0", "c15", "2", "3", "21"),
                        ld!("a", "c16", "c31", "4", "5", "20"),
                        ld!("b", "c16", "c31", "6", "7", "21"),
                        ld!("a", "c32", "c47", "8", "9", "20"),
                        ld!("b", "c32", "c47", "10", "11", "21"),
                        ld!("a", "c48", "c63", "12", "13", "20"),
                        ld!("b", "c48", "c63", "14", "15", "21"),
                        $op!("0", "0", "2"),
                        $op!("4", "4", "6"),
                        $op!("8", "8", "10"),
                        $op!("12", "12", "14"),
                        "stvx 0, {o}, {c0}",
                        "stvx 4, {o}, {c16}",
                        "stvx 8, {o}, {c32}",
                        "stvx 12, {o}, {c48}",
                        "addi {a}, {a}, 64",
                        "addi {b}, {b}, 64",
                        "addi {o}, {o}, 64",
                        "bdnz 2b",
                        a = inout(reg_nonzero) a,
                        b = inout(reg_nonzero) b,
                        o = inout(reg_nonzero) o,
                        n = in(reg) n4,
                        c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                        c31 = in(reg) 31usize, c32 = in(reg) 32usize, c47 = in(reg) 47usize,
                        c48 = in(reg) 48usize, c63 = in(reg) 63usize,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                        out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                        out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                        out("v15") _, out("v19") _, out("v20") _, out("v21") _, out("ctr") _,
                        options(nostack),
                    );
                }
            }
            if n1 > 0 {
                // SAFETY: as above; `n1` more vectors follow in bounds.
                unsafe {
                    core::arch::asm!(
                        "lvsl 20, 0, {a}",
                        "lvsl 21, 0, {b}",
                        "vspltisw 19, -1",
                        "vslw 19, 19, 19",
                        "mtctr {n}",
                        "2:",
                        ld!("a", "c0", "c15", "0", "1", "20"),
                        ld!("b", "c0", "c15", "2", "3", "21"),
                        $op!("0", "0", "2"),
                        "stvx 0, {o}, {c0}",
                        "addi {a}, {a}, 16",
                        "addi {b}, {b}, 16",
                        "addi {o}, {o}, 16",
                        "bdnz 2b",
                        a = inout(reg_nonzero) a => _,
                        b = inout(reg_nonzero) b => _,
                        o = inout(reg_nonzero) o => _,
                        n = in(reg) n1,
                        c0 = in(reg) 0usize, c15 = in(reg) 15usize,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v19") _, out("v20") _, out("v21") _, out("ctr") _,
                        options(nostack),
                    );
                }
            }
        }
    };
}

macro_rules! op_add_f32 {
    ($d:literal, $a:literal, $b:literal) => {
        concat!("vaddfp ", $d, ", ", $a, ", ", $b)
    };
}
macro_rules! op_sub_f32 {
    ($d:literal, $a:literal, $b:literal) => {
        concat!("vsubfp ", $d, ", ", $a, ", ", $b)
    };
}
// VMX has no plain multiply: `a * b + (-0.0)` rounds once and keeps the sign of zero.
macro_rules! op_mul_f32 {
    ($d:literal, $a:literal, $b:literal) => {
        concat!("vmaddfp ", $d, ", ", $a, ", ", $b, ", 19")
    };
}
macro_rules! op_add_i32 {
    ($d:literal, $a:literal, $b:literal) => {
        concat!("vadduwm ", $d, ", ", $a, ", ", $b)
    };
}

map2!(add_f32_body, f32, op_add_f32);
map2!(sub_f32_body, f32, op_sub_f32);
map2!(mul_f32_body, f32, op_mul_f32);
map2!(add_i32_body, i32, op_add_i32);

impl Vmx {
    /// Token if the cached feature set has AltiVec.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has AltiVec.
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains(Feature::Altivec).then_some(Vmx(()))
    }

    /// Shared driver of the two-input element-wise ops: scalar head until `out`
    /// is aligned, `body` for whole vectors, scalar tail.
    #[inline]
    fn map2<T: Copy>(
        a: &[T],
        b: &[T],
        out: &mut [T],
        scalar: impl Fn(T, T) -> T,
        body: unsafe fn(*const T, *const T, *mut T, usize),
    ) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let n = out.len();
        let h = head(out.as_ptr(), n);
        for i in 0..h {
            out[i] = scalar(a[i], b[i]);
        }
        let lanes = 16 / core::mem::size_of::<T>();
        let whole = (n - h) / lanes * lanes;
        // SAFETY: the token proves VMX; `a`, `b` and `out` have `n` elements and
        // `h + whole <= n`; `out[h..]` is 16-byte aligned.
        unsafe {
            body(
                a[h..].as_ptr(),
                b[h..].as_ptr(),
                out[h..].as_mut_ptr(),
                whole,
            );
        }
        for i in h + whole..n {
            out[i] = scalar(a[i], b[i]);
        }
    }

    /// `out[i] = a[i] + b[i]`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn add_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x + y, add_f32_body);
    }

    /// `out[i] = a[i] - b[i]`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn sub_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x - y, sub_f32_body);
    }

    /// `out[i] = a[i] * b[i]` (one rounding, as `vmaddfp` with `-0.0`).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn mul_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        Self::map2(a, b, out, |x, y| x * y, mul_f32_body);
    }

    /// `out[i] = a[i] + b[i]`, wrapping.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn add_i32(self, a: &[i32], b: &[i32], out: &mut [i32]) {
        Self::map2(a, b, out, i32::wrapping_add, add_i32_body);
    }

    /// `out[i] = a[i] * b[i] + c[i]`, fused (`vmaddfp`, one rounding) in the
    /// vector part; the scalar head and tail round twice.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
        let n = out.len();
        let h = head(out.as_ptr(), n);
        for i in 0..h {
            out[i] = a[i] * b[i] + c[i];
        }
        let nv = (n - h) / 4;
        if nv > 0 {
            let (pa, pb, pc, po) = (
                a[h..].as_ptr(),
                b[h..].as_ptr(),
                c[h..].as_ptr(),
                out[h..].as_mut_ptr(),
            );
            // SAFETY: token proves VMX; `nv` whole vectors of each slice from `h`
            // are in bounds and `po` is 16-byte aligned.
            unsafe {
                core::arch::asm!(
                    "lvsl 20, 0, {a}",
                    "lvsl 21, 0, {b}",
                    "lvsl 22, 0, {c}",
                    "mtctr {n}",
                    "2:",
                    ld!("a", "c0", "c15", "0", "1", "20"),
                    ld!("b", "c0", "c15", "2", "3", "21"),
                    ld!("c", "c0", "c15", "4", "5", "22"),
                    "vmaddfp 0, 0, 2, 4",
                    "stvx 0, {o}, {c0}",
                    "addi {a}, {a}, 16",
                    "addi {b}, {b}, 16",
                    "addi {c}, {c}, 16",
                    "addi {o}, {o}, 16",
                    "bdnz 2b",
                    a = inout(reg_nonzero) pa => _,
                    b = inout(reg_nonzero) pb => _,
                    c = inout(reg_nonzero) pc => _,
                    o = inout(reg_nonzero) po => _,
                    n = in(reg) nv,
                    c0 = in(reg) 0usize, c15 = in(reg) 15usize,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                    out("v5") _, out("v20") _, out("v21") _, out("v22") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        for i in h + nv * 4..n {
            out[i] = a[i] * b[i] + c[i];
        }
    }

    /// Sum of all elements, [`ACCUMULATORS`] vector sums plus a scalar tail.
    /// The order differs from the scalar path, so the last bits can too. NaN
    /// propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 32;
        if trips == 0 {
            return crate::reduce_common::sum4(a);
        }
        let mut sum = 0f32;
        // SAFETY: token proves VMX; `trips` x 32 floats of `a` are in bounds;
        // `stvewx` writes one aligned word to `sum`.
        unsafe {
            core::arch::asm!(
                "lvsl 20, 0, {a}",
                "vxor 24, 24, 24", "vxor 25, 25, 25", "vxor 26, 26, 26", "vxor 27, 27, 27",
                "vxor 28, 28, 28", "vxor 29, 29, 29", "vxor 30, 30, 30", "vxor 31, 31, 31",
                "mtctr {n}",
                "2:",
                ld!("a", "c0", "c15", "0", "1", "20"),
                ld!("a", "c16", "c31", "2", "3", "20"),
                ld!("a", "c32", "c47", "4", "5", "20"),
                ld!("a", "c48", "c63", "6", "7", "20"),
                "addi {a}, {a}, 64",
                ld!("a", "c0", "c15", "8", "9", "20"),
                ld!("a", "c16", "c31", "10", "11", "20"),
                ld!("a", "c32", "c47", "12", "13", "20"),
                ld!("a", "c48", "c63", "14", "15", "20"),
                "addi {a}, {a}, 64",
                "vaddfp 24, 24, 0", "vaddfp 25, 25, 2", "vaddfp 26, 26, 4", "vaddfp 27, 27, 6",
                "vaddfp 28, 28, 8", "vaddfp 29, 29, 10", "vaddfp 30, 30, 12", "vaddfp 31, 31, 14",
                "bdnz 2b",
                "vaddfp 24, 24, 25", "vaddfp 26, 26, 27", "vaddfp 28, 28, 29", "vaddfp 30, 30, 31",
                "vaddfp 24, 24, 26", "vaddfp 28, 28, 30", "vaddfp 24, 24, 28",
                "vsldoi 0, 24, 24, 8", "vaddfp 24, 24, 0",
                "vsldoi 0, 24, 24, 4", "vaddfp 24, 24, 0",
                "stvewx 24, 0, {s}",
                a = inout(reg_nonzero) a.as_ptr() => _,
                s = in(reg) &mut sum as *mut f32,
                n = in(reg) trips,
                c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                c31 = in(reg) 31usize, c32 = in(reg) 32usize, c47 = in(reg) 47usize,
                c48 = in(reg) 48usize, c63 = in(reg) 63usize,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                out("v15") _, out("v20") _, out("v24") _, out("v25") _, out("v26") _,
                out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::sum4(&a[trips * 32..])
    }

    /// Dot product `sum(a[i] * b[i])`: fused per lane, [`ACCUMULATORS`] vector
    /// sums, scalar tail. NaN and inf propagate.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let trips = a.len() / 32;
        if trips == 0 {
            return crate::reduce_common::dot4(a, b, 0);
        }
        let mut sum = 0f32;
        // SAFETY: token proves VMX; `trips` x 32 floats of `a` and `b` are in
        // bounds; `stvewx` writes one aligned word to `sum`.
        unsafe {
            core::arch::asm!(
                "lvsl 20, 0, {a}",
                "lvsl 21, 0, {b}",
                "vxor 24, 24, 24", "vxor 25, 25, 25", "vxor 26, 26, 26", "vxor 27, 27, 27",
                "vxor 28, 28, 28", "vxor 29, 29, 29", "vxor 30, 30, 30", "vxor 31, 31, 31",
                "mtctr {n}",
                "2:",
                ld!("a", "c0", "c15", "0", "1", "20"),
                ld!("b", "c0", "c15", "2", "3", "21"),
                ld!("a", "c16", "c31", "4", "5", "20"),
                ld!("b", "c16", "c31", "6", "7", "21"),
                ld!("a", "c32", "c47", "8", "9", "20"),
                ld!("b", "c32", "c47", "10", "11", "21"),
                ld!("a", "c48", "c63", "12", "13", "20"),
                ld!("b", "c48", "c63", "14", "15", "21"),
                "vmaddfp 24, 0, 2, 24", "vmaddfp 25, 4, 6, 25",
                "vmaddfp 26, 8, 10, 26", "vmaddfp 27, 12, 14, 27",
                "addi {a}, {a}, 64",
                "addi {b}, {b}, 64",
                ld!("a", "c0", "c15", "0", "1", "20"),
                ld!("b", "c0", "c15", "2", "3", "21"),
                ld!("a", "c16", "c31", "4", "5", "20"),
                ld!("b", "c16", "c31", "6", "7", "21"),
                ld!("a", "c32", "c47", "8", "9", "20"),
                ld!("b", "c32", "c47", "10", "11", "21"),
                ld!("a", "c48", "c63", "12", "13", "20"),
                ld!("b", "c48", "c63", "14", "15", "21"),
                "vmaddfp 28, 0, 2, 28", "vmaddfp 29, 4, 6, 29",
                "vmaddfp 30, 8, 10, 30", "vmaddfp 31, 12, 14, 31",
                "addi {a}, {a}, 64",
                "addi {b}, {b}, 64",
                "bdnz 2b",
                "vaddfp 24, 24, 25", "vaddfp 26, 26, 27", "vaddfp 28, 28, 29", "vaddfp 30, 30, 31",
                "vaddfp 24, 24, 26", "vaddfp 28, 28, 30", "vaddfp 24, 24, 28",
                "vsldoi 0, 24, 24, 8", "vaddfp 24, 24, 0",
                "vsldoi 0, 24, 24, 4", "vaddfp 24, 24, 0",
                "stvewx 24, 0, {s}",
                a = inout(reg_nonzero) a.as_ptr() => _,
                b = inout(reg_nonzero) b.as_ptr() => _,
                s = in(reg) &mut sum as *mut f32,
                n = in(reg) trips,
                c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                c31 = in(reg) 31usize, c32 = in(reg) 32usize, c47 = in(reg) 47usize,
                c48 = in(reg) 48usize, c63 = in(reg) 63usize,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                out("v15") _, out("v20") _, out("v21") _, out("v24") _, out("v25") _,
                out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
                out("v31") _, out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::dot4(a, b, trips * 32)
    }

    /// Largest (`MAX = true`) or smallest element, NaN ignored, over whole
    /// 32-float blocks, and how many elements that covered. The identity
    /// (`-inf` / `+inf`) if no block was complete or every lane was NaN.
    fn extreme<const MAX: bool>(a: &[f32]) -> (f32, usize) {
        let trips = a.len() / 32;
        let ident = if MAX {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        if trips == 0 {
            return (ident, 0);
        }
        let k = Splat::f32(ident);
        let mut out = ident;
        macro_rules! body {
            ($op:literal) => {
                // SAFETY: token proves VMX; `trips` x 32 floats of `a` are in
                // bounds; `k` is 16-byte aligned; `stvewx` writes one word.
                unsafe {
                    core::arch::asm!(
                        "lvsl 20, 0, {a}",
                        "lvx 22, 0, {k}",
                        "vor 24, 22, 22", "vor 25, 22, 22", "vor 26, 22, 22", "vor 27, 22, 22",
                        "vor 28, 22, 22", "vor 29, 22, 22", "vor 30, 22, 22", "vor 31, 22, 22",
                        "mtctr {n}",
                        "2:",
                        ld!("a", "c0", "c15", "0", "1", "20"),
                        ld!("a", "c16", "c31", "2", "3", "20"),
                        ld!("a", "c32", "c47", "4", "5", "20"),
                        ld!("a", "c48", "c63", "6", "7", "20"),
                        "addi {a}, {a}, 64",
                        ld!("a", "c0", "c15", "8", "9", "20"),
                        ld!("a", "c16", "c31", "10", "11", "20"),
                        ld!("a", "c32", "c47", "12", "13", "20"),
                        ld!("a", "c48", "c63", "14", "15", "20"),
                        "addi {a}, {a}, 64",
                        // NaN lanes (x != x) become the identity before the max/min.
                        "vcmpeqfp 1, 0, 0", "vsel 0, 22, 0, 1", concat!($op, " 24, 24, 0"),
                        "vcmpeqfp 3, 2, 2", "vsel 2, 22, 2, 3", concat!($op, " 25, 25, 2"),
                        "vcmpeqfp 5, 4, 4", "vsel 4, 22, 4, 5", concat!($op, " 26, 26, 4"),
                        "vcmpeqfp 7, 6, 6", "vsel 6, 22, 6, 7", concat!($op, " 27, 27, 6"),
                        "vcmpeqfp 9, 8, 8", "vsel 8, 22, 8, 9", concat!($op, " 28, 28, 8"),
                        "vcmpeqfp 11, 10, 10", "vsel 10, 22, 10, 11", concat!($op, " 29, 29, 10"),
                        "vcmpeqfp 13, 12, 12", "vsel 12, 22, 12, 13", concat!($op, " 30, 30, 12"),
                        "vcmpeqfp 15, 14, 14", "vsel 14, 22, 14, 15", concat!($op, " 31, 31, 14"),
                        "bdnz 2b",
                        concat!($op, " 24, 24, 25"), concat!($op, " 26, 26, 27"),
                        concat!($op, " 28, 28, 29"), concat!($op, " 30, 30, 31"),
                        concat!($op, " 24, 24, 26"), concat!($op, " 28, 28, 30"),
                        concat!($op, " 24, 24, 28"),
                        "vsldoi 0, 24, 24, 8", concat!($op, " 24, 24, 0"),
                        "vsldoi 0, 24, 24, 4", concat!($op, " 24, 24, 0"),
                        "stvewx 24, 0, {s}",
                        a = inout(reg_nonzero) a.as_ptr() => _,
                        k = in(reg) &k as *const Splat,
                        s = in(reg) &mut out as *mut f32,
                        n = in(reg) trips,
                        c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                        c31 = in(reg) 31usize, c32 = in(reg) 32usize, c47 = in(reg) 47usize,
                        c48 = in(reg) 48usize, c63 = in(reg) 63usize,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                        out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                        out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                        out("v15") _, out("v20") _, out("v22") _, out("v24") _, out("v25") _,
                        out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
                        out("v31") _, out("ctr") _,
                        options(nostack),
                    );
                }
            };
        }
        if MAX {
            body!("vmaxfp");
        } else {
            body!("vminfp");
        }
        (out, trips * 32)
    }

    /// Largest element, NaN ignored; `-inf` if there is none.
    pub fn reduce_max_f32(self, a: &[f32]) -> f32 {
        let (m, done) = Self::extreme::<true>(a);
        a[done..].iter().fold(m, |m, &x| if x > m { x } else { m })
    }

    /// Smallest element, NaN ignored; `+inf` if there is none.
    pub fn reduce_min_f32(self, a: &[f32]) -> f32 {
        let (m, done) = Self::extreme::<false>(a);
        a[done..].iter().fold(m, |m, &x| if x < m { x } else { m })
    }

    /// First index of the largest element and its value, NaN ignored; `None`
    /// if the slice is empty or all NaN. Vector max, then a scalar search.
    pub fn argmax_f32(self, a: &[f32]) -> Option<(usize, f32)> {
        let m = self.reduce_max_f32(a);
        a.iter().position(|&x| x == m).map(|i| (i, m))
    }

    /// bfloat16 bits to `f32`, exact (`vmrghh` / `vmrglh` with zero).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn bf16_to_f32(self, a: &[u16], out: &mut [f32]) {
        assert_eq!(a.len(), out.len());
        let n = out.len();
        let h = head(out.as_ptr(), n);
        crate::reduce_common::bf16_to_f32_scalar(&a[..h], &mut out[..h], 0);
        let trips = (n - h) / 8;
        if trips > 0 {
            // SAFETY: token proves VMX; `trips` x 8 elements of `a` and `out`
            // from `h` are in bounds; `out[h..]` is 16-byte aligned.
            unsafe {
                core::arch::asm!(
                    "lvsl 20, 0, {a}",
                    "vxor 18, 18, 18",
                    "mtctr {n}",
                    "2:",
                    ld!("a", "c0", "c15", "0", "1", "20"),
                    // Big-endian: (x, 0) halfword pairs are the words `x << 16`.
                    "vmrghh 2, 0, 18",
                    "vmrglh 3, 0, 18",
                    "stvx 2, {o}, {c0}",
                    "stvx 3, {o}, {c16}",
                    "addi {a}, {a}, 16",
                    "addi {o}, {o}, 32",
                    "bdnz 2b",
                    a = inout(reg_nonzero) a[h..].as_ptr() => _,
                    o = inout(reg_nonzero) out[h..].as_mut_ptr() => _,
                    n = in(reg) trips,
                    c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v18") _,
                    out("v20") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        crate::reduce_common::bf16_to_f32_scalar(a, out, h + trips * 8);
    }

    /// `out[i] = a * q[i] + b` with two roundings, bit-identical to the
    /// portable code (`vcfux` is exact for 16-bit values; the multiply is
    /// `vmaddfp` with `-0.0`, then `vaddfp`).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn affine_u16_f32(self, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
        assert_eq!(q.len(), out.len());
        let n = out.len();
        let h = head(out.as_ptr(), n);
        crate::reduce_common::affine_u16_f32_scalar(&q[..h], a, b, &mut out[..h], 0);
        let trips = (n - h) / 8;
        if trips > 0 {
            let (ka, kb) = (Splat::f32(a), Splat::f32(b));
            // SAFETY: token proves VMX; `trips` x 8 elements of `q` and `out`
            // from `h` are in bounds; `out[h..]` and the splats are aligned.
            unsafe {
                core::arch::asm!(
                    "lvsl 20, 0, {q}",
                    "lvx 16, 0, {ka}",
                    "lvx 17, 0, {kb}",
                    "vxor 18, 18, 18",
                    "vspltisw 19, -1",
                    "vslw 19, 19, 19",
                    "mtctr {n}",
                    "2:",
                    ld!("q", "c0", "c15", "0", "1", "20"),
                    // Big-endian: (0, x) halfword pairs are the words `x`.
                    "vmrghh 2, 18, 0",
                    "vmrglh 3, 18, 0",
                    "vcfux 2, 2, 0",
                    "vcfux 3, 3, 0",
                    "vmaddfp 2, 2, 16, 19",
                    "vmaddfp 3, 3, 16, 19",
                    "vaddfp 2, 2, 17",
                    "vaddfp 3, 3, 17",
                    "stvx 2, {o}, {c0}",
                    "stvx 3, {o}, {c16}",
                    "addi {q}, {q}, 16",
                    "addi {o}, {o}, 32",
                    "bdnz 2b",
                    q = inout(reg_nonzero) q[h..].as_ptr() => _,
                    o = inout(reg_nonzero) out[h..].as_mut_ptr() => _,
                    ka = in(reg) &ka as *const Splat,
                    kb = in(reg) &kb as *const Splat,
                    n = in(reg) trips,
                    c0 = in(reg) 0usize, c15 = in(reg) 15usize, c16 = in(reg) 16usize,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v16") _,
                    out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        crate::reduce_common::affine_u16_f32_scalar(q, a, b, out, h + trips * 8);
    }
}
