//! VSX slice kernels (Power ISA 2.06+: POWER7 and later, both endians) in
//! stable `asm!`.
//!
//! - **Loads/stores.** `lxvw4x` / `stxvw4x` take any address and put word `i`
//!   of memory into word element `i` (big-endian element numbering) on both
//!   big- and little-endian systems, so there is no alignment head, no permute
//!   and no endian special case for the element-wise kernels.
//! - **Registers.** VSX has 64 x 128-bit registers (the FPRs are `vs0`-`vs31`,
//!   the VMX registers `vs32`-`vs63`). The kernels keep everything in
//!   `vs32`-`vs63`, so VMX integer instructions (`v0` = `vs32`) mix freely
//!   with VSX float ones, and reductions run [`ACCUMULATORS`] independent sums
//!   over 32 floats per trip.
//! - **IEEE.** Unlike VMX, VSX float ops honour denormals (no `VSCR[NJ]`), and
//!   there is a real multiply, so `mul_f32` and `affine_u16_f32` match the
//!   portable code bit for bit.
//!
//! The whole-vector parts are `pub(crate)` so the ISA 3.0 kernels
//! (`isa300`, powerpc64) can add their length-controlled tails to them.

use crate::powerpc::{Feature, FeatureSet};

/// Vectors per trip in the element-wise loops.
pub const UNROLL: usize = 4;
/// Independent vector accumulators in the reductions.
pub const ACCUMULATORS: usize = 8;

/// Proof of VSX (ISA 2.06), switched on.
#[derive(Debug, Clone, Copy)]
pub struct Vsx(());

/// 16-byte splat for `lxvw4x` (any alignment works; 16 keeps it in one line).
#[repr(C, align(16))]
pub(crate) struct Splat(pub(crate) [u32; 4]);

impl Splat {
    pub(crate) fn f32(x: f32) -> Self {
        Splat([x.to_bits(); 4])
    }
}

/// `out = op(a, b)` over `n` whole vectors: four per trip, then one by one.
/// `$op` receives (dest, a, b) as VSX numbers and as VMX numbers.
macro_rules! vsx_map2 {
    ($name:ident, $t:ty, $op:ident) => {
        /// # Safety
        /// VSX on; `a`, `b` readable and `o` writable for `nv * 16` bytes.
        pub(crate) unsafe fn $name(mut a: *const $t, mut b: *const $t, mut o: *mut $t, nv: usize) {
            let (n4, n1) = (nv / UNROLL, nv % UNROLL);
            if n4 > 0 {
                // SAFETY: caller's contract; `n4` trips of 64 bytes.
                unsafe {
                    core::arch::asm!(
                        "mtctr {n}",
                        "2:",
                        "lxvw4x 32, {a}, {c0}", "lxvw4x 33, {a}, {c16}",
                        "lxvw4x 34, {a}, {c32}", "lxvw4x 35, {a}, {c48}",
                        "lxvw4x 36, {b}, {c0}", "lxvw4x 37, {b}, {c16}",
                        "lxvw4x 38, {b}, {c32}", "lxvw4x 39, {b}, {c48}",
                        $op!("32", "32", "36", "0", "0", "4"),
                        $op!("33", "33", "37", "1", "1", "5"),
                        $op!("34", "34", "38", "2", "2", "6"),
                        $op!("35", "35", "39", "3", "3", "7"),
                        "stxvw4x 32, {o}, {c0}", "stxvw4x 33, {o}, {c16}",
                        "stxvw4x 34, {o}, {c32}", "stxvw4x 35, {o}, {c48}",
                        "addi {a}, {a}, 64",
                        "addi {b}, {b}, 64",
                        "addi {o}, {o}, 64",
                        "bdnz 2b",
                        a = inout(reg_nonzero) a,
                        b = inout(reg_nonzero) b,
                        o = inout(reg_nonzero) o,
                        n = in(reg) n4,
                        c0 = in(reg) 0usize, c16 = in(reg) 16usize,
                        c32 = in(reg) 32usize, c48 = in(reg) 48usize,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("ctr") _,
                        options(nostack),
                    );
                }
            }
            if n1 > 0 {
                // SAFETY: as above; `n1` more vectors.
                unsafe {
                    core::arch::asm!(
                        "mtctr {n}",
                        "2:",
                        "lxvw4x 32, 0, {a}",
                        "lxvw4x 36, 0, {b}",
                        $op!("32", "32", "36", "0", "0", "4"),
                        "stxvw4x 32, 0, {o}",
                        "addi {a}, {a}, 16",
                        "addi {b}, {b}, 16",
                        "addi {o}, {o}, 16",
                        "bdnz 2b",
                        a = inout(reg_nonzero) a => _,
                        b = inout(reg_nonzero) b => _,
                        o = inout(reg_nonzero) o => _,
                        n = in(reg) n1,
                        out("v0") _, out("v4") _, out("ctr") _,
                        options(nostack),
                    );
                }
            }
        }
    };
}

macro_rules! xv_add {
    ($d:literal, $a:literal, $b:literal, $vd:literal, $va:literal, $vb:literal) => {
        concat!("xvaddsp ", $d, ", ", $a, ", ", $b)
    };
}
macro_rules! xv_sub {
    ($d:literal, $a:literal, $b:literal, $vd:literal, $va:literal, $vb:literal) => {
        concat!("xvsubsp ", $d, ", ", $a, ", ", $b)
    };
}
macro_rules! xv_mul {
    ($d:literal, $a:literal, $b:literal, $vd:literal, $va:literal, $vb:literal) => {
        concat!("xvmulsp ", $d, ", ", $a, ", ", $b)
    };
}
macro_rules! v_add_i32 {
    ($d:literal, $a:literal, $b:literal, $vd:literal, $va:literal, $vb:literal) => {
        concat!("vadduwm ", $vd, ", ", $va, ", ", $vb)
    };
}

vsx_map2!(add_f32_body, f32, xv_add);
vsx_map2!(sub_f32_body, f32, xv_sub);
vsx_map2!(mul_f32_body, f32, xv_mul);
vsx_map2!(add_i32_body, i32, v_add_i32);

/// `out = a * b + c` over `nv` whole vectors (`xvmaddasp`, fused).
///
/// # Safety
/// VSX on; the four pointers cover `nv * 16` bytes.
pub(crate) unsafe fn fmadd_f32_body(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    o: *mut f32,
    nv: usize,
) {
    if nv == 0 {
        return;
    }
    // SAFETY: caller's contract.
    unsafe {
        core::arch::asm!(
            "mtctr {n}",
            "2:",
            "lxvw4x 32, 0, {a}",
            "lxvw4x 33, 0, {b}",
            "lxvw4x 34, 0, {c}",
            // vs34 = vs32 * vs33 + vs34
            "xvmaddasp 34, 32, 33",
            "stxvw4x 34, 0, {o}",
            "addi {a}, {a}, 16",
            "addi {b}, {b}, 16",
            "addi {c}, {c}, 16",
            "addi {o}, {o}, 16",
            "bdnz 2b",
            a = inout(reg_nonzero) a => _,
            b = inout(reg_nonzero) b => _,
            c = inout(reg_nonzero) c => _,
            o = inout(reg_nonzero) o => _,
            n = in(reg) nv,
            out("v0") _, out("v1") _, out("v2") _, out("ctr") _,
            options(nostack),
        );
    }
}

/// Sum of `trips * 32` floats from `a` in eight accumulators.
///
/// # Safety
/// VSX on; `a` readable for `trips * 128` bytes, `trips > 0`.
pub(crate) unsafe fn sum_body(a: *const f32, trips: usize) -> f32 {
    let mut out = [0f32; 4];
    // SAFETY: caller's contract; `stxvw4x` writes the 16 bytes of `out`.
    unsafe {
        core::arch::asm!(
            "xxlxor 40, 40, 40", "xxlxor 41, 41, 41", "xxlxor 42, 42, 42", "xxlxor 43, 43, 43",
            "xxlxor 44, 44, 44", "xxlxor 45, 45, 45", "xxlxor 46, 46, 46", "xxlxor 47, 47, 47",
            "mtctr {n}",
            "2:",
            "lxvw4x 32, {a}, {c0}", "lxvw4x 33, {a}, {c16}",
            "lxvw4x 34, {a}, {c32}", "lxvw4x 35, {a}, {c48}",
            "lxvw4x 36, {a}, {c64}", "lxvw4x 37, {a}, {c80}",
            "lxvw4x 38, {a}, {c96}", "lxvw4x 39, {a}, {c112}",
            "xvaddsp 40, 40, 32", "xvaddsp 41, 41, 33", "xvaddsp 42, 42, 34", "xvaddsp 43, 43, 35",
            "xvaddsp 44, 44, 36", "xvaddsp 45, 45, 37", "xvaddsp 46, 46, 38", "xvaddsp 47, 47, 39",
            "addi {a}, {a}, 128",
            "bdnz 2b",
            "xvaddsp 40, 40, 41", "xvaddsp 42, 42, 43", "xvaddsp 44, 44, 45", "xvaddsp 46, 46, 47",
            "xvaddsp 40, 40, 42", "xvaddsp 44, 44, 46", "xvaddsp 40, 40, 44",
            "xxsldwi 32, 40, 40, 2", "xvaddsp 40, 40, 32",
            "xxsldwi 32, 40, 40, 1", "xvaddsp 40, 40, 32",
            "stxvw4x 40, 0, {s}",
            a = inout(reg_nonzero) a => _,
            s = in(reg) out.as_mut_ptr(),
            n = in(reg) trips,
            c0 = in(reg) 0usize, c16 = in(reg) 16usize, c32 = in(reg) 32usize,
            c48 = in(reg) 48usize, c64 = in(reg) 64usize, c80 = in(reg) 80usize,
            c96 = in(reg) 96usize, c112 = in(reg) 112usize,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("ctr") _,
            options(nostack),
        );
    }
    out[0]
}

/// Dot product of `trips * 32` floats in eight fused accumulators.
///
/// # Safety
/// VSX on; `a` and `b` readable for `trips * 128` bytes, `trips > 0`.
pub(crate) unsafe fn dot_body(a: *const f32, b: *const f32, trips: usize) -> f32 {
    let mut out = [0f32; 4];
    // SAFETY: caller's contract; `stxvw4x` writes the 16 bytes of `out`.
    unsafe {
        core::arch::asm!(
            "xxlxor 48, 48, 48", "xxlxor 49, 49, 49", "xxlxor 50, 50, 50", "xxlxor 51, 51, 51",
            "xxlxor 52, 52, 52", "xxlxor 53, 53, 53", "xxlxor 54, 54, 54", "xxlxor 55, 55, 55",
            "mtctr {n}",
            "2:",
            "lxvw4x 32, {a}, {c0}", "lxvw4x 33, {a}, {c16}",
            "lxvw4x 34, {a}, {c32}", "lxvw4x 35, {a}, {c48}",
            "lxvw4x 36, {a}, {c64}", "lxvw4x 37, {a}, {c80}",
            "lxvw4x 38, {a}, {c96}", "lxvw4x 39, {a}, {c112}",
            "lxvw4x 40, {b}, {c0}", "lxvw4x 41, {b}, {c16}",
            "lxvw4x 42, {b}, {c32}", "lxvw4x 43, {b}, {c48}",
            "lxvw4x 44, {b}, {c64}", "lxvw4x 45, {b}, {c80}",
            "lxvw4x 46, {b}, {c96}", "lxvw4x 47, {b}, {c112}",
            // acc = a * b + acc
            "xvmaddasp 48, 32, 40", "xvmaddasp 49, 33, 41",
            "xvmaddasp 50, 34, 42", "xvmaddasp 51, 35, 43",
            "xvmaddasp 52, 36, 44", "xvmaddasp 53, 37, 45",
            "xvmaddasp 54, 38, 46", "xvmaddasp 55, 39, 47",
            "addi {a}, {a}, 128",
            "addi {b}, {b}, 128",
            "bdnz 2b",
            "xvaddsp 48, 48, 49", "xvaddsp 50, 50, 51", "xvaddsp 52, 52, 53", "xvaddsp 54, 54, 55",
            "xvaddsp 48, 48, 50", "xvaddsp 52, 52, 54", "xvaddsp 48, 48, 52",
            "xxsldwi 32, 48, 48, 2", "xvaddsp 48, 48, 32",
            "xxsldwi 32, 48, 48, 1", "xvaddsp 48, 48, 32",
            "stxvw4x 48, 0, {s}",
            a = inout(reg_nonzero) a => _,
            b = inout(reg_nonzero) b => _,
            s = in(reg) out.as_mut_ptr(),
            n = in(reg) trips,
            c0 = in(reg) 0usize, c16 = in(reg) 16usize, c32 = in(reg) 32usize,
            c48 = in(reg) 48usize, c64 = in(reg) 64usize, c80 = in(reg) 80usize,
            c96 = in(reg) 96usize, c112 = in(reg) 112usize,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("ctr") _,
            options(nostack),
        );
    }
    out[0]
}

/// The two halfwords of each word, as whole words in memory order.
/// `lxvw4x` makes word `w` hold halfwords `h0` (lower address) and `h1`:
/// little-endian `w = h0 | h1 << 16`, big-endian `w = h0 << 16 | h1`.
/// `half_lo!` builds the `h0` vector in `v2`, `half_hi!` the `h1` vector in
/// `v3`, from `w` in `v0` (`v29` = `0x0000ffff`, `v30` = `0xffff0000`,
/// `v31` = shift count 16).
#[cfg(target_endian = "little")]
macro_rules! half_lo {
    (bf16) => {
        "vslw 2, 0, 31"
    }; // h0 << 16
    (u16) => {
        "vand 2, 0, 29"
    }; // h0
}
#[cfg(target_endian = "little")]
macro_rules! half_hi {
    (bf16) => {
        "vand 3, 0, 30"
    }; // h1 << 16
    (u16) => {
        "vsrw 3, 0, 31"
    }; // h1
}
#[cfg(target_endian = "big")]
macro_rules! half_lo {
    (bf16) => {
        "vand 2, 0, 30"
    };
    (u16) => {
        "vsrw 2, 0, 31"
    };
}
#[cfg(target_endian = "big")]
macro_rules! half_hi {
    (bf16) => {
        "vslw 3, 0, 31"
    };
    (u16) => {
        "vand 3, 0, 29"
    };
}

impl Vsx {
    /// Token if the cached feature set has VSX.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has VSX.
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains_all(&[Feature::Altivec, Feature::Vsx])
            .then_some(Vsx(()))
    }

    /// For tokens that imply VSX (ISA 3.1): their own proof covers it.
    #[cfg_attr(not(target_arch = "powerpc64"), allow(dead_code))]
    #[inline]
    pub(crate) const fn implied() -> Self {
        Vsx(())
    }

    #[inline]
    fn map2<T: Copy>(
        a: &[T],
        b: &[T],
        out: &mut [T],
        scalar: impl Fn(T, T) -> T,
        body: unsafe fn(*const T, *const T, *mut T, usize),
    ) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let lanes = 16 / core::mem::size_of::<T>();
        let nv = out.len() / lanes;
        // SAFETY: the token proves VSX; `nv` whole vectors of each slice.
        unsafe { body(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), nv) };
        for i in nv * lanes..out.len() {
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

    /// `out[i] = a[i] * b[i]`.
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

    /// `out[i] = a[i] * b[i] + c[i]`, fused in the vector part (`xvmaddasp`).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: token proves VSX; `nv` whole vectors of each slice.
        unsafe { fmadd_f32_body(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), nv) };
        for i in nv * 4..out.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }

    /// Sum of all elements: [`ACCUMULATORS`] vector sums, scalar tail. NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 32;
        if trips == 0 {
            return crate::reduce_common::sum4(a);
        }
        // SAFETY: token proves VSX; `trips * 32` floats are in bounds.
        let s = unsafe { sum_body(a.as_ptr(), trips) };
        s + crate::reduce_common::sum4(&a[trips * 32..])
    }

    /// Dot product: [`ACCUMULATORS`] fused sums, scalar tail. NaN and inf propagate.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let trips = a.len() / 32;
        if trips == 0 {
            return crate::reduce_common::dot4(a, b, 0);
        }
        // SAFETY: token proves VSX; `trips * 32` floats of each are in bounds.
        let s = unsafe { dot_body(a.as_ptr(), b.as_ptr(), trips) };
        s + crate::reduce_common::dot4(a, b, trips * 32)
    }

    /// Max (`MAX`) or min over whole 32-float blocks, NaN ignored; the
    /// identity if there is no block. Returns the value and the count covered.
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
        let mut out = [0f32; 4];
        macro_rules! body {
            ($op:literal) => {
                // SAFETY: token proves VSX; `trips * 32` floats are in bounds;
                // `stxvw4x` writes the 16 bytes of `out`.
                unsafe {
                    core::arch::asm!(
                        "lxvw4x 48, 0, {k}",
                        "xxlor 40, 48, 48", "xxlor 41, 48, 48", "xxlor 42, 48, 48", "xxlor 43, 48, 48",
                        "xxlor 44, 48, 48", "xxlor 45, 48, 48", "xxlor 46, 48, 48", "xxlor 47, 48, 48",
                        "mtctr {n}",
                        "2:",
                        "lxvw4x 32, {a}, {c0}", "lxvw4x 33, {a}, {c16}",
                        "lxvw4x 34, {a}, {c32}", "lxvw4x 35, {a}, {c48}",
                        "lxvw4x 36, {a}, {c64}", "lxvw4x 37, {a}, {c80}",
                        "lxvw4x 38, {a}, {c96}", "lxvw4x 39, {a}, {c112}",
                        // NaN lanes (x != x) become the identity.
                        "xvcmpeqsp 56, 32, 32", "xxsel 32, 48, 32, 56", concat!($op, " 40, 40, 32"),
                        "xvcmpeqsp 57, 33, 33", "xxsel 33, 48, 33, 57", concat!($op, " 41, 41, 33"),
                        "xvcmpeqsp 58, 34, 34", "xxsel 34, 48, 34, 58", concat!($op, " 42, 42, 34"),
                        "xvcmpeqsp 59, 35, 35", "xxsel 35, 48, 35, 59", concat!($op, " 43, 43, 35"),
                        "xvcmpeqsp 60, 36, 36", "xxsel 36, 48, 36, 60", concat!($op, " 44, 44, 36"),
                        "xvcmpeqsp 61, 37, 37", "xxsel 37, 48, 37, 61", concat!($op, " 45, 45, 37"),
                        "xvcmpeqsp 62, 38, 38", "xxsel 38, 48, 38, 62", concat!($op, " 46, 46, 38"),
                        "xvcmpeqsp 63, 39, 39", "xxsel 39, 48, 39, 63", concat!($op, " 47, 47, 39"),
                        "addi {a}, {a}, 128",
                        "bdnz 2b",
                        concat!($op, " 40, 40, 41"), concat!($op, " 42, 42, 43"),
                        concat!($op, " 44, 44, 45"), concat!($op, " 46, 46, 47"),
                        concat!($op, " 40, 40, 42"), concat!($op, " 44, 44, 46"),
                        concat!($op, " 40, 40, 44"),
                        "xxsldwi 32, 40, 40, 2", concat!($op, " 40, 40, 32"),
                        "xxsldwi 32, 40, 40, 1", concat!($op, " 40, 40, 32"),
                        "stxvw4x 40, 0, {s}",
                        a = inout(reg_nonzero) a.as_ptr() => _,
                        k = in(reg) &k as *const Splat,
                        s = in(reg) out.as_mut_ptr(),
                        n = in(reg) trips,
                        c0 = in(reg) 0usize, c16 = in(reg) 16usize, c32 = in(reg) 32usize,
                        c48 = in(reg) 48usize, c64 = in(reg) 64usize, c80 = in(reg) 80usize,
                        c96 = in(reg) 96usize, c112 = in(reg) 112usize,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                        out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                        out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                        out("v15") _, out("v16") _, out("v24") _, out("v25") _, out("v26") _,
                        out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                        out("ctr") _,
                        options(nostack),
                    );
                }
            };
        }
        if MAX {
            body!("xvmaxsp");
        } else {
            body!("xvminsp");
        }
        (out[0], trips * 32)
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

    /// bfloat16 bits to `f32`, exact (shifts and masks, no float ops).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn bf16_to_f32(self, a: &[u16], out: &mut [f32]) {
        assert_eq!(a.len(), out.len());
        let trips = a.len() / 8;
        if trips > 0 {
            // SAFETY: token proves VSX; `trips` x 8 elements of `a` and `out`.
            unsafe {
                core::arch::asm!(
                    "vspltisw 31, -16", // shift count 16 (low 5 bits)
                    "vspltisw 30, -1",
                    "vslw 30, 30, 31", // 0xffff0000
                    "mtctr {n}",
                    "2:",
                    "lxvw4x 32, 0, {a}",
                    half_lo!(bf16),
                    half_hi!(bf16),
                    "vmrghw 4, 2, 3",
                    "vmrglw 5, 2, 3",
                    "stxvw4x 36, 0, {o}",
                    "stxvw4x 37, {o}, {c16}",
                    "addi {a}, {a}, 16",
                    "addi {o}, {o}, 32",
                    "bdnz 2b",
                    a = inout(reg_nonzero) a.as_ptr() => _,
                    o = inout(reg_nonzero) out.as_mut_ptr() => _,
                    n = in(reg) trips,
                    c16 = in(reg) 16usize,
                    out("v0") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                    out("v30") _, out("v31") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        crate::reduce_common::bf16_to_f32_scalar(a, out, trips * 8);
    }

    /// `out[i] = a * q[i] + b`, two roundings (`xvmulsp`, `xvaddsp`), bit-identical
    /// to the portable code.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn affine_u16_f32(self, q: &[u16], a: f32, b: f32, out: &mut [f32]) {
        assert_eq!(q.len(), out.len());
        let trips = q.len() / 8;
        if trips > 0 {
            let (ka, kb) = (Splat::f32(a), Splat::f32(b));
            // SAFETY: token proves VSX; `trips` x 8 elements of `q` and `out`.
            unsafe {
                core::arch::asm!(
                    "vspltisw 31, -16", // shift count 16
                    "vspltisw 29, -1",
                    "vsrw 29, 29, 31", // 0x0000ffff
                    "lxvw4x 40, 0, {ka}",
                    "lxvw4x 41, 0, {kb}",
                    "mtctr {n}",
                    "2:",
                    "lxvw4x 32, 0, {q}",
                    half_lo!(u16),
                    half_hi!(u16),
                    "vmrghw 4, 2, 3",
                    "vmrglw 5, 2, 3",
                    "xvcvuxwsp 36, 36",
                    "xvcvuxwsp 37, 37",
                    "xvmulsp 36, 36, 40",
                    "xvmulsp 37, 37, 40",
                    "xvaddsp 36, 36, 41",
                    "xvaddsp 37, 37, 41",
                    "stxvw4x 36, 0, {o}",
                    "stxvw4x 37, {o}, {c16}",
                    "addi {q}, {q}, 16",
                    "addi {o}, {o}, 32",
                    "bdnz 2b",
                    q = inout(reg_nonzero) q.as_ptr() => _,
                    o = inout(reg_nonzero) out.as_mut_ptr() => _,
                    ka = in(reg) &ka as *const Splat,
                    kb = in(reg) &kb as *const Splat,
                    n = in(reg) trips,
                    c16 = in(reg) 16usize,
                    out("v0") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                    out("v8") _, out("v9") _, out("v29") _, out("v31") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        crate::reduce_common::affine_u16_f32_scalar(q, a, b, out, trips * 8);
    }
}
