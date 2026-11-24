//! Power ISA 3.0 (POWER9+) kernels: length-controlled tails and FP16.
//!
//! - **Tails without a scalar loop.** `lxvl` / `stxvl` load and store only the
//!   first `n` bytes (`n` in bits 0:7 of the length register) and zero the rest
//!   of the register: the same idea as RVV's `vsetvl`. The whole vectors run
//!   the [`super::vsx`] bodies; the last `0..=3` elements are one `lxvl` /
//!   `stxvl` instead of a scalar loop. In little-endian mode both reverse the
//!   loaded bytes as a pair, so every lane holds a whole, correct value (in
//!   reverse lane order), which an element-wise op or a sum does not notice.
//! - **FP16.** `xvcvhpsp` / `xvcvsphp` convert the halfword in the low half of
//!   each word. `lxvh8x` / `stxvh8x` put halfword `i` of memory into halfword
//!   element `i` on both endians, so the packing below has no endian case.

use super::vsx::{
    add_f32_body, add_i32_body, dot_body, fmadd_f32_body, mul_f32_body, sub_f32_body, sum_body,
};
use crate::formats::{MiniFloat, Overflow};
use crate::powerpc::{Feature, FeatureSet};

/// Proof of the ISA 3.0 vector facility (POWER9+) with VSX on.
#[derive(Debug, Clone, Copy)]
pub struct Isa300(());

/// `lxvl` length operand: the byte count in bits 0:7.
#[inline]
const fn len_operand(bytes: usize) -> usize {
    bytes << 56
}

/// Two-input tail: the last `bytes` (< 16) bytes through `lxvl`/`stxvl`.
macro_rules! tail2 {
    ($a:expr, $b:expr, $o:expr, $bytes:expr, $op:literal) => {
        // SAFETY: the token proves ISA 3.0; `lxvl`/`stxvl` touch exactly
        // `$bytes` bytes from each pointer, all inside the slices.
        unsafe {
            core::arch::asm!(
                "lxvl 32, {a}, {len}",
                "lxvl 33, {b}, {len}",
                $op,
                "stxvl 32, {o}, {len}",
                a = in(reg_nonzero) $a,
                b = in(reg_nonzero) $b,
                o = in(reg_nonzero) $o,
                len = in(reg) len_operand($bytes),
                out("v0") _, out("v1") _,
                options(nostack),
            );
        }
    };
}

impl Isa300 {
    /// Token if the cached feature set has ISA 3.0 vector and VSX.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has ISA 3.0 vector and VSX.
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains_all(&[Feature::Altivec, Feature::Vsx, Feature::Power9Vector])
            .then_some(Isa300(()))
    }

    /// `out[i] = a[i] + b[i]`; the tail is one `lxvl`/`stxvl`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn add_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: ISA 3.0 implies VSX; `nv` whole vectors.
        unsafe { add_f32_body(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), nv) };
        let (d, bytes) = (nv * 4, out.len() % 4 * 4);
        if bytes > 0 {
            tail2!(
                a[d..].as_ptr(),
                b[d..].as_ptr(),
                out[d..].as_mut_ptr(),
                bytes,
                "xvaddsp 32, 32, 33"
            );
        }
    }

    /// `out[i] = a[i] - b[i]`; the tail is one `lxvl`/`stxvl`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn sub_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: ISA 3.0 implies VSX; `nv` whole vectors.
        unsafe { sub_f32_body(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), nv) };
        let (d, bytes) = (nv * 4, out.len() % 4 * 4);
        if bytes > 0 {
            tail2!(
                a[d..].as_ptr(),
                b[d..].as_ptr(),
                out[d..].as_mut_ptr(),
                bytes,
                "xvsubsp 32, 32, 33"
            );
        }
    }

    /// `out[i] = a[i] * b[i]`; the tail is one `lxvl`/`stxvl`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn mul_f32(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: ISA 3.0 implies VSX; `nv` whole vectors.
        unsafe { mul_f32_body(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), nv) };
        let (d, bytes) = (nv * 4, out.len() % 4 * 4);
        if bytes > 0 {
            tail2!(
                a[d..].as_ptr(),
                b[d..].as_ptr(),
                out[d..].as_mut_ptr(),
                bytes,
                "xvmulsp 32, 32, 33"
            );
        }
    }

    /// `out[i] = a[i] + b[i]`, wrapping; the tail is one `lxvl`/`stxvl`.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn add_i32(self, a: &[i32], b: &[i32], out: &mut [i32]) {
        assert!(a.len() == b.len() && b.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: ISA 3.0 implies VSX; `nv` whole vectors.
        unsafe { add_i32_body(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), nv) };
        let (d, bytes) = (nv * 4, out.len() % 4 * 4);
        if bytes > 0 {
            tail2!(
                a[d..].as_ptr(),
                b[d..].as_ptr(),
                out[d..].as_mut_ptr(),
                bytes,
                "vadduwm 0, 0, 1"
            );
        }
    }

    /// `out[i] = a[i] * b[i] + c[i]`, fused everywhere (`xvmaddasp`, the tail too).
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn fmadd_f32(self, a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len() && c.len() == out.len());
        let nv = out.len() / 4;
        // SAFETY: ISA 3.0 implies VSX; `nv` whole vectors.
        unsafe { fmadd_f32_body(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), nv) };
        let (d, bytes) = (nv * 4, out.len() % 4 * 4);
        if bytes > 0 {
            // SAFETY: as in `tail2!`, three inputs.
            unsafe {
                core::arch::asm!(
                    "lxvl 32, {a}, {len}",
                    "lxvl 33, {b}, {len}",
                    "lxvl 34, {c}, {len}",
                    "xvmaddasp 34, 32, 33",
                    "stxvl 34, {o}, {len}",
                    a = in(reg_nonzero) a[d..].as_ptr(),
                    b = in(reg_nonzero) b[d..].as_ptr(),
                    c = in(reg_nonzero) c[d..].as_ptr(),
                    o = in(reg_nonzero) out[d..].as_mut_ptr(),
                    len = in(reg) len_operand(bytes),
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack),
                );
            }
        }
    }

    /// Sum of the `< 32` floats after the whole blocks: full vectors, then one
    /// `lxvl` (whose unused lanes are zero). `B` = with a second input (dot).
    fn tail_reduce<const DOT: bool>(a: &[f32], b: &[f32]) -> f32 {
        debug_assert!(a.len() < 32);
        let (nv, bytes) = (a.len() / 4, a.len() % 4 * 4);
        let mut out = [0f32; 4];
        macro_rules! body {
            ($step:literal) => {
                // SAFETY: the token proves ISA 3.0; `nv` vectors plus `bytes`
                // bytes of each input are in bounds (`b` is `a` for a sum);
                // `stxvw4x` writes the 16 bytes of `out`.
                unsafe {
                    core::arch::asm!(
                        "xxlxor 34, 34, 34",
                        "cmpdi {nv}, 0",
                        "beq 3f",
                        "mtctr {nv}",
                        "2:",
                        "lxvw4x 32, 0, {a}",
                        "lxvw4x 33, 0, {b}",
                        $step,
                        "addi {a}, {a}, 16",
                        "addi {b}, {b}, 16",
                        "bdnz 2b",
                        "3:",
                        "lxvl 32, {a}, {len}",
                        "lxvl 33, {b}, {len}",
                        $step,
                        "xxsldwi 32, 34, 34, 2", "xvaddsp 34, 34, 32",
                        "xxsldwi 32, 34, 34, 1", "xvaddsp 34, 34, 32",
                        "stxvw4x 34, 0, {s}",
                        a = inout(reg_nonzero) a.as_ptr() => _,
                        b = inout(reg_nonzero) b.as_ptr() => _,
                        nv = in(reg) nv,
                        len = in(reg) len_operand(bytes),
                        s = in(reg) out.as_mut_ptr(),
                        out("v0") _, out("v1") _, out("v2") _, out("ctr") _, out("cr0") _,
                        options(nostack),
                    );
                }
            };
        }
        if DOT {
            body!("xvmaddasp 34, 32, 33");
        } else {
            body!("xvaddsp 34, 34, 32");
        }
        out[0]
    }

    /// Sum of all elements: the [`super::vsx`] blocks plus an `lxvl` tail.
    /// NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 32;
        // SAFETY: ISA 3.0 implies VSX; `trips * 32` floats are in bounds.
        let head = if trips > 0 {
            unsafe { sum_body(a.as_ptr(), trips) }
        } else {
            0.0
        };
        let rest = &a[trips * 32..];
        head + Self::tail_reduce::<false>(rest, rest)
    }

    /// Dot product: the [`super::vsx`] blocks plus an `lxvl` tail, fused
    /// throughout. NaN and inf propagate.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let trips = a.len() / 32;
        // SAFETY: ISA 3.0 implies VSX; `trips * 32` floats of each are in bounds.
        let head = if trips > 0 {
            unsafe { dot_body(a.as_ptr(), b.as_ptr(), trips) }
        } else {
            0.0
        };
        head + Self::tail_reduce::<true>(&a[trips * 32..], &b[trips * 32..])
    }

    /// IEEE binary16 bits to `f32` (`xvcvhpsp`), exact; the last `< 8`
    /// elements use the scalar codec.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn cvt_f16_to_f32_slice(self, src: &[u16], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len());
        let trips = src.len() / 8;
        if trips > 0 {
            // SAFETY: the token proves ISA 3.0; `trips` x 8 elements of each.
            unsafe {
                core::arch::asm!(
                    "vspltisw 31, -16", // shift count 16
                    "mtctr {n}",
                    "2:",
                    // halfword i of memory -> halfword element i; word i holds
                    // h(2i) in its high half and h(2i+1) in its low half
                    "lxvh8x 32, 0, {s}",
                    "vsrw 1, 0, 31",
                    "xvcvhpsp 33, 33", // f(h(2i))
                    "xvcvhpsp 32, 32", // f(h(2i+1))
                    "xxmrghw 34, 33, 32",
                    "xxmrglw 35, 33, 32",
                    "stxvw4x 34, 0, {d}",
                    "stxvw4x 35, {d}, {c16}",
                    "addi {s}, {s}, 16",
                    "addi {d}, {d}, 32",
                    "bdnz 2b",
                    s = inout(reg_nonzero) src.as_ptr() => _,
                    d = inout(reg_nonzero) dst.as_mut_ptr() => _,
                    n = in(reg) trips,
                    c16 = in(reg) 16usize,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v31") _,
                    out("ctr") _,
                    options(nostack),
                );
            }
        }
        for i in trips * 8..src.len() {
            dst[i] = MiniFloat::F16.decode(src[i] as u32);
        }
    }

    /// `f32` to IEEE binary16 bits (`xvcvsphp`): round to nearest even, overflow
    /// to infinity ([`Overflow::Special`]); NaN payloads may differ from the
    /// scalar codec. The last `< 8` elements use the scalar codec.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn cvt_f32_to_f16_slice(self, src: &[f32], dst: &mut [u16]) {
        assert_eq!(src.len(), dst.len());
        let trips = src.len() / 8;
        if trips > 0 {
            // SAFETY: the token proves ISA 3.0; `trips` x 8 elements of each.
            unsafe {
                core::arch::asm!(
                    "mtctr {n}",
                    "2:",
                    "lxvw4x 32, 0, {s}",
                    "lxvw4x 33, {s}, {c16}",
                    "xvcvsphp 32, 32",
                    "xvcvsphp 33, 33",
                    // low halfword of each word, in order: h0..h3, h4..h7
                    "vpkuwum 2, 0, 1",
                    "stxvh8x 34, 0, {d}",
                    "addi {s}, {s}, 32",
                    "addi {d}, {d}, 16",
                    "bdnz 2b",
                    s = inout(reg_nonzero) src.as_ptr() => _,
                    d = inout(reg_nonzero) dst.as_mut_ptr() => _,
                    n = in(reg) trips,
                    c16 = in(reg) 16usize,
                    out("v0") _, out("v1") _, out("v2") _, out("ctr") _,
                    options(nostack),
                );
            }
        }
        for i in trips * 8..src.len() {
            dst[i] = MiniFloat::F16.encode(src[i], Overflow::Special) as u16;
        }
    }
}
