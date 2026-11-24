//! Power ISA 3.1 (POWER10 / POWER11) kernels beyond MMA.
//!
//! - **`lxvp`** loads 32 bytes into an even/odd VSR pair: the 32-float
//!   reduction blocks need four loads per input instead of eight. In
//!   little-endian mode the pair comes in byte-reversed (registers and lanes
//!   swapped, every value intact); `a` and `b` get the same mapping, so sums
//!   and dot products do not notice.
//! - **`vextractwm`** moves the sign bits of the four word lanes into a GPR
//!   (an x86 `movmskps`). `compress_gt_f32` and the position search of
//!   `argmax_f32` compare four lanes at a time and only look at lanes whose
//!   bit is set. Lane `i` (memory order, as `lxvw4x` loads it) is bit `3 - i`.
//! - **`xvcvspbf16`** rounds `f32` to bfloat16 (nearest even, overflow to
//!   infinity, subnormals kept); it backs `MiniFloat::Bf16` encoding with
//!   `Overflow::Special`.

use super::isa300::Isa300;
use super::vsx::{Splat, Vsx};
use crate::formats::{Overflow, bf16_fast};
use crate::powerpc::{Feature, FeatureSet};

/// Proof of Power ISA 3.1 with VSX on (POWER10+).
#[derive(Debug, Clone, Copy)]
pub struct Isa31(());

/// Vectors per mask batch (one mask byte each).
const BATCH: usize = 64;

/// Writes one 4-bit lane mask per vector: `cmp(a[4v..4v+4], k)`.
///
/// # Safety
/// ISA 3.1 on; `a` readable for `nv * 16` bytes, `out` writable for `nv`,
/// `0 < nv`.
unsafe fn masks<const EQ: bool>(a: *const f32, k: f32, out: *mut u8, nv: usize) {
    let splat = Splat::f32(k);
    macro_rules! body {
        ($cmp:literal) => {
            // SAFETY: caller's contract; `splat` is 16 readable bytes.
            unsafe {
                core::arch::asm!(
                    "lxvw4x 33, 0, {k}",
                    "mtctr {n}",
                    "2:",
                    "lxvw4x 32, 0, {a}",
                    $cmp,
                    "vextractwm {t}, 0",
                    "stb {t}, 0({o})",
                    "addi {a}, {a}, 16",
                    "addi {o}, {o}, 1",
                    "bdnz 2b",
                    a = inout(reg_nonzero) a => _,
                    o = inout(reg_nonzero) out => _,
                    k = in(reg) &splat as *const Splat,
                    n = in(reg) nv,
                    t = out(reg) _,
                    out("v0") _, out("v1") _, out("ctr") _,
                    options(nostack),
                );
            }
        };
    }
    if EQ {
        body!("xvcmpeqsp 32, 32, 33");
    } else {
        body!("xvcmpgtsp 32, 32, 33");
    }
}

impl Isa31 {
    /// Token if the cached feature set has ISA 3.1 with VSX.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has ISA 3.1 with VSX (ISA 3.0 included, as on every 3.1 core).
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains_all(&[
            Feature::Altivec,
            Feature::Vsx,
            Feature::Power9Vector,
            Feature::Isa31,
        ])
        .then_some(Isa31(()))
    }

    /// Sum or dot product of `trips * 32` floats, eight accumulators, `lxvp` loads.
    fn blocks<const DOT: bool>(a: &[f32], b: &[f32], trips: usize) -> f32 {
        let mut out = [0f32; 4];
        macro_rules! body {
            ($($step:literal),*) => {
                // SAFETY: the token proves ISA 3.1; `trips * 128` bytes of `a`
                // (and `b` for a dot) are in bounds; `stxvw4x` writes `out`.
                unsafe {
                    core::arch::asm!(
                        "xxlxor 48, 48, 48", "xxlxor 49, 49, 49", "xxlxor 50, 50, 50", "xxlxor 51, 51, 51",
                        "xxlxor 52, 52, 52", "xxlxor 53, 53, 53", "xxlxor 54, 54, 54", "xxlxor 55, 55, 55",
                        "mtctr {n}",
                        "2:",
                        "lxvp 32, 0({a})", "lxvp 34, 32({a})", "lxvp 36, 64({a})", "lxvp 38, 96({a})",
                        "lxvp 40, 0({b})", "lxvp 42, 32({b})", "lxvp 44, 64({b})", "lxvp 46, 96({b})",
                        $($step,)*
                        "addi {a}, {a}, 128",
                        "addi {b}, {b}, 128",
                        "bdnz 2b",
                        "xvaddsp 48, 48, 49", "xvaddsp 50, 50, 51", "xvaddsp 52, 52, 53", "xvaddsp 54, 54, 55",
                        "xvaddsp 48, 48, 50", "xvaddsp 52, 52, 54", "xvaddsp 48, 48, 52",
                        "xxsldwi 32, 48, 48, 2", "xvaddsp 48, 48, 32",
                        "xxsldwi 32, 48, 48, 1", "xvaddsp 48, 48, 32",
                        "stxvw4x 48, 0, {s}",
                        a = inout(reg_nonzero) a.as_ptr() => _,
                        b = inout(reg_nonzero) b.as_ptr() => _,
                        s = in(reg) out.as_mut_ptr(),
                        n = in(reg) trips,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                        out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                        out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                        out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                        out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("ctr") _,
                        options(nostack),
                    );
                }
            };
        }
        if DOT {
            body!(
                "xvmaddasp 48, 32, 40",
                "xvmaddasp 49, 33, 41",
                "xvmaddasp 50, 34, 42",
                "xvmaddasp 51, 35, 43",
                "xvmaddasp 52, 36, 44",
                "xvmaddasp 53, 37, 45",
                "xvmaddasp 54, 38, 46",
                "xvmaddasp 55, 39, 47"
            );
        } else {
            body!(
                "xvaddsp 48, 48, 32",
                "xvaddsp 49, 49, 33",
                "xvaddsp 50, 50, 34",
                "xvaddsp 51, 51, 35",
                "xvaddsp 52, 52, 36",
                "xvaddsp 53, 53, 37",
                "xvaddsp 54, 54, 38",
                "xvaddsp 55, 55, 39"
            );
        }
        out[0]
    }

    /// Sum of all elements: `lxvp` blocks of 32, then the ISA 3.0 `lxvl` tail.
    /// NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 32;
        let head = if trips > 0 {
            Self::blocks::<false>(a, a, trips)
        } else {
            0.0
        };
        let rest = &a[trips * 32..];
        head + Isa300::tail_reduce::<false>(rest, rest)
    }

    /// Dot product: `lxvp` blocks of 32 with fused accumulators, then the
    /// ISA 3.0 `lxvl` tail. NaN and inf propagate.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn dot_f32(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let trips = a.len() / 32;
        let head = if trips > 0 {
            Self::blocks::<true>(a, b, trips)
        } else {
            0.0
        };
        head + Isa300::tail_reduce::<true>(&a[trips * 32..], &b[trips * 32..])
    }

    /// Indices of all `a[i] > threshold`, ascending; returns the count. NaN
    /// never matches. Four lanes per `xvcmpgtsp` + `vextractwm`.
    ///
    /// # Panics
    /// If `out.len() < a.len()`.
    pub fn compress_gt_f32(self, a: &[f32], threshold: f32, out: &mut [u32]) -> usize {
        assert!(out.len() >= a.len(), "out must hold a.len() indices");
        let nv = a.len() / 4;
        let mut buf = [0u8; BATCH];
        let mut n = 0;
        let mut v0 = 0;
        while v0 < nv {
            let k = (nv - v0).min(BATCH);
            // SAFETY: the token proves ISA 3.1; vectors `v0..v0 + k` are in
            // bounds; `buf` holds `k <= BATCH` bytes.
            unsafe { masks::<false>(a[v0 * 4..].as_ptr(), threshold, buf.as_mut_ptr(), k) };
            for (v, &m) in buf[..k].iter().enumerate() {
                if m == 0 {
                    continue;
                }
                for j in 0..4 {
                    if m & (8 >> j) != 0 {
                        out[n] = ((v0 + v) * 4 + j) as u32;
                        n += 1;
                    }
                }
            }
            v0 += k;
        }
        for (i, &x) in a.iter().enumerate().skip(nv * 4) {
            if x > threshold {
                out[n] = i as u32;
                n += 1;
            }
        }
        n
    }

    /// First index of the largest element and its value, NaN ignored; `None`
    /// if the slice is empty or all NaN. VSX max, then a mask search for the
    /// first lane equal to it.
    pub fn argmax_f32(self, a: &[f32]) -> Option<(usize, f32)> {
        let m = Vsx::implied().reduce_max_f32(a);
        let nv = a.len() / 4;
        let mut buf = [0u8; BATCH];
        let mut v0 = 0;
        while v0 < nv {
            let k = (nv - v0).min(BATCH);
            // SAFETY: as in `compress_gt_f32`.
            unsafe { masks::<true>(a[v0 * 4..].as_ptr(), m, buf.as_mut_ptr(), k) };
            if let Some((v, &mm)) = buf[..k].iter().enumerate().find(|&(_, &mm)| mm != 0) {
                let j = (0..4).find(|&j| mm & (8 >> j) != 0).unwrap_or(0);
                return Some(((v0 + v) * 4 + j, m));
            }
            v0 += k;
        }
        (nv * 4..a.len()).find(|&i| a[i] == m).map(|i| (i, m))
    }

    /// `f32` to bfloat16 bits (`xvcvspbf16`): round to nearest even, overflow to
    /// infinity ([`Overflow::Special`]), subnormals kept. NaN results are NaN;
    /// their payload bits may differ from the scalar codec. The last `< 8`
    /// elements use the scalar codec.
    ///
    /// # Panics
    /// If the lengths differ.
    pub fn cvt_f32_to_bf16_slice(self, src: &[f32], dst: &mut [u16]) {
        assert_eq!(src.len(), dst.len());
        let trips = src.len() / 8;
        if trips > 0 {
            // SAFETY: the token proves ISA 3.1; `trips` x 8 elements of each.
            unsafe {
                core::arch::asm!(
                    "mtctr {n}",
                    "2:",
                    "lxvw4x 32, 0, {s}",
                    "lxvw4x 33, {s}, {c16}",
                    "xvcvspbf16 32, 32",
                    "xvcvspbf16 33, 33",
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
            dst[i] = bf16_fast(src[i], Overflow::Special);
        }
    }
}
