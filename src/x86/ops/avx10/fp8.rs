//! AVX10.2 FP8 converts. Token: [`Avx10Fp8`].
//!
//! FP8 <-> binary16 in 512-bit registers (32 elements per instruction), plus
//! FP8 -> `f32` through `vcvtph2ps`. Both OCP formats exist: `HF8` is E4M3
//! ([`Fp8Format::E4m3`]) and `BF8` is E5M2 ([`Fp8Format::E5m2`]).
//!
//! **Not run on real hardware.** `core::arch` has no AVX10.2 intrinsics, so
//! the bodies are `asm!`, and the machine this was written on has no AVX10.2.
//! It is verified by disassembling the compiled code with GNU `objdump` (an
//! assembler independent of LLVM's) and checking the instructions, not by
//! execution. The tests in `test/ops/avx10/fp8.rs` run on AVX10.2 CPUs and
//! compare against the scalar codec in [`crate::formats`].
//!
//! Notes:
//! - The narrowing converts take binary16 input, so `f32` -> FP8 would round
//!   twice (`f32` -> `f16` -> FP8). Use [`crate::formats::MiniFloat`] when
//!   you need one rounding from `f32`; [`crate::formats`] does not dispatch to
//!   these for that reason.
//! - BF8 -> binary16 has no instruction. BF8 is the top byte of an E5M2
//!   binary16, so the code widens each byte and shifts it left by 8 (exact).
//! - Narrowing rounds to nearest even. With `saturate`, overflow clamps to
//!   the largest finite value (`...S` forms); without, E5M2 gives infinity
//!   and E4M3 gives NaN.

use core::arch::asm;

use super::super::super::tiers::Avx10;
use super::super::super::{Feature, FeatureSet};

/// FP8 flavour for [`Avx10Fp8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// OCP E5M2 (`BF8`).
    E5m2,
    /// OCP E4M3 FN (`HF8`).
    E4m3,
}

/// Proof that AVX10 version 2 or later (with 512-bit vectors) is present.
/// Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Avx10Fp8(());

impl Avx10Fp8 {
    /// `None` if the CPU lacks AVX10.2.
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::x86::detect_features())
    }

    /// From a raw feature bitset (reads CPUID leaf `0x24` for the AVX10 version).
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        let ok = set.contains(Feature::Avx10)
            && set.contains(Feature::Avx512f)
            && Avx10::version(set).is_some_and(|v| v >= 2);
        ok.then_some(Avx10Fp8(()))
    }
}

macro_rules! enc {
    ($name:ident, $mnemonic:literal) => {
        /// # Safety
        /// AVX10.2 available; `s` points to 32 `u16`, `d` to 32 bytes.
        #[target_feature(enable = "avx512f")]
        unsafe fn $name(s: *const u16, d: *mut u8) {
            unsafe {
                asm!(
                    "vmovdqu64 zmm0, zmmword ptr [{s}]",
                    concat!($mnemonic, " ymm1, zmm0"),
                    "vmovdqu ymmword ptr [{d}], ymm1",
                    s = in(reg) s, d = in(reg) d,
                    out("zmm0") _, out("zmm1") _,
                    options(nostack),
                );
            }
        }
    };
}

enc!(enc_hf8, "vcvtph2hf8");
enc!(enc_hf8s, "vcvtph2hf8s");
enc!(enc_bf8, "vcvtph2bf8");
enc!(enc_bf8s, "vcvtph2bf8s");

/// # Safety
/// AVX10.2 available; `s` points to 32 bytes, `d` to 32 `u16`.
#[target_feature(enable = "avx512f")]
unsafe fn dec_hf8(s: *const u8, d: *mut u16) {
    unsafe {
        asm!(
            "vcvthf82ph zmm0, ymmword ptr [{s}]",
            "vmovdqu64 zmmword ptr [{d}], zmm0",
            s = in(reg) s, d = in(reg) d,
            out("zmm0") _,
            options(nostack),
        );
    }
}

/// # Safety
/// AVX512F available; `s` points to 32 bytes, `d` to 32 `u16`.
#[target_feature(enable = "avx512f")]
unsafe fn dec_bf8(s: *const u8, d: *mut u16) {
    unsafe {
        asm!(
            "vpmovzxbw zmm0, ymmword ptr [{s}]",
            "vpsllw zmm0, zmm0, 8",
            "vmovdqu64 zmmword ptr [{d}], zmm0",
            s = in(reg) s, d = in(reg) d,
            out("zmm0") _,
            options(nostack),
        );
    }
}

/// # Safety
/// AVX10.2 available; `s` points to 32 bytes, `d` to 32 `f32`.
#[target_feature(enable = "avx512f")]
unsafe fn dec_hf8_f32(s: *const u8, d: *mut f32) {
    unsafe {
        asm!(
            "vcvthf82ph zmm0, ymmword ptr [{s}]",
            "vcvtph2ps zmm1, ymm0",
            "vextracti64x4 ymm2, zmm0, 1",
            "vcvtph2ps zmm2, ymm2",
            "vmovups zmmword ptr [{d}], zmm1",
            "vmovups zmmword ptr [{d} + 64], zmm2",
            s = in(reg) s, d = in(reg) d,
            out("zmm0") _, out("zmm1") _, out("zmm2") _,
            options(nostack),
        );
    }
}

/// # Safety
/// AVX512F available; `s` points to 32 bytes, `d` to 32 `f32`.
#[target_feature(enable = "avx512f")]
unsafe fn dec_bf8_f32(s: *const u8, d: *mut f32) {
    unsafe {
        asm!(
            "vpmovzxbw zmm0, ymmword ptr [{s}]",
            "vpsllw zmm0, zmm0, 8",
            "vcvtph2ps zmm1, ymm0",
            "vextracti64x4 ymm2, zmm0, 1",
            "vcvtph2ps zmm2, ymm2",
            "vmovups zmmword ptr [{d}], zmm1",
            "vmovups zmmword ptr [{d} + 64], zmm2",
            s = in(reg) s, d = in(reg) d,
            out("zmm0") _, out("zmm1") _, out("zmm2") _,
            options(nostack),
        );
    }
}

impl Avx10Fp8 {
    /// 32 FP8 values to binary16 bits. Exact.
    #[inline]
    pub fn fp8_to_f16x32(self, fmt: Fp8Format, a: [u8; 32]) -> [u16; 32] {
        let mut out = [0u16; 32];
        unsafe {
            match fmt {
                Fp8Format::E4m3 => dec_hf8(a.as_ptr(), out.as_mut_ptr()),
                Fp8Format::E5m2 => dec_bf8(a.as_ptr(), out.as_mut_ptr()),
            }
        }
        out
    }

    /// 32 FP8 values to `f32`. Exact.
    #[inline]
    pub fn fp8_to_f32x32(self, fmt: Fp8Format, a: [u8; 32]) -> [f32; 32] {
        let mut out = [0f32; 32];
        unsafe {
            match fmt {
                Fp8Format::E4m3 => dec_hf8_f32(a.as_ptr(), out.as_mut_ptr()),
                Fp8Format::E5m2 => dec_bf8_f32(a.as_ptr(), out.as_mut_ptr()),
            }
        }
        out
    }

    /// 32 binary16 values to FP8 (`vcvtph2hf8` / `vcvtph2bf8`, with the
    /// `s` suffix when `saturate`), round to nearest even.
    #[inline]
    pub fn f16x32_to_fp8(self, fmt: Fp8Format, saturate: bool, a: [u16; 32]) -> [u8; 32] {
        let mut out = [0u8; 32];
        let (s, d) = (a.as_ptr(), out.as_mut_ptr());
        unsafe {
            match (fmt, saturate) {
                (Fp8Format::E4m3, false) => enc_hf8(s, d),
                (Fp8Format::E4m3, true) => enc_hf8s(s, d),
                (Fp8Format::E5m2, false) => enc_bf8(s, d),
                (Fp8Format::E5m2, true) => enc_bf8s(s, d),
            }
        }
        out
    }

    /// [`Self::fp8_to_f16x32`] over slices; the tail uses a zero-padded chunk.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn fp8_to_f16_slice(self, fmt: Fp8Format, a: &[u8], out: &mut [u16]) {
        assert_eq!(out.len(), a.len());
        let whole = a.len() - a.len() % 32;
        for i in (0..whole).step_by(32) {
            out[i..i + 32]
                .copy_from_slice(&self.fp8_to_f16x32(fmt, a[i..i + 32].try_into().unwrap()));
        }
        if whole < a.len() {
            let mut pad = [0u8; 32];
            pad[..a.len() - whole].copy_from_slice(&a[whole..]);
            out[whole..].copy_from_slice(&self.fp8_to_f16x32(fmt, pad)[..a.len() - whole]);
        }
    }

    /// [`Self::fp8_to_f32x32`] over slices; the tail uses a zero-padded chunk.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn fp8_to_f32_slice(self, fmt: Fp8Format, a: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), a.len());
        let whole = a.len() - a.len() % 32;
        for i in (0..whole).step_by(32) {
            out[i..i + 32]
                .copy_from_slice(&self.fp8_to_f32x32(fmt, a[i..i + 32].try_into().unwrap()));
        }
        if whole < a.len() {
            let mut pad = [0u8; 32];
            pad[..a.len() - whole].copy_from_slice(&a[whole..]);
            out[whole..].copy_from_slice(&self.fp8_to_f32x32(fmt, pad)[..a.len() - whole]);
        }
    }

    /// [`Self::f16x32_to_fp8`] over slices; the tail uses a zero-padded chunk.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn f16_to_fp8_slice(self, fmt: Fp8Format, saturate: bool, a: &[u16], out: &mut [u8]) {
        assert_eq!(out.len(), a.len());
        let whole = a.len() - a.len() % 32;
        for i in (0..whole).step_by(32) {
            out[i..i + 32].copy_from_slice(&self.f16x32_to_fp8(
                fmt,
                saturate,
                a[i..i + 32].try_into().unwrap(),
            ));
        }
        if whole < a.len() {
            let mut pad = [0u16; 32];
            pad[..a.len() - whole].copy_from_slice(&a[whole..]);
            out[whole..]
                .copy_from_slice(&self.f16x32_to_fp8(fmt, saturate, pad)[..a.len() - whole]);
        }
    }
}

#[cfg(test)]
#[path = "../../test/ops/avx10/fp8.rs"]
mod tests;
