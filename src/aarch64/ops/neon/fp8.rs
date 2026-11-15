//! Armv9.5-A `FEAT_FP8` conversions: FP8 (OCP E4M3 / E5M2) <-> `f32`.
//! Token: [`Fp8`].
//!
//! No stable intrinsics exist, so the bodies are `asm!` (`F1CVTL`, `FCVTL`,
//! `FCVTN`) and select the FP8 format through `FPMR`, which every call
//! writes itself. FP8 values are bit patterns (`u8`). Only the OCP formats
//! exist in hardware; the AMD FNUZ variants need the scalar codec in
//! [`crate::formats`].

use core::arch::asm;

use super::super::macros::arm_token;

arm_token! {
    /// Proof that `FEAT_FP8` conversions and the `FPMR` register are available.
    token = Fp8, requires = [F8cvt, Fpmr],
}

/// FP8 flavour for [`Fp8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// OCP E5M2.
    E5m2,
    /// OCP E4M3 (FN).
    E4m3,
}

impl Fp8Format {
    const fn code(self) -> u64 {
        match self {
            Fp8Format::E5m2 => 0,
            Fp8Format::E4m3 => 1,
        }
    }
}

const SAT: u64 = 1 << 15;

#[inline]
unsafe fn dec8(fmt: Fp8Format, src: *const u8, dst: *mut f32) {
    unsafe {
        asm!(
            ".arch_extension fp8",
            "msr S3_3_C4_C4_2, {fpmr}",
            "ldr d0, [{s}]",
            "f1cvtl v1.8h, v0.8b",
            "fcvtl v2.4s, v1.4h",
            "fcvtl2 v3.4s, v1.8h",
            "stp q2, q3, [{d}]",
            fpmr = in(reg) fmt.code(), s = in(reg) src, d = in(reg) dst,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            options(nostack),
        );
    }
}

#[inline]
unsafe fn enc8(fmt: Fp8Format, saturate: bool, src: *const f32, dst: *mut u8) {
    let fpmr = (fmt.code() << 6) | if saturate { SAT } else { 0 };
    unsafe {
        asm!(
            ".arch_extension fp8",
            "msr S3_3_C4_C4_2, {fpmr}",
            "ldp q0, q1, [{s}]",
            "fcvtn v2.8b, v0.4s, v1.4s",
            "str d2, [{d}]",
            fpmr = in(reg) fpmr, s = in(reg) src, d = in(reg) dst,
            out("v0") _, out("v1") _, out("v2") _,
            options(nostack),
        );
    }
}

impl Fp8 {
    /// 8 FP8 values to `f32` (`f1cvtl` then `fcvtl`/`fcvtl2`). Exact.
    #[inline]
    pub fn fp8_to_f32x8(self, fmt: Fp8Format, a: [u8; 8]) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { dec8(fmt, a.as_ptr(), out.as_mut_ptr()) };
        out
    }

    /// 8 `f32` to FP8 (`fcvtn`), round to nearest even. With `saturate`,
    /// out-of-range values clamp to the largest finite value; otherwise
    /// they become infinity (E5M2) or NaN (E4M3).
    #[inline]
    pub fn f32x8_to_fp8(self, fmt: Fp8Format, saturate: bool, a: [f32; 8]) -> [u8; 8] {
        let mut out = [0u8; 8];
        unsafe { enc8(fmt, saturate, a.as_ptr(), out.as_mut_ptr()) };
        out
    }

    /// [`Self::fp8_to_f32x8`] over slices; the tail uses a zero-padded chunk.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn fp8_to_f32_slice(self, fmt: Fp8Format, a: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), a.len());
        let whole = a.len() - a.len() % 8;
        for i in (0..whole).step_by(8) {
            unsafe { dec8(fmt, a[i..].as_ptr(), out[i..].as_mut_ptr()) };
        }
        if whole < a.len() {
            let mut pad = [0u8; 8];
            pad[..a.len() - whole].copy_from_slice(&a[whole..]);
            let r = self.fp8_to_f32x8(fmt, pad);
            out[whole..].copy_from_slice(&r[..a.len() - whole]);
        }
    }

    /// [`Self::f32x8_to_fp8`] over slices; the tail uses a zero-padded chunk.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn f32_to_fp8_slice(self, fmt: Fp8Format, saturate: bool, a: &[f32], out: &mut [u8]) {
        assert_eq!(out.len(), a.len());
        let whole = a.len() - a.len() % 8;
        for i in (0..whole).step_by(8) {
            unsafe { enc8(fmt, saturate, a[i..].as_ptr(), out[i..].as_mut_ptr()) };
        }
        if whole < a.len() {
            let mut pad = [0f32; 8];
            pad[..a.len() - whole].copy_from_slice(&a[whole..]);
            let r = self.f32x8_to_fp8(fmt, saturate, pad);
            out[whole..].copy_from_slice(&r[..a.len() - whole]);
        }
    }
}
