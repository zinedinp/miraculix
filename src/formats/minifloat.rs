//! Small floating-point formats as bit-exact scalar codecs.
//!
//! One generic codec covers every sign/exponent/mantissa format used by GPU
//! and accelerator matrix engines: FP8 (OCP E4M3/E5M2 and AMD FNUZ), the MX
//! element types FP6 (E2M3, E3M2) and FP4 (E2M1), plus IEEE FP16 and BF16.
//! Encoding rounds to nearest even.

/// A sign + exponent + mantissa format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MiniFloat {
    /// FP8 E4M3, OCP "FN": bias 7, no infinity, one NaN per sign (`S.1111.111`), max 448.
    E4M3Fn,
    /// FP8 E4M3 "FNUZ" (AMD MI300): bias 8, no infinity, no negative zero, NaN is `0x80`, max 240.
    E4M3Fnuz,
    /// FP8 E5M2, IEEE-like: bias 15, infinity and NaN, max 57344.
    E5M2,
    /// FP8 E5M2 "FNUZ": bias 16, no infinity, no negative zero, NaN is `0x80`, max 57344.
    E5M2Fnuz,
    /// MXFP4 / NVFP4 element E2M1: 4 bits, bias 1, no infinity or NaN, max 6.
    E2M1,
    /// MXFP6 element E2M3: 6 bits, bias 1, no infinity or NaN, max 7.5.
    E2M3,
    /// MXFP6 element E3M2: 6 bits, bias 3, no infinity or NaN, max 28.
    E3M2,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    Bf16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Top exponent is inf/NaN.
    Ieee,
    /// Only the all-ones pattern is NaN.
    Fn,
    /// Only `0x80` is NaN; no negative zero.
    Fnuz,
    /// Every pattern is finite.
    Finite,
}

/// How [`MiniFloat::encode`] treats values above the largest finite value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    /// Clamp to the largest finite value (what MX quantizers do).
    Saturate,
    /// Produce infinity, or NaN where the format has no infinity.
    /// Formats with neither (E2M1, E2M3, E3M2) always saturate.
    Special,
}

pub(super) const fn pow2(k: i32) -> f32 {
    if k >= -126 {
        f32::from_bits(((k + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (k + 149))
    }
}

const fn rne(m: u32, shift: u32) -> u32 {
    if shift >= 32 {
        return 0;
    }
    let q = m >> shift;
    let rem = m & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rem > half || (rem == half && (q & 1) == 1) {
        q + 1
    } else {
        q
    }
}

impl MiniFloat {
    const fn params(self) -> (u32, u32, i32, Kind) {
        match self {
            MiniFloat::E4M3Fn => (4, 3, 7, Kind::Fn),
            MiniFloat::E4M3Fnuz => (4, 3, 8, Kind::Fnuz),
            MiniFloat::E5M2 => (5, 2, 15, Kind::Ieee),
            MiniFloat::E5M2Fnuz => (5, 2, 16, Kind::Fnuz),
            MiniFloat::E2M1 => (2, 1, 1, Kind::Finite),
            MiniFloat::E2M3 => (2, 3, 1, Kind::Finite),
            MiniFloat::E3M2 => (3, 2, 3, Kind::Finite),
            MiniFloat::F16 => (5, 10, 15, Kind::Ieee),
            MiniFloat::Bf16 => (8, 7, 127, Kind::Ieee),
        }
    }

    /// Storage width in bits (4, 6, 8 or 16).
    pub const fn bits(self) -> u32 {
        let (e, m, _, _) = self.params();
        1 + e + m
    }

    /// Bit pattern of the largest finite positive value.
    pub const fn max_code(self) -> u32 {
        let (e, m, _, kind) = self.params();
        match kind {
            Kind::Ieee => (((1u32 << e) - 2) << m) | ((1u32 << m) - 1),
            Kind::Fn => (1u32 << (e + m)) - 2,
            Kind::Fnuz | Kind::Finite => (1u32 << (e + m)) - 1,
        }
    }

    /// Largest finite positive value.
    pub const fn max_finite(self) -> f32 {
        self.decode(self.max_code())
    }

    /// Unbiased exponent of the largest finite value (the `emax` of the OCP MX spec).
    pub const fn emax(self) -> i32 {
        let (e, m, bias, _) = self.params();
        let top = (self.max_code() >> m) & ((1u32 << e) - 1);
        top as i32 - bias
    }

    /// Decodes a bit pattern (low [`bits`](Self::bits) bits of `code`) to `f32`.
    /// Every format here is exactly representable in `f32`.
    pub const fn decode(self, code: u32) -> f32 {
        let (e, m, bias, kind) = self.params();
        let sign = (code >> (e + m)) & 1;
        let exp = (code >> m) & ((1u32 << e) - 1);
        let man = code & ((1u32 << m) - 1);
        let all_ones = (1u32 << e) - 1;
        let neg = sign == 1;
        match kind {
            Kind::Ieee if exp == all_ones => {
                return if man == 0 {
                    if neg { f32::NEG_INFINITY } else { f32::INFINITY }
                } else {
                    f32::NAN
                };
            }
            Kind::Fn if exp == all_ones && man == (1u32 << m) - 1 => return f32::NAN,
            Kind::Fnuz if code & ((1u32 << (e + m + 1)) - 1) == 1u32 << (e + m) => {
                return f32::NAN;
            }
            _ => {}
        }
        let mag = if exp == 0 {
            man as f32 * pow2(1 - bias - m as i32)
        } else {
            ((1u32 << m) | man) as f32 * pow2(exp as i32 - bias - m as i32)
        };
        if neg { -mag } else { mag }
    }

    /// Encodes `x`, rounding to nearest even. NaN maps to the format's NaN
    /// (or to `0` for formats without one).
    pub const fn encode(self, x: f32, overflow: Overflow) -> u32 {
        let (e, m, bias, kind) = self.params();
        let bits = x.to_bits();
        let sign = bits >> 31;
        let sign_bit = sign << (e + m);
        let abs = bits & 0x7FFF_FFFF;
        let nan = match kind {
            Kind::Ieee => (((1u32 << e) - 1) << m) | (1u32 << (m - 1)),
            Kind::Fn => (1u32 << (e + m)) - 1,
            Kind::Fnuz => 1u32 << (e + m),
            Kind::Finite => 0,
        };
        if abs > 0x7F80_0000 {
            return match kind {
                Kind::Fnuz => nan,
                Kind::Finite => 0,
                _ => sign_bit | nan,
            };
        }
        let max = self.max_code();
        let emin = 1 - bias;
        let code = if abs >= 0x7F80_0000 {
            max + 1
        } else {
            let fexp = (abs >> 23) as i32;
            let (ee, mant24) = if fexp == 0 {
                (-126, abs & 0x7F_FFFF)
            } else {
                (fexp - 127, (abs & 0x7F_FFFF) | 0x80_0000)
            };
            if ee >= emin && mant24 >= 0x80_0000 {
                let r = rne(mant24, 23 - m);
                (((ee + bias) as u32) << m) + (r - (1u32 << m))
            } else {
                rne(mant24, 23 - m + (emin - ee) as u32)
            }
        };
        let code = if code > max {
            match (overflow, kind) {
                (Overflow::Saturate, _) | (_, Kind::Finite) => max,
                (Overflow::Special, Kind::Ieee) => ((1u32 << e) - 1) << m,
                (Overflow::Special, Kind::Fn) => return sign_bit | nan,
                (Overflow::Special, Kind::Fnuz) => return nan,
            }
        } else {
            code
        };
        if matches!(kind, Kind::Fnuz) && code == 0 {
            0
        } else {
            sign_bit | code
        }
    }
}

const fn build_lut(f: MiniFloat) -> [f32; 256] {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = f.decode(i as u32);
        i += 1;
    }
    t
}

static E4M3FN: [f32; 256] = build_lut(MiniFloat::E4M3Fn);
static E4M3FNUZ: [f32; 256] = build_lut(MiniFloat::E4M3Fnuz);
static E5M2: [f32; 256] = build_lut(MiniFloat::E5M2);
static E5M2FNUZ: [f32; 256] = build_lut(MiniFloat::E5M2Fnuz);

impl MiniFloat {
    /// 256-entry decode table for the 8-bit formats; `None` for other widths.
    /// A table lookup is the CPU equivalent of a hardware FP8 convert
    /// (`pshufb`/`vpermi2b` planes, Neon `tbl`, `vgather`).
    pub fn lut_u8(self) -> Option<&'static [f32; 256]> {
        match self {
            MiniFloat::E4M3Fn => Some(&E4M3FN),
            MiniFloat::E4M3Fnuz => Some(&E4M3FNUZ),
            MiniFloat::E5M2 => Some(&E5M2),
            MiniFloat::E5M2Fnuz => Some(&E5M2FNUZ),
            _ => None,
        }
    }

    /// Hardware FP8 (Arm `FEAT_FP8`) for the OCP formats, when present.
    #[cfg(target_arch = "aarch64")]
    fn arm_fp8(
        self,
    ) -> Option<(crate::aarch64::ops::neon::fp8::Fp8, crate::aarch64::ops::neon::fp8::Fp8Format)> {
        use crate::aarch64::ops::neon::fp8::{Fp8, Fp8Format};
        let hw = match self {
            MiniFloat::E4M3Fn => Fp8Format::E4m3,
            MiniFloat::E5M2 => Fp8Format::E5m2,
            _ => return None,
        };
        Fp8::detect().map(|t| (t, hw))
    }

    /// Decodes `src` (one code per byte, `bits() <= 8`) into `dst`. The OCP FP8
    /// formats use Arm `FEAT_FP8` converts when the CPU has them.
    ///
    /// # Panics
    /// If `bits() > 8` or the slices differ in length.
    pub fn decode_u8_slice(self, src: &[u8], dst: &mut [f32]) {
        assert!(self.bits() <= 8, "use decode_u16_slice for 16-bit formats");
        assert_eq!(src.len(), dst.len());
        #[cfg(target_arch = "aarch64")]
        if let Some((t, hw)) = self.arm_fp8() {
            t.fp8_to_f32_slice(hw, src, dst);
            return;
        }
        match self.lut_u8() {
            Some(t) => dst.iter_mut().zip(src).for_each(|(d, &s)| *d = t[s as usize]),
            None => dst.iter_mut().zip(src).for_each(|(d, &s)| *d = self.decode(s as u32)),
        }
    }

    /// Encodes `src` into one code per byte (`bits() <= 8`).
    ///
    /// # Panics
    /// If `bits() > 8` or the slices differ in length.
    pub fn encode_u8_slice(self, src: &[f32], dst: &mut [u8], overflow: Overflow) {
        assert!(self.bits() <= 8, "use encode_u16_slice for 16-bit formats");
        assert_eq!(src.len(), dst.len());
        #[cfg(target_arch = "aarch64")]
        if let Some((t, hw)) = self.arm_fp8() {
            t.f32_to_fp8_slice(hw, overflow == Overflow::Saturate, src, dst);
            return;
        }
        dst.iter_mut().zip(src).for_each(|(d, &s)| *d = self.encode(s, overflow) as u8);
    }

    /// Decodes 16-bit codes ([`F16`](Self::F16) / [`Bf16`](Self::Bf16)).
    ///
    /// FP16 uses F16C on x86, `fcvtl` on AArch64 and `Zvfh` on RISC-V when
    /// present; BF16 is a shift (`Zvfbfmin` on RISC-V). Both fall back to the
    /// scalar codec.
    ///
    /// # Panics
    /// If `bits() != 16` or the slices differ in length.
    pub fn decode_u16_slice(self, src: &[u16], dst: &mut [f32]) {
        assert_eq!(self.bits(), 16);
        assert_eq!(src.len(), dst.len());
        if self == MiniFloat::Bf16 {
            #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
            if let Some(t) = crate::riscv::ops::zvfbf::Zvfbfmin::detect() {
                t.cvt_bf16_to_f32_slice(src, dst);
                return;
            }
            dst.iter_mut().zip(src).for_each(|(d, &s)| *d = f32::from_bits((s as u32) << 16));
            return;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(t) = crate::x86::ops::avx::f16c::F16c::detect() {
            t.f16_to_f32_slice(src, dst);
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = crate::aarch64::ops::neon::neon::Neon::detect() {
            t.cvt_f16_to_f32_slice(src, dst);
            return;
        }
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        if let Some(t) = crate::riscv::ops::zvfh::Zvfh::detect() {
            t.cvt_f16_to_f32_slice(src, dst);
            return;
        }
        dst.iter_mut().zip(src).for_each(|(d, &s)| *d = self.decode(s as u32));
    }

    /// Encodes to 16-bit codes ([`F16`](Self::F16) / [`Bf16`](Self::Bf16)).
    ///
    /// FP16 with [`Overflow::Special`] uses F16C / `fcvtn` when present;
    /// NaN payloads may then differ from the scalar codec. BF16 uses an
    /// integer round-to-nearest-even that the compiler vectorizes (the
    /// AVX512-BF16 convert is not used: it flushes subnormals).
    ///
    /// # Panics
    /// If `bits() != 16` or the slices differ in length.
    pub fn encode_u16_slice(self, src: &[f32], dst: &mut [u16], overflow: Overflow) {
        assert_eq!(self.bits(), 16);
        assert_eq!(src.len(), dst.len());
        if self == MiniFloat::Bf16 {
            #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
            if overflow == Overflow::Special {
                if let Some(t) = crate::riscv::ops::zvfbf::Zvfbfmin::detect() {
                    t.cvt_f32_to_bf16_slice(src, dst);
                    return;
                }
            }
            dst.iter_mut().zip(src).for_each(|(d, &s)| *d = bf16_fast(s, overflow));
            return;
        }
        if overflow == Overflow::Special {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(t) = crate::x86::ops::avx::f16c::F16c::detect() {
                t.f32_to_f16_slice::<{ core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT }>(src, dst);
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(t) = crate::aarch64::ops::neon::neon::Neon::detect() {
                t.cvt_f32_to_f16_slice(src, dst);
                return;
            }
            #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
            if let Some(t) = crate::riscv::ops::zvfh::Zvfh::detect() {
                t.cvt_f32_to_f16_slice(src, dst);
                return;
            }
        }
        dst.iter_mut().zip(src).for_each(|(d, &s)| *d = self.encode(s, overflow) as u16);
    }
}

fn bf16_fast(x: f32, overflow: Overflow) -> u16 {
    let b = x.to_bits();
    if (b & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((b >> 16) | 0x40) as u16; // quiet NaN, payload kept
    }
    let r = (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16;
    if overflow == Overflow::Saturate && (r & 0x7FFF) == 0x7F80 {
        return (r & 0x8000) | 0x7F7F;
    }
    r
}
