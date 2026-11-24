//! Instruction words for what no assembler accepts: the paired-single
//! instructions of Gekko / Broadway / Espresso and the VMX128 instructions of
//! Xenon. LLVM, GCC's `as` and rustc know neither, so the kernels emit them as
//! `.long {}` with `const` operands built here.
//!
//! Pure `const fn`s with no `asm!`: the file also builds on the host, where the
//! tests below check every encoder against independent references:
//!
//! - paired singles: the expected words of binutils'
//!   `gas/testsuite/gas/ppc/ppc750ps.d` (`as -m750cl`);
//! - VMX128: a decoder transcribed from Xenia's generated opcode table
//!   (`src/xenia/cpu/ppc/ppc_opcode_lookup_gen.cc`) and its field layouts
//!   (`ppc_decode_data.h`, `FormatVX128*`), while the encoders follow the bit
//!   pictures in Xenia's `docs/ppc/vmx128.txt`. The two sources were written
//!   separately, so agreement is a real check.
//!
//! Register arguments are plain numbers: `f0`-`f31`, `r0`-`r31`, `v0`-`v127`.

#![allow(dead_code)] // each target uses a different subset

// ------------------------------------------------------------ paired singles

/// Quantized loads/stores, D-form: `op frD, d(rA), W, I`.
const fn psq_d(op: u32, fr: u32, ra: u32, d: i16, w: bool, gqr: u32) -> u32 {
    op << 26 | fr << 21 | ra << 16 | (w as u32) << 15 | (gqr & 7) << 12 | (d as u16 as u32 & 0xfff)
}

/// `psq_l frD, d(rA), W, I`: load one (`w`) or two values through GQR `I`.
pub(crate) const fn psq_l(fr: u32, ra: u32, d: i16, w: bool, gqr: u32) -> u32 {
    psq_d(56, fr, ra, d, w, gqr)
}

/// `psq_lu`: as [`psq_l`], then `rA += d`.
pub(crate) const fn psq_lu(fr: u32, ra: u32, d: i16, w: bool, gqr: u32) -> u32 {
    psq_d(57, fr, ra, d, w, gqr)
}

/// `psq_st frS, d(rA), W, I`.
pub(crate) const fn psq_st(fr: u32, ra: u32, d: i16, w: bool, gqr: u32) -> u32 {
    psq_d(60, fr, ra, d, w, gqr)
}

/// `psq_stu`: as [`psq_st`], then `rA += d`.
pub(crate) const fn psq_stu(fr: u32, ra: u32, d: i16, w: bool, gqr: u32) -> u32 {
    psq_d(61, fr, ra, d, w, gqr)
}

/// Indexed forms: `op frD, rA, rB, W, I` (primary 4, 6-bit xo).
const fn psq_x(xo: u32, fr: u32, ra: u32, rb: u32, w: bool, gqr: u32) -> u32 {
    4 << 26 | fr << 21 | ra << 16 | rb << 11 | (w as u32) << 10 | (gqr & 7) << 7 | xo << 1
}

/// `psq_lx frD, rA, rB, W, I`.
pub(crate) const fn psq_lx(fr: u32, ra: u32, rb: u32, w: bool, gqr: u32) -> u32 {
    psq_x(6, fr, ra, rb, w, gqr)
}

/// `psq_stx frS, rA, rB, W, I`.
pub(crate) const fn psq_stx(fr: u32, ra: u32, rb: u32, w: bool, gqr: u32) -> u32 {
    psq_x(7, fr, ra, rb, w, gqr)
}

/// A-form arithmetic: `frD, frA, frC, frB` (primary 4, 5-bit xo).
const fn ps_a(xo: u32, d: u32, a: u32, c: u32, b: u32) -> u32 {
    4 << 26 | d << 21 | a << 16 | b << 11 | c << 6 | xo << 1
}

/// `ps_add frD, frA, frB`: both halves `a + b`.
pub(crate) const fn ps_add(d: u32, a: u32, b: u32) -> u32 {
    ps_a(21, d, a, 0, b)
}

/// `ps_sub frD, frA, frB`.
pub(crate) const fn ps_sub(d: u32, a: u32, b: u32) -> u32 {
    ps_a(20, d, a, 0, b)
}

/// `ps_mul frD, frA, frC`.
pub(crate) const fn ps_mul(d: u32, a: u32, c: u32) -> u32 {
    ps_a(25, d, a, c, 0)
}

/// `ps_madd frD, frA, frC, frB`: `a * c + b` per half.
pub(crate) const fn ps_madd(d: u32, a: u32, c: u32, b: u32) -> u32 {
    ps_a(29, d, a, c, b)
}

/// `ps_sum0 frD, frA, frC, frB`: `d.ps0 = a.ps0 + b.ps1`, `d.ps1 = c.ps1`.
pub(crate) const fn ps_sum0(d: u32, a: u32, c: u32, b: u32) -> u32 {
    ps_a(10, d, a, c, b)
}

/// `ps_muls0 frD, frA, frC`: both halves times `c.ps0`.
pub(crate) const fn ps_muls0(d: u32, a: u32, c: u32) -> u32 {
    ps_a(12, d, a, c, 0)
}

/// `ps_madds0 frD, frA, frC, frB`: `a * c.ps0 + b`.
pub(crate) const fn ps_madds0(d: u32, a: u32, c: u32, b: u32) -> u32 {
    ps_a(14, d, a, c, b)
}

/// X-form register ops with a 10-bit xo: `op frD, frA, frB`.
const fn ps_x(xo: u32, d: u32, a: u32, b: u32) -> u32 {
    4 << 26 | d << 21 | a << 16 | b << 11 | xo << 1
}

/// `ps_merge00 frD, frA, frB`: `(a.ps0, b.ps0)`.
pub(crate) const fn ps_merge00(d: u32, a: u32, b: u32) -> u32 {
    ps_x(528, d, a, b)
}

/// `ps_merge11 frD, frA, frB`: `(a.ps1, b.ps1)`.
pub(crate) const fn ps_merge11(d: u32, a: u32, b: u32) -> u32 {
    ps_x(624, d, a, b)
}

/// `ps_mr frD, frB`.
pub(crate) const fn ps_mr(d: u32, b: u32) -> u32 {
    ps_x(72, d, 0, b)
}

/// GQR field values (Dolphin `UGQReg`, bits from the LSB): store type 0..2,
/// store scale 8..13, load type 16..18, load scale 24..29.
pub(crate) mod gqr {
    /// `f32`, no scaling.
    pub(crate) const F32: u32 = 0;
    /// Unsigned 8-bit.
    pub(crate) const U8: u32 = 4;
    /// Unsigned 16-bit.
    pub(crate) const U16: u32 = 5;
    /// Signed 8-bit.
    pub(crate) const S8: u32 = 6;
    /// Signed 16-bit.
    pub(crate) const S16: u32 = 7;

    /// A GQR value with scale 0 on both sides.
    pub(crate) const fn value(load: u32, store: u32) -> u32 {
        (load & 7) << 16 | (store & 7)
    }
}

// ------------------------------------------------------------------- VMX128

/// VX128 arithmetic (primary 5): `vD, vA, vB` with 7-bit vA and vD/vB.
const fn vx128(op: u32, xo4: u32, vd: u32, va: u32, vb: u32) -> u32 {
    op << 26
        | (vd & 31) << 21
        | (va & 31) << 16
        | (vb & 31) << 11
        | ((va >> 6) & 1) << 10
        | (xo4 & 15) << 6
        | ((va >> 5) & 1) << 5
        | 1 << 4
        | ((vd >> 5) & 3) << 2
        | ((vb >> 5) & 3)
}

/// `vaddfp128 vD, vA, vB`.
pub(crate) const fn vaddfp128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 0, vd, va, vb)
}

/// `vsubfp128 vD, vA, vB`.
pub(crate) const fn vsubfp128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 1, vd, va, vb)
}

/// `vmulfp128 vD, vA, vB` (VMX128 has a real multiply; VMX does not).
pub(crate) const fn vmulfp128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 2, vd, va, vb)
}

/// `vmaddfp128 vD, vA, vB`: **`vD = vA * vB + vD`** (destructive, per Xenia).
pub(crate) const fn vmaddfp128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 3, vd, va, vb)
}

/// `vmsum4fp128 vD, vA, vB`: the 4-lane dot product, splat to every lane.
pub(crate) const fn vmsum4fp128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 7, vd, va, vb)
}

/// `vor128 vD, vA, vB` (`vor128 vD, vS, vS` is the register move).
pub(crate) const fn vor128(vd: u32, va: u32, vb: u32) -> u32 {
    vx128(5, 11, vd, va, vb)
}

/// `vperm128 vD, vA, vB, vC` with `vC` in `v0`-`v7` only.
pub(crate) const fn vperm128(vd: u32, va: u32, vb: u32, vc: u32) -> u32 {
    5 << 26
        | (vd & 31) << 21
        | (va & 31) << 16
        | (vb & 31) << 11
        | ((va >> 6) & 1) << 10
        | (vc & 7) << 6
        | ((va >> 5) & 1) << 5
        | ((vd >> 5) & 3) << 2
        | ((vb >> 5) & 3)
}

/// VX128_1 loads/stores (primary 4): `vD, rA, rB` with a 7-bit xo.
const fn vx128_1(xo7: u32, vd: u32, ra: u32, rb: u32) -> u32 {
    4 << 26 | (vd & 31) << 21 | ra << 16 | rb << 11 | (xo7 & 127) << 4 | ((vd >> 5) & 3) << 2 | 3
}

/// `lvx128 vD, rA, rB`: aligned 16-byte load (low 4 address bits ignored).
pub(crate) const fn lvx128(vd: u32, ra: u32, rb: u32) -> u32 {
    vx128_1(12, vd, ra, rb)
}

/// `stvx128 vS, rA, rB`.
pub(crate) const fn stvx128(vs: u32, ra: u32, rb: u32) -> u32 {
    vx128_1(28, vs, ra, rb)
}

/// `lvsl128 vD, rA, rB`.
pub(crate) const fn lvsl128(vd: u32, ra: u32, rb: u32) -> u32 {
    vx128_1(0, vd, ra, rb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected words from binutils `gas/testsuite/gas/ppc/ppc750ps.d`.
    #[test]
    fn paired_singles_match_binutils() {
        assert_eq!(psq_l(0, 3, 4, true, 5), 0xe003_d004); // psq_l f0,4(r3),1,5
        assert_eq!(psq_lu(1, 2, 8, false, 3), 0xe422_3008); // psq_lu f1,8(r2),0,3
        assert_eq!(psq_lx(3, 2, 4, false, 5), 0x1062_228c); // psq_lx f3,r2,r4,0,5
        assert_eq!(psq_st(3, 2, 8, false, 3), 0xf062_3008); // psq_st f3,8(r2),0,3
        assert_eq!(psq_stu(3, 2, 8, false, 7), 0xf462_7008); // psq_stu f3,8(r2),0,7
        assert_eq!(psq_stx(6, 7, 8, true, 4), 0x10c7_460e); // psq_stx f6,r7,r8,1,4
        assert_eq!(ps_add(1, 2, 3), 0x1022_182a); // ps_add f1,f2,f3
        assert_eq!(ps_sub(5, 11, 2), 0x10ab_1028); // ps_sub f5,f11,f2
        assert_eq!(ps_mul(2, 3, 5), 0x1043_0172); // ps_mul f2,f3,f5
        assert_eq!(ps_madd(0, 1, 2, 3), 0x1001_18ba); // ps_madd f0,f1,f2,f3
        assert_eq!(ps_sum0(2, 5, 9, 10), 0x1045_5254); // ps_sum0 f2,f5,f9,f10
        assert_eq!(ps_muls0(3, 4, 7), 0x1064_01d8); // ps_muls0 f3,f4,f7
        assert_eq!(ps_madds0(1, 2, 3, 4), 0x1022_20dc); // ps_madds0 f1,f2,f3,f4
        assert_eq!(ps_merge00(2, 4, 6), 0x1044_3420); // ps_merge00 f2,f4,f6
        assert_eq!(ps_merge11(2, 4, 6), 0x1044_34e0); // ps_merge11 f2,f4,f6
        assert_eq!(ps_mr(3, 5), 0x1060_2890); // ps_mr f3,f5
    }

    #[test]
    fn psq_offsets_are_12_bit_signed() {
        assert_eq!(psq_l(1, 5, -8, false, 7) & 0xfff, 0xff8);
        assert_eq!(psq_l(1, 5, 2047, false, 7) & 0xfff, 0x7ff);
    }

    #[test]
    fn gqr_values() {
        assert_eq!(gqr::value(gqr::U16, gqr::F32), 0x0005_0000);
        assert_eq!(gqr::value(gqr::F32, gqr::F32), 0);
        assert_eq!(gqr::value(gqr::U8, gqr::S16), 0x0004_0007);
    }

    /// Xenia's decoder for the VMX128 forms used here, transcribed from
    /// `ppc_opcode_lookup_gen.cc` (IBM bit numbers: bit 0 is the MSB) and the
    /// `FormatVX128` / `FormatVX128_1` / `FormatVX128_2` field layouts.
    fn xenia_decode(w: u32) -> Option<(&'static str, u32, u32, u32)> {
        let bits = |a: u32, b: u32| (w >> (31 - b)) & ((1 << (b - a + 1)) - 1);
        let vd = bits(6, 10) | (w >> 2 & 3) << 5;
        let va = bits(11, 15) | (w >> 5 & 1) << 5 | (w >> 10 & 1) << 6;
        let vb = bits(16, 20) | (w & 3) << 5;
        match bits(0, 5) {
            4 => {
                let name = match bits(21, 27) << 4 | bits(30, 31) {
                    // Verbatim from Xenia (11-bit keys, no digit grouping).
                    0b00000000011 => "lvsl128",
                    0b00011000011 => "lvx128",
                    0b00111000011 => "stvx128",
                    _ => return None,
                };
                Some((name, vd, bits(11, 15), bits(16, 20)))
            }
            5 => {
                if bits(22, 22) << 5 | bits(27, 27) == 0 {
                    return Some(("vperm128", vd, va, vb | (w >> 6 & 7) << 8));
                }
                let name = match bits(22, 25) << 2 | bits(27, 27) {
                    0b000001 => "vaddfp128",
                    0b000101 => "vsubfp128",
                    0b001001 => "vmulfp128",
                    0b001101 => "vmaddfp128",
                    0b011101 => "vmsum4fp128",
                    0b101101 => "vor128",
                    _ => return None,
                };
                Some((name, vd, va, vb))
            }
            _ => None,
        }
    }

    #[test]
    fn vmx128_arithmetic_roundtrips_through_xenias_decoder() {
        type Enc = fn(u32, u32, u32) -> u32;
        let ops: [(&str, Enc); 6] = [
            ("vaddfp128", vaddfp128),
            ("vsubfp128", vsubfp128),
            ("vmulfp128", vmulfp128),
            ("vmaddfp128", vmaddfp128),
            ("vmsum4fp128", vmsum4fp128),
            ("vor128", vor128),
        ];
        for (name, enc) in ops {
            for vd in [0, 1, 31, 32, 63, 64, 100, 127] {
                for va in [0, 5, 31, 32, 64, 96, 127] {
                    for vb in [0, 17, 31, 33, 95, 127] {
                        let w = enc(vd, va, vb);
                        assert_eq!(
                            xenia_decode(w),
                            Some((name, vd, va, vb)),
                            "{name} {w:#010x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn vmx128_vperm_keeps_vc_in_three_bits() {
        for vc in 0..8 {
            let w = vperm128(70, 100, 33, vc);
            assert_eq!(xenia_decode(w), Some(("vperm128", 70, 100, 33 | vc << 8)));
        }
    }

    #[test]
    fn vmx128_loads_and_stores_roundtrip() {
        type Enc = fn(u32, u32, u32) -> u32;
        let ops: [(&str, Enc); 3] = [
            ("lvx128", lvx128),
            ("stvx128", stvx128),
            ("lvsl128", lvsl128),
        ];
        for (name, enc) in ops {
            for vd in [0, 31, 32, 64, 127] {
                for (ra, rb) in [(0, 5), (5, 7), (31, 30)] {
                    assert_eq!(
                        xenia_decode(enc(vd, ra, rb)),
                        Some((name, vd, ra, rb)),
                        "{name}"
                    );
                }
            }
        }
    }

    /// A word from the bit picture in Xenia's `vmx128.txt`, written out by hand.
    #[test]
    fn vmsum4fp128_matches_the_picture() {
        // |000101| VD=00001 | VA=00010 | VB=00011 |A=0|0111|a=0|1|VDh=00|VBh=00|
        assert_eq!(vmsum4fp128(1, 2, 3), 0x1422_19d0);
    }
}
