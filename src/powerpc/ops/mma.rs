//! Matrix-Multiply Assist (Power ISA 3.1, POWER10 / POWER11): the GEMM
//! micro-kernel on the eight 512-bit accumulators.
//!
//! Each accumulator `ACCt` is a 4 x 4 `f32` tile backed by `vs(4t)`-`vs(4t+3)`;
//! `xvf32gerpp ACCt, XA, XB` adds the outer product `XA[i] * XB[j]` to it (one
//! rounding per step). The 8 x 32 block runs as two 8 x 16 halves (2 row
//! groups x 4 column groups = 8 tiles). While an accumulator is primed its
//! VSRs must not be touched, so the operands live in `vs32`-`vs37`.
//!
//! `lxvw4x` / `stxvw4x` give big-endian element numbering on both endians,
//! the numbering the MMA instructions use, so there is no endian case.

use crate::powerpc::{Feature, FeatureSet};
use crate::reduce_common::{GEMM_MR, GEMM_NR};

/// Proof of MMA on an ISA 3.1 core with VSX on.
#[derive(Debug, Clone, Copy)]
pub struct Mma(());

/// One 8 x 16 half: `acc` points at column `16 * half` of row 0, `b` at
/// `b[16 * half]`.
///
/// # Safety
/// MMA on; `acc` is the 8 x 32 block (rows 128 bytes apart), `a` holds
/// `kc * 8` and `b` `kc * 32` floats from the given starts, `kc > 0`.
unsafe fn half(kc: usize, a: *const f32, b: *const f32, acc: *mut f32) {
    // SAFETY: caller's contract; every load and store stays in the block and
    // the panels.
    unsafe {
        core::arch::asm!(
                "mr {p}, {acc}",
                "lxvw4x 0, {p}, {c0}",
                "lxvw4x 4, {p}, {c16}",
                "lxvw4x 8, {p}, {c32}",
                "lxvw4x 12, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 1, {p}, {c0}",
                "lxvw4x 5, {p}, {c16}",
                "lxvw4x 9, {p}, {c32}",
                "lxvw4x 13, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 2, {p}, {c0}",
                "lxvw4x 6, {p}, {c16}",
                "lxvw4x 10, {p}, {c32}",
                "lxvw4x 14, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 3, {p}, {c0}",
                "lxvw4x 7, {p}, {c16}",
                "lxvw4x 11, {p}, {c32}",
                "lxvw4x 15, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 16, {p}, {c0}",
                "lxvw4x 20, {p}, {c16}",
                "lxvw4x 24, {p}, {c32}",
                "lxvw4x 28, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 17, {p}, {c0}",
                "lxvw4x 21, {p}, {c16}",
                "lxvw4x 25, {p}, {c32}",
                "lxvw4x 29, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 18, {p}, {c0}",
                "lxvw4x 22, {p}, {c16}",
                "lxvw4x 26, {p}, {c32}",
                "lxvw4x 30, {p}, {c48}",
                "addi {p}, {p}, 128",
                "lxvw4x 19, {p}, {c0}",
                "lxvw4x 23, {p}, {c16}",
                "lxvw4x 27, {p}, {c32}",
                "lxvw4x 31, {p}, {c48}",
                "addi {p}, {p}, 128",
                "xxmtacc 0",
                "xxmtacc 1",
                "xxmtacc 2",
                "xxmtacc 3",
                "xxmtacc 4",
                "xxmtacc 5",
                "xxmtacc 6",
                "xxmtacc 7",
                "mtctr {n}",
                "2:",
                "lxvw4x 32, 0, {a}",
                "lxvw4x 33, {a}, {c16}",
                "lxvw4x 34, {b}, {c0}",
                "lxvw4x 35, {b}, {c16}",
                "lxvw4x 36, {b}, {c32}",
                "lxvw4x 37, {b}, {c48}",
                "xvf32gerpp 0, 32, 34",
                "xvf32gerpp 1, 32, 35",
                "xvf32gerpp 2, 32, 36",
                "xvf32gerpp 3, 32, 37",
                "xvf32gerpp 4, 33, 34",
                "xvf32gerpp 5, 33, 35",
                "xvf32gerpp 6, 33, 36",
                "xvf32gerpp 7, 33, 37",
                "addi {a}, {a}, 32",
                "addi {b}, {b}, 128",
                "bdnz 2b",
                "xxmfacc 0",
                "xxmfacc 1",
                "xxmfacc 2",
                "xxmfacc 3",
                "xxmfacc 4",
                "xxmfacc 5",
                "xxmfacc 6",
                "xxmfacc 7",
                "mr {p}, {acc}",
                "stxvw4x 0, {p}, {c0}",
                "stxvw4x 4, {p}, {c16}",
                "stxvw4x 8, {p}, {c32}",
                "stxvw4x 12, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 1, {p}, {c0}",
                "stxvw4x 5, {p}, {c16}",
                "stxvw4x 9, {p}, {c32}",
                "stxvw4x 13, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 2, {p}, {c0}",
                "stxvw4x 6, {p}, {c16}",
                "stxvw4x 10, {p}, {c32}",
                "stxvw4x 14, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 3, {p}, {c0}",
                "stxvw4x 7, {p}, {c16}",
                "stxvw4x 11, {p}, {c32}",
                "stxvw4x 15, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 16, {p}, {c0}",
                "stxvw4x 20, {p}, {c16}",
                "stxvw4x 24, {p}, {c32}",
                "stxvw4x 28, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 17, {p}, {c0}",
                "stxvw4x 21, {p}, {c16}",
                "stxvw4x 25, {p}, {c32}",
                "stxvw4x 29, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 18, {p}, {c0}",
                "stxvw4x 22, {p}, {c16}",
                "stxvw4x 26, {p}, {c32}",
                "stxvw4x 30, {p}, {c48}",
                "addi {p}, {p}, 128",
                "stxvw4x 19, {p}, {c0}",
                "stxvw4x 23, {p}, {c16}",
                "stxvw4x 27, {p}, {c32}",
                "stxvw4x 31, {p}, {c48}",
                "addi {p}, {p}, 128",
                acc = in(reg_nonzero) acc,
                p = out(reg_nonzero) _,
                a = inout(reg_nonzero) a => _,
                b = inout(reg_nonzero) b => _,
                n = in(reg) kc,
                c0 = in(reg) 0usize, c16 = in(reg) 16usize,
                c32 = in(reg) 32usize, c48 = in(reg) 48usize,
                out("vs0") _, out("vs1") _, out("vs2") _, out("vs3") _, out("vs4") _, out("vs5") _, out("vs6") _, out("vs7") _, out("vs8") _, out("vs9") _, out("vs10") _, out("vs11") _, out("vs12") _, out("vs13") _, out("vs14") _, out("vs15") _, out("vs16") _, out("vs17") _, out("vs18") _, out("vs19") _, out("vs20") _, out("vs21") _, out("vs22") _, out("vs23") _, out("vs24") _, out("vs25") _, out("vs26") _, out("vs27") _, out("vs28") _, out("vs29") _, out("vs30") _, out("vs31") _, out("vs32") _, out("vs33") _, out("vs34") _, out("vs35") _, out("vs36") _, out("vs37") _,
                out("ctr") _,
                options(nostack),
        );
    }
}

impl Mma {
    /// Token if the cached feature set has MMA.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has MMA on an ISA 3.1 core with VSX.
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains_all(&[Feature::Altivec, Feature::Vsx, Feature::Isa31, Feature::Mma])
            .then_some(Mma(()))
    }

    /// `acc[i][j] += sum_k a[k*MR + i] * b[k*NR + j]`, k ascending, one rounding
    /// per step; same contract as `x86::reduce::gemm_f32_kernel`.
    ///
    /// # Panics
    /// If `a.len() < kc * GEMM_MR` or `b.len() < kc * GEMM_NR`.
    pub fn gemm_f32_kernel(
        self,
        kc: usize,
        a: &[f32],
        b: &[f32],
        acc: &mut [[f32; GEMM_NR]; GEMM_MR],
    ) {
        crate::reduce_common::gemm_check(kc, a, b);
        if kc == 0 {
            return;
        }
        let base = acc.as_mut_ptr().cast::<f32>();
        for h in 0..2 {
            // SAFETY: the token proves MMA; `gemm_check` sized the panels; the
            // block is `GEMM_MR` rows of `GEMM_NR` floats.
            unsafe { half(kc, a.as_ptr(), b.as_ptr().add(16 * h), base.add(16 * h)) };
        }
    }
}
