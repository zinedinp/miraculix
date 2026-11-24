//! VMX128 reductions for Xenon (Xbox 360): 128 vector registers per hardware
//! thread instead of VMX's 32.
//!
//! - **Register plan.** Accumulators live in `v32`-`v39`, load temporaries in
//!   `v64`-`v95`; only the two permute controls (`v0`, `v1`, because
//!   `vperm128` takes its control from `v0`-`v7`) and two scratch registers
//!   touch the VMX file the compiler knows. Eight vectors (32 floats) per input
//!   are in flight per trip, which hides the in-order core's FP latency.
//! - **Loads.** As in [`super::vmx`]: two aligned `lvx128` per vector plus a
//!   `vperm128`, so nothing outside the slice's 16-byte blocks is read.
//! - **Encodings.** No assembler knows VMX128; every such instruction is a
//!   `.long` from `encode`, checked on the host against a decoder transcribed
//!   from Xenia. GPRs are fixed (`r0`, `r3`-`r11`) because their numbers are in
//!   the words. Not executed in CI: there is no runner for Xenon code.
//!
//! The element-wise ops gain nothing from the larger file (they are
//! load/store bound), so [`super::vmx::Vmx`] serves those on Xenon.
//!
//! Only reported on bare metal (`target_os = "none"`): no mainline kernel
//! saves `v32`-`v127`. See [`crate::powerpc::features`].

use super::encode::{lvx128, vaddfp128, vmaddfp128, vmsum4fp128, vor128, vperm128};
use crate::powerpc::{Feature, FeatureSet};

/// Independent vector accumulators (`v32`-`v39`).
pub const ACCUMULATORS: usize = 8;

/// Proof of Xenon VMX128 with `MSR[VEC]` set.
#[derive(Debug, Clone, Copy)]
pub struct Vmx128(());

// GPRs baked into the words: r3 = a, r4 = b, r0/r5..r11 = byte offsets.
const RA: u32 = 3;
const RB: u32 = 4;
const OFF: [u32; 8] = [0, 5, 6, 7, 8, 9, 10, 11]; // 0, 15, 16, 31, 32, 47, 48, 63

/// Unaligned vector `k` (0..4) of the current 64-byte group of input `ptr`
/// into `v$dst`, using `v$dst + 1` as the high block; control in `v$ctl`.
const fn lo(ptr: u32, k: usize, dst: u32) -> u32 {
    lvx128(dst, ptr, OFF[2 * k])
}
const fn hi(ptr: u32, k: usize, dst: u32) -> u32 {
    lvx128(dst + 1, ptr, OFF[2 * k + 1])
}
const fn perm(dst: u32, ctl: u32) -> u32 {
    vperm128(dst, dst, dst + 1, ctl)
}

impl Vmx128 {
    /// Token if the cached feature set has VMX128.
    #[inline]
    pub fn detect() -> Option<Self> {
        Self::from_features(crate::powerpc::detect_features())
    }

    /// Token if `set` has VMX128 (which implies VMX on Xenon).
    #[inline]
    pub fn from_features(set: FeatureSet) -> Option<Self> {
        set.contains_all(&[Feature::Altivec, Feature::Vmx128])
            .then_some(Vmx128(()))
    }

    /// Dot product: eight `vmaddfp128` accumulators (`acc = a * b + acc`) over
    /// 32 floats per trip, summed with `vaddfp128` and a final `vmsum4fp128`
    /// against 1.0; scalar tail. NaN and inf propagate.
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
        // SAFETY: the token proves VMX128 on bare metal; `trips * 32` floats of
        // `a` and `b` are in bounds (the `+15` loads stay in the same blocks);
        // `stvewx` writes one aligned word to `sum`.
        unsafe {
            asm_mixed!(
                s("lvsl 0, 0, 3"),
                s("lvsl 1, 0, 4"),
                s("vxor 2, 2, 2"),
                w(vor128(32, 2, 2)), w(vor128(33, 2, 2)), w(vor128(34, 2, 2)), w(vor128(35, 2, 2)),
                w(vor128(36, 2, 2)), w(vor128(37, 2, 2)), w(vor128(38, 2, 2)), w(vor128(39, 2, 2)),
                s("mtctr {n}"),
                s("2:"),
                // first 64 bytes: a -> v64/68/72/76, b -> v80/84/88/92 (+1 = high block)
                w(lo(RA, 0, 64)), w(hi(RA, 0, 64)), w(lo(RB, 0, 80)), w(hi(RB, 0, 80)),
                w(lo(RA, 1, 68)), w(hi(RA, 1, 68)), w(lo(RB, 1, 84)), w(hi(RB, 1, 84)),
                w(lo(RA, 2, 72)), w(hi(RA, 2, 72)), w(lo(RB, 2, 88)), w(hi(RB, 2, 88)),
                w(lo(RA, 3, 76)), w(hi(RA, 3, 76)), w(lo(RB, 3, 92)), w(hi(RB, 3, 92)),
                w(perm(64, 0)), w(perm(80, 1)), w(perm(68, 0)), w(perm(84, 1)),
                w(perm(72, 0)), w(perm(88, 1)), w(perm(76, 0)), w(perm(92, 1)),
                w(vmaddfp128(32, 64, 80)), w(vmaddfp128(33, 68, 84)),
                w(vmaddfp128(34, 72, 88)), w(vmaddfp128(35, 76, 92)),
                s("addi 3, 3, 64"),
                s("addi 4, 4, 64"),
                // second 64 bytes: v66/70/74/78 and v82/86/90/94
                w(lo(RA, 0, 66)), w(hi(RA, 0, 66)), w(lo(RB, 0, 82)), w(hi(RB, 0, 82)),
                w(lo(RA, 1, 70)), w(hi(RA, 1, 70)), w(lo(RB, 1, 86)), w(hi(RB, 1, 86)),
                w(lo(RA, 2, 74)), w(hi(RA, 2, 74)), w(lo(RB, 2, 90)), w(hi(RB, 2, 90)),
                w(lo(RA, 3, 78)), w(hi(RA, 3, 78)), w(lo(RB, 3, 94)), w(hi(RB, 3, 94)),
                w(perm(66, 0)), w(perm(82, 1)), w(perm(70, 0)), w(perm(86, 1)),
                w(perm(74, 0)), w(perm(90, 1)), w(perm(78, 0)), w(perm(94, 1)),
                w(vmaddfp128(36, 66, 82)), w(vmaddfp128(37, 70, 86)),
                w(vmaddfp128(38, 74, 90)), w(vmaddfp128(39, 78, 94)),
                s("addi 3, 3, 64"),
                s("addi 4, 4, 64"),
                s("bdnz 2b"),
                w(vaddfp128(32, 32, 33)), w(vaddfp128(34, 34, 35)),
                w(vaddfp128(36, 36, 37)), w(vaddfp128(38, 38, 39)),
                w(vaddfp128(32, 32, 34)), w(vaddfp128(36, 36, 38)), w(vaddfp128(32, 32, 36)),
                // horizontal: dot with (1, 1, 1, 1), splat to every lane
                s("vspltisw 2, 1"),
                s("vcfsx 2, 2, 0"),
                w(vmsum4fp128(33, 32, 2)),
                w(vor128(3, 33, 33)),
                s("stvewx 3, 0, {s}"),
                ;
                n = in(reg) trips,
                s = in(reg) &mut sum as *mut f32,
                inout("r3") a.as_ptr() => _, inout("r4") b.as_ptr() => _,
                in("r0") 0usize, in("r5") 15usize, in("r6") 16usize, in("r7") 31usize,
                in("r8") 32usize, in("r9") 47usize, in("r10") 48usize, in("r11") 63usize,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::dot4(a, b, trips * 32)
    }

    /// Sum of all elements: eight `vaddfp128` accumulators over 32 floats per
    /// trip, then `vmsum4fp128` against 1.0; scalar tail. NaN propagates.
    pub fn reduce_sum_f32(self, a: &[f32]) -> f32 {
        let trips = a.len() / 32;
        if trips == 0 {
            return crate::reduce_common::sum4(a);
        }
        let mut sum = 0f32;
        // SAFETY: as in `dot_f32`, one input.
        unsafe {
            asm_mixed!(
                s("lvsl 0, 0, 3"),
                s("vxor 2, 2, 2"),
                w(vor128(32, 2, 2)), w(vor128(33, 2, 2)), w(vor128(34, 2, 2)), w(vor128(35, 2, 2)),
                w(vor128(36, 2, 2)), w(vor128(37, 2, 2)), w(vor128(38, 2, 2)), w(vor128(39, 2, 2)),
                s("mtctr {n}"),
                s("2:"),
                w(lo(RA, 0, 64)), w(hi(RA, 0, 64)), w(lo(RA, 1, 66)), w(hi(RA, 1, 66)),
                w(lo(RA, 2, 68)), w(hi(RA, 2, 68)), w(lo(RA, 3, 70)), w(hi(RA, 3, 70)),
                s("addi 3, 3, 64"),
                w(lo(RA, 0, 72)), w(hi(RA, 0, 72)), w(lo(RA, 1, 74)), w(hi(RA, 1, 74)),
                w(lo(RA, 2, 76)), w(hi(RA, 2, 76)), w(lo(RA, 3, 78)), w(hi(RA, 3, 78)),
                s("addi 3, 3, 64"),
                w(perm(64, 0)), w(perm(66, 0)), w(perm(68, 0)), w(perm(70, 0)),
                w(perm(72, 0)), w(perm(74, 0)), w(perm(76, 0)), w(perm(78, 0)),
                w(vaddfp128(32, 32, 64)), w(vaddfp128(33, 33, 66)),
                w(vaddfp128(34, 34, 68)), w(vaddfp128(35, 35, 70)),
                w(vaddfp128(36, 36, 72)), w(vaddfp128(37, 37, 74)),
                w(vaddfp128(38, 38, 76)), w(vaddfp128(39, 39, 78)),
                s("bdnz 2b"),
                w(vaddfp128(32, 32, 33)), w(vaddfp128(34, 34, 35)),
                w(vaddfp128(36, 36, 37)), w(vaddfp128(38, 38, 39)),
                w(vaddfp128(32, 32, 34)), w(vaddfp128(36, 36, 38)), w(vaddfp128(32, 32, 36)),
                s("vspltisw 2, 1"),
                s("vcfsx 2, 2, 0"),
                w(vmsum4fp128(33, 32, 2)),
                w(vor128(3, 33, 33)),
                s("stvewx 3, 0, {s}"),
                ;
                n = in(reg) trips,
                s = in(reg) &mut sum as *mut f32,
                inout("r3") a.as_ptr() => _,
                in("r0") 0usize, in("r5") 15usize, in("r6") 16usize, in("r7") 31usize,
                in("r8") 32usize, in("r9") 47usize, in("r10") 48usize, in("r11") 63usize,
                out("v0") _, out("v2") _, out("v3") _, out("ctr") _,
                options(nostack),
            );
        }
        sum + crate::reduce_common::sum4(&a[trips * 32..])
    }
}
