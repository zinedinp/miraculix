//! Cache prefetch hints: the CPU starts loading a line before the code needs it.
//!
//! | Target | [`read`] | [`write`] | [`instruction`] |
//! |---|---|---|---|
//! | x86 / x86_64 | `prefetcht0` / `t1` / `t2` / `nta` (SSE: always on x86_64, detected on 32-bit x86) | `prefetchw` (PRFCHW, detected), else as [`read`] | - |
//! | aarch64 | `prfm pldl1keep` / `pldl2keep` / `pldl3keep` / `pldl1strm` | `prfm pstl1keep` / ... | `prfm plil1keep` |
//! | arm (`nightly-arm-neon`) | `pld` (ARM state from ARMv5TE, Thumb-2) | `pld` | `pli` (ARMv7) |
//! | riscv32 / riscv64 | `prefetch.r` (Zicbop) | `prefetch.w` | `prefetch.i` |
//! | loongarch64 | `preld 0` | `preld 8` | - |
//! | powerpc / powerpc64 (`nightly-powerpc-asm`) | `dcbt` | `dcbtst` | - |
//! | wasm and the rest | - | - | - |
//!
//! A prefetch never faults, whatever the address (null, unmapped, past an allocation's end), so
//! these functions are safe; with `-` they do nothing. A [`Locality`] the target has no hint for
//! gets its plain one.
//!
//! - x86: `prefetchit0`/`prefetchit1` (PREFETCHI) take only RIP-relative addresses, so
//!   [`instruction`] has nothing to emit for a pointer. A build with `-C target-feature=+prfchw`
//!   skips the PRFCHW check in [`write`].
//! - arm: stable rustc reports no ARM architecture version (`arm_target_feature` is unstable),
//!   and `pld` on ARMv4T or in Thumb-1 code is undefined, so the hints need the
//!   `nightly-arm-neon` feature; without it they do nothing. `pldw` needs the Multiprocessing
//!   Extensions, which no `target_feature` reports, so [`write`] uses `pld`.
//! - RISC-V: the Zicbop encodings are `ori x0` hints, which harts without Zicbop run as no-ops,
//!   so no detection is needed.
//! - PowerPC: `asm!` there is unstable, so the hints need the `nightly-powerpc-asm` feature;
//!   without it they do nothing.

/// Which caches the line should end up in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Locality {
    /// Used again soon: every level, L1 included (`prefetcht0`, `pldl1keep`).
    L1,
    /// L2 and outward (`prefetcht1`, `pldl2keep`).
    L2,
    /// L3 and outward (`prefetcht2`, `pldl3keep`).
    L3,
    /// Used once: kept from pushing other data out (`prefetchnta`, `pldl1strm`).
    Streaming,
}

impl Locality {
    /// Every locality, nearest first.
    pub const ALL: [Locality; 4] = [
        Locality::L1,
        Locality::L2,
        Locality::L3,
        Locality::Streaming,
    ];
}

/// Prefetch the line holding `p` to be read.
#[inline(always)]
pub fn read<T>(p: *const T, locality: Locality) {
    imp::read(p.cast(), locality);
}

/// Prefetch the line holding `p` to be written: on most targets it arrives owned, so the store
/// needn't ask the other cores first.
#[inline(always)]
pub fn write<T>(p: *const T, locality: Locality) {
    imp::write(p.cast(), locality);
}

/// Prefetch the code at `p` (a function pointer cast with `as *const ()`) into the instruction cache.
#[inline(always)]
pub fn instruction<T>(p: *const T) {
    imp::instruction(p.cast());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod imp {
    use super::Locality;
    use crate::x86::{Feature, detect_features};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
    use core::sync::atomic::{AtomicU8, Ordering};

    /// One feature's presence, probed on first use: a relaxed load per hint, no call.
    struct Probe {
        feature: Feature,
        /// 0 not probed yet, 1 absent, 2 present.
        state: AtomicU8,
    }

    impl Probe {
        const fn new(feature: Feature) -> Self {
            Self {
                feature,
                state: AtomicU8::new(0),
            }
        }

        #[inline(always)]
        fn present(&self) -> bool {
            match self.state.load(Ordering::Relaxed) {
                0 => self.probe(),
                state => state == 2,
            }
        }

        #[cold]
        #[inline(never)]
        fn probe(&self) -> bool {
            let present = detect_features().contains(self.feature);
            self.state.store(1 + u8::from(present), Ordering::Relaxed);
            present
        }
    }

    static SSE: Probe = Probe::new(Feature::Sse);
    static PRFCHW: Probe = Probe::new(Feature::Prfchw);

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        if cfg!(target_feature = "sse") || SSE.present() {
            // SAFETY: the CPU has SSE.
            unsafe { sse(p, locality) }
        }
    }

    #[inline]
    #[target_feature(enable = "sse")]
    fn sse(p: *const u8, locality: Locality) {
        let p = p.cast::<i8>();
        match locality {
            Locality::L1 => _mm_prefetch::<_MM_HINT_T0>(p),
            Locality::L2 => _mm_prefetch::<_MM_HINT_T1>(p),
            Locality::L3 => _mm_prefetch::<_MM_HINT_T2>(p),
            Locality::Streaming => _mm_prefetch::<_MM_HINT_NTA>(p),
        }
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        if cfg!(target_feature = "prfchw") || PRFCHW.present() {
            // SAFETY: the CPU has PRFCHW; a prefetch never faults.
            unsafe {
                core::arch::asm!(
                    "prefetchw byte ptr [{p}]",
                    p = in(reg) p,
                    options(nostack, preserves_flags, readonly),
                );
            }
        } else {
            read(p, locality);
        }
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        let _ = p;
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::Locality;
    use core::arch::asm;

    macro_rules! prfm {
        ($op:literal, $p:expr) => {
            // SAFETY: `prfm` is base A64 and never faults.
            unsafe {
                asm!(
                    concat!("prfm ", $op, ", [{p}]"),
                    p = in(reg) $p,
                    options(nostack, preserves_flags, readonly),
                )
            }
        };
    }

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        match locality {
            Locality::L1 => prfm!("pldl1keep", p),
            Locality::L2 => prfm!("pldl2keep", p),
            Locality::L3 => prfm!("pldl3keep", p),
            Locality::Streaming => prfm!("pldl1strm", p),
        }
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        match locality {
            Locality::L1 => prfm!("pstl1keep", p),
            Locality::L2 => prfm!("pstl2keep", p),
            Locality::L3 => prfm!("pstl3keep", p),
            Locality::Streaming => prfm!("pstl1strm", p),
        }
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        prfm!("plil1keep", p);
    }
}

#[cfg(all(target_arch = "arm", feature = "nightly-arm-neon"))]
mod imp {
    use super::Locality;

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        let _ = locality;
        // `pld`: ARM state from ARMv5TE, Thumb state from Thumb-2 (not ARMv6-M, ARMv8-M Baseline).
        #[cfg(any(
            all(target_feature = "v5te", not(target_feature = "thumb-mode")),
            target_feature = "thumb2"
        ))]
        // SAFETY: the target has `pld`, which never faults.
        unsafe {
            core::arch::asm!("pld [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
        }
        #[cfg(not(any(
            all(target_feature = "v5te", not(target_feature = "thumb-mode")),
            target_feature = "thumb2"
        )))]
        let _ = p;
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        read(p, locality);
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        #[cfg(target_feature = "v7")]
        // SAFETY: ARMv7 has `pli`, which never faults.
        unsafe {
            core::arch::asm!("pli [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
        }
        #[cfg(not(target_feature = "v7"))]
        let _ = p;
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod imp {
    use super::Locality;

    // Zicbop, offset 0: `prefetch.i` = `ori x0, rs1, 0`, `.r` = `ori x0, rs1, 1`, `.w` = `ori x0, rs1, 3`.
    macro_rules! zicbop {
        ($imm:literal, $p:expr) => {
            // SAFETY: an `ori` to `x0` is a hint: a prefetch with Zicbop, a no-op without.
            unsafe {
                core::arch::asm!(
                    concat!("ori x0, {p}, ", $imm),
                    p = in(reg) $p,
                    options(nostack, preserves_flags, readonly),
                )
            }
        };
    }

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        let _ = locality;
        zicbop!("1", p);
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        let _ = locality;
        zicbop!("3", p);
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        zicbop!("0", p);
    }
}

#[cfg(target_arch = "loongarch64")]
mod imp {
    use super::Locality;

    // `preld` hint 0 loads into L1 to read, 8 to write.
    macro_rules! preld {
        ($hint:literal, $p:expr) => {
            // SAFETY: `preld` is base LoongArch64 and never faults.
            unsafe {
                core::arch::asm!(
                    concat!("preld ", $hint, ", {p}, 0"),
                    p = in(reg) $p,
                    options(nostack, preserves_flags, readonly),
                )
            }
        };
    }

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        let _ = locality;
        preld!("0", p);
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        let _ = locality;
        preld!("8", p);
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        let _ = p;
    }
}

#[cfg(all(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    feature = "nightly-powerpc-asm"
))]
mod imp {
    use super::Locality;

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        let _ = locality;
        // SAFETY: `dcbt` is in every PowerPC and does nothing for an inaccessible block.
        unsafe {
            core::arch::asm!("dcbt 0, {p}", p = in(reg) p, options(nostack, preserves_flags, readonly));
        }
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        let _ = locality;
        // SAFETY: as `dcbt`.
        unsafe {
            core::arch::asm!("dcbtst 0, {p}", p = in(reg) p, options(nostack, preserves_flags, readonly));
        }
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        let _ = p;
    }
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", feature = "nightly-arm-neon"),
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "loongarch64",
    all(
        any(target_arch = "powerpc", target_arch = "powerpc64"),
        feature = "nightly-powerpc-asm"
    )
)))]
mod imp {
    use super::Locality;

    #[inline(always)]
    pub fn read(p: *const u8, locality: Locality) {
        let _ = (p, locality);
    }

    #[inline(always)]
    pub fn write(p: *const u8, locality: Locality) {
        let _ = (p, locality);
    }

    #[inline(always)]
    pub fn instruction(p: *const u8) {
        let _ = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hint_takes_any_address() {
        let data = [0u64; 64];
        let addresses = [
            data.as_ptr(),
            data.as_ptr().wrapping_add(64),
            core::ptr::null(),
            core::ptr::without_provenance(usize::MAX),
        ];
        for p in addresses {
            for locality in Locality::ALL {
                read(p, locality);
                write(p, locality);
            }
            instruction(p);
        }
        instruction(every_hint_takes_any_address as *const ());
    }
}
