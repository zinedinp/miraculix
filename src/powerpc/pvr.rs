//! Pure core identification from the Processor Version Register (PVR, SPR 287).
//!
//! No `asm!` and no OS calls here, so the whole file also builds and runs its
//! tests on the host (see `lib.rs`). Reading the register lives in
//! [`super::class::pvr`].
//!
//! Values are checked against Linux `arch/powerpc/include/asm/reg.h` and
//! `kernel/cpu_specs_book3s_{32,64}.h`. The two console values Linux does not
//! carry come from Free60 (Xenon, `0x0071`) and linux-wiiu (Espresso,
//! `0x7001`); Microwatt's is from its `common.vhdl` (`0x0063`).

/// CPU core family, decoded from the PVR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PowerCore {
    /// Nintendo GameCube "Gekko" (PVR exactly `0x00083214`, which Linux lists
    /// as a 750CXe). Paired singles.
    Gekko,
    /// IBM 750CL, also the Wii's "Broadway" (mask `0xfffff0e0` = `0x00087000`).
    /// Paired singles. The two cannot be told apart by PVR.
    Ppc750Cl,
    /// Nintendo Wii U "Espresso" (three 750CL-class cores, PVR `0x7001xxxx`).
    /// Paired singles.
    Espresso,
    /// Any other 740/750 (G3) part: no paired singles, no AltiVec.
    Ppc750,
    /// Motorola/Freescale 7400/7410/74x5/7448 (G4): AltiVec.
    G4,
    /// IBM 970 / 970FX / 970MP / 970GX (G5): AltiVec.
    G5,
    /// Cell Broadband Engine PPE (PS3, QS20/QS22 blades): AltiVec, no VSX.
    Cell,
    /// Xbox 360 "Xenon" (PPE-class cores with VMX128).
    Xenon,
    /// POWER7 / POWER7+ (Power ISA 2.06, VSX).
    Power7,
    /// POWER8 / 8E / 8NVL (Power ISA 2.07).
    Power8,
    /// POWER9 (Power ISA 3.0).
    Power9,
    /// POWER10 (Power ISA 3.1, MMA).
    Power10,
    /// POWER11 (Power ISA 3.1, same user ISA as POWER10).
    Power11,
    /// Microwatt, the open OpenPOWER core (scalar SFFS, no VMX/VSX).
    Microwatt,
    /// Not in the table.
    Unknown,
}

impl PowerCore {
    /// Decodes a raw PVR value.
    pub const fn from_pvr(pvr: u32) -> Self {
        if pvr == 0x0008_3214 {
            return PowerCore::Gekko;
        }
        if pvr & 0xffff_f0e0 == 0x0008_7000 {
            return PowerCore::Ppc750Cl;
        }
        match pvr >> 16 {
            0x7001 => PowerCore::Espresso,
            0x0008 | 0x1008 => PowerCore::Ppc750,
            0x000c | 0x800c | 0x8000..=0x8004 => PowerCore::G4,
            0x0039 | 0x003c | 0x0044 | 0x0045 => PowerCore::G5,
            0x0070 => PowerCore::Cell,
            0x0071 => PowerCore::Xenon,
            0x003f | 0x004a => PowerCore::Power7,
            0x004b..=0x004d => PowerCore::Power8,
            0x004e => PowerCore::Power9,
            0x0080 => PowerCore::Power10,
            0x0082 => PowerCore::Power11,
            0x0063 => PowerCore::Microwatt,
            _ => PowerCore::Unknown,
        }
    }

    /// Gekko, Broadway/750CL and Espresso implement paired singles.
    pub const fn has_paired_single(self) -> bool {
        matches!(
            self,
            PowerCore::Gekko | PowerCore::Ppc750Cl | PowerCore::Espresso
        )
    }

    /// The core has an AltiVec/VMX unit (it may still be switched off in MSR).
    pub const fn has_vmx(self) -> bool {
        matches!(
            self,
            PowerCore::G4
                | PowerCore::G5
                | PowerCore::Cell
                | PowerCore::Xenon
                | PowerCore::Power7
                | PowerCore::Power8
                | PowerCore::Power9
                | PowerCore::Power10
                | PowerCore::Power11
        )
    }

    /// L1 data-cache line size in bytes (what `dcbz` clears), if known.
    ///
    /// 32 on the 750 family and the G4, 128 on the G5, Cell, Xenon and POWER.
    /// On Xenon plain `dcbz` clears only 32 bytes; `dcbz128` clears the line.
    pub const fn cache_line_bytes(self) -> Option<u32> {
        match self {
            PowerCore::Gekko
            | PowerCore::Ppc750Cl
            | PowerCore::Espresso
            | PowerCore::Ppc750
            | PowerCore::G4 => Some(32),
            PowerCore::G5
            | PowerCore::Cell
            | PowerCore::Xenon
            | PowerCore::Power7
            | PowerCore::Power8
            | PowerCore::Power9
            | PowerCore::Power10
            | PowerCore::Power11 => Some(128),
            PowerCore::Microwatt | PowerCore::Unknown => None,
        }
    }
}

/// `MSR[FP]`: floating point (and with it paired singles) is enabled.
pub(crate) const MSR_FP: u64 = 0x0000_2000;
/// `MSR[VEC]`: AltiVec/VMX (and VMX128) is enabled.
pub(crate) const MSR_VEC: u64 = 0x0200_0000;
/// `MSR[VSX]`: VSX is enabled.
pub(crate) const MSR_VSX: u64 = 0x0080_0000;
/// `HID2[LSQE]` (quantized load/store) and `HID2[PSE]` (paired singles), SPR 920.
pub(crate) const HID2_LSQE_PSE: u32 = 0x8000_0000 | 0x2000_0000;

/// What a bare-metal (supervisor-mode) probe proves, before it becomes a
/// `FeatureSet`. Built from the PVR, the MSR and, on paired-single cores, HID2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct BareMetal {
    pub altivec: bool,
    pub vsx: bool,
    pub isa207: bool,
    pub isa300: bool,
    pub isa31: bool,
    pub cell: bool,
    pub vmx128: bool,
    pub paired_single: bool,
}

impl BareMetal {
    /// A unit only counts when the core has it **and** the MSR has it switched
    /// on: with `MSR[VEC]` or `MSR[FP]` clear the first vector or FP
    /// instruction would trap. Paired singles also need `HID2[PSE|LSQE]`.
    pub(crate) const fn from_regs(pvr: u32, msr: u64, hid2: u32) -> Self {
        let core = PowerCore::from_pvr(pvr);
        let vec = msr & MSR_VEC != 0;
        let vsx = vec && msr & MSR_VSX != 0;
        let level = match core {
            PowerCore::Power7 => 1,
            PowerCore::Power8 => 2,
            PowerCore::Power9 => 3,
            PowerCore::Power10 | PowerCore::Power11 => 4,
            _ => 0,
        };
        BareMetal {
            altivec: vec && core.has_vmx(),
            vsx: vsx && level >= 1,
            isa207: vsx && level >= 2,
            isa300: vsx && level >= 3,
            isa31: vsx && level >= 4,
            cell: matches!(core, PowerCore::Cell),
            vmx128: vec && matches!(core, PowerCore::Xenon),
            paired_single: core.has_paired_single()
                && msr & MSR_FP != 0
                && hid2 & HID2_LSQE_PSE == HID2_LSQE_PSE,
        }
    }
}

/// Number of CPUs in a Linux CPU list such as `0-7`, `0,8,16,24` or
/// `0-3,64-67` (the `thread_siblings_list` format). `None` if malformed.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android")),
    allow(dead_code, reason = "read from sysfs / auxv on Linux only")
)]
pub(crate) fn cpu_list_len(list: &[u8]) -> Option<u32> {
    let text = core::str::from_utf8(list).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    let mut n = 0u32;
    for part in text.split(',') {
        let mut ends = part.split('-');
        let lo: u32 = ends.next()?.trim().parse().ok()?;
        let hi: u32 = match ends.next() {
            Some(h) => h.trim().parse().ok()?,
            None => lo,
        };
        if ends.next().is_some() || hi < lo {
            return None;
        }
        n = n.checked_add(hi - lo + 1)?;
    }
    Some(n)
}

/// One cache level as the PowerPC auxv reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheGeometry {
    /// Total size in bytes.
    pub size: u32,
    /// Line size in bytes.
    pub line: u32,
    /// Associativity (ways); 0 if unknown, `0xffff` for fully associative.
    pub ways: u32,
}

impl CacheGeometry {
    /// From `AT_*_CACHESIZE` and `AT_*_CACHEGEOMETRY` (line size in bits 0..16,
    /// associativity in bits 16..32). `None` when the kernel reports nothing.
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android")),
        allow(dead_code, reason = "read from auxv on Linux only")
    )]
    pub(crate) const fn from_auxv(size: u64, geometry: u64) -> Option<Self> {
        if size == 0 && geometry == 0 {
            return None;
        }
        Some(CacheGeometry {
            size: size as u32,
            line: (geometry & 0xffff) as u32,
            ways: ((geometry >> 16) & 0xffff) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_pvrs_decode() {
        assert_eq!(PowerCore::from_pvr(0x0008_3214), PowerCore::Gekko);
        // Broadway as reported by Wii Linux, and a 750CL DD2.0.
        assert_eq!(PowerCore::from_pvr(0x0008_7102), PowerCore::Ppc750Cl);
        assert_eq!(PowerCore::from_pvr(0x0008_7200), PowerCore::Ppc750Cl);
        assert_eq!(PowerCore::from_pvr(0x7001_0201), PowerCore::Espresso);
        assert_eq!(PowerCore::from_pvr(0x0070_0501), PowerCore::Cell);
        assert_eq!(PowerCore::from_pvr(0x0071_0200), PowerCore::Xenon);
        // A plain 750CXe is not a Gekko.
        assert_eq!(PowerCore::from_pvr(0x0008_3311), PowerCore::Ppc750);
    }

    #[test]
    fn desktop_and_server_pvrs_decode() {
        assert_eq!(PowerCore::from_pvr(0x000c_0209), PowerCore::G4);
        assert_eq!(PowerCore::from_pvr(0x8004_0202), PowerCore::G4);
        assert_eq!(PowerCore::from_pvr(0x003c_0300), PowerCore::G5);
        assert_eq!(PowerCore::from_pvr(0x004a_0201), PowerCore::Power7);
        assert_eq!(PowerCore::from_pvr(0x004d_0200), PowerCore::Power8);
        assert_eq!(PowerCore::from_pvr(0x004e_1202), PowerCore::Power9);
        assert_eq!(PowerCore::from_pvr(0x0080_0200), PowerCore::Power10);
        assert_eq!(PowerCore::from_pvr(0x0082_0200), PowerCore::Power11);
        assert_eq!(PowerCore::from_pvr(0x0063_0000), PowerCore::Microwatt);
        assert_eq!(PowerCore::from_pvr(0x0f00_0006), PowerCore::Unknown);
    }

    #[test]
    fn cache_lines() {
        assert_eq!(PowerCore::Gekko.cache_line_bytes(), Some(32));
        assert_eq!(PowerCore::G4.cache_line_bytes(), Some(32));
        assert_eq!(PowerCore::Cell.cache_line_bytes(), Some(128));
        assert_eq!(PowerCore::Xenon.cache_line_bytes(), Some(128));
        assert_eq!(PowerCore::Power10.cache_line_bytes(), Some(128));
        assert_eq!(PowerCore::Unknown.cache_line_bytes(), None);
    }

    #[test]
    fn bare_metal_needs_the_unit_switched_on() {
        let on = MSR_FP | MSR_VEC | MSR_VSX;
        // Wii with paired singles enabled by the SDK.
        let wii = BareMetal::from_regs(0x0008_7102, MSR_FP, HID2_LSQE_PSE);
        assert!(wii.paired_single && !wii.altivec);
        // Same core, HID2[PSE] clear: no paired singles.
        assert!(!BareMetal::from_regs(0x0008_7102, MSR_FP, 0x8000_0000).paired_single);
        // FP off: no paired singles either.
        assert!(!BareMetal::from_regs(0x0008_3214, 0, HID2_LSQE_PSE).paired_single);
        // Xenon: VMX and VMX128 only with MSR[VEC].
        let x = BareMetal::from_regs(0x0071_0200, on, 0);
        assert!(x.altivec && x.vmx128 && !x.vsx && !x.paired_single);
        assert_eq!(
            BareMetal::from_regs(0x0071_0200, MSR_FP, 0),
            BareMetal::default()
        );
        // Cell is flagged even with VMX off; VMX itself is not.
        let c = BareMetal::from_regs(0x0070_0501, MSR_FP, 0);
        assert!(c.cell && !c.altivec);
        // POWER10 bare metal: the whole ladder.
        let p = BareMetal::from_regs(0x0080_0200, on, 0);
        assert!(p.altivec && p.vsx && p.isa207 && p.isa300 && p.isa31);
        // POWER9 with VSX off in MSR: AltiVec only.
        let q = BareMetal::from_regs(0x004e_1202, MSR_VEC, 0);
        assert!(q.altivec && !q.vsx && !q.isa300);
        // Microwatt: scalar.
        assert_eq!(
            BareMetal::from_regs(0x0063_0000, on, 0),
            BareMetal::default()
        );
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(
            cpu_list_len(
                b"0-7
"
            ),
            Some(8)
        );
        assert_eq!(cpu_list_len(b"0,8,16,24"), Some(4));
        assert_eq!(cpu_list_len(b"0-3,64-67"), Some(8));
        assert_eq!(cpu_list_len(b"5"), Some(1));
        assert_eq!(cpu_list_len(b""), None);
        assert_eq!(cpu_list_len(b"3-1"), None);
        assert_eq!(cpu_list_len(b"0-1-2"), None);
        assert_eq!(cpu_list_len(b"x"), None);
    }

    #[test]
    fn cache_geometry_decodes() {
        // POWER9 L1d: 32 KiB, 128-byte lines, 8 ways.
        assert_eq!(
            CacheGeometry::from_auxv(32768, 8 << 16 | 128),
            Some(CacheGeometry {
                size: 32768,
                line: 128,
                ways: 8
            })
        );
        assert_eq!(CacheGeometry::from_auxv(0, 0), None);
    }
}
