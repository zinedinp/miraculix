//! Core identification and the platform class: which console, desktop or
//! server family the code runs on.
//!
//! Like `riscv::class` this is a **hint** for tuning, not a capability proof:
//! use [`super::detect_features`] and the tokens for that. The decoding is pure
//! ([`PowerCore::from_pvr`], [`PowerClass::from_parts`]); only [`pvr`] and
//! [`cache_line_bytes`] touch the hardware or the OS.

pub use super::pvr::{CacheGeometry, PowerCore};
use super::{Feature, FeatureSet, detect_features};

/// Reads the Processor Version Register (`mfspr 287`) where that is safe.
///
/// - Linux / Android: user-mode `mfpvr` traps and the kernel emulates it.
///   That works but costs a trap, so call this once and keep the result.
/// - Bare metal (`target_os = "none"`): code runs in supervisor mode.
/// - Everything else (FreeBSD, PS3 GameOS, ...): `None`. There the
///   instruction may raise an illegal-instruction signal.
pub fn pvr() -> Option<u32> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "none"))]
    {
        Some(read_pvr())
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "none")))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "none"))]
fn read_pvr() -> u32 {
    let v: usize;
    // SAFETY: SPR 287 is readable in supervisor mode (bare metal) and emulated
    // for user mode by Linux; the instruction has no other effect.
    unsafe {
        core::arch::asm!("mfspr {0}, 287", out(reg) v, options(nomem, nostack, preserves_flags));
    }
    v as u32
}

/// PVR, MSR and (on paired-single cores) HID2, for the bare-metal probe.
#[cfg(target_os = "none")]
pub(crate) fn bare_metal_regs() -> super::pvr::BareMetal {
    let pvr = read_pvr();
    let msr: usize;
    // SAFETY: bare metal runs in supervisor mode, where `mfmsr` is allowed.
    unsafe {
        core::arch::asm!("mfmsr {0}", out(reg) msr, options(nomem, nostack, preserves_flags));
    }
    let hid2 = if PowerCore::from_pvr(pvr).has_paired_single() {
        let v: usize;
        // SAFETY: HID2 (SPR 920) exists on exactly these cores (checked just
        // above) and is readable in supervisor mode.
        unsafe {
            core::arch::asm!("mfspr {0}, 920", out(reg) v, options(nomem, nostack, preserves_flags));
        }
        v as u32
    } else {
        0
    };
    super::pvr::BareMetal::from_regs(pvr, msr as u64, hid2)
}

impl PowerCore {
    /// The running core, or [`PowerCore::Unknown`] where [`pvr`] is `None`.
    pub fn detect() -> Self {
        pvr().map_or(PowerCore::Unknown, PowerCore::from_pvr)
    }
}

/// Game consoles with a PowerPC CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Console {
    /// Nintendo GameCube (Gekko).
    GameCube,
    /// Nintendo Wii (Broadway; a 750CL board looks the same).
    Wii,
    /// Nintendo Wii U (Espresso).
    WiiU,
    /// Microsoft Xbox 360 (Xenon).
    Xbox360,
    /// Sony PlayStation 3 (Cell PPE; Cell blades look the same).
    Ps3,
}

/// Coarse platform class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PowerClass {
    /// A game console.
    Console(Console),
    /// Desktop G3/G4/G5 class (750, 74xx, 970).
    Desktop,
    /// IBM POWER7 and later: wide out-of-order cores with SMT4/SMT8.
    Server,
    /// An open OpenPOWER core such as Microwatt (scalar; no VMX/VSX).
    OpenCore,
    /// Nothing more specific is known.
    Unknown,
}

impl PowerClass {
    /// Pure classification from already-known facts (testable anywhere).
    pub fn from_parts(features: FeatureSet, core: PowerCore) -> Self {
        match core {
            PowerCore::Gekko => PowerClass::Console(Console::GameCube),
            PowerCore::Ppc750Cl => PowerClass::Console(Console::Wii),
            PowerCore::Espresso => PowerClass::Console(Console::WiiU),
            PowerCore::Xenon => PowerClass::Console(Console::Xbox360),
            PowerCore::Cell => PowerClass::Console(Console::Ps3),
            PowerCore::Ppc750 | PowerCore::G4 | PowerCore::G5 => PowerClass::Desktop,
            PowerCore::Power7
            | PowerCore::Power8
            | PowerCore::Power9
            | PowerCore::Power10
            | PowerCore::Power11 => PowerClass::Server,
            PowerCore::Microwatt => PowerClass::OpenCore,
            // No PVR (PS3 GameOS, FreeBSD): fall back to what the features say.
            PowerCore::Unknown if features.contains(Feature::Cell) => {
                PowerClass::Console(Console::Ps3)
            }
            PowerCore::Unknown if features.contains(Feature::Power8) => PowerClass::Server,
            PowerCore::Unknown => PowerClass::Unknown,
        }
    }

    /// Classifies the running system.
    pub fn detect() -> Self {
        Self::from_parts(detect_features(), PowerCore::detect())
    }
}

/// L1 data-cache line size in bytes: what `dcbz` clears and the natural
/// prefetch and padding unit (32 on 750/G4, 128 on G5/Cell/Xenon/POWER).
///
/// Linux and Android report it in `AT_DCACHEBSIZE`; elsewhere it comes from
/// the PVR table. `None` if neither knows.
pub fn cache_line_bytes() -> Option<u32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        const AT_DCACHEBSIZE: libc::c_ulong = 19;
        // SAFETY: read-only auxv lookup; 0 means "not present".
        let v = unsafe { libc::getauxval(AT_DCACHEBSIZE) };
        if v != 0 {
            return Some(v as u32);
        }
    }
    PowerCore::detect().cache_line_bytes()
}

/// Hardware threads per core (the SMT mode: 1, 2, 4 or 8 on POWER), from
/// Linux's `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`.
///
/// POWER cores share their vector units between threads (POWER9/10: per SMT4
/// half), so a vector-bound loop often runs best with fewer threads than the
/// OS shows CPUs. Use this to size a thread pool; measure before settling on a
/// number. `None` off Linux, or when sysfs is not readable (some containers,
/// and qemu-user, which shows the host's topology).
pub fn threads_per_core() -> Option<u32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let path = c"/sys/devices/system/cpu/cpu0/topology/thread_siblings_list";
        let mut buf = [0u8; 256];
        // SAFETY: `path` is NUL-terminated; `read` writes at most `buf.len()`
        // bytes into `buf`; the descriptor is closed on every path.
        let n = unsafe {
            let fd = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
            if fd < 0 {
                return None;
            }
            let n = libc::read(fd, buf.as_mut_ptr().cast(), buf.len());
            libc::close(fd);
            n
        };
        if n <= 0 {
            return None;
        }
        super::pvr::cpu_list_len(&buf[..n as usize])
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// A data or unified cache level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// L1 data cache.
    L1d,
    /// L2.
    L2,
    /// L3.
    L3,
}

/// Size, line size and associativity of one cache level, from the
/// `AT_L1D/L2/L3_CACHESIZE` and `..._CACHEGEOMETRY` auxv entries (Linux 4.11+
/// on PowerPC). `None` elsewhere or when the kernel reports nothing (qemu-user
/// passes none).
pub fn cache_geometry(level: CacheLevel) -> Option<CacheGeometry> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let (size, geo): (libc::c_ulong, libc::c_ulong) = match level {
            CacheLevel::L1d => (42, 43),
            CacheLevel::L2 => (44, 45),
            CacheLevel::L3 => (46, 47),
        };
        // SAFETY: read-only auxv lookups; 0 means "not present".
        let (s, g) = unsafe { (libc::getauxval(size), libc::getauxval(geo)) };
        #[allow(
            clippy::useless_conversion,
            reason = "`c_ulong` is `u32` on 32-bit PowerPC"
        )]
        CacheGeometry::from_auxv(u64::from(s), u64::from(g))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = level;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consoles_by_core() {
        let none = FeatureSet::default();
        assert_eq!(
            PowerClass::from_parts(none, PowerCore::Gekko),
            PowerClass::Console(Console::GameCube)
        );
        assert_eq!(
            PowerClass::from_parts(none, PowerCore::Ppc750Cl),
            PowerClass::Console(Console::Wii)
        );
        assert_eq!(
            PowerClass::from_parts(none, PowerCore::Espresso),
            PowerClass::Console(Console::WiiU)
        );
        assert_eq!(
            PowerClass::from_parts(none, PowerCore::Xenon),
            PowerClass::Console(Console::Xbox360)
        );
        assert_eq!(
            PowerClass::from_parts(none, PowerCore::Cell),
            PowerClass::Console(Console::Ps3)
        );
    }

    #[test]
    fn unknown_core_falls_back_to_features() {
        let cell = FeatureSet::default()
            .with(Feature::Altivec)
            .with(Feature::Cell);
        assert_eq!(
            PowerClass::from_parts(cell, PowerCore::Unknown),
            PowerClass::Console(Console::Ps3)
        );
        let p8 = FeatureSet::default().with(Feature::Power8);
        assert_eq!(
            PowerClass::from_parts(p8, PowerCore::Unknown),
            PowerClass::Server
        );
        assert_eq!(
            PowerClass::from_parts(FeatureSet::default(), PowerCore::Unknown),
            PowerClass::Unknown
        );
    }

    #[test]
    fn detect_is_consistent() {
        // Whatever the host is, the class must agree with its own parts.
        assert_eq!(
            PowerClass::detect(),
            PowerClass::from_parts(detect_features(), PowerCore::detect())
        );
        if let Some(t) = threads_per_core() {
            assert!((1..=64).contains(&t), "{t}");
        }
        for level in [CacheLevel::L1d, CacheLevel::L2, CacheLevel::L3] {
            if let Some(g) = cache_geometry(level) {
                assert!(g.line == 0 || g.line.is_power_of_two(), "{g:?}");
            }
        }
        if let Some(line) = cache_line_bytes() {
            assert!(
                line.is_power_of_two() && (16..=512).contains(&line),
                "{line}"
            );
        }
    }
}
