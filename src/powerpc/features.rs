//! PowerPC [`Feature`] bits and their probes.
//!
//! | Platform | Probe |
//! |---|---|
//! | Linux / Android | `getauxval(AT_HWCAP / AT_HWCAP2)` |
//! | FreeBSD | `elf_aux_info(AT_HWCAP / AT_HWCAP2)` |
//! | bare metal (`target_os = "none"`: libogc, wut, libxenon) | PVR, MSR and HID2, read in supervisor mode |
//! | PS3 GameOS (`target_os = "ps3"`) | assumed: the PPE always has AltiVec |
//! | anything else | compile-time `target_feature` only |
//!
//! HWCAP masks are from the Linux UAPI `asm/cputable.h`.
//!
//! **Extended register state.** [`Feature::PairedSingle`] (`ps1`, `HID2`,
//! the GQRs) and [`Feature::Vmx128`] (`v32`-`v127`) are state no mainline
//! kernel saves on a context switch, so they are reported only on bare metal,
//! never under an OS, even when the PVR says the core has them.

/// Bit masks (not indices) of `AT_HWCAP` / `AT_HWCAP2`.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap {
    // AT_HWCAP
    pub const HAS_ALTIVEC: u64 = 0x1000_0000;
    pub const CELL: u64 = 0x0001_0000;
    pub const HAS_VSX: u64 = 0x0000_0080;
    // AT_HWCAP2
    pub const ARCH_2_07: u64 = 0x8000_0000;
    pub const VEC_CRYPTO: u64 = 0x0200_0000;
    pub const ARCH_3_00: u64 = 0x0080_0000;
    pub const ARCH_3_1: u64 = 0x0004_0000;
    pub const MMA: u64 = 0x0002_0000;
}

/// Power ISA vector features and console extensions.
///
/// New features are appended to [`Feature::ALL`], so existing bits in
/// [`FeatureSet`] keep their index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// AltiVec / VMX: 32 x 128-bit `v` registers.
    Altivec,
    /// VSX (Power ISA 2.06, POWER7): 64 x 128-bit `vs` registers.
    Vsx,
    /// Power ISA 2.07 (POWER8).
    Power8,
    /// POWER8 AltiVec additions (with ISA 2.07).
    Power8Altivec,
    /// POWER8 vector additions (with ISA 2.07).
    Power8Vector,
    /// POWER8 vector crypto (`VEC_CRYPTO`, else ISA 2.07).
    Power8Crypto,
    /// Power ISA 3.0 (POWER9).
    Power9,
    /// POWER9 AltiVec additions (with ISA 3.0).
    Power9Altivec,
    /// POWER9 vector additions (with ISA 3.0).
    Power9Vector,
    /// Power ISA 3.1 (POWER10, POWER11): prefixed instructions, `lxvp`.
    Isa31,
    /// Matrix-Multiply Assist (POWER10+): eight 512-bit accumulators.
    /// Its own HWCAP2 bit; a 3.1 core without it is possible.
    Mma,
    /// Cell Broadband Engine PPE (PS3, QS20/QS22): `lvlx`/`lvrx`, in-order,
    /// AltiVec without VSX.
    Cell,
    /// Gekko / Broadway / Espresso paired singles (GameCube, Wii, Wii U).
    /// Bare metal only; see the module docs.
    PairedSingle,
    /// Xenon VMX128 (Xbox 360): 128 vector registers, `vmsum4fp128`.
    /// Bare metal only; see the module docs.
    Vmx128,
}

impl Feature {
    /// Every feature; index = bit in [`FeatureSet`].
    pub const ALL: &'static [Feature] = &[
        Feature::Altivec,
        Feature::Vsx,
        Feature::Power8,
        Feature::Power8Altivec,
        Feature::Power8Vector,
        Feature::Power8Crypto,
        Feature::Power9,
        Feature::Power9Altivec,
        Feature::Power9Vector,
        Feature::Isa31,
        Feature::Mma,
        Feature::Cell,
        Feature::PairedSingle,
        Feature::Vmx128,
    ];

    /// Bit index in [`FeatureSet`].
    #[inline]
    pub const fn bit(self) -> u32 {
        // No explicit discriminants and `ALL` is in declaration order, so the
        // discriminant is the index (checked by `bit_is_the_index_in_all`).
        self as u32
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
    hwcap: u64,
    hwcap2: u64,
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
    fn read() -> Self {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        // SAFETY: read-only auxv; tags are libc consts.
        unsafe {
            Self {
                hwcap: libc::getauxval(libc::AT_HWCAP) as u64,
                hwcap2: libc::getauxval(libc::AT_HWCAP2) as u64,
            }
        }
        #[cfg(target_os = "freebsd")]
        {
            fn one(key: libc::c_int) -> u64 {
                let mut out: libc::c_ulong = 0;
                // SAFETY: elf_aux_info writes size_of out; FreeBSD 12+.
                let _ = unsafe {
                    libc::elf_aux_info(
                        key,
                        (&mut out) as *mut libc::c_ulong as *mut libc::c_void,
                        core::mem::size_of::<libc::c_ulong>() as libc::c_int,
                    )
                };
                out as u64
            }
            Self {
                hwcap: one(libc::AT_HWCAP),
                hwcap2: one(libc::AT_HWCAP2),
            }
        }
    }

    fn supports(&self, feature: Feature) -> bool {
        let p8 = self.hwcap2 & hwcap::ARCH_2_07 != 0;
        let p9 = self.hwcap2 & hwcap::ARCH_3_00 != 0;
        let crypto = self.hwcap2 & hwcap::VEC_CRYPTO != 0 || p8;
        match feature {
            Feature::Altivec => self.hwcap & hwcap::HAS_ALTIVEC != 0,
            Feature::Vsx => self.hwcap & hwcap::HAS_VSX != 0,
            Feature::Power8 | Feature::Power8Altivec | Feature::Power8Vector => p8,
            Feature::Power8Crypto => crypto,
            Feature::Power9 | Feature::Power9Altivec | Feature::Power9Vector => p9,
            Feature::Isa31 => self.hwcap2 & hwcap::ARCH_3_1 != 0,
            Feature::Mma => self.hwcap2 & hwcap::MMA != 0,
            Feature::Cell => self.hwcap & hwcap::CELL != 0,
            // Never under an OS: the kernel does not save this state.
            Feature::PairedSingle | Feature::Vmx128 => false,
        }
    }
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
    /// Probes the host: auxv (Linux/Android/FreeBSD), the core registers (bare
    /// metal), or the compile-time features (everything else).
    pub fn detect() -> Self {
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        {
            let aux = AuxVal::read();
            let mut set = Self::default();
            for &feature in Feature::ALL {
                if aux.supports(feature) {
                    set = set.with(feature);
                }
            }
            set
        }

        #[cfg(target_os = "none")]
        {
            Self::from_bare_metal(super::class::bare_metal_regs())
        }

        #[cfg(not(any(
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "none"
        )))]
        {
            Self::compile_time()
        }
    }

    /// The bits a bare-metal register probe proves.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    pub(crate) const fn from_bare_metal(b: super::pvr::BareMetal) -> Self {
        let mut set = Self(0);
        if b.altivec {
            set = set.with(Feature::Altivec);
        }
        if b.vsx {
            set = set.with(Feature::Vsx);
        }
        if b.isa207 {
            set = set
                .with(Feature::Power8)
                .with(Feature::Power8Altivec)
                .with(Feature::Power8Vector)
                .with(Feature::Power8Crypto);
        }
        if b.isa300 {
            set = set
                .with(Feature::Power9)
                .with(Feature::Power9Altivec)
                .with(Feature::Power9Vector);
        }
        if b.isa31 {
            // Every POWER10/11 ships MMA; bare metal has no separate bit.
            set = set.with(Feature::Isa31).with(Feature::Mma);
        }
        if b.cell {
            set = set.with(Feature::Cell);
        }
        if b.vmx128 {
            set = set.with(Feature::Vmx128);
        }
        if b.paired_single {
            set = set.with(Feature::PairedSingle);
        }
        set
    }

    /// Bits the target guarantees: `-C target-feature` / `target-cpu` (visible
    /// on nightly only: every PowerPC target feature is still unstable) and
    /// the PS3 target, whose PPE always has AltiVec.
    pub(crate) const fn compile_time() -> Self {
        let mut set = Self(0);
        if cfg!(any(target_feature = "altivec", target_os = "ps3")) {
            set = set.with(Feature::Altivec);
        }
        if cfg!(target_os = "ps3") {
            set = set.with(Feature::Cell);
        }
        if cfg!(target_feature = "vsx") {
            set = set.with(Feature::Altivec).with(Feature::Vsx);
        }
        if cfg!(target_feature = "power8-vector") {
            set = set
                .with(Feature::Altivec)
                .with(Feature::Vsx)
                .with(Feature::Power8)
                .with(Feature::Power8Altivec)
                .with(Feature::Power8Vector);
        }
        if cfg!(target_feature = "power8-crypto") {
            set = set.with(Feature::Power8Crypto);
        }
        if cfg!(target_feature = "power9-vector") {
            set = set
                .with(Feature::Power9)
                .with(Feature::Power9Altivec)
                .with(Feature::Power9Vector);
        }
        if cfg!(target_feature = "power10-vector") {
            set = set.with(Feature::Isa31);
        }
        set
    }

    /// `self` plus `feature`.
    #[inline]
    pub(crate) const fn with(self, feature: Feature) -> Self {
        Self(self.0 | 1 << feature.bit())
    }

    /// Features present in **either** set (bitwise OR).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub fn contains(self, feature: Feature) -> bool {
        self.0 & (1 << feature.bit()) != 0
    }

    pub fn contains_all(self, required: &[Feature]) -> bool {
        required.iter().all(|&f| self.contains(f))
    }
}

/// Process-wide cache for one full [`FeatureSet`] probe. Thin wrapper over
/// [`crate::level_cache::CachedU128`], shared by every arch that keeps a
/// 128-bit feature set.
pub(crate) struct CachedFeatureSet(crate::level_cache::CachedU128);

impl CachedFeatureSet {
    pub(crate) const fn new() -> Self {
        Self(crate::level_cache::CachedU128::new())
    }

    /// Cached value, or run `init` once (races may double-init; same CPU => same set).
    #[inline]
    pub(crate) fn get_or_init(&self, init: impl FnOnce() -> FeatureSet) -> FeatureSet {
        FeatureSet(self.0.get_or_init(|| init().0))
    }
}

#[cfg(test)]
mod bit_index_tests {
    use super::Feature;

    /// `Feature::bit` relies on declaration order matching `Feature::ALL`;
    /// reordering one without the other fails here, not silently at runtime.
    #[test]
    fn bit_is_the_index_in_all() {
        for (i, &f) in Feature::ALL.iter().enumerate() {
            assert_eq!(f.bit() as usize, i, "{f:?}");
        }
        assert!(Feature::ALL.len() <= 128, "FeatureSet is a u128");
    }
}

#[cfg(test)]
mod tests {
    use super::super::pvr::{BareMetal, HID2_LSQE_PSE, MSR_FP, MSR_VEC, MSR_VSX};
    use super::*;

    #[test]
    fn bare_metal_wii_has_paired_singles_only() {
        let set =
            FeatureSet::from_bare_metal(BareMetal::from_regs(0x0008_7102, MSR_FP, HID2_LSQE_PSE));
        assert!(set.contains(Feature::PairedSingle));
        assert!(!set.contains(Feature::Altivec));
    }

    #[test]
    fn bare_metal_xenon_has_vmx_and_vmx128() {
        let set =
            FeatureSet::from_bare_metal(BareMetal::from_regs(0x0071_0200, MSR_FP | MSR_VEC, 0));
        assert!(set.contains_all(&[Feature::Altivec, Feature::Vmx128]));
        assert!(!set.contains(Feature::Vsx));
    }

    #[test]
    fn bare_metal_power10_fills_the_ladder() {
        let set = FeatureSet::from_bare_metal(BareMetal::from_regs(
            0x0080_0200,
            MSR_FP | MSR_VEC | MSR_VSX,
            0,
        ));
        for &f in Feature::ALL {
            let console = matches!(f, Feature::Cell | Feature::PairedSingle | Feature::Vmx128);
            assert_eq!(set.contains(f), !console, "{f:?}");
        }
    }

    /// Under an OS the console extensions never appear, whatever the target.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    #[test]
    fn no_extended_state_under_an_os() {
        let set = FeatureSet::detect();
        assert!(!set.contains(Feature::PairedSingle));
        assert!(!set.contains(Feature::Vmx128));
    }
}
