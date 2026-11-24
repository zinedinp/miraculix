//! RISC-V base ISA (M/A/F/D/C/V). Auxv or compile-time cfg. [`hwcap`] restates
//! `COMPAT_HWCAP_ISA_*` (UAPI bit is letter minus `'A'`). No Feature::I (always
//! implied). `Zvfh` / `Zvfbfmin` / `Zvfbfwma` come from `riscv_hwprobe(2)`
//! (Linux, Android).

/// `unsigned long`: u32 on RV32, u64 on RV64.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
#[cfg(target_pointer_width = "32")]
type Hwcap = u32;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
#[cfg(target_pointer_width = "64")]
type Hwcap = u64;

/// `COMPAT_HWCAP_ISA_*`. Kernel UAPI `asm/hwcap.h`.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap {
    use super::Hwcap;

    pub const M: Hwcap = 1 << (b'M' - b'A');
    pub const A: Hwcap = 1; // 1 << ('A' - 'A')
    pub const F: Hwcap = 1 << (b'F' - b'A');
    pub const D: Hwcap = 1 << (b'D' - b'A');
    pub const C: Hwcap = 1 << (b'C' - b'A');
    pub const V: Hwcap = 1 << (b'V' - b'A');
}

/// Enum list for RISC-V HWCAP ISA letters (no I: always implied).
///
/// New features (e.g. Zba/Zbb/Zvfh from `riscv_hwprobe(2)`) are appended to
/// [`Feature::ALL`], so existing bits in [`FeatureSet`] keep their index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// Mul/div.
    M,
    /// Atomics.
    A,
    /// f32 FP.
    F,
    /// f64 FP.
    D,
    /// Compressed 16-bit encodings.
    C,
    /// Vector (app processors).
    V,
    /// Vector half-precision FP (`Zvfh`), from `riscv_hwprobe(2)`. Implies V.
    Zvfh,
    /// Vector BF16 converts (`Zvfbfmin`), from `riscv_hwprobe(2)`. Implies V.
    Zvfbfmin,
    /// Vector widening BF16 multiply-accumulate (`Zvfbfwma`), from `riscv_hwprobe(2)`. Implies V.
    Zvfbfwma,
}

impl Feature {
    /// HWCAP letter order; index = bit in [`FeatureSet`].
    pub const ALL: &'static [Feature] = &[
        Feature::M,
        Feature::A,
        Feature::F,
        Feature::D,
        Feature::C,
        Feature::V,
        Feature::Zvfh,
        Feature::Zvfbfmin,
        Feature::Zvfbfwma,
    ];

    /// Bit index in [`FeatureSet`].
    pub const fn bit(self) -> u32 {
        // The enum has no explicit discriminants and `ALL` lists the
        // variants in declaration order, so the discriminant *is* the index
        // in `ALL` (checked by `bit_is_the_index_in_all`). O(1), no search.
        self as u32
    }
}

/// One auxv `AT_HWCAP` snapshot.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
    hwcap: Hwcap,
    /// `RISCV_HWPROBE_KEY_IMA_EXT_0`, or 0 when unavailable.
    ext0: u64,
}

/// `RISCV_HWPROBE_KEY_IMA_EXT_0` bits (Linux UAPI `asm/hwprobe.h`).
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod ext0 {
    pub const ZVFH: u64 = 1 << 30;
    pub const ZVFBFMIN: u64 = 1 << 53;
    pub const ZVFBFWMA: u64 = 1 << 54;
}

/// `riscv_hwprobe(2)` for one key (`RISCV_HWPROBE_KEY_*`). `None` if the kernel
/// lacks the syscall or does not know the key. Linux and Android (same kernel
/// syscall) on RISC-V only.
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
pub(super) fn hwprobe(key: i64) -> Option<u64> {
    const NR_RISCV_HWPROBE: usize = 258;
    let mut pair = [key as usize, 0usize];
    let ret: usize;
    // SAFETY: Linux riscv syscall ABI; `pair` is one valid {key, value} entry
    // and the cpu set is empty (all online harts). Unknown syscall -> error.
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a7") NR_RISCV_HWPROBE,
            inlateout("a0") pair.as_mut_ptr() => ret,
            in("a1") 1usize, in("a2") 0usize, in("a3") 0usize, in("a4") 0usize,
            options(nostack),
        );
    }
    (ret == 0 && pair[0] as i64 == key).then_some(pair[1] as u64)
}

#[cfg(not(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "riscv32", target_arch = "riscv64")
)))]
pub(super) fn hwprobe(_key: i64) -> Option<u64> {
    None
}

/// `RISCV_HWPROBE_KEY_IMA_EXT_0`, or 0 when unavailable.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
fn hwprobe_ext0() -> u64 {
    hwprobe(4).unwrap_or(0)
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
    fn read() -> Self {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        // SAFETY: read-only auxv; tag is a libc const.
        unsafe {
            return Self {
                hwcap: libc::getauxval(libc::AT_HWCAP) as Hwcap,
                ext0: hwprobe_ext0(),
            };
        }
        #[cfg(target_os = "freebsd")]
        {
            let mut out: libc::c_ulong = 0;
            // SAFETY: elf_aux_info writes size_of out; FreeBSD 12+.
            let _ = unsafe {
                libc::elf_aux_info(
                    libc::AT_HWCAP,
                    (&mut out) as *mut libc::c_ulong as *mut libc::c_void,
                    core::mem::size_of::<libc::c_ulong>() as libc::c_int,
                )
            };
            Self {
                hwcap: out as Hwcap,
                ext0: hwprobe_ext0(),
            }
        }
    }

    /// `match` so a new Feature arm is a compile error.
    fn supports(&self, feature: Feature) -> bool {
        match feature {
            Feature::M => self.hwcap & hwcap::M != 0,
            Feature::A => self.hwcap & hwcap::A != 0,
            Feature::F => self.hwcap & hwcap::F != 0,
            Feature::D => self.hwcap & hwcap::D != 0,
            Feature::C => self.hwcap & hwcap::C != 0,
            Feature::V => self.hwcap & hwcap::V != 0,
            Feature::Zvfh => self.ext0 & ext0::ZVFH != 0 && self.hwcap & hwcap::V != 0,
            Feature::Zvfbfmin => self.ext0 & ext0::ZVFBFMIN != 0 && self.hwcap & hwcap::V != 0,
            Feature::Zvfbfwma => self.ext0 & ext0::ZVFBFWMA != 0 && self.hwcap & hwcap::V != 0,
        }
    }
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
    /// Host via auxv, or compile-time cfg (bare-metal/other).
    pub fn detect() -> Self {
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        {
            let aux = AuxVal::read();
            let mut set = 0u128;
            for &feature in Feature::ALL {
                if aux.supports(feature) {
                    set |= 1 << feature.bit();
                }
            }
            return Self(set);
        }
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
        {
            // Many none-elf targets enable m/a/f/d/c in the base ISA string without
            // exposing every letter as target_feature; assume gc floor when no
            // letter fired but we are on a typical imac/gc-style target.
            let set = Self::compile_time();
            if set.0 == 0 {
                return Self(
                    1 << Feature::M.bit()
                        | 1 << Feature::A.bit()
                        | 1 << Feature::F.bit()
                        | 1 << Feature::D.bit()
                        | 1 << Feature::C.bit(),
                );
            }
            set
        }
    }

    /// Bits the compile flags guarantee (`-C target-feature`, `target-cpu`).
    /// [`crate::riscv::detect_features`] unions these with the runtime probe,
    /// so a `+v` build dispatches to RVV without a runtime branch miss.
    #[inline]
    pub(crate) fn compile_time() -> Self {
        {
            #[allow(unused_mut, reason = "mutated under target_feature cfgs")]
            let mut set = 0u128;
            #[cfg(target_feature = "m")]
            {
                set |= 1 << Feature::M.bit();
            }
            #[cfg(target_feature = "a")]
            {
                set |= 1 << Feature::A.bit();
            }
            #[cfg(target_feature = "f")]
            {
                set |= 1 << Feature::F.bit();
            }
            #[cfg(target_feature = "d")]
            {
                set |= 1 << Feature::D.bit();
            }
            #[cfg(target_feature = "c")]
            {
                set |= 1 << Feature::C.bit();
            }
            #[cfg(target_feature = "v")]
            {
                set |= 1 << Feature::V.bit();
            }
            #[cfg(all(target_feature = "v", target_feature = "zvfh"))]
            {
                set |= 1 << Feature::Zvfh.bit();
            }
            #[cfg(all(target_feature = "v", target_feature = "zvfbfmin"))]
            {
                set |= 1 << Feature::Zvfbfmin.bit();
            }
            #[cfg(all(target_feature = "v", target_feature = "zvfbfwma"))]
            {
                set |= 1 << Feature::Zvfbfwma.bit();
            }
            Self(set)
        }
    }

    /// Features present in **either** set (bitwise OR).
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

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
