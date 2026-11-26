//! x86 `CPUID`: leaf 4 (Intel) or `0x8000_001D` (AMD, topology extensions),
//! else AMD's legacy `0x8000_0005` / `0x8000_0006`.
//!
//! `CPUID` answers for the core it runs on, so this is one row. On a hybrid
//! CPU (leaf 7 EDX bit 15) the other class is missing and the matrix is marked
//! incomplete. `shared_by` is the "maximum logical processors sharing this
//! cache" field: an upper bound (it counts SMT threads even when SMT is off).

use super::{Cache, CacheMatrix, CoreClass, Source};
use crate::x86::arch::{__cpuid, __cpuid_count};
use raw_cpuid::{Associativity, CacheType, CpuId, CpuIdResult};

pub(super) fn probe() -> Option<CacheMatrix> {
    // 486-class CPUs (and some 32-bit emulators) have no CPUID at all.
    #[cfg(target_arch = "x86")]
    if !crate::x86::features::has_cpuid() {
        return None;
    }
    fn reader(leaf: u32, sub: u32) -> CpuIdResult {
        let r = __cpuid_count(leaf, sub);
        CpuIdResult {
            eax: r.eax,
            ebx: r.ebx,
            ecx: r.ecx,
            edx: r.edx,
        }
    }
    let cpuid = CpuId::with_cpuid_reader(reader as fn(u32, u32) -> CpuIdResult);

    let mut class = CoreClass::default();
    if let Some(params) = cpuid.get_cache_parameters() {
        for p in params {
            let kind = match p.cache_type() {
                CacheType::Data => super::parse::Kind::Data,
                CacheType::Instruction => super::parse::Kind::Instruction,
                CacheType::Unified => super::parse::Kind::Unified,
                _ => continue,
            };
            let size = p.associativity()
                * p.physical_line_partitions()
                * p.coherency_line_size()
                * p.sets();
            let cache = Cache {
                size: u32::try_from(size).unwrap_or(u32::MAX),
                line: u16::try_from(p.coherency_line_size()).unwrap_or(0),
                ways: if p.is_fully_associative() {
                    Cache::FULLY_ASSOCIATIVE
                } else {
                    u16::try_from(p.associativity()).unwrap_or(0)
                },
                shared_by: u16::try_from(p.max_cores_for_cache()).unwrap_or(u16::MAX),
            };
            let (a, b) = super::parse::columns(u32::from(p.level()), kind);
            for level in [a, b].into_iter().flatten() {
                class.caches[level as usize] = cache;
            }
        }
    }
    if !class.caches.iter().any(|c| c.is_known()) {
        amd_legacy(&cpuid, &mut class);
    }
    if !class.caches.iter().any(|c| c.is_known()) {
        return None;
    }
    let mut m = CacheMatrix::with_source(Source::Cpuid);
    m.push(class, 0);
    m.partial = is_hybrid();
    Some(m)
}

/// K8 .. Bulldozer: sizes in KiB (L3 in 512 KiB units), no sharing info.
fn amd_legacy(cpuid: &CpuId<impl raw_cpuid::CpuIdReader>, class: &mut CoreClass) {
    use super::Level;
    let ways = |a: Associativity| match a {
        Associativity::DirectMapped => 1,
        Associativity::NWay(n) => u16::from(n),
        Associativity::FullyAssociative => Cache::FULLY_ASSOCIATIVE,
        _ => 0,
    };
    if let Some(l1) = cpuid.get_l1_cache_and_tlb_info() {
        class.caches[Level::L1d as usize] = Cache::new(
            u32::from(l1.dcache_size()) << 10,
            u16::from(l1.dcache_line_size()),
            ways(l1.dcache_associativity()),
            0,
        );
        class.caches[Level::L1i as usize] = Cache::new(
            u32::from(l1.icache_size()) << 10,
            u16::from(l1.icache_line_size()),
            ways(l1.icache_associativity()),
            0,
        );
    }
    if let Some(l23) = cpuid.get_l2_l3_cache_and_tlb_info() {
        class.caches[Level::L2 as usize] = Cache::new(
            u32::from(l23.l2cache_size()) << 10,
            u16::from(l23.l2cache_line_size()),
            ways(l23.l2cache_associativity()),
            0,
        );
        class.caches[Level::L3 as usize] = Cache::new(
            u32::from(l23.l3cache_size()) << 19,
            u16::from(l23.l3cache_line_size()),
            ways(l23.l3cache_associativity()),
            0,
        );
    }
}

/// Leaf 7 EDX bit 15 (Intel hybrid). Leaf 7 is read only when it exists:
/// Intel answers a leaf above the maximum with the highest leaf's data.
fn is_hybrid() -> bool {
    __cpuid(0).eax >= 7 && __cpuid_count(7, 0).edx >> 15 & 1 != 0
}

#[cfg(test)]
mod tests {
    use super::super::{Level, Source, detect_fresh};

    /// On a symmetric machine the OS and `CPUID` describe the same caches
    /// (the OS reads them from `CPUID` itself). `shared_by` may differ:
    /// `CPUID` gives an upper bound.
    #[test]
    fn cpuid_agrees_with_the_os() {
        let (Some(c), os) = (super::probe(), detect_fresh()) else {
            return;
        };
        if !matches!(os.source(), Source::Windows | Source::Sysfs) || !os.is_symmetric() {
            return;
        }
        let (c, os) = (c.classes()[0], os.classes()[0]);
        for level in Level::ALL {
            let (a, b) = (c.get(level), os.get(level));
            if a.is_known() && b.is_known() {
                assert_eq!((a.size, a.line), (b.size, b.line), "{level:?}");
            }
        }
    }
}
