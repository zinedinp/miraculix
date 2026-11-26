//! Pure decoders for the OS cache reports: Linux sysfs text, the Windows
//! `GetLogicalProcessorInformationEx` buffer, PowerPC auxv words.
//!
//! No OS calls here, so every decoder is tested on every host.

use super::{Cache, CacheMatrix, CoreClass, Level, Source};

/// Which matrix columns a cache of `level` and `kind` fills. A unified L1
/// counts as both L1d and L1i; L4 and above are not columns.
pub(super) fn columns(level: u32, kind: Kind) -> (Option<Level>, Option<Level>) {
    match (level, kind) {
        (1, Kind::Data) => (Some(Level::L1d), None),
        (1, Kind::Instruction) => (Some(Level::L1i), None),
        (1, Kind::Unified) => (Some(Level::L1d), Some(Level::L1i)),
        (2, Kind::Data | Kind::Unified) => (Some(Level::L2), None),
        (3, Kind::Data | Kind::Unified) => (Some(Level::L3), None),
        _ => (None, None),
    }
}

/// What a cache holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Data,
    Instruction,
    Unified,
}

/// Unsigned decimal, surrounding whitespace allowed (sysfs ends with `\n`).
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code, reason = "sysfs decoder")
)]
pub(super) fn parse_u32(text: &[u8]) -> Option<u32> {
    core::str::from_utf8(text).ok()?.trim().parse().ok()
}

/// A sysfs cache `size`: `"32K"`, `"1024K"`, `"8M"`, or plain bytes.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code, reason = "sysfs decoder")
)]
pub(super) fn parse_size(text: &[u8]) -> Option<u32> {
    let s = core::str::from_utf8(text).ok()?.trim();
    let (digits, shift) = match s.as_bytes().last()? {
        b'K' | b'k' => (&s[..s.len() - 1], 10),
        b'M' | b'm' => (&s[..s.len() - 1], 20),
        b'G' | b'g' => (&s[..s.len() - 1], 30),
        _ => (s, 0),
    };
    let n: u32 = digits.parse().ok()?;
    n.checked_mul(1 << shift)
}

/// A sysfs cache `type`.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code, reason = "sysfs decoder")
)]
pub(super) fn parse_kind(text: &[u8]) -> Option<Kind> {
    match core::str::from_utf8(text).ok()?.trim() {
        "Data" => Some(Kind::Data),
        "Instruction" => Some(Kind::Instruction),
        "Unified" => Some(Kind::Unified),
        _ => None,
    }
}

/// Calls `f` for every CPU in a Linux CPU list (`"0-3,8,10-11"`). `false`
/// on a malformed list (`f` may have run for the part before the error).
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code, reason = "sysfs decoder")
)]
pub(super) fn for_each_cpu(text: &[u8], mut f: impl FnMut(u32)) -> bool {
    let Ok(s) = core::str::from_utf8(text) else {
        return false;
    };
    let s = s.trim();
    if s.is_empty() {
        return true;
    }
    for part in s.split(',') {
        let mut ends = part.split('-');
        let lo: Option<u32> = ends.next().and_then(|v| v.trim().parse().ok());
        let hi = match ends.next() {
            Some(h) => h.trim().parse().ok(),
            None => lo,
        };
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return false;
        };
        if ends.next().is_some() || hi < lo {
            return false;
        }
        (lo..=hi).for_each(&mut f);
    }
    true
}

/// Number of CPUs in a Linux CPU list.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", test)),
    allow(dead_code, reason = "sysfs decoder")
)]
pub(super) fn cpu_list_count(text: &[u8]) -> Option<u32> {
    let mut n = 0u32;
    for_each_cpu(text, |_| n = n.saturating_add(1)).then_some(n)
}

/// One cache from the PowerPC auxv pair `AT_*_CACHESIZE` /
/// `AT_*_CACHEGEOMETRY` (line size in bits 0..16, ways in bits 16..32).
#[cfg_attr(
    not(all(
        any(target_os = "linux", target_os = "android"),
        any(target_arch = "powerpc", target_arch = "powerpc64")
    )),
    allow(dead_code, reason = "PowerPC auxv decoder")
)]
pub(super) fn from_auxv(size: u64, geometry: u64) -> Cache {
    Cache {
        size: u32::try_from(size).unwrap_or(u32::MAX),
        line: (geometry & 0xffff) as u16,
        ways: ((geometry >> 16) & 0xffff) as u16,
        shared_by: 0,
    }
}

// ---- Windows ---------------------------------------------------------------

/// `RelationCache` in `LOGICAL_PROCESSOR_RELATIONSHIP`.
const RELATION_CACHE: u32 = 2;
/// Offset of `CACHE_RELATIONSHIP.GroupMask(s)` in a record.
const WIN_GROUPS_AT: usize = 40;
/// `CACHE_FULLY_ASSOCIATIVE`.
const WIN_FULLY_ASSOCIATIVE: u8 = 0xff;

/// One `CACHE_RELATIONSHIP` record of a `GetLogicalProcessorInformationEx`
/// buffer. Layout (all little-endian):
///
/// ```text
///  0 u32 Relationship   4 u32 Size
///  8 u8  Level          9 u8  Associativity  10 u16 LineSize
/// 12 u32 CacheSize     16 u32 Type (0 unified, 1 instruction, 2 data, 3 trace)
/// 20 [u8; 18] Reserved 38 u16 GroupCount (0 before Windows 11: one group)
/// 40 GROUP_AFFINITY[GroupCount] { KAFFINITY Mask; u16 Group; u16 Reserved[3] }
/// ```
#[derive(Clone, Copy)]
struct WinRecord<'a> {
    level: u32,
    kind: Kind,
    cache: Cache,
    groups: &'a [u8],
    /// `size_of::<KAFFINITY>()`: 8 on 64-bit Windows, 4 on 32-bit.
    mask_bytes: usize,
}

impl WinRecord<'_> {
    fn group_stride(&self) -> usize {
        self.mask_bytes + 8
    }

    /// `(group, mask)` of each `GROUP_AFFINITY`.
    fn masks(&self) -> impl Iterator<Item = (u16, u64)> + '_ {
        self.groups.chunks_exact(self.group_stride()).map(|g| {
            let mut m = [0u8; 8];
            m[..self.mask_bytes].copy_from_slice(&g[..self.mask_bytes]);
            let group = u16::from_le_bytes([g[self.mask_bytes], g[self.mask_bytes + 1]]);
            (group, u64::from_le_bytes(m))
        })
    }

    fn contains(&self, cpu: u32) -> bool {
        let bits = (self.mask_bytes * 8) as u32;
        let (group, bit) = (cpu / bits, cpu % bits);
        self.masks()
            .any(|(g, m)| u32::from(g) == group && m >> bit & 1 != 0)
    }

    /// Lowest CPU number in the record, numbered `group * mask bits + bit`.
    fn first_cpu(&self) -> Option<u32> {
        let bits = (self.mask_bytes * 8) as u32;
        self.masks()
            .filter(|&(_, m)| m != 0)
            .map(|(g, m)| u32::from(g) * bits + m.trailing_zeros())
            .min()
    }

    fn cpu_count(&self) -> u32 {
        self.masks().map(|(_, m)| m.count_ones()).sum()
    }
}

/// Every cache record in the buffer; stops at the first malformed one.
fn win_records(buf: &[u8], mask_bytes: usize) -> impl Iterator<Item = WinRecord<'_>> + '_ {
    let mut at = 0usize;
    core::iter::from_fn(move || {
        loop {
            let head = buf.get(at..at + 8)?;
            let relation = u32::from_le_bytes(head[0..4].try_into().ok()?);
            let size = u32::from_le_bytes(head[4..8].try_into().ok()?) as usize;
            if size < 8 {
                return None;
            }
            let rec = buf.get(at..at.checked_add(size)?)?;
            at += size;
            if relation != RELATION_CACHE || rec.len() < WIN_GROUPS_AT {
                continue;
            }
            let kind = match u32::from_le_bytes(rec[16..20].try_into().ok()?) {
                0 => Kind::Unified,
                1 => Kind::Instruction,
                2 => Kind::Data,
                _ => continue, // trace cache
            };
            let count = usize::from(u16::from_le_bytes([rec[38], rec[39]])).max(1);
            let stride = mask_bytes + 8;
            let groups = rec.get(WIN_GROUPS_AT..WIN_GROUPS_AT + count * stride)?;
            let ways = match rec[9] {
                WIN_FULLY_ASSOCIATIVE => Cache::FULLY_ASSOCIATIVE,
                w => u16::from(w),
            };
            let mut r = WinRecord {
                level: u32::from(rec[8]),
                kind,
                cache: Cache {
                    size: u32::from_le_bytes(rec[12..16].try_into().ok()?),
                    line: u16::from_le_bytes([rec[10], rec[11]]),
                    ways,
                    shared_by: 0,
                },
                groups,
                mask_bytes,
            };
            r.cache.shared_by = u16::try_from(r.cpu_count()).unwrap_or(u16::MAX);
            return Some(r);
        }
    })
}

/// Builds the matrix from a `GetLogicalProcessorInformationEx(RelationCache)`
/// buffer. Each L1 data (or unified L1) cache is one core: its first CPU picks
/// the L1i, L2 and L3 records, and the class counts all CPUs of that L1.
#[cfg_attr(
    not(any(target_os = "windows", test)),
    allow(dead_code, reason = "Windows decoder")
)]
pub(super) fn windows_matrix(buf: &[u8], mask_bytes: usize) -> Option<CacheMatrix> {
    let mut m = CacheMatrix::with_source(Source::Windows);
    for core in win_records(buf, mask_bytes) {
        if core.level != 1 || core.kind == Kind::Instruction {
            continue;
        }
        let Some(cpu) = core.first_cpu() else {
            continue;
        };
        let mut class = CoreClass::default();
        for r in win_records(buf, mask_bytes).filter(|r| r.contains(cpu)) {
            let (a, b) = columns(r.level, r.kind);
            for level in [a, b].into_iter().flatten() {
                class.caches[level as usize] = r.cache;
            }
        }
        m.push(class, u16::try_from(core.cpu_count()).unwrap_or(u16::MAX));
    }
    (!m.is_empty()).then_some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    #[test]
    fn sizes() {
        assert_eq!(parse_size(b"32K\n"), Some(32 * 1024));
        assert_eq!(parse_size(b"1280K"), Some(1280 * 1024));
        assert_eq!(parse_size(b"36M"), Some(36 << 20));
        assert_eq!(parse_size(b"512"), Some(512));
        assert_eq!(parse_size(b"8G"), None, "overflows u32");
        assert_eq!(parse_size(b""), None);
        assert_eq!(parse_size(b"K"), None);
        assert_eq!(parse_u32(b"64\n"), Some(64));
        assert_eq!(parse_kind(b"Instruction\n"), Some(Kind::Instruction));
        assert_eq!(parse_kind(b"Trace"), None);
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(cpu_list_count(b"0-3,8,10-11\n"), Some(7));
        assert_eq!(cpu_list_count(b"5"), Some(1));
        assert_eq!(cpu_list_count(b"\n"), Some(0));
        assert_eq!(cpu_list_count(b"3-1"), None);
        assert_eq!(cpu_list_count(b"1-2-3"), None);
        assert_eq!(cpu_list_count(b"a"), None);
        let mut seen = Vec::new();
        assert!(for_each_cpu(b"0-1,4", |c| seen.push(c)));
        assert_eq!(seen, [0, 1, 4]);
    }

    #[test]
    fn columns_map_unified_l1_to_both() {
        assert_eq!(
            columns(1, Kind::Unified),
            (Some(Level::L1d), Some(Level::L1i))
        );
        assert_eq!(columns(2, Kind::Unified), (Some(Level::L2), None));
        assert_eq!(columns(2, Kind::Instruction), (None, None));
        assert_eq!(columns(4, Kind::Unified), (None, None));
    }

    #[test]
    fn auxv_geometry() {
        let c = from_auxv(32768, 8 << 16 | 128);
        assert_eq!((c.size, c.line, c.ways), (32768, 128, 8));
    }

    /// One 64-bit `CACHE_RELATIONSHIP` record over the given `(group, mask)` pairs.
    fn win_rec(level: u8, kind: u32, size: u32, ways: u8, groups: &[(u16, u64)]) -> Vec<u8> {
        let mut r = Vec::new();
        r.extend_from_slice(&RELATION_CACHE.to_le_bytes());
        r.extend_from_slice(&0u32.to_le_bytes()); // size, patched below
        r.push(level);
        r.push(ways);
        r.extend_from_slice(&64u16.to_le_bytes());
        r.extend_from_slice(&size.to_le_bytes());
        r.extend_from_slice(&kind.to_le_bytes());
        r.extend_from_slice(&[0; 18]);
        r.extend_from_slice(&(groups.len() as u16).to_le_bytes());
        for &(g, m) in groups {
            r.extend_from_slice(&m.to_le_bytes());
            r.extend_from_slice(&g.to_le_bytes());
            r.extend_from_slice(&[0; 6]);
        }
        let len = r.len() as u32;
        r[4..8].copy_from_slice(&len.to_le_bytes());
        r
    }

    /// Intel hybrid shape: 2 P-cores with HT (CPUs 0-3, private 2 MiB L2) and
    /// 4 E-cores (CPUs 4-7, one shared 4 MiB L2), one 12 MiB L3 for all.
    fn hybrid_buffer() -> Vec<u8> {
        let mut b = Vec::new();
        for core in 0..2u64 {
            let mask = 0b11 << (core * 2);
            b.extend(win_rec(1, 2, 48 << 10, 12, &[(0, mask)]));
            b.extend(win_rec(1, 1, 32 << 10, 8, &[(0, mask)]));
            b.extend(win_rec(2, 0, 2 << 20, 16, &[(0, mask)]));
        }
        for cpu in 4..8u64 {
            b.extend(win_rec(1, 2, 32 << 10, 8, &[(0, 1 << cpu)]));
            b.extend(win_rec(1, 1, 64 << 10, 8, &[(0, 1 << cpu)]));
        }
        b.extend(win_rec(2, 0, 4 << 20, 16, &[(0, 0xf0)]));
        b.extend(win_rec(3, 0, 12 << 20, 12, &[(0, 0xff)]));
        // A non-cache record in between must be skipped.
        let mut other = win_rec(1, 2, 0, 0, &[(0, 1)]);
        other[0..4].copy_from_slice(&0u32.to_le_bytes());
        b.extend(other);
        b
    }

    #[test]
    fn windows_hybrid_gives_two_classes() {
        let m = windows_matrix(&hybrid_buffer(), 8).unwrap();
        assert_eq!(m.source(), Source::Windows);
        assert_eq!(m.classes().len(), 2);
        assert!(!m.is_symmetric());
        assert_eq!(m.cpus(), 8);
        let (p, e) = (&m.classes()[0], &m.classes()[1]);
        assert_eq!(p.cpus(), 4);
        assert_eq!(p.get(Level::L1d), Cache::new(48 << 10, 64, 12, 2));
        assert_eq!(p.get(Level::L2), Cache::new(2 << 20, 64, 16, 2));
        assert_eq!(e.cpus(), 4);
        assert_eq!(e.get(Level::L1i), Cache::new(64 << 10, 64, 8, 1));
        assert_eq!(e.get(Level::L2), Cache::new(4 << 20, 64, 16, 4));
        assert_eq!(e.get(Level::L3), p.get(Level::L3));
        assert_eq!(m.min_size(Level::L1d), Some(32 << 10));
        assert_eq!(m.max_size(Level::L2), Some(4 << 20));
        // P-core L1d per thread: 48K / 2 = 24K, below the E-core's 32K.
        assert_eq!(m.min_per_thread(Level::L1d), Some(24 << 10));
        assert_eq!(m.min_per_thread(Level::L2), Some(1 << 20));
    }

    #[test]
    fn windows_second_group_and_full_associativity() {
        // CPU 64 (group 1, bit 0) with a fully associative L1d.
        let mut b = win_rec(1, 2, 32 << 10, WIN_FULLY_ASSOCIATIVE, &[(1, 1)]);
        b.extend(win_rec(3, 0, 32 << 20, 16, &[(0, 0), (1, 1)]));
        let m = windows_matrix(&b, 8).unwrap();
        let c = m.classes()[0];
        assert_eq!(c.get(Level::L1d).ways, Cache::FULLY_ASSOCIATIVE);
        assert_eq!(c.get(Level::L3).size, 32 << 20);
        assert!(!c.get(Level::L2).is_known());
    }

    #[test]
    fn windows_truncated_buffer_is_rejected() {
        let b = hybrid_buffer();
        assert!(windows_matrix(&b[..20], 8).is_none());
        assert!(windows_matrix(&[], 8).is_none());
    }
}
