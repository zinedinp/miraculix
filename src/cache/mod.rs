//! Cache sizes (L1d, L1i, L2, L3) per core class, for sizing tables and
//! blocks: Huffman decode tables, hash tables, GEMM tiles.
//!
//! Like the SIMD feature probes, [`detect`] asks the OS once and caches the
//! answer process-wide; [`detect_fresh`] probes again. The result is a
//! [`CacheMatrix`]: one row ([`CoreClass`]) per distinct cache configuration,
//! one column per [`Level`]. A symmetric CPU has one row. Asymmetric ones have
//! more:
//!
//! - Intel hybrid (Alder Lake and later): P-cores (private L2) and E-cores
//!   (one L2 per 4-core cluster).
//! - Arm big.LITTLE / DynamIQ, Apple P and E clusters.
//! - AMD X3D parts where only one CCD has the stacked L3.
//!
//! | Target | Source | Rows |
//! |---|---|---|
//! | Linux / Android, any arch | `/sys/devices/system/cpu/cpuN/cache/indexM/` | every online CPU |
//! | Windows | `GetLogicalProcessorInformationEx(RelationCache)` | every CPU |
//! | macOS / iOS | `sysctlbyname("hw.perflevelN.*")` | one per performance level |
//! | x86, fallback or no OS | `CPUID` leaf 4 / `0x8000_001D` / AMD `0x8000_0005..6` | the current core only |
//! | PowerPC Linux without sysfs | auxv `AT_L1I/L1D/L2/L3_CACHE*` | one |
//! | wasm, other bare metal | - | none ([`CacheMatrix::EMPTY`]) |
//!
//! `CPUID` describes only the core it runs on, so on a hybrid x86 without an
//! OS source the matrix says so: [`CacheMatrix::is_complete`] is `false`.
//!
//! # Sizing a table
//!
//! A thread may migrate between core classes, so size for the smallest one
//! unless you pin threads. [`table_bits`] turns a byte budget into a power of
//! two:
//!
//! ```rust
//! use miraculix::cache::{self, Level};
//!
//! let l1d = cache::detect().min_per_thread(Level::L1d).unwrap_or(32 * 1024);
//! // Leave half of L1d to the input and output streams; 4-byte entries.
//! let bits = cache::table_bits(l1d / 2, 4).clamp(9, 13);
//! assert!((9..=13).contains(&bits));
//! ```
//!
//! # Cost
//!
//! The first [`detect`] reads about 24 small sysfs files per CPU on Linux
//! (a few ms on a 256-thread server), one API call on Windows, ~10 sysctls
//! per performance level on Apple. Later calls are one atomic load. Call
//! [`warm_up`] at startup to move the probe out of a latency-sensitive path.

use core::fmt;

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod cpuid;
mod parse;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod sysfs;
#[cfg(target_os = "windows")]
mod win;

/// Maximum number of distinct core classes a [`CacheMatrix`] stores. Phones
/// have up to four (e.g. Cortex-X3 + A715 + A710 + A510); more are counted in
/// the totals and set [`CacheMatrix::is_complete`] to `false`.
pub const MAX_CLASSES: usize = 8;

/// A column of the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    /// Level-1 data cache (a unified L1 fills both L1 columns).
    L1d = 0,
    /// Level-1 instruction cache.
    L1i = 1,
    /// Level-2 (data or unified).
    L2 = 2,
    /// Level-3 (data or unified). A system-level cache that the OS does not
    /// report as L3 (Apple SLC, many Arm SoCs) is not here.
    L3 = 3,
}

impl Level {
    /// Every level, nearest first.
    pub const ALL: [Level; 4] = [Level::L1d, Level::L1i, Level::L2, Level::L3];

    /// `"L1d"`, `"L1i"`, `"L2"` or `"L3"`.
    pub const fn name(self) -> &'static str {
        match self {
            Level::L1d => "L1d",
            Level::L1i => "L1i",
            Level::L2 => "L2",
            Level::L3 => "L3",
        }
    }
}

/// One cache as seen from one core. `0` in any field means "not reported".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Cache {
    /// Size of one instance in bytes.
    pub size: u32,
    /// Line size in bytes.
    pub line: u16,
    /// Associativity; [`Cache::FULLY_ASSOCIATIVE`] for fully associative.
    pub ways: u16,
    /// Logical CPUs sharing this instance (SMT siblings for L1, a cluster or
    /// CCD for L2/L3). From `CPUID` alone it is the maximum the core could
    /// share with, not the number of threads actually enabled.
    pub shared_by: u16,
}

impl Cache {
    /// Nothing reported.
    pub const UNKNOWN: Cache = Cache::new(0, 0, 0, 0);

    /// [`Cache::ways`] of a fully associative cache.
    pub const FULLY_ASSOCIATIVE: u16 = u16::MAX;

    /// A cache with the given fields.
    pub const fn new(size: u32, line: u16, ways: u16, shared_by: u16) -> Self {
        Cache {
            size,
            line,
            ways,
            shared_by,
        }
    }

    /// The size is known.
    pub const fn is_known(self) -> bool {
        self.size != 0
    }

    /// Size divided by [`Cache::shared_by`]: the share one thread gets when
    /// every sharer is busy. Equals `size` when sharing is unknown.
    pub const fn per_thread(self) -> u32 {
        if self.shared_by > 1 {
            self.size / self.shared_by as u32
        } else {
            self.size
        }
    }

    /// Number of sets, when size, line and a finite associativity are known.
    /// Addresses `sets * line` bytes apart compete for the same set.
    pub const fn sets(self) -> Option<u32> {
        if self.size == 0 || self.line == 0 || self.ways == 0 {
            return None;
        }
        if self.ways == Self::FULLY_ASSOCIATIVE {
            return Some(1);
        }
        let way_bytes = self.line as u32 * self.ways as u32;
        if way_bytes == 0 || !self.size.is_multiple_of(way_bytes) {
            return None;
        }
        Some(self.size / way_bytes)
    }
}

/// One row: the caches a set of cores sees, and how many logical CPUs have
/// exactly this configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CoreClass {
    caches: [Cache; 4],
    cpus: u16,
}

impl CoreClass {
    /// A row with these caches and no CPUs yet (see [`CacheMatrix::push`]).
    pub const fn new(l1d: Cache, l1i: Cache, l2: Cache, l3: Cache) -> Self {
        CoreClass {
            caches: [l1d, l1i, l2, l3],
            cpus: 0,
        }
    }

    /// The cache in column `level`.
    pub const fn get(&self, level: Level) -> Cache {
        self.caches[level as usize]
    }

    /// Logical CPUs in this class; `0` when unknown (`CPUID` only).
    pub const fn cpus(&self) -> u16 {
        self.cpus
    }
}

/// Where a [`CacheMatrix`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Source {
    /// Nothing reported anything.
    None,
    /// Built by the caller with [`CacheMatrix::push`].
    Manual,
    /// Linux / Android sysfs.
    Sysfs,
    /// `GetLogicalProcessorInformationEx`.
    Windows,
    /// Apple `sysctlbyname`.
    Sysctl,
    /// x86 `CPUID` on the calling core.
    Cpuid,
    /// PowerPC auxv.
    Auxv,
}

/// Cache sizes per core class: rows are [`CoreClass`]es, columns [`Level`]s.
///
/// Rows come in detection order (by CPU number, or Apple performance level
/// 0 first); the order does not say which class is faster. Identical rows are
/// merged, so a symmetric CPU has exactly one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CacheMatrix {
    rows: [CoreClass; MAX_CLASSES],
    len: u8,
    source: Source,
    truncated: bool,
    partial: bool,
    cpus: u32,
    min_size: [u32; 4],
    max_size: [u32; 4],
    min_per_thread: [u32; 4],
    max_line: u16,
}

impl CacheMatrix {
    /// No rows: nothing is known.
    pub const EMPTY: CacheMatrix = CacheMatrix::with_source(Source::None);

    /// An empty matrix to fill with [`CacheMatrix::push`], e.g. with a
    /// datasheet's numbers on bare metal.
    pub const fn new() -> Self {
        Self::with_source(Source::Manual)
    }

    const fn with_source(source: Source) -> Self {
        CacheMatrix {
            rows: [CoreClass::new(
                Cache::UNKNOWN,
                Cache::UNKNOWN,
                Cache::UNKNOWN,
                Cache::UNKNOWN,
            ); MAX_CLASSES],
            len: 0,
            source,
            truncated: false,
            partial: false,
            cpus: 0,
            min_size: [0; 4],
            max_size: [0; 4],
            min_per_thread: [0; 4],
            max_line: 0,
        }
    }

    /// Adds `cpus` logical CPUs with `class`'s caches: merged into an
    /// identical row, else a new row. Past [`MAX_CLASSES`] rows the CPUs still
    /// count in the totals and min/max, and [`Self::is_complete`] turns false.
    pub fn push(&mut self, class: CoreClass, cpus: u16) {
        for (i, c) in class.caches.iter().enumerate() {
            if !c.is_known() {
                continue;
            }
            let pt = c.per_thread();
            if self.min_size[i] == 0 || c.size < self.min_size[i] {
                self.min_size[i] = c.size;
            }
            self.max_size[i] = self.max_size[i].max(c.size);
            if self.min_per_thread[i] == 0 || pt < self.min_per_thread[i] {
                self.min_per_thread[i] = pt;
            }
            self.max_line = self.max_line.max(c.line);
        }
        self.cpus = self.cpus.saturating_add(u32::from(cpus));
        let len = usize::from(self.len);
        if let Some(row) = self.rows[..len]
            .iter_mut()
            .find(|r| r.caches == class.caches)
        {
            row.cpus = row.cpus.saturating_add(cpus);
        } else if len < MAX_CLASSES {
            self.rows[len] = CoreClass {
                caches: class.caches,
                cpus,
            };
            self.len += 1;
        } else {
            self.truncated = true;
        }
    }

    /// The rows.
    pub fn classes(&self) -> &[CoreClass] {
        &self.rows[..usize::from(self.len)]
    }

    /// No rows at all.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// At most one row: every core sees the same caches.
    pub const fn is_symmetric(&self) -> bool {
        self.len <= 1
    }

    /// `false` when rows may be missing: more than [`MAX_CLASSES`] classes,
    /// or `CPUID` saw one core of a hybrid x86 CPU. Min/max still cover every
    /// CPU in the first case, but only the calling core in the second.
    pub const fn is_complete(&self) -> bool {
        !self.truncated && !self.partial
    }

    /// Where the numbers came from.
    pub const fn source(&self) -> Source {
        self.source
    }

    /// Logical CPUs counted over all rows; `0` when unknown (`CPUID` only).
    pub const fn cpus(&self) -> u32 {
        self.cpus
    }

    /// Smallest instance size of `level` over all classes.
    pub const fn min_size(&self, level: Level) -> Option<u32> {
        nonzero(self.min_size[level as usize])
    }

    /// Largest instance size of `level` over all classes.
    pub const fn max_size(&self, level: Level) -> Option<u32> {
        nonzero(self.max_size[level as usize])
    }

    /// Smallest [`Cache::per_thread`] share of `level` over all classes: the
    /// budget that fits wherever a thread runs, even with every SMT sibling
    /// and cluster neighbour busy.
    pub const fn min_per_thread(&self, level: Level) -> Option<u32> {
        nonzero(self.min_per_thread[level as usize])
    }

    /// Largest line size of any cache in any class: the padding unit that
    /// keeps two values off one line everywhere.
    pub const fn line_size(&self) -> Option<u16> {
        if self.max_line == 0 {
            None
        } else {
            Some(self.max_line)
        }
    }
}

impl Default for CacheMatrix {
    fn default() -> Self {
        Self::new()
    }
}

const fn nonzero(v: u32) -> Option<u32> {
    if v == 0 { None } else { Some(v) }
}

impl fmt::Debug for CacheMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheMatrix")
            .field("source", &self.source)
            .field("complete", &self.is_complete())
            .field("cpus", &self.cpus)
            .field("classes", &self.classes())
            .finish()
    }
}

/// A table, one row per class:
///
/// ```text
/// cache matrix: Windows, 8 CPUs, 2 classes
/// class  cpus  L1d               L1i               L2                L3
/// 0         4  48K 12w 64B x2    32K 8w 64B x2     2M 16w 64B x2     12M 12w 64B x8
/// 1         4  32K 8w 64B        64K 8w 64B        4M 16w 64B x4     12M 12w 64B x8
/// ```
///
/// `x2` is [`Cache::shared_by`] (left out when 1); `-` is unknown.
impl fmt::Display for CacheMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cache matrix: {:?}", self.source)?;
        if self.cpus != 0 {
            write!(f, ", {} CPUs", self.cpus)?;
        }
        let n = self.classes().len();
        write!(f, ", {n} class{}", if n == 1 { "" } else { "es" })?;
        if !self.is_complete() {
            f.write_str(" (incomplete)")?;
        }
        write!(f, "\nclass  cpus")?;
        for level in Level::ALL {
            let w = if level == Level::L3 { 0 } else { 16 };
            write!(f, "  {:<w$}", level.name())?;
        }
        for (i, row) in self.classes().iter().enumerate() {
            write!(f, "\n{i:<5}  {:>4}", row.cpus)?;
            for level in Level::ALL {
                let mut cell = Cell::default();
                write_cache(&mut cell, row.get(level))?;
                let w = if level == Level::L3 { 0 } else { 16 };
                write!(f, "  {:<w$}", cell.as_str())?;
            }
        }
        Ok(())
    }
}

fn write_cache(w: &mut impl fmt::Write, c: Cache) -> fmt::Result {
    if !c.is_known() {
        return w.write_char('-');
    }
    match c.size {
        s if s >= 1 << 20 && s.is_multiple_of(1 << 20) => write!(w, "{}M", s >> 20)?,
        s if s >= 1 << 10 && s.is_multiple_of(1 << 10) => write!(w, "{}K", s >> 10)?,
        s => write!(w, "{s}B")?,
    }
    match c.ways {
        0 => {}
        Cache::FULLY_ASSOCIATIVE => w.write_str(" full")?,
        n => write!(w, " {n}w")?,
    }
    if c.line != 0 {
        write!(w, " {}B", c.line)?;
    }
    if c.shared_by > 1 {
        write!(w, " x{}", c.shared_by)?;
    }
    Ok(())
}

/// Fixed buffer for one table cell (no `alloc`).
#[derive(Default)]
struct Cell {
    buf: [u8; 32],
    len: usize,
}

impl Cell {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("?")
    }
}

impl fmt::Write for Cell {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        self.buf
            .get_mut(self.len..end)
            .ok_or(fmt::Error)?
            .copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// Largest `bits` such that a table of `1 << bits` entries of `entry_bytes`
/// each fits in `budget` bytes. `0` when at most one entry fits.
///
/// ```rust
/// use miraculix::cache::table_bits;
/// assert_eq!(table_bits(16 * 1024, 4), 12); // 4096 x 4 B = 16 KiB
/// assert_eq!(table_bits(16 * 1024 - 1, 4), 11);
/// assert_eq!(table_bits(3, 4), 0);
/// ```
pub const fn table_bits(budget: u32, entry_bytes: u32) -> u32 {
    if entry_bytes == 0 || budget < entry_bytes {
        return 0;
    }
    (budget / entry_bytes).ilog2()
}

/// The cache matrix, probed on the first call and cached process-wide.
///
/// Racing first callers may each probe; one result is kept. On targets
/// without compare-and-swap (`thumbv6m`, `riscv32i`: all bare metal, where no
/// source exists) this is [`CacheMatrix::EMPTY`].
pub fn detect() -> &'static CacheMatrix {
    #[cfg(target_has_atomic = "8")]
    {
        static SLOT: once::Slot = once::Slot::new();
        SLOT.get_or_init(detect_fresh)
    }
    #[cfg(not(target_has_atomic = "8"))]
    {
        static EMPTY: CacheMatrix = CacheMatrix::EMPTY;
        &EMPTY
    }
}

/// Fills the [`detect`] cache now. Safe to call any number of times.
pub fn warm_up() {
    let _ = detect();
}

/// Probes again, ignoring the cache (tests, diagnostics, CPU hotplug).
pub fn detect_fresh() -> CacheMatrix {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Some(m) = sysfs::probe() {
        return m;
    }
    #[cfg(target_os = "windows")]
    if let Some(m) = win::probe() {
        return m;
    }
    #[cfg(target_vendor = "apple")]
    if let Some(m) = apple::probe() {
        return m;
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "android"),
        any(target_arch = "powerpc", target_arch = "powerpc64")
    ))]
    if let Some(m) = sysfs::probe_powerpc_auxv() {
        return m;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(m) = cpuid::probe() {
        return m;
    }
    CacheMatrix::EMPTY
}

#[cfg(target_has_atomic = "8")]
mod once {
    use super::CacheMatrix;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicU8, Ordering};

    const EMPTY: u8 = 0;
    const BUSY: u8 = 1;
    const READY: u8 = 2;

    /// Write-once cell. The value is probed *before* claiming the slot, so a
    /// waiter only spins while the winner copies it in (no probe, no panic).
    pub(super) struct Slot {
        state: AtomicU8,
        value: UnsafeCell<CacheMatrix>,
    }

    // SAFETY: `value` is written only by the one thread that moved `state`
    // from EMPTY to BUSY, before it stores READY (Release). Readers access it
    // only after an Acquire load saw READY, and nobody writes after that.
    unsafe impl Sync for Slot {}

    impl Slot {
        pub(super) const fn new() -> Self {
            Slot {
                state: AtomicU8::new(EMPTY),
                value: UnsafeCell::new(CacheMatrix::EMPTY),
            }
        }

        pub(super) fn get_or_init(
            &'static self,
            init: impl FnOnce() -> CacheMatrix,
        ) -> &'static CacheMatrix {
            if self.state.load(Ordering::Acquire) != READY {
                let m = init();
                if self
                    .state
                    .compare_exchange(EMPTY, BUSY, Ordering::Acquire, Ordering::Acquire)
                    .is_ok()
                {
                    // SAFETY: we own the slot (EMPTY -> BUSY); no reader yet.
                    unsafe { *self.value.get() = m };
                    self.state.store(READY, Ordering::Release);
                } else {
                    while self.state.load(Ordering::Acquire) != READY {
                        core::hint::spin_loop();
                    }
                }
            }
            // SAFETY: READY was observed with Acquire; the value is final.
            unsafe { &*self.value.get() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::string::ToString;

    const fn kib(n: u32) -> u32 {
        n << 10
    }

    #[test]
    fn push_merges_identical_rows() {
        let big = CoreClass::new(
            Cache::new(kib(64), 64, 4, 1),
            Cache::new(kib(64), 64, 4, 1),
            Cache::new(kib(1024), 64, 8, 4),
            Cache::UNKNOWN,
        );
        let little = CoreClass::new(
            Cache::new(kib(32), 64, 4, 1),
            Cache::new(kib(32), 64, 4, 1),
            Cache::new(kib(512), 64, 8, 4),
            Cache::UNKNOWN,
        );
        let mut m = CacheMatrix::new();
        for _ in 0..4 {
            m.push(big, 1);
            m.push(little, 1);
        }
        assert_eq!(m.classes().len(), 2);
        assert_eq!(m.classes()[0].cpus(), 4);
        assert_eq!(m.cpus(), 8);
        assert!(!m.is_symmetric() && m.is_complete());
        assert_eq!(m.min_size(Level::L1d), Some(kib(32)));
        assert_eq!(m.max_size(Level::L1d), Some(kib(64)));
        assert_eq!(m.min_per_thread(Level::L2), Some(kib(128)));
        assert_eq!(m.min_size(Level::L3), None);
        assert_eq!(m.line_size(), Some(64));
    }

    #[test]
    fn overflow_keeps_totals() {
        let mut m = CacheMatrix::new();
        for i in 1..=(MAX_CLASSES as u32 + 1) {
            let l1 = Cache::new(kib(i), 64, 8, 1);
            m.push(CoreClass::new(l1, l1, Cache::UNKNOWN, Cache::UNKNOWN), 1);
        }
        assert_eq!(m.classes().len(), MAX_CLASSES);
        assert!(!m.is_complete());
        assert_eq!(m.cpus(), MAX_CLASSES as u32 + 1);
        assert_eq!(m.max_size(Level::L1d), Some(kib(MAX_CLASSES as u32 + 1)));
    }

    #[test]
    fn cache_helpers() {
        let c = Cache::new(kib(48), 64, 12, 2);
        assert_eq!(c.per_thread(), kib(24));
        assert_eq!(c.sets(), Some(64));
        assert_eq!(Cache::new(kib(32), 64, 0, 0).sets(), None);
        assert_eq!(
            Cache::new(kib(32), 64, Cache::FULLY_ASSOCIATIVE, 1).sets(),
            Some(1)
        );
        // Non-power-of-two way counts (AMD 0x8000_001D, Intel 10/12/20-way) still divide.
        assert_eq!(Cache::new(1280 * 1024, 64, 10, 1).sets(), Some(2048));
        assert_eq!(Cache::UNKNOWN.per_thread(), 0);
    }

    #[test]
    fn table_bits_rounds_down() {
        assert_eq!(table_bits(kib(16), 4), 12);
        assert_eq!(table_bits(kib(16) - 1, 4), 11);
        assert_eq!(table_bits(4, 4), 0);
        assert_eq!(table_bits(0, 4), 0);
        assert_eq!(table_bits(kib(16), 0), 0);
        assert_eq!(table_bits(u32::MAX, 1), 31);
    }

    #[test]
    fn display_is_a_table() {
        let mut m = CacheMatrix::new();
        m.push(
            CoreClass::new(
                Cache::new(kib(48), 64, 12, 2),
                Cache::new(kib(32), 64, Cache::FULLY_ASSOCIATIVE, 2),
                Cache::new(1280 * 1024, 64, 10, 2),
                Cache::UNKNOWN,
            ),
            16,
        );
        let s = m.to_string();
        assert!(
            s.starts_with("cache matrix: Manual, 16 CPUs, 1 class\n"),
            "{s}"
        );
        assert!(s.contains("48K 12w 64B x2"), "{s}");
        assert!(s.contains("32K full 64B x2"), "{s}");
        assert!(s.contains("1280K 10w 64B x2"), "{s}");
        assert!(s.lines().nth(2).unwrap().trim_end().ends_with('-'), "{s}");
    }

    #[test]
    fn empty_matrix() {
        let m = CacheMatrix::EMPTY;
        assert!(m.is_empty() && m.is_symmetric() && m.is_complete());
        assert_eq!(m.source(), Source::None);
        assert_eq!(m.min_per_thread(Level::L1d), None);
        assert_eq!(m.line_size(), None);
    }

    /// Whatever the host is, the probe must be self-consistent, and the
    /// cached and fresh answers must agree.
    #[test]
    fn host_probe_is_consistent() {
        let fresh = detect_fresh();
        assert_eq!(*detect(), fresh);
        std::println!("{fresh}");
        for row in fresh.classes() {
            for level in Level::ALL {
                let c = row.get(level);
                assert!(c.line == 0 || c.line.is_power_of_two(), "{c:?}");
                if c.is_known() {
                    assert!(c.size >= 1024, "{c:?}");
                }
            }
        }
        let l1d = fresh.min_size(Level::L1d);
        let l2 = fresh.max_size(Level::L2);
        if let (Some(a), Some(b)) = (l1d, l2) {
            assert!(a <= b, "L1d {a} > L2 {b}");
        }
        #[cfg(any(
            target_os = "windows",
            all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64"))
        ))]
        {
            assert!(
                !fresh.is_empty(),
                "no cache info on a desktop OS: {fresh:?}"
            );
            assert!(fresh.min_size(Level::L1d).is_some(), "{fresh:?}");
        }
    }

    #[test]
    fn concurrent_first_calls_agree() {
        let handles: std::vec::Vec<_> = (0..8).map(|_| std::thread::spawn(|| *detect())).collect();
        let first = *detect();
        for h in handles {
            assert_eq!(h.join().unwrap(), first);
        }
    }
}
