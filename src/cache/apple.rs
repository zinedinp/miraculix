//! macOS / iOS: `sysctlbyname`.
//!
//! macOS 12+ / iOS 15+ describe each performance level (`hw.perflevel0` =
//! P cores, `hw.perflevel1` = E cores) with its own L1/L2 and the CPUs per
//! L2. Older systems have only the flat `hw.l1dcachesize` ... plus
//! `hw.cacheconfig` (CPUs sharing each level). No associativity is reported.

use super::{Cache, CacheMatrix, CoreClass, Level, Source};

pub(super) fn probe() -> Option<CacheMatrix> {
    let line = sysctl(b"hw.cachelinesize").unwrap_or(0);
    let mut m = CacheMatrix::with_source(Source::Sysctl);
    match sysctl(b"hw.nperflevels") {
        Some(n) if n > 0 => {
            for level in 0..n.min(8) {
                perflevel(&mut m, level as u8, line);
            }
        }
        _ => flat(&mut m, line),
    }
    (!m.is_empty()).then_some(m)
}

fn perflevel(m: &mut CacheMatrix, level: u8, line: u64) {
    let get = |key: &[u8]| {
        let mut name = Name::default();
        name.push(b"hw.perflevel");
        name.push(&[b'0' + level]);
        name.push(b".");
        name.push(key);
        sysctl(name.as_bytes()).unwrap_or(0)
    };
    let logical = get(b"logicalcpu");
    let physical = get(b"physicalcpu");
    let smt = logical.checked_div(physical).unwrap_or(0);
    let cache = |size: u64, shared: u64| make(size, line, shared);
    let class = CoreClass::new(
        cache(get(b"l1dcachesize"), smt),
        cache(get(b"l1icachesize"), smt),
        cache(get(b"l2cachesize"), get(b"cpusperl2")),
        cache(get(b"l3cachesize"), get(b"cpusperl3")),
    );
    push(m, class, logical);
}

fn flat(m: &mut CacheMatrix, line: u64) {
    // `hw.cacheconfig`: CPUs sharing [memory, L1, L2, L3, ...], u64 each.
    let mut config = [0u64; 8];
    let mut len = size_of_val(&config);
    // SAFETY: `config` is `len` writable bytes; the name is NUL-terminated.
    let ok = unsafe {
        libc::sysctlbyname(
            c"hw.cacheconfig".as_ptr(),
            config.as_mut_ptr().cast(),
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    } == 0;
    let shared = |i: usize| {
        if ok && len >= (i + 1) * 8 {
            config[i]
        } else {
            0
        }
    };
    let class = CoreClass::new(
        make(sysctl(b"hw.l1dcachesize").unwrap_or(0), line, shared(1)),
        make(sysctl(b"hw.l1icachesize").unwrap_or(0), line, shared(1)),
        make(sysctl(b"hw.l2cachesize").unwrap_or(0), line, shared(2)),
        make(sysctl(b"hw.l3cachesize").unwrap_or(0), line, shared(3)),
    );
    push(m, class, sysctl(b"hw.logicalcpu").unwrap_or(0));
}

fn push(m: &mut CacheMatrix, class: CoreClass, cpus: u64) {
    if Level::ALL.iter().any(|&l| class.get(l).is_known()) {
        m.push(class, u16::try_from(cpus).unwrap_or(u16::MAX));
    }
}

fn make(size: u64, line: u64, shared: u64) -> Cache {
    if size == 0 {
        return Cache::UNKNOWN;
    }
    Cache {
        size: u32::try_from(size).unwrap_or(u32::MAX),
        line: u16::try_from(line).unwrap_or(0),
        ways: 0,
        shared_by: u16::try_from(shared).unwrap_or(u16::MAX),
    }
}

/// An integer sysctl (`int32` or `int64`); `None` if missing.
fn sysctl(name: &[u8]) -> Option<u64> {
    let mut n = Name::default();
    n.push(name);
    let mut v = [0u8; 8];
    let mut len = v.len();
    // SAFETY: `v` is `len` writable bytes; `n` is NUL-terminated.
    let r = unsafe {
        libc::sysctlbyname(
            n.as_ptr(),
            v.as_mut_ptr().cast(),
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    match (r, len) {
        (0, 4) => Some(u64::from(u32::from_ne_bytes([v[0], v[1], v[2], v[3]]))),
        (0, 8) => Some(u64::from_ne_bytes(v)),
        _ => None,
    }
}

/// A NUL-terminated sysctl name in a stack buffer (the longest used,
/// `hw.perflevel0.l1dcachesize`, is 26 bytes).
#[derive(Default)]
struct Name {
    buf: [u8; 32],
    len: usize,
}

impl Name {
    fn push(&mut self, s: &[u8]) {
        let end = (self.len + s.len()).min(self.buf.len() - 1);
        self.buf[self.len..end].copy_from_slice(&s[..end - self.len]);
        self.len = end;
    }

    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    fn as_ptr(&self) -> *const libc::c_char {
        self.buf.as_ptr().cast()
    }
}
