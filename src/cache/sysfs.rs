//! Linux / Android: `/sys/devices/system/cpu/cpuN/cache/indexM/`.
//!
//! Each `indexM` directory is one cache the CPU sees, with `level`, `type`,
//! `size`, `coherency_line_size`, `ways_of_associativity` and
//! `shared_cpu_list`. Offline CPUs have no `cache` directory and are skipped.
//! On Arm the files come from the device tree or ACPI PPTT, which may leave
//! out sizes; those caches stay unknown.

use super::parse::{columns, cpu_list_count, for_each_cpu, parse_kind, parse_size, parse_u32};
use super::{Cache, CacheMatrix, CoreClass, Source};

/// Upper bound on CPU numbers walked, against a corrupt `possible` list.
const MAX_CPU: u32 = 1 << 16;
/// `indexM` directories tried per CPU; real CPUs have 3 to 5.
const MAX_INDEX: u32 = 16;

pub(super) fn probe() -> Option<CacheMatrix> {
    let (mut possible, mut present) = ([0u8; 4096], [0u8; 4096]);
    let list = read(c"/sys/devices/system/cpu/possible", &mut possible)
        .or_else(|| read(c"/sys/devices/system/cpu/present", &mut present))?;
    let mut m = CacheMatrix::with_source(Source::Sysfs);
    let ok = for_each_cpu(list, |cpu| {
        if cpu < MAX_CPU
            && let Some(class) = cpu_class(cpu)
        {
            m.push(class, 1);
        }
    });
    (ok && !m.is_empty()).then_some(m)
}

/// The caches of one CPU, or `None` when it reports none.
fn cpu_class(cpu: u32) -> Option<CoreClass> {
    let mut class = CoreClass::default();
    let mut any = false;
    for index in 0..MAX_INDEX {
        let dir = Path::index_dir(cpu, index);
        let mut buf = [0u8; 64];
        let Some(level) = read(dir.file("level").as_cstr(), &mut buf).and_then(parse_u32) else {
            break;
        };
        let Some(kind) = read(dir.file("type").as_cstr(), &mut buf).and_then(parse_kind) else {
            continue;
        };
        let (a, b) = columns(level, kind);
        if a.is_none() {
            continue;
        }
        let field = |name: &str| {
            let mut buf = [0u8; 64];
            read(dir.file(name).as_cstr(), &mut buf).and_then(parse_u32)
        };
        let line = field("coherency_line_size").unwrap_or(0);
        let ways = field("ways_of_associativity").unwrap_or(0);
        let size = read(dir.file("size").as_cstr(), &mut buf)
            .and_then(parse_size)
            .unwrap_or(0);
        let mut list = [0u8; 1024];
        let shared = read(dir.file("shared_cpu_list").as_cstr(), &mut list)
            .and_then(cpu_list_count)
            .unwrap_or(0);
        let cache = Cache {
            size,
            line: u16::try_from(line).unwrap_or(0),
            ways: u16::try_from(ways).unwrap_or(Cache::FULLY_ASSOCIATIVE),
            shared_by: u16::try_from(shared).unwrap_or(u16::MAX),
        };
        for level in [a, b].into_iter().flatten() {
            class.caches[level as usize] = cache;
        }
        any = true;
    }
    any.then_some(class)
}

/// PowerPC fallback: auxv `AT_L1I_CACHESIZE` (40) .. `AT_L3_CACHEGEOMETRY`
/// (47), Linux 4.11+. One row, sharing unknown.
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
pub(super) fn probe_powerpc_auxv() -> Option<CacheMatrix> {
    use super::Level;
    use super::parse::from_auxv;
    let pairs = [
        (Level::L1d, 42, 43),
        (Level::L1i, 40, 41),
        (Level::L2, 44, 45),
        (Level::L3, 46, 47),
    ];
    let mut class = CoreClass::default();
    for (level, size, geo) in pairs {
        // SAFETY: read-only auxv lookups; 0 means "not present".
        let (s, g) = unsafe { (libc::getauxval(size), libc::getauxval(geo)) };
        #[allow(
            clippy::useless_conversion,
            reason = "`c_ulong` is `u32` on 32-bit PowerPC"
        )]
        let cache = from_auxv(u64::from(s), u64::from(g));
        class.caches[level as usize] = cache;
    }
    if class.caches.iter().all(|c| !c.is_known()) {
        return None;
    }
    let mut m = CacheMatrix::with_source(Source::Auxv);
    m.push(class, 0);
    Some(m)
}

/// Reads a small file into `buf`; `None` if it is missing or empty.
fn read<'a>(path: &core::ffi::CStr, buf: &'a mut [u8]) -> Option<&'a [u8]> {
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
    let n = usize::try_from(n).ok().filter(|&n| n > 0)?;
    Some(&buf[..n])
}

/// A NUL-terminated sysfs path in a stack buffer.
#[derive(Clone, Copy)]
struct Path {
    buf: [u8; 96],
    len: usize,
}

impl Path {
    /// `/sys/devices/system/cpu/cpu{cpu}/cache/index{index}/`.
    fn index_dir(cpu: u32, index: u32) -> Self {
        let mut p = Path {
            buf: [0; 96],
            len: 0,
        };
        p.push(b"/sys/devices/system/cpu/cpu");
        p.push_u32(cpu);
        p.push(b"/cache/index");
        p.push_u32(index);
        p.push(b"/");
        p
    }

    fn file(mut self, name: &str) -> Self {
        self.push(name.as_bytes());
        self
    }

    fn push(&mut self, s: &[u8]) {
        // Leave room for the NUL; the longest path is ~70 bytes.
        let end = (self.len + s.len()).min(self.buf.len() - 1);
        self.buf[self.len..end].copy_from_slice(&s[..end - self.len]);
        self.len = end;
    }

    fn push_u32(&mut self, mut v: u32) {
        let mut digits = [0u8; 10];
        let mut i = digits.len();
        loop {
            i -= 1;
            digits[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.push(&digits[i..]);
    }

    fn as_cstr(&self) -> &core::ffi::CStr {
        // `buf` is zeroed past `len` and `push` never writes the last byte.
        core::ffi::CStr::from_bytes_until_nul(&self.buf).unwrap_or(c"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths() {
        let p = Path::index_dir(127, 3).file("shared_cpu_list");
        assert_eq!(
            p.as_cstr().to_bytes(),
            b"/sys/devices/system/cpu/cpu127/cache/index3/shared_cpu_list"
        );
        assert_eq!(
            Path::index_dir(0, 0).as_cstr().to_bytes(),
            b"/sys/devices/system/cpu/cpu0/cache/index0/"
        );
    }
}
