//! Copies into and out of memory mapped write-combining: PCIe BARs of accelerator cards
//! (Tenstorrent TLB windows over L1 and GDDR, GPU apertures).
//!
//! | Tier | Store body | Load body | Needs |
//! |---|---|---|---|
//! | [`WcTier::Avx512`] | `vmovntdq` 64 B (one full write-combining line; usually one PCIe write, not guaranteed) | `vmovntdqa` 64 B | AVX-512F |
//! | [`WcTier::Avx2`] | `vmovntdq` 32 B | `vmovntdqa` 32 B | AVX2 |
//! | [`WcTier::Sse`] | `movntdq` 16 B | `movntdqa` 16 B | SSE4.1 (loads) |
//! | [`WcTier::Volatile`] | volatile `u32` | volatile `u32` | - |
//!
//! Head and tail bytes go through volatile `u32` accesses where the address is 4-aligned and
//! single bytes otherwise, so a device that takes 4-byte granular accesses sees no wider ones.
//! Every copy ends with one store fence (`sfence` on x86, `dmb oshst` on AArch64, `fence ow,ow`
//! on RISC-V) after **all** its stores, the volatile head and tail included: stores to
//! write-combining memory are weakly ordered even against later uncached stores (a doorbell, a
//! mailbox word). The streaming loads read a write-combining line once instead of one bus round
//! trip per access.
//!
//! Only for memory behind the mapping (L1, GDDR): device registers need the caller's own
//! dword-exact accesses.

/// Which instructions a copy uses; [`wc_tier`] is the best this CPU has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WcTier {
    /// Volatile `u32` accesses, any architecture.
    Volatile,
    /// 16-byte non-temporal stores and streaming loads (x86, SSE4.1).
    Sse,
    /// 32-byte non-temporal stores and streaming loads (x86, AVX2).
    Avx2,
    /// 64-byte non-temporal stores and streaming loads (x86, AVX-512F).
    Avx512,
}

impl WcTier {
    /// Whether this CPU can run the tier.
    pub fn available(self) -> bool {
        match self {
            WcTier::Volatile => true,
            #[cfg(target_arch = "x86_64")]
            WcTier::Sse => crate::x86::ops::sse::sse41::Sse41::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            WcTier::Avx2 => crate::x86::ops::avx::avx2::Avx2::detect().is_some(),
            #[cfg(target_arch = "x86_64")]
            WcTier::Avx512 => crate::x86::ops::avx512::avx512f::Avx512f::detect().is_some(),
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// The best [`WcTier`] this CPU has.
pub fn wc_tier() -> WcTier {
    [WcTier::Avx512, WcTier::Avx2, WcTier::Sse]
        .into_iter()
        .find(|t| t.available())
        .unwrap_or(WcTier::Volatile)
}

/// Copy `src` into write-combining memory at `dst` with the best tier.
///
/// # Safety
/// `dst` is valid for writes of `src.len()` bytes and does not overlap `src`.
pub unsafe fn copy_to_wc(dst: *mut u8, src: &[u8]) {
    // SAFETY: the caller's contract; `wc_tier` only returns available tiers.
    unsafe { copy_to_wc_with(wc_tier(), dst, src) }
}

/// Copy `dst.len()` bytes out of write-combining memory at `src` with the best tier.
///
/// # Safety
/// `src` is valid for reads of `dst.len()` bytes and does not overlap `dst`.
pub unsafe fn copy_from_wc(src: *const u8, dst: &mut [u8]) {
    // SAFETY: the caller's contract; `wc_tier` only returns available tiers.
    unsafe { copy_from_wc_with(wc_tier(), src, dst) }
}

/// [`copy_to_wc`] with a given tier.
///
/// # Safety
/// As [`copy_to_wc`].
///
/// # Panics
/// If the CPU lacks `tier` ([`WcTier::available`]).
pub unsafe fn copy_to_wc_with(tier: WcTier, dst: *mut u8, src: &[u8]) {
    assert!(tier.available(), "{tier:?} is not available on this CPU");
    let body = match tier {
        WcTier::Volatile => 0,
        WcTier::Sse | WcTier::Avx2 | WcTier::Avx512 => 16,
    };
    let head = if body == 0 {
        src.len()
    } else {
        head_len(dst as usize, body, src.len())
    };
    // SAFETY: `head <= src.len()`, inside the caller's range.
    unsafe { volatile_write(dst, &src[..head]) };
    let rest = &src[head..];
    #[cfg(target_arch = "x86_64")]
    let whole = rest.len() / 16 * 16;
    #[cfg(target_arch = "x86_64")]
    if whole > 0 {
        // SAFETY: `dst + head` is 16-aligned and writable for `whole` bytes; the tier is
        // available (asserted above).
        unsafe {
            let d = dst.add(head);
            match tier {
                WcTier::Avx512 => x86::store_512(d, &rest[..whole]),
                WcTier::Avx2 => x86::store_256(d, &rest[..whole]),
                _ => x86::store_128(d, &rest[..whole]),
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let whole = 0;
    // SAFETY: the tail lies inside the caller's range.
    unsafe { volatile_write(dst.add(head + whole), &rest[whole..]) };
    // Every store above (volatile head and tail included) went to
    // write-combining memory and is weakly ordered, so fence once after all
    // of them: a later doorbell or mailbox write must not overtake the data.
    wc_store_fence();
}

/// Orders every earlier store to write-combining / device memory before any
/// later store, as seen by the device.
#[inline]
fn wc_store_fence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier, no memory access of its own. `oshst`: the device
    // sits in the outer shareable domain, and only stores need ordering.
    unsafe {
        core::arch::asm!("dmb oshst", options(nostack, preserves_flags))
    };
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    // SAFETY: a fence, no memory access of its own. `ow, ow`: device-output
    // and memory writes before, the same after.
    unsafe {
        core::arch::asm!("fence ow, ow", options(nostack, preserves_flags))
    };
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64"
    )))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// [`copy_from_wc`] with a given tier.
///
/// # Safety
/// As [`copy_from_wc`].
///
/// # Panics
/// If the CPU lacks `tier` ([`WcTier::available`]).
pub unsafe fn copy_from_wc_with(tier: WcTier, src: *const u8, dst: &mut [u8]) {
    assert!(tier.available(), "{tier:?} is not available on this CPU");
    let body = match tier {
        WcTier::Volatile => 0,
        WcTier::Sse | WcTier::Avx2 | WcTier::Avx512 => 16,
    };
    let n = dst.len();
    let head = if body == 0 {
        n
    } else {
        head_len(src as usize, body, n)
    };
    // SAFETY: `head <= dst.len()`, inside the caller's range.
    unsafe { volatile_read(src, &mut dst[..head]) };
    #[cfg(target_arch = "x86_64")]
    let whole = (n - head) / 16 * 16;
    #[cfg(target_arch = "x86_64")]
    if whole > 0 {
        // SAFETY: `src + head` is 16-aligned and readable for `whole` bytes; the tier is
        // available (asserted above).
        unsafe {
            let s = src.add(head);
            let out = &mut dst[head..head + whole];
            match tier {
                WcTier::Avx512 => x86::load_512(s, out),
                WcTier::Avx2 => x86::load_256(s, out),
                _ => x86::load_128(s, out),
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let whole = 0;
    // SAFETY: the tail lies inside the caller's range.
    unsafe { volatile_read(src.add(head + whole), &mut dst[head + whole..]) };
}

/// Bytes before `addr` reaches `align`, at most `len`.
fn head_len(addr: usize, align: usize, len: usize) -> usize {
    ((align - addr % align) % align).min(len)
}

/// Volatile `u32` stores where `dst` is 4-aligned, single bytes elsewhere.
///
/// # Safety
/// `dst` is valid for writes of `src.len()` bytes.
unsafe fn volatile_write(dst: *mut u8, src: &[u8]) {
    let mut i = 0;
    while i < src.len() {
        // SAFETY: `i < src.len()`, inside the caller's range; the `u32` store only where the
        // address is 4-aligned and four bytes remain.
        unsafe {
            let at = dst.add(i);
            if (at as usize).is_multiple_of(4) && src.len() - i >= 4 {
                let v = u32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
                core::ptr::write_volatile(at.cast::<u32>(), v);
                i += 4;
            } else {
                core::ptr::write_volatile(at, src[i]);
                i += 1;
            }
        }
    }
}

/// Volatile `u32` loads where `src` is 4-aligned, single bytes elsewhere.
///
/// # Safety
/// `src` is valid for reads of `dst.len()` bytes.
unsafe fn volatile_read(src: *const u8, dst: &mut [u8]) {
    let mut i = 0;
    while i < dst.len() {
        // SAFETY: as in `volatile_write`.
        unsafe {
            let at = src.add(i);
            if (at as usize).is_multiple_of(4) && dst.len() - i >= 4 {
                let v = core::ptr::read_volatile(at.cast::<u32>());
                dst[i..i + 4].copy_from_slice(&v.to_le_bytes());
                i += 4;
            } else {
                dst[i] = core::ptr::read_volatile(at);
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    /// 16-byte stores until `dst` is `align`-aligned; returns the bytes done.
    ///
    /// # Safety
    /// `dst` is 16-aligned and writable for `src.len()` bytes (a multiple of 16); SSE2.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn store_head(dst: *mut u8, src: &[u8], align: usize) -> usize {
        let mut i = 0;
        // SAFETY: each 16-byte store lies inside `src.len()`; `dst + i` stays 16-aligned.
        unsafe {
            while !(dst.add(i) as usize).is_multiple_of(align) && i < src.len() {
                _mm_stream_si128(
                    dst.add(i).cast(),
                    _mm_loadu_si128(src.as_ptr().add(i).cast()),
                );
                i += 16;
            }
        }
        i
    }

    /// # Safety
    /// `dst` is 16-aligned and writable for `src.len()` bytes (a multiple of 16); SSE2.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn store_128(dst: *mut u8, src: &[u8]) {
        // SAFETY: as documented; every store is 16-aligned and inside `src.len()`.
        unsafe {
            let mut i = 0;
            while i < src.len() {
                _mm_stream_si128(
                    dst.add(i).cast(),
                    _mm_loadu_si128(src.as_ptr().add(i).cast()),
                );
                i += 16;
            }
        }
    }

    /// # Safety
    /// As [`store_128`], with AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn store_256(dst: *mut u8, src: &[u8]) {
        // SAFETY: as documented; after the head `dst + i` is 32-aligned.
        unsafe {
            let mut i = store_head(dst, src, 32);
            while i + 32 <= src.len() {
                let v = _mm256_loadu_si256(src.as_ptr().add(i).cast());
                _mm256_stream_si256(dst.add(i).cast(), v);
                i += 32;
            }
            while i < src.len() {
                _mm_stream_si128(
                    dst.add(i).cast(),
                    _mm_loadu_si128(src.as_ptr().add(i).cast()),
                );
                i += 16;
            }
        }
    }

    /// # Safety
    /// As [`store_128`], with AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn store_512(dst: *mut u8, src: &[u8]) {
        // SAFETY: as documented; after the head `dst + i` is 64-aligned.
        unsafe {
            let mut i = store_head(dst, src, 64);
            while i + 64 <= src.len() {
                let v = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                _mm512_stream_si512(dst.add(i).cast(), v);
                i += 64;
            }
            while i < src.len() {
                _mm_stream_si128(
                    dst.add(i).cast(),
                    _mm_loadu_si128(src.as_ptr().add(i).cast()),
                );
                i += 16;
            }
        }
    }

    /// 16-byte streaming loads until `src` is `align`-aligned; returns the bytes done.
    ///
    /// # Safety
    /// `src` is 16-aligned and readable for `dst.len()` bytes (a multiple of 16); SSE4.1.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn load_head(src: *const u8, dst: &mut [u8], align: usize) -> usize {
        let mut i = 0;
        // SAFETY: each 16-byte load lies inside `dst.len()`; `src + i` stays 16-aligned.
        unsafe {
            while !(src.add(i) as usize).is_multiple_of(align) && i < dst.len() {
                let v = _mm_stream_load_si128(src.add(i).cast());
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
                i += 16;
            }
        }
        i
    }

    /// # Safety
    /// `src` is 16-aligned and readable for `dst.len()` bytes (a multiple of 16); SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn load_128(src: *const u8, dst: &mut [u8]) {
        // SAFETY: as documented.
        unsafe {
            let mut i = 0;
            while i < dst.len() {
                let v = _mm_stream_load_si128(src.add(i).cast());
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
                i += 16;
            }
        }
    }

    /// # Safety
    /// As [`load_128`], with AVX2.
    #[target_feature(enable = "avx2,sse4.1")]
    pub(super) unsafe fn load_256(src: *const u8, dst: &mut [u8]) {
        // SAFETY: as documented; after the head `src + i` is 32-aligned.
        unsafe {
            let mut i = load_head(src, dst, 32);
            while i + 32 <= dst.len() {
                let v = _mm256_stream_load_si256(src.add(i).cast());
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), v);
                i += 32;
            }
            while i < dst.len() {
                let v = _mm_stream_load_si128(src.add(i).cast());
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
                i += 16;
            }
        }
    }

    /// # Safety
    /// As [`load_128`], with AVX-512F.
    #[target_feature(enable = "avx512f,sse4.1")]
    pub(super) unsafe fn load_512(src: *const u8, dst: &mut [u8]) {
        // SAFETY: as documented; after the head `src + i` is 64-aligned.
        unsafe {
            let mut i = load_head(src, dst, 64);
            while i + 64 <= dst.len() {
                let v = _mm512_stream_load_si512(src.add(i).cast());
                _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), v);
                i += 64;
            }
            while i < dst.len() {
                let v = _mm_stream_load_si128(src.add(i).cast());
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
                i += 16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn tiers() -> Vec<WcTier> {
        [WcTier::Volatile, WcTier::Sse, WcTier::Avx2, WcTier::Avx512]
            .into_iter()
            .filter(|t| t.available())
            .collect()
    }

    /// Every tier, every start offset up to two lines and every length up to five lines: the
    /// copy lands exactly, and nothing outside it is touched.
    #[test]
    fn every_offset_and_length_copies_exactly() {
        let src: Vec<u8> = (0..400u32).map(|i| (i * 37 + 11) as u8).collect();
        for tier in tiers() {
            for off in 0..128 {
                for len in (0..330).step_by(7).chain([16, 32, 64, 128, 256]) {
                    let mut buf = std::vec![0xA5u8; 512];
                    // SAFETY: `off + len <= 458 < 512`.
                    unsafe { copy_to_wc_with(tier, buf.as_mut_ptr().add(off), &src[..len]) };
                    assert_eq!(&buf[off..off + len], &src[..len], "{tier:?} to {off}+{len}");
                    assert!(
                        buf[..off]
                            .iter()
                            .chain(&buf[off + len..])
                            .all(|&b| b == 0xA5)
                    );
                    let mut back = std::vec![0u8; len];
                    // SAFETY: as above.
                    unsafe { copy_from_wc_with(tier, buf.as_ptr().add(off), &mut back) };
                    assert_eq!(back, &src[..len], "{tier:?} from {off}+{len}");
                }
            }
        }
    }

    #[test]
    fn the_best_tier_is_available() {
        assert!(wc_tier().available());
        assert!(WcTier::Volatile.available());
    }
}
