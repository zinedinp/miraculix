//! Core identification and the vector-accelerator class.
//!
//! Some RISC-V parts are vector-first accelerator cores, not application
//! cores: for example the SiFive X280 (RV64GCV, RVV 1.0, VLEN 512) in the
//! Tenstorrent Blackhole. They run the same `V` code, but tuning differs (long
//! vectors, in-order, bandwidth-bound), so callers can branch on
//! [`RiscvClass`].
//!
//! The class is a heuristic over facts the OS reports: the JEDEC vendor id
//! from `riscv_hwprobe` (`mvendorid`), the `V` extension, and the vector
//! register length read from the `vlenb` CSR. Treat it as a hint. The X280's
//! `marchid` / `mimpid` values are **not** hard-coded here (unverified), so
//! check [`core_id`] on the device before relying on a narrower match.
//!
//! The Tensix "baby" cores on the same card are RV32 without `V` and are
//! programmed through the tt-metal toolchain, so they never reach this
//! detection.

use super::features::hwprobe;
use super::{Feature, FeatureSet, detect_features};

/// `mvendorid` of SiFive (JEDEC bank 5, id 0x09: `0x489`).
pub const VENDOR_SIFIVE: u64 = 0x489;

/// Raw identification registers from `riscv_hwprobe`. Zero when unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoreId {
    /// `mvendorid` (JEDEC manufacturer id).
    pub vendor: u64,
    /// `marchid`.
    pub arch: u64,
    /// `mimpid`.
    pub imp: u64,
}

/// Reads `mvendorid` / `marchid` / `mimpid` through `riscv_hwprobe`
/// (Linux 6.4+). All zeros elsewhere or on failure.
pub fn core_id() -> CoreId {
    CoreId {
        vendor: hwprobe(0).unwrap_or(0),
        arch: hwprobe(1).unwrap_or(0),
        imp: hwprobe(2).unwrap_or(0),
    }
}

/// Vector register length in bits (`vlenb * 8`), or `None` without `V`.
pub fn vlen_bits() -> Option<u32> {
    if !detect_features().contains(Feature::V) {
        return None;
    }
    let vlenb: usize;
    // SAFETY: `vlenb` is a read-only, user-accessible CSR when `V` exists,
    // which was just checked (reading it without `V` would trap).
    unsafe {
        core::arch::asm!(
            ".option push",
            ".option arch, +v",
            "csrr {0}, vlenb",
            ".option pop",
            out(reg) vlenb,
            options(nomem, nostack)
        );
    }
    Some(vlenb as u32 * 8)
}

/// Coarse core class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RiscvClass {
    /// Ordinary application core (or anything not recognized as an accelerator).
    Application,
    /// Vector-first accelerator core: `V` present, SiFive vendor id, VLEN >= 512
    /// (the X280 shape).
    VectorAccelerator,
}

/// Minimum VLEN (bits) for [`RiscvClass::VectorAccelerator`].
pub const ACCELERATOR_MIN_VLEN: u32 = 512;

impl RiscvClass {
    /// Pure classification from already-known facts (testable anywhere).
    pub fn from_parts(features: FeatureSet, id: CoreId, vlen: Option<u32>) -> Self {
        if features.contains(Feature::V)
            && id.vendor == VENDOR_SIFIVE
            && vlen.is_some_and(|v| v >= ACCELERATOR_MIN_VLEN)
        {
            RiscvClass::VectorAccelerator
        } else {
            RiscvClass::Application
        }
    }

    /// Classifies the running core.
    pub fn detect() -> Self {
        Self::from_parts(detect_features(), core_id(), vlen_bits())
    }
}
