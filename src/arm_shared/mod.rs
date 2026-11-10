//! Code shared by the two Arm back-ends, [`crate::aarch64`] and
//! [`crate::aarch32`]: the Neon wrapper DSL (fixed-width form, `_slice` form,
//! the `#[target_feature]` bodies behind both) and the token macro.
//!
//! The macros never name an arch module. Intrinsics, loads and stores arrive
//! as paths the call site imports from [`simd_arch`], and tokens resolve
//! `Feature` / `FeatureSet` through [`host`]. Both aliases are set per arch
//! here and nowhere else.
//!
//! What stays per arch: SVE (`aarch64::ops::macros::sve`, no A32 encoding),
//! DSP/SIMD32 (`aarch32::ops::macros`, GPR-only, no A64 encoding) and the
//! aarch64 test helpers.

/// `core::arch` module of the host Arm arch.
#[cfg(target_arch = "aarch64")]
pub(crate) use core::arch::aarch64 as simd_arch;
/// `core::arch` module of the host Arm arch.
#[cfg(target_arch = "arm")]
pub(crate) use core::arch::arm as simd_arch;

/// Crate module of the host Arm arch (`Feature`, `FeatureSet`).
#[cfg(target_arch = "arm")]
pub(crate) use crate::aarch32 as host;
/// Crate module of the host Arm arch (`Feature`, `FeatureSet`).
#[cfg(target_arch = "aarch64")]
pub(crate) use crate::aarch64 as host;

pub(crate) mod macros;
