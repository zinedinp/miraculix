//! AVX10.2 additions. Token files, one per extension group:
//!
//! | File | Extension | Token |
//! |---|---|---|
//! | [`fp8`] | AVX10.2 FP8 converts (`vcvthf82ph`, `vcvtph2bf8`, ...) | `Avx10Fp8` |
//!
//! `core::arch` has no AVX10.2 intrinsics yet, so these are `asm!`. No
//! shipping hardware was available when they were written: they are checked
//! by disassembly (see `fp8`), not by execution.

pub mod fp8;
