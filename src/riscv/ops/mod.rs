//! # Token-gated vector ops (RISC-V)
//!
//! Same shape as [`crate::x86::ops`] and [`crate::aarch64::ops`]: a token is
//! a zero-sized proof of a live feature. `core::arch::riscv64` has no stable
//! vector intrinsics, so the bodies are `asm!` loops that are
//! vector-length-agnostic (`vsetvli` strip-mining), like the SVE ops.
//!
//! | File | Extension | Token |
//! |---|---|---|
//! | [`zvfh`] | `Zvfh` (vector half-precision FP) | `Zvfh` |
//! | [`zvfbf`] | `Zvfbfmin` (BF16 converts), `Zvfbfwma` (BF16 widening FMA) | `Zvfbfmin`, `Zvfbfwma` |

pub mod zvfbf;
pub mod zvfh;
