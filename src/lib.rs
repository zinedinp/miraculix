//! # miraculix
//!
//! Safe CPU SIMD for Rust: **detect** host capabilities, then either call
//! **auto slice ops** or run **token-gated** kernels.
//!
//! This crate is `no_std` (tests excepted). Each supported architecture is a
//! top-level module gated by `cfg(target_arch = ...)`.
//!
//! ## Which module do I import?
//!
//! | Your target | Module |
//! |---|---|
//! | x86 / x86_64 | [`x86`] |
//! | aarch64 | [`aarch64`] |
//! | arm (AArch32) | [`aarch32`] |
//! | riscv32 / riscv64 | [`riscv`] |
//! | loongarch32 / loongarch64 | [`loongarch`] |
//! | powerpc / powerpc64 | [`powerpc`] |
//! | wasm32 / wasm64 | [`wasm`] |
//!
//! ## Three ways to call (x86 example)
//!
//! ### 1. Auto slices (easiest)
//!
//! Whole-buffer elementwise ops. miraculix picks the best tier and falls
//! back to scalar when needed.
//!
//! ```rust
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//! # {
//! use miraculix::x86::auto_up;
//!
//! let a = [1i32, 2, 3, 4];
//! let b = [10, 20, 30, 40];
//! let mut out = [0i32; 4];
//! auto_up::add_i32(&a, &b, &mut out);
//! assert_eq!(out, [11, 22, 33, 44]);
//! # }
//! ```
//!
//! Full list: [`x86::auto_up`].
//!
//! ### 2. Detect + tokens
//!
//! Ask what the CPU can do, then call fixed-width ops only when a proof
//! token exists.
//!
//! ```rust
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//! # {
//! use miraculix::x86::{detect_level, detect_features, Feature, GenericLevel};
//! use miraculix::x86::ops::avx::avx2::Avx2;
//!
//! let level = detect_level(); // cached process-wide
//! let set = detect_features();
//!
//! if let Some(t) = Avx2::from_features(set) {
//!     let _ = t.add_i32x8([1; 8], [2; 8]);
//! }
//! assert!(level >= GenericLevel::V1);
//! assert!(set.contains(Feature::Sse2) || cfg!(not(target_arch = "x86_64")));
//! # }
//! ```
//!
//! ### 3. Multi-op kernels
//!
//! Wrap a body so several token ops compile inside one `target_feature`
//! function: [`avx_fn!`], [`avx2_fn!`], [`avx512_fn!`], [`fma_fn!`],
//! [`aes_fn!`], etc: full catalog in [`x86::fn_macros`].
//!
//! ## Detect API (every arch)
//!
//! | Call | Use for |
//! |---|---|
//! | `detect_level()` | Normal code: best tier, process cache |
//! | `detect_level_fresh()` | Tests / re-audit: ignore cache |
//! | `warm_up()` | Optional: fill cache at startup |
//! | `shortpath::verify_or_panic()` | Startup check if binary assumes a high tier |
//!
//! x86 also has [`x86::detect_features`] / [`x86::detect_features_fresh`] for
//! the raw per-extension bitset used by auto dispatch.
//!
//! ## Further reading
//!
//! Crate README (install, full auto catalog, token layout, Cargo features).

#![cfg_attr(not(test), no_std)]
#![cfg_attr(feature = "nightly-altivec", feature(stdarch_powerpc, powerpc_target_feature))]
#![cfg_attr(feature = "nightly-rtm", feature(stdarch_x86_rtm, rtm_target_feature))]
#![cfg_attr(
	all(feature = "nightly-arm-neon", target_arch = "arm"),
	feature(stdarch_arm_dsp, stdarch_arm_neon_intrinsics, stdarch_arm_sat, arm_target_feature)
)]
// `stdarch_aarch32_crc32` (`aarch32::ops::crc32`) has no unconditional
// re-export carrying its name on `target_arch = "arm"` (only inside the
// `v7`-gated `neon` submodule, unlike the other three features above), so
// it needs its own `target_feature = "v7"`-gated block - otherwise it's an
// "unknown feature" (E0635) on a sub-`v7` compile, not just unused.
#![cfg_attr(
	all(feature = "nightly-arm-neon", target_arch = "arm", target_feature = "v7"),
	feature(stdarch_aarch32_crc32)
)]
// `aarch32::auto_down`'s FullFP16 scalar-remainder fallback needs the
// unstable `f16` primitive (no hardware to fall back to in software; `core`
// has no `f16` arithmetic without it). Same compile-time gate as `auto_down`
// itself (`nightly-arm-neon` + `v7`).
#![cfg_attr(all(feature = "nightly-arm-neon", target_arch = "arm", target_feature = "v7"), feature(f16))]

mod level_cache;

/// x86 / x86_64: detect tiers and features, auto slice ops, tokens, kernel macros.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// AArch64: NEON / SVE / SME detect (OS-specific probes).
// AArch64: getauxval / elf_aux_info / sysctl / PF / bare compile-time.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// AArch32: VFP / NEON detect.
// AArch32: auxv or bare compile-time. Windows CE: no rustc target (TODO).
#[cfg(target_arch = "arm")]
pub mod aarch32;

/// RISC-V: base + vector detect.
// RISC-V: auxv or bare compile-time.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub mod riscv;

/// LoongArch: LSX / LASX detect.
// LoongArch: Linux LSX/LASX auxv; bare compile-time.
#[cfg(any(target_arch = "loongarch32", target_arch = "loongarch64"))]
pub mod loongarch;

/// PowerPC: AltiVec / VSX detect (and optional ops behind `nightly-altivec`).
// powerpc64 (LE first): auxv AltiVec/VSX or bare cfg.
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
pub mod powerpc;

/// WebAssembly: compile-time simd128 / relaxed-simd only.
// WASM32/64: compile-time only.
#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub mod wasm;
