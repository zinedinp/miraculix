//! # Token-gated SIMD ops (AArch64)
//!
//! Same shape as [`crate::x86::ops`] and [`crate::aarch32::ops`]: each
//! extension is a zero-sized **token** (e.g. [`Neon`](neon::neon::Neon)). You
//! only get one after a real capability check. Call methods on the token; they
//! are safe because the token proves the feature is live.
//!
//! ```
//! # #[cfg(target_arch = "aarch64")] {
//! use miraculix::aarch64::ops::neon::neon::Neon;
//! if let Some(t) = Neon::detect() {
//!     assert_eq!(t.add_i32x4([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 33, 44]);
//! }
//! # }
//! ```
//!
//! ## How to get a token
//!
//! | Call | When |
//! |---|---|
//! | `Token::detect()` | One-off probe |
//! | `Token::from_features(set)` | You already have [`super::detect_features`] |
//!
//! Detect on demand, not cached, not auto-dispatched (same policy as
//! [`crate::x86::ops`]). For whole-buffer work that picks the tier for you,
//! use [`super::auto_up`] instead.
//!
//! ## Layout
//!
//! Folders follow the **Arm Intrinsics Guide**'s grouping, the way
//! `x86::ops` follows the Intel Intrinsics Guide:
//!
//! - [`neon`]: Advanced SIMD. The v8.0 baseline plus one file per optional
//!   extension that adds vector ops (RDM, FP16, DotProd, FHM, FCMA, JSCVT,
//!   FRINTTS, I8MM, BF16).
//! - [`crypto`]: AES, PMULL, SHA1/SHA2, SHA3, SHA512, SM3, SM4.
//! - [`other`]: CRC32 and the `RNDR` random-number instructions.
//! - [`sve`]: SVE / SVE2 and their matrix, crypto and bit-permute
//!   extensions, plus the vector-length tokens. `asm!`-based. No stable
//!   intrinsics exist.
//!
//! Unlike aarch32, none of this needs a Cargo feature or a nightly toolchain:
//! `core::arch::aarch64`'s Neon intrinsics are stable.

/// Where the intrinsics come from: `core::arch::aarch64`, through the alias
/// in [`crate::arm_shared`] that also serves aarch32.
pub(crate) use crate::arm_shared::simd_arch;

pub mod crypto;
pub(crate) mod macros;
pub mod neon;
pub mod other;
pub mod sve;

// Not wrapped (TODO, see notes/miraculix/TODO.md aarch64 section):
// - SME/SME2 ops: ZA tile state plus streaming-mode entry/exit
//   (SMSTART/SMSTOP) are process state, not a value-in/value-out op. The
//   token-plus-safe-wrapper model does not cover them. The `sme`/`sme2`
//   target features are also still unstable (1.98, 1.99). Detect-only for now
//   (Feature::Sme*).
// - SME2p1/SME2p2, SME_LUTV2, SME_F8F16/F8F32: detect-only, same reason.
// - Unsigned variable shifts (`vshlq_u32(uint32x4_t, int32x4_t)`): the count
//   vector is signed even for unsigned data, so the operand types do not
//   match and `neon_binop!` does not fit. Signed `shl_i*` is wrapped;
//   unsigned is reachable through `slli_u*`/`srli_u*` with a constant count.
// - SVE: `f16` lanes, gather/scatter and first-fault loads, SVE2 widening /
//   narrowing / pairwise ops, `MATCH`/`HISTCNT`, and the SVE2-SHA3 / SVE2-SM4
//   crypto (`Feature::Sve2Sha3`/`Sve2Sm4` are detect-only). `ops/sve` covers
//   the elementwise set `auto_up` needs plus the matrix / bit-permute / AES
//   extensions.
// - Table lookups (`vqtbl*`), zips/unzips/transposes and lane
//   inserts/extracts: structural, not elementwise. x86's equivalents live in
//   `macros/structural.rs`; there is no aarch64 caller for them yet.
