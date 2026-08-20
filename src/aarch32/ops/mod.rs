//! # Token-gated SIMD ops (AArch32)
//!
//! Same shape as [`crate::x86::ops`]: each extension is a zero-sized
//! **token** (e.g. `Dsp`, `Neon`). You only get one after a real capability
//! check. Call methods on the token; they are safe because the token proves
//! the feature is live.
//!
//! ```
//! # #[cfg(all(target_arch = "arm", feature = "nightly-arm-neon"))] {
//! use miraculix::aarch32::ops::dsp::Dsp;
//! if let Some(t) = Dsp::detect() {
//!     let _ = t.qadd(i32::MAX, 1);
//! }
//! # }
//! ```
//!
//! ## How to get a token
//!
//! | Call | When |
//! |---|---|
//! | `Token::detect()` | One-off probe |
//! | `Token::from_features(set)` | You already have [`super::FeatureSet::detect`] |
//!
//! Detect on demand, not cached, not auto-dispatched (same policy as
//! [`crate::x86::ops`]). Two distinct base ISAs: [`dsp`] (ARMv6 DSP/SIMD32
//! in a plain GPR, [`super::Feature::Edsp`]) and [`neon`] (ARMv7-A vector
//! regs, [`super::Feature::Neon`]). Further modules add optional layers on
//! Neon (`vfpv4neon`, `crc32`, `crypto`, `fp16`, `dotprod`, `i8mm`) or
//! baseline ARMv6 free functions ([`sat`]).

pub mod dsp;
/// Needs a `v7` compile-time baseline (`armv7-*` or `-C target-feature=+v7`),
/// matching `core::arch::arm`'s gate on its `neon` module. Sub-`v7` targets
/// (e.g. Pi Zero / ARMv6) can use [`dsp`] only; Neon is compiled out entirely,
/// separate from [`super::Feature::Neon`]'s runtime check in
/// [`Neon::detect`](neon::Neon::detect).
#[cfg(any(target_feature = "v7", doc))]
pub mod neon;
/// Neon + VFPv4 fused multiply-add (`+FMA`). Separate token from [`neon`],
/// same `v7` compile-time gate.
#[cfg(any(target_feature = "v7", doc))]
pub mod vfpv4neon;
/// ARMv8-A32 CRC32 (Cortex-A32 class). Same upstream `neon` module / `v7`
/// compile gate as [`neon`]/[`vfpv4neon`]; `crc`/`v8` are runtime
/// `#[target_feature]` only.
#[cfg(any(target_feature = "v7", doc))]
pub mod crc32;
/// ARMv8-A32 Crypto (AES + SHA1/SHA256). Same `v7` gate as [`crc32`]; lives
/// under ordinary `stdarch_arm_neon_intrinsics` (no separate feature like CRC).
#[cfg(any(target_feature = "v7", doc))]
pub mod crypto;
/// ARMv6 baseline `SSAT`/`USAT`: free functions, no token (compile-time `v6`
/// only, not a runtime extension like [`dsp`]/[`neon`]).
#[cfg(any(target_feature = "v6", doc))]
pub mod sat;
/// ARMv8.2-A FullFP16 Neon. Same `v7` compile gate as [`crc32`]/[`crypto`].
#[cfg(any(target_feature = "v7", doc))]
pub mod fp16;
/// ARMv8.2-A Dot Product (`vdotq_s32`/`vdotq_u32`). Same `v7` gate.
#[cfg(any(target_feature = "v7", doc))]
pub mod dotprod;
/// ARMv8.6-A Int8 Matrix Multiply. Same `v7` gate as [`crc32`]/[`crypto`].
#[cfg(any(target_feature = "v7", doc))]
pub mod i8mm;
mod macros;

// Not wrapped (stdarch / HWCAP gaps; see notes/miraculix/TODO.md B4):
// - RDM (Armv8.1): no arm stdarch path, no 32-bit Linux HWCAP bit.
// - Complex (Armv8.3 VCMLA/VCADD): no stdarch on any arch.
// - Feature::Fhm (vfmlal/vfmlsl): aarch64-only stdarch; detect-only like Pmull.
// - Feature::AsimdBf16: no stdarch on any arch; detect-only.
// - Feature::Pmull: no arm `vmull_p64` in stdarch; detect-only.
