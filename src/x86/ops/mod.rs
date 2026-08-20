//! # Token-gated SIMD ops (x86)
//!
//! Each extension is a zero-sized **token** (e.g. `Avx2`). You only get one
//! after a real capability check. Call methods on the token; they are safe
//! because the token proves the feature is live.
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::ops::sse::sse2::Sse2;
//! if let Some(t) = Sse2::detect() {
//!     let _ = t.add_i32x4([1, 2, 3, 4], [5, 6, 7, 8]);
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
//! | `Token::from_level(level)` | When implemented: tier implies the feature |
//!
//! For whole buffers without picking a width, prefer [`super::auto_up`].
//!
//! Folders follow the Intel Intrinsics Guide: [`mmx`], [`sse`],
//! [`avx`], [`avx512`], plus [`other`] (AES, SHA, POPCNT, AMD extras, ...).

pub mod avx;
pub mod avx512;
mod macros;
pub mod mmx;
pub mod other;
pub mod sse;
