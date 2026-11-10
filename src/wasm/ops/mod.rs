//! # Token-gated SIMD ops (wasm32)
//!
//! Same shape as [`crate::x86::ops`] and `aarch64::ops`: each proposal is a
//! zero-sized **token**. Get one with `Token::detect()` (or
//! `from_features`), then call methods on it.
//!
//! ```
//! # #[cfg(target_arch = "wasm32")] {
//! use miraculix::wasm::ops::simd128::Simd128;
//! miraculix::wasm::enable_simd128(); // after the host checked support
//! if let Some(t) = Simd128::detect() {
//!     assert_eq!(t.add_i32x4([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 33, 44]);
//! }
//! # }
//! ```
//!
//! | Module | Token | Needs |
//! |---|---|---|
//! | [`simd128`] | [`simd128::Simd128`] | `+simd128` or [`crate::wasm::enable_simd128`] |
//! | `relaxed` | `relaxed::RelaxedSimd` | as above for relaxed-simd, **and** the `wasm-relaxed-simd` feature or `+relaxed-simd` at build time |
//!
//! `detect()` is not a hardware probe (wasm has none): it reads `cfg` plus
//! the runtime flags. See [`crate::wasm::features`] for why the flags cannot
//! make a module load on an engine that lacks the proposal.

pub(crate) mod macros;
/// relaxed-simd FMA family. Only with the `wasm-relaxed-simd` feature or
/// `+relaxed-simd` (its opcodes would make the module fail to load on an
/// engine without relaxed-simd).
#[cfg(any(feature = "wasm-relaxed-simd", target_feature = "relaxed-simd"))]
pub mod relaxed;
pub mod simd128;
mod token_only;
