//! Compile-time only (see [`super::features`]). Returns [`WasmLevel`] directly,
//! not `Assumed`/`Baseline`. Scalar/Simd128/RelaxedSimd all from
//! `cfg(target_feature)`, never a hardware read.

use super::tiers::WasmLevel;

#[cfg(target_feature = "relaxed-simd")]
const COMPILE_TIME_LEVEL: WasmLevel = WasmLevel::RelaxedSimd;

#[cfg(all(target_feature = "simd128", not(target_feature = "relaxed-simd")))]
const COMPILE_TIME_LEVEL: WasmLevel = WasmLevel::Simd128;

#[cfg(not(target_feature = "simd128"))]
const COMPILE_TIME_LEVEL: WasmLevel = WasmLevel::Scalar;

/// The only level: fixed for the life of the compiled module.
pub const fn resolve() -> WasmLevel {
	COMPILE_TIME_LEVEL
}

/// Permanently a no-op: a WASM host refuses to instantiate a module whose
/// compiled-in opcodes it does not support, before any Rust code runs, so no
/// compiled-vs-host mismatch can ever reach this line.
pub fn verify_or_panic() {}
