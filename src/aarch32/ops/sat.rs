//! ARMv6 baseline `SSAT`/`USAT`. Not a runtime-detectable extension: plain
//! ARMv6 ISA (stdarch gates on `target_feature = "v6"`, no HWCAP bit). Free
//! functions, no token. Upstream: `core::arch::arm::{__ssat, __usat}`.

/// `SSAT`: saturate a signed 32-bit integer to the signed range representable
/// in `WIDTH` bits (`1..=32`). Out-of-range `WIDTH` is a compile error via
/// the upstream intrinsic's `static_assert!`.
#[inline]
pub fn ssat<const WIDTH: u32>(x: i32) -> i32 {
	unsafe { core::arch::arm::__ssat::<WIDTH>(x) }
}

/// `USAT`: saturate a signed 32-bit integer to the unsigned range
/// representable in `WIDTH` bits (`1..=32`; same compile-time bound as
/// [`ssat`]).
#[inline]
pub fn usat<const WIDTH: u32>(x: i32) -> u32 {
	unsafe { core::arch::arm::__usat::<WIDTH>(x) }
}

#[cfg(test)]
#[path = "../test/ops/sat.rs"]
mod tests;
