//! ARMv8.3-A `FEAT_JSCVT`: JavaScript `ToInt32` in one instruction. Token:
//! [`Jscvt`].
//!
//! `FJCVTZS` truncates an `f64` toward zero and wraps it **modulo 2³²** into
//! an `i32`; NaN and ±∞ give 0. That is ECMAScript's `ToInt32`, and it is
//! not Rust's `as i32`, which saturates instead of wrapping.
//!
//! Intrinsic `__jcvt`, stable since Rust 1.95 (`stdarch_aarch64_jscvt`).
//! Scalar instruction, not a vector op - it lives under `neon/` because Arm
//! files it with the FP/SIMD extensions.

use super::super::macros::arm_token;
use super::super::simd_arch::__jcvt;

arm_token! {
    /// Proof that `FEAT_JSCVT` (`FJCVTZS`) is available.
    token = Jscvt, requires = [Jscvt],
}

impl Jscvt {
    /// ECMAScript `ToInt32`: truncate toward zero, wrap modulo 2³², NaN/±∞ → 0
    /// (`fjcvtzs w, d`).
    #[inline]
    pub fn jscvt(self, x: f64) -> i32 {
        unsafe { fjcvtzs(x) }
    }

    /// [`Jscvt::jscvt`] over a whole slice.
    ///
    /// # Panics
    /// `out.len() != a.len()`.
    pub fn jscvt_slice(self, a: &[f64], out: &mut [i32]) {
        assert_eq!(out.len(), a.len());
        unsafe { fjcvtzs_slice(a, out) }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "jsconv")]
unsafe fn fjcvtzs(x: f64) -> i32 {
    __jcvt(x)
}

/// # Safety
/// Caller proved the feature via the token.
#[target_feature(enable = "jsconv")]
unsafe fn fjcvtzs_slice(a: &[f64], out: &mut [i32]) {
    for (&x, o) in a.iter().zip(out) {
        *o = unsafe { fjcvtzs(x) };
    }
}

#[cfg(test)]
#[path = "../../test/ops/neon/jscvt.rs"]
mod tests;
