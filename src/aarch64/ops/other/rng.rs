//! ARMv8.5-A `FEAT_RNG`: hardware random numbers from the `RNDR` and
//! `RNDRRS` system registers. Token: [`Rng`].
//!
//! Either read can **fail** (the hardware sets `Z` and returns 0 when it
//! cannot produce a value in reasonable time), so both methods return
//! `Option<u64>`, like x86's `rdrand` wrapper. `rndrrs` reseeds the DRBG
//! before generating, trading throughput for fresher entropy.
//!
//! **Why `asm!`:** `__rndr`/`__rndrrs` are unstable
//! (`stdarch_aarch64_rand`); the `rand` target feature is stable.
//!
//! Not an `auto` candidate and not deterministic, so its tests check
//! plausibility only (see `test/ops/other/rng.rs`).

use core::arch::asm;

use super::super::macros::arm_token;

arm_token! {
    /// Proof that `FEAT_RNG` (`RNDR`/`RNDRRS`) is available.
    token = Rng, requires = [Rng],
}

impl Rng {
    /// One 64-bit random value from `RNDR`, or `None` if the hardware reported
    /// failure (`mrs x, RNDR`; `Z` set means "no value").
    #[inline]
    pub fn rndr(self) -> Option<u64> {
        unsafe { rndr_fixed() }
    }

    /// As [`Rng::rndr`], but reseeds first (`mrs x, RNDRRS`).
    #[inline]
    pub fn rndrrs(self) -> Option<u64> {
        unsafe { rndrrs_fixed() }
    }
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "rand")]
unsafe fn rndr_fixed() -> Option<u64> {
    let (v, ok): (u64, u64);
    // Not `pure`: two reads must be allowed to differ. Writes NZCV.
    unsafe {
        asm!("mrs {v}, RNDR", "cset {ok}, ne", v = out(reg) v, ok = out(reg) ok, options(nomem, nostack))
    };
    (ok != 0).then_some(v)
}

/// # Safety
/// Caller proved the feature via the token.
#[inline]
#[target_feature(enable = "rand")]
unsafe fn rndrrs_fixed() -> Option<u64> {
    let (v, ok): (u64, u64);
    unsafe {
        asm!("mrs {v}, RNDRRS", "cset {ok}, ne", v = out(reg) v, ok = out(reg) ok, options(nomem, nostack))
    };
    (ok != 0).then_some(v)
}

#[cfg(test)]
#[path = "../../test/ops/other/rng.rs"]
mod tests;
