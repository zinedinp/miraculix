//! Process-wide `AtomicU8` cache for arch tier enums (`#[repr(u8)]`).
//! `EMPTY = 0xFF` so a valid level may be `0` (V8_0, M1, Gc, Scalar, etc).

#[cfg(not(target_has_atomic = "64"))]
use core::sync::atomic::AtomicU32;
#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicU64;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Sentinel: cache not filled. Not a valid tier discriminant.
pub const EMPTY: u8 = 0xFF;

/// One byte of process-global level state.
pub struct CachedU8(AtomicU8);

impl CachedU8 {
    pub const fn new() -> Self {
        Self(AtomicU8::new(EMPTY))
    }

    /// Cached value, or run `init` once (races may double-init; same CPU => same level).
    #[inline]
    pub fn get_or_init(&self, init: impl FnOnce() -> u8) -> u8 {
        let v = self.0.load(Ordering::Acquire);
        if v != EMPTY {
            return v;
        }
        let computed = init();
        debug_assert!(computed != EMPTY, "tier discriminant must not equal EMPTY");
        #[cfg(target_has_atomic = "8")]
        match self
            .0
            .compare_exchange(EMPTY, computed, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => computed,
            Err(actual) => actual,
        }
        // No compare-and-swap (e.g. `riscv32im`, no `A`): a plain store. Every racing
        // `init` computes the same value, so whichever store lands last is still right.
        #[cfg(not(target_has_atomic = "8"))]
        {
            self.0.store(computed, Ordering::Release);
            computed
        }
    }
}

/// Process-wide cache for one 128-bit value (an arch's feature bitset).
///
/// `core` has no stable `AtomicU128`, so the value is split into halves
/// (or quarters on targets without 64-bit atomics, e.g. Cortex-M) behind a
/// `filled` flag: the halves are stored first, then `filled` with
/// `Release`; a reader that sees `filled` with `Acquire` sees the halves.
/// Only plain loads and stores are used, so it also works on cores without
/// compare-and-swap (`thumbv6m`, `riscv32im`).
///
/// Racing first callers may each run `init` and store; that is harmless only
/// because `init` returns the same value on every core (a feature probe of
/// one machine). Do not use it for values that can differ between callers.
pub struct CachedU128 {
    filled: AtomicBool,
    #[cfg(target_has_atomic = "64")]
    parts: [AtomicU64; 2],
    #[cfg(not(target_has_atomic = "64"))]
    parts: [AtomicU32; 4],
}

impl CachedU128 {
    pub const fn new() -> Self {
        Self {
            filled: AtomicBool::new(false),
            #[cfg(target_has_atomic = "64")]
            parts: [AtomicU64::new(0), AtomicU64::new(0)],
            #[cfg(not(target_has_atomic = "64"))]
            parts: [
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
            ],
        }
    }

    /// Cached value, or run `init` and cache it.
    #[inline]
    pub fn get_or_init(&self, init: impl FnOnce() -> u128) -> u128 {
        if self.filled.load(Ordering::Acquire) {
            return self.load_parts();
        }
        let v = init();
        self.store_parts(v);
        self.filled.store(true, Ordering::Release);
        v
    }

    #[inline]
    fn load_parts(&self) -> u128 {
        let w = (u128::BITS / self.parts.len() as u32) as usize;
        self.parts.iter().enumerate().fold(0u128, |acc, (i, p)| {
            acc | (u128::from(p.load(Ordering::Relaxed)) << (i * w))
        })
    }

    fn store_parts(&self, v: u128) {
        let w = (u128::BITS / self.parts.len() as u32) as usize;
        for (i, p) in self.parts.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, reason = "one part of the value")]
            p.store((v >> (i * w)) as _, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_u128_round_trips_every_bit() {
        let v = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        let c = CachedU128::new();
        assert_eq!(c.get_or_init(|| v), v);
        assert_eq!(c.get_or_init(|| unreachable!("cached")), v);
    }
}
