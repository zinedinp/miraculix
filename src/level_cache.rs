//! Process-wide `AtomicU8` cache for arch tier enums (`#[repr(u8)]`).
//! `EMPTY = 0xFF` so a valid level may be `0` (V8_0, M1, Gc, Scalar, etc).

use core::sync::atomic::{AtomicU8, Ordering};

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
		match self.0.compare_exchange(EMPTY, computed, Ordering::Release, Ordering::Acquire) {
			Ok(_) => computed,
			Err(actual) => actual,
		}
	}
}
