//! Vector-length tokens: [`SveVl128`], [`SveVl256`], [`SveVl512`].
//!
//! Each proves `Feature::Sve` **and** a hardware vector length (VL) of at
//! least the named width, and carries the fixed-width forms of that width
//! (`add_i32x4` / `add_i32x8` / `add_i32x16`), generated next to the slice
//! ops in [`super::sve`]. Those names match the Neon and x86 counterparts
//! (`add_i32x8` is AVX2's width), which is what keeps code translatable.
//!
//! The VL guarantee is a **performance** promise, not a correctness one.
//! Fixed-width forms use the same `whilelo` loop as the slice ops, so they
//! are exact on any VL. One iteration only if VL is at least the token width.
//! Linux can change VL later (`prctl(PR_SVE_SET_VL)`). The result stays
//! right. Only the single-iteration promise lapses.
//!
//! The VL is read with `cntb` (bytes per vector; one instruction, no OS
//! call, works on every OS) on **every** call, never cached: Linux sets the
//! VL per thread, so a process-wide cache could hand one thread another
//! thread's length.

use super::super::super::FeatureSet;
use super::sve::Sve;

/// # Safety
/// Caller proved `Feature::Sve`.
#[target_feature(enable = "sve")]
unsafe fn cntb() -> u64 {
    let n: u64;
    unsafe { core::arch::asm!("cntb {}", out(reg) n, options(nomem, nostack, preserves_flags)) };
    n
}

impl Sve {
    /// The **calling thread's** vector length in bits (128 to 2048, a
    /// multiple of 128), read with `cntb` each time.
    ///
    /// 128 is common (Neoverse N2/V2, Grace, Graviton4, Cortex-A710 and
    /// later) and fully supported: every SVE op here is `whilelo`-predicated
    /// and exact at any length. Linux can change the length per thread
    /// (`prctl(PR_SVE_SET_VL)`), so size buffers from a fresh call on the
    /// thread that uses them.
    #[inline]
    pub fn vector_bits(self) -> usize {
        // SAFETY: `self` proves SVE.
        let bytes = unsafe { cntb() } as usize;
        bytes * 8
    }
}

macro_rules! vl_token {
	($(#[$doc:meta])* $Token:ident, $bits:literal) => {
		$(#[$doc])*
		///
		/// Zero-sized, `Copy`. Obtain via `detect()` or `from_features()`.
		#[derive(Debug, Clone, Copy)]
		pub struct $Token(());

		impl $Token {
			/// Probe once: `Some(token)` if SVE is available **and** its vector
			#[doc = concat!("length is at least ", $bits, " bits.")]
			pub fn detect() -> Option<Self> {
				Self::from_features(crate::aarch64::detect_features())
			}

			/// Build a token from a set you already have (e.g.
			/// [`crate::aarch64::detect_features`]). Reads the vector length
			/// (one `cntb`, on the calling thread) only once `Feature::Sve` is confirmed.
			pub fn from_features(set: FeatureSet) -> Option<Self> {
				let sve = Sve::from_features(set)?;
				(sve.vector_bits() >= $bits).then_some($Token(()))
			}

			/// The base SVE token (every VL token implies it).
			#[inline]
			pub fn sve(self) -> Sve {
				Sve::implied_by_stronger_token()
			}
		}
	};
}

vl_token!(
    /// Proof of SVE with a vector length of at least 128 bits. True of every
    /// SVE implementation (the architectural minimum). Carries the 128-bit
    /// fixed-width forms (`add_i32x4`), a drop-in for the `Neon` token's.
    SveVl128,
    128
);
vl_token!(
    /// Proof of SVE with a vector length of at least 256 bits (Neoverse V1,
    /// Graviton3). Carries the 256-bit fixed-width forms (`add_i32x8`).
    SveVl256,
    256
);
vl_token!(
    /// Proof of SVE with a vector length of at least 512 bits (A64FX).
    /// Carries the 512-bit fixed-width forms (`add_i32x16`).
    SveVl512,
    512
);

#[cfg(test)]
#[path = "../../test/ops/sve/vl.rs"]
mod tests;
