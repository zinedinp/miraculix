//! WASM SIMD proposals and the runtime flags: compile-time
//! `cfg(target_feature)` **or** a flag the embedder sets (pulp-style, see
//! [`enable_simd128`]). No hardware probe exists on wasm.
//!
//! The flags choose which code runs; they cannot make a module load on an
//! engine that lacks the proposal (validation is per module). Why, and
//! what the `wasm-simd` / `wasm-relaxed-simd` cargo features put into the
//! module: see the [`super`] module doc, "What ends up in the module".

use core::sync::atomic::{AtomicBool, Ordering};

static SIMD128: AtomicBool = AtomicBool::new(cfg!(target_feature = "simd128"));
static RELAXED_SIMD: AtomicBool = AtomicBool::new(cfg!(target_feature = "relaxed-simd"));

/// Tell the crate the engine runs simd128. Call once, after the host checked.
pub fn enable_simd128() {
    SIMD128.store(true, Ordering::Relaxed);
}

/// Back to the scalar path (also drops relaxed-simd, which implies simd128).
/// No effect when the binary was built with `+simd128`: the compiler already
/// emits simd128 everywhere, so the flag cannot take it back.
pub fn disable_simd128() {
    disable_relaxed_simd();
    SIMD128.store(false, Ordering::Relaxed);
}

/// `cfg(target_feature = "simd128")` or [`enable_simd128`] was called.
pub fn is_simd128_enabled() -> bool {
    cfg!(target_feature = "simd128") || SIMD128.load(Ordering::Relaxed)
}

/// Tell the crate the engine runs relaxed-simd (enables simd128 too).
pub fn enable_relaxed_simd() {
    enable_simd128();
    RELAXED_SIMD.store(true, Ordering::Relaxed);
}

/// Drop relaxed-simd only; simd128 stays as it was. No effect when built
/// with `+relaxed-simd`.
pub fn disable_relaxed_simd() {
    RELAXED_SIMD.store(false, Ordering::Relaxed);
}

/// `cfg(target_feature = "relaxed-simd")` or [`enable_relaxed_simd`] was
/// called.
pub fn is_relaxed_simd_enabled() -> bool {
    cfg!(target_feature = "relaxed-simd") || RELAXED_SIMD.load(Ordering::Relaxed)
}

/// Enum list for WASM SIMD proposals. `relaxed-simd` implies `simd128` (LLVM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// Fixed-width 128-bit SIMD (`v128`, `i32x4_add`, ...). Stable since 1.54.
    Simd128,
    /// Relaxed-SIMD: implementation-defined-precision ops (fma, swizzle, dot).
    /// Stable since 1.82.
    RelaxedSimd,
}

impl Feature {
    /// Index = bit in [`FeatureSet`].
    pub const ALL: &'static [Feature] = &[Feature::Simd128, Feature::RelaxedSimd];

    /// Bit index in [`FeatureSet`].
    pub const fn bit(self) -> u32 {
        // The enum has no explicit discriminants and `ALL` lists the
        // variants in declaration order, so the discriminant *is* the index
        // in `ALL` (checked by `bit_is_the_index_in_all`). O(1), no search.
        self as u32
    }
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
    /// This binary's `cfg(target_feature)` plus the runtime flags
    /// ([`enable_simd128`], [`enable_relaxed_simd`]). Never a hardware read:
    /// wasm has none. Two atomic loads, so there is no cache.
    pub fn detect() -> Self {
        let mut set = 0u128;
        if is_simd128_enabled() {
            set |= 1 << Feature::Simd128.bit();
        }
        if is_relaxed_simd_enabled() {
            set |= 1 << Feature::RelaxedSimd.bit();
        }
        Self(set)
    }

    pub fn contains(self, feature: Feature) -> bool {
        self.0 & (1 << feature.bit()) != 0
    }

    pub fn contains_all(self, required: &[Feature]) -> bool {
        required.iter().all(|&f| self.contains(f))
    }
}

#[cfg(test)]
mod bit_index_tests {
    use super::Feature;

    /// `Feature::bit` relies on declaration order matching `Feature::ALL`;
    /// reordering one without the other fails here, not silently at runtime.
    #[test]
    fn bit_is_the_index_in_all() {
        for (i, &f) in Feature::ALL.iter().enumerate() {
            assert_eq!(f.bit() as usize, i, "{f:?}");
        }
        assert!(Feature::ALL.len() <= 128, "FeatureSet is a u128");
    }
}
