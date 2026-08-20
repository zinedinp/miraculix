//! powerpc64 SIMD op wrappers, one file per extension. AltiVec needs
//! nightly (`nightly-altivec`; see `notes/miraculix/TODO.md`).

#[cfg(feature = "nightly-altivec")]
pub mod altivec;
