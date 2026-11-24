//! powerpc64 SIMD op wrappers, one file per extension. AltiVec needs
//! nightly (`nightly-altivec`). powerpc64 only: the wrappers import
//! `core::arch::powerpc64`, so 32-bit PowerPC gets detection but no ops.

#[cfg(all(feature = "nightly-altivec", target_arch = "powerpc64"))]
pub mod altivec;
