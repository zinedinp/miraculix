//! AVX family modules: `avx`, `avx2`, `f16c`, `fma`, `avx_vnni`, `avx_ifma`,
//! `avx_ne_convert`, and related extensions. See the Intel Intrinsics Guide
//! for reference.

// The family module and its primary/oldest extension share a name by design (zero path
// stutter: `ops::avx::Avx`, not `ops::avx::avx::Avx`; see this file's own doc).
#[allow(clippy::module_inception)]
pub mod avx;
pub mod avx2;
pub mod avx_ifma;
pub mod avx_ne_convert;
pub mod avx_vnni;
pub mod avx_vnni_int8;
pub mod avx_vnni_int16;
pub mod f16c;
pub mod fma;
pub mod sha512;
pub mod sm3;
pub mod sm4;
