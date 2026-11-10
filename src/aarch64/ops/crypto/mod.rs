//! ARMv8-A Cryptographic Extension. Every file here is optional even on v8.0
//! and each carries its own token, because CPUs ship these in different
//! combinations (a Cortex-A53 may have none, a server core all of them).
//!
//! | File | Feature | Token |
//! |---|---|---|
//! | [`aes`] | `FEAT_AES` | `Aes` |
//! | [`pmull`] | `FEAT_PMULL` | `Pmull` |
//! | [`sha2`] | `FEAT_SHA1` + `FEAT_SHA256` | `Sha` |
//! | [`sha3`] | `FEAT_SHA3` | `Sha3` |
//! | [`sha512`] | `FEAT_SHA512` | `Sha512` |
//! | [`sm3`] | `FEAT_SM3` | `Sm3` |
//! | [`sm4`] | `FEAT_SM4` | `Sm4` |

pub mod aes;
pub mod pmull;
pub mod sha2;
pub mod sha3;
pub mod sha512;
pub mod sm3;
pub mod sm4;
