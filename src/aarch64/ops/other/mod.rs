//! AArch64 extensions that are neither Advanced SIMD nor crypto: they operate
//! on general-purpose registers, so they have no lane structure and no `auto`
//! rung.
//!
//! | File | Feature | Token |
//! |---|---|---|
//! | [`crc32`] | `FEAT_CRC32` | `Crc32` |
//! | [`rng`] | `FEAT_RNG` | `Rng` |

pub mod crc32;
pub mod rng;
