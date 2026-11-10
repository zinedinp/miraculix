//! Advanced SIMD (Neon). [`neon`] is the ARMv8.0-A baseline every AArch64 CPU
//! has; the other files are optional extensions, one per `FEAT_*` bit, each
//! with its own token.
//!
//! | File | Arm version | Feature | Token |
//! |---|---|---|---|
//! | [`neon`] | v8.0 (mandatory) | `Neon` | `Neon` |
//! | [`rdm`] | v8.1 | `FEAT_RDM` | `Rdm` |
//! | [`fp16`] | v8.2 | `FEAT_FP16` | `Fp16Neon` |
//! | [`dotprod`] | v8.2 | `FEAT_DotProd` | `Dotprod` |
//! | [`fhm`] | v8.2 | `FEAT_FHM` | `Fhm` |
//! | [`fcma`] | v8.3 | `FEAT_FCMA` | `Fcma` |
//! | [`jscvt`] | v8.3 | `FEAT_JSCVT` | `Jscvt` |
//! | [`frintts`] | v8.5 | `FEAT_FRINTTS` | `Frintts` |
//! | [`i8mm`] | v8.6 | `FEAT_I8MM` | `I8mm` |
//! | [`bf16`] | v8.6 | `FEAT_BF16` | `Bf16` |

pub mod bf16;
pub mod dotprod;
pub mod fcma;
pub mod fhm;
pub mod fp16;
pub mod frintts;
pub mod i8mm;
pub mod jscvt;
// Family module and its baseline file share a name, as `x86::ops::avx::avx`.
#[allow(clippy::module_inception)]
pub mod neon;
pub mod rdm;
