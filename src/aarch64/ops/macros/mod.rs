//! Macro generators for AArch64 wrappers. The Neon DSL (binop/unop/cmp/
//! ternop/shift/cvt/reduce) and the token macro live in
//! [`crate::arm_shared::macros`], shared with aarch32, and are re-exported
//! here so op files keep importing `super::super::macros::neon_binop`.
//! Only the AArch64-only parts are local: SVE (no A32 encoding) and the test
//! oracles.

mod sve;
#[cfg(test)]
mod test_helpers;

pub(crate) use crate::arm_shared::macros::*;
pub(crate) use sve::*;
#[cfg(test)]
pub(crate) use test_helpers::*;
