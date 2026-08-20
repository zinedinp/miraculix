//! Macro generators for x86 SIMD wrappers: fixed-width, slice, and `#[target_feature]` bodies.
//! Split by op shape (binop/unop/ternop/shift/cvt/structural + masked/imm variants), not by width.
//! Flat re-exports from this module (`super::macros::simd_binop`, etc.).

mod binop;
mod cvt;
mod reduce;
mod shift;
mod structural;
mod ternop;
mod unop;
#[cfg(test)]
mod test_helpers;

pub(crate) use binop::*;
pub(crate) use cvt::*;
pub(crate) use reduce::*;
pub(crate) use shift::*;
pub(crate) use structural::*;
pub(crate) use ternop::*;
pub(crate) use unop::*;
#[cfg(test)]
pub(crate) use test_helpers::*;
