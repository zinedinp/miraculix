//! Macro generators for Arm Neon wrappers (aarch64 and aarch32): fixed-width
//! form, slice form, and the `#[target_feature]` bodies behind both. Split by
//! op shape (binop/unop/cmp/ternop/shift/cvt/reduce), not by width. Flat
//! re-exports from this module (`macros::neon_binop`, ...).
//!
//! Grammar is x86's: `key = value,` pairs with a mandatory trailing comma,
//! so a call site reads as a small table and a missing field is a compile
//! error rather than a silently shifted positional argument
//! (`src/x86/ops/macros/binop.rs`).
//!
//! Three properties let one DSL serve both archs (`plans/AArch64-A
//! SIMD-Plan.md`, stages 1.1 and 3):
//!
//! 1. **No hard-coded `core::arch::*`.** Every intrinsic, load and store
//!    arrives as a `$path`; op files import them from
//!    [`super::simd_arch`]. A load may also be a small local helper (aarch32
//!    `fp16` passes a `vld1q_u16` + reinterpret pair as its `loadu`).
//! 2. **`lanes` and `target_feature` are parameters.** AArch64 needs 1 to 16
//!    lanes for its D and Q forms; aarch32 passes its extra baseline bits in
//!    the feature string (`"neon,v7"`).
//! 3. **The `_slice` rule is separate from the fixed-width rule.** SVE fills
//!    the same `_slice` signature with a `whilelo`-predicated loop instead of
//!    chunk + scalar remainder, so the two must be independently replaceable.

// aarch32 uses only part of the DSL (no shifts by immediate, converts or
// reductions yet).
#![cfg_attr(target_arch = "arm", allow(unused_macros, unused_imports))]

mod binop;
mod cmp;
mod cvt;
mod reduce;
mod shift;
mod ternop;
mod token;
mod unop;

pub(crate) use {binop::*, cmp::*, cvt::*, reduce::*, shift::*, ternop::*, token::*, unop::*};
