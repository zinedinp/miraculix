//! AVX-512 k-mask ops by kind: [`logic`], [`shift`], [`ktest`], [`bridge`] (vector<->mask).
//! Covers `u8`/`u16`/`u32`/`u64` for 512-bit ops; 128/256-bit bridges via `Avx512BwVl`/`Avx512DqVl`.
//! No `auto` (AVX-512 only). Gating: F mask16, DQ for mask8 add/test, BW for mask32/64, VL for bridges.

mod bridge;
mod build;
mod ktest;
mod logic;
mod shift;
