//! # Matrix-engine number formats on the CPU
//!
//! GPUs and accelerator cards (NVIDIA Tensor Cores, AMD Matrix Cores, Intel
//! XMX, TPUs) run FP8, MXFP4/6/8, NVFP4, TF32 and packed INT1/2/4. CPUs have
//! no such types, but each has a CPU equivalent, and this module provides it.
//! It is architecture-neutral: same names and results on x86, Arm, RISC-V and
//! wasm, and safe to call without a token.
//!
//! | Format | CPU equivalent | Where |
//! |---|---|---|
//! | FP8 E4M3 / E5M2, OCP and FNUZ | Arm `FEAT_FP8` converts (OCP only), else 256-entry `f32` table / RNE encode; AVX10.2 in `x86::ops::avx10::fp8` (f16 input, not wired in) | [`MiniFloat`] |
//! | FP6 (E2M3, E3M2), FP4 (E2M1) | same codec, 6 / 4-bit codes | [`MiniFloat`] |
//! | FP16 | F16C (x86) / `fcvtl`+`fcvtn` (AArch64) / `Zvfh` (RISC-V) slice converts, scalar fallback | [`MiniFloat`] |
//! | BF16 | shift / integer round-to-nearest-even (vectorizes; `avx512bf16` flushes subnormals); `Zvfbfmin` on RISC-V | [`MiniFloat`] |
//! | MXFP4/6/8, MXINT8 | decode, scale by `2^e`, `f32` math | [`mx`] |
//! | NVFP4 | FP4 + E4M3 block scale + tensor scale | [`mx`] |
//! | TF32 | `f32` with 13 low mantissa bits rounded off | [`round_to_tf32`] |
//! | INT4 / INT2 / INT1 | packed bytes, unpack to `i8`/`u8` | [`int`] |
//! | INT1 `and.popc` / `xor.popc` | `count_ones` on bytes | [`int::popcount_and`] |
//! | Tenstorrent `BFP_B` (`Bfp8_b`/`Bfp4_b`/`Bfp2_b`) | 16-value rows with a shared max exponent, bit-exact with tt-metal | [`bfp`] |
//! | GGUF `Q8_0`, `Q6_K` (decode only) | ggml's dequantizers | [`gguf`] |
//! | FP64, FP32, INT8/16/32 | native, see the per-arch `ops` | - |
//!
//! Everything is plain scalar Rust over slices (the compiler auto-vectorizes
//! it), except the FP16 slice converts, which dispatch to F16C or Neon at
//! run time. Hardware FP8 converts (AVX10.2, Arm FEAT_FP8) can later replace
//! the bodies without changing the API.
//!
//! ```
//! use miraculix::formats::{MiniFloat, Overflow};
//! let code = MiniFloat::E4M3Fn.encode(1.5, Overflow::Saturate);
//! assert_eq!(MiniFloat::E4M3Fn.decode(code), 1.5);
//! assert_eq!(MiniFloat::E4M3Fn.max_finite(), 448.0);
//! ```

pub mod bfp;
pub mod gguf;
pub mod int;
mod minifloat;
pub mod mx;
mod tf32;

#[cfg_attr(not(target_arch = "powerpc64"), allow(unused_imports))]
pub(crate) use minifloat::bf16_fast;
pub use minifloat::{MiniFloat, Overflow, Special};
pub use tf32::{round_to_tf32, round_to_tf32_slice};

#[cfg(test)]
mod tests;
