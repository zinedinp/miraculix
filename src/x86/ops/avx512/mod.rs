//! AVX-512 family modules. Phi-only native ops are `asm!`-encoded and gated
//! by `phi-asm`; mainstream equivalents appear in `avx512f`/`avx512vnni`
//! when available.

#[cfg(feature = "phi-asm")]
pub mod avx512_4fmaps;
#[cfg(feature = "phi-asm")]
pub mod avx512_4vnniw;
#[cfg(feature = "phi-asm")]
pub mod avx512er;
#[cfg(feature = "phi-asm")]
pub mod avx512pf;
pub mod avx512bf16;
pub mod avx512bitalg;
pub mod avx512bw;
pub mod avx512cd;
pub mod avx512dq;
pub mod avx512f;
pub mod avx512fp16;
pub mod avx512ifma;
pub mod kmask;
pub mod avx512vbmi;
pub mod avx512vbmi2;
pub mod avx512vl;
pub mod avx512vnni;
pub mod avx512vpopcntdq;
