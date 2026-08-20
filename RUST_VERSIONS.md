# Rust version per feature

Crate MSRV: **1.94.0** (`rust-version` in [Cargo.toml](Cargo.toml)).

## How versions are chosen

Source of truth: `#[stable(feature = ..., since = ...)]` (or `#[unstable]`)
on the **intrinsics this crate actually calls**, from installed `core_arch`
(`rustup component add rust-src`, then
`~/.rustup/toolchains/<t>/lib/rustlib/src/rust/library/stdarch/crates/core_arch/src/x86/`).
Do not copy a neighbor row. Before raising `rust-version`, re-check with
clippy's MSRV lint.

## `x86::ops` (default build)

| Family | Extension | Min `rustc` | Notes |
|---|---|---|---|
| `mmx` | MMX | 1.59.0 | Plain `asm!` (no `core::arch` target feature). |
| `sse` | SSE | 1.27.0 | `sqrt`/`rcp`/`rsqrt` fixed-width only (no `_slice` / `auto_up`: remainder needs `f32::sqrt`, absent under `no_std`). |
| `sse` | SSE2 | 1.27.0 | `sqrt_f64x2` fixed-width only. `mul`/`shl`/`shr`/`sra` on i8/u8 are composed (no native 8-bit SIMD mul/shift); `pub` + `auto_up`. |
| `sse` | SSE3 | 1.27.0 | |
| `sse` | SSSE3 | 1.27.0 | |
| `sse` | SSE4.1 | 1.27.0 | |
| `sse` | SSE4.2 | 1.27.0 | |
| `other` | SSE4a | 1.82.0 | AMD-only; insert/extract i64 stabilized late. |
| `avx` | AVX | 1.27.0 | `sqrt`/`rcp`/`rsqrt` (f32; f64 has `sqrt` only) fixed-width only, same `no_std` reason as SSE. |
| `avx` | AVX2 | 1.27.0 | i8/u8 mul and byte shifts: composed, 256-bit mirror of SSE2. |
| `avx` | F16C | 1.68.0 | |
| `avx` | FMA | 1.27.0 | |
| `other` | FMA4 | 1.59.0 | AMD-only, dead; `asm!`. |
| `other` | XOP | 1.59.0 | AMD-only, dead; `asm!`. |
| `avx512` | AVX-512F | 1.89.0 | `rcp14`/`rsqrt14`/`sqrt` fixed-width only (`no_std`). VL forms: `Avx512FVl`. |
| `avx512` | AVX-512BW | 1.89.0 | 512-bit i8/u8/i16/u16; 128/256 already on SSE2/AVX2 (no `*Vl` token). |
| `avx512` | AVX-512CD | 1.89.0 | |
| `avx512` | AVX-512DQ | 1.89.0 | Native `mullo` i64/u64; VL: `Avx512DqVl` (not in `auto_up`; 512-bit DQ already wins that cascade). |
| `avx512` | AVX512VPOPCNTDQ | 1.89.0 | VL needs extra bit: `Avx512VpopcntdqVl`. |
| `avx512` | AVX512VBMI | 1.89.0 | Permute/bit-shuffle; VL: `Avx512VbmiVl`. |
| `avx512` | AVX512VBMI2 | 1.89.0 | Funnel shifts + compress/expand i8/u16; VL: `Avx512Vbmi2Vl`. |
| `avx512` | AVX512BITALG | 1.89.0 | VL: `Avx512BitalgVl`. |
| `avx512` | AVX512IFMA | 1.89.0 | VL: `Avx512IfmaVl`. Distinct from AVX-IFMA. |
| `avx512` | AVX512VNNI | 1.89.0 | VL: `Avx512VnniVl`. Distinct from AVX-VNNI*. |
| `avx512` | AVX512BF16 | 1.89.0 | Lanes as raw `u16`; token `Avx512Bf16` (`bf16`+`f`). VL token deferred. |
| `avx512` | AVX512FP16 | **1.94.0** | Vector arith stable; scalar `f16` / `_ph` load still unstable. Lanes as raw `u16`. VL: `Avx512Fp16Vl`. `sqrt`/`rsqrt`/`rcp` fixed-width only. |
| `avx` | AVX-VNNI | 1.89.0 | |
| `avx` | AVX-VNNI-INT8 | 1.89.0 | |
| `avx` | AVX-VNNI-INT16 | 1.89.0 | |
| `avx` | AVX-IFMA | 1.89.0 | |
| `avx` | AVX-NE-CONVERT | **1.94.0** | Even/odd half converts need 1.94; rest of file is fine on 1.89. No scalar `bf16`/`f16` broadcast (types unstable). |
| `avx` | SHA512 | 1.89.0 | |
| `avx` | SM3 | 1.89.0 | |
| `avx` | SM4 | 1.89.0 | |
| `other` | AES | 1.27.0 | |
| `other` | PCLMULQDQ | 1.27.0 | |
| `other` | POPCNT | 1.27.0 | |
| `other` | SHA (SHA-1/256-NI) | 1.27.0 | Distinct from AVX SHA512. Feature `"sha"` only. |
| `other` | GFNI | 1.89.0 | 128: `"gfni"`; 256: `"gfni,avx"`; 512: `Gfni512` (`gfni`+`avx512f`, not assumed). |
| `other` | VAES | 1.89.0 | 256: `"vaes"`; 512: `Vaes512`. No 128 (AES-NI covers it). |
| `other` | VPCLMULQDQ | 1.89.0 | 256: `"vpclmulqdq"`; 512: `Vpclmulqdq512`. No 128 (PCLMUL covers it). |
| `other` | 3DNow! (`amd3dnow`) | 1.59.0 | `asm!`; no AMD CPUs since ~2011. |

## Optional / not default

These do **not** set the crate MSRV.

| Family | Extension | Min / gate | Notes |
|---|---|---|---|
| `avx512` | AVX512VP2INTERSECT | nightly, **not shipped** | All 6 intrinsics unstable (`stdarch_x86_avx512vp2intersect`, rust#111137). |
| `other` | RTM | **nightly** (`nightly-rtm`) | `stdarch_x86_rtm` + `rtm_target_feature`. HLE has no `core::arch` coverage. |
| `avx512` | AVX512_4FMAPS | 1.59.0 + `phi-asm` | Knights Mill only; `asm!` (`v4fmaddps`/`v4fnmaddps`); `detect()` always `None`. |
| `avx512` | AVX512_4VNNIW | 1.59.0 + `phi-asm` | Same as 4FMAPS; `vp4dpwssd`/`vp4dpwssds`. |
| `avx512` | AVX512PF | 1.59.0 + `phi-asm` | Knights Landing; prefetch hints only; first raw-pointer public API (`unsafe`). |
| `avx512` | AVX512ER | 1.59.0 + `phi-asm` | Knights Landing; `vrcp28`/`vrsqrt28`/`vexp2` packed. |

## `powerpc::ops`

| Extension | Min `rustc` | Notes |
|---|---|---|
| AltiVec | **nightly** (`nightly-altivec`) | `stdarch_powerpc` + `powerpc_target_feature`. |

## Adding a new extension

1. Grep each intrinsic against `core_arch` `#[stable(since = ...)]` / `#[unstable]`.
2. Take the max `since` across every intrinsic the new file calls.
3. If above current `rust-version`, bump [Cargo.toml](Cargo.toml) and confirm
   with `cargo clippy --lib --all-targets -- -D warnings`.
4. Add a row here; if the floor moves, note it in
   `notes/miraculix/design/Unsafe-Policy.md`.
