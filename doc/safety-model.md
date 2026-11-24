# Safety model

What miraculix guarantees, how it enforces it, and where `unsafe` is still
yours to handle.

## The guarantee

**No safe call runs an instruction the CPU or OS does not support.** A
missing feature gives you `None`, a narrower SIMD path, or scalar code, never
`SIGILL`.

## How: tokens are proofs

A token (`Avx2`, `Neon`, `Rvv`, ...) is a zero-sized value with a private
field. The only ways to get one are its constructors, and they all check
detection:

| Constructor | Returns `Some` when |
|---|---|
| `Token::detect()` | the cached feature set has the feature |
| `Token::from_features(set)` | `set` has it. A `FeatureSet` only comes from detection; you cannot add bits to one |
| `Token::from_level(level)` (x86) | `level` implies the feature **and** the process really runs at `level` or above |

Holding a token is therefore proof, and every method on it is safe.

Detection itself checks both the CPU and the OS (x86 `XGETBV`, see
[detection](detection-and-caching.md#what-detection-checks)).

## Other proof types

The same rule covers the other places that pick a code path for you:

| Type | Comes from | Used by |
|---|---|---|
| `x86::reduce::Tier` | `reduce::best_tier()`, `reduce::available_tiers()` | `x86::reduce::*_with(tier, ..)` |
| `Option<Neon>` | `Neon::detect()` | `aarch64::reduce::*_with(neon, ..)` |
| `x86::bundles::{V3, V4, V4Vbmi}` | `detect()` / `from_features()` | bundle fields, `*_fn!` kernels |
| `mmio::WcTier` | any value; `copy_*_with` **asserts** it is available | `mmio::copy_to_wc_with` (an `unsafe fn` anyway) |

None of them can be built by hand. `Tier(...)` does not compile, because its
field is private.

## Kernel macros

`avx2_fn!` and friends wrap a body in a `#[target_feature]` function and
give you a **safe** wrapper. That is only sound if the wrapper cannot be
called without proof, so:

- **The first parameter must be a token that proves every feature the
  macro enables.** This is checked at compile time through a sealed trait;
  nothing outside the crate can implement it.
- A kernel with no parameters, or with a non-token first parameter, does not
  compile.
- "Proves" follows LLVM's feature implications, the same rule
  `#[target_feature]` uses (`avx2` implies `avx`, `avx512f` implies `avx2`,
  `fma` and `f16c`, ...). Features with no implication between them take a
  tuple: `(Avx2, Fma)`, `(Gfni, Avx)`.

Details: [kernel macros](kernel-macros.md).

## Where `unsafe` stays with you

| API | Why it is `unsafe` |
|---|---|
| `mmio::copy_to_wc`, `copy_from_wc` (+ `_with`) | raw device pointers: you vouch that the range is mapped and valid |
| `prefetch::*` | **safe**: a prefetch never faults, even on a bad address |

## Trust you hand to the host (wasm)

WebAssembly cannot ask the engine which proposals it supports, so the host
calls `wasm::enable_simd128()` after checking. This is not a soundness hole:
an engine without simd128 refuses to **load** a module that contains SIMD
opcodes at all. If the module loaded, the instructions exist. The flag only
picks which code runs. See [WebAssembly](platforms/wasm.md).

## Compile-time builds

Building with `-C target-feature=...` or `-C target-cpu=...` tells the
compiler it may use those instructions **anywhere** in the binary. miraculix
treats those features as present (detection is unioned with them), exactly
like the rest of Rust does. Running such a binary on an older CPU is a
deployment error; call `shortpath::verify_or_panic()` at startup to catch it
with a clear message.

## Not public API

Items marked `#[doc(hidden)]` (for example `__miraculix_tf_fn!` and
`x86::fn_macros::__private`) exist for the crate's own macro expansions.
They are not covered by semver and not part of the guarantee above.
