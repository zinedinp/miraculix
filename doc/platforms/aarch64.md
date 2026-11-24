# AArch64 (Neon / SVE)

**Module:** `miraculix::aarch64` · **Toolchain:** stable · **Status:** close
behind x86; SME is detect-only.

**Probe:** `getauxval` (Linux, Android), `elf_aux_info` (FreeBSD),
`sysctlbyname` (macOS, iOS, tvOS, visionOS), `IsProcessorFeaturePresent`
(Windows; FP16, I8MM, BF16, SHA3/512 and the SVE family included), compile-time
features elsewhere. Always unioned with the compile-time features.

## Detect

| Item | Role |
|---|---|
| `detect_features() -> FeatureSet` | every extension bit, cached |
| `detect_level()` | `ArmLevel` (`V8_0..V9_5`, also iOS); `AppleLevel` on macOS; `SnapdragonLevel` on Windows |
| `warm_up()`, `detect_*_fresh()`, `shortpath::verify_or_panic()` | as on x86 ([caching](../detection-and-caching.md)) |

Because `detect_level()`'s type changes with the OS, portable code should
gate on `detect_features()` and tokens.

## Auto slices

`aarch64::auto_up` (alias `auto`). Neon is mandatory on AArch64, so it is
the floor and needs no probe. SVE / SVE2 rungs sit above it and run with no
scalar tail (`whilelo` predication). `Fp16Neon` adds a native `f16` rung.

Covers, for every integer width and `f32`/`f64`:

- `add`, `sub`, `mul`, `qadd`, `qsub`, `hadd`, `hsub`, `avg`
- bitwise ops, `min`, `max`, `abs`, `neg`
- compares and `select`
- shifts: `shl` (per-lane count) and `slli`/`srli`/`srai` (immediate)
- FMA, and `sum_*`, `maxv_*`, `minv_*` reductions
- `clamp_*` and int/float converts (`cvt_f32_to_i32`, ...)
- `f16_to_f32`, `f32_to_f16`, `fmadd_f16`

Several of these differ from the x86 ops of the same name (`andnot`,
`select`, `shl`). See [cross-architecture differences](../cross-arch-semantics.md).

**SVE vs Neon:** `auto_up` takes the SVE rung whenever SVE is present. A
128-bit SVE length is normal (Neoverse N2/V2, Grace, Graviton4, recent
Cortex-A) and fully supported: every SVE op is `whilelo`-predicated, so it is
exact at any length, and no code assumes a particular one. At 128 bits SVE is
no wider than Neon and can be slightly slower. When you pick a path by hand,
use `SveVl256::detect()` to take SVE only where it is actually wider.

## Reductions

`aarch64::reduce` has the same names and rules as `x86::reduce`
(`reduce_sum_f32`, `argmax_f32`, `topk_f32`, `dot_f32`, `exp_f32`, ...).
Each has a `*_with(neon, ..)` twin: pass `Neon::detect()` once, or `None` for
the portable path.

## Tokens

| Area | Tokens (in `aarch64::ops::...`) |
|---|---|
| Neon base | `neon::neon::Neon`, plus x86-named helpers in `neon::extra` (`movemask_i8x16`, `tbl_u8x16`, ...) |
| Neon extensions | `Dotprod`, `I8mm`, `Bf16`, `Fp16Neon`, `Fhm`, `Rdm`, `Fcma`, `Frintts`, `Jscvt`, `Fp8` |
| Crypto | `Aes`, `Pmull`, `Sha` (SHA1 + SHA2), `Sha3`, `Sha512`, `Sm3`, `Sm4` |
| Other | `Crc32`, `Rng` |
| SVE | `Sve`, `Sve2`, `Sve2Aes`, `Sve2Pmull`, `Sve2Bitperm`, `SveBf16`, `SveI8mm`, `SveF32mm`, `SveF64mm` |
| SVE width | `SveVl128`, `SveVl256`, `SveVl512`: prove SVE **and** at least that vector length. `Sve::vector_bits()` reads the calling thread's length (`cntb`, not cached: Linux sets it per thread) |

There are no kernel macros for Arm yet; token methods are called directly.

## Big-endian

Big-endian AArch64 builds without a feature flag (`cfg(target_endian)`
selects the lane-order fixes). Checked under `qemu-aarch64_be` with
`scripts/be-probe` (`cd scripts/be-probe && cargo +nightly run`): the Neon
extras and a sample of element-wise ops pass. Widen/narrow halves, zips and
pairwise ops in `neon.rs` are not yet audited for lane order there.

## Testing

Verified under QEMU-user, including SVE at vector lengths 128, 256 and 512
(`max,sveNNN=on`, `neoverse-v1`, `a64fx`). See [testing](../testing.md).
