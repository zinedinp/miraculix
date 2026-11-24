# AArch32 (ARMv7)

**Module:** `miraculix::aarch32` · **Toolchain:** detection on stable; ops
need **nightly** + the `nightly-arm-neon` feature + a `v7` target.

`core::arch::arm`'s Neon and DSP intrinsics are still unstable, which is why
the ops are gated.

| Item | Role |
|---|---|
| `detect_level() -> FpuLevel` | `None`, `Vfpv3Neon`, `Vfpv4Neon`, `Crypto` |
| `detect_features()` | feature bitset (auxv on Linux, compile-time elsewhere) |
| `warm_up()`, `detect_*_fresh()`, `shortpath::verify_or_panic()` | as on x86 ([caching](../detection-and-caching.md)) |
| Tokens | `Neon`, `Vfpv4Neon`, `Dsp`, `Fp16Neon`, `Fp16Fma`, `Dotprod`, `I8mm`, `Aes`, `Sha`, `Crc32`, saturating ops |
| `auto_up` | Neon slice ops with the same semantics as [AArch64](aarch64.md) |

```toml
[dependencies]
miraculix = { version = "0.6.5-beta", features = ["nightly-arm-neon"] }
```
