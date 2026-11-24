# Architecture-independent modules

These exist on every target and need no token.

## `prefetch`

`miraculix::prefetch::{read, write, instruction}` with a `Locality` (`L1`,
`L2`, `L3`, `Streaming`).

```rust
use miraculix::prefetch::{read, Locality};
read(data.as_ptr().wrapping_add(256), Locality::L1);
```

Safe: a prefetch is a hint and never faults, even on a bad address.

| Target | Instructions |
|---|---|
| x86 | `prefetcht0/t1/t2/nta`, `prefetchw` (detected, PRFCHW) |
| AArch64 | `prfm` |
| ARM | `pld` / `pli` (with `nightly-arm-neon`) |
| RISC-V | Zicbop hints (no detection needed: they are `ori x0` encodings) |
| LoongArch64 | `preld` |
| PowerPC | `dcbt` / `dcbtst` |
| elsewhere | no-op |

## `formats`

Number formats used by matrix engines and quantised models, converted on the
CPU:

| Format | Where |
|---|---|
| FP8 (E4M3, E5M2), FP6, FP4, FP16, BF16 | `MiniFloat` |
| OCP MX, NVFP4 | `mx` (block size via the `*_block` functions) |
| Packed INT1 / INT2 / INT4 | `int` |
| TF32 | `tf32` |
| Tenstorrent `BFP_B` (`Bfp8_b`, `Bfp4_b`, `Bfp2_b`) | `bfp`, with AVX-512 and RVV fast paths; checked bit for bit against tt-metal via `scripts/tt-oracle` |
| GGUF `Q8_0`, `Q6_K` decode | `gguf`, checked against ggml via `scripts/gguf-oracle` |

## `mmio`

`miraculix::mmio::{copy_to_wc, copy_from_wc}` copy into and out of
write-combining memory (PCIe BARs of accelerator cards).

| Tier | Store / load body | Needs |
|---|---|---|
| `WcTier::Avx512` | 64-byte `vmovntdq` / `vmovntdqa` | AVX-512 |
| `WcTier::Avx2` | 32-byte non-temporal | AVX2 |
| `WcTier::Sse` | 16-byte non-temporal | SSE4.1 |
| `WcTier::Volatile` | volatile `u32` | any arch |

Every copy ends with one store fence after **all** its stores (`sfence` on
x86, `dmb oshst` on AArch64, `fence ow,ow` on RISC-V), so a doorbell write
you issue afterwards cannot overtake the data. `wc_tier()` tells you which
tier runs; `copy_*_with(tier, ..)` panics if the CPU lacks `tier`. These functions are
`unsafe`: you vouch that the device range is mapped and valid. They are for
memory behind the mapping (L1, GDDR); device registers need your own
dword-exact accesses.
