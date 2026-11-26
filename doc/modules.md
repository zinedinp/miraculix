# Architecture-independent modules

These exist on every target and need no token.

## `cache`

Cache sizes per core class, detected once and cached like the SIMD features:
`cache::detect()` (process-wide, one atomic load after the first call),
`cache::detect_fresh()`, `cache::warm_up()`.

The result is a `CacheMatrix`: one row (`CoreClass`) per distinct cache
configuration, one column per `Level` (`L1d`, `L1i`, `L2`, `L3`). Each cell
is a `Cache` with `size`, `line`, `ways` and `shared_by` (logical CPUs
sharing the instance). Identical rows merge, so a symmetric CPU has one row.
Asymmetric CPUs get several: Intel P/E cores, Arm big.LITTLE, Apple P/E
clusters, AMD X3D with one stacked CCD.

```rust
use miraculix::cache::{self, Level};

let m = cache::detect();
// Fits on every core class, even with SMT siblings busy.
let l1d = m.min_per_thread(Level::L1d).unwrap_or(32 * 1024);
// Huffman decode table: 4-byte entries in half of L1d.
let bits = cache::table_bits(l1d / 2, 4).clamp(9, 13);
println!("{m}"); // prints the matrix as a table
```

| Target | Source | Rows |
|---|---|---|
| Linux / Android (any arch) | sysfs `cpuN/cache/indexM/` | every online CPU |
| Windows | `GetLogicalProcessorInformationEx(RelationCache)` | every CPU |
| macOS / iOS | `sysctlbyname("hw.perflevelN.*")` | one per performance level |
| x86 fallback, no OS | `CPUID` leaf 4 / `0x8000_001D` / AMD `0x8000_0005..6` | calling core only |
| PowerPC Linux without sysfs | auxv `AT_L1I/L1D/L2/L3_CACHE*` | one |
| wasm, other bare metal | - | none (`CacheMatrix::EMPTY`; fill one with `CacheMatrix::push`) |

Things to know:

- `CPUID` describes only the core it runs on. On a hybrid Intel CPU without
  an OS source the matrix has one row and `is_complete()` is `false`. Its
  `shared_by` is the maximum the core *could* share with (it counts SMT
  threads even with SMT off).
- Row order is detection order and does not say which class is faster.
- `min_size`, `max_size`, `min_per_thread` and `line_size` cover every CPU,
  even past `MAX_CLASSES` (8) rows.
- Apple reports no associativity (`ways == 0`) and no SLC; Arm device trees
  sometimes report levels without sizes. Unknown fields are `0`.

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
