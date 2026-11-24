# Feature detection and caching

How miraculix finds out what the CPU can do, what each check costs, and how
to structure a program so the check happens once instead of in your hot
loop.

## The short version

```rust
fn main() {
    miraculix::x86::warm_up();                    // probe once, fill the caches
    miraculix::x86::shortpath::verify_or_panic(); // only matters for target-cpu builds
    let simd = Simd::init();                      // resolve tokens once (below)
    run(simd);                                    // pass them down by value
}
```

Then, inside hot loops, branch on a token you already hold and call a
[kernel](kernel-macros.md). Do not call `auto_up::*` once per tiny slice.

## What detection checks

A feature counts as present only if **both** are true:

1. **The CPU has it.** x86: `CPUID`. Arm, RISC-V, LoongArch, PowerPC: the
   OS's view of the hardware (`getauxval`, `elf_aux_info`, `sysctlbyname` on
   every Apple OS, `IsProcessorFeaturePresent`, `riscv_hwprobe`).
2. **The OS has enabled it.** On x86 this is the `XCR0` register (read with
   `XGETBV`). A CPU can report AVX-512 while the OS (or a VM) has left the
   wide register state off; the instructions then fault. miraculix checks
   this the same way `std::is_x86_feature_detected!` does:

| Extensions | Needs these `XCR0` bits |
|---|---|
| SSE family, AES, PCLMULQDQ, SHA, GFNI (128-bit) | none |
| AVX, AVX2, FMA, F16C, VAES, VPCLMULQDQ, AVX-VNNI, AVX-IFMA, SHA512, SM3, SM4, ... | XMM + YMM (`0b110`) |
| Every AVX-512 extension, AVX10 | + opmask, ZMM (`0b1110_0110`) |
| AMX | + tile config and data (bits 17, 18) |

**Intel Macs** enable AVX-512 state lazily (on a thread's first AVX-512
instruction), so `XCR0` cannot be trusted there. miraculix asks the kernel
instead, like Go's runtime: AVX-512 counts only if AVX state is on,
`hw.optional.avx512f` is 1, and the kernel is at least 21.3.0 (macOS 12.2;
earlier kernels can corrupt the `k0..k7` mask registers on signal return).
Under Rosetta 2 the kernel reports no AVX-512. Linux additionally
needs `arch_prctl(ARCH_REQ_XCOMP_PERM)` before AMX instructions; miraculix
has no AMX ops yet.

The result is **unioned with the compile-time features**: anything you built
with (`-C target-feature=+avx2`, `-C target-cpu=...`) counts as present,
because the whole binary already assumes it.

## What each call costs

| Call | First call | Every later call |
|---|---|---|
| `detect_level()` | probe | one atomic byte load |
| `detect_features()` | probe | a few atomic loads + OR with the compile-time set |
| `Token::detect()` | via `detect_features()` | `detect_features()` + a feature check |
| `Token::from_features(set)` | - | a feature check on the value you pass, no shared state |
| `auto_up::*`, `reduce::*` | via `detect_features()` | `detect_features()` + an `if` cascade, **every call** |
| `reduce::*_with(tier, ..)` | - | nothing: the tier was resolved earlier |
| `*_fresh()` | full probe | full probe (never cached) |

Detection is cheap, but not free. For large slices the auto-dispatch cost is
invisible. For thousands of tiny calls inside a loop it can be larger than
the work.

## Caching at program start

### 1. Fill the caches

```rust
miraculix::x86::warm_up(); // or aarch64::warm_up(), riscv::warm_up(), ...
```

Optional: without it the first `detect_*` call fills the cache. Calling it at
startup moves that one-time probe out of your first latency-sensitive path.
Safe to call any number of times, from any thread.

### 2. Guard `target-cpu` builds

If you compile with `-C target-cpu=x86-64-v3` (or similar), the binary uses
those instructions everywhere and would crash with `SIGILL` on an older CPU.
`shortpath::verify_or_panic()` turns that into a clear panic at startup. It
does nothing in a build without such flags.

### 3. Resolve tokens once

Tokens are zero-sized and `Copy`. Put the ones you need in a struct and pass
it by value. This costs nothing at runtime.

```rust
use miraculix::x86::{self, ops::avx::avx2::Avx2, ops::avx512::avx512bw::Avx512Bw};

#[derive(Clone, Copy)]
pub struct Simd {
    pub avx2: Option<Avx2>,
    pub avx512bw: Option<Avx512Bw>,
    pub reduce: x86::reduce::Tier,
}

impl Simd {
    pub fn init() -> Self {
        x86::warm_up();
        x86::shortpath::verify_or_panic();
        let f = x86::detect_features(); // one snapshot
        Self {
            avx2: Avx2::from_features(f),
            avx512bw: Avx512Bw::from_features(f),
            reduce: x86::reduce::best_tier(),
        }
    }
}
```

Bundles save typing when you want a whole level: `x86::bundles::V3::detect()`
gives `v.avx2`, `v.fma`, `v.f16c`, ... in one go.

### 4. Branch once per loop, not once per element

```rust
fn process(simd: Simd, rows: &[[f32; 64]], out: &mut [f32]) {
    // Reductions: the tier was resolved at startup, no detection here.
    for (r, o) in rows.iter().zip(out) {
        *o = miraculix::x86::reduce::dot_f32_with(simd.reduce, r, r);
    }
}
```

For element-wise work, branch on the token outside the loop and run a
[`*_fn!` kernel](kernel-macros.md) inside it.

## Resolve-once APIs per architecture

| Arch | Resolve once | Use many times |
|---|---|---|
| x86 | `Token::from_features(detect_features())`, `bundles::V3/V4::detect()` | token methods, `*_fn!` kernels |
| x86 | `reduce::best_tier()` (or `available_tiers()`) | `reduce::*_with(tier, ..)` |
| AArch64 | `Token::from_features(detect_features())` | token methods |
| AArch64 | `Neon::detect()` | `reduce::*_with(neon, ..)` (`None` = portable path) |
| RISC-V | `Rvv::from_features(detect_features())` | `Rvv` methods |
| PowerPC | `Vsx::from_features(detect_features())` (also `Isa300`, `Mma`, `Vmx`, `PairedSingle`, `Vmx128`) | token methods |
| wasm | `Simd128::from_features(detect_features())` | token methods |

A `reduce::Tier` and a `Neon` token can only come from detection, so the
`_with` functions are safe: they cannot be told to run a tier the CPU lacks.

## `detect_level` vs `detect_features`

- **`detect_level()`** returns a coarse tier: x86 `V1..V4`, Arm
  `V8_0..V9_5`, RISC-V `Gc`/`Vector`, ... Good for logging and policy ("this
  machine is x86-64-v3").
- **`detect_features()`** returns every extension bit. Use it to build
  tokens. `auto_up` dispatches on it, so a CPU with AVX-512F but not the full
  V4 bundle still gets the AVX-512 path.

Two things to know about `detect_level()`:

- In a `-C target-cpu=...` build the compiled-in level is a **floor**: the
  result is the higher of that and what the host reports.
- On AArch64 its type depends on the OS: `ArmLevel` (Linux, Android, BSD,
  iOS), `AppleLevel` (macOS), `SnapdragonLevel` (Windows). Portable code
  should gate on `detect_features()` and tokens instead.

## Fresh probes

`detect_features_fresh()` and `detect_level_fresh()` skip the cache and probe
again. They are for tests and diagnostics. To build a token from a fresh
probe: `Avx2::from_features(miraculix::x86::detect_features_fresh())`.

## Threads and races

The caches are process-wide atomics. If several threads hit the first call
at once, each may probe, and all store the same answer (the CPU does not
change between them). No lock and no `init` call are needed.

## One codebase, several architectures

Each arch module exists only on its own target, so gate with `cfg`:

```rust
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    miraculix::x86::auto_up::add_f32(a, b, out);
    #[cfg(target_arch = "aarch64")]
    miraculix::aarch64::auto_up::add_f32(a, b, out);
    #[cfg(target_arch = "wasm32")]
    miraculix::wasm::auto_up::add_f32(a, b, out);
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    miraculix::riscv::auto::add_f32(a, b, out);
}
```

Before sharing code across architectures, read
[cross-architecture differences](cross-arch-semantics.md): a few ops share a
name but not a meaning.
