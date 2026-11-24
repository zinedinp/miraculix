# miraculix

Safe SIMD for Rust with runtime CPU detection. Detect what the CPU can do
**once**, then call ready-made slice functions that pick the fastest path, or
write your own kernels that only run where the instructions really exist. No
hand-written `#[target_feature]`, no raw intrinsics, no illegal-instruction
crashes on older hardware.

`no_std`, no allocation. The only dependencies are the OS probes (`raw-cpuid`,
`libc`, `windows-sys`, per target).

> **Status: beta (0.6).** APIs still change between minor versions. x86 is
> the most complete platform, AArch64 is close behind.

## Install

```toml
[dependencies]
miraculix = "0.6.5-beta"
```

MSRV **1.100.0**. Some extensions need a newer `rustc`, and a few need
nightly: see [RUST_VERSIONS.md](RUST_VERSIONS.md).

## Start here

| I want to... | Read |
|---|---|
| Run whole-slice math as fast as the CPU allows | [Auto slice ops](#1-auto-slice-ops), below |
| Sum / argmax / top-k / dot / softmax a slice | [Reductions](#2-reductions), below |
| Detect once at startup and keep hot loops check-free | [Detection and caching](doc/detection-and-caching.md) |
| Write my own kernel for AVX2 / AVX-512 / Neon / ... | [Tokens](#3-tokens) + [kernel macros](doc/kernel-macros.md) |
| Share code between x86 and Arm | [Cross-arch differences](doc/cross-arch-semantics.md) |
| Find the miraculix name for an intrinsic | [Op finder](doc/op-finder.md) |
| Know why it cannot crash on old CPUs | [Safety model](doc/safety-model.md) |

All topics: [doc/](doc/README.md).

## Platforms

| Platform | Detect | Auto slices | Reductions | Tokens | Kernel macros |
|---|---|---|---|---|---|
| [x86 / x86_64](doc/platforms/x86.md) ([AVX-512](doc/platforms/avx512.md)) | ✓ | ✓ | ✓ | SSE..AVX-512, AVX10, crypto | ✓ |
| [AArch64](doc/platforms/aarch64.md) | ✓ | ✓ Neon + SVE | ✓ | Neon, v8.x, SVE/SVE2, crypto | - |
| [AArch32](doc/platforms/aarch32.md) | ✓ | ✓ nightly | - | Neon, VFPv4, DSP (nightly) | - |
| [RISC-V](doc/platforms/riscv.md) | ✓ | f32/i32 core | ✓ | `Rvv`, `Zvfh`, `Zvfbf*` | - |
| [WebAssembly](doc/platforms/wasm.md) | host flags | ✓ | - | `Simd128`, `RelaxedSimd` | - |
| [PowerPC](doc/platforms/powerpc.md) (consoles, G4/G5, POWER) | ✓ | ✓ VMX, paired singles | ✓ | `Vmx`, `PairedSingle`, `Vmx128` | - |
| [LoongArch](doc/platforms/loongarch.md) | ✓ | - | - | - | - |

Every target also gets [`prefetch`, `formats` and `mmio`](doc/modules.md).

## The four ways to use it

### 1) Auto slice ops

Pass equal-length slices. miraculix picks the best SIMD tier and falls back
to narrower SIMD, then scalar.

```rust
use miraculix::x86::auto_up; // or aarch64::auto_up, wasm::auto_up, riscv::auto

let a = [1i32, 2, 3, 4, 5, 6, 7, 8];
let b = [10, 20, 30, 40, 50, 60, 70, 80];
let mut out = [0i32; 8];
auto_up::add_i32(&a, &b, &mut out);

let mut mask = [0i32; 8];
auto_up::cmplt_i32(&a, &[4; 8], &mut mask);   // lane masks, not bool
auto_up::select_i32(&a, &b, &mask, &mut out); // mask != 0 ? b : a
```

Full catalog per platform: [x86](doc/platforms/x86.md#catalog),
[AArch64](doc/platforms/aarch64.md#auto-slices).

### 2) Reductions

Whole-slice functions that return a value or an index. Same names on
`x86::reduce`, `aarch64::reduce` and `riscv::auto`.

```rust
use miraculix::x86::reduce;

let logits = [0.5f32, -1.0, 7.25, 3.0, f32::NAN];
assert_eq!(reduce::argmax_f32(&logits), Some((2, 7.25))); // NaN ignored

let tier = reduce::best_tier();                       // resolve once...
let d = reduce::dot_f32_with(tier, &[1.0, 2.0], &[3.0, 4.0]); // ...reuse in a hot loop
```

`reduce_sum/max/min_f32`, `argmax_f32`, `topk_f32`, `compress_gt_f32`,
`exp_f32`, `logsumexp_f32`, `log_softmax_f32`, `dot_f32`, `sliding_dot_f32`,
`affine_u16_f32`, `lerp_u8`, `bf16_to_f32`, `gemm_f32_kernel`.

### 3) Tokens

A token is a zero-sized, `Copy` proof that a feature is present. You only
get one from detection, so its methods are safe to call.

```rust
use miraculix::x86::ops::avx::avx2::Avx2;

if let Some(t) = Avx2::detect() {
    let v = t.add_i32x8([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]);
}
```

Every token has `detect()` (cached) and `from_features(set)` (from a set you
already hold). Methods are `<op>_<type>x<lanes>` on arrays.

### 4) Kernel macros (x86)

Several token calls in a row from normal code cost one call and a memory
round trip each. A `*_fn!` macro compiles the whole body as one
`#[target_feature]` function, so it all inlines. The **first parameter must
be the proving token**; anything else does not compile.

```rust
use miraculix::x86::ops::avx::avx::Avx;

miraculix::avx_fn! {
    fn kernel(avx: Avx, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        let t = avx.add_f32x8(a, b);
        avx.mul_f32x8(t, c)
    }
}

if let Some(avx) = Avx::detect() {
    let r = kernel(avx, [1.0; 8], [2.0; 8], [3.0; 8]);
}
```

Catalog and patterns: [kernel macros](doc/kernel-macros.md).

## Detect once at startup

Detection is cached, but `auto_up` and `reduce` still check the cache on
**every** call. That is invisible for big slices and noticeable for many tiny
ones. The recommended shape:

```rust
fn main() {
    miraculix::x86::warm_up();                    // 1. probe once
    miraculix::x86::shortpath::verify_or_panic(); // 2. guard target-cpu builds
    let f = miraculix::x86::detect_features();    // 3. resolve tokens once...
    let avx2 = Avx2::from_features(f);            //    ...and pass them down
    let tier = miraculix::x86::reduce::best_tier();
    // 4. in hot loops: branch on `avx2` once, call a kernel / `*_with(tier, ..)`
}
```

Cost table, a full `Simd` struct example, and per-arch details:
[detection and caching](doc/detection-and-caching.md).

## Cargo features

| Feature | Default | Purpose |
|---|---|---|
| `wider-bus-lift` | on | Experimental extra `auto_up` rung on x86 |
| `wasm-simd` | on | wasm: compile `wasm::ops` / `wasm::auto`. Off for modules that must load without simd128 |
| `wasm-relaxed-simd` | off | wasm relaxed-SIMD bodies |
| `phi-asm` | off | Xeon Phi 4FMAPS / 4VNNIW / ER / PF ops |
| `nightly-arm-neon` | off | AArch32 Neon / DSP ops and ARM `prefetch` (nightly) |
| `nightly-altivec` | off | PowerPC AltiVec intrinsic wrappers (nightly; the `asm!` kernels need nothing) |
| `nightly-rtm` | off | x86 RTM / TSX (nightly) |

## Safety in one paragraph

Detection checks both the CPU and the OS (x86 reads `XCR0`, like
`std::is_x86_feature_detected!`). Tokens, `reduce::Tier` and the level
bundles can only be obtained from detection. Kernel macros only compile with
a proving token as the first parameter. So **no safe call runs an
instruction the machine lacks**. The `unsafe` left to you is raw device
memory in `mmio`. Details: [safety model](doc/safety-model.md).

## License

MIT OR Apache-2.0.
