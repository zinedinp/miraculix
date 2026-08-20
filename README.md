# miraculix

miraculix is a Rust library for safe SIMD programming. Instead of manually
wrangling `#[target_feature]`, CPU feature checks, and raw intrinsics, it
lets you detect what the host supports at runtime and then choose the
appropriate fast path.

The main idea is simple: **detect the available CPU features once, then use
safe SIMD wrappers that either dispatch automatically or only run when the
required instruction set is actually present**. That keeps the fast path fast
without silently producing illegal instructions on unsupported hardware.

This is aimed at performance-sensitive numeric code: arrays, slices, vector
kernels, and other hot loops where SIMD helps but the usual unsafe setup is
more hassle than it is worth.

Still in Alpha. Expect unintended behaviour.

x86/x86_64 is the furthest developed part. Other platforms already have
detect support, and the most-used targets get richer operation support first.

## Installation

```toml
[dependencies]
miraculix = "0.1"
```

## Quick start: common SIMD patterns

These are the usual patterns in SIMD code: element-wise arithmetic, masked
selection, and clamping. In raw intrinsics, they become verbose and
architecture-specific. miraculix keeps the same ideas, but the call site stays
close to normal Rust.

### 1) Element-wise add / multiply

```rust
use miraculix::x86::auto_up;

let a = [1i32, 2, 3, 4, 5, 6, 7, 8];
let b = [10, 20, 30, 40, 50, 60, 70, 80];
let mut sum = [0i32; 8];
let mut product = [0i32; 8];

auto_up::add_i32(&a, &b, &mut sum);
auto_up::mul_i32(&a, &b, &mut product);

assert_eq!(sum, [11, 22, 33, 44, 55, 66, 77, 88]);
assert_eq!(product, [10, 40, 90, 160, 250, 360, 490, 640]);
```

### 2) Compare + select (masking)

```rust
use miraculix::x86::auto_up;

let x = [0i32, 5, 10, 15];
let y = [4i32, 4, 9, 20];
let mut mask = [0i32; 4];
let mut out = [0i32; 4];

auto_up::cmplt_i32(&x, &y, &mut mask);
auto_up::select_i32(&x, &y, &mask, &mut out);

assert_eq!(mask, [!0, 0, 0, !0]);
assert_eq!(out, [0, 4, 9, 15]);
```

### 3) Clamp to a valid range

```rust
use miraculix::x86::auto_up;

let v = [0i32, 8, 15, 30, 80];
let lo = [5i32; 5];
let hi = [20i32; 5];
let mut clamped = [0i32; 5];

auto_up::clamp_i32(&v, &lo, &hi, &mut clamped);

assert_eq!(clamped, [5, 8, 15, 20, 20]);
```

The point is not that miraculix removes SIMD entirely. It removes most of the
repetitive feature- and width-management boilerplate while keeping the
original SIMD pattern intact.

## Three ways to use it

If you are not sure which path to choose, this is the short version:

- **Auto slices**: you want fast element-wise operations on normal arrays or
  slices and do not care which exact ISA rung is used. This is the easiest
  path.
- **Tokens**: you know the ISA you need and want a fixed-width SIMD register
  API with explicit feature proof.
- **Kernel macros**: you need multiple SIMD operations in one compiled
  target-feature body, without building a messy chain of calls.

| Path | When to use | Entry point |
|---|---|---|
| **1. Auto slices** | "Just add these buffers as fast as possible" | `miraculix::x86::auto_up::*` (alias: `x86::auto`) |
| **2. Tokens** | You pick the ISA and call fixed-width ops | `Avx2::detect()`, then `t.add_i32x8(...)` |
| **3. Kernel macros** | Multi-op body must compile as real SIMD (no `callq` soup) | `miraculix::avx2_fn! { ... }` |

You do not have to hand-roll `#[target_feature]` + raw intrinsics for normal
work. Auto dispatch picks a safe tier for you, while tokens prove a feature is
live before the code runs.

---

## Path 1: Auto slice ops (easiest)

Pick an element type and an op. Pass equal-length slices. miraculix probes
once (cached), picks the best available SIMD tier, and falls back to scalar
when needed.

```rust
use miraculix::x86::auto_up;

let a = [1i32, 2, 3, 4];
let b = [10, 20, 30, 40];
let mut out = [0i32; 4];

auto_up::add_i32(&a, &b, &mut out);
assert_eq!(out, [11, 22, 33, 44]);
```

**Rules every auto fn shares**

- All input/output slices must have the **same length** (or documented length
  relationship for a few multi-slice ops). Mismatch panics.
- Integer add/sub/mul are **wrapping** unless the name says `adds`/`subs`
  (saturating).
- Compare ops write **lane masks** (all-1s or all-0s per element), not `bool`.
  Feed those masks into `select_*`.
- Float `min`/`max` on SIMD rungs use x86 **second-operand-on-NaN** (`minps`/
  `maxps` family), not Rust `f32::min`. Scalar remainder uses Rust `min`/`max`,
  so a short tail can differ on NaN. Ordered compares are false if either is
  NaN; `cmpeq` never matches NaN. Full note: `auto_up` module rustdoc.
- `f16` values are stored as raw `u16` bit patterns (IEEE half).
- Optional `wider-bus-lift` feature (default on) can run two narrower chains
  on a wider bus when it helps; you do not call anything extra for that.
- Dispatch uses the raw **feature bitset** (`detect_features`), not the coarse
  V1..V4 tier. A host with AVX-512F but incomplete V4 still gets the F path.

### Auto catalog (x86)

Call as `miraculix::x86::auto_up::NAME(...)`. Short alias: `miraculix::x86::auto::NAME(...)` (same module).

#### Arithmetic (per element)

| Call | Meaning |
|---|---|
| `add_{i,u}{8,16,32,64}`, `add_f{32,64}`, `add_f16` | `out = a + b` (int: wrapping) |
| `sub_{i,u}{8,16,32,64}`, `sub_f{32,64}`, `sub_f16` | `out = a - b` |
| `mul_{i,u}{8,16,32}`, `mul_f{32,64}`, `mul_f16` | `out = a * b` (int: wrapping) |
| `mullo_{i,u}64` | low 64 bits of 64x64 product |
| `div_{i,u}32`, `div_f{32,64}`, `div_f16` | `out = a / b` (int: not vectorized; panics like Rust `/`) |
| `adds_{i,u}{8,16}` | saturating add |
| `subs_{i,u}{8,16}` | saturating sub |
| `abs_i{8,16,32,64}`, `abs_f16` | absolute value (int: wrapping) |
| `min_*` / `max_*` | per-element min/max (many integer widths + f32/f64/f16) |
| `avg_u{8,16}` | rounded average `(a + b + 1) / 2` |
| `clamp_{i,u}{32,64}`, `clamp_f{32,64}` | `out = min(max(a, lo), hi)` |

#### Bitwise

| Call | Meaning |
|---|---|
| `and_*` / `or_*` / `xor_*` | `out = a &\| ^ b` (also bitwise on float bits) |
| `andnot_*` | `out = (!a) & b` |

#### Compare (lane masks)

| Call | Meaning |
|---|---|
| `cmpeq_*` | all-1s if equal, else 0 |
| `cmpgt_*` / `cmplt_*` / `cmpge_*` / `cmple_*` | ordered compares (float: false if either NaN) |

#### Select / blend

| Call | Meaning |
|---|---|
| `select_{i,u}{8,16,32,64}` | `out = mask != 0 ? b : a` |
| `select_f{32,64}` | float: **sign bit** of mask selects `b`, not "nonzero" |

Typical pipeline: `cmpeq_i32(&a, &b, &mut mask); select_i32(&x, &y, &mask, &mut out);`

#### Shifts

| Call | Meaning |
|---|---|
| `shl_*` / `shr_*` / `sra_*` with `<const IMM: u32>` | same shift amount for every lane |
| `sllv_*` / `srlv_*` / `srav_*` | per-lane shift counts from a second slice |

#### Fused multiply-add (float)

| Call | Meaning |
|---|---|
| `fmadd_f{32,64,16}` | `a*b + c` (hardware fused when FMA available) |
| `fmsub_*` | `a*b - c` |
| `fnmadd_*` | `-(a*b) + c` |
| `fnmsub_*` | `-(a*b) - c` |
| `fmaddsub_f16` / `fmsubadd_f16` | alternating add/sub by lane |

#### Converts / specialty

| Call | Meaning |
|---|---|
| `f16_to_f32` / `f32_to_f16` | half float widen/narrow |
| `popcnt_u{8,16,32,64}` | population count |
| `dpbf16_ps_f32`, `cvtneps_pbh_u16`, `cvtne2ps_pbh_u16` | BF16 helpers |
| `madd52lo_u64` / `madd52hi_u64` | 52-bit integer multiply-add |
| `dpbusd_i32` / `dpbusds_i32` / `dpwssd_i32` / `dpwssds_i32` | VNNI-style dot products |
| `p4fmadd_f32` / `p4fnmadd_f32` / `p4dpwssd_i32` / `p4dpwssds_i32` | Xeon Phi only (`phi-asm` feature) |

Exact cascade per op lives on each function's rustdoc (`cargo doc --open`).

---

## Path 2: Detect + tokens

Use this when you write a custom kernel or need fixed-width registers.

### Detect (every arch module)

Same shape on `x86`, `aarch64`, `aarch32`, `riscv`, `loongarch`, `powerpc`, `wasm`
(where the target exists):

| Call | What it does |
|---|---|
| `detect_level()` | Best **coarse tier** V1..V4 for this process (cached). Use for policy / "how good is this CPU?". |
| `detect_level_fresh()` | Same answer, but always re-probes. Tests / rare re-audit only. |
| `warm_up()` | Optional: fill both level and feature caches at startup. |
| `shortpath::verify_or_panic()` | If the binary was built for a high tier, panic if the CPU is weaker. |

**x86 only extras** (raw features, not coarse tiers):

| Call | What it does |
|---|---|
| `detect_features()` | Full `FeatureSet` bitset (cached). **What `auto_up` and tokens use.** Prefer this for fine-grained gates. |
| `detect_features_fresh()` | Fresh `CPUID` unioned with compile-time lower bound. |

**Level vs features vs auto:** `detect_level` folds many flags into V1..V4
(psABI buckets). `detect_features` keeps every extension bit. `auto_up::*`
always consults the feature set, so partial AVX-512 (or other non-bucket
flags) still light up. Use `detect_level` when you only need a coarse floor;
use `detect_features` / `Token::from_features` when you care about a specific
ISA bit.

```rust
use miraculix::x86::{detect_level, detect_features, warm_up, Feature, GenericLevel};

// Optional at app start:
warm_up();

let level = detect_level(); // GenericLevel::V1 .. V4
if level >= GenericLevel::V3 {
    // Host has the full x86-64-v3 bundle (AVX2, FMA, ...).
}

let set = detect_features();
if set.contains(Feature::Avx512f) {
    // Host has AVX-512F specifically (even if not full V4).
}
```

| Type | Role |
|---|---|
| `Feature` | One ISA flag (e.g. `Avx2`, `Sse41`) |
| `FeatureSet` | Bitset: `contains`, `contains_all`, `union`, `detect` |
| `GenericLevel` | Coarse psABI tier V1..V4 (`required_features`, `detect`) |
| `Avx10` | Presence / best-effort version helper |

Other arches export their own `Feature` / `FeatureSet` / `*Level` types from
their module (`aarch64::ArmLevel`, macOS `AppleLevel`, etc.).

### Tokens (x86 ops)

A token is a zero-sized proof value. You only get one if the feature is real.

```rust
use miraculix::x86::ops::avx::avx2::Avx2;

if let Some(t) = Avx2::detect() {
    // Safe: only runs if AVX2 is present.
    let sum = t.add_i32x8([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]);
    let _ = sum;
}

// Or from a FeatureSet you already have:
let set = miraculix::x86::detect_features();
if let Some(t) = Avx2::from_features(set) {
    let _ = t;
}
```

Common token constructors (every extension token follows this shape):

| Call | Meaning |
|---|---|
| `Token::detect()` | Probe now; `Some` if available |
| `Token::from_features(set)` | Use a set you already probed |
| `Token::from_level(level)` | When the tier implies the feature (where implemented) |

Layout mirrors the Intel Intrinsics Guide:

```
miraculix::x86::ops::
  mmx::  sse::{sse,sse2,sse3,ssse3,sse41,sse42}::
  avx::{avx,avx2,fma,f16c,...}::
  avx512::{avx512f,avx512bw,...}::
  other::{aes,sha,popcnt,...}::
```

Fixed-width methods look like `add_i32x8`, `loadu_si256`, etc. Slice helpers
on tokens are often named `add_i32_slice`. Prefer `auto_up` for "whole buffer"
work unless you need a specific width.

---

## Path 3: Multi-op kernel macros

Closures and separate token methods can leave a chain of calls. When several
ops must live in one `#[target_feature]` function, wrap the body:

```rust
use miraculix::x86::ops::avx::avx::Avx;

miraculix::avx_fn! {
    fn kernel(avx: Avx, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        let t = avx.add_f32x8(a, b);
        avx.mul_f32x8(t, c)
    }
}

if let Some(avx) = Avx::detect() {
    let _ = kernel(avx, [1.0; 8], [2.0; 8], [3.0; 8]);
}
```

| Macro | Enables (must hold matching tokens) |
|---|---|
| **SSE** | |
| `ssse3_fn!` | `ssse3` |
| `sse41_fn!` | `sse2,ssse3,sse4.1` |
| `sse41_f16c_fn!` | `sse2,ssse3,sse4.1,f16c` |
| `sse42_fn!` | `sse4.2` |
| **AVX / FMA / VNNI** | |
| `avx_fn!` | `avx` |
| `f16c_fn!` | `f16c` |
| `fma_fn!` | `fma` |
| `avx_fma_fn!` | `avx,fma` |
| `avx2_fn!` | `avx2,sse2,ssse3` |
| `avx2_fma_fn!` | `avx2,fma,sse2,ssse3` |
| `avx_v3_fn!` | x86-64-v3 SIMD core |
| `avx_vnni_fn!` | `avxvnni` |
| **AVX-512** | |
| `avx512f_fn!` | `avx512f` only |
| `avx512_fn!` | `avx512f,avx,avx512dq` |
| `avx512bw_fn!` | `avx512f,avx512bw,avx2,sse2,ssse3` |
| `avx512vl_fn!` | `avx512f,avx512vl` |
| `avx512vnni_fn!` | `avx512vnni` |
| `avx512vbmi2_fn!` | `avx512vbmi2` |
| `avx512fp16_fn!` | `avx512fp16` |
| `avx512bf16_fn!` | `avx512bf16,avx512f` |
| `avx512_v4_fn!` | full x86-64-v4 SIMD core |
| **Crypto / specialty** | |
| `aes_fn!` | `aes` |
| `pclmulqdq_fn!` | `pclmulqdq` |
| `aes_pclmul_fn!` | `aes,pclmulqdq` |
| `sha_fn!` | `sha` |
| `gfni_fn!` | `gfni` |
| `gfni_avx_fn!` | `gfni,avx` |
| `gfni512_fn!` | `gfni,avx512f` |
| `vaes_fn!` | `vaes` |
| `vaes512_fn!` | `vaes,avx512f` |

Only call the outer function when you already hold the tokens that prove those
features. That is the safety contract. Full feature-string detail and
selection rules live in the rustdoc for `x86::fn_macros`.

---

## Features (Cargo)

| Feature | Default | Purpose |
|---|---|---|
| `wider-bus-lift` | on | Extra dispatch rung: two narrower chains when a wider bus helps |
| `phi-asm` | off | Xeon Phi 4FMAPS / 4VNNIW / ER / PF ops |
| `nightly-altivec` | off | Power AltiVec via nightly stdarch |
| `nightly-rtm` | off | x86 RTM via nightly stdarch |

MSRV: **1.94.0** (see `RUST_VERSIONS.md` for per-extension floors).

---

## Safety model (short)

1. Public detect APIs and tokens are **safe**.
2. `unsafe` lives inside probes and `target_feature` bodies after a real check.
3. Missing feature => `None` token or narrower/scalar auto path. Never silent
   illegal instruction from a public auto call.
4. No app-wide `init` required: first `detect_*` fills a process cache.