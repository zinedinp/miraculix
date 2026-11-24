# Kernel macros (`*_fn!`, x86)

Write a multi-op SIMD kernel that compiles as **one** `#[target_feature]`
function, so every op inlines and the data stays in registers.

## Why you need them

Each token method is its own `#[target_feature]` function. Rust will not
inline such a function into a caller that lacks the same feature, so this:

```rust
fn kernel(avx: Avx, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
    let t = avx.add_f32x8(a, b); // call + store to stack
    avx.mul_f32x8(t, c)          // call + load from stack
}
```

becomes a chain of calls with memory round trips between them. Wrapping the
body in a macro gives the whole function the feature, so the ops inline:

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

## The one rule: the first parameter is the proof

The first parameter must be a token that proves every feature the macro
enables. Anything else is a **compile error**:

```rust,compile_fail
miraculix::avx512f_fn! {
    fn bad(avx: Avx, a: [f32; 8]) -> [f32; 8] { avx.add_f32x8(a, a) }
    //     ^^^ error: `Avx: Proves<Avx512fFn>` is not satisfied
}
```

That is what makes the generated function safe to call: you cannot reach it
without a token, and you cannot get a token without detection.

Pass any further tokens you want to call methods on as later parameters.
Their features are covered by the first one (or the macro enables them):

```rust
miraculix::avx2_fn! {
    fn mix(avx2: Avx2, sse2: Sse2, a: [u8; 32]) -> [u8; 32] {
        let lo = avx2.extract_u8x16_from_x32::<0>(a);
        let lo = sse2.slli_u8x16::<1>(lo);
        avx2.insert_u8x16_into_x32::<0>(a, lo)
    }
}
```

### What counts as proof

"Proves" follows LLVM's feature implications, the same rule
`#[target_feature]` itself uses:

- `avx2` implies `avx`, which implies the SSE family.
- `fma` and `f16c` imply `avx`.
- `avx512f` implies `avx2`, `fma` and `f16c`. `avx512vbmi` implies
  `avx512bw`. `avx512dq` implies `avx512f`.
- Level bundles (`V3`, `V4`, `V4Vbmi`) prove every macro of their level and
  below.

Two features with no implication between them take a **tuple**:

```rust
miraculix::avx2_fma_fn! {
    fn fused(p: (Avx2, Fma), a: [f32; 8]) -> [f32; 8] {
        let (avx2, fma) = p;
        let _ = avx2;
        fma.fmadd_f32x8(a, a, a)
    }
}
// call: fused((avx2, fma), a)  or use a V3 bundle as the first parameter
```

## Catalog

| Macro | Enables | First parameter |
|---|---|---|
| `ssse3_fn!` | `ssse3` | `Ssse3`, `Sse41`, `Sse42`, `Avx`, `Avx2`, a bundle |
| `sse41_fn!` | `sse2,ssse3,sse4.1` | `Sse41`, `Sse42`, `Avx`, `Avx2`, a bundle |
| `sse41_f16c_fn!` | `sse2,ssse3,sse4.1,f16c` | `F16c`, a bundle |
| `sse42_fn!` | `sse4.2` | `Sse42`, `Avx`, `Avx2`, a bundle |
| `avx_fn!` | `avx` | `Avx`, `Avx2`, `Fma`, `F16c`, a bundle |
| `f16c_fn!` | `f16c` | `F16c`, a bundle |
| `fma_fn!` | `fma` | `Fma`, a bundle |
| `avx_fma_fn!` | `avx,fma` | `Fma`, a bundle |
| `avx2_fn!` | `avx2,sse2,ssse3` | `Avx2`, a bundle |
| `avx2_fma_fn!` | `avx2,fma,sse2,ssse3` | `(Avx2, Fma)`, a bundle |
| `avx_v3_fn!` | x86-64-v3 core | `V3`, `V4`, `V4Vbmi` |
| `avx_vnni_fn!` | `avxvnni` | `AvxVnni` |
| `avx512f_fn!` | `avx512f` | `Avx512f`, `V4`, `V4Vbmi` |
| `avx512_fn!` | `avx512f,avx,avx512dq` | `Avx512Dq`, `V4`, `V4Vbmi` |
| `avx512bw_fn!` | `avx512f,avx512bw,avx2,sse2,ssse3` | `Avx512Bw`, `V4`, `V4Vbmi` |
| `avx512vl_fn!` | `avx512f,avx512vl` | `Avx512FVl`, `V4`, `V4Vbmi` |
| `avx512vnni_fn!` | `avx512vnni` | `Avx512Vnni` |
| `avx512vbmi_fn!` | `avx512f,avx512bw,avx512vbmi` | `Avx512Vbmi`, `V4Vbmi` |
| `avx512vbmi2_fn!` | `avx512vbmi2` | `Avx512Vbmi2` |
| `avx512fp16_fn!` | `avx512fp16` | `Avx512Fp16` |
| `avx512bf16_fn!` | `avx512bf16,avx512f` | `Avx512Bf16` |
| `avx512_v4_fn!` | x86-64-v4 core | `V4`, `V4Vbmi` |
| `avx512_v4_vbmi_fn!` | v4 core + `avx512vbmi` | `V4Vbmi` |
| `aes_fn!` | `aes` | `Aes` |
| `pclmulqdq_fn!` | `pclmulqdq` | `Pclmulqdq` |
| `aes_pclmul_fn!` | `aes,pclmulqdq` | `(Aes, Pclmulqdq)` |
| `sha_fn!` | `sha` | `Sha` |
| `gfni_fn!` | `gfni` | `Gfni` |
| `gfni_avx_fn!` | `gfni,avx` | `(Gfni, Avx)`: GFNI exists without AVX (Tremont, Jasper Lake) |
| `gfni512_fn!` | `gfni,avx512f` | `Gfni512` |
| `vaes_fn!` | `vaes` | `Vaes` |
| `vaes512_fn!` | `vaes,avx512f` | `Vaes512` |

Pick the **narrowest** macro that covers what the body calls. The full
feature strings are in the rustdoc of `x86::fn_macros`.

## Writing a good kernel

- **Loop inside the kernel**, not around it. One call per buffer, not one
  per 8 elements.
- **Use plain `while`/`for` loops.** Closures and `[T; N].map` can stop the
  inlining you wrapped the body for.
- **Handle the tail yourself.** Fixed-width methods take full arrays; finish
  the last `len % lanes` elements with scalar code.
- **Branch on the token outside the loop.** Get the token once at startup
  (see [detection](detection-and-caching.md)), then call the kernel.

```rust
miraculix::avx2_fn! {
    fn add_rows(t: Avx2, a: &[i32], b: &[i32], out: &mut [i32]) {
        let n = a.len() / 8 * 8;
        let mut i = 0;
        while i < n {
            let x: [i32; 8] = a[i..i + 8].try_into().unwrap();
            let y: [i32; 8] = b[i..i + 8].try_into().unwrap();
            out[i..i + 8].copy_from_slice(&t.add_i32x8(x, y));
            i += 8;
        }
        for j in n..a.len() {
            out[j] = a[j].wrapping_add(b[j]);
        }
    }
}
```

## Generics

Kernels can be generic over const parameters, lifetimes and type parameters
with a single bound:

```rust
miraculix::avx2_fn! {
    fn splat<const K: usize>(t: Avx2, a: [u32; K]) -> [u32; K] { let _ = t; a }
}
```

Multi-bound lists (`T: A + B`) and `where` clauses are not supported.

## Missing a combination?

Use the next wider macro your first parameter proves, or open a PR that adds
a named macro, its marker, and the token types that prove it (in
`x86::fn_macros::__private`).
