# WebAssembly

**Module:** `miraculix::wasm` · **Toolchain:** stable · **Ops:** wasm32.

## Turning SIMD on

Wasm code cannot ask the engine which proposals it supports, so **the host
turns SIMD on** after checking:

```rust
#[cfg(target_arch = "wasm32")]
{
    miraculix::wasm::enable_simd128(); // e.g. from an export the JS / wasmtime host calls
    let mut out = [0f32; 3];
    miraculix::wasm::auto::add_f32(&[1.0, 2.0, 3.0], &[0.5; 3], &mut out);
}
```

Building with `-C target-feature=+simd128` turns it on at compile time
instead.

| Item | Role |
|---|---|
| `enable_simd128()`, `enable_relaxed_simd()` (+ `disable_*`, `is_*_enabled`) | runtime switches |
| `detect_level() -> WasmLevel` | `Scalar`, `Simd128`, `RelaxedSimd`. **Not cached**, so `enable_*` takes effect immediately |
| `detect_features()` | the current flags; also not cached |
| Tokens | `ops::simd128::Simd128`, `ops::relaxed::RelaxedSimd` |
| `auto_up` (alias `auto`) | ~200 element-wise ops with Arm-style names (`qadd`, `sum_*`) |

`warm_up()` and `shortpath::verify_or_panic()` exist for API parity and do
nothing.

## Module validation

LLVM has no per-function target features on wasm. With the default
`wasm-simd` feature the whole crate is compiled with simd128, so an engine
without simd128 rejects the **module at load time**, even if no SIMD code
would run. That is also why the enable flag is safe: if the module loaded,
the instructions exist.

For a module that must load everywhere, build miraculix with
`default-features = false`. Detection stays; the SIMD ops go. Relaxed-SIMD
bodies need the `wasm-relaxed-simd` feature or `+relaxed-simd`, for the same
reason.

Tested under node (V8, wasip1) and wasmtime (wasip1, wasip2, wasip3).
