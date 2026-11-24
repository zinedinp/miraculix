# Testing

## Host tests

```sh
cargo test            # unit tests + doctests for the host architecture
cargo test --release
```

Token tests skip themselves when the host lacks the feature, so a pass on a
machine without AVX-512 says nothing about the AVX-512 paths. Run those
under an emulator or on matching hardware.

## Emulated architectures

| Target | How | Script |
|---|---|---|
| x86 extensions the host lacks | Intel SDE (`sde64 -spr`, `-gnr`, `-snb`, ...) | - |
| AArch64 (Neon, SVE at VL 128/256/512) | QEMU-user (`max,sveNNN=on`, `neoverse-v1`, `a64fx`) | - |
| Big-endian AArch64 | `qemu-aarch64_be` | `scripts/be-probe` (`cargo +nightly run`) |
| RISC-V RVV, VLEN 128 to 1024 | `qemu-riscv64` | `scripts/rv-probe` |
| wasm32 | node (V8, wasip1), wasmtime (wasip1/2/3) | - |

## Oracles

Format conversions are checked bit for bit against the reference
implementations:

| Oracle | Checks |
|---|---|
| `scripts/tt-oracle` | Tenstorrent `BFP_B` against tt-metal |
| `scripts/gguf-oracle` | GGUF `Q8_0` / `Q6_K` against ggml |

## Cross-checking builds

```sh
cargo check --tests --target aarch64-unknown-linux-gnu
cargo check --tests --target riscv64gc-unknown-linux-gnu
cargo check --tests --target i686-unknown-linux-gnu
cargo check --target wasm32-unknown-unknown
```

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request:

| Job | What |
|---|---|
| `test` | `cargo test` (debug, release, `--no-default-features`) on real x86_64 Linux, aarch64 Linux, Apple Silicon macOS and x86_64 Windows |
| `test-target-cpu` | the tests again with `-C target-cpu=x86-64-v2` (compile flags are a floor) |
| `lint` | `cargo fmt --check`, `cargo clippy`, `cargo doc` |
| `cross-check` | `cargo check` for every supported target, including Apple iOS, Windows on Arm, i586, RISC-V, LoongArch, PowerPC, wasm and bare-metal Cortex-M |
| `msrv` | `cargo check --tests` on the declared `rust-version` |

GitHub's runners have no AVX-512, so AVX-512 paths still need SDE or real
hardware.
