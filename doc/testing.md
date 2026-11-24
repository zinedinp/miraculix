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
| PowerPC G4 / BE ppc64 AltiVec (`Vmx`) | `qemu-ppc -cpu 7400`, `qemu-ppc64 -cpu power8` | CI job `test-powerpc-qemu` |
| ppc64le POWER8-10 | `qemu-ppc64le -cpu power8/9/10` | CI job `test-powerpc-qemu` |
| GameCube / Wii paired singles | Dolphin (no QEMU model has them) | `scripts/ps-probe` (nightly, run by hand) |
| Xbox 360 VMX128 | none automatable | encodings only (host tests in `powerpc/ops/encode.rs`) |

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
cargo build --lib --target powerpc64-unknown-linux-gnu   # assembles the asm! kernels
```

`cargo check` never runs LLVM, so it cannot catch a bad mnemonic inside
`asm!`; for PowerPC (and anything else written in `asm!`) build the library.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request:

| Job | What |
|---|---|
| `test` | `cargo test` (debug, release, `--no-default-features`) on real x86_64 Linux, aarch64 Linux, Apple Silicon macOS and x86_64 Windows |
| `test-target-cpu` | the tests again with `-C target-cpu=x86-64-v2` (compile flags are a floor) |
| `lint` | `cargo fmt --check`, `cargo clippy`, `cargo doc` |
| `cross-check` | `cargo check` for every supported target, including Apple iOS, Windows on Arm, i586, RISC-V, LoongArch, PowerPC (32-bit, BE and LE, plus a `cargo build --lib` to assemble the kernels), wasm and bare-metal Cortex-M |
| `test-powerpc-qemu` | `cargo test` under qemu-user: G4 (`7400`, 32-bit BE, VMX), POWER8 BE (VMX + VSX), ppc64le POWER8/9/10 (VSX, ISA 3.0, MMA); `MIRACULIX_EXPECT_POWER` fails the job if the model's kernels were not detected |
| `msrv` | `cargo check --tests` on the declared `rust-version` |

GitHub's runners have no AVX-512, so AVX-512 paths still need SDE or real
hardware.
