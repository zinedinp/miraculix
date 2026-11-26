# miraculix documentation

Start with the [crate README](../README.md) for the overview and a first
example. These pages go deeper on one topic each.

## Concepts

| Page | Read it when |
|---|---|
| [Feature detection and caching](detection-and-caching.md) | you want to know what a check costs and how to detect once at startup |
| [Safety model](safety-model.md) | you want to know why the safe API cannot crash on old CPUs, and where `unsafe` is still yours |
| [Kernel macros](kernel-macros.md) | you write multi-op SIMD kernels with `avx2_fn!` and friends |
| [Slice-op rules and cross-arch differences](cross-arch-semantics.md) | you share code between x86 and Arm, or care about NaN / overflow |
| [Op finder](op-finder.md) | you know the Intel / Arm intrinsic and need the miraculix name |
| [Architecture-independent modules](modules.md) | you need cache sizes (`cache`), `prefetch`, number `formats`, or `mmio` copies |
| [Testing](testing.md) | you want to run the tests under emulators or oracles |

## Platforms

| Page | Covers |
|---|---|
| [x86 / x86_64](platforms/x86.md) | detect, auto catalog, reductions, token layout |
| [AVX-512](platforms/avx512.md) | which AVX-512 token you need, OS support, 512 vs 256-bit |
| [AArch64](platforms/aarch64.md) | Neon, SVE/SVE2, crypto, Apple and Windows detection |
| [AArch32](platforms/aarch32.md) | ARMv7 Neon (nightly) |
| [RISC-V](platforms/riscv.md) | RVV, Zvfh, Zvfbf*, accelerator detection |
| [WebAssembly](platforms/wasm.md) | simd128 / relaxed-simd, host-enabled SIMD, module validation |
| [PowerPC](platforms/powerpc.md) | GameCube/Wii/Wii U paired singles, PS3 Cell, Xbox 360 VMX128, G4/G5 AltiVec, POWER7-11 VSX / ISA 3.0 / MMA, SMT topology |
| [LoongArch](platforms/loongarch.md) | detection |

Per-extension minimum Rust versions: [RUST_VERSIONS.md](../RUST_VERSIONS.md).
