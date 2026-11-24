# PowerPC

**Module:** `miraculix::powerpc` · **Toolchain:** stable (`asm!` is stable on
PowerPC since Rust 1.95) · **Status:** consoles and big-endian AltiVec first;
VSX / POWER9 / POWER10 kernels come next.

One module covers four very different worlds. Pick your island:

| You target | Read | Kernel |
|---|---|---|
| GameCube, Wii, Wii U | [Consoles: paired singles](#gamecube-wii-wii-u-paired-singles) | `ops::paired::PairedSingle` |
| PS3 | [PS3: Cell PPE](#ps3-cell-ppe) | `ops::vmx::Vmx` |
| Xbox 360 | [Xbox 360: VMX128](#xbox-360-vmx128) | `ops::vmx128::Vmx128` + `Vmx` |
| Power Mac G4 / G5 | [Desktop AltiVec](#g4--g5-altivec) | `ops::vmx::Vmx` |
| OpenPOWER, IBM POWER7-11 | [OpenPOWER and POWER](#openpower-and-ibm-power) | portable for now |

Not sure? `powerpc::auto::*` picks the best kernel at run time on every one
of them, and `class::PowerClass::detect()` tells you where you are.

## Detection

| Platform | Probe |
|---|---|
| Linux, Android | `getauxval(AT_HWCAP / AT_HWCAP2)` |
| FreeBSD | `elf_aux_info(AT_HWCAP / AT_HWCAP2)` |
| Bare metal (`target_os = "none"`: libogc, wut, libxenon) | PVR, MSR and HID2, read in supervisor mode |
| PS3 GameOS (`target_os = "ps3"`) | assumed AltiVec + Cell |
| Anything else | compile-time `target_feature` |

Every PowerPC `target_feature` is still unstable, so on stable the
compile-time part is empty except for the PS3 target; the run-time probe
decides. Both are cached once per process
([caching](../detection-and-caching.md)).

| Item | Role |
|---|---|
| `detect_level() -> PowerLevel` | `Scalar`, `Altivec`, `Vsx`, `Isa207`, `Isa300`, `Isa31` |
| `detect_features()` | feature bitset, cached |
| `class::PowerClass::detect()` | `Console(GameCube / Wii / WiiU / Xbox360 / Ps3)`, `Desktop`, `Server`, `OpenCore` |
| `class::PowerCore::detect()` | core family from the PVR (Gekko, 750CL, Espresso, G4, G5, Cell, Xenon, POWER7-11, Microwatt) |
| `class::pvr()` | raw PVR: Linux (kernel-emulated, costs a trap) and bare metal only |
| `class::cache_line_bytes()` | 32 (750, G4) or 128 (G5, Cell, Xenon, POWER); `AT_DCACHEBSIZE` on Linux |

The levels are named after the open Power ISA versions (OpenPOWER
Foundation), not IBM chips: `Isa207` = POWER8, `Isa300` = POWER9, `Isa31` =
POWER10/11. `PowerLevel::Power8` / `Power9` still compile as deprecated
aliases. Paired singles, Cell and VMX128 are features, not levels: a Gekko is
`Scalar` with `Feature::PairedSingle`, a Xenon is `Altivec` with
`Feature::Vmx128`.

### Extended register state: bare metal only

`Feature::PairedSingle` (the second FPR half, HID2, the GQRs) and
`Feature::Vmx128` (`v32`-`v127`) are register state no mainline kernel saves
on a context switch. They are therefore reported **only on bare metal**, never
under Linux, even on a Wii or Xbox 360 whose PVR says the core has them. On
bare metal they also need the unit switched on: `MSR[FP]` and `HID2[PSE|LSQE]`
for paired singles, `MSR[VEC]` for VMX and VMX128.

## Registers: more of them, used

PowerPC has 32 GPRs, 32 FPRs and 32 vector registers (64 with VSX, 128 on
Xenon), against 16 on SSE/AVX2. The kernels spend them on independent work, so
in-order cores (Cell, Xenon, 750) never wait on one FP result:

| Kernel | Registers | In flight per trip | Reduction accumulators |
|---|---|---|---|
| `Vmx` | 32 x 128-bit | 4 vectors (element-wise), 8 (reductions) | 8 |
| `PairedSingle` | 32 FPR x 2 lanes | 4-8 pairs | 8 |
| `Vmx128` | 128 x 128-bit | 8 vectors per input | 8 (in `v32`-`v39`) |

Each module exports its numbers as `UNROLL` / `ACCUMULATORS`.

## GameCube, Wii, Wii U: paired singles

`ops::paired::PairedSingle` (32-bit `powerpc`). Every FPR holds two `f32`,
and the quantized load unit converts `u8`/`s8`/`u16`/`s16` to `f32` for free.

- Ops: `add/sub/mul/fmadd_f32`, `reduce_sum_f32`, `dot_f32`,
  `affine_u16_f32` (hardware `u16` dequantization through `psq_l`).
- No assembler knows these instructions, so they are `.long` words from
  `ops::encode`, checked on the host against binutils' expected encodings.
- Each kernel is a single `asm!` block (LLVM would drop the second half on a
  spill) and saves and restores GQR7; GQR0-6 are untouched.
- No Rust target exists upstream: use a custom target JSON (as `ogc-rs` and
  `rust-wiiu` do) with `"os": "none"`.
- **Not run in CI**: no QEMU model has paired singles. Run
  `scripts/ps-probe` in Dolphin (see [testing](../testing.md)).

## PS3: Cell PPE

The PPE is a 64-bit big-endian in-order core with AltiVec and no VSX. Both the
Tier-3 target `powerpc64-sony-ps3` (GameOS, needs `-Zbuild-std`) and Linux on
PS3 (`AT_HWCAP` reports `CELL`) use `ops::vmx::Vmx`. LLVM has no `cell` CPU;
`-C target-cpu=970` is the closest model.

## Xbox 360: VMX128

Xenon has three PPE-class cores with VMX128: 128 vector registers per thread
and a 4-lane dot product (`vmsum4fp128`). `ops::vmx128::Vmx128` keeps eight
accumulators and eight vectors per input in the high registers for
`dot_f32` and `reduce_sum_f32`; element-wise ops are load/store bound and use
`Vmx`.

The VMX128 words are built by `ops::encode` and checked against a decoder
transcribed from Xenia. **They are not executed anywhere**: there is no Rust
target, no assembler and no automatable emulator. Treat them as untested
until run on a console or in Xenia.

## G4 / G5: AltiVec

`ops::vmx::Vmx` (big-endian, 32- and 64-bit):

| Ops | Notes |
|---|---|
| `add/sub/mul/fmadd_f32`, `add_i32` | `mul` is `vmaddfp` with `-0.0` (VMX has no plain multiply); `fmadd` is fused |
| `reduce_sum/max/min_f32`, `dot_f32`, `argmax_f32` | NaN ignored by max/min/argmax |
| `bf16_to_f32`, `affine_u16_f32` | bit-exact against the portable code |

Unaligned loads use `lvx` + `lvx` + `vperm`, so nothing outside the slice's
own 16-byte blocks is read; stores are aligned `stvx` after a short scalar
head. Linux and Mac OS run VMX in non-Java mode (`VSCR[NJ] = 1`): denormal
inputs and results become zero.

## OpenPOWER and IBM POWER

Detection is complete (ISA 2.06 to 3.1, MMA as its own bit, POWER11 =
`Isa31`), but little-endian ppc64le runs the portable `auto` code until the
VSX kernels land. Planned, in this order:

1. VSX (ISA 2.06/2.07), little- and big-endian.
2. ISA 3.0: `lxvl` / `stxvl` length-controlled tails (the RVV `vsetvl` idea,
   no scalar remainder), `xvcvhpsp` for f16.
3. ISA 3.1: `lxvp` pairs, `xvcvbf16spn`, `vextractbm`.
4. MMA: a `gemm_f32_kernel` on the eight 512-bit accumulators.
5. SMT topology (threads per core, cache geometry) for POWER's SMT4/SMT8.

Open cores such as Microwatt implement the scalar SFFS subset without VMX/VSX,
so they stay on `Scalar`.

## Nightly extras

`ops::altivec::Altivec` (`nightly-altivec`, powerpc64) wraps the
`core::arch::powerpc64` intrinsics (`add_f32x4`, `sub_f32x4`). It predates the
`asm!` kernels; prefer `Vmx`.
