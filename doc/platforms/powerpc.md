# PowerPC

**Module:** `miraculix::powerpc` · **Toolchain:** stable (`asm!` is stable on
PowerPC since Rust 1.95) · **Status:** kernels for consoles, G4/G5 AltiVec
and POWER7-11 (VSX, ISA 3.0, MMA).

One module covers four very different worlds. Pick your island:

| You target | Read | Kernel |
|---|---|---|
| GameCube, Wii, Wii U | [Consoles: paired singles](#gamecube-wii-wii-u-paired-singles) | `ops::paired::PairedSingle` |
| PS3 | [PS3: Cell PPE](#ps3-cell-ppe) | `ops::vmx::Vmx` |
| Xbox 360 | [Xbox 360: VMX128](#xbox-360-vmx128) | `ops::vmx128::Vmx128` + `Vmx` |
| Power Mac G4 / G5 | [Desktop AltiVec](#g4--g5-altivec) | `ops::vmx::Vmx` |
| OpenPOWER, IBM POWER7-11 | [OpenPOWER and POWER](#openpower-and-ibm-power) | `ops::vsx::Vsx`, `ops::isa300::Isa300`, `ops::isa31::Isa31`, `ops::mma::Mma` |

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
| `class::threads_per_core()` | SMT width from sysfs (Linux) |
| `class::cache_geometry(level)` | L1d / L2 / L3 size, line, ways from auxv (Linux) |

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
| `Vsx` / `Isa300` | 64 x 128-bit | 4 vectors (element-wise), 8-16 (reductions) | 8 |
| `Mma` | 8 x 512-bit accumulators on `vs0`-`vs31` | 2 + 4 operand vectors per k | 8 tiles of 4 x 4 |

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

The Cell-only `lvlx` / `lvrx` loads are deliberately **not** used. They would
replace `lvx` + `lvx` + `vperm` by `lvlx` + `lvrx` + `vor`, which is only a
win if the permute pipe is the bottleneck, and nothing here can tell: QEMU
has no Cell model (and no `lvlx`), and RPCS3 is not cycle-accurate. Without a
measurement on a real PS3 the generic path stays, which is correct on every
AltiVec core.

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

POWER7 and later, little- or big-endian, plus any OpenPOWER core with VSX:

| Token | ISA | Ops |
|---|---|---|
| `ops::vsx::Vsx` | 2.06 (POWER7+) | the full `Vmx` set: element-wise, sum/max/min/argmax/dot, `bf16_to_f32`, `affine_u16_f32` |
| `ops::isa300::Isa300` | 3.0 (POWER9+) | element-wise, sum and dot with `lxvl` / `stxvl` tails (no scalar remainder, the RVV `vsetvl` idea); `cvt_f16_to_f32_slice` / `cvt_f32_to_f16_slice` (`xvcvhpsp` / `xvcvsphp`, also used by `formats::MiniFloat::F16`) |
| `ops::isa31::Isa31` | 3.1 (POWER10/11) | sum and dot with `lxvp` paired loads; `compress_gt_f32` and the `argmax_f32` search with `vextractwm` lane masks; `cvt_f32_to_bf16_slice` (`xvcvspbf16`, also used by `formats::MiniFloat::Bf16` with `Overflow::Special`) |
| `ops::mma::Mma` | 3.1 + MMA (POWER10/11) | `gemm_f32_kernel` on the eight 512-bit accumulators (two 8 x 16 halves of the 8 x 32 block) |

- `lxvw4x` / `stxvw4x` load and store in the same element order on both
  endians and take any address, so the VSX kernels have no alignment head
  and no endian case (the `u16` widening is the one exception, handled per
  endianness). VSX float ops are IEEE (no non-Java flush), so `mul_f32` and
  `affine_u16_f32` match the portable code bit for bit.
- 64 vector registers: the reductions keep 8 accumulators over 32 floats per
  trip. MMA puts its accumulators on `vs0`-`vs31`, operands live in `vs32`+.

**SMT.** POWER8-11 run SMT4 or SMT8, and the vector/MMA units are shared per
SMT4 half (POWER9/10). For vector-bound loops, fewer threads per core than
the OS shows CPUs is often faster: `class::threads_per_core()` gives the SMT
width to size a thread pool, `class::cache_geometry()` the cache sizes for
blocking. Measure before settling on a number. Cache lines are 128 bytes;
pad per-thread data to that.

Open cores such as Microwatt implement the scalar SFFS subset without
VMX/VSX, so they stay on `Scalar`.

## Nightly extras

`ops::altivec::Altivec` (`nightly-altivec`, powerpc64) wraps the
`core::arch::powerpc64` intrinsics (`add_f32x4`, `sub_f32x4`). It predates the
`asm!` kernels; prefer `Vmx`.
