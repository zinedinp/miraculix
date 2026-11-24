# RISC-V

**Module:** `miraculix::riscv` · **Toolchain:** stable · **Status:** RVV
element-wise core and the full reduction family.

**Probe:** auxv `AT_HWCAP` for the base letters (Linux, Android, FreeBSD),
plus `riscv_hwprobe(2)` on Linux and Android for `Zvfh` / `Zvfbfmin` /
`Zvfbfwma`. Always unioned with the compile-time features, so a `+v` build
always takes the RVV path.

| Item | Role |
|---|---|
| `detect_level() -> RiscvLevel` | `Gc` or `Vector` |
| `detect_features()` | feature bitset, cached |
| `warm_up()`, `detect_*_fresh()`, `shortpath::verify_or_panic()` | as on x86 ([caching](../detection-and-caching.md)) |
| `ops::rvv::Rvv` | f32/i32 element-wise, FMA, reductions, compare/compress, `exp_f32`, `dot_i8_i32`, BFP_B row encode/decode. `asm!` loops that work for **any VLEN** |
| `ops::zvfh::Zvfh`, `ops::zvfbf::{Zvfbfmin, Zvfbfwma}` | half / bfloat16 vectors |
| `riscv::auto` | `add/sub/mul/fmadd_f32`, `add_i32` plus every reduction (same names as `x86::reduce`) |
| `class::RiscvClass::detect()` | `VectorAccelerator` for a SiFive core with V and VLEN >= 512 (the X280 in Tenstorrent Blackhole) |
| `class::core_id()`, `class::vlen_bits()` | raw vendor / arch / impl IDs and VLEN |

Note the module is `riscv::auto`, not `auto_up` as on other arches.

Reductions add in an order that depends on VLEN, so sums can differ in the
last bits between CPUs.

Tested under `qemu-riscv64` at VLEN 128 to 1024 with `scripts/rv-probe`.
