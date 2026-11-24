# Slice-op rules and cross-architecture differences

The rules every `auto_up` / `reduce` function follows, and the few places
where the same name means different things on different architectures.

## Rules every slice op shares

- **Lengths.** All slices must have the same length (or the ratio the op
  documents). A mismatch panics.
- **Integer overflow.** `add` / `sub` / `mul` wrap. Saturating forms have
  their own names (below).
- **Compares** write lane masks (all ones or all zeros per element), not
  `bool`. Feed them to `select_*`.
- **Half floats.** `f16` and `bf16` values are raw `u16` bit patterns.
- **NaN.** Ordered compares (`cmplt`, `cmpgt`, ...) are false if either side
  is NaN; `cmpeq` never matches NaN. SIMD `min`/`max` follow the hardware
  rule (below). The scalar tail uses Rust's `f32::min`/`max`, so a short tail
  can disagree with the vector lanes on NaN inputs.
- **Reductions** (`reduce_*`, `argmax`, `topk`, `compress_gt`) ignore NaN.
  `reduce_sum_f32` propagates it. Ties pick the lowest index. Sums use
  several vector accumulators, so the last bits can differ between CPUs.

## Same name, different meaning

Check this table before sharing code between x86 and Arm:

| Op | x86 | AArch64 / AArch32 |
|---|---|---|
| `andnot_*(a, b)` | `!a & b` (`pandn`) | `a & !b` (`bic`) |
| `select_f32/f64(a, b, mask, out)` | **sign bit** of the mask picks `b` | **per bit** (`bsl`) |
| `shl_*` | one `const IMM` count for all lanes | per-lane **signed** count from a slice; negative shifts right |
| Shift by an immediate | `shl_*::<IMM>`, `shr_*`, `sra_*` | `slli_*`, `srli_*`, `srai_*` |
| Saturating add / sub | `adds_*` / `subs_*` | `qadd_*` / `qsub_*` |
| Compare mask element type | same as the input (`cmpeq_i8 -> i8`) | always unsigned (`cmpeq_i8 -> u8`) |
| Float `min` / `max` on NaN | returns the **second** operand (`minps`) | propagates NaN (`fmin`) |
| Horizontal sums | `reduce::*` | also `sum_*`, `maxv_*`, `minv_*` in `auto_up` |
| FMA tail | scalar tail is **not** fused | Neon tail not fused; SVE has no tail, every lane fused |

WebAssembly follows the Arm naming (`qadd`, `sum_*`). RISC-V `auto` follows
the x86 `reduce` names.

## Writing portable code anyway

- Stick to ops whose meaning is identical: `add`, `sub`, `mul`, `and`, `or`,
  `xor`, `min`/`max` on integers, compares, `fmadd`, and the `reduce` family.
- For `andnot` and float `select`, write the expression out with `and` /
  `xor` / integer `select` instead.
- Keep NaN out of `min`/`max` inputs, or post-process.
