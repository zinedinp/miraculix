# Op finder: intrinsic name to miraculix name

miraculix names ops `<op>_<type>x<lanes>` (`movemask_i8x16`,
`select_i8x64`), not after the Intel / Arm intrinsics. Grepping for
`_mm512_test_epi8_mask` finds nothing, or only a private helper. Use these
tables, or search the rustdoc (`cargo doc --open`) for the operation.

## x86

| You want | miraculix | File |
|---|---|---|
| `vpmovb2m` (byte sign bits to `u64`) | `Avx512Bw::movepi8_mask`, also `movepi16/32/64_mask`; `Vl` tokens for 128/256 | `x86/ops/avx512/kmask/bridge.rs` |
| `vpmovm2b` (mask to vector) | `Avx512Bw::movm_epi8` and siblings | `kmask/bridge.rs` |
| `kand`, `kshift`, `ktest` | `kmask/logic.rs`, `shift.rs`, `ktest.rs` | `x86/ops/avx512/kmask/` |
| 64-lane blend | `Avx512Bw::select_i8x64(a, b, mask)` (vector mask), `mask_blend_u8x64(k, a, b)` (`u64` mask) | `avx512bw.rs` |
| Byte bit test to `u64` | `Avx512Bw::test_u8x64_mask(a, b)` | `avx512bw.rs` |
| `pmovmskb` | `Sse2::movemask_i8x16`, `Avx2::movemask_i8x32` | `sse2.rs`, `avx2.rs` |
| `pshufb` | `Ssse3::shuffle_i8x16`, `Avx2::shuffle_i8x32`, `Avx512Bw::shuffle_i8x64` | |
| `vpermb` / `vpermi2b` | `Avx512Vbmi::permutexvar_u8x64` / `permutex2var_u8x64` | `avx512vbmi.rs` |
| `cvtps2pd` / `cvtpd2ps` | `Sse2::cvt_f32x4_to_f64x2`, `Avx::cvt_f32x4_to_f64x4`, `Avx512f::cvt_f32x8_to_f64x8` (+ reverse) | |
| `punpckl*` / `punpckh*` | `unpack{lo,hi}_u8x16` (Sse2), `_u8x32/_i16x16/_i32x8/_i64x4` (Avx2), `_u8x64/_i16x32` (Avx512Bw), `_i32x16/_i64x8` (Avx512f) | |
| `pmovzxbw` | `Sse41::cvtepu8_u16x8`, `Avx2::cvtepu8_u16x16`, `Avx512Bw::cvtepu8_u16x32` | |
| `psllw` / `psrlw` (runtime count) | `sll_u16x8` / `srl_u16x8` (Sse2), `x16` (Avx2), `x32` (Avx512Bw) | |
| `movq`, `movd` loads | `Sse2::loadl_u8x8`, `cvtsi32_u32x4`, `cvtsi64_u64x2` | |
| `vperm2i128`, `vbroadcasti128`, `vpermt2q` | `Avx2::permute2x128_i64x4`, `Avx2::broadcast_u8x16_to_x32`, `Avx512f::broadcast_u8x16_to_x64`, `Avx512f::permutex2var_u64x8` | |
| Uncached CPUID probe | `x86::detect_features_fresh()` + `Token::from_features` | `x86/mod.rs` |

## Neon (AArch64)

The `Neon` token's extras (`aarch64/ops/neon/extra.rs`) use the x86 names
where the operation is the same, so a byte kernel can be written once:

| Need | Name |
|---|---|
| f32 / f64 converts | `cvt_f32x4_to_f64x2`, `cvt_f64x2_to_f32x4`, `cvt_f32x4_to_f64x4`, `cvt_f64x4_to_f32x4`, `cvt_f32x2_to_f64x2`, `cvt_high_f32x4_to_f64x2`, `cvt_f64x2_to_f32x2` |
| Byte `movemask` | `movemask_i8x16` (sign-bit semantics, like `Sse2`) |
| Byte interleave | `zip1_u8x16` / `zip2_u8x16` (also `unpacklo_u8x16` / `unpackhi_u8x16`), `zip1_u8x8` / `zip2_u8x8` |
| Bit test per byte (`vtstq_u8`) | `test_u8x16` |
| Table lookup (`vqtbl*`) | `tbl_u8x16`, `tbx_u8x16`, `tbl4_u8x64`, `tbx4_u8x64`, `lookup_u8x256`. Out of range gives 0 / keeps the fallback; not `pshufb` semantics |
| Horizontal add | `vaddvq_u8` and friends (128-bit forms) |
| 8-byte ops | `srli_u8x8::<N>`, `sum_u8x8` |
