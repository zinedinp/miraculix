# LoongArch and PowerPC

Detection only for now. PowerPC has a few nightly ops.

| | LoongArch | PowerPC |
|---|---|---|
| Module | `miraculix::loongarch` | `miraculix::powerpc` |
| Level | `LoongArchLevel`: `Scalar`, `Lsx`, `Lasx` | `PowerLevel`: `Scalar`, `Altivec`, `Vsx`, `Power8`, `Power9` |
| Probe | Linux/Android auxv, compile-time elsewhere | Linux/Android `AT_HWCAP` + `AT_HWCAP2`, FreeBSD `elf_aux_info`, compile-time elsewhere |
| Ops | none yet | `ops::altivec::Altivec` (`add_f32x4`, `sub_f32x4`) with `nightly-altivec`, powerpc64 only |
| API | `detect_level`, `detect_features` (both cached), `*_fresh`, `warm_up`, `shortpath::verify_or_panic` | same; `Altivec::detect()` uses the cache |

Both arches union the probe with the compile-time features, as everywhere
else ([caching](../detection-and-caching.md)).
