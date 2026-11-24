# LoongArch and PowerPC

Detection only for now. PowerPC has a few nightly ops.

| | LoongArch | PowerPC |
|---|---|---|
| Module | `miraculix::loongarch` | `miraculix::powerpc` |
| Level | `LoongArchLevel`: `Scalar`, `Lsx`, `Lasx` | `PowerLevel`: `Scalar`, `Altivec`, `Vsx`, `Power8`, `Power9` |
| Probe | Linux/Android auxv, compile-time elsewhere | Linux/Android `AT_HWCAP` + `AT_HWCAP2`, FreeBSD `elf_aux_info`, compile-time elsewhere |
| Ops | none yet | `ops::altivec::Altivec` (`add_f32x4`, `sub_f32x4`) with `nightly-altivec`, powerpc64 |
| API | `detect_level`, `detect_level_fresh`, `warm_up`, `shortpath::verify_or_panic`, `features::FeatureSet::detect()` (uncached) | same |

There is no cached `detect_features()` on these two yet; call
`detect_level()` (cached) or take one `FeatureSet::detect()` at startup and
keep it.
