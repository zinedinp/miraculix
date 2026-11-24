# LoongArch

**Module:** `miraculix::loongarch` · **Status:** detection only.

| | |
|---|---|
| Level | `LoongArchLevel`: `Scalar`, `Lsx`, `Lasx` |
| Probe | Linux/Android auxv, compile-time elsewhere |
| Ops | none yet |
| API | `detect_level`, `detect_features` (both cached), `*_fresh`, `warm_up`, `shortpath::verify_or_panic` |

The probe is unioned with the compile-time features, as everywhere else
([caching](../detection-and-caching.md)).

PowerPC has its own page now: [PowerPC](powerpc.md).
