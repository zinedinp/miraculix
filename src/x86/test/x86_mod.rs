use super::*;

/// MMX/SSE/SSE2: x86_64 ABI baseline, always present.
#[test]
#[cfg(target_arch = "x86_64")]
fn baseline_is_always_available_on_x86_64() {
    let set = FeatureSet::detect();
    assert!(set.contains(Feature::Mmx));
    assert!(set.contains(Feature::Sse));
    assert!(set.contains(Feature::Sse2));
}

/// `Feature::bit` keys the bitset; every variant needs a distinct index.
#[test]
fn every_feature_has_a_unique_bit() {
    let mut seen = std::collections::HashSet::new();
    for &feature in Feature::ALL {
        assert!(seen.insert(feature.bit()), "duplicate bit for {feature:?}");
    }
}

/// `detect` matches manual required-features on the same set.
#[test]
fn generic_level_matches_manual_check() {
    let set = FeatureSet::detect();
    let detected = GenericLevel::detect(set);

    for &level in GenericLevel::ALL {
        let should_hold = set.contains_all(level.required_features());
        assert_eq!(
            level <= detected,
            should_hold || level == GenericLevel::V1,
            "level {level:?} required-features check disagrees with detect() result"
        );
    }
}

/// V4 contains V3 contains V2 (copy-paste guard).
#[test]
fn generic_levels_are_cumulative() {
    let v2 = GenericLevel::V2.required_features();
    let v3 = GenericLevel::V3.required_features();
    let v4 = GenericLevel::V4.required_features();

    assert!(
        v2.iter().all(|f| v3.contains(f)),
        "V3 must be a superset of V2"
    );
    assert!(
        v3.iter().all(|f| v4.contains(f)),
        "V4 must be a superset of V3"
    );
}

/// No target-cpu in plain `cargo test`: verify no-op.
#[test]
fn verify_or_panic_is_a_no_op_without_a_compile_time_target_cpu() {
    shortpath::verify_or_panic();
}

/// Cached path matches a fresh probe (shortpath or CPUID).
#[test]
fn detect_level_matches_fresh() {
    assert_eq!(detect_level(), detect_level_fresh());
}

/// Repeated calls are stable (process cache / Assumed const).
#[test]
fn detect_level_is_stable_across_calls() {
    let a = detect_level();
    let b = detect_level();
    let c = detect_level();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

/// warm_up is optional and must not change the resolved level.
#[test]
fn warm_up_then_detect_agrees_with_fresh() {
    warm_up();
    assert_eq!(detect_level(), detect_level_fresh());
}

/// Cached path matches a fresh probe, same contract as `detect_level_matches_fresh`.
#[test]
fn detect_features_matches_fresh() {
    assert_eq!(detect_features(), detect_features_fresh());
}

/// `detect_features()`'s cached/union path must agree bit-for-bit with a truly
/// raw, unassisted `FeatureSet::detect()` on this host: the compile-time lower
/// bound `compile_time_features()` unions in must never disagree with reality.
#[test]
fn detect_features_matches_raw_cpuid() {
    assert_eq!(detect_features(), FeatureSet::detect());
}

/// Repeated calls are stable (process cache).
#[test]
fn detect_features_is_stable_across_calls() {
    let a = detect_features();
    let b = detect_features();
    assert_eq!(a, b);
}

/// `x86::auto` is only a re-export of `auto_up` (same functions).
#[test]
fn auto_alias_matches_auto_up() {
    use crate::x86::{auto, auto_up};
    let a = [1i32, 2, 3, 4];
    let b = [10, 20, 30, 40];
    let mut via_up = [0i32; 4];
    let mut via_alias = [0i32; 4];
    auto_up::add_i32(&a, &b, &mut via_up);
    auto::add_i32(&a, &b, &mut via_alias);
    assert_eq!(via_up, via_alias);
    assert_eq!(via_up, [11, 22, 33, 44]);
}

/// 32-bit x86: the stand-ins for the x86-64-only intrinsics in `arch` give
/// what those do (SSE4a, the one user of the two `cvtsi*64` without other
/// coverage, exists on AMD only).
#[test]
#[cfg(target_arch = "x86")]
fn x86_64_only_intrinsics_have_32bit_stand_ins() {
    use crate::x86::arch::{
        _mm_crc32_u8, _mm_crc32_u64, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_srli_si128,
        _popcnt64,
    };
    let set = FeatureSet::detect();
    let values = [
        0i64,
        1,
        -1,
        i64::MIN,
        i64::MAX,
        0x0123_4567_89ab_cdef,
        -0x7654_3210_fedc_ba98,
    ];
    for v in values {
        // SAFETY: SSE2 is the x86 baseline numerobis targets (i686).
        unsafe {
            let x = _mm_cvtsi64_si128(v);
            assert_eq!(_mm_cvtsi128_si64(x), v);
            assert_eq!(_mm_cvtsi128_si64(_mm_srli_si128::<8>(x)), 0, "high lane zero");
        }
        if set.contains(Feature::Popcnt) {
            // SAFETY: POPCNT detected.
            let n = unsafe { _popcnt64(v) };
            assert_eq!(n.cast_unsigned(), v.count_ones());
        }
        if set.contains(Feature::Sse42) {
            let mut want = 0x1234_5678u32;
            for b in v.to_le_bytes() {
                // SAFETY: SSE4.2 detected.
                want = unsafe { _mm_crc32_u8(want, b) };
            }
            // SAFETY: SSE4.2 detected.
            let got = unsafe { _mm_crc32_u64(0xffff_ffff_1234_5678, v.cast_unsigned()) };
            assert_eq!(got, u64::from(want), "low 32 bits of crc used, result zero-extended");
        }
    }
}
