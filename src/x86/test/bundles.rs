use crate::x86::bundles::{V3, V4, V4Vbmi};
use crate::x86::ops::avx512::avx512vbmi::Avx512Vbmi;
use crate::x86::{GenericLevel, detect_features, detect_level};

#[test]
fn bundles_match_level_and_token_probes() {
    let set = detect_features();
    assert_eq!(V3::detect().is_some(), detect_level() >= GenericLevel::V3);
    assert_eq!(V4::detect().is_some(), detect_level() >= GenericLevel::V4);
    assert_eq!(
        V4Vbmi::detect().is_some(),
        V4::detect().is_some() && Avx512Vbmi::from_features(set).is_some()
    );
    // A higher bundle implies the lower ones.
    if V4::detect().is_some() {
        assert!(V3::detect().is_some());
    }
}

#[test]
fn bundles_reject_an_empty_set() {
    let empty = crate::x86::FeatureSet::default();
    assert!(V3::from_features(empty).is_none());
    assert!(V4::from_features(empty).is_none());
    assert!(V4Vbmi::from_features(empty).is_none());
}

crate::avx512_v4_vbmi_fn! {
    fn reverse_bytes(v: V4Vbmi, a: [u8; 64]) -> [u8; 64] {
        let idx: [u8; 64] = core::array::from_fn(|i| 63 - i as u8);
        v.vbmi.permutexvar_u8x64(idx, a)
    }
}

#[test]
fn v4_vbmi_fn_runs_with_bundle() {
    let Some(v) = V4Vbmi::detect() else { return };
    let a: [u8; 64] = core::array::from_fn(|i| i as u8);
    let want: [u8; 64] = core::array::from_fn(|i| 63 - i as u8);
    assert_eq!(reverse_bytes(v, a), want);
}
