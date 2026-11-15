//! Runs only on AVX10.2 hardware; elsewhere `detect()` is `None` and each test returns.
use super::*;
use crate::formats::{MiniFloat, Overflow};

fn f16_inputs() -> Vec<u16> {
    // every binary16 pattern: encodes must equal the scalar codec on the exact value
    (0..=u16::MAX).collect()
}

#[test]
fn decode_matches_scalar_codec() {
    let Some(t) = Avx10Fp8::detect() else { return };
    let codes: Vec<u8> = (0..=255).collect();
    for (hw, f) in [(Fp8Format::E4m3, MiniFloat::E4M3Fn), (Fp8Format::E5m2, MiniFloat::E5M2)] {
        let mut h = vec![0u16; 256];
        t.fp8_to_f16_slice(hw, &codes, &mut h);
        let mut s = vec![0f32; 256];
        t.fp8_to_f32_slice(hw, &codes, &mut s);
        for i in 0..256 {
            let w = f.decode(i as u32);
            let h32 = MiniFloat::F16.decode(h[i] as u32);
            assert!(h32.to_bits() == w.to_bits() || (h32.is_nan() && w.is_nan()), "{f:?} f16 {i:#x}");
            assert!(s[i].to_bits() == w.to_bits() || (s[i].is_nan() && w.is_nan()), "{f:?} f32 {i:#x}");
        }
    }
}

#[test]
fn encode_matches_scalar_codec() {
    let Some(t) = Avx10Fp8::detect() else { return };
    let src = f16_inputs();
    for (hw, f) in [(Fp8Format::E4m3, MiniFloat::E4M3Fn), (Fp8Format::E5m2, MiniFloat::E5M2)] {
        for (sat, ov) in [(true, Overflow::Saturate), (false, Overflow::Special)] {
            let mut got = vec![0u8; src.len()];
            t.f16_to_fp8_slice(hw, sat, &src, &mut got);
            for (i, &b) in src.iter().enumerate() {
                let x = MiniFloat::F16.decode(b as u32);
                if x.is_nan() {
                    assert!(f.decode(got[i] as u32).is_nan(), "{f:?} nan {b:#x}");
                } else {
                    assert_eq!(got[i] as u32, f.encode(x, ov), "{f:?} sat={sat} f16={b:#x}");
                }
            }
        }
    }
}
