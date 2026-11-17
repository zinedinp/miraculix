use super::int::*;
use super::mx::*;
use super::*;

const ALL: [MiniFloat; 9] = [
    MiniFloat::E4M3Fn,
    MiniFloat::E4M3Fnuz,
    MiniFloat::E5M2,
    MiniFloat::E5M2Fnuz,
    MiniFloat::E2M1,
    MiniFloat::E2M3,
    MiniFloat::E3M2,
    MiniFloat::F16,
    MiniFloat::Bf16,
];

#[test]
fn known_values() {
    assert_eq!(MiniFloat::E4M3Fn.max_finite(), 448.0);
    assert_eq!(MiniFloat::E4M3Fnuz.max_finite(), 240.0);
    assert_eq!(MiniFloat::E5M2.max_finite(), 57344.0);
    assert_eq!(MiniFloat::E5M2Fnuz.max_finite(), 57344.0);
    assert_eq!(MiniFloat::E2M1.max_finite(), 6.0);
    assert_eq!(MiniFloat::E2M3.max_finite(), 7.5);
    assert_eq!(MiniFloat::E3M2.max_finite(), 28.0);
    assert_eq!(MiniFloat::F16.max_finite(), 65504.0);
    assert_eq!(MiniFloat::Bf16.decode(0x3F80), 1.0);
    assert_eq!(MiniFloat::F16.decode(0x3C00), 1.0);
    assert_eq!(MiniFloat::F16.decode(0xC000), -2.0);
    // E2M1 code table: 0, .5, 1, 1.5, 2, 3, 4, 6
    let e2m1: Vec<f32> = (0..8).map(|c| MiniFloat::E2M1.decode(c)).collect();
    assert_eq!(e2m1, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    assert!(MiniFloat::E4M3Fn.decode(0x7F).is_nan());
    assert!(MiniFloat::E4M3Fnuz.decode(0x80).is_nan());
    assert_eq!(MiniFloat::E5M2.decode(0x7C), f32::INFINITY);
    assert_eq!(MiniFloat::E4M3Fn.decode(1), 2.0f32.powi(-9));
}

#[test]
fn roundtrip_every_code() {
    for f in ALL {
        for c in 0..(1u32 << f.bits()) {
            let x = f.decode(c);
            if x.is_nan() {
                assert!(f.decode(f.encode(x, Overflow::Special)).is_nan(), "{f:?} {c:#x}");
                continue;
            }
            let back = f.encode(x, Overflow::Special);
            // -0 collapses to +0 on FNUZ; everything else is exact.
            assert_eq!(f.decode(back).to_bits(), x.to_bits(), "{f:?} code {c:#x}");
        }
    }
}

#[test]
fn matches_native_f16_f32_bits() {
    // bf16 = top 16 bits of f32 with RNE
    for &x in &[0.1f32, -3.14159, 1e-40, 1e30, 65504.0, 1.0 + 1.0 / 256.0, 1.0 + 3.0 / 256.0] {
        let b = x.to_bits();
        let want = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u32;
        assert_eq!(MiniFloat::Bf16.encode(x, Overflow::Special), want, "{x}");
    }
}

#[test]
fn rounding_overflow_and_nan() {
    let f = MiniFloat::E4M3Fn;
    assert_eq!(f.encode(1000.0, Overflow::Saturate), 0x7E);
    assert_eq!(f.encode(-1000.0, Overflow::Saturate), 0xFE);
    assert_eq!(f.encode(1000.0, Overflow::Special), 0x7F);
    assert_eq!(f.encode(f32::INFINITY, Overflow::Saturate), 0x7E);
    assert_eq!(MiniFloat::E5M2.encode(1e9, Overflow::Special), 0x7C);
    assert_eq!(MiniFloat::E2M1.encode(100.0, Overflow::Special), 7);
    // ties to even: 0.75 is halfway between E2M1 1.0 (code 2) and 0.5 (code 1)
    assert_eq!(MiniFloat::E2M1.encode(0.75, Overflow::Saturate), 2);
    assert_eq!(MiniFloat::E2M1.encode(1.25, Overflow::Saturate), 2);
    assert_eq!(MiniFloat::E2M1.encode(1.75, Overflow::Saturate), 4);
    assert_eq!(MiniFloat::E4M3Fnuz.encode(-1e-9, Overflow::Saturate), 0);
    assert!(f.decode(f.encode(f32::NAN, Overflow::Saturate)).is_nan());
}

#[test]
fn slices_and_lut() {
    let src: Vec<u8> = (0..=255).collect();
    let mut out = vec![0.0; 256];
    MiniFloat::E5M2.decode_u8_slice(&src, &mut out);
    for (i, v) in out.iter().enumerate() {
        let w = MiniFloat::E5M2.decode(i as u32);
        assert!(v.to_bits() == w.to_bits() || (v.is_nan() && w.is_nan()));
    }
    let mut codes = vec![0u8; 256];
    MiniFloat::E4M3Fn.encode_u8_slice(&[0.5; 256], &mut codes, Overflow::Saturate);
    assert!(codes.iter().all(|&c| MiniFloat::E4M3Fn.decode(c as u32) == 0.5));
}

#[test]
fn tf32() {
    assert_eq!(round_to_tf32(1.0), 1.0);
    let x = f32::from_bits(0x3F80_1000); // exactly half an ulp: ties to even (down)
    assert_eq!(round_to_tf32(x).to_bits(), 0x3F80_0000);
    let y = f32::from_bits(0x3F80_3000); // tie, odd neighbour: rounds up
    assert_eq!(round_to_tf32(y).to_bits(), 0x3F80_4000);
    assert_eq!(round_to_tf32(f32::from_bits(0x3F80_1001)).to_bits(), 0x3F80_2000);
    assert!(round_to_tf32(f32::NAN).is_nan());
    assert_eq!(round_to_tf32(f32::INFINITY), f32::INFINITY);
    assert_eq!(round_to_tf32(f32::MAX), f32::INFINITY);
}

#[test]
fn int_pack_roundtrip() {
    let u4: Vec<u8> = (0..33).map(|i| (i * 7 % 16) as u8).collect();
    let mut p = vec![0u8; packed_len::<4>(u4.len())];
    pack_uint::<4>(&u4, &mut p);
    let mut o = vec![0u8; u4.len()];
    unpack_uint::<4>(&p, &mut o);
    assert_eq!(o, u4);

    let i2: Vec<i8> = (0..25).map(|i| (i % 4) as i8 - 2).collect();
    let mut p = vec![0u8; packed_len::<2>(i2.len())];
    pack_int::<2>(&i2, &mut p);
    let mut o = vec![0i8; i2.len()];
    unpack_int::<2>(&p, &mut o);
    assert_eq!(o, i2);

    let i4: Vec<i8> = (0..40).map(|i| (i % 16) as i8 - 8).collect();
    let mut p = vec![0u8; packed_len::<4>(i4.len())];
    pack_int::<4>(&i4, &mut p);
    let mut o = vec![0i8; i4.len()];
    unpack_int::<4>(&p, &mut o);
    assert_eq!(o, i4);

    let b: Vec<u8> = (0..19).map(|i| (i % 2) as u8).collect();
    let mut p = vec![0u8; packed_len::<1>(b.len())];
    pack_uint::<1>(&b, &mut p);
    assert_eq!(p[0], 0b1010_1010);
    let mut o = vec![0u8; b.len()];
    unpack_uint::<1>(&p, &mut o);
    assert_eq!(o, b);
    // saturation
    let mut p = [0u8; 1];
    pack_int::<4>(&[100, -100], &mut p);
    assert_eq!(p[0], 0x87);
}

#[test]
fn int_dots() {
    let a: Vec<i8> = (0..31).map(|i| (i * 5 % 16) as i8 - 8).collect();
    let b: Vec<i8> = (0..31).map(|i| (i * 3 % 16) as i8 - 8).collect();
    let (mut pa, mut pb) = (vec![0u8; 16], vec![0u8; 16]);
    pack_int::<4>(&a, &mut pa);
    pack_int::<4>(&b, &mut pb);
    let want: i32 = a.iter().zip(&b).map(|(x, y)| *x as i32 * *y as i32).sum();
    assert_eq!(dot_i4(&pa, &pb, 31), want);

    let ua: Vec<u8> = a.iter().map(|&v| (v + 8) as u8).collect();
    let ub: Vec<u8> = b.iter().map(|&v| (v + 8) as u8).collect();
    pack_uint::<4>(&ua, &mut pa);
    pack_uint::<4>(&ub, &mut pb);
    let want: u32 = ua.iter().zip(&ub).map(|(x, y)| *x as u32 * *y as u32).sum();
    assert_eq!(dot_u4(&pa, &pb, 31), want);

    assert_eq!(popcount_and(&[0xFF, 0x0F], &[0x0F, 0xFF]), 8);
    assert_eq!(popcount_xor(&[0xFF, 0x0F], &[0x0F, 0xFF]), 8);
    assert_eq!(dot_pm1(&[0b1111], &[0b1111], 4), 4);
    assert_eq!(dot_pm1(&[0b1111], &[0b0000], 4), -4);
    assert_eq!(dot_pm1(&[0xFF], &[0x00], 3), -3);
}

fn data(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173 * (1.0 + (i / 32) as f32)).collect()
}

#[test]
fn mx_roundtrip_within_format_error() {
    let src = data(96);
    for (elem, tol) in [
        (MxElem::Fp8E4M3, 0.07),
        (MxElem::Fp8E5M2, 0.13),
        (MxElem::Fp6E2M3, 0.14),
        (MxElem::Fp6E3M2, 0.27),
        (MxElem::Fp4, 0.35),
        (MxElem::Int8, 0.02),
    ] {
        let mut sc = [0u8; 3];
        let mut d = vec![0u8; 3 * elem.block_bytes()];
        mx_quantize(elem, &src, &mut sc, &mut d);
        let mut out = vec![0.0; 96];
        mx_dequantize(elem, &sc, &d, &mut out);
        for b in 0..3 {
            let amax = src[b * 32..][..32].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for i in b * 32..(b + 1) * 32 {
                assert!((src[i] - out[i]).abs() <= tol * amax, "{elem:?} {} vs {}", src[i], out[i]);
            }
        }
    }
}

#[test]
fn mx_exact_and_scale() {
    // MXFP4 block with exactly representable values and amax = 6 -> scale 2^0.
    let mut src = [0.0f32; 32];
    src[0] = 6.0;
    src[1] = -1.5;
    src[2] = 0.5;
    let (mut sc, mut d) = ([0u8; 1], [0u8; 16]);
    mx_quantize(MxElem::Fp4, &src, &mut sc, &mut d);
    assert_eq!(sc[0], 127);
    let mut out = [0.0; 32];
    mx_dequantize(MxElem::Fp4, &sc, &d, &mut out);
    assert_eq!(out, src);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert!(e8m0_to_f32(255).is_nan());
}

#[test]
fn nvfp4_roundtrip() {
    let src = data(64);
    let ts = nvfp4_tensor_scale(&src);
    let (mut sc, mut d) = ([0u8; 4], [0u8; 32]);
    nvfp4_quantize(&src, ts, &mut sc, &mut d);
    let mut out = vec![0.0; 64];
    nvfp4_dequantize(&sc, ts, &d, &mut out);
    for b in 0..4 {
        let amax = src[b * 16..][..16].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for i in b * 16..(b + 1) * 16 {
            assert!((src[i] - out[i]).abs() <= 0.2 * amax + 1e-6, "{} vs {}", src[i], out[i]);
        }
    }
}

#[test]
fn half_slices_match_scalar_codec() {
    let mut vals: Vec<f32> = (0..2000).map(|i| ((i as f32 - 1000.0) * 0.37).powi(3) * 1e-3).collect();
    vals.extend([0.0, -0.0, 1e-8, 6.0e-8, 65504.0, 65520.0, 1e9, -1e9, f32::INFINITY, f32::MIN_POSITIVE, 1e-40]);
    for f in [MiniFloat::F16, MiniFloat::Bf16] {
        for ov in [Overflow::Special, Overflow::Saturate] {
            let mut got = vec![0u16; vals.len()];
            f.encode_u16_slice(&vals, &mut got, ov);
            // F16 hardware path only exists for Special; scalar otherwise.
            for (i, &x) in vals.iter().enumerate() {
                assert_eq!(got[i] as u32, f.encode(x, ov), "{f:?} {ov:?} {x}");
            }
        }
        let codes: Vec<u16> = (0..=u16::MAX).collect();
        let mut dec = vec![0.0f32; codes.len()];
        f.decode_u16_slice(&codes, &mut dec);
        for (i, d) in dec.iter().enumerate() {
            let w = f.decode(i as u32);
            assert!(d.to_bits() == w.to_bits() || (d.is_nan() && w.is_nan()), "{f:?} {i:#x}");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn arm_fp8_hardware_matches_scalar() {
    use crate::aarch64::ops::neon::fp8::{Fp8, Fp8Format};
    let Some(t) = Fp8::detect() else {
        std::eprintln!("FP8 hardware not detected, skipping");
        return;
    };
    std::eprintln!("FP8 hardware path active");
    let mut vals: Vec<f32> = (0..3000).map(|i| ((i as f32 - 1500.0) * 0.29).powi(3) * 1e-2).collect();
    vals.extend([0.0, -0.0, 1e-9, 1e-3, 448.0, 449.0, 464.0, 480.0, 1e5, 57344.0, 61440.0, 1e30, -1e30, f32::INFINITY, f32::NEG_INFINITY]);
    for (hw, f) in [(Fp8Format::E4m3, MiniFloat::E4M3Fn), (Fp8Format::E5m2, MiniFloat::E5M2)] {
        for (sat, ov) in [(true, Overflow::Saturate), (false, Overflow::Special)] {
            let mut got = vec![0u8; vals.len()];
            t.f32_to_fp8_slice(hw, sat, &vals, &mut got);
            for (i, &x) in vals.iter().enumerate() {
                assert_eq!(got[i] as u32, f.encode(x, ov), "{f:?} sat={sat} x={x}");
            }
        }
        let codes: Vec<u8> = (0..=255).collect();
        let mut dec = vec![0.0f32; 256];
        t.fp8_to_f32_slice(hw, &codes, &mut dec);
        for i in 0..256 {
            let w = f.decode(i as u32);
            assert!(dec[i].to_bits() == w.to_bits() || (dec[i].is_nan() && w.is_nan()), "{f:?} {i:#x}");
        }
    }
}

#[test]
fn fp8_u8_slices_match_scalar_codec() {
    let vals: Vec<f32> = (0..777).map(|i| ((i as f32 - 400.0) * 0.41).powi(3) * 1e-2).collect();
    for f in [MiniFloat::E4M3Fn, MiniFloat::E5M2, MiniFloat::E4M3Fnuz, MiniFloat::E5M2Fnuz] {
        for ov in [Overflow::Saturate, Overflow::Special] {
            let mut got = vec![0u8; vals.len()];
            f.encode_u8_slice(&vals, &mut got, ov);
            for (i, &x) in vals.iter().enumerate() {
                assert_eq!(got[i] as u32, f.encode(x, ov), "{f:?} {ov:?} {x}");
            }
        }
    }
}

// ---- Tenstorrent BFP_B ----
use super::bfp::*;

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Mirrors `gen_word` in scripts/tt-oracle/oracle.cpp.
fn bfp_word(rng: &mut Xs, row: u32) -> u32 {
    let u = rng.next();
    let kind = u & 7;
    if kind == 0 {
        return (u >> 40) as u32 & 0x8000_0000;
    }
    if kind == 1 {
        return (u >> 8) as u32 & 0x807F_FFFF;
    }
    if kind == 2 && row % 97 == 0 {
        return (u >> 8) as u32 | 0x7F80_0000;
    }
    let base = 120 + row % 20;
    let e = base + ((u >> 40) % 10) as u32 - 5;
    (((u >> 63) as u32) << 31) | (e << 23) | ((u >> 8) as u32 & 0x7F_FFFF)
}

fn bfp_words() -> Vec<u32> {
    let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
    (0..5000u32).flat_map(|r| (0..16).map(move |_| r)).map(|r| bfp_word(&mut rng, r)).collect()
}

fn fnv(h: &mut u64, b: u8) {
    *h ^= b as u64;
    *h = h.wrapping_mul(0x100_0000_01b3);
}

fn bfp_hash(elem: BfpElem, exps: &[u8], data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let rb = elem.row_bytes();
    for (r, &e) in exps.iter().enumerate() {
        fnv(&mut h, e);
        data[r * rb..][..rb].iter().for_each(|&b| fnv(&mut h, b));
    }
    h
}

// FNV-1a of tt-metal's `row_scalar` output, from scripts/tt-oracle (real bfp_simd.hpp).
const BFP_ORACLE: [(BfpElem, bool, u64); 6] = [
    (BfpElem::M7, false, 0xb1fe9e3c9d5acc27),
    (BfpElem::M7, true, 0xf36a025efdf7cfe7),
    (BfpElem::M3, false, 0xae31e6784c4da60c),
    (BfpElem::M3, true, 0xdf714a5662d559f4),
    (BfpElem::M1, false, 0x71fe9f9c13841376),
    (BfpElem::M1, true, 0x3f456ba7344920d1),
];

#[test]
fn bfp_oracle_matches_tt_metal() {
    let words = bfp_words();
    let f32s: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
    let bf: Vec<u16> = words.iter().map(|&w| (w >> 16) as u16).collect();
    for (elem, is_bf, want) in BFP_ORACLE {
        for hw in [false, true] {
            let mut exps = vec![0u8; 5000];
            let mut data = vec![0u8; 5000 * elem.row_bytes()];
            match (is_bf, hw) {
                (false, true) => bfp_encode_rows(elem, &f32s, &mut exps, &mut data),
                (false, false) => bfp_encode_rows_scalar(elem, &f32s, &mut exps, &mut data),
                (true, true) => bfp_encode_rows_bf16(elem, &bf, &mut exps, &mut data),
                (true, false) => bfp_encode_rows_bf16_scalar(elem, &bf, &mut exps, &mut data),
            }
            assert_eq!(bfp_hash(elem, &exps, &data), want, "{elem:?} bf16={is_bf} dispatch={hw}");
        }
    }
}

#[test]
fn bfp_decode_roundtrip_and_edges() {
    // exactly representable row: 1, -0.5, 0.25, ... all multiples of 2^-6 under shared exp 127
    let row: [f32; 16] = [1.0, -0.5, 0.25, 0.0, -0.0, 1.984375, -1.0, 0.015625, 0.5, 0.75, -0.125, 0.0625, 1.5, 1.25, -1.75, 0.03125];
    let (mut e, mut d) = ([0u8; 1], [0u8; 16]);
    bfp_encode_rows(BfpElem::M7, &row, &mut e, &mut d);
    assert_eq!(e[0], 127);
    let mut back = [0f32; 16];
    bfp_decode_rows(BfpElem::M7, &e, &d, &mut back);
    for i in 0..16 {
        assert_eq!(back[i], if row[i] == 0.0 { 0.0 } else { row[i] }, "{i}"); // -0 suppressed
    }
    let mut bb = [0u16; 16];
    bfp_decode_rows_bf16(BfpElem::M7, &e, &d, &mut bb);
    for i in 0..16 {
        assert_eq!(f32::from_bits((bb[i] as u32) << 16), back[i]);
    }
    // exponent-0 inputs flush; all-zero row has shared 0 and decodes to zero
    let z = [f32::from_bits(1); 16];
    bfp_encode_rows(BfpElem::M3, &z, &mut e, &mut d[..8]);
    assert_eq!(e[0], 0);
    assert!(d[..8].iter().all(|&b| b == 0));
    // M1 keeps the top element at magnitude 1 (hidden one), small ones round away
    let mut r1 = [0.001f32; 16];
    r1[0] = 8.0;
    r1[1] = -8.0;
    let mut d1 = [0u8; 4];
    bfp_encode_rows(BfpElem::M1, &r1, &mut e, &mut d1);
    let mut b1 = [0f32; 16];
    bfp_decode_rows(BfpElem::M1, &e, &d1, &mut b1);
    assert_eq!((b1[0], b1[1], b1[2]), (8.0, -8.0, 0.0));
}

#[test]
fn mx_block_size_is_a_parameter() {
    let src = data(64);
    for blk in [16usize, 32, 64] {
        let nb = 64 / blk;
        let elem = MxElem::Fp4;
        let mut sc = vec![0u8; nb];
        let mut d = vec![0u8; nb * elem.block_bytes_for(blk)];
        mx_quantize_block(elem, blk, &src, &mut sc, &mut d);
        let mut out = vec![0.0; 64];
        mx_dequantize_block(elem, blk, &sc, &d, &mut out);
        for b in 0..nb {
            let amax = src[b * blk..][..blk].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for i in b * blk..(b + 1) * blk {
                assert!((src[i] - out[i]).abs() <= 0.35 * amax, "{blk}: {} vs {}", src[i], out[i]);
            }
        }
    }
    // default wrappers are the 32-block case
    let (mut s1, mut d1, mut s2, mut d2) = ([0u8; 2], [0u8; 32], [0u8; 2], [0u8; 32]);
    mx_quantize(MxElem::Fp4, &src, &mut s1, &mut d1);
    mx_quantize_block(MxElem::Fp4, 32, &src, &mut s2, &mut d2);
    assert_eq!((s1, d1), (s2, d2));
}

// ---- GGUF, against scripts/gguf-oracle (ggml's dequantizers) ----
use super::gguf::*;

fn gh(h: &mut u64, f: f32) {
    for i in 0..4 {
        fnv(h, (f.to_bits() >> (8 * i)) as u8);
    }
}

fn rhalf(rng: &mut Xs) -> u16 {
    let u = rng.next();
    (((u >> 20) & 0x8000) as u16) | ((8 + (u >> 5) % 14) as u16) << 10 | (u & 0x3FF) as u16
}

#[test]
fn gguf_matches_ggml() {
    let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
    let mut b8 = Vec::new();
    for _ in 0..64 {
        b8.extend(rhalf(&mut rng).to_le_bytes());
        (0..32).for_each(|_| b8.push(rng.next() as u8));
    }
    let mut y = vec![0f32; 64 * 32];
    dequantize_q8_0(&b8, &mut y);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    y.iter().for_each(|&v| gh(&mut h, v));
    assert_eq!(h, 0xb680d70beff8c7b9, "q8_0");

    let mut b6 = Vec::new();
    for _ in 0..16 {
        (0..128 + 64 + 16).for_each(|_| b6.push(rng.next() as u8));
        b6.extend(rhalf(&mut rng).to_le_bytes());
    }
    let mut y = vec![0f32; 16 * 256];
    dequantize_q6_k(&b6, &mut y);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    y.iter().for_each(|&v| gh(&mut h, v));
    assert_eq!(h, 0xe74fcf89b24037b8, "q6_k");
}
