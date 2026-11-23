//! `Rvv` kernels against portable references. Run under `qemu-riscv64 -cpu rv64,v=true`;
//! without `V` every test returns early.
use crate::formats::bfp::*;
use crate::riscv::ops::rvv::Rvv;

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 97 == 96 {
                f32::NAN
            } else {
                ((s % 4001) as f32 - 2000.0) * 0.0625
            }
        })
        .collect()
}

fn lengths() -> [usize; 8] {
    [1, 3, 15, 16, 17, 64, 257, 4099]
}

#[test]
fn reductions_match_the_portable_rules() {
    let Some(t) = Rvv::detect() else { return };
    for n in lengths() {
        let a = data(n, n as u32);
        let max = a
            .iter()
            .fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
        let min = a
            .iter()
            .fold(f32::INFINITY, |m, &x| if x < m { x } else { m });
        assert_eq!(t.reduce_max_f32(&a).to_bits(), max.to_bits(), "max {n}");
        assert_eq!(t.reduce_min_f32(&a).to_bits(), min.to_bits(), "min {n}");
        let first = a.iter().position(|&x| x == max).map(|i| (i, max));
        assert_eq!(t.argmax_f32(&a), first, "argmax {n}");
        let clean: Vec<f32> = a
            .iter()
            .map(|&x| if x.is_nan() { 0.0 } else { x })
            .collect();
        let sum: f64 = clean.iter().map(|&x| x as f64).sum();
        assert!(
            (t.reduce_sum_f32(&clean) as f64 - sum).abs() <= 1e-3 * (1.0 + sum.abs()),
            "sum {n}"
        );
        assert!(
            t.reduce_sum_f32(&a).is_nan() == a.iter().any(|x| x.is_nan()),
            "sum NaN {n}"
        );
        for thr in [-1000.0, 0.0, 17.5, f32::INFINITY] {
            let want: Vec<u32> = (0..n as u32).filter(|&i| a[i as usize] > thr).collect();
            let mut out = vec![0u32; n];
            let k = t.compress_gt_f32(&a, thr, &mut out);
            assert_eq!(&out[..k], &want[..], "compress {n} > {thr}");
        }
    }
    assert_eq!(t.reduce_max_f32(&[f32::NAN; 5]), f32::NEG_INFINITY);
    assert_eq!(t.argmax_f32(&[f32::NAN; 5]), None);
}

#[test]
fn dot_i8_wraps_as_the_portable_sum() {
    let Some(t) = Rvv::detect() else { return };
    for n in lengths() {
        let a: Vec<i8> = (0..n).map(|i| (i * 37 + 11) as i8).collect();
        let b: Vec<i8> = (0..n).map(|i| (i * 91 + 5) as i8).collect();
        let want = a
            .iter()
            .zip(&b)
            .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32));
        assert_eq!(t.dot_i8_i32(&a, &b), want, "{n}");
    }
}

#[test]
fn exp_keeps_its_contract() {
    let Some(t) = Rvv::detect() else { return };
    let a: Vec<f32> = (-900..=900)
        .map(|i| i as f32 * 0.1)
        .chain([-1e30, 1e30, f32::NAN, -0.0])
        .collect();
    let mut out = vec![0f32; a.len()];
    t.exp_f32(&a, &mut out);
    for (&x, &y) in a.iter().zip(&out) {
        let want = crate::reduce_common::exp_scalar(x);
        if x.is_nan() {
            assert!(y.is_nan());
        } else if x < -87.33655 {
            assert_eq!(y, 0.0, "{x}");
        } else if x > 88.72283 {
            assert_eq!(y, f32::INFINITY, "{x}");
        } else {
            assert!(
                ((y - want) / want).abs() <= 3e-7,
                "exp({x}) = {y}, scalar {want}"
            );
        }
    }
}

#[test]
fn bfp_rows_match_the_portable_codec() {
    let Some(t) = Rvv::detect() else { return };
    let a: Vec<f32> = data(16 * 300, 7)
        .into_iter()
        .map(|x| if x.is_nan() { 1.0 } else { x })
        .collect();
    let bf: Vec<u16> = a.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    for elem in [BfpElem::M7, BfpElem::M3, BfpElem::M1] {
        let rows = a.len() / 16;
        let (mut e1, mut d1) = (vec![0u8; rows], vec![0u8; rows * elem.row_bytes()]);
        let (mut e2, mut d2) = (e1.clone(), d1.clone());
        t.bfp_encode_rows(elem, &a, &mut e1, &mut d1);
        bfp_encode_rows_scalar(elem, &a, &mut e2, &mut d2);
        assert_eq!((&e1, &d1), (&e2, &d2), "{elem:?} f32");
        t.bfp_encode_rows_bf16(elem, &bf, &mut e1, &mut d1);
        bfp_encode_rows_bf16_scalar(elem, &bf, &mut e2, &mut d2);
        assert_eq!((&e1, &d1), (&e2, &d2), "{elem:?} bf16");
    }
}

/// With `MIRACULIX_EXPECT_RVV=1` (set when running under `qemu-riscv64 -cpu rv64,v=true`) the
/// tests above must not return early.
#[test]
fn v_is_detected_where_expected() {
    if std::env::var_os("MIRACULIX_EXPECT_RVV").is_some() {
        assert!(
            Rvv::detect().is_some(),
            "MIRACULIX_EXPECT_RVV is set but V is not detected"
        );
    }
}

/// Quarters in `-8..=8`, no NaN: every product and sum here is exact in `f32`.
fn grid(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s % 65) as f32 - 32.0) * 0.25
        })
        .collect()
}

#[test]
fn sliding_dot_is_the_definition() {
    let Some(t) = Rvv::detect() else { return };
    for taps_len in [0, 1, 3, 64, 65] {
        for n in lengths() {
            let taps = grid(taps_len, 3 + taps_len as u32);
            let signal = grid(n + taps_len.saturating_sub(1), n as u32);
            let mut out = vec![f32::NAN; n];
            t.sliding_dot_f32(&signal, &taps, &mut out);
            for (i, &got) in out.iter().enumerate() {
                let want: f64 = taps
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| signal[i + k] as f64 * c as f64)
                    .sum();
                assert_eq!(got, want as f32, "taps {taps_len} n {n} i {i}");
            }
        }
    }
}

#[test]
fn dot_and_affine_match_the_portable_code() {
    let Some(t) = Rvv::detect() else { return };
    for n in lengths() {
        let (a, b) = (grid(n, n as u32), grid(n, 7 + n as u32));
        let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        assert_eq!(t.dot_f32(&a, &b), want as f32, "dot {n}");
        let q: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(40503)).collect();
        let mut out = vec![0.0; n];
        t.affine_u16_f32(&q, 0.37, -5.5, &mut out);
        for (i, (&x, &got)) in q.iter().zip(&out).enumerate() {
            assert_eq!(
                got.to_bits(),
                (0.37f32 * x as f32 + -5.5).to_bits(),
                "affine {n} {i}"
            );
        }
    }
    assert_eq!(t.dot_f32(&[], &[]), 0.0);
}
