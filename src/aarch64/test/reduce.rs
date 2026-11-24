use super::*;
use crate::aarch64::ops::neon::neon::Neon;

/// `false` = portable path, `true` = Neon (skipped if Neon is somehow absent).
fn tiers() -> Vec<bool> {
    if Neon::detect().is_some() {
        vec![false, true]
    } else {
        vec![false]
    }
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn data(n: usize, seed: u64, nan_every: usize) -> Vec<f32> {
    let mut r = rng(seed);
    (0..n)
        .map(|i| {
            if nan_every > 0 && i % nan_every == nan_every - 1 {
                f32::NAN
            } else {
                // coarse grid so ties are common
                ((r() % 401) as f32 - 200.0) * 0.25
            }
        })
        .collect()
}

fn ref_max(a: &[f32]) -> f32 {
    a.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f32::NEG_INFINITY, f32::max)
}

#[test]
fn max_min_argmax_match_scalar_on_every_tier() {
    for t in tiers() {
        for n in (0..140).chain([1000, 4097]) {
            for nan in [0, 3, 1] {
                let a = data(n, n as u64 + 7, nan);
                assert_eq!(
                    reduce_max_f32_with(t, &a),
                    ref_max(&a),
                    "{t:?} n={n} nan={nan}"
                );
                let mn = a
                    .iter()
                    .copied()
                    .filter(|x| !x.is_nan())
                    .fold(f32::INFINITY, f32::min);
                assert_eq!(reduce_min_f32_with(t, &a), mn, "{t:?} min n={n}");
                let want = a.iter().enumerate().filter(|(_, x)| !x.is_nan()).fold(
                    None,
                    |b: Option<(usize, f32)>, (i, &x)| match b {
                        Some((_, bv)) if bv >= x => b,
                        _ => Some((i, x)),
                    },
                );
                assert_eq!(argmax_f32_with(t, &a), want, "{t:?} argmax n={n} nan={nan}");
            }
        }
    }
    assert_eq!(argmax_f32(&[]), None);
    assert_eq!(argmax_f32(&[f32::NAN; 40]), None);
    assert_eq!(
        argmax_f32(&[f32::NEG_INFINITY; 20]),
        Some((0, f32::NEG_INFINITY))
    );
}

#[test]
fn sum_is_close_on_every_tier() {
    for t in tiers() {
        for n in (0..140).chain([1000, 100_003]) {
            let a = data(n, 3, 0);
            let want: f64 = a.iter().map(|&x| x as f64).sum();
            let got = reduce_sum_f32_with(t, &a) as f64;
            assert!(
                (got - want).abs() <= 1e-3 + want.abs() * 1e-6,
                "{t:?} n={n}: {got} vs {want}"
            );
        }
    }
    assert!(reduce_sum_f32(&[1.0, f32::NAN]).is_nan());
}

#[test]
fn bf16_widening_is_exact() {
    for t in tiers() {
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 33, 200] {
            let a: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(40503)).collect();
            let mut out = vec![0f32; n];
            bf16_to_f32_with(t, &a, &mut out);
            for i in 0..n {
                assert_eq!(out[i].to_bits(), (a[i] as u32) << 16, "{t:?} n={n} i={i}");
            }
        }
    }
}

#[test]
fn exp_error_bound_on_every_tier() {
    let mut xs: Vec<f32> = (-9000..=8900).map(|i| i as f32 * 0.01).collect();
    xs.extend([
        0.0,
        -0.0,
        1e-8,
        -1e-8,
        88.72283,
        88.8,
        -87.33654,
        -87.4,
        -200.0,
        200.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]);
    for t in tiers() {
        let mut out = vec![0f32; xs.len()];
        exp_f32_with(t, &xs, &mut out);
        let mut worst = 0f64;
        for (&x, &y) in xs.iter().zip(&out) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else if x < -87.33655 {
                assert_eq!(y, 0.0, "{t:?} {x}");
            } else if x > 88.72283 {
                assert_eq!(y, f32::INFINITY, "{t:?} {x}");
            } else {
                let want = (x as f64).exp();
                let rel = ((y as f64) - want).abs() / want;
                worst = worst.max(rel);
                assert!(rel <= 3e-7, "{t:?} exp({x}) = {y}, want {want}, rel {rel}");
            }
        }
        eprintln!("{t:?}: worst exp rel error {worst:e}");
    }
}

#[test]
fn compress_and_topk_match_reference_on_every_tier() {
    for t in tiers() {
        for n in [0usize, 1, 5, 8, 15, 16, 17, 100, 255, 256, 257, 1000, 5000] {
            for nan in [0, 7] {
                let a = data(n, 11 + n as u64, nan);
                let thr = 10.0;
                let mut out = vec![0u32; n];
                let c = compress_gt_f32_with(t, &a, thr, &mut out);
                let want: Vec<u32> = (0..n as u32).filter(|&i| a[i as usize] > thr).collect();
                assert_eq!(&out[..c], &want[..], "{t:?} compress n={n}");
                for k in [1usize, 3, 40, 300] {
                    let (mut idx, mut val) = (vec![0u32; k], vec![0f32; k]);
                    let got = topk_f32_with(t, &a, &mut idx, &mut val);
                    let mut all: Vec<(usize, f32)> = a
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, x)| !x.is_nan())
                        .collect();
                    all.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
                    all.truncate(k);
                    assert_eq!(got, all.len(), "{t:?} topk count n={n} k={k}");
                    for j in 0..got {
                        assert_eq!(
                            (idx[j] as usize, val[j]),
                            all[j],
                            "{t:?} topk n={n} k={k} j={j}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn affine_dot_logsumexp_match_f64() {
    let mut r = rng(3);
    for t in tiers() {
        for n in (0..70).chain([1000, 4099]) {
            let q: Vec<u16> = (0..n).map(|_| r() as u16).collect();
            let mut out = vec![0f32; n];
            affine_u16_f32_with(t, &q, 3.0517578e-5, -1.25, &mut out);
            for i in 0..n {
                let want = 3.0517578e-5f64 * q[i] as f64 - 1.25;
                assert!((out[i] as f64 - want).abs() <= 2e-7 * want.abs().max(1.0));
            }
            let a: Vec<f32> = (0..n).map(|_| (r() % 2001) as f32 * 0.01 - 10.0).collect();
            let b: Vec<f32> = (0..n).map(|_| (r() % 2001) as f32 * 0.01 - 10.0).collect();
            let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let abs: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| (x as f64 * y as f64).abs())
                .sum();
            assert!((dot_f32_with(t, &a, &b) as f64 - want).abs() <= 1e-5 * abs.max(1.0));
            if n > 0 {
                let m = a.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y)) as f64;
                let want = m + a.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
                let got = logsumexp_f32_with(t, &a) as f64;
                assert!(
                    (got - want).abs() <= 2e-6 * want.abs().max(1.0),
                    "n={n}: {got} vs {want}"
                );
            }
        }
    }
    assert_eq!(logsumexp_f32(&[]), f32::NEG_INFINITY);
    assert!(logsumexp_f32(&[1.0, f32::NAN]).is_nan());
}

fn gemm_want(kc: usize, a: &[f32], b: &[f32], init: &[[f32; GEMM_NR]; GEMM_MR]) -> Vec<f64> {
    let mut w = vec![0f64; GEMM_MR * GEMM_NR];
    for i in 0..GEMM_MR {
        for j in 0..GEMM_NR {
            w[i * GEMM_NR + j] = init[i][j] as f64
                + (0..kc)
                    .map(|k| a[k * GEMM_MR + i] as f64 * b[k * GEMM_NR + j] as f64)
                    .sum::<f64>();
        }
    }
    w
}

#[test]
fn gemm_kernel_matches_f64_on_every_tier() {
    let mut r = rng(11);
    let mut f = move || (r() % 2001) as f32 * 1e-3 - 1.0;
    for t in tiers() {
        for kc in [0usize, 1, 2, 7, 64, 256, 513] {
            let a: Vec<f32> = (0..kc * GEMM_MR).map(|_| f()).collect();
            let b: Vec<f32> = (0..kc * GEMM_NR).map(|_| f()).collect();
            let mut acc = [[0f32; GEMM_NR]; GEMM_MR];
            for row in acc.iter_mut() {
                row.iter_mut().for_each(|x| *x = f());
            }
            let want = gemm_want(kc, &a, &b, &acc);
            let before = acc;
            gemm_f32_kernel_with(t, kc, &a, &b, &mut acc);
            if kc == 0 {
                assert_eq!(acc, before, "{t:?}: kc = 0 must leave acc unchanged");
            }
            for i in 0..GEMM_MR {
                for j in 0..GEMM_NR {
                    let (g, w) = (acc[i][j] as f64, want[i * GEMM_NR + j]);
                    assert!(
                        (g - w).abs() <= 2e-6 * (kc as f64 + 1.0),
                        "{t:?} kc={kc} [{i}][{j}]: {g} vs {w}"
                    );
                }
            }
        }
    }
}

#[test]
fn gemm_kernel_propagates_nan() {
    let mut a = vec![1f32; 3 * GEMM_MR];
    a[GEMM_MR + 2] = f32::NAN;
    let b = vec![1f32; 3 * GEMM_NR];
    let mut acc = [[0f32; GEMM_NR]; GEMM_MR];
    gemm_f32_kernel(3, &a, &b, &mut acc);
    assert!(acc[2].iter().all(|x| x.is_nan()));
    assert!(acc[1].iter().all(|&x| x == 3.0));
}

#[test]
#[should_panic]
fn gemm_kernel_panics_on_short_panel() {
    let mut acc = [[0f32; GEMM_NR]; GEMM_MR];
    gemm_f32_kernel(2, &[0.0; GEMM_MR], &[0.0; 2 * GEMM_NR], &mut acc);
}

/// Products and sums of the quarter grid stay exact in `f32`, so every tier
/// must give the definition bit for bit.
#[test]
fn sliding_dot_matches_the_definition_on_every_tier() {
    for t in tiers() {
        for taps_len in [0, 1, 3, 64, 65] {
            for n in [0, 1, 3, 4, 15, 16, 31, 33, 127, 128, 129, 300] {
                let taps = data(taps_len, 11 + taps_len as u64, 0);
                let signal = data(n + taps_len.saturating_sub(1), 5, 0);
                let mut out = vec![f32::NAN; n];
                sliding_dot_f32_with(t, &signal, &taps, &mut out);
                for (i, &got) in out.iter().enumerate() {
                    let want: f64 = taps
                        .iter()
                        .enumerate()
                        .map(|(k, &c)| signal[i + k] as f64 * c as f64)
                        .sum();
                    assert_eq!(got, want as f32, "{t:?} taps={taps_len} n={n} i={i}");
                }
            }
        }
    }
}

/// Every byte pair, small and big `den`s, lengths around the block of 16:
/// bit for bit the integer definition.
#[test]
fn lerp_matches_the_definition_on_every_tier() {
    let a: Vec<u8> = (0..=255u8).flat_map(|x| [x; 256]).collect();
    let b: Vec<u8> = (0..=255u8).cycle().take(65536).collect();
    let cases = (0..=7).map(|n| (n, 7)).chain([
        (0, 1),
        (1, 1),
        (333, 1000),
        (999, 1000),
        (1, 32767),
        (16384, 32767),
        (32766, 32767),
    ]);
    for t in tiers() {
        for (num, den) in cases.clone() {
            for len in [0, 1, 15, 16, 17, 31, 32, 33, 65536] {
                let mut out = vec![0xAA; len];
                lerp_u8_with(t, &a[..len], &b[..len], num, den, &mut out);
                for (i, &got) in out.iter().enumerate() {
                    let sum =
                        u32::from(a[i]) * u32::from(den - num) + u32::from(b[i]) * u32::from(num);
                    let want = (2 * sum + u32::from(den)) / (2 * u32::from(den));
                    assert_eq!(u32::from(got), want, "{t:?} {num}/{den} len={len} i={i}");
                }
            }
        }
    }
}

#[test]
fn sliding_dot_propagates_nan() {
    for t in tiers() {
        let mut signal = vec![1.0f32; 100];
        signal[40] = f32::NAN;
        let mut out = vec![0.0; 97];
        sliding_dot_f32_with(t, &signal, &[1.0; 4], &mut out);
        for (i, &x) in out.iter().enumerate() {
            assert_eq!(x.is_nan(), (37..=40).contains(&i), "{t:?} i={i}");
        }
    }
}

#[test]
#[should_panic(expected = "sliding_dot_f32")]
fn sliding_dot_wants_every_window() {
    sliding_dot_f32(&[0.0; 10], &[0.0; 4], &mut [0.0; 8]);
}
