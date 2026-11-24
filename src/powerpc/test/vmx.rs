//! `Vmx` kernels and `powerpc::auto` against portable references.
//!
//! Run under `qemu-ppc -cpu 7400` (G4, 32-bit) or `qemu-ppc64 -cpu 970` (G5);
//! without AltiVec the token tests return early and only `auto` is checked.
//! The paired-single and VMX128 kernels never run here (bare metal only);
//! their encodings are tested in `ops/encode.rs`.

use crate::powerpc::auto;

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

fn lengths() -> [usize; 8] {
    [1, 3, 15, 16, 17, 64, 257, 4099]
}

/// Every misalignment of the inputs against the output (f32 steps).
fn offsets() -> [(usize, usize, usize); 4] {
    [(0, 0, 0), (1, 2, 3), (3, 1, 0), (2, 0, 1)]
}

#[test]
fn elementwise_is_exact_at_every_alignment() {
    for (oa, ob, oo) in offsets() {
        for n in lengths() {
            let a = grid(n + 4, n as u32);
            let b = grid(n + 4, n as u32 + 9);
            let c = grid(n + 4, n as u32 + 17);
            let (a, b, c) = (&a[oa..oa + n], &b[ob..ob + n], &c[oo..oo + n]);
            let mut buf = vec![f32::NAN; n + 4];
            let out = &mut buf[oo..oo + n];
            auto::add_f32(a, b, out);
            assert!(
                out.iter()
                    .zip(a.iter().zip(b))
                    .all(|(&o, (&x, &y))| o == x + y),
                "add {n}"
            );
            auto::sub_f32(a, b, out);
            assert!(
                out.iter()
                    .zip(a.iter().zip(b))
                    .all(|(&o, (&x, &y))| o == x - y),
                "sub {n}"
            );
            auto::mul_f32(a, b, out);
            assert!(
                out.iter()
                    .zip(a.iter().zip(b))
                    .all(|(&o, (&x, &y))| o == x * y),
                "mul {n}"
            );
            // Products and sums of quarters are exact, so fused == unfused here.
            auto::fmadd_f32(a, b, c, out);
            for i in 0..n {
                assert_eq!(out[i], a[i] * b[i] + c[i], "fmadd {n} {i}");
            }
            // Nothing outside `out` was written.
            assert!(
                buf[..oo].iter().chain(&buf[oo + n..]).all(|x| x.is_nan()),
                "bounds {n}"
            );

            let ia: Vec<i32> = (0..n as i32).map(|i| i.wrapping_mul(0x3779_b9b1)).collect();
            let ib: Vec<i32> = (0..n as i32)
                .map(|i| i.wrapping_mul(-0x61c8_8647))
                .collect();
            let mut io = vec![0i32; n];
            auto::add_i32(&ia, &ib, &mut io);
            for i in 0..n {
                assert_eq!(io[i], ia[i].wrapping_add(ib[i]), "add_i32 {n} {i}");
            }
        }
    }
}

#[test]
fn reductions_match_the_portable_rules() {
    for n in lengths() {
        for off in [0, 1, 3] {
            let all = data(n + off, n as u32);
            let a = &all[off..];
            let max = a
                .iter()
                .fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
            let min = a
                .iter()
                .fold(f32::INFINITY, |m, &x| if x < m { x } else { m });
            assert_eq!(auto::reduce_max_f32(a).to_bits(), max.to_bits(), "max {n}");
            assert_eq!(auto::reduce_min_f32(a).to_bits(), min.to_bits(), "min {n}");
            let first = a.iter().position(|&x| x == max).map(|i| (i, max));
            assert_eq!(auto::argmax_f32(a), first, "argmax {n}");
            assert_eq!(
                auto::reduce_sum_f32(a).is_nan(),
                a.iter().any(|x| x.is_nan()),
                "sum NaN {n}"
            );
            let clean: Vec<f32> = a
                .iter()
                .map(|&x| if x.is_nan() { 0.0 } else { x })
                .collect();
            let sum: f64 = clean.iter().map(|&x| x as f64).sum();
            let got = auto::reduce_sum_f32(&clean) as f64;
            assert!(
                (got - sum).abs() <= 1e-3 * (1.0 + sum.abs()),
                "sum {n}: {got} vs {sum}"
            );
        }
    }
    assert_eq!(auto::reduce_max_f32(&[f32::NAN; 40]), f32::NEG_INFINITY);
    assert_eq!(auto::reduce_min_f32(&[f32::NAN; 40]), f32::INFINITY);
    assert_eq!(auto::argmax_f32(&[f32::NAN; 40]), None);
    let mut inf = [f32::NAN; 40];
    inf[33] = f32::NEG_INFINITY;
    assert_eq!(auto::argmax_f32(&inf), Some((33, f32::NEG_INFINITY)));
}

#[test]
fn dot_is_exact_on_the_grid() {
    for n in lengths() {
        for (oa, ob) in [(0, 0), (1, 3), (2, 0)] {
            let a = grid(n + oa, 5 + n as u32);
            let b = grid(n + ob, 7 + n as u32);
            let (a, b) = (&a[oa..], &b[ob..]);
            // Quarter products sum exactly (well below 2^24 quarters), so every
            // summation order gives the same answer.
            let want: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
            assert_eq!(auto::dot_f32(a, b), want as f32, "dot {n}");
            let s: f64 = a.iter().map(|&x| x as f64).sum();
            assert_eq!(auto::reduce_sum_f32(a), s as f32, "sum {n}");
        }
    }
    let mut a = grid(100, 1);
    a[70] = f32::INFINITY;
    assert_eq!(auto::dot_f32(&a, &[1.0; 100]), f32::INFINITY);
    a[71] = f32::NAN;
    assert!(auto::dot_f32(&a, &[1.0; 100]).is_nan());
}

#[test]
fn conversions_are_bit_exact() {
    for n in lengths() {
        for (oq, oo) in [(0, 0), (1, 0), (0, 3), (5, 1)] {
            let q: Vec<u16> = (0..n + oq)
                .map(|i| (i as u16).wrapping_mul(40503))
                .collect();
            let q = &q[oq..];
            let mut buf = vec![0f32; n + 4];
            let out = &mut buf[oo..oo + n];
            auto::bf16_to_f32(q, out);
            for i in 0..n {
                assert_eq!(out[i].to_bits(), (q[i] as u32) << 16, "bf16 {n} {i}");
            }
            for (a, b) in [(0.5f32, -3.0f32), (1.0 / 3.0, 0.1), (-7.25, 1e6)] {
                auto::affine_u16_f32(q, a, b, out);
                for i in 0..n {
                    assert_eq!(
                        out[i].to_bits(),
                        (a * q[i] as f32 + b).to_bits(),
                        "affine {n} {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn portable_only_ops_keep_their_contract() {
    let a = data(300, 3);
    let mut out = vec![0u32; a.len()];
    let k = auto::compress_gt_f32(&a, 17.5, &mut out);
    let want: Vec<u32> = (0..a.len() as u32)
        .filter(|&i| a[i as usize] > 17.5)
        .collect();
    assert_eq!(&out[..k], &want[..]);
    let x: Vec<i8> = (0..300).map(|i| (i * 37 + 11) as i8).collect();
    let y: Vec<i8> = (0..300).map(|i| (i * 91 + 5) as i8).collect();
    let d = x
        .iter()
        .zip(&y)
        .fold(0i32, |s, (&p, &q)| s.wrapping_add(p as i32 * q as i32));
    assert_eq!(auto::dot_i8_i32(&x, &y), d);
}

/// With `MIRACULIX_EXPECT_VMX=1` (set in CI under `qemu-ppc -cpu 7400` and
/// `qemu-ppc64 -cpu 970`) the kernels above must really have run.
#[test]
fn vmx_is_detected_where_expected() {
    if std::env::var_os("MIRACULIX_EXPECT_VMX").is_some() {
        #[cfg(target_endian = "big")]
        assert!(
            crate::powerpc::ops::vmx::Vmx::detect().is_some(),
            "MIRACULIX_EXPECT_VMX is set but AltiVec is not detected"
        );
        #[cfg(not(target_endian = "big"))]
        panic!("MIRACULIX_EXPECT_VMX is set, but Vmx is only built for big-endian targets");
    }
}
