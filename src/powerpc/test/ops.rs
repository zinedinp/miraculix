//! PowerPC kernels and `powerpc::auto` against portable references.
//!
//! Run under qemu-user (`qemu-ppc -cpu 7400`, `qemu-ppc64 -cpu power8`,
//! `qemu-ppc64le -cpu power8/9/10`). Each token is checked directly, not only
//! through `auto`, so the big-endian POWER8 job still runs `Vmx` although
//! `auto` prefers `Vsx` there. Tokens the CPU lacks are skipped; the
//! `MIRACULIX_EXPECT_*` guards at the end make sure CI did not skip them all.
//! Paired singles and VMX128 never run here (bare metal only): their encodings
//! are tested in `ops/encode.rs`, paired singles also by `scripts/ps-probe`.

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

fn lengths() -> [usize; 9] {
    [0, 1, 3, 15, 16, 17, 64, 257, 4099]
}

type Bin<T> = dyn Fn(&[T], &[T], &mut [T]);
type Fma = dyn Fn(&[f32], &[f32], &[f32], &mut [f32]);
type ArgMax = dyn Fn(&[f32]) -> Option<(usize, f32)>;
type Affine = dyn Fn(&[u16], f32, f32, &mut [f32]);

/// Element-wise ops at every relative misalignment; nothing outside `out` written.
fn check_elementwise(
    name: &str,
    add: &Bin<f32>,
    sub: &Bin<f32>,
    mul: &Bin<f32>,
    fma: &Fma,
    add_i: &Bin<i32>,
) {
    for (oa, ob, oo) in [(0, 0, 0), (1, 2, 3), (3, 1, 0), (2, 0, 1)] {
        for n in lengths() {
            let a = grid(n + 4, n as u32);
            let b = grid(n + 4, n as u32 + 9);
            let c = grid(n + 4, n as u32 + 17);
            let (a, b, c) = (&a[oa..oa + n], &b[ob..ob + n], &c[oo..oo + n]);
            let mut buf = vec![f32::NAN; n + 4];
            let out = &mut buf[oo..oo + n];
            add(a, b, out);
            assert!((0..n).all(|i| out[i] == a[i] + b[i]), "{name} add {n}");
            sub(a, b, out);
            assert!((0..n).all(|i| out[i] == a[i] - b[i]), "{name} sub {n}");
            mul(a, b, out);
            assert!((0..n).all(|i| out[i] == a[i] * b[i]), "{name} mul {n}");
            // Quarter products and sums are exact, so fused == unfused here.
            fma(a, b, c, out);
            assert!(
                (0..n).all(|i| out[i] == a[i] * b[i] + c[i]),
                "{name} fmadd {n}"
            );
            assert!(
                buf[..oo].iter().chain(&buf[oo + n..]).all(|x| x.is_nan()),
                "{name} bounds {n}"
            );

            let ia: Vec<i32> = (0..n as i32).map(|i| i.wrapping_mul(0x3779_b9b1)).collect();
            let ib: Vec<i32> = (0..n as i32)
                .map(|i| i.wrapping_mul(-0x61c8_8647))
                .collect();
            let mut io = vec![0i32; n + 1];
            add_i(&ia, &ib, &mut io[..n]);
            assert!(
                (0..n).all(|i| io[i] == ia[i].wrapping_add(ib[i])),
                "{name} add_i32 {n}"
            );
            assert_eq!(io[n], 0, "{name} add_i32 bounds {n}");
        }
    }
}

/// Sum and dot: exact on the grid (any order), NaN/inf propagate.
fn check_sum_dot(name: &str, sum: &dyn Fn(&[f32]) -> f32, dot: &dyn Fn(&[f32], &[f32]) -> f32) {
    for n in lengths() {
        for (oa, ob) in [(0, 0), (1, 3), (2, 0)] {
            let a = grid(n + oa, 5 + n as u32);
            let b = grid(n + ob, 7 + n as u32);
            let (a, b) = (&a[oa..], &b[ob..]);
            let want: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
            assert_eq!(dot(a, b), want as f32, "{name} dot {n}");
            let s: f64 = a.iter().map(|&x| x as f64).sum();
            assert_eq!(sum(a), s as f32, "{name} sum {n}");
        }
        let a = data(n, n as u32);
        assert_eq!(
            sum(&a).is_nan(),
            a.iter().any(|x| x.is_nan()),
            "{name} sum NaN {n}"
        );
    }
    let mut a = grid(100, 1);
    a[70] = f32::INFINITY;
    assert_eq!(dot(&a, &[1.0; 100]), f32::INFINITY, "{name}");
    a[99] = f32::NAN; // in the tail
    assert!(dot(&a, &[1.0; 100]).is_nan(), "{name}");
}

/// Max, min, argmax with NaN ignored, bit-exact against the scalar folds.
fn check_extremes(
    name: &str,
    max: &dyn Fn(&[f32]) -> f32,
    min: &dyn Fn(&[f32]) -> f32,
    argmax: &ArgMax,
) {
    for n in lengths() {
        for off in [0, 1, 3] {
            let all = data(n + off, n as u32);
            let a = &all[off..];
            let mx = a
                .iter()
                .fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
            let mn = a
                .iter()
                .fold(f32::INFINITY, |m, &x| if x < m { x } else { m });
            assert_eq!(max(a).to_bits(), mx.to_bits(), "{name} max {n}");
            assert_eq!(min(a).to_bits(), mn.to_bits(), "{name} min {n}");
            assert_eq!(
                argmax(a),
                a.iter().position(|&x| x == mx).map(|i| (i, mx)),
                "{name} argmax {n}"
            );
        }
    }
    assert_eq!(max(&[f32::NAN; 40]), f32::NEG_INFINITY, "{name}");
    assert_eq!(min(&[f32::NAN; 40]), f32::INFINITY, "{name}");
    assert_eq!(argmax(&[f32::NAN; 40]), None, "{name}");
    let mut inf = [f32::NAN; 40];
    inf[33] = f32::NEG_INFINITY;
    assert_eq!(argmax(&inf), Some((33, f32::NEG_INFINITY)), "{name}");
}

/// bf16 and affine u16 conversions, bit-exact at several alignments.
fn check_conversions(name: &str, bf16: &dyn Fn(&[u16], &mut [f32]), affine: &Affine) {
    for n in lengths() {
        for (oq, oo) in [(0, 0), (1, 0), (0, 3), (5, 1)] {
            let q: Vec<u16> = (0..n + oq)
                .map(|i| (i as u16).wrapping_mul(40503))
                .collect();
            let q = &q[oq..];
            let mut buf = vec![0f32; n + 4];
            let out = &mut buf[oo..oo + n];
            bf16(q, out);
            for i in 0..n {
                assert_eq!(out[i].to_bits(), (q[i] as u32) << 16, "{name} bf16 {n} {i}");
            }
            for (a, b) in [(0.5f32, -3.0f32), (1.0 / 3.0, 0.1), (-7.25, 1e6)] {
                affine(q, a, b, out);
                for i in 0..n {
                    assert_eq!(
                        out[i].to_bits(),
                        (a * q[i] as f32 + b).to_bits(),
                        "{name} affine {n} {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn auto_matches_the_references() {
    check_elementwise(
        "auto",
        &auto::add_f32,
        &auto::sub_f32,
        &auto::mul_f32,
        &auto::fmadd_f32,
        &auto::add_i32,
    );
    check_sum_dot("auto", &auto::reduce_sum_f32, &auto::dot_f32);
    check_extremes(
        "auto",
        &auto::reduce_max_f32,
        &auto::reduce_min_f32,
        &auto::argmax_f32,
    );
    check_conversions("auto", &auto::bf16_to_f32, &auto::affine_u16_f32);
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

#[test]
fn gemm_matches_the_scalar_kernel_exactly_on_the_grid() {
    use crate::reduce_common::{GEMM_MR, GEMM_NR};
    for kc in [0, 1, 3, 17, 64] {
        let a = grid(kc * GEMM_MR, kc as u32 + 1);
        let b = grid(kc * GEMM_NR, kc as u32 + 2);
        let init: Vec<f32> = grid(GEMM_MR * GEMM_NR, 3);
        let mut want = [[0f32; GEMM_NR]; GEMM_MR];
        for (i, row) in want.iter_mut().enumerate() {
            row.copy_from_slice(&init[i * GEMM_NR..][..GEMM_NR]);
        }
        let mut got = want;
        crate::reduce_common::gemm_kernel_scalar(kc, &a, &b, &mut want);
        auto::gemm_f32_kernel(kc, &a, &b, &mut got);
        assert_eq!(got, want, "kc {kc}");
    }
}

#[cfg(target_endian = "big")]
#[test]
fn vmx_kernels() {
    let Some(t) = crate::powerpc::ops::vmx::Vmx::detect() else {
        return;
    };
    check_elementwise(
        "vmx",
        &move |a, b, o| t.add_f32(a, b, o),
        &move |a, b, o| t.sub_f32(a, b, o),
        &move |a, b, o| t.mul_f32(a, b, o),
        &move |a, b, c, o| t.fmadd_f32(a, b, c, o),
        &move |a, b, o| t.add_i32(a, b, o),
    );
    check_sum_dot("vmx", &move |a| t.reduce_sum_f32(a), &move |a, b| {
        t.dot_f32(a, b)
    });
    check_extremes(
        "vmx",
        &move |a| t.reduce_max_f32(a),
        &move |a| t.reduce_min_f32(a),
        &move |a| t.argmax_f32(a),
    );
    check_conversions(
        "vmx",
        &move |a, o| t.bf16_to_f32(a, o),
        &move |q, a, b, o| t.affine_u16_f32(q, a, b, o),
    );
}

#[test]
fn vsx_kernels() {
    let Some(t) = crate::powerpc::ops::vsx::Vsx::detect() else {
        return;
    };
    check_elementwise(
        "vsx",
        &move |a, b, o| t.add_f32(a, b, o),
        &move |a, b, o| t.sub_f32(a, b, o),
        &move |a, b, o| t.mul_f32(a, b, o),
        &move |a, b, c, o| t.fmadd_f32(a, b, c, o),
        &move |a, b, o| t.add_i32(a, b, o),
    );
    check_sum_dot("vsx", &move |a| t.reduce_sum_f32(a), &move |a, b| {
        t.dot_f32(a, b)
    });
    check_extremes(
        "vsx",
        &move |a| t.reduce_max_f32(a),
        &move |a| t.reduce_min_f32(a),
        &move |a| t.argmax_f32(a),
    );
    check_conversions(
        "vsx",
        &move |a, o| t.bf16_to_f32(a, o),
        &move |q, a, b, o| t.affine_u16_f32(q, a, b, o),
    );
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn isa300_kernels() {
    let Some(t) = crate::powerpc::ops::isa300::Isa300::detect() else {
        return;
    };
    check_elementwise(
        "isa300",
        &move |a, b, o| t.add_f32(a, b, o),
        &move |a, b, o| t.sub_f32(a, b, o),
        &move |a, b, o| t.mul_f32(a, b, o),
        &move |a, b, c, o| t.fmadd_f32(a, b, c, o),
        &move |a, b, o| t.add_i32(a, b, o),
    );
    check_sum_dot("isa300", &move |a| t.reduce_sum_f32(a), &move |a, b| {
        t.dot_f32(a, b)
    });
}

#[cfg(target_arch = "powerpc64")]
#[test]
fn isa300_f16_matches_the_scalar_codec() {
    use crate::formats::{MiniFloat, Overflow};
    let Some(t) = crate::powerpc::ops::isa300::Isa300::detect() else {
        return;
    };
    // Every binary16 code, decoded: exact, NaN stays NaN.
    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let mut f = vec![0f32; codes.len()];
    t.cvt_f16_to_f32_slice(&codes, &mut f);
    for (&c, &x) in codes.iter().zip(&f) {
        let want = MiniFloat::F16.decode(c as u32);
        if want.is_nan() {
            assert!(x.is_nan(), "{c:#06x}");
        } else {
            assert_eq!(x.to_bits(), want.to_bits(), "{c:#06x}");
        }
    }
    // Encode: halfway cases, subnormals, overflow; NaN only checked as NaN.
    let mut src: Vec<f32> = f.iter().copied().filter(|x| !x.is_nan()).collect();
    src.extend(
        f.windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .filter(|x| x.is_finite()),
    );
    src.extend([
        65520.0,
        65519.0,
        1e9,
        -1e9,
        5.96e-8,
        2.98e-8,
        f32::MIN_POSITIVE,
    ]);
    let mut h = vec![0u16; src.len()];
    t.cvt_f32_to_f16_slice(&src, &mut h);
    for (&x, &got) in src.iter().zip(&h) {
        let want = MiniFloat::F16.encode(x, Overflow::Special) as u16;
        assert_eq!(got, want, "{x:e}");
    }
}

/// CI sets `MIRACULIX_EXPECT_POWER` to the kernels the QEMU CPU model must
/// provide (`vmx`, `power8`, `power9`, `power10`), so a detection bug cannot
/// turn every token test above into a silent skip.
#[test]
fn expected_kernels_are_detected() {
    let Ok(want) = std::env::var("MIRACULIX_EXPECT_POWER") else {
        return;
    };
    let f = crate::powerpc::detect_features();
    let level = crate::powerpc::detect_level();
    use crate::powerpc::{Feature, PowerLevel};
    match want.as_str() {
        "vmx" => assert!(f.contains(Feature::Altivec), "{f:?}"),
        "power8" => assert!(level >= PowerLevel::Isa207, "{level:?}"),
        "power9" => assert!(level >= PowerLevel::Isa300, "{level:?}"),
        "power10" => {
            assert!(level >= PowerLevel::Isa31, "{level:?}");
            assert!(f.contains(Feature::Mma), "{f:?}");
        }
        other => panic!("unknown MIRACULIX_EXPECT_POWER={other}"),
    }
}
