//! Oracle: round (trunc or ties-even), then map anything outside
//! `[-2^(n-1), 2^(n-1))` - and NaN - to `-2^(n-1)`.

use super::Frintts;

fn frint(x: f64, bits: i32, toward_zero: bool) -> f64 {
    let r = if toward_zero {
        x.trunc()
    } else {
        x.round_ties_even()
    };
    let lim = 2f64.powi(bits - 1);
    if r.is_nan() || r < -lim || r >= lim {
        -lim
    } else {
        r
    }
}

const SAMPLES: [f64; 16] = [
    0.5,
    -0.5,
    1.5,
    2.5,
    -2.5,
    2147483647.5,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    9.3e18,
    -9.3e18,
    1e30,
    3.75,
    -7.25,
    f64::INFINITY,
    f64::NAN,
];

#[test]
fn rnd_f32x4_matches_oracle() {
    let Some(t) = Frintts::detect() else { return };
    for chunk in SAMPLES.chunks(4) {
        let a: [f32; 4] = core::array::from_fn(|i| chunk[i] as f32);
        let o = |bits, z| a.map(|x| frint(x as f64, bits, z) as f32);
        assert_eq!(
            t.rnd32z_f32x4(a).map(f32::to_bits),
            o(32, true).map(f32::to_bits),
            "32z {a:?}"
        );
        assert_eq!(
            t.rnd32x_f32x4(a).map(f32::to_bits),
            o(32, false).map(f32::to_bits),
            "32x {a:?}"
        );
        assert_eq!(
            t.rnd64z_f32x4(a).map(f32::to_bits),
            o(64, true).map(f32::to_bits),
            "64z {a:?}"
        );
        assert_eq!(
            t.rnd64x_f32x4(a).map(f32::to_bits),
            o(64, false).map(f32::to_bits),
            "64x {a:?}"
        );
    }
}

#[test]
fn rnd_f64x2_matches_oracle() {
    let Some(t) = Frintts::detect() else { return };
    for chunk in SAMPLES.chunks(2) {
        let a = [chunk[0], chunk[1]];
        let o = |bits, z| a.map(|x| frint(x, bits, z));
        assert_eq!(
            t.rnd32z_f64x2(a).map(f64::to_bits),
            o(32, true).map(f64::to_bits),
            "32z {a:?}"
        );
        assert_eq!(
            t.rnd32x_f64x2(a).map(f64::to_bits),
            o(32, false).map(f64::to_bits),
            "32x {a:?}"
        );
        assert_eq!(
            t.rnd64z_f64x2(a).map(f64::to_bits),
            o(64, true).map(f64::to_bits),
            "64z {a:?}"
        );
        assert_eq!(
            t.rnd64x_f64x2(a).map(f64::to_bits),
            o(64, false).map(f64::to_bits),
            "64x {a:?}"
        );
    }
}
