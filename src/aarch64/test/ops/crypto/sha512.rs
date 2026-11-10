//! The message-schedule steps have short closed forms (FIPS 180-4 §4.1.3
//! `σ0`/`σ1`) and are checked exactly. The two compression halves have no
//! identity short of a full SHA-512 run, whose 80 64-bit round constants
//! cannot be derived with `f64` precision (they need 64 fractional bits of a
//! cube root) - so `sha512h`/`sha512h2` get the documented smoke test used
//! for aarch32's SHA rounds: deterministic, and sensitive to each operand.

use super::Sha512;
use crate::aarch64::ops::macros::mix;

fn sig0(x: u64) -> u64 {
    x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
}
fn sig1(x: u64) -> u64 {
    x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
}
fn v2(i: u64) -> [u64; 2] {
    [mix(i), mix(i + 1)]
}

#[test]
fn sha512su0_is_w_plus_sigma0_of_the_next_word() {
    let Some(t) = Sha512::detect() else { return };
    for i in 0..200 {
        let (w01, w23) = (v2(2 * i), v2(2 * i + 500));
        let want = [
            w01[0].wrapping_add(sig0(w01[1])),
            w01[1].wrapping_add(sig0(w23[0])),
        ];
        assert_eq!(t.sha512su0(w01, w23), want);
    }
}

#[test]
fn sha512su1_adds_sigma1_and_w9_10() {
    let Some(t) = Sha512::detect() else { return };
    for i in 0..200 {
        let (s, w14, w9) = (v2(3 * i), v2(3 * i + 500), v2(3 * i + 900));
        let want: [u64; 2] =
            core::array::from_fn(|k| s[k].wrapping_add(sig1(w14[k])).wrapping_add(w9[k]));
        assert_eq!(t.sha512su1(s, w14, w9), want);
    }
}

#[test]
fn sha512h_and_h2_are_deterministic_and_input_sensitive() {
    let Some(t) = Sha512::detect() else { return };
    for f in [Sha512::sha512h, Sha512::sha512h2] {
        let (a, b, c) = (v2(1), v2(2), v2(3));
        let r = f(t, a, b, c);
        assert_eq!(r, f(t, a, b, c));
        assert_ne!(r, f(t, v2(9), b, c));
        assert_ne!(r, f(t, a, v2(9), c));
        assert_ne!(r, f(t, a, b, v2(9)));
    }
}
