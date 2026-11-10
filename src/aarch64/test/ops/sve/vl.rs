//! VL probe and VL-token tests. Under `scripts/qemu-test.sh` the CPU model
//! arrives in `QEMU_CPU`, and each row pins the VL it must report - otherwise
//! the `SveVl256`/`SveVl512` tests of the generated file would pass vacuously.

use super::super::sve::Sve;
use super::{SveVl128, SveVl256, SveVl512};

/// VL QEMU user mode gives each model (observed with QEMU 11.1): the widest
/// `sveNNN=on` length for `max` rows, else the model's own, capped at QEMU's
/// default of 512 bits.
fn expected_vl_bits(cpu: &str) -> Option<usize> {
    let widest_on = cpu
        .split(',')
        .filter_map(|p| {
            p.strip_prefix("sve")?
                .strip_suffix("=on")?
                .parse::<usize>()
                .ok()
        })
        .max();
    match cpu.split(',').next()? {
        "max" => Some(widest_on.unwrap_or(512)),
        "a64fx" => Some(512),
        "neoverse-v1" => Some(256),
        "neoverse-n2" => Some(128),
        _ => None,
    }
}

#[test]
fn vector_bits_is_a_legal_sve_length() {
    let Some(t) = Sve::detect() else { return };
    let bits = t.vector_bits();
    assert!((128..=2048).contains(&bits) && bits % 128 == 0, "VL {bits}");
    assert_eq!(bits, t.vector_bits(), "cached VL must be stable");
}

#[test]
fn vector_bits_matches_the_qemu_row() {
    let Ok(cpu) = std::env::var("QEMU_CPU") else {
        return;
    };
    let Some(want) = expected_vl_bits(&cpu) else {
        return;
    };
    let t = Sve::detect().expect("every QEMU row with an expected VL has SVE");
    assert_eq!(t.vector_bits(), want, "QEMU_CPU={cpu}");
}

#[test]
fn vl_tokens_agree_with_vector_bits() {
    let Some(t) = Sve::detect() else {
        assert!(
            SveVl128::detect().is_none()
                && SveVl256::detect().is_none()
                && SveVl512::detect().is_none()
        );
        return;
    };
    let bits = t.vector_bits();
    assert!(SveVl128::detect().is_some(), "every SVE VL is at least 128");
    assert_eq!(SveVl256::detect().is_some(), bits >= 256);
    assert_eq!(SveVl512::detect().is_some(), bits >= 512);
}

/// The fixed-width forms loop, so a VL token's methods stay exact even when
/// called through a token that promises more VL than a later `prctl` leaves.
/// Here: the 512-bit form must be right on any VL the process actually has.
#[test]
fn fixed_width_forms_are_exact_on_the_real_vl() {
    let Some(t) = SveVl128::detect() else { return };
    let a: [i32; 4] = [i32::MAX, -1, 7, i32::MIN];
    assert_eq!(t.add_i32x4(a, [1, 1, 1, -1]), [i32::MIN, 0, 8, i32::MAX]);
    let _ = t.sve();
    if let Some(t) = SveVl512::detect() {
        let x: [u8; 64] = core::array::from_fn(|i| i as u8);
        let y = t.add_u8x64(x, x);
        assert!(y.iter().enumerate().all(|(i, &v)| v == (2 * i) as u8));
    }
}
