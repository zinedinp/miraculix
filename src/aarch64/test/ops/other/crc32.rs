//! Known answers: the CRC catalogue's check value for `"123456789"` -
//! CRC-32/ISO-HDLC `0xCBF43926`, CRC-32C/ISCSI `0xE3069283` - with the
//! conventional pre- and post-inversion done by the caller.

use super::Crc32;
use crate::aarch64::ops::macros::mix;

const CHECK: &[u8] = b"123456789";

#[test]
fn crc32_check_values() {
    let Some(t) = Crc32::detect() else { return };
    assert_eq!(!t.crc32_slice(!0, CHECK), 0xCBF4_3926);
    assert_eq!(!t.crc32c_slice(!0, CHECK), 0xE306_9283);
    let bytewise = CHECK.iter().fold(!0, |c, &b| t.crc32b(c, b));
    assert_eq!(!bytewise, 0xCBF4_3926);
    let bytewise_c = CHECK.iter().fold(!0, |c, &b| t.crc32cb(c, b));
    assert_eq!(!bytewise_c, 0xE306_9283);
}

#[test]
fn wide_steps_equal_byte_steps() {
    let Some(t) = Crc32::detect() else { return };
    for i in 0..200 {
        let x = mix(i);
        let b = x.to_le_bytes();
        let seed = mix(i + 1) as u32;
        let by = |n: usize, c: bool| {
            b[..n].iter().fold(
                seed,
                |s, &v| if c { t.crc32cb(s, v) } else { t.crc32b(s, v) },
            )
        };
        assert_eq!(t.crc32h(seed, x as u16), by(2, false));
        assert_eq!(t.crc32w(seed, x as u32), by(4, false));
        assert_eq!(t.crc32d(seed, x), by(8, false));
        assert_eq!(t.crc32ch(seed, x as u16), by(2, true));
        assert_eq!(t.crc32cw(seed, x as u32), by(4, true));
        assert_eq!(t.crc32cd(seed, x), by(8, true));
    }
}

#[test]
fn slice_forms_handle_every_tail_length() {
    let Some(t) = Crc32::detect() else { return };
    let data: Vec<u8> = (0..100u64).map(|i| mix(i) as u8).collect();
    for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 100] {
        let d = &data[..len];
        assert_eq!(
            t.crc32_slice(7, d),
            d.iter().fold(7, |s, &v| t.crc32b(s, v)),
            "len={len}"
        );
        assert_eq!(
            t.crc32c_slice(7, d),
            d.iter().fold(7, |s, &v| t.crc32cb(s, v)),
            "len={len}"
        );
    }
}
