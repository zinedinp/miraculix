use super::Sha3;
use crate::aarch64::ops::macros::mix;

fn v2(i: u64) -> [u64; 2] {
    [mix(i), mix(i + 1)]
}
fn v16(i: u64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&mix(i).to_le_bytes());
    b[8..].copy_from_slice(&mix(i + 1).to_le_bytes());
    b
}

#[test]
fn keccak_helpers_match_their_definitions() {
    let Some(t) = Sha3::detect() else { return };
    for i in 0..200 {
        let (a, b, c) = (v2(3 * i), v2(3 * i + 100), v2(3 * i + 200));
        assert_eq!(
            t.eor3_u64x2(a, b, c),
            core::array::from_fn(|k| a[k] ^ b[k] ^ c[k])
        );
        assert_eq!(
            t.bcax_u64x2(a, b, c),
            core::array::from_fn(|k| a[k] ^ (b[k] & !c[k]))
        );
        assert_eq!(
            t.rax1_u64x2(a, b),
            core::array::from_fn(|k| a[k] ^ b[k].rotate_left(1))
        );
        assert_eq!(
            t.xar_u64x2::<0>(a, b),
            core::array::from_fn(|k| a[k] ^ b[k])
        );
        assert_eq!(
            t.xar_u64x2::<13>(a, b),
            core::array::from_fn(|k| (a[k] ^ b[k]).rotate_right(13))
        );
        assert_eq!(
            t.xar_u64x2::<63>(a, b),
            core::array::from_fn(|k| (a[k] ^ b[k]).rotate_right(63))
        );
        let (x, y, z) = (v16(i), v16(i + 50), v16(i + 90));
        assert_eq!(
            t.eor3_u8x16(x, y, z),
            core::array::from_fn(|k| x[k] ^ y[k] ^ z[k])
        );
        assert_eq!(
            t.bcax_u8x16(x, y, z),
            core::array::from_fn(|k| x[k] ^ (y[k] & !z[k]))
        );
    }
}
