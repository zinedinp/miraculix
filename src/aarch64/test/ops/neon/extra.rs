use super::super::neon::Neon;

fn bytes<const N: usize>(seed: u8) -> [u8; N] {
    core::array::from_fn(|i| seed.wrapping_add(i as u8).wrapping_mul(37) ^ 0x5A)
}

#[test]
fn cvt_f32_f64() {
    let t = Neon::assume_baseline();
    let a = [1.5f32, -2.25, 3.0e10, 0.1];
    assert_eq!(t.cvt_f32x2_to_f64x2([a[0], a[1]]), [1.5, -2.25]);
    assert_eq!(t.cvt_f32x4_to_f64x2(a), [f64::from(a[0]), f64::from(a[1])]);
    assert_eq!(
        t.cvt_high_f32x4_to_f64x2(a),
        [f64::from(a[2]), f64::from(a[3])]
    );
    assert_eq!(t.cvt_f32x4_to_f64x4(a), a.map(f64::from));
    let d = [0.1f64, -3.0, 1e300, 7.7];
    assert_eq!(
        t.cvt_f64x2_to_f32x2([d[0], d[1]]),
        [d[0] as f32, d[1] as f32]
    );
    assert_eq!(
        t.cvt_f64x2_to_f32x4([d[0], d[1]]),
        [d[0] as f32, d[1] as f32, 0.0, 0.0]
    );
    assert_eq!(t.cvt_f64x4_to_f32x4(d), d.map(|x| x as f32));
}

#[test]
fn test_zip_shift_sum() {
    let t = Neon::assume_baseline();
    let (a, b) = (bytes::<16>(1), bytes::<16>(99));
    let want_test: [u8; 16] = core::array::from_fn(|i| if a[i] & b[i] != 0 { 0xFF } else { 0 });
    assert_eq!(t.test_u8x16(a, b), want_test);
    let lo: [u8; 16] = core::array::from_fn(|i| if i % 2 == 0 { a[i / 2] } else { b[i / 2] });
    let hi: [u8; 16] = core::array::from_fn(|i| {
        if i % 2 == 0 {
            a[8 + i / 2]
        } else {
            b[8 + i / 2]
        }
    });
    assert_eq!(t.zip1_u8x16(a, b), lo);
    assert_eq!(t.zip2_u8x16(a, b), hi);
    assert_eq!(t.unpacklo_u8x16(a, b), lo);
    assert_eq!(t.unpackhi_u8x16(a, b), hi);
    let (c, d) = (bytes::<8>(3), bytes::<8>(50));
    assert_eq!(
        t.zip1_u8x8(c, d),
        core::array::from_fn(|i| if i % 2 == 0 { c[i / 2] } else { d[i / 2] })
    );
    assert_eq!(
        t.zip2_u8x8(c, d),
        core::array::from_fn(|i| if i % 2 == 0 {
            c[4 + i / 2]
        } else {
            d[4 + i / 2]
        })
    );
    assert_eq!(t.srli_u8x8::<4>(c), c.map(|x| x >> 4));
    assert_eq!(t.srli_u8x8::<8>(c), [0; 8]);
    assert_eq!(t.sum_u8x8(c), c.iter().fold(0u8, |s, &x| s.wrapping_add(x)));
}

#[test]
fn movemask_matches_scalar() {
    let t = Neon::assume_baseline();
    for seed in 0..64u8 {
        let a: [i8; 16] = bytes::<16>(seed).map(|x| x as i8);
        let want = (0..16).fold(0u16, |m, i| m | (u16::from(a[i] < 0) << i));
        assert_eq!(t.movemask_i8x16(a), want);
    }
    assert_eq!(t.movemask_i8x16([-1; 16]), 0xFFFF);
    assert_eq!(t.movemask_i8x16([0; 16]), 0);
    // Bit `k` of every byte via test + movemask.
    let a = bytes::<16>(7);
    for bit in 0..8 {
        let m = t.test_u8x16(a, [1 << bit; 16]).map(|x| x as i8);
        let want = (0..16).fold(0u16, |m, i| m | (u16::from(a[i] >> bit & 1) << i));
        assert_eq!(t.movemask_i8x16(m), want);
    }
}

#[test]
fn table_lookups() {
    let t = Neon::assume_baseline();
    let table16 = bytes::<16>(11);
    let idx = [0, 15, 16, 255, 3, 3, 8, 200, 1, 2, 17, 5, 9, 14, 13, 128];
    let want: [u8; 16] = idx.map(|i| if i < 16 { table16[i as usize] } else { 0 });
    assert_eq!(t.tbl_u8x16(table16, idx), want);
    let fb = bytes::<16>(200);
    let want_x: [u8; 16] = core::array::from_fn(|k| {
        if idx[k] < 16 {
            table16[idx[k] as usize]
        } else {
            fb[k]
        }
    });
    assert_eq!(t.tbx_u8x16(fb, table16, idx), want_x);

    let table64 = bytes::<64>(5);
    let idx64: [u8; 16] = core::array::from_fn(|i| (i * 9) as u8);
    let want: [u8; 16] = idx64.map(|i| if i < 64 { table64[i as usize] } else { 0 });
    assert_eq!(t.tbl4_u8x64(table64, idx64), want);
    let want_x: [u8; 16] = core::array::from_fn(|k| {
        if idx64[k] < 64 {
            table64[idx64[k] as usize]
        } else {
            fb[k]
        }
    });
    assert_eq!(t.tbx4_u8x64(fb, table64, idx64), want_x);

    let table256: [u8; 256] = core::array::from_fn(|i| (i as u8).wrapping_mul(13) ^ 0xA5);
    for base in 0..16u8 {
        let idx: [u8; 16] =
            core::array::from_fn(|i| base.wrapping_mul(16).wrapping_add((i * 17) as u8));
        let want = idx.map(|i| table256[i as usize]);
        assert_eq!(t.lookup_u8x256(&table256, idx), want, "base {base}");
    }
}
