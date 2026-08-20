    use super::super::super::super::super::{Feature, FeatureSet};
    use super::*;

    macro_rules! movm_roundtrip_test {
        ($name:ident, $token:ty, $movm:ident, $movepi:ident, $mask:ty) => {
            #[test]
            fn $name() {
                let Some(t) = <$token>::detect() else { return };
                let width = <$mask>::BITS;
                let cases: [$mask; 4] = [0, !0, 0b0101_0101, (1 as $mask).rotate_left(width - 1)];
                for k in cases {
                    let vec = t.$movm(k);
                    for (i, &lane) in vec.iter().enumerate() {
                        assert_eq!(lane == 0, (k >> i) & 1 == 0, "lane {i}: k={k:#x}");
                    }
                    assert_eq!(t.$movepi(vec), k, "k={k:#x}");
                }
            }
        };
    }

    movm_roundtrip_test!(
        movm_movepi_epi8_roundtrip,
        Avx512Bw,
        movm_epi8,
        movepi8_mask,
        u64
    );
    movm_roundtrip_test!(
        movm_movepi_epi16_roundtrip,
        Avx512Bw,
        movm_epi16,
        movepi16_mask,
        u32
    );
    movm_roundtrip_test!(
        movm_movepi_epi32_roundtrip,
        Avx512Dq,
        movm_epi32,
        movepi32_mask,
        u16
    );
    movm_roundtrip_test!(
        movm_movepi_epi64_roundtrip,
        Avx512Dq,
        movm_epi64,
        movepi64_mask,
        u8
    );

    #[test]
    fn movepi8_mask_matches_sign_bits() {
        let Some(t) = Avx512Bw::detect() else { return };
        let mut a = [1i8; 64];
        let mut expected: u64 = 0;
        for i in (0..64).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi8_mask(a), expected);
    }

    #[test]
    fn movepi16_mask_matches_sign_bits() {
        let Some(t) = Avx512Bw::detect() else { return };
        let mut a = [1i16; 32];
        let mut expected: u32 = 0;
        for i in (0..32).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi16_mask(a), expected);
    }

    #[test]
    fn movepi32_mask_matches_sign_bits() {
        let Some(t) = Avx512Dq::detect() else { return };
        let mut a = [1i32; 16];
        let mut expected: u16 = 0;
        for i in (0..16).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi32_mask(a), expected);
    }

    #[test]
    fn movepi64_mask_matches_sign_bits() {
        let Some(t) = Avx512Dq::detect() else { return };
        let mut a = [1i64; 8];
        let mut expected: u8 = 0;
        for i in (0..8).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi64_mask(a), expected);
    }

    // 128/256-bit bridge (`Avx512BwVl`/`Avx512DqVl`)
    //
    // `movm_roundtrip_test!` (above) assumes `$mask::BITS == lane count`,
    // true for the `epi8`/`epi16` forms here (16/32/8/16 lanes match
    // `u16`/`u32`/`u8`/`u16` exactly) but not for `epi32_x4`/`epi64_x2`/
    // `epi64_x4`, which use `__mmask8` (the AVX-512 minimum) despite fewer
    // than 8 lanes: those need cases restricted to the significant bits.

    movm_roundtrip_test!(
        movm_movepi_epi8_x16_roundtrip,
        Avx512BwVl,
        movm_epi8_x16,
        movepi8_mask_x16,
        u16
    );
    movm_roundtrip_test!(
        movm_movepi_epi8_x32_roundtrip,
        Avx512BwVl,
        movm_epi8_x32,
        movepi8_mask_x32,
        u32
    );
    movm_roundtrip_test!(
        movm_movepi_epi16_x8_roundtrip,
        Avx512BwVl,
        movm_epi16_x8,
        movepi16_mask_x8,
        u8
    );
    movm_roundtrip_test!(
        movm_movepi_epi16_x16_roundtrip,
        Avx512BwVl,
        movm_epi16_x16,
        movepi16_mask_x16,
        u16
    );
    movm_roundtrip_test!(
        movm_movepi_epi32_x8_roundtrip,
        Avx512DqVl,
        movm_epi32_x8,
        movepi32_mask_x8,
        u8
    );

    macro_rules! movm_roundtrip_narrow_test {
        ($name:ident, $token:ty, $movm:ident, $movepi:ident, $lanes:literal) => {
            #[test]
            fn $name() {
                let Some(t) = <$token>::detect() else { return };
                let bit_mask: u8 = (1u16 << $lanes) as u8 - 1;
                let cases: [u8; 4] = [0, bit_mask, 0b0101 & bit_mask, 1 << ($lanes - 1)];
                for k in cases {
                    let vec = t.$movm(k);
                    for (i, &lane) in vec.iter().enumerate() {
                        assert_eq!(lane == 0, (k >> i) & 1 == 0, "lane {i}: k={k:#x}");
                    }
                    assert_eq!(t.$movepi(vec), k, "k={k:#x}");
                }
            }
        };
    }

    movm_roundtrip_narrow_test!(
        movm_movepi_epi32_x4_roundtrip,
        Avx512DqVl,
        movm_epi32_x4,
        movepi32_mask_x4,
        4
    );
    movm_roundtrip_narrow_test!(
        movm_movepi_epi64_x2_roundtrip,
        Avx512DqVl,
        movm_epi64_x2,
        movepi64_mask_x2,
        2
    );
    movm_roundtrip_narrow_test!(
        movm_movepi_epi64_x4_roundtrip,
        Avx512DqVl,
        movm_epi64_x4,
        movepi64_mask_x4,
        4
    );

    #[test]
    fn movepi8_mask_x16_matches_sign_bits() {
        let Some(t) = Avx512BwVl::detect() else {
            return;
        };
        let mut a = [1i8; 16];
        let mut expected: u16 = 0;
        for i in (0..16).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi8_mask_x16(a), expected);
    }

    #[test]
    fn movepi32_mask_x4_matches_sign_bits() {
        let Some(t) = Avx512DqVl::detect() else {
            return;
        };
        let mut a = [1i32; 4];
        let mut expected: u8 = 0;
        for i in (0..4).step_by(3) {
            a[i] = -1;
            expected |= 1 << i;
        }
        assert_eq!(t.movepi32_mask_x4(a), expected);
    }

    #[test]
    fn detect_avx512bw_vl_matches_features() {
        let fs = FeatureSet::detect();
        let expect = fs.contains(Feature::Avx512bw) && fs.contains(Feature::Avx512vl);
        assert_eq!(Avx512BwVl::detect().is_some(), expect);
    }

    #[test]
    fn detect_avx512dq_vl_matches_features() {
        let fs = FeatureSet::detect();
        let expect = fs.contains(Feature::Avx512dq) && fs.contains(Feature::Avx512vl);
        assert_eq!(Avx512DqVl::detect().is_some(), expect);
    }
