    use super::*;

    #[test]
    fn mask_between_mask16_sets_low_n_bits() {
        let Some(t) = Avx512f::detect() else { return };
        assert_eq!(t.mask_between_mask16(0), 0b0000_0000_0000_0000);
        assert_eq!(t.mask_between_mask16(1), 0b0000_0000_0000_0001);
        assert_eq!(t.mask_between_mask16(5), 0b0000_0000_0001_1111);
        assert_eq!(t.mask_between_mask16(16), u16::MAX);
        assert_eq!(t.mask_between_mask16(1000), u16::MAX);
    }

    #[test]
    fn mask_between_mask8_sets_low_n_bits() {
        let Some(t) = Avx512Dq::detect() else { return };
        assert_eq!(t.mask_between_mask8(0), 0);
        assert_eq!(t.mask_between_mask8(3), 0b0000_0111);
        assert_eq!(t.mask_between_mask8(8), u8::MAX);
        assert_eq!(t.mask_between_mask8(9), u8::MAX);
    }

    #[test]
    fn mask_between_mask32_sets_low_n_bits() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert_eq!(t.mask_between_mask32(0), 0);
        assert_eq!(t.mask_between_mask32(17), (1u32 << 17) - 1);
        assert_eq!(t.mask_between_mask32(32), u32::MAX);
        assert_eq!(t.mask_between_mask32(33), u32::MAX);
    }

    #[test]
    fn mask_between_mask64_sets_low_n_bits() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert_eq!(t.mask_between_mask64(0), 0);
        assert_eq!(t.mask_between_mask64(40), (1u64 << 40) - 1);
        assert_eq!(t.mask_between_mask64(64), u64::MAX);
        assert_eq!(t.mask_between_mask64(65), u64::MAX);
    }
