    use super::*;

    #[test]
    fn kshift_mask16_matches_shl_shr() {
        let Some(t) = Avx512f::detect() else { return };
        let a: u16 = 0b1010_1010_1010_1010;
        assert_eq!(t.kshiftli_mask16::<3>(a), a << 3);
        assert_eq!(t.kshiftri_mask16::<3>(a), a >> 3);
        assert_eq!(t.kshiftli_mask16::<16>(a), 0);
        assert_eq!(t.kshiftri_mask16::<16>(a), 0);
    }

    #[test]
    fn kshift_mask8_matches_shl_shr() {
        let Some(t) = Avx512Dq::detect() else { return };
        let a: u8 = 0b1010_1010;
        assert_eq!(t.kshiftli_mask8::<3>(a), a << 3);
        assert_eq!(t.kshiftri_mask8::<3>(a), a >> 3);
        assert_eq!(t.kshiftli_mask8::<8>(a), 0);
        assert_eq!(t.kshiftri_mask8::<8>(a), 0);
    }

    #[test]
    fn kshift_mask32_matches_shl_shr() {
        let Some(t) = Avx512Bw::detect() else { return };
        let a: u32 = 0xAAAA_AAAA;
        assert_eq!(t.kshiftli_mask32::<5>(a), a << 5);
        assert_eq!(t.kshiftri_mask32::<5>(a), a >> 5);
    }

    #[test]
    fn kshift_mask64_matches_shl_shr() {
        let Some(t) = Avx512Bw::detect() else { return };
        let a: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        assert_eq!(t.kshiftli_mask64::<5>(a), a << 5);
        assert_eq!(t.kshiftri_mask64::<5>(a), a >> 5);
    }
