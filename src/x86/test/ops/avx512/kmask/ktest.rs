    use super::*;

    #[test]
    fn kortest_mask16_matches_or_all_ones_all_zero() {
        let Some(t) = Avx512f::detect() else { return };
        assert!(t.kortestc_mask16(0xFFFF, 0));
        assert!(!t.kortestc_mask16(0x00FF, 0));
        assert!(t.kortestz_mask16(0, 0));
        assert!(!t.kortestz_mask16(1, 0));
    }

    #[test]
    fn ktest_mask8_matches_and_andn_zero() {
        let Some(t) = Avx512Dq::detect() else { return };
        assert!(t.ktestz_mask8(0b1100, 0b0011));
        assert!(!t.ktestz_mask8(0b1100, 0b1000));
        assert!(t.ktestc_mask8(0b1100, 0b1000));
        assert!(!t.ktestc_mask8(0b0011, 0b1000));
    }

    #[test]
    fn ktest_mask16_matches_and_andn_zero() {
        let Some(t) = Avx512Dq::detect() else { return };
        assert!(t.ktestz_mask16(0b1100, 0b0011));
        assert!(!t.ktestz_mask16(0b1100, 0b1000));
        assert!(t.ktestc_mask16(0b1100, 0b1000));
        assert!(!t.ktestc_mask16(0b0011, 0b1000));
    }

    #[test]
    fn kortest_mask32_matches_or_all_ones_all_zero() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert!(t.kortestc_mask32(0xFFFF_FFFF, 0));
        assert!(!t.kortestc_mask32(0x0000_00FF, 0));
        assert!(t.kortestz_mask32(0, 0));
        assert!(!t.kortestz_mask32(1, 0));
    }

    #[test]
    fn kortest_mask64_matches_or_all_ones_all_zero() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert!(t.kortestc_mask64(!0u64, 0));
        assert!(!t.kortestc_mask64(0xFF, 0));
        assert!(t.kortestz_mask64(0, 0));
        assert!(!t.kortestz_mask64(1, 0));
    }

    #[test]
    fn ktest_mask32_matches_and_andn_zero() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert!(t.ktestz_mask32(0b1100, 0b0011));
        assert!(!t.ktestz_mask32(0b1100, 0b1000));
        assert!(t.ktestc_mask32(0b1100, 0b1000));
        assert!(!t.ktestc_mask32(0b0011, 0b1000));
    }

    #[test]
    fn ktest_mask64_matches_and_andn_zero() {
        let Some(t) = Avx512Bw::detect() else { return };
        assert!(t.ktestz_mask64(0b1100, 0b0011));
        assert!(!t.ktestz_mask64(0b1100, 0b1000));
        assert!(t.ktestc_mask64(0b1100, 0b1000));
        assert!(!t.ktestc_mask64(0b0011, 0b1000));
    }
