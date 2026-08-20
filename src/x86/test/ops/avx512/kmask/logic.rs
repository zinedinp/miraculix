    use super::*;

    macro_rules! k_binop_test {
        ($name:ident, $token:ty, $method:ident, $mask:ty, $reference:expr) => {
            #[test]
            fn $name() {
                let Some(t) = <$token>::detect() else { return };
                let width = <$mask>::BITS;
                let cases: [($mask, $mask); 5] = [
                    (0, 0),
                    (!0, !0),
                    (0, !0),
                    (0b0101_0101 as $mask, 0b1010_1010 as $mask),
                    (1, (1 as $mask).rotate_left(width - 1)),
                ];
                let reference: fn($mask, $mask) -> $mask = $reference;
                for (a, b) in cases {
                    assert_eq!(t.$method(a, b), reference(a, b), "a={a:#x} b={b:#x}");
                }
            }
        };
    }

    k_binop_test!(
        kand_mask16_matches_bitand,
        Avx512f,
        kand_mask16,
        u16,
        |a, b| a & b
    );
    k_binop_test!(
        kor_mask16_matches_bitor,
        Avx512f,
        kor_mask16,
        u16,
        |a, b| a | b
    );
    k_binop_test!(
        kxor_mask16_matches_bitxor,
        Avx512f,
        kxor_mask16,
        u16,
        |a, b| a ^ b
    );
    k_binop_test!(
        kxnor_mask16_matches_bitxor_not,
        Avx512f,
        kxnor_mask16,
        u16,
        |a, b| !(a ^ b)
    );
    k_binop_test!(
        kandn_mask16_matches_not_a_and_b,
        Avx512f,
        kandn_mask16,
        u16,
        |a, b| !a & b
    );

    #[test]
    fn knot_mask16_matches_bitnot() {
        let Some(t) = Avx512f::detect() else { return };
        for a in [0u16, !0, 0b0101_0101_0101_0101] {
            assert_eq!(t.knot_mask16(a), !a, "a={a:#x}");
        }
    }

    k_binop_test!(
        kand_mask8_matches_bitand,
        Avx512Dq,
        kand_mask8,
        u8,
        |a, b| a & b
    );
    k_binop_test!(kor_mask8_matches_bitor, Avx512Dq, kor_mask8, u8, |a, b| a
        | b);
    k_binop_test!(
        kxor_mask8_matches_bitxor,
        Avx512Dq,
        kxor_mask8,
        u8,
        |a, b| a ^ b
    );
    k_binop_test!(
        kxnor_mask8_matches_bitxor_not,
        Avx512Dq,
        kxnor_mask8,
        u8,
        |a, b| !(a ^ b)
    );
    k_binop_test!(
        kandn_mask8_matches_not_a_and_b,
        Avx512Dq,
        kandn_mask8,
        u8,
        |a, b| !a & b
    );
    k_binop_test!(
        kadd_mask8_matches_wrapping_add,
        Avx512Dq,
        kadd_mask8,
        u8,
        |a, b| a.wrapping_add(b)
    );
    k_binop_test!(
        kadd_mask16_matches_wrapping_add,
        Avx512Dq,
        kadd_mask16,
        u16,
        |a, b| a.wrapping_add(b)
    );

    #[test]
    fn knot_mask8_matches_bitnot() {
        let Some(t) = Avx512Dq::detect() else { return };
        for a in [0u8, !0, 0b0101_0101] {
            assert_eq!(t.knot_mask8(a), !a, "a={a:#x}");
        }
    }

    k_binop_test!(
        kand_mask32_matches_bitand,
        Avx512Bw,
        kand_mask32,
        u32,
        |a, b| a & b
    );
    k_binop_test!(
        kor_mask32_matches_bitor,
        Avx512Bw,
        kor_mask32,
        u32,
        |a, b| a | b
    );
    k_binop_test!(
        kxor_mask32_matches_bitxor,
        Avx512Bw,
        kxor_mask32,
        u32,
        |a, b| a ^ b
    );
    k_binop_test!(
        kxnor_mask32_matches_bitxor_not,
        Avx512Bw,
        kxnor_mask32,
        u32,
        |a, b| !(a ^ b)
    );
    k_binop_test!(
        kandn_mask32_matches_not_a_and_b,
        Avx512Bw,
        kandn_mask32,
        u32,
        |a, b| !a & b
    );
    k_binop_test!(
        kadd_mask32_matches_wrapping_add,
        Avx512Bw,
        kadd_mask32,
        u32,
        |a, b| a.wrapping_add(b)
    );

    k_binop_test!(
        kand_mask64_matches_bitand,
        Avx512Bw,
        kand_mask64,
        u64,
        |a, b| a & b
    );
    k_binop_test!(
        kor_mask64_matches_bitor,
        Avx512Bw,
        kor_mask64,
        u64,
        |a, b| a | b
    );
    k_binop_test!(
        kxor_mask64_matches_bitxor,
        Avx512Bw,
        kxor_mask64,
        u64,
        |a, b| a ^ b
    );
    k_binop_test!(
        kxnor_mask64_matches_bitxor_not,
        Avx512Bw,
        kxnor_mask64,
        u64,
        |a, b| !(a ^ b)
    );
    k_binop_test!(
        kandn_mask64_matches_not_a_and_b,
        Avx512Bw,
        kandn_mask64,
        u64,
        |a, b| !a & b
    );
    k_binop_test!(
        kadd_mask64_matches_wrapping_add,
        Avx512Bw,
        kadd_mask64,
        u64,
        |a, b| a.wrapping_add(b)
    );

    #[test]
    fn knot_mask32_matches_bitnot() {
        let Some(t) = Avx512Bw::detect() else { return };
        for a in [0u32, !0, 0x5555_5555] {
            assert_eq!(t.knot_mask32(a), !a, "a={a:#x}");
        }
    }

    #[test]
    fn knot_mask64_matches_bitnot() {
        let Some(t) = Avx512Bw::detect() else { return };
        for a in [0u64, !0, 0x5555_5555_5555_5555] {
            assert_eq!(t.knot_mask64(a), !a, "a={a:#x}");
        }
    }
