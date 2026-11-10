use super::I8mm;
use crate::aarch64::ops::macros::TestElem;

/// `acc[2i+j] += Σₖ a[8i+k]·b[8j+k]`, in i64 so the oracle cannot overflow.
fn mmla(acc: [i64; 4], a: [i64; 16], b: [i64; 16]) -> [i64; 4] {
    core::array::from_fn(|n| {
        let (i, j) = (n / 2, n % 2);
        (0..8).fold(acc[n], |s, k| s + a[8 * i + k] * b[8 * j + k])
    })
}

fn dot4(acc: [i64; 4], a: [i64; 16], b: [i64; 16]) -> [i64; 4] {
    core::array::from_fn(|l| (0..4).fold(acc[l], |s, k| s + a[4 * l + k] * b[4 * l + k]))
}

macro_rules! i8mm_test {
    ($test:ident, $method:ident, $Acc:ty, $A:ty, $B:ty, $oracle:ident) => {
        #[test]
        fn $test() {
            let Some(t) = I8mm::detect() else { return };
            for r in 0..16usize {
                // The i64 oracle is exact; cast back to the accumulator type to
                // get the instruction's modulo-2^32 result.
                let acc: [$Acc; 4] = core::array::from_fn(|i| (<i16>::sample(r * 31 + i) as $Acc));
                let a: [$A; 16] = core::array::from_fn(|i| <$A>::sample(r * 97 + i));
                let b: [$B; 16] = core::array::from_fn(|i| <$B>::sample(r * 97 + i + 50));
                let want = $oracle(
                    acc.map(|x| x as i64),
                    a.map(|x| x as i64),
                    b.map(|x| x as i64),
                );
                assert_eq!(
                    t.$method(acc, a, b),
                    want.map(|x| x as $Acc),
                    "{} round {r}",
                    stringify!($method)
                );
            }
        }
    };
}

i8mm_test!(mmla_s32_matches_matrix_product, mmla_s32, i32, i8, i8, mmla);
i8mm_test!(mmla_u32_matches_matrix_product, mmla_u32, u32, u8, u8, mmla);
i8mm_test!(
    mmla_us32_matches_matrix_product,
    mmla_us32,
    i32,
    u8,
    i8,
    mmla
);
i8mm_test!(dot_us32_matches_widened_sum, dot_us32, i32, u8, i8, dot4);

#[test]
fn mmla_s32_layout_is_row_major() {
    let Some(t) = I8mm::detect() else { return };
    // A = rows [1..8], [0..]; Bᵀ rows = [1,0,..], [0,1,0,..]: result picks
    // A[0][0], A[0][1], A[1][0], A[1][1] - pins which operand is which.
    let a: [i8; 16] = core::array::from_fn(|k| {
        if k < 8 {
            k as i8 + 1
        } else {
            10 * (k as i8 - 7)
        }
    });
    let mut b = [0i8; 16];
    b[0] = 1;
    b[9] = 1;
    assert_eq!(t.mmla_s32([0; 4], a, b), [1, 2, 10, 20]);
}
