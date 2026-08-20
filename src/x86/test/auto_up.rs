use super::*;
use super::super::ops::{avx::avx::Avx, sse::sse41::Sse41};

macro_rules! matches_scalar_across_lengths_test {
	($test_name:ident, $fn_name:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				$fn_name(&a, &b, &mut out);

				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

macro_rules! panics_on_length_mismatch_test {
	($test_name:ident, $fn_name:ident, $Elem:ty) => {
		#[test]
		#[should_panic]
		fn $test_name() {
			let a: [$Elem; 2] = [1 as $Elem, 2 as $Elem];
			let b: [$Elem; 3] = [1 as $Elem, 2 as $Elem, 3 as $Elem];
			let mut out: [$Elem; 2] = [Default::default(); 2];
			$fn_name(&a, &b, &mut out);
		}
	};
}

matches_scalar_across_lengths_test!(add_f32_matches_scalar_across_lengths, add_f32, |x, y| x + y, f32);
matches_scalar_across_lengths_test!(sub_f32_matches_scalar_across_lengths, sub_f32, |x, y| x - y, f32);
matches_scalar_across_lengths_test!(mul_f32_matches_scalar_across_lengths, mul_f32, |x, y| x * y, f32);
matches_scalar_across_lengths_test!(div_f32_matches_scalar_across_lengths, div_f32, |x, y| x / y, f32);
matches_scalar_across_lengths_test!(min_f32_matches_scalar_across_lengths, min_f32, |x, y| x.min(y), f32);
matches_scalar_across_lengths_test!(max_f32_matches_scalar_across_lengths, max_f32, |x, y| x.max(y), f32);
panics_on_length_mismatch_test!(add_f32_panics_on_length_mismatch, add_f32, f32);
panics_on_length_mismatch_test!(sub_f32_panics_on_length_mismatch, sub_f32, f32);
panics_on_length_mismatch_test!(mul_f32_panics_on_length_mismatch, mul_f32, f32);
panics_on_length_mismatch_test!(div_f32_panics_on_length_mismatch, div_f32, f32);
panics_on_length_mismatch_test!(min_f32_panics_on_length_mismatch, min_f32, f32);
panics_on_length_mismatch_test!(max_f32_panics_on_length_mismatch, max_f32, f32);

matches_scalar_across_lengths_test!(add_f64_matches_scalar_across_lengths, add_f64, |x, y| x + y, f64);
matches_scalar_across_lengths_test!(sub_f64_matches_scalar_across_lengths, sub_f64, |x, y| x - y, f64);
matches_scalar_across_lengths_test!(mul_f64_matches_scalar_across_lengths, mul_f64, |x, y| x * y, f64);
matches_scalar_across_lengths_test!(div_f64_matches_scalar_across_lengths, div_f64, |x, y| x / y, f64);
matches_scalar_across_lengths_test!(min_f64_matches_scalar_across_lengths, min_f64, |x, y| x.min(y), f64);
matches_scalar_across_lengths_test!(max_f64_matches_scalar_across_lengths, max_f64, |x, y| x.max(y), f64);
panics_on_length_mismatch_test!(add_f64_panics_on_length_mismatch, add_f64, f64);
panics_on_length_mismatch_test!(sub_f64_panics_on_length_mismatch, sub_f64, f64);
panics_on_length_mismatch_test!(mul_f64_panics_on_length_mismatch, mul_f64, f64);
panics_on_length_mismatch_test!(div_f64_panics_on_length_mismatch, div_f64, f64);
panics_on_length_mismatch_test!(min_f64_panics_on_length_mismatch, min_f64, f64);
panics_on_length_mismatch_test!(max_f64_panics_on_length_mismatch, max_f64, f64);

/// Bit-exact variant: arbitrary floats OR/XORed bitwise routinely land in the NaN exponent
/// range, and NaN != NaN under `PartialEq` even when the bit patterns are identical.
macro_rules! bitop_matches_scalar_across_lengths_test {
	($test_name:ident, $fn_name:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				$fn_name(&a, &b, &mut out);

				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
				let out_bits: Vec<_> = out.iter().map(|x| x.to_bits()).collect();
				let expect_bits: Vec<_> = expect.iter().map(|x| x.to_bits()).collect();
				assert_eq!(out_bits, expect_bits, "len={len}");
			}
		}
	};
}

bitop_matches_scalar_across_lengths_test!(
	and_f32_matches_scalar_across_lengths, and_f32, |x: f32, y: f32| f32::from_bits(x.to_bits() & y.to_bits()), f32
);
bitop_matches_scalar_across_lengths_test!(
	or_f32_matches_scalar_across_lengths, or_f32, |x: f32, y: f32| f32::from_bits(x.to_bits() | y.to_bits()), f32
);
bitop_matches_scalar_across_lengths_test!(
	xor_f32_matches_scalar_across_lengths, xor_f32, |x: f32, y: f32| f32::from_bits(x.to_bits() ^ y.to_bits()), f32
);
bitop_matches_scalar_across_lengths_test!(
	andnot_f32_matches_scalar_across_lengths, andnot_f32, |x: f32, y: f32| f32::from_bits(!x.to_bits() & y.to_bits()), f32
);
bitop_matches_scalar_across_lengths_test!(
	and_f64_matches_scalar_across_lengths, and_f64, |x: f64, y: f64| f64::from_bits(x.to_bits() & y.to_bits()), f64
);
bitop_matches_scalar_across_lengths_test!(
	or_f64_matches_scalar_across_lengths, or_f64, |x: f64, y: f64| f64::from_bits(x.to_bits() | y.to_bits()), f64
);
bitop_matches_scalar_across_lengths_test!(
	xor_f64_matches_scalar_across_lengths, xor_f64, |x: f64, y: f64| f64::from_bits(x.to_bits() ^ y.to_bits()), f64
);
bitop_matches_scalar_across_lengths_test!(
	andnot_f64_matches_scalar_across_lengths, andnot_f64, |x: f64, y: f64| f64::from_bits(!x.to_bits() & y.to_bits()), f64
);
bitop_matches_scalar_across_lengths_test!(
	cmpeq_f32_matches_scalar_across_lengths, cmpeq_f32,
	|x, y| if x == y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
bitop_matches_scalar_across_lengths_test!(
	cmpgt_f32_matches_scalar_across_lengths, cmpgt_f32,
	|x, y| if x > y { f32::from_bits(!0) } else { f32::from_bits(0) }, f32
);
bitop_matches_scalar_across_lengths_test!(
	cmple_f64_matches_scalar_across_lengths, cmple_f64,
	|x, y| if x <= y { f64::from_bits(!0) } else { f64::from_bits(0) }, f64
);

matches_scalar_across_lengths_test!(and_i64_matches_scalar_across_lengths, and_i64, |x, y| x & y, i64);
matches_scalar_across_lengths_test!(xor_u64_matches_scalar_across_lengths, xor_u64, |x, y| x ^ y, u64);

#[test]
fn abs_i32_wrapping_abs_of_min() {
	let a = [i32::MIN, -5, 0, 7];
	let mut out = [0i32; 4];
	abs_i32(&a, &mut out);
	assert_eq!(out, [i32::MIN, 5, 0, 7]);
}

#[test]
fn clamp_i32_bounds_lanes() {
	let a = [-10i32, 0, 5, 100];
	let lo = [0i32, 0, 0, 0];
	let hi = [10i32, 10, 10, 10];
	let mut out = [0i32; 4];
	clamp_i32(&a, &lo, &hi, &mut out);
	assert_eq!(out, [0, 0, 5, 10]);
}

#[test]
fn clamp_f32_bounds_lanes() {
	let a = [-1.0f32, 0.5, 2.0, 100.0];
	let lo = [0.0f32; 4];
	let hi = [1.0f32; 4];
	let mut out = [0.0f32; 4];
	clamp_f32(&a, &lo, &hi, &mut out);
	assert_eq!(out, [0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn sllv_i64_shifts_by_the_count_vector() {
	let a = [1i64, 1, 1, 1];
	let count = [0i64, 1, 2, 64];
	let mut out = [0i64; 4];
	sllv_i64(&a, &count, &mut out);
	assert_eq!(out, [1, 2, 4, 0]);
}

#[test]
fn srav_i64_sign_fills_past_bit_width() {
	let a = [-8i64, -8, -8, -8];
	let count = [0i64, 1, 2, 64];
	let mut out = [0i64; 4];
	srav_i64(&a, &count, &mut out);
	assert_eq!(out, [-8, -4, -2, -1]);
}
panics_on_length_mismatch_test!(and_f32_panics_on_length_mismatch, and_f32, f32);
panics_on_length_mismatch_test!(and_f64_panics_on_length_mismatch, and_f64, f64);

#[test]
fn mullo_i64_wraps_on_overflow() {
	let a = [i64::MAX, 3];
	let b = [2, 4];
	let mut out = [0i64; 2];
	mullo_i64(&a, &b, &mut out);
	assert_eq!(out, [i64::MAX.wrapping_mul(2), 12]);
}

#[test]
fn mullo_u64_wraps_on_overflow() {
	let a = [u64::MAX, 3];
	let b = [2, 4];
	let mut out = [0u64; 2];
	mullo_u64(&a, &b, &mut out);
	assert_eq!(out, [u64::MAX.wrapping_mul(2), 12]);
}

#[test]
fn mullo_i64_matches_scalar_for_cross_term_carry_values() {
	let a = [0xFFFF_FFFF_FFFF_FFFFu64 as i64, 0x1_0000_0002];
	let b = [0xFFFF_FFFF_FFFF_FFFFu64 as i64, 0x1_0000_0003];
	let mut out = [0i64; 2];
	mullo_i64(&a, &b, &mut out);
	let expect: [i64; 2] = core::array::from_fn(|i| a[i].wrapping_mul(b[i]));
	assert_eq!(out, expect);
}

matches_scalar_across_lengths_test!(mullo_i64_matches_scalar_across_lengths, mullo_i64, |x: i64, y: i64| x.wrapping_mul(y), i64);
matches_scalar_across_lengths_test!(mullo_u64_matches_scalar_across_lengths, mullo_u64, |x: u64, y: u64| x.wrapping_mul(y), u64);
panics_on_length_mismatch_test!(mullo_i64_panics_on_length_mismatch, mullo_i64, i64);
panics_on_length_mismatch_test!(mullo_u64_panics_on_length_mismatch, mullo_u64, u64);

matches_scalar_across_lengths_test!(mul_i8_matches_scalar_across_lengths, mul_i8, |x: i8, y: i8| x.wrapping_mul(y), i8);
matches_scalar_across_lengths_test!(mul_u8_matches_scalar_across_lengths, mul_u8, |x: u8, y: u8| x.wrapping_mul(y), u8);
panics_on_length_mismatch_test!(mul_i8_panics_on_length_mismatch, mul_i8, i8);
panics_on_length_mismatch_test!(mul_u8_panics_on_length_mismatch, mul_u8, u8);

#[test]
fn mul_i8_wraps_on_overflow() {
	let a = [i8::MIN, 3];
	let b = [i8::MIN, 4];
	let mut out = [0i8; 2];
	mul_i8(&a, &b, &mut out);
	assert_eq!(out, [i8::MIN.wrapping_mul(i8::MIN), 12]);
}

#[test]
fn shl_i8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17, 32, 33, 100] {
		let a: Vec<i8> = (0..len).map(|i| (i as i8).wrapping_mul(3).wrapping_add(1)).collect();
		let mut out = vec![0i8; len];
		shl_i8::<3>(&a, &mut out);
		let expect: Vec<i8> = a.iter().map(|&x| x.wrapping_shl(3)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shl_u8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17, 32, 33, 100] {
		let a: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(3).wrapping_add(1)).collect();
		let mut out = vec![0u8; len];
		shl_u8::<3>(&a, &mut out);
		let expect: Vec<u8> = a.iter().map(|&x| x.wrapping_shl(3)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shr_i8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17, 32, 33, 100] {
		let a: Vec<i8> = (0..len).map(|i| (i as i8).wrapping_mul(3).wrapping_add(1)).collect();
		let mut out = vec![0i8; len];
		shr_i8::<2>(&a, &mut out);
		let expect: Vec<i8> = a.iter().map(|&x| ((x as u8).wrapping_shr(2)) as i8).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shr_u8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17, 32, 33, 100] {
		let a: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(3).wrapping_add(1)).collect();
		let mut out = vec![0u8; len];
		shr_u8::<2>(&a, &mut out);
		let expect: Vec<u8> = a.iter().map(|&x| x.wrapping_shr(2)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn sra_i8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17, 32, 33, 100] {
		let a: Vec<i8> = (0..len).map(|i| (i as i8).wrapping_mul(5).wrapping_sub(20)).collect();
		let mut out = vec![0i8; len];
		sra_i8::<1>(&a, &mut out);
		let expect: Vec<i8> = a.iter().map(|&x| x.wrapping_shr(1)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn sra_i8_of_i8_min_matches_scalar() {
	let a = [i8::MIN; 4];
	let mut out = [0i8; 4];
	sra_i8::<3>(&a, &mut out);
	assert_eq!(out, [i8::MIN.wrapping_shr(3); 4]);
}

#[test]
fn abs_i64_wrapping_abs_of_min() {
	let a = [i64::MIN, -5, 0, 7];
	let mut out = [0i64; 4];
	abs_i64(&a, &mut out);
	assert_eq!(out, [i64::MIN, 5, 0, 7]);
}

#[test]
fn abs_i64_matches_scalar_across_lengths() {
	for len in [0usize, 1, 7, 8, 9, 16, 17, 100] {
		let a: Vec<i64> = (0..len).map(|i| (i as i64 - len as i64 / 2) * 0x1_0000_0007).collect();
		let mut out = vec![0i64; len];
		abs_i64(&a, &mut out);
		let expect: Vec<i64> = a.iter().map(|&x| x.wrapping_abs()).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

matches_scalar_across_lengths_test!(add_i32_matches_scalar_across_lengths, add_i32, |x: i32, y: i32| x.wrapping_add(y), i32);
matches_scalar_across_lengths_test!(sub_i32_matches_scalar_across_lengths, sub_i32, |x: i32, y: i32| x.wrapping_sub(y), i32);
matches_scalar_across_lengths_test!(mul_i32_matches_scalar_across_lengths, mul_i32, |x: i32, y: i32| x.wrapping_mul(y), i32);
matches_scalar_across_lengths_test!(div_i32_matches_scalar_across_lengths, div_i32, |x: i32, y: i32| x / y, i32);
matches_scalar_across_lengths_test!(min_i32_matches_scalar_across_lengths, min_i32, |x, y| x.min(y), i32);
matches_scalar_across_lengths_test!(max_i32_matches_scalar_across_lengths, max_i32, |x, y| x.max(y), i32);
panics_on_length_mismatch_test!(add_i32_panics_on_length_mismatch, add_i32, i32);
panics_on_length_mismatch_test!(sub_i32_panics_on_length_mismatch, sub_i32, i32);
panics_on_length_mismatch_test!(mul_i32_panics_on_length_mismatch, mul_i32, i32);
panics_on_length_mismatch_test!(div_i32_panics_on_length_mismatch, div_i32, i32);
panics_on_length_mismatch_test!(min_i32_panics_on_length_mismatch, min_i32, i32);
panics_on_length_mismatch_test!(max_i32_panics_on_length_mismatch, max_i32, i32);

matches_scalar_across_lengths_test!(add_u32_matches_scalar_across_lengths, add_u32, |x: u32, y: u32| x.wrapping_add(y), u32);
matches_scalar_across_lengths_test!(sub_u32_matches_scalar_across_lengths, sub_u32, |x: u32, y: u32| x.wrapping_sub(y), u32);
matches_scalar_across_lengths_test!(mul_u32_matches_scalar_across_lengths, mul_u32, |x: u32, y: u32| x.wrapping_mul(y), u32);
matches_scalar_across_lengths_test!(div_u32_matches_scalar_across_lengths, div_u32, |x: u32, y: u32| x / y, u32);
matches_scalar_across_lengths_test!(min_u32_matches_scalar_across_lengths, min_u32, |x, y| x.min(y), u32);
matches_scalar_across_lengths_test!(max_u32_matches_scalar_across_lengths, max_u32, |x, y| x.max(y), u32);
panics_on_length_mismatch_test!(add_u32_panics_on_length_mismatch, add_u32, u32);
panics_on_length_mismatch_test!(sub_u32_panics_on_length_mismatch, sub_u32, u32);
panics_on_length_mismatch_test!(mul_u32_panics_on_length_mismatch, mul_u32, u32);
panics_on_length_mismatch_test!(div_u32_panics_on_length_mismatch, div_u32, u32);
panics_on_length_mismatch_test!(min_u32_panics_on_length_mismatch, min_u32, u32);
panics_on_length_mismatch_test!(max_u32_panics_on_length_mismatch, max_u32, u32);

macro_rules! branch_matches_scalar_test {
	($test_name:ident, $Tier:ty, $from:expr, $len:literal, $Elem:ty; $(($scalar:expr, $slice_fn:path)),+ $(,)?) => {
		#[test]
		fn $test_name() {
			let level = super::super::detect_level();
			let Some(t) = $from(level) else { return };
			let a: Vec<$Elem> = (0..$len).map(|i| i as $Elem + 1 as $Elem).collect();
			let b: Vec<$Elem> = (0..$len).map(|i| ($len - i) as $Elem + 1 as $Elem).collect();

			type ScalarFn = fn($Elem, $Elem) -> $Elem;
			type SliceFn = fn($Tier, &[$Elem], &[$Elem], &mut [$Elem]);
			let ops: &[(ScalarFn, SliceFn)] = &[$(($scalar, $slice_fn)),+];

			for (scalar, slice_fn) in ops {
				let mut out = vec![Default::default(); $len];
				slice_fn(t, &a, &b, &mut out);
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
				assert_eq!(out, expect);
			}
		}
	};
}

branch_matches_scalar_test!(
	avx_branch_matches_scalar_for_all_f32_ops, Avx, Avx::from_level, 9, f32;
	(|x, y| x + y, Avx::add_f32_slice),
	(|x, y| x - y, Avx::sub_f32_slice),
	(|x, y| x * y, Avx::mul_f32_slice),
	(|x, y| x / y, Avx::div_f32_slice),
	(|x, y| x.min(y), Avx::min_f32_slice),
	(|x, y| x.max(y), Avx::max_f32_slice),
);
branch_matches_scalar_test!(
	avx512_branch_matches_scalar_for_all_f32_ops, Avx512f, Avx512f::from_level, 17, f32;
	(|x, y| x + y, Avx512f::add_f32_slice),
	(|x, y| x - y, Avx512f::sub_f32_slice),
	(|x, y| x * y, Avx512f::mul_f32_slice),
	(|x, y| x / y, Avx512f::div_f32_slice),
	(|x, y| x.min(y), Avx512f::min_f32_slice),
	(|x, y| x.max(y), Avx512f::max_f32_slice),
);

branch_matches_scalar_test!(
	avx_branch_matches_scalar_for_all_f64_ops, Avx, Avx::from_level, 9, f64;
	(|x, y| x + y, Avx::add_f64_slice),
	(|x, y| x - y, Avx::sub_f64_slice),
	(|x, y| x * y, Avx::mul_f64_slice),
	(|x, y| x / y, Avx::div_f64_slice),
	(|x, y| x.min(y), Avx::min_f64_slice),
	(|x, y| x.max(y), Avx::max_f64_slice),
);
branch_matches_scalar_test!(
	avx512_branch_matches_scalar_for_all_f64_ops, Avx512f, Avx512f::from_level, 17, f64;
	(|x, y| x + y, Avx512f::add_f64_slice),
	(|x, y| x - y, Avx512f::sub_f64_slice),
	(|x, y| x * y, Avx512f::mul_f64_slice),
	(|x, y| x / y, Avx512f::div_f64_slice),
	(|x, y| x.min(y), Avx512f::min_f64_slice),
	(|x, y| x.max(y), Avx512f::max_f64_slice),
);

branch_matches_scalar_test!(
	avx2_branch_matches_scalar_for_all_i32_ops, Avx2, Avx2::from_level, 9, i32;
	(|x: i32, y: i32| x.wrapping_add(y), Avx2::add_i32_slice),
	(|x: i32, y: i32| x.wrapping_sub(y), Avx2::sub_i32_slice),
	(|x: i32, y: i32| x.wrapping_mul(y), Avx2::mul_i32_slice),
	(|x: i32, y: i32| x / y, Avx2::div_i32_slice),
	(|x, y| x.min(y), Avx2::min_i32_slice),
	(|x, y| x.max(y), Avx2::max_i32_slice),
);
branch_matches_scalar_test!(
	avx512_branch_matches_scalar_for_all_i32_ops, Avx512f, Avx512f::from_level, 17, i32;
	(|x: i32, y: i32| x.wrapping_add(y), Avx512f::add_i32_slice),
	(|x: i32, y: i32| x.wrapping_sub(y), Avx512f::sub_i32_slice),
	(|x: i32, y: i32| x.wrapping_mul(y), Avx512f::mul_i32_slice),
	(|x: i32, y: i32| x / y, Avx512f::div_i32_slice),
	(|x, y| x.min(y), Avx512f::min_i32_slice),
	(|x, y| x.max(y), Avx512f::max_i32_slice),
);

branch_matches_scalar_test!(
	avx2_branch_matches_scalar_for_all_u32_ops, Avx2, Avx2::from_level, 9, u32;
	(|x: u32, y: u32| x.wrapping_add(y), Avx2::add_u32_slice),
	(|x: u32, y: u32| x.wrapping_sub(y), Avx2::sub_u32_slice),
	(|x: u32, y: u32| x.wrapping_mul(y), Avx2::mul_u32_slice),
	(|x: u32, y: u32| x / y, Avx2::div_u32_slice),
	(|x, y| x.min(y), Avx2::min_u32_slice),
	(|x, y| x.max(y), Avx2::max_u32_slice),
);
branch_matches_scalar_test!(
	avx512_branch_matches_scalar_for_all_u32_ops, Avx512f, Avx512f::from_level, 17, u32;
	(|x: u32, y: u32| x.wrapping_add(y), Avx512f::add_u32_slice),
	(|x: u32, y: u32| x.wrapping_sub(y), Avx512f::sub_u32_slice),
	(|x: u32, y: u32| x.wrapping_mul(y), Avx512f::mul_u32_slice),
	(|x: u32, y: u32| x / y, Avx512f::div_u32_slice),
	(|x, y| x.min(y), Avx512f::min_u32_slice),
	(|x, y| x.max(y), Avx512f::max_u32_slice),
);

/// Directly exercises the SSE4.1 bottom rung for i32/u32 mul (min/max now
/// baseline SSE2-composed; mul still needs the probe).
#[test]
fn sse41_branch_matches_scalar_for_i32_and_u32_mul() {
	let Some(t) = Sse41::detect() else { return };
	let a_i: [i32; 4] = [1, 2, 3, 4];
	let b_i: [i32; 4] = [4, 3, 2, 1];
	let mut out_i = [0i32; 4];
	t.mul_i32_slice(&a_i, &b_i, &mut out_i);
	assert_eq!(out_i, [4, 6, 6, 4]);

	let a_u: [u32; 4] = [1, 2, 3, 4];
	let b_u: [u32; 4] = [4, 3, 2, 1];
	let mut out_u = [0u32; 4];
	t.mul_u32_slice(&a_u, &b_u, &mut out_u);
	assert_eq!(out_u, [4, 6, 6, 4]);
}

matches_scalar_across_lengths_test!(and_i32_matches_scalar_across_lengths, and_i32, |x, y| x & y, i32);
matches_scalar_across_lengths_test!(xor_u32_matches_scalar_across_lengths, xor_u32, |x, y| x ^ y, u32);
matches_scalar_across_lengths_test!(
	cmpeq_i32_matches_scalar_across_lengths, cmpeq_i32,
	|x, y| if x == y { -1 } else { 0 }, i32
);
matches_scalar_across_lengths_test!(
	cmpgt_i32_matches_scalar_across_lengths, cmpgt_i32,
	|x, y| if x > y { -1 } else { 0 }, i32
);
matches_scalar_across_lengths_test!(
	cmplt_i32_matches_scalar_across_lengths, cmplt_i32,
	|x, y| if x < y { -1 } else { 0 }, i32
);
matches_scalar_across_lengths_test!(
	cmple_i32_matches_scalar_across_lengths, cmple_i32,
	|x, y| if x <= y { -1 } else { 0 }, i32
);
matches_scalar_across_lengths_test!(
	cmpge_i32_matches_scalar_across_lengths, cmpge_i32,
	|x, y| if x >= y { -1 } else { 0 }, i32
);
matches_scalar_across_lengths_test!(
	cmplt_u32_matches_scalar_across_lengths, cmplt_u32,
	|x, y| if x < y { !0 } else { 0 }, u32
);
panics_on_length_mismatch_test!(and_i32_panics_on_length_mismatch, and_i32, i32);

matches_scalar_across_lengths_test!(
	cmpeq_i64_matches_scalar_across_lengths, cmpeq_i64,
	|x, y| if x == y { -1 } else { 0 }, i64
);
matches_scalar_across_lengths_test!(
	cmpgt_i64_matches_scalar_across_lengths, cmpgt_i64,
	|x, y| if x > y { -1 } else { 0 }, i64
);
matches_scalar_across_lengths_test!(
	cmplt_i64_matches_scalar_across_lengths, cmplt_i64,
	|x, y| if x < y { -1 } else { 0 }, i64
);
matches_scalar_across_lengths_test!(
	cmple_u64_matches_scalar_across_lengths, cmple_u64,
	|x, y| if x <= y { !0 } else { 0 }, u64
);
matches_scalar_across_lengths_test!(
	cmpge_i64_matches_scalar_across_lengths, cmpge_i64,
	|x, y| if x >= y { -1 } else { 0 }, i64
);

#[test]
fn select_i32_picks_b_where_mask_set() {
	let a = [1, 2, 3, 4];
	let b = [10, 20, 30, 40];
	let mask = [-1, 0, -1, 0];
	let mut out = [0i32; 4];
	select_i32(&a, &b, &mask, &mut out);
	assert_eq!(out, [10, 2, 30, 4]);
}

#[test]
fn select_f32_uses_sign_bit_not_zero_test() {
	let a = [1.0f32; 4];
	let b = [2.0f32; 4];
	let mask = [-0.0f32; 4];
	let mut out = [0f32; 4];
	select_f32(&a, &b, &mask, &mut out);
	assert_eq!(out, [2.0; 4]);
}

#[test]
fn select_i64_picks_b_where_mask_set() {
	let a = [1i64, 2, 3, 4];
	let b = [10i64, 20, 30, 40];
	let mask = [-1i64, 0, -1, 0];
	let mut out = [0i64; 4];
	select_i64(&a, &b, &mask, &mut out);
	assert_eq!(out, [10, 2, 30, 4]);
}

#[test]
fn select_f64_uses_sign_bit_not_zero_test() {
	let a = [1.0f64; 4];
	let b = [2.0f64; 4];
	let mask = [-0.0f64; 4];
	let mut out = [0f64; 4];
	select_f64(&a, &b, &mask, &mut out);
	assert_eq!(out, [2.0; 4]);
}

#[test]
fn select_i8_picks_b_where_mask_set() {
	let a = [1i8, 2, 3, 4];
	let b = [10i8, 20, 30, 40];
	let mask = [-1i8, 0, -1, 0];
	let mut out = [0i8; 4];
	select_i8(&a, &b, &mask, &mut out);
	assert_eq!(out, [10, 2, 30, 4]);
}

#[test]
fn select_i16_picks_b_where_mask_set() {
	let a = [1i16, 2, 3, 4];
	let b = [10i16, 20, 30, 40];
	let mask = [-1i16, 0, -1, 0];
	let mut out = [0i16; 4];
	select_i16(&a, &b, &mask, &mut out);
	assert_eq!(out, [10, 2, 30, 4]);
}

#[test]
fn sllv_i32_shifts_by_the_count_vector() {
	let a = [1i32, 1, 1, 1];
	let count = [0i32, 1, 2, 32];
	let mut out = [0i32; 4];
	sllv_i32(&a, &count, &mut out);
	assert_eq!(out, [1, 2, 4, 0]);
}

#[test]
fn srav_i32_sign_fills_past_bit_width() {
	let a = [-8i32, -8, -8, -8];
	let count = [0i32, 1, 2, 32];
	let mut out = [0i32; 4];
	srav_i32(&a, &count, &mut out);
	assert_eq!(out, [-8, -4, -2, -1]);
}

matches_scalar_across_lengths_test!(
	cmpgt_i8_matches_scalar_across_lengths, cmpgt_i8,
	|x, y| if x > y { -1 } else { 0 }, i8
);
matches_scalar_across_lengths_test!(
	cmple_u8_matches_scalar_across_lengths, cmple_u8,
	|x, y| if x <= y { !0 } else { 0 }, u8
);
matches_scalar_across_lengths_test!(
	cmpgt_i16_matches_scalar_across_lengths, cmpgt_i16,
	|x, y| if x > y { -1 } else { 0 }, i16
);
matches_scalar_across_lengths_test!(
	cmpge_u16_matches_scalar_across_lengths, cmpge_u16,
	|x, y| if x >= y { !0 } else { 0 }, u16
);

// i8/u8/i16/u16 cascade: spot checks across the new macro variants
// (avx2_baseline_bottom, avx2_probed_bottom, shift_imm_avx2_baseline_bottom).
// The underlying ops are already exhaustively tested in `ops::sse::sse2`/
// `ops::avx::avx2`; this tier only needs to confirm dispatch wiring.
matches_scalar_across_lengths_test!(add_i8_matches_scalar_across_lengths, add_i8, |x: i8, y: i8| x.wrapping_add(y), i8);
matches_scalar_across_lengths_test!(adds_i8_matches_scalar_across_lengths, adds_i8, |x: i8, y: i8| x.saturating_add(y), i8);
matches_scalar_across_lengths_test!(cmpeq_i8_matches_scalar_across_lengths, cmpeq_i8, |x, y| if x == y { -1 } else { 0 }, i8);
matches_scalar_across_lengths_test!(min_i8_matches_scalar_across_lengths, min_i8, |x, y| x.min(y), i8);
panics_on_length_mismatch_test!(add_i8_panics_on_length_mismatch, add_i8, i8);

matches_scalar_across_lengths_test!(add_u8_matches_scalar_across_lengths, add_u8, |x: u8, y: u8| x.wrapping_add(y), u8);
matches_scalar_across_lengths_test!(min_u8_matches_scalar_across_lengths, min_u8, |x, y| x.min(y), u8);
matches_scalar_across_lengths_test!(and_u8_matches_scalar_across_lengths, and_u8, |x, y| x & y, u8);

matches_scalar_across_lengths_test!(add_i16_matches_scalar_across_lengths, add_i16, |x: i16, y: i16| x.wrapping_add(y), i16);
matches_scalar_across_lengths_test!(mul_i16_matches_scalar_across_lengths, mul_i16, |x: i16, y: i16| x.wrapping_mul(y), i16);
matches_scalar_across_lengths_test!(min_i16_matches_scalar_across_lengths, min_i16, |x, y| x.min(y), i16);

matches_scalar_across_lengths_test!(add_u16_matches_scalar_across_lengths, add_u16, |x: u16, y: u16| x.wrapping_add(y), u16);
matches_scalar_across_lengths_test!(mul_u16_matches_scalar_across_lengths, mul_u16, |x: u16, y: u16| x.wrapping_mul(y), u16);
matches_scalar_across_lengths_test!(min_u16_matches_scalar_across_lengths, min_u16, |x, y| x.min(y), u16);
panics_on_length_mismatch_test!(add_u16_panics_on_length_mismatch, add_u16, u16);

#[test]
fn shl_i16_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<i16> = (0..len).map(|i| (i as i16).wrapping_mul(3).wrapping_add(1)).collect();
		let mut out = vec![0i16; len];
		shl_i16::<3>(&a, &mut out);
		let expect: Vec<i16> = a.iter().map(|&x| x.wrapping_shl(3)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shr_u16_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<u16> = (0..len).map(|i| (i as u16).wrapping_mul(37)).collect();
		let mut out = vec![0u16; len];
		shr_u16::<2>(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| x.wrapping_shr(2)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn sra_i16_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17] {
		let a: Vec<i16> = (0..len).map(|i| (i as i16) * 5 - 20).collect();
		let mut out = vec![0i16; len];
		sra_i16::<1>(&a, &mut out);
		let expect: Vec<i16> = a.iter().map(|&x| x.wrapping_shr(1)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shl_i32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<i32> = (0..len).map(|i| (i as i32).wrapping_mul(3) + 1).collect();
		let mut out = vec![0i32; len];
		shl_i32::<3>(&a, &mut out);
		let expect: Vec<i32> = a.iter().map(|&x| x.wrapping_shl(3)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn sra_i32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 8, 16, 17] {
		let a: Vec<i32> = (0..len).map(|i| (i as i32) * 5 - 20).collect();
		let mut out = vec![0i32; len];
		sra_i32::<1>(&a, &mut out);
		let expect: Vec<i32> = a.iter().map(|&x| x.wrapping_shr(1)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn fmadd_f32_matches_mul_add_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
		let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 + 1.0).collect();
		let c: Vec<f32> = (0..len).map(|i| i as f32 * 0.25).collect();
		let mut out = vec![0f32; len];
		fmadd_f32(&a, &b, &c, &mut out);
		let expect: Vec<f32> = a.iter().zip(&b).zip(&c).map(|((&x, &y), &z)| x * y + z).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn fmadd_f64_matches_mul_add_across_lengths() {
	for len in [0usize, 1, 2, 3, 4, 8, 9] {
		let a: Vec<f64> = (0..len).map(|i| i as f64 + 1.0).collect();
		let b: Vec<f64> = (0..len).map(|i| (len - i) as f64 + 1.0).collect();
		let c: Vec<f64> = vec![0.5; len];
		let mut out = vec![0f64; len];
		fmadd_f64(&a, &b, &c, &mut out);
		let expect: Vec<f64> = a.iter().zip(&b).zip(&c).map(|((&x, &y), &z)| x * y + z).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn fmadd_f32_panics_on_length_mismatch() {
	let a = [1.0f32, 2.0];
	let b = [1.0f32, 2.0, 3.0];
	let c = [1.0f32, 2.0];
	let mut out = [0f32; 2];
	fmadd_f32(&a, &b, &c, &mut out);
}

macro_rules! popcnt_matches_scalar_across_lengths_test {
	($test_name:ident, $fn_name:ident, $Elem:ty) => {
		#[test]
		fn $test_name() {
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| (i as $Elem).wrapping_mul(0x9E37_79B9u32 as $Elem)).collect();
				let mut out = vec![Default::default(); len];
				$fn_name(&a, &mut out);
				let expect: Vec<$Elem> = a.iter().map(|&x| x.count_ones() as $Elem).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

macro_rules! popcnt_panics_on_length_mismatch_test {
	($test_name:ident, $fn_name:ident, $Elem:ty) => {
		#[test]
		#[should_panic]
		fn $test_name() {
			let a: [$Elem; 2] = [1 as $Elem, 2 as $Elem];
			let mut out: [$Elem; 1] = [Default::default(); 1];
			$fn_name(&a, &mut out);
		}
	};
}

popcnt_matches_scalar_across_lengths_test!(popcnt_u8_matches_scalar_across_lengths, popcnt_u8, u8);
popcnt_matches_scalar_across_lengths_test!(popcnt_u16_matches_scalar_across_lengths, popcnt_u16, u16);
popcnt_matches_scalar_across_lengths_test!(popcnt_u32_matches_scalar_across_lengths, popcnt_u32, u32);
popcnt_matches_scalar_across_lengths_test!(popcnt_u64_matches_scalar_across_lengths, popcnt_u64, u64);
popcnt_panics_on_length_mismatch_test!(popcnt_u8_panics_on_length_mismatch, popcnt_u8, u8);
popcnt_panics_on_length_mismatch_test!(popcnt_u16_panics_on_length_mismatch, popcnt_u16, u16);
popcnt_panics_on_length_mismatch_test!(popcnt_u32_panics_on_length_mismatch, popcnt_u32, u32);
popcnt_panics_on_length_mismatch_test!(popcnt_u64_panics_on_length_mismatch, popcnt_u64, u64);

// f16_to_f32/f32_to_f16/dpbf16_ps_f32/cvtneps_pbh_u16/cvtne2ps_pbh_u16 always
// bottom at a portable scalar reference, so these run unconditionally
// (no `detect() == None` skip needed): correct on every host either way.

#[test]
fn f16_to_f32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<u16> = (0..len).map(|i| 0x3c00 | (i as u16 & 0x3ff)).collect();
		let mut out = vec![0f32; len];
		f16_to_f32(&a, &mut out);
		let expect: Vec<f32> = a.iter().map(|&x| f16_to_f32_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn f16_to_f32_panics_on_length_mismatch() {
	let a = [0u16; 2];
	let mut out = [0f32; 1];
	f16_to_f32(&a, &mut out);
}

#[test]
fn f32_to_f16_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| (i as f32 - 50.0) * 12.375).collect();
		let mut out = vec![0u16; len];
		f32_to_f16(&a, &mut out);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_f16_scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn f32_to_f16_panics_on_length_mismatch() {
	let a = [0f32; 2];
	let mut out = [0u16; 1];
	f32_to_f16(&a, &mut out);
}

#[test]
fn dpbf16_ps_f32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 15, 16, 17, 31, 32, 33, 100] {
		let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let a: Vec<u16> = (0..2 * len).map(|i| f32_to_bf16_scalar((i as f32 - 40.0) * 0.25)).collect();
		let b: Vec<u16> = (0..2 * len).map(|i| f32_to_bf16_scalar((i as f32 % 5.0) - 2.0)).collect();
		let mut got = vec![0f32; len];
		dpbf16_ps_f32(&src, &a, &b, &mut got);

		let expect: Vec<f32> = (0..len)
			.map(|j| {
				let mut acc = src[j];
				acc += bf16_to_f32_scalar(a[2 * j + 1]) * bf16_to_f32_scalar(b[2 * j + 1]);
				acc += bf16_to_f32_scalar(a[2 * j]) * bf16_to_f32_scalar(b[2 * j]);
				acc
			})
			.collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn dpbf16_ps_f32_panics_on_length_mismatch() {
	let src = [0f32; 1];
	let a = [0u16; 2];
	let b = [0u16; 3];
	let mut out = [0f32; 1];
	dpbf16_ps_f32(&src, &a, &b, &mut out);
}

#[test]
fn cvtneps_pbh_u16_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| (i as f32 - 50.0) * 12.375).collect();
		let mut got = vec![0u16; len];
		cvtneps_pbh_u16(&a, &mut got);
		let expect: Vec<u16> = a.iter().map(|&x| f32_to_bf16_scalar(x)).collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn cvtneps_pbh_u16_panics_on_length_mismatch() {
	let a = [0f32; 2];
	let mut out = [0u16; 1];
	cvtneps_pbh_u16(&a, &mut out);
}

#[test]
fn cvtne2ps_pbh_u16_matches_scalar_across_lengths_and_lane_order() {
	for len in [0usize, 1, 3, 15, 16, 17, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 + 0.3).collect();
		let b: Vec<f32> = (0..len).map(|i| -(i as f32) - 0.7).collect();
		let mut got = vec![0u16; 2 * len];
		cvtne2ps_pbh_u16(&a, &b, &mut got);
		let expect: Vec<u16> =
			b.iter().map(|&x| f32_to_bf16_scalar(x)).chain(a.iter().map(|&x| f32_to_bf16_scalar(x))).collect();
		assert_eq!(got, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn cvtne2ps_pbh_u16_panics_on_length_mismatch() {
	let a = [0f32; 2];
	let b = [0f32; 3];
	let mut out = [0u16; 4];
	cvtne2ps_pbh_u16(&a, &b, &mut out);
}

// madd52lo_u64/madd52hi_u64/dpbusd_i32/dpbusds_i32/dpwssd_i32/dpwssds_i32
// also bottom at a portable scalar reference (Avx512Ifma/Avx512Vnni ->
// AvxIfma/AvxVnni -> scalar), so these run unconditionally too.

#[test]
fn madd52lo_u64_matches_scalar_across_lengths() {
	for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let src: Vec<u64> = (0..len).map(|i| i as u64 * 1000).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		madd52lo_u64(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52lo_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn madd52lo_u64_panics_on_length_mismatch() {
	let src = [0u64; 2];
	let a = [0u64; 2];
	let b = [0u64; 1];
	let mut out = [0u64; 2];
	madd52lo_u64(&src, &a, &b, &mut out);
}

#[test]
fn madd52hi_u64_matches_scalar_across_lengths() {
	for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
		let src: Vec<u64> = (0..len).map(|i| i as u64).collect();
		let a: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x1_0000_0007)).collect();
		let b: Vec<u64> = (0..len).map(|i| (i as u64).wrapping_mul(0x3_0000_0001) ^ 0xdead_beef).collect();
		let mut out = vec![0u64; len];
		madd52hi_u64(&src, &a, &b, &mut out);
		let expect: Vec<u64> = (0..len).map(|i| madd52hi_scalar(src[i], a[i], b[i])).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn madd52hi_u64_panics_on_length_mismatch() {
	let src = [0u64; 2];
	let a = [0u64; 2];
	let b = [0u64; 1];
	let mut out = [0u64; 2];
	madd52hi_u64(&src, &a, &b, &mut out);
}

#[test]
fn dpbusd_i32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<u8> = (0..len * 4).map(|i| (i % 20) as u8 + 1).collect();
		let b: Vec<i8> = (0..len * 4).map(|i| ((i % 7) as i8) - 3).collect();
		let mut out = vec![0i32; len];
		dpbusd_i32(&src, &a, &b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let sum: i64 = (0..4).map(|k| a[4 * j + k] as i64 * b[4 * j + k] as i64).sum();
				vnni_acc_wrapping(src[j], sum)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn dpbusd_i32_panics_on_length_mismatch() {
	let src = [0i32; 2];
	let a = [0u8; 8];
	let b = [0i8; 7];
	let mut out = [0i32; 2];
	dpbusd_i32(&src, &a, &b, &mut out);
}

#[test]
fn dpbusds_i32_saturates_at_i32_max() {
	let src = vec![i32::MAX; 5];
	let a = vec![u8::MAX; 20];
	let b = vec![i8::MAX; 20];
	let mut out = vec![0i32; 5];
	dpbusds_i32(&src, &a, &b, &mut out);
	assert!(out.iter().all(|&x| x == i32::MAX));
}

#[test]
fn dpwssd_i32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: Vec<i16> = (0..len * 2).map(|i| (i as i16) - 50).collect();
		let b: Vec<i16> = (0..len * 2).map(|i| (i as i16) - 25).collect();
		let mut out = vec![0i32; len];
		dpwssd_i32(&src, &a, &b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let sum: i64 = (0..2).map(|k| a[2 * j + k] as i64 * b[2 * j + k] as i64).sum();
				vnni_acc_wrapping(src[j], sum)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
#[should_panic]
fn dpwssd_i32_panics_on_length_mismatch() {
	let src = [0i32; 2];
	let a = [0i16; 4];
	let b = [0i16; 3];
	let mut out = [0i32; 2];
	dpwssd_i32(&src, &a, &b, &mut out);
}

#[test]
fn dpwssds_i32_saturates_at_i32_min() {
	let src = vec![i32::MIN; 4];
	let a = vec![i16::MIN; 8];
	let b = vec![i16::MAX; 8];
	let mut out = vec![0i32; 4];
	dpwssds_i32(&src, &a, &b, &mut out);
	assert!(out.iter().all(|&x| x == i32::MIN));
}

#[test]
fn p4fmadd_f32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let b: [Vec<f32>; 4] =
			core::array::from_fn(|n| (0..len).map(|i| (n * len + i) as f32 * 0.25 - 10.0).collect());
		let c = [1.0f32, -2.0, 0.5, 4.0];
		let mut out = vec![0f32; len];
		p4fmadd_f32(&a, [&b[0], &b[1], &b[2], &b[3]], c, &mut out);
		let expect: Vec<f32> = (0..len)
			.map(|i| a[i] + b[0][i] * c[0] + b[1][i] * c[1] + b[2][i] * c[2] + b[3][i] * c[3])
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4fnmadd_f32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
		let b: [Vec<f32>; 4] =
			core::array::from_fn(|n| (0..len).map(|i| (n * len + i) as f32 * 0.25 - 10.0).collect());
		let c = [1.0f32, -2.0, 0.5, 4.0];
		let mut out = vec![0f32; len];
		p4fnmadd_f32(&a, [&b[0], &b[1], &b[2], &b[3]], c, &mut out);
		let expect: Vec<f32> = (0..len)
			.map(|i| a[i] - b[0][i] * c[0] - b[1][i] * c[1] - b[2][i] * c[2] - b[3][i] * c[3])
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4fmadd_f32_panics_on_length_mismatch() {
	let a = [0f32; 4];
	let b0 = [0f32; 4];
	let b1 = [0f32; 3];
	let b2 = [0f32; 4];
	let b3 = [0f32; 4];
	let mut out = [0f32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		p4fmadd_f32(&a, [&b0, &b1, &b2, &b3], [0.0; 4], &mut out);
	}));
	assert!(result.is_err());
}

#[test]
fn p4dpwssd_i32_matches_scalar_across_lengths() {
	for len in [0usize, 1, 15, 16, 17, 33, 100] {
		let src: Vec<i32> = (0..len).map(|i| i as i32).collect();
		let a: [Vec<i16>; 4] =
			core::array::from_fn(|n| (0..len * 2).map(|i| ((n * len * 2 + i) as i16) - 50).collect());
		let b: [i16; 8] = [3, -2, 1, 0, -1, 5, 2, -4];
		let mut out = vec![0i32; len];
		p4dpwssd_i32(&src, [&a[0], &a[1], &a[2], &a[3]], b, &mut out);
		let expect: Vec<i32> = (0..len)
			.map(|j| {
				let mut acc: i64 = src[j] as i64;
				for n in 0..4 {
					acc += a[n][2 * j] as i64 * b[2 * n] as i64 + a[n][2 * j + 1] as i64 * b[2 * n + 1] as i64;
				}
				acc as i32
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn p4dpwssds_i32_saturates_at_i32_max() {
	let src = [i32::MAX; 4];
	let a = [[i16::MAX; 8]; 4];
	let b = [i16::MAX; 8];
	let mut out = [0i32; 4];
	p4dpwssds_i32(&src, [&a[0], &a[1], &a[2], &a[3]], b, &mut out);
	assert!(out.iter().all(|&x| x == i32::MAX));
}

#[test]
fn p4dpwssd_i32_panics_on_length_mismatch() {
	let src = [0i32; 4];
	let a0 = [0i16; 8];
	let a1 = [0i16; 7];
	let a2 = [0i16; 8];
	let a3 = [0i16; 8];
	let mut out = [0i32; 4];
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		p4dpwssd_i32(&src, [&a0, &a1, &a2, &a3], [0i16; 8], &mut out);
	}));
	assert!(result.is_err());
}
