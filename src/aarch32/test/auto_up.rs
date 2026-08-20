use super::*;

fn check_binop_i32(f: fn(&[i32], &[i32], &mut [i32]), scalar: fn(i32, i32) -> i32) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 16] {
		let a: Vec<i32> = (0..len as i32).map(|i| i * 3 - 7).collect();
		let b: Vec<i32> = (0..len as i32).map(|i| i * -2 + 5).collect();
		let mut out = vec![0i32; len];
		f(&a, &b, &mut out);
		let expect: Vec<i32> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_binop_f32(f: fn(&[f32], &[f32], &mut [f32]), scalar: fn(f32, f32) -> f32) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<f32> = (0..len as i32).map(|i| i as f32 * 1.5 - 3.0).collect();
		let b: Vec<f32> = (0..len as i32).map(|i| i as f32 * -0.5 + 2.0).collect();
		let mut out = vec![0f32; len];
		f(&a, &b, &mut out);
		let expect: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_binop_u32(f: fn(&[u32], &[u32], &mut [u32]), scalar: fn(u32, u32) -> u32) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(0x1357_9BDF)).collect();
		let b: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(0x2468_ACE0) ^ 0xFFFF_0000).collect();
		let mut out = vec![0u32; len];
		f(&a, &b, &mut out);
		let expect: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_binop_i8(f: fn(&[i8], &[i8], &mut [i8]), scalar: fn(i8, i8) -> i8) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<i8> = (0..len as i32).map(|i| ((i * 37 - 61) % 128) as i8).collect();
		let b: Vec<i8> = (0..len as i32).map(|i| ((i * 53 + 17) % 128) as i8).collect();
		let mut out = vec![0i8; len];
		f(&a, &b, &mut out);
		let expect: Vec<i8> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn add_sub_mul_i32_match_scalar_across_lengths() {
	check_binop_i32(add_i32, i32::wrapping_add);
	check_binop_i32(sub_i32, i32::wrapping_sub);
	check_binop_i32(mul_i32, i32::wrapping_mul);
}

#[test]
fn add_sub_mul_f32_match_scalar_across_lengths() {
	check_binop_f32(add_f32, |a, b| a + b);
	check_binop_f32(sub_f32, |a, b| a - b);
	check_binop_f32(mul_f32, |a, b| a * b);
}

#[test]
fn and_or_xor_u32_match_scalar_across_lengths() {
	check_binop_u32(and_u32, |a, b| a & b);
	check_binop_u32(or_u32, |a, b| a | b);
	check_binop_u32(xor_u32, |a, b| a ^ b);
}

#[test]
fn qadd_qsub_sadd_ssub_i8_match_scalar_across_lengths() {
	check_binop_i8(qadd_i8, i8::saturating_add);
	check_binop_i8(qsub_i8, i8::saturating_sub);
	check_binop_i8(sadd_i8, i8::wrapping_add);
	check_binop_i8(ssub_i8, i8::wrapping_sub);
}

#[test]
#[should_panic]
fn mismatched_lengths_panic() {
	let a = [1i32, 2, 3];
	let b = [1i32, 2];
	let mut out = [0i32; 3];
	add_i32(&a, &b, &mut out);
}

fn check_unop_i32(f: fn(&[i32], &mut [i32]), scalar: fn(i32) -> i32) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<i32> = (0..len as i32).map(|i| i * 3 - 7).collect();
		let mut out = vec![0i32; len];
		f(&a, &mut out);
		let expect: Vec<i32> = a.iter().map(|&x| scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_unop_f32(f: fn(&[f32], &mut [f32]), scalar: fn(f32) -> f32) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<f32> = (0..len as i32).map(|i| i as f32 * 1.5 - 3.0).collect();
		let mut out = vec![0f32; len];
		f(&a, &mut out);
		let expect: Vec<f32> = a.iter().map(|&x| scalar(x)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_cmp_i32(f: fn(&[i32], &[i32], &mut [u32]), scalar: fn(i32, i32) -> bool) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<i32> = (0..len as i32).map(|i| i * 3 - 7).collect();
		let b: Vec<i32> = (0..len as i32).map(|i| i * -2 + 5).collect();
		let mut out = vec![0u32; len];
		f(&a, &b, &mut out);
		let expect: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| if scalar(x, y) { u32::MAX } else { 0 }).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_cmp_f32(f: fn(&[f32], &[f32], &mut [u32]), scalar: fn(f32, f32) -> bool) {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<f32> = (0..len as i32).map(|i| i as f32 * 1.5 - 3.0).collect();
		let b: Vec<f32> = (0..len as i32).map(|i| i as f32 * -0.5 + 2.0).collect();
		let mut out = vec![0u32; len];
		f(&a, &b, &mut out);
		let expect: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| if scalar(x, y) { u32::MAX } else { 0 }).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn max_min_i32_f32_match_scalar_across_lengths() {
	check_binop_i32(max_i32, |a, b| a.max(b));
	check_binop_i32(min_i32, |a, b| a.min(b));
	check_binop_f32(max_f32, f32::max);
	check_binop_f32(min_f32, f32::min);
}

#[test]
fn andnot_u32_matches_scalar_across_lengths() {
	check_binop_u32(andnot_u32, |a, b| a & !b);
}

#[test]
fn abs_neg_i32_f32_not_u32_match_scalar_across_lengths() {
	check_unop_i32(abs_i32, i32::wrapping_abs);
	check_unop_i32(neg_i32, i32::wrapping_neg);
	check_unop_f32(abs_f32, f32::abs);
	check_unop_f32(neg_f32, |x| -x);
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(0x1357_9BDF)).collect();
		let mut out = vec![0u32; len];
		not_u32(&a, &mut out);
		let expect: Vec<u32> = a.iter().map(|&x| !x).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn cmp_i32_family_matches_scalar_across_lengths() {
	check_cmp_i32(cmpeq_i32, |a, b| a == b);
	check_cmp_i32(cmpgt_i32, |a, b| a > b);
	check_cmp_i32(cmpge_i32, |a, b| a >= b);
	check_cmp_i32(cmplt_i32, |a, b| a < b);
	check_cmp_i32(cmple_i32, |a, b| a <= b);
}

#[test]
fn cmp_f32_family_matches_scalar_across_lengths() {
	check_cmp_f32(cmpeq_f32, |a, b| a == b);
	check_cmp_f32(cmpgt_f32, |a, b| a > b);
	check_cmp_f32(cmpge_f32, |a, b| a >= b);
	check_cmp_f32(cmplt_f32, |a, b| a < b);
	check_cmp_f32(cmple_f32, |a, b| a <= b);
}

#[test]
fn shl_i32_matches_vshl_semantics() {
	// a is fixed, b covers 0, small, boundary (31/32), negative, and
	// out-of-range-magnitude (>=32) shift amounts.
	let a = [1i32, -1, i32::MIN, 100, -100, 7, -7, 1 << 30];
	let b = [0i32, 1, 31, 32, -1, -31, -32, 40];
	let mut out = [0i32; 8];
	shl_i32(&a, &b, &mut out);
	let expect: [i32; 8] = core::array::from_fn(|i| {
		let amt = b[i];
		if amt >= 32 {
			0
		} else if amt <= -32 {
			if a[i] < 0 { -1 } else { 0 }
		} else if amt >= 0 {
			a[i].wrapping_shl(amt as u32)
		} else {
			a[i] >> (-amt) as u32
		}
	});
	assert_eq!(out, expect);
}

#[test]
fn fmadd_f32_matches_unfused_scalar() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<f32> = (0..len as i32).map(|i| i as f32 * 1.5 - 3.0).collect();
		let b: Vec<f32> = (0..len as i32).map(|i| i as f32 * -0.5 + 2.0).collect();
		let c: Vec<f32> = (0..len as i32).map(|i| i as f32 * 0.25 + 1.0).collect();
		let mut out = vec![0f32; len];
		fmadd_f32(&a, &b, &c, &mut out);
		let expect: Vec<f32> = (0..len).map(|i| a[i] * b[i] + c[i]).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn select_i32_u32_f32_match_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a_i: Vec<i32> = (0..len as i32).map(|i| i * 3 - 7).collect();
		let b_i: Vec<i32> = (0..len as i32).map(|i| i * -2 + 5).collect();
		let mask: Vec<u32> = (0..len).map(|i| if i % 2 == 0 { u32::MAX } else { 0 }).collect();
		let mut out_i = vec![0i32; len];
		select_i32(&a_i, &b_i, &mask, &mut out_i);
		let expect_i: Vec<i32> = (0..len).map(|i| if mask[i] != 0 { b_i[i] } else { a_i[i] }).collect();
		assert_eq!(out_i, expect_i, "select_i32 len={len}");

		let a_u: Vec<u32> = a_i.iter().map(|&x| x as u32).collect();
		let b_u: Vec<u32> = b_i.iter().map(|&x| x as u32).collect();
		let mut out_u = vec![0u32; len];
		select_u32(&a_u, &b_u, &mask, &mut out_u);
		let expect_u: Vec<u32> = (0..len).map(|i| if mask[i] != 0 { b_u[i] } else { a_u[i] }).collect();
		assert_eq!(out_u, expect_u, "select_u32 len={len}");

		let a_f: Vec<f32> = (0..len as i32).map(|i| i as f32 * 1.5 - 3.0).collect();
		let b_f: Vec<f32> = (0..len as i32).map(|i| i as f32 * -0.5 + 2.0).collect();
		let mut out_f = vec![0f32; len];
		select_f32(&a_f, &b_f, &mask, &mut out_f);
		let expect_f: Vec<f32> = (0..len).map(|i| if mask[i] != 0 { b_f[i] } else { a_f[i] }).collect();
		assert_eq!(out_f, expect_f, "select_f32 len={len}");
	}
}

/// FullFP16 oracle: `half::f16` for bit-exact rounding, deliberately
/// independent of the `auto_down` production fallback's nightly `f16`
/// primitive (no test that only confirms it agrees with itself).
fn f16_bits(x: f32) -> u16 {
	half::f16::from_f32(x).to_bits()
}
fn f16_oracle(a: u16, b: u16, op: impl Fn(f32, f32) -> f32) -> u16 {
	let x = half::f16::from_bits(a).to_f32();
	let y = half::f16::from_bits(b).to_f32();
	half::f16::from_f32(op(x, y)).to_bits()
}

fn check_binop_f16(f: fn(&[u16], &[u16], &mut [u16]), scalar: fn(f32, f32) -> f32) {
	for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17] {
		let a: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * 1.5 - 3.0)).collect();
		let b: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * -0.5 + 2.0)).collect();
		let mut out = vec![0u16; len];
		f(&a, &b, &mut out);
		let expect: Vec<u16> = a.iter().zip(&b).map(|(&x, &y)| f16_oracle(x, y, scalar)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_cmp_f16(f: fn(&[u16], &[u16], &mut [u16]), scalar: fn(f32, f32) -> bool) {
	for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17] {
		let a: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * 1.5 - 3.0)).collect();
		let b: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * -0.5 + 2.0)).collect();
		let mut out = vec![0u16; len];
		f(&a, &b, &mut out);
		let expect: Vec<u16> = a
			.iter()
			.zip(&b)
			.map(|(&x, &y)| {
				let xf = half::f16::from_bits(x).to_f32();
				let yf = half::f16::from_bits(y).to_f32();
				if scalar(xf, yf) { u16::MAX } else { 0 }
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn add_sub_mul_max_min_f16_match_scalar_across_lengths() {
	check_binop_f16(add_f16, |a, b| a + b);
	check_binop_f16(sub_f16, |a, b| a - b);
	check_binop_f16(mul_f16, |a, b| a * b);
	check_binop_f16(max_f16, f32::max);
	check_binop_f16(min_f16, f32::min);
}

#[test]
fn abs_neg_f16_match_scalar_across_lengths() {
	for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17] {
		let a: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * 1.5 - 3.0)).collect();
		let mut out_abs = vec![0u16; len];
		abs_f16(&a, &mut out_abs);
		let expect_abs: Vec<u16> =
			a.iter().map(|&x| f16_bits(half::f16::from_bits(x).to_f32().abs())).collect();
		assert_eq!(out_abs, expect_abs, "abs len={len}");

		let mut out_neg = vec![0u16; len];
		neg_f16(&a, &mut out_neg);
		let expect_neg: Vec<u16> =
			a.iter().map(|&x| f16_bits(-half::f16::from_bits(x).to_f32())).collect();
		assert_eq!(out_neg, expect_neg, "neg len={len}");
	}
}

#[test]
fn cmp_f16_family_matches_scalar_across_lengths() {
	check_cmp_f16(cmpeq_f16, |a, b| a == b);
	check_cmp_f16(cmpgt_f16, |a, b| a > b);
	check_cmp_f16(cmpge_f16, |a, b| a >= b);
	check_cmp_f16(cmplt_f16, |a, b| a < b);
	check_cmp_f16(cmple_f16, |a, b| a <= b);
}

#[test]
fn fmadd_f16_matches_unfused_scalar() {
	for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17] {
		let a: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * 1.5 - 3.0)).collect();
		let b: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * -0.5 + 2.0)).collect();
		let c: Vec<u16> = (0..len as i32).map(|i| f16_bits(i as f32 * 0.25 + 1.0)).collect();
		let mut out = vec![0u16; len];
		fmadd_f16(&a, &b, &c, &mut out);
		let expect: Vec<u16> = (0..len)
			.map(|i| {
				let x = half::f16::from_bits(a[i]).to_f32();
				let y = half::f16::from_bits(b[i]).to_f32();
				let z = half::f16::from_bits(c[i]).to_f32();
				f16_bits(x * y + z)
			})
			.collect();
		assert_eq!(out, expect, "len={len}");
	}
}

fn check_binop_i16(f: fn(&[i16], &[i16], &mut [i16]), scalar: fn(i16, i16) -> i16) {
	for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9] {
		let a: Vec<i16> = (0..len as i32).map(|i| (i * 37 - 61) as i16).collect();
		let b: Vec<i16> = (0..len as i32).map(|i| (i * 53 + 17) as i16).collect();
		let mut out = vec![0i16; len];
		f(&a, &b, &mut out);
		let expect: Vec<i16> = a.iter().zip(&b).map(|(&x, &y)| scalar(x, y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn shadd_shsub_i8_match_scalar_across_lengths() {
	check_binop_i8(shadd_i8, |a, b| ((a as i16 + b as i16) >> 1) as i8);
	check_binop_i8(shsub_i8, |a, b| ((a as i16 - b as i16) >> 1) as i8);
}

#[test]
fn usub_u8_matches_scalar_across_lengths() {
	for len in [0usize, 1, 3, 4, 5, 7, 8, 9] {
		let a: Vec<u8> = (0..len as i32).map(|i| ((i * 37) % 251) as u8).collect();
		let b: Vec<u8> = (0..len as i32).map(|i| ((i * 53 + 17) % 251) as u8).collect();
		let mut out = vec![0u8; len];
		usub_u8(&a, &b, &mut out);
		let expect: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_sub(y)).collect();
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn qadd_qsub_sadd_shadd_shsub_i16_match_scalar_across_lengths() {
	check_binop_i16(qadd_i16, i16::saturating_add);
	check_binop_i16(qsub_i16, i16::saturating_sub);
	check_binop_i16(sadd_i16, i16::wrapping_add);
	check_binop_i16(shadd_i16, |a, b| ((a as i32 + b as i32) >> 1) as i16);
	check_binop_i16(shsub_i16, |a, b| ((a as i32 - b as i32) >> 1) as i16);
}

fn check_cross_i16(f: fn(&[i16], &[i16], &mut [i16]), scalar: fn(i16, i16, i16, i16) -> (i16, i16)) {
	for len in [0usize, 2, 4, 6, 8, 10] {
		let a: Vec<i16> = (0..len as i32).map(|i| (i * 37 - 61) as i16).collect();
		let b: Vec<i16> = (0..len as i32).map(|i| (i * 53 + 17) as i16).collect();
		let mut out = vec![0i16; len];
		f(&a, &b, &mut out);
		let mut expect = vec![0i16; len];
		for pair in 0..len / 2 {
			let (r0, r1) = scalar(a[2 * pair], a[2 * pair + 1], b[2 * pair], b[2 * pair + 1]);
			expect[2 * pair] = r0;
			expect[2 * pair + 1] = r1;
		}
		assert_eq!(out, expect, "len={len}");
	}
}

#[test]
fn qasx_qsax_sasx_i16_match_scalar_across_lengths() {
	check_cross_i16(qasx_i16, |a0, a1, b0, b1| (a0.saturating_sub(b1), a1.saturating_add(b0)));
	check_cross_i16(qsax_i16, |a0, a1, b0, b1| (a0.saturating_add(b1), a1.saturating_sub(b0)));
	check_cross_i16(sasx_i16, |a0, a1, b0, b1| (a0.wrapping_sub(b1), a1.wrapping_add(b0)));
}

#[test]
#[should_panic]
fn qasx_i16_odd_length_panics() {
	let a = [1i16, 2, 3];
	let b = [1i16, 2, 3];
	let mut out = [0i16; 3];
	qasx_i16(&a, &b, &mut out);
}
