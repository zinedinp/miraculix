//! Pure scalar fallback for [`super::auto_up`]'s slice ops: no SIMD, no
//! token. Used when no hardware token is available and as the remainder
//! handler after a SIMD-chunked loop.

pub fn add_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_add(b[i]);
	}
}
pub fn sub_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_sub(b[i]);
	}
}
pub fn mul_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_mul(b[i]);
	}
}

pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i] + b[i];
	}
}
pub fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i] - b[i];
	}
}
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i] * b[i];
	}
}

pub fn and_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = a[i] & b[i];
	}
}
pub fn or_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = a[i] | b[i];
	}
}
pub fn xor_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = a[i] ^ b[i];
	}
}
pub fn andnot_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = a[i] & !b[i];
	}
}

pub fn max_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].max(b[i]);
	}
}
pub fn min_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].min(b[i]);
	}
}
pub fn max_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i].max(b[i]);
	}
}
pub fn min_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i].min(b[i]);
	}
}

/// `VSHL.S32` semantics: `b[i]` is a signed shift count, positive shifts
/// left, negative shifts right (arithmetic); `|b[i]| >= 32` saturates (0 for
/// left, sign-fill for right) instead of panicking like Rust's `<<`/`>>`.
pub fn shl_i32(a: &[i32], b: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		let amt = b[i];
		out[i] = if amt >= 32 {
			0
		} else if amt <= -32 {
			if a[i] < 0 { -1 } else { 0 }
		} else if amt >= 0 {
			a[i].wrapping_shl(amt as u32)
		} else {
			a[i] >> (-amt) as u32
		};
	}
}

pub fn abs_i32(a: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_abs();
	}
}
pub fn neg_i32(a: &[i32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_neg();
	}
}
pub fn abs_f32(a: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i].abs();
	}
}
pub fn neg_f32(a: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = -a[i];
	}
}
pub fn not_u32(a: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = !a[i];
	}
}

const MASK_TRUE: u32 = u32::MAX;
const MASK_FALSE: u32 = 0;

pub fn cmpeq_i32(a: &[i32], b: &[i32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] == b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmpgt_i32(a: &[i32], b: &[i32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] > b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmpge_i32(a: &[i32], b: &[i32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] >= b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmplt_i32(a: &[i32], b: &[i32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] < b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmple_i32(a: &[i32], b: &[i32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] <= b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmpeq_f32(a: &[f32], b: &[f32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] == b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmpgt_f32(a: &[f32], b: &[f32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] > b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmpge_f32(a: &[f32], b: &[f32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] >= b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmplt_f32(a: &[f32], b: &[f32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] < b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}
pub fn cmple_f32(a: &[f32], b: &[f32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if a[i] <= b[i] { MASK_TRUE } else { MASK_FALSE };
	}
}

pub fn fmadd_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = a[i] * b[i] + c[i];
	}
}

pub fn select_i32(a: &[i32], b: &[i32], mask: &[u32], out: &mut [i32]) {
	for i in 0..a.len() {
		out[i] = if mask[i] != 0 { b[i] } else { a[i] };
	}
}
pub fn select_u32(a: &[u32], b: &[u32], mask: &[u32], out: &mut [u32]) {
	for i in 0..a.len() {
		out[i] = if mask[i] != 0 { b[i] } else { a[i] };
	}
}
pub fn select_f32(a: &[f32], b: &[f32], mask: &[u32], out: &mut [f32]) {
	for i in 0..a.len() {
		out[i] = if mask[i] != 0 { b[i] } else { a[i] };
	}
}

/// FullFP16 scalar fallback (`[u16]` bit patterns, see
/// [`super::ops::fp16`]). Uses the unstable `f16` primitive (`core` has no
/// `f16` arithmetic otherwise) - hardware-independent, real IEEE-754
/// binary16 rounding, not an approximation.
fn f16_binop(a: &[u16], b: &[u16], out: &mut [u16], op: impl Fn(f16, f16) -> f16) {
	for i in 0..a.len() {
		out[i] = op(f16::from_bits(a[i]), f16::from_bits(b[i])).to_bits();
	}
}

pub fn add_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	f16_binop(a, b, out, |x, y| x + y);
}
pub fn sub_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	f16_binop(a, b, out, |x, y| x - y);
}
pub fn mul_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	f16_binop(a, b, out, |x, y| x * y);
}
pub fn max_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	f16_binop(a, b, out, |x, y| if x > y { x } else { y });
}
pub fn min_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	f16_binop(a, b, out, |x, y| if x < y { x } else { y });
}

pub fn abs_f16(a: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		let x = f16::from_bits(a[i]);
		out[i] = (if x < 0.0 { -x } else { x }).to_bits();
	}
}
pub fn neg_f16(a: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = (-f16::from_bits(a[i])).to_bits();
	}
}

const MASK_TRUE_16: u16 = u16::MAX;
const MASK_FALSE_16: u16 = 0;

pub fn cmpeq_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = if f16::from_bits(a[i]) == f16::from_bits(b[i]) { MASK_TRUE_16 } else { MASK_FALSE_16 };
	}
}
pub fn cmpgt_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = if f16::from_bits(a[i]) > f16::from_bits(b[i]) { MASK_TRUE_16 } else { MASK_FALSE_16 };
	}
}
pub fn cmpge_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = if f16::from_bits(a[i]) >= f16::from_bits(b[i]) { MASK_TRUE_16 } else { MASK_FALSE_16 };
	}
}
pub fn cmplt_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = if f16::from_bits(a[i]) < f16::from_bits(b[i]) { MASK_TRUE_16 } else { MASK_FALSE_16 };
	}
}
pub fn cmple_f16(a: &[u16], b: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		out[i] = if f16::from_bits(a[i]) <= f16::from_bits(b[i]) { MASK_TRUE_16 } else { MASK_FALSE_16 };
	}
}

pub fn fmadd_f16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
	for i in 0..a.len() {
		let r = f16::from_bits(a[i]) * f16::from_bits(b[i]) + f16::from_bits(c[i]);
		out[i] = r.to_bits();
	}
}

pub fn qadd_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = a[i].saturating_add(b[i]);
	}
}
pub fn qsub_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = a[i].saturating_sub(b[i]);
	}
}
pub fn sadd_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_add(b[i]);
	}
}
pub fn ssub_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_sub(b[i]);
	}
}
/// `SHADD8`: halving add, full-precision sum before the arithmetic `>> 1`
/// (not `a.wrapping_add(b) / 2`, which would overflow-wrap first).
pub fn shadd_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = ((a[i] as i16 + b[i] as i16) >> 1) as i8;
	}
}
/// `SHSUB8`: halving subtract, same full-precision-then-shift shape as [`shadd_i8`].
pub fn shsub_i8(a: &[i8], b: &[i8], out: &mut [i8]) {
	for i in 0..a.len() {
		out[i] = ((a[i] as i16 - b[i] as i16) >> 1) as i8;
	}
}
pub fn usub_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_sub(b[i]);
	}
}

pub fn qadd_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for i in 0..a.len() {
		out[i] = a[i].saturating_add(b[i]);
	}
}
pub fn qsub_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for i in 0..a.len() {
		out[i] = a[i].saturating_sub(b[i]);
	}
}
pub fn sadd_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for i in 0..a.len() {
		out[i] = a[i].wrapping_add(b[i]);
	}
}
/// `SHADD16`: halving add, same full-precision-then-shift shape as [`shadd_i8`].
pub fn shadd_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for i in 0..a.len() {
		out[i] = ((a[i] as i32 + b[i] as i32) >> 1) as i16;
	}
}
/// `SHSUB16`: halving subtract, same shape as [`shadd_i16`].
pub fn shsub_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for i in 0..a.len() {
		out[i] = ((a[i] as i32 - b[i] as i32) >> 1) as i16;
	}
}

/// `QASX`: saturating cross add-subtract on **packed pairs**
/// (`out[2k]=a[2k]-b[2k+1], out[2k+1]=a[2k+1]+b[2k]`, per whole `[i16;2]`
/// lane pair, not adjacent scalars in isolation).
pub fn qasx_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for pair in 0..a.len() / 2 {
		let (a0, a1) = (a[2 * pair], a[2 * pair + 1]);
		let (b0, b1) = (b[2 * pair], b[2 * pair + 1]);
		out[2 * pair] = a0.saturating_sub(b1);
		out[2 * pair + 1] = a1.saturating_add(b0);
	}
}
/// `QSAX`: saturating cross subtract-add on packed pairs, mirror of [`qasx_i16`].
pub fn qsax_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for pair in 0..a.len() / 2 {
		let (a0, a1) = (a[2 * pair], a[2 * pair + 1]);
		let (b0, b1) = (b[2 * pair], b[2 * pair + 1]);
		out[2 * pair] = a0.saturating_add(b1);
		out[2 * pair + 1] = a1.saturating_sub(b0);
	}
}
/// `SASX`: wrapping cross add-subtract on packed pairs, same lane layout as [`qasx_i16`].
pub fn sasx_i16(a: &[i16], b: &[i16], out: &mut [i16]) {
	for pair in 0..a.len() / 2 {
		let (a0, a1) = (a[2 * pair], a[2 * pair + 1]);
		let (b0, b1) = (b[2 * pair], b[2 * pair + 1]);
		out[2 * pair] = a0.wrapping_sub(b1);
		out[2 * pair + 1] = a1.wrapping_add(b0);
	}
}