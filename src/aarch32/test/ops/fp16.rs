use super::{Fp16Fma, Fp16Neon};
use half::f16;

fn require() -> Option<Fp16Neon> {
	Fp16Neon::detect()
}

fn require_fma() -> Option<Fp16Fma> {
	Fp16Fma::detect()
}

/// `[f32; 8]` -> `[u16; 8]` f16 bit pattern (test input construction).
fn bits(v: [f32; 8]) -> [u16; 8] {
	v.map(|x| f16::from_f32(x).to_bits())
}

/// Elementwise oracle: round each input to f16, do the op in f32 (exact for
/// a single rounded-input op), round the result back to f16. No
/// multi-step error accumulation risk: see `ops/fp16.rs`'s module doc for
/// why `half` (not hand-rolled IEEE-754 rounding) is used for this.
fn oracle(a: [f32; 8], b: [f32; 8], f: impl Fn(f32, f32) -> f32) -> [u16; 8] {
	core::array::from_fn(|i| {
		let (x, y) = (f16::from_f32(a[i]).to_f32(), f16::from_f32(b[i]).to_f32());
		f16::from_f32(f(x, y)).to_bits()
	})
}

fn oracle_unop(a: [f32; 8], f: impl Fn(f32) -> f32) -> [u16; 8] {
	core::array::from_fn(|i| {
		let x = f16::from_f32(a[i]).to_f32();
		f16::from_f32(f(x)).to_bits()
	})
}

const MASK_TRUE: u16 = u16::MAX;
const MASK_FALSE: u16 = 0;

fn oracle_cmp(a: [f32; 8], b: [f32; 8], f: impl Fn(f32, f32) -> bool) -> [u16; 8] {
	core::array::from_fn(|i| {
		let (x, y) = (f16::from_f32(a[i]).to_f32(), f16::from_f32(b[i]).to_f32());
		if f(x, y) { MASK_TRUE } else { MASK_FALSE }
	})
}

const A: [f32; 8] = [1.5, -2.25, 0.0, 100.0, -0.5, 3.0, f32::INFINITY, f32::NEG_INFINITY];
const B: [f32; 8] = [10.25, -20.5, 1.0, -1.0, 2.0, 3.0, 1.0, -1.0];

#[test]
fn add_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.add_f16x8(bits(A), bits(B)), oracle(A, B, |x, y| x + y));
}

#[test]
fn sub_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.sub_f16x8(bits(A), bits(B)), oracle(A, B, |x, y| x - y));
}

#[test]
fn mul_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.mul_f16x8(bits(A), bits(B)), oracle(A, B, |x, y| x * y));
}

#[test]
fn max_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.max_f16x8(bits(A), bits(B)), oracle(A, B, f32::max));
}

#[test]
fn min_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.min_f16x8(bits(A), bits(B)), oracle(A, B, f32::min));
}

#[test]
fn abs_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.abs_f16x8(bits(A)), oracle_unop(A, f32::abs));
}

#[test]
fn neg_f16x8_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.neg_f16x8(bits(A)), oracle_unop(A, |x| -x));
}

#[test]
fn compare_family_matches_scalar() {
	let Some(t) = require() else { return };
	assert_eq!(t.cmpeq_f16x8(bits(A), bits(B)), oracle_cmp(A, B, |x, y| x == y));
	assert_eq!(t.cmpgt_f16x8(bits(A), bits(B)), oracle_cmp(A, B, |x, y| x > y));
	assert_eq!(t.cmpge_f16x8(bits(A), bits(B)), oracle_cmp(A, B, |x, y| x >= y));
	assert_eq!(t.cmplt_f16x8(bits(A), bits(B)), oracle_cmp(A, B, |x, y| x < y));
	assert_eq!(t.cmple_f16x8(bits(A), bits(B)), oracle_cmp(A, B, |x, y| x <= y));
}

#[test]
fn fma_f16x8_matches_scalar_b_times_c_plus_a() {
	let Some(t) = require_fma() else { return };
	let a = [1.0, -2.5, 0.0, 100.0, 0.5, -0.5, 2.0, -2.0];
	let b = [2.0, 3.0, 5.0, -1.0, 1.0, 1.0, 3.0, 3.0];
	let c = [3.0, -4.0, 7.0, 0.5, -1.0, 1.0, -1.0, 1.0];
	let expect = oracle_ternop(a, b, c, |x, y, z| y * z + x);
	assert_eq!(t.fma_f16x8(bits(a), bits(b), bits(c)), expect);
}

fn oracle_ternop(a: [f32; 8], b: [f32; 8], c: [f32; 8], f: impl Fn(f32, f32, f32) -> f32) -> [u16; 8] {
	core::array::from_fn(|i| {
		let (x, y, z) = (
			f16::from_f32(a[i]).to_f32(),
			f16::from_f32(b[i]).to_f32(),
			f16::from_f32(c[i]).to_f32(),
		);
		f16::from_f32(f(x, y, z)).to_bits()
	})
}
