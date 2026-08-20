//! Test-only helpers: slice-vs-scalar oracle checks shared across
//! binop/ternop/shift/imm-binop tier tests.

/// Test helper: token `detect()`, then slice vs scalar over several lengths.
#[cfg(test)]
macro_rules! slice_binop_matches_scalar_test {
	($test_name:ident, $Token:ty, $slice_method:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method(&a, &b, &mut out);
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

/// Test helper for the experimental wider-bus-lift `_lifted` methods:
/// same shape as [`slice_binop_matches_scalar_test`], but needs both the
/// op's own token *and* the lift-target proof token (`_lifted` methods take
/// both: see `ops/macros/binop.rs`'s `simd_binop_lifted!` doc for why).
/// Skips silently unless the host has both.
#[cfg(all(test, feature = "wider-bus-lift"))]
macro_rules! slice_binop_lifted_matches_scalar_test {
	($test_name:ident, $Token:ty, $LiftProof:ty, $slice_method:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			let Some(proof) = <$LiftProof>::detect() else {
				return;
			};
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method(proof, &a, &b, &mut out);
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

/// Bit-exact variant of [`slice_binop_matches_scalar_test`] for float bitwise ops: arbitrary
/// floats AND/OR/XORed bitwise routinely land in the NaN exponent range, and NaN != NaN under
/// `PartialEq` even when the bit patterns (what a bitwise op actually guarantees) are identical.
#[cfg(test)]
macro_rules! slice_bitop_matches_scalar_bits_test {
	($test_name:ident, $Token:ty, $slice_method:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method(&a, &b, &mut out);
				let op: fn($Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y)).collect();
				let out_bits: Vec<_> = out.iter().map(|x| x.to_bits()).collect();
				let expect_bits: Vec<_> = expect.iter().map(|x| x.to_bits()).collect();
				assert_eq!(out_bits, expect_bits, "len={len}");
			}
		}
	};
}

/// Test helper: ternary slice vs scalar (FMA-shaped).
#[cfg(test)]
macro_rules! slice_ternop_matches_scalar_test {
	($test_name:ident, $Token:ty, $slice_method:ident, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| i as $Elem + 1 as $Elem).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem + 1 as $Elem).collect();
				let c: Vec<$Elem> = (0..len).map(|i| i as $Elem * 0 as $Elem + 2 as $Elem).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method(&a, &b, &c, &mut out);
				let op: fn($Elem, $Elem, $Elem) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a
					.iter()
					.zip(&b)
					.zip(&c)
					.map(|((&x, &y), &z)| op(x, y, z))
					.collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

/// Test helper: const-imm shift slice vs scalar (IMM fixed at call site).
#[cfg(test)]
macro_rules! slice_shift_imm_matches_scalar_test {
	($test_name:ident, $Token:ty, $slice_method:ident, $imm:expr, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			const IMM: u32 = $imm;
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| (i as $Elem).wrapping_mul(3 as $Elem).wrapping_add(1 as $Elem)).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method::<IMM>(&a, &mut out);
				let op: fn($Elem, u32) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().map(|&x| op(x, IMM)).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

/// Test helper: const-imm 2-op binop slice vs scalar (IMM fixed at call site).
#[cfg(test)]
macro_rules! slice_binop_imm_matches_scalar_test {
	($test_name:ident, $Token:ty, $slice_method:ident, $imm:expr, $scalar:expr, $Elem:ty) => {
		#[test]
		fn $test_name() {
			let Some(t) = <$Token>::detect() else {
				return;
			};
			const IMM8: i32 = $imm;
			for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100] {
				let a: Vec<$Elem> = (0..len).map(|i| (i as $Elem).wrapping_mul(3 as $Elem).wrapping_add(1 as $Elem)).collect();
				let b: Vec<$Elem> = (0..len).map(|i| (len - i) as $Elem).collect();
				let mut out = vec![Default::default(); len];
				t.$slice_method::<IMM8>(&a, &b, &mut out);
				let op: fn($Elem, $Elem, i32) -> $Elem = $scalar;
				let expect: Vec<$Elem> = a.iter().zip(&b).map(|(&x, &y)| op(x, y, IMM8)).collect();
				assert_eq!(out, expect, "len={len}");
			}
		}
	};
}

#[cfg(test)]
pub(crate) use slice_binop_imm_matches_scalar_test;
#[cfg(test)]
pub(crate) use slice_binop_matches_scalar_test;
#[cfg(all(test, feature = "wider-bus-lift"))]
pub(crate) use slice_binop_lifted_matches_scalar_test;
#[cfg(test)]
pub(crate) use slice_bitop_matches_scalar_bits_test;
#[cfg(test)]
pub(crate) use slice_shift_imm_matches_scalar_test;
#[cfg(test)]
pub(crate) use slice_ternop_matches_scalar_test;
