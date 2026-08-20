//! AVX512PF: Xeon Phi prefetch-hint instructions encoded via `asm!`.
//! No Rust `target_feature` exists for this CPUID bit; detection is
//! compile-checked only. API is `unsafe` pointer-based (pure hints).

use core::arch::asm;

use super::super::super::{Feature, FeatureSet};

/// Proof token: AVX512PF available. Zero-sized, `Copy`.
///
/// `detect()` is `None` on hosts reachable by this crate's test matrix.
#[derive(Debug, Clone, Copy)]
pub struct Avx512Pf(());

impl Avx512Pf {
	/// `None` on every CPU this crate can detect on (Xeon Phi only; platform
	/// EOL).
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Avx512pf).then_some(Avx512Pf(()))
	}

	/// T0-hint gather prefetch: `f32` at `base + idx[i]*4` for every lane `i`
	/// with bit `i` set in `mask` (bits 16..=15 unused: 16 active lanes).
	///
	/// # Safety
	/// `base + idx[i]*4` must be a validly mapped, readable address for
	/// every `i` with bit `i` set in `mask`: same addressing contract as a
	/// real gather at those addresses, even though this instruction only
	/// prefetches and never materializes a value.
	#[inline]
	pub unsafe fn gatherpf0_dps(self, base: *const f32, idx: &[u32; 16], mask: u16) {
		unsafe { gatherpf0_dps_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::gatherpf0_dps`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dps`].
	#[inline]
	pub unsafe fn gatherpf1_dps(self, base: *const f32, idx: &[u32; 16], mask: u16) {
		unsafe { gatherpf1_dps_native(base.cast(), idx, mask) }
	}

	/// T0-hint scatter prefetch (prefetch-for-write): `f32` at
	/// `base + idx[i]*4` for every lane `i` with bit `i` set in `mask`.
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dps`].
	#[inline]
	pub unsafe fn scatterpf0_dps(self, base: *const f32, idx: &[u32; 16], mask: u16) {
		unsafe { scatterpf0_dps_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::scatterpf0_dps`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dps`].
	#[inline]
	pub unsafe fn scatterpf1_dps(self, base: *const f32, idx: &[u32; 16], mask: u16) {
		unsafe { scatterpf1_dps_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::gatherpf0_dps`], qword indices (8 active lanes: bits
	/// 8..=15 of `mask` unused).
	///
	/// # Safety
	/// `base + idx[i]*4` must be a validly mapped, readable address for
	/// every `i` with bit `i` set in `mask`.
	#[inline]
	pub unsafe fn gatherpf0_qps(self, base: *const f32, idx: &[u64; 8], mask: u16) {
		unsafe { gatherpf0_qps_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::gatherpf0_qps`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qps`].
	#[inline]
	pub unsafe fn gatherpf1_qps(self, base: *const f32, idx: &[u64; 8], mask: u16) {
		unsafe { gatherpf1_qps_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::scatterpf0_dps`], qword indices (8 active lanes).
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qps`].
	#[inline]
	pub unsafe fn scatterpf0_qps(self, base: *const f32, idx: &[u64; 8], mask: u16) {
		unsafe { scatterpf0_qps_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::scatterpf0_qps`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qps`].
	#[inline]
	pub unsafe fn scatterpf1_qps(self, base: *const f32, idx: &[u64; 8], mask: u16) {
		unsafe { scatterpf1_qps_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::gatherpf0_dps`], `f64` elements (8 active lanes, dword
	/// indices).
	///
	/// # Safety
	/// `base + idx[i]*8` must be a validly mapped, readable address for
	/// every `i` with bit `i` set in `mask`.
	#[inline]
	pub unsafe fn gatherpf0_dpd(self, base: *const f64, idx: &[u32; 8], mask: u16) {
		unsafe { gatherpf0_dpd_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::gatherpf0_dpd`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dpd`].
	#[inline]
	pub unsafe fn gatherpf1_dpd(self, base: *const f64, idx: &[u32; 8], mask: u16) {
		unsafe { gatherpf1_dpd_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::scatterpf0_dps`], `f64` elements (8 active lanes, dword
	/// indices).
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dpd`].
	#[inline]
	pub unsafe fn scatterpf0_dpd(self, base: *const f64, idx: &[u32; 8], mask: u16) {
		unsafe { scatterpf0_dpd_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::scatterpf0_dpd`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_dpd`].
	#[inline]
	pub unsafe fn scatterpf1_dpd(self, base: *const f64, idx: &[u32; 8], mask: u16) {
		unsafe { scatterpf1_dpd_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::gatherpf0_dps`], `f64` elements, qword indices (8 active
	/// lanes).
	///
	/// # Safety
	/// `base + idx[i]*8` must be a validly mapped, readable address for
	/// every `i` with bit `i` set in `mask`.
	#[inline]
	pub unsafe fn gatherpf0_qpd(self, base: *const f64, idx: &[u64; 8], mask: u16) {
		unsafe { gatherpf0_qpd_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::gatherpf0_qpd`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qpd`].
	#[inline]
	pub unsafe fn gatherpf1_qpd(self, base: *const f64, idx: &[u64; 8], mask: u16) {
		unsafe { gatherpf1_qpd_native(base.cast(), idx, mask) }
	}

	/// [`Avx512Pf::scatterpf0_dps`], `f64` elements, qword indices (8 active
	/// lanes).
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qpd`].
	#[inline]
	pub unsafe fn scatterpf0_qpd(self, base: *const f64, idx: &[u64; 8], mask: u16) {
		unsafe { scatterpf0_qpd_native(base.cast(), idx, mask) }
	}

	/// T1-hint variant of [`Avx512Pf::scatterpf0_qpd`].
	///
	/// # Safety
	/// Same as [`Avx512Pf::gatherpf0_qpd`].
	#[inline]
	pub unsafe fn scatterpf1_qpd(self, base: *const f64, idx: &[u64; 8], mask: u16) {
		unsafe { scatterpf1_qpd_native(base.cast(), idx, mask) }
	}
}

/// Emits one `<native fn>(base: *const u8, idx: &[$idx_elem; $idx_len],
/// mask: u16)` per invocation. `$idx_reg` is the vector register holding the
/// VSIB index (`zmm1` for 16 dword/8 qword indices, `ymm1` for the 8-dword
/// `DPD` case where only 8 lanes are active); `$load` is the matching
/// dword/qword unaligned-move mnemonic; `$scale` is the element size in
/// bytes (also the VSIB scale factor). Written in AT&T syntax
/// (`options(att_syntax)`) because LLVM's integrated Intel-syntax parser
/// rejects the `{k1}`-masked VSIB memory operand these instructions use
/// (confirmed against `llvm-mc` directly: AT&T syntax for the same bytes
/// parses and round-trips through `objdump` cleanly). `k1`/`$idx_reg` are
/// hardcoded register names, not asm! register-class operands, matching the
/// `zmm0`/`zmm4..7` hardcoding already used in `avx512_4fmaps`/
/// `avx512_4vnniw`: `kmovw` (not `kmovb`) is used for every mask width
/// since `kmovb` needs AVX512DQ, which real Knights Landing/Knights Mill
/// silicon never has.
macro_rules! prefetch_op {
	($name:ident, $mnemonic:literal, $load:literal, $idx_reg:literal, $scale:literal, $idx_elem:ty, $idx_len:literal) => {
		/// # Safety
		/// Caller proved AVX512PF via [`Avx512Pf`]. `target_feature =
		/// "avx512f"` only covers the register classes the `asm!` block
		/// uses: there is no Rust-recognized feature string for the
		/// `avx512pf` CPUID bit itself, so [`Avx512Pf::detect`] plus the
		/// VSIB addressing contract documented on the calling public method
		/// are the sole real gates.
		#[target_feature(enable = "avx512f")]
		unsafe fn $name(base: *const u8, idx: &[$idx_elem; $idx_len], mask: u16) {
			unsafe {
				asm!(
					concat!($load, " ({idx}), %", $idx_reg),
					"kmovw {mask:e}, %k1",
					concat!($mnemonic, " ({base},%", $idx_reg, ",", $scale, "){{%k1}}"),
					idx = in(reg) idx.as_ptr(),
					mask = in(reg) mask as u32,
					base = in(reg) base,
					out("zmm1") _,
					out("k1") _,
					options(att_syntax, nostack),
				);
			}
		}
	};
}

prefetch_op!(gatherpf0_dps_native, "vgatherpf0dps", "vmovdqu32", "zmm1", 4, u32, 16);
prefetch_op!(gatherpf1_dps_native, "vgatherpf1dps", "vmovdqu32", "zmm1", 4, u32, 16);
prefetch_op!(scatterpf0_dps_native, "vscatterpf0dps", "vmovdqu32", "zmm1", 4, u32, 16);
prefetch_op!(scatterpf1_dps_native, "vscatterpf1dps", "vmovdqu32", "zmm1", 4, u32, 16);
prefetch_op!(gatherpf0_qps_native, "vgatherpf0qps", "vmovdqu64", "zmm1", 4, u64, 8);
prefetch_op!(gatherpf1_qps_native, "vgatherpf1qps", "vmovdqu64", "zmm1", 4, u64, 8);
prefetch_op!(scatterpf0_qps_native, "vscatterpf0qps", "vmovdqu64", "zmm1", 4, u64, 8);
prefetch_op!(scatterpf1_qps_native, "vscatterpf1qps", "vmovdqu64", "zmm1", 4, u64, 8);
prefetch_op!(gatherpf0_dpd_native, "vgatherpf0dpd", "vmovdqu32", "ymm1", 8, u32, 8);
prefetch_op!(gatherpf1_dpd_native, "vgatherpf1dpd", "vmovdqu32", "ymm1", 8, u32, 8);
prefetch_op!(scatterpf0_dpd_native, "vscatterpf0dpd", "vmovdqu32", "ymm1", 8, u32, 8);
prefetch_op!(scatterpf1_dpd_native, "vscatterpf1dpd", "vmovdqu32", "ymm1", 8, u32, 8);
prefetch_op!(gatherpf0_qpd_native, "vgatherpf0qpd", "vmovdqu64", "zmm1", 8, u64, 8);
prefetch_op!(gatherpf1_qpd_native, "vgatherpf1qpd", "vmovdqu64", "zmm1", 8, u64, 8);
prefetch_op!(scatterpf0_qpd_native, "vscatterpf0qpd", "vmovdqu64", "zmm1", 8, u64, 8);
prefetch_op!(scatterpf1_qpd_native, "vscatterpf1qpd", "vmovdqu64", "zmm1", 8, u64, 8);

#[cfg(test)]
#[path = "../../test/ops/avx512/avx512pf.rs"]
mod tests;
