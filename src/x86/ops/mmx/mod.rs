//! MMX (1997): 8-byte SIMD via `asm!` (no `core::arch`/`target_feature`). Each op ends with `emms`.
//! Token: [`Mmx::detect`]. Fixed-width only (no slice: per-call `movq`+`emms`).
//! Not on the x86_64 auto ladder (SSE2 is the ABI floor); standalone for completeness / 32-bit floor.

use core::arch::asm;

use super::super::{Feature, FeatureSet};

/// Proof token: MMX available. Zero-sized, `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct Mmx(());

impl Mmx {
	/// `None` if the CPU (or the compile-time target) lacks MMX.
	pub fn detect() -> Option<Self> {
		Self::from_features(FeatureSet::detect())
	}

	/// From a raw feature bitset (e.g. `x86::detect_features()`).
	pub fn from_features(set: FeatureSet) -> Option<Self> {
		set.contains(Feature::Mmx).then_some(Mmx(()))
	}
}

/// Binop over one MMX mnemonic: `impl Mmx` method (fixed-width only) plus
/// its `asm!` helper (`movq` load x2, op, `movq` store, `emms`).
macro_rules! mmx_binop_asm {
	($fixed_fn:ident, $asm_fn:ident, $mnemonic:literal, $Elem:ty, $width:literal, $fixed_doc:literal) => {
		impl Mmx {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn(self, a: [$Elem; $width], b: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $asm_fn(&a, &b) }
			}
		}

		/// # Safety
		/// Caller proved MMX via [`Mmx`].
		#[inline]
		unsafe fn $asm_fn(a: &[$Elem; $width], b: &[$Elem; $width]) -> [$Elem; $width] {
			let mut out = [Default::default(); $width];
			unsafe {
				asm!(
					"movq mm0, [{a}]",
					concat!($mnemonic, " mm0, [{b}]"),
					"movq [{out}], mm0",
					"emms",
					a = in(reg) a.as_ptr(),
					b = in(reg) b.as_ptr(),
					out = in(reg) out.as_mut_ptr(),
					out("mm0") _,
				);
			}
			out
		}
	};
}

/// Const-imm shift over one MMX mnemonic (real MMX immediate-shift form,
/// not a register-count trick): `impl Mmx` method plus its `asm!` helper.
macro_rules! mmx_shift_imm_asm {
	($fixed_fn:ident, $asm_fn:ident, $mnemonic:literal, $Elem:ty, $width:literal, $fixed_doc:literal) => {
		impl Mmx {
			#[doc = $fixed_doc]
			#[inline]
			pub fn $fixed_fn<const IMM: u32>(self, a: [$Elem; $width]) -> [$Elem; $width] {
				unsafe { $asm_fn::<IMM>(&a) }
			}
		}

		/// # Safety
		/// Caller proved MMX via [`Mmx`].
		#[inline]
		unsafe fn $asm_fn<const IMM: u32>(a: &[$Elem; $width]) -> [$Elem; $width] {
			let mut out = [Default::default(); $width];
			unsafe {
				asm!(
					"movq mm0, [{a}]",
					concat!($mnemonic, " mm0, {imm}"),
					"movq [{out}], mm0",
					"emms",
					a = in(reg) a.as_ptr(),
					imm = const IMM,
					out = in(reg) out.as_mut_ptr(),
					out("mm0") _,
				);
			}
			out
		}
	};
}

// Byte (x8) family.
mmx_binop_asm!(add_i8x8, paddb_i, "paddb", i8, 8, "`a + b` per lane, wrapping mod 256 (`paddb`, no saturation).");
mmx_binop_asm!(sub_i8x8, psubb_i, "psubb", i8, 8, "`a - b` per lane, wrapping mod 256 (`psubb`, no saturation).");
mmx_binop_asm!(add_u8x8, paddb_u, "paddb", u8, 8, "`a + b` per lane, wrapping mod 256 (`paddb`, no saturation).");
mmx_binop_asm!(sub_u8x8, psubb_u, "psubb", u8, 8, "`a - b` per lane, wrapping mod 256 (`psubb`, no saturation).");
mmx_binop_asm!(adds_i8x8, paddsb, "paddsb", i8, 8, "`a + b` per lane, saturating (`paddsb`).");
mmx_binop_asm!(subs_i8x8, psubsb, "psubsb", i8, 8, "`a - b` per lane, saturating (`psubsb`).");
mmx_binop_asm!(adds_u8x8, paddusb, "paddusb", u8, 8, "`a + b` per lane, saturating (`paddusb`).");
mmx_binop_asm!(subs_u8x8, psubusb, "psubusb", u8, 8, "`a - b` per lane, saturating (`psubusb`).");
mmx_binop_asm!(cmpeq_i8x8, pcmpeqb_i, "pcmpeqb", i8, 8, "Lane equality mask (`pcmpeqb`): all-1s if equal, else 0.");
mmx_binop_asm!(cmpeq_u8x8, pcmpeqb_u, "pcmpeqb", u8, 8, "Lane equality mask (`pcmpeqb`): all-1s if equal, else 0.");
mmx_binop_asm!(and_u8x8, pand_mmx, "pand", u8, 8, "`a & b` per lane (`pand`, whole 64-bit register).");
mmx_binop_asm!(or_u8x8, por_mmx, "por", u8, 8, "`a | b` per lane (`por`, whole 64-bit register).");
mmx_binop_asm!(xor_u8x8, pxor_mmx, "pxor", u8, 8, "`a ^ b` per lane (`pxor`, whole 64-bit register).");
mmx_binop_asm!(andnot_u8x8, pandn_mmx, "pandn", u8, 8, "`!a & b` per lane (`pandn`, whole 64-bit register).");

// Word (x4) family.
mmx_binop_asm!(add_i16x4, paddw_i, "paddw", i16, 4, "`a + b` per lane, wrapping (`paddw`, no saturation).");
mmx_binop_asm!(sub_i16x4, psubw_i, "psubw", i16, 4, "`a - b` per lane, wrapping (`psubw`, no saturation).");
mmx_binop_asm!(add_u16x4, paddw_u, "paddw", u16, 4, "`a + b` per lane, wrapping (`paddw`, no saturation).");
mmx_binop_asm!(sub_u16x4, psubw_u, "psubw", u16, 4, "`a - b` per lane, wrapping (`psubw`, no saturation).");
mmx_binop_asm!(adds_i16x4, paddsw, "paddsw", i16, 4, "`a + b` per lane, saturating (`paddsw`).");
mmx_binop_asm!(subs_i16x4, psubsw, "psubsw", i16, 4, "`a - b` per lane, saturating (`psubsw`).");
mmx_binop_asm!(adds_u16x4, paddusw, "paddusw", u16, 4, "`a + b` per lane, saturating (`paddusw`).");
mmx_binop_asm!(subs_u16x4, psubusw, "psubusw", u16, 4, "`a - b` per lane, saturating (`psubusw`).");
mmx_binop_asm!(cmpeq_i16x4, pcmpeqw_i, "pcmpeqw", i16, 4, "Lane equality mask (`pcmpeqw`): all-1s if equal, else 0.");
mmx_binop_asm!(cmpeq_u16x4, pcmpeqw_u, "pcmpeqw", u16, 4, "Lane equality mask (`pcmpeqw`): all-1s if equal, else 0.");
mmx_binop_asm!(mullo_i16x4, pmullw_i, "pmullw", i16, 4, "`a * b` per lane, low 16 bits (`pmullw`).");
mmx_binop_asm!(mullo_u16x4, pmullw_u, "pmullw", u16, 4, "`a * b` per lane, low 16 bits (`pmullw`).");
mmx_binop_asm!(mulhi_i16x4, pmulhw, "pmulhw", i16, 4, "`a * b` per lane, high 16 bits, signed (`pmulhw`).");

// Dword (x2) family.
mmx_binop_asm!(add_i32x2, paddd_i, "paddd", i32, 2, "`a + b` per lane, wrapping (`paddd`, no saturation).");
mmx_binop_asm!(sub_i32x2, psubd_i, "psubd", i32, 2, "`a - b` per lane, wrapping (`psubd`, no saturation).");
mmx_binop_asm!(add_u32x2, paddd_u, "paddd", u32, 2, "`a + b` per lane, wrapping (`paddd`, no saturation).");
mmx_binop_asm!(sub_u32x2, psubd_u, "psubd", u32, 2, "`a - b` per lane, wrapping (`psubd`, no saturation).");
mmx_binop_asm!(cmpeq_i32x2, pcmpeqd_i, "pcmpeqd", i32, 2, "Lane equality mask (`pcmpeqd`): all-1s if equal, else 0.");
mmx_binop_asm!(cmpeq_u32x2, pcmpeqd_u, "pcmpeqd", u32, 2, "Lane equality mask (`pcmpeqd`): all-1s if equal, else 0.");

// Shifts: word, then dword (register-count-immediate via const IMM).
mmx_shift_imm_asm!(shl_i16x4, psllw_i, "psllw", i16, 4, "`a << IMM` per lane (`psllw`, immediate count).");
mmx_shift_imm_asm!(shr_i16x4, psrlw_i, "psrlw", i16, 4, "`a >> IMM` logical per lane (`psrlw`, immediate count).");
mmx_shift_imm_asm!(sra_i16x4, psraw_i, "psraw", i16, 4, "`a >> IMM` arithmetic per lane (`psraw`, immediate count).");
mmx_shift_imm_asm!(shl_u16x4, psllw_u, "psllw", u16, 4, "`a << IMM` per lane (`psllw`, immediate count).");
mmx_shift_imm_asm!(shr_u16x4, psrlw_u, "psrlw", u16, 4, "`a >> IMM` per lane (`psrlw`, immediate count).");

mmx_shift_imm_asm!(shl_i32x2, pslld_i, "pslld", i32, 2, "`a << IMM` per lane (`pslld`, immediate count).");
mmx_shift_imm_asm!(shr_i32x2, psrld_i, "psrld", i32, 2, "`a >> IMM` logical per lane (`psrld`, immediate count).");
mmx_shift_imm_asm!(sra_i32x2, psrad_i, "psrad", i32, 2, "`a >> IMM` arithmetic per lane (`psrad`, immediate count).");
mmx_shift_imm_asm!(shl_u32x2, pslld_u, "pslld", u32, 2, "`a << IMM` per lane (`pslld`, immediate count).");
mmx_shift_imm_asm!(shr_u32x2, psrld_u, "psrld", u32, 2, "`a >> IMM` per lane (`psrld`, immediate count).");

impl Mmx {
	/// Multiply pairs of 16-bit lanes, sum each pair into one 32-bit lane
	/// (`pmaddwd`): `[out[0] = a[0]*b[0] + a[1]*b[1], out[1] = a[2]*b[2] +
	/// a[3]*b[3]]`. Width-converting (4 lanes in, 2 lanes out).
	#[inline]
	pub fn maddwd_i16x4(self, a: [i16; 4], b: [i16; 4]) -> [i32; 2] {
		unsafe { pmaddwd(&a, &b) }
	}
}

/// # Safety
/// Caller proved MMX via [`Mmx`].
#[inline]
unsafe fn pmaddwd(a: &[i16; 4], b: &[i16; 4]) -> [i32; 2] {
	let mut out = [0i32; 2];
	unsafe {
		asm!(
			"movq mm0, [{a}]",
			"pmaddwd mm0, [{b}]",
			"movq [{out}], mm0",
			"emms",
			a = in(reg) a.as_ptr(),
			b = in(reg) b.as_ptr(),
			out = in(reg) out.as_mut_ptr(),
			out("mm0") _,
		);
	}
	out
}

#[cfg(test)]
#[path = "../../test/ops/mmx/mmx_mod.rs"]
mod tests;
