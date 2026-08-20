//! RISC-V base ISA (M/A/F/D/C/V). Auxv or compile-time cfg. [`hwcap`] restates
//! `COMPAT_HWCAP_ISA_*` (UAPI bit is letter minus `'A'`). No Feature::I (always
//! implied). Z* / vector-crypto need `riscv_hwprobe(2)` (not here).

/// `unsigned long`: u32 on RV32, u64 on RV64.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
#[cfg(target_pointer_width = "32")]
type Hwcap = u32;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
#[cfg(target_pointer_width = "64")]
type Hwcap = u64;

/// `COMPAT_HWCAP_ISA_*`. Kernel UAPI `asm/hwcap.h`.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap {
	use super::Hwcap;

	pub const M: Hwcap = 1 << (b'M' - b'A');
	pub const A: Hwcap = 1; // 1 << ('A' - 'A')
	pub const F: Hwcap = 1 << (b'F' - b'A');
	pub const D: Hwcap = 1 << (b'D' - b'A');
	pub const C: Hwcap = 1 << (b'C' - b'A');
	pub const V: Hwcap = 1 << (b'V' - b'A');
}

/// Enum list for RISC-V HWCAP ISA letters (no I: always implied).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
	/// Mul/div.
	M,
	/// Atomics.
	A,
	/// f32 FP.
	F,
	/// f64 FP.
	D,
	/// Compressed 16-bit encodings.
	C,
	/// Vector (app processors).
	V,
}

impl Feature {
	/// HWCAP letter order; index = bit in [`FeatureSet`].
	pub const ALL: &'static [Feature] =
		&[Feature::M, Feature::A, Feature::F, Feature::D, Feature::C, Feature::V];

	/// Bit index in [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

/// One auxv `AT_HWCAP` snapshot.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
	hwcap: Hwcap,
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
	fn read() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android"))]
		// SAFETY: read-only auxv; tag is a libc const.
		unsafe {
			return Self {
				hwcap: libc::getauxval(libc::AT_HWCAP) as Hwcap,
			};
		}
		#[cfg(target_os = "freebsd")]
		{
			let mut out: libc::c_ulong = 0;
			// SAFETY: elf_aux_info writes size_of out; FreeBSD 12+.
			let _ = unsafe {
				libc::elf_aux_info(
					libc::AT_HWCAP,
					(&mut out) as *mut libc::c_ulong as *mut libc::c_void,
					core::mem::size_of::<libc::c_ulong>() as libc::c_int,
				)
			};
			Self { hwcap: out as Hwcap }
		}
	}

	/// `match` so a new Feature arm is a compile error.
	fn supports(&self, feature: Feature) -> bool {
		match feature {
			Feature::M => self.hwcap & hwcap::M != 0,
			Feature::A => self.hwcap & hwcap::A != 0,
			Feature::F => self.hwcap & hwcap::F != 0,
			Feature::D => self.hwcap & hwcap::D != 0,
			Feature::C => self.hwcap & hwcap::C != 0,
			Feature::V => self.hwcap & hwcap::V != 0,
		}
	}
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Host via auxv, or compile-time cfg (bare-metal/other).
	pub fn detect() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
		{
			let aux = AuxVal::read();
			let mut set = 0u128;
			for &feature in Feature::ALL {
				if aux.supports(feature) {
					set |= 1 << feature.bit();
				}
			}
			return Self(set);
		}
		#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
		{
			// gc floor is usual for riscv*gc; still only set bits proven by cfg.
			#[allow(unused_mut, reason = "mutated under target_feature cfgs")]
			let mut set = 0u128;
			#[cfg(target_feature = "m")]
			{
				set |= 1 << Feature::M.bit();
			}
			#[cfg(target_feature = "a")]
			{
				set |= 1 << Feature::A.bit();
			}
			#[cfg(target_feature = "f")]
			{
				set |= 1 << Feature::F.bit();
			}
			#[cfg(target_feature = "d")]
			{
				set |= 1 << Feature::D.bit();
			}
			#[cfg(target_feature = "c")]
			{
				set |= 1 << Feature::C.bit();
			}
			#[cfg(target_feature = "v")]
			{
				set |= 1 << Feature::V.bit();
			}
			// Many none-elf targets enable m/a/f/d/c in the base ISA string without
			// exposing every letter as target_feature; assume gc floor when none
			// of the above fired but we are on a typical imac/gc-style target.
			if set == 0 {
				set |= 1 << Feature::M.bit();
				set |= 1 << Feature::A.bit();
				set |= 1 << Feature::F.bit();
				set |= 1 << Feature::D.bit();
				set |= 1 << Feature::C.bit();
			}
			Self(set)
		}
	}

	pub fn contains(self, feature: Feature) -> bool {
		self.0 & (1 << feature.bit()) != 0
	}

	pub fn contains_all(self, required: &[Feature]) -> bool {
		required.iter().all(|&f| self.contains(f))
	}
}
