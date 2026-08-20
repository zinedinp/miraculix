//! LoongArch64 Feature enum. Linux `getauxval(AT_HWCAP)` for LSX/LASX/FPU;
//! bare/other compile-time `cfg(target_feature)`. UAPI `asm/hwcap.h` bits.

/// `HWCAP_LOONGARCH_*`. Kernel UAPI `asm/hwcap.h`.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod hwcap {
	pub const FPU: u64 = 1 << 3;
	pub const LSX: u64 = 1 << 4;
	pub const LASX: u64 = 1 << 5;
	pub const CRC32: u64 = 1 << 6;
	pub const COMPLEX: u64 = 1 << 7;
	pub const CRYPTO: u64 = 1 << 8;
	pub const LVZ: u64 = 1 << 9;
	pub const UAL: u64 = 1 << 2;
}

/// Enum list for LoongArch SIMD / FP extensions (detect subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
	/// Scalar FPU (f/d class; HWCAP FPU bit).
	Fpu,
	/// Unaligned access (UAL).
	Ual,
	/// LSX 128-bit SIMD.
	Lsx,
	/// LASX 256-bit SIMD (implies LSX on hardware).
	Lasx,
	/// CRC32 instructions.
	Crc32,
	/// Complex number ops.
	Complex,
	/// Crypto instructions.
	Crypto,
	/// Virtualization (LVZ).
	Lvz,
}

impl Feature {
	/// Index = bit in [`FeatureSet`].
	pub const ALL: &'static [Feature] = &[
		Feature::Fpu,
		Feature::Ual,
		Feature::Lsx,
		Feature::Lasx,
		Feature::Crc32,
		Feature::Complex,
		Feature::Crypto,
		Feature::Lvz,
	];

	/// Bit index in [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

#[cfg(any(target_os = "linux", target_os = "android"))]
struct AuxVal {
	hwcap: u64,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl AuxVal {
	fn read() -> Self {
		// SAFETY: read-only auxv; tag is a libc const.
		unsafe { Self { hwcap: libc::getauxval(libc::AT_HWCAP) as u64 } }
	}

	fn supports(&self, feature: Feature) -> bool {
		match feature {
			Feature::Fpu => self.hwcap & hwcap::FPU != 0,
			Feature::Ual => self.hwcap & hwcap::UAL != 0,
			Feature::Lsx => self.hwcap & hwcap::LSX != 0,
			Feature::Lasx => self.hwcap & hwcap::LASX != 0,
			Feature::Crc32 => self.hwcap & hwcap::CRC32 != 0,
			Feature::Complex => self.hwcap & hwcap::COMPLEX != 0,
			Feature::Crypto => self.hwcap & hwcap::CRYPTO != 0,
			Feature::Lvz => self.hwcap & hwcap::LVZ != 0,
		}
	}
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Host via `AT_HWCAP` (Linux/Android) or compile-time cfg (else).
	pub fn detect() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android"))]
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

		#[cfg(not(any(target_os = "linux", target_os = "android")))]
		{
			// Fail-closed compile-time only (bare-metal, other OS).
			#[allow(unused_mut, reason = "mutated under target_feature cfgs")]
			let mut set = 0u128;
			#[cfg(target_feature = "f")]
			{
				set |= 1 << Feature::Fpu.bit();
			}
			#[cfg(target_feature = "lsx")]
			{
				set |= 1 << Feature::Lsx.bit();
			}
			#[cfg(target_feature = "lasx")]
			{
				set |= 1 << Feature::Lasx.bit();
				set |= 1 << Feature::Lsx.bit();
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
