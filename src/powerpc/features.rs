//! powerpc64 Feature enum (LE first: powerpc64le). Auxv for AltiVec/VSX/POWER8/9;
//! bare/other compile-time cfg. Masks from UAPI `asm/cputable.h`.

/// PPC_FEATURE / PPC_FEATURE2 bit masks (not bit indices).
#[cfg(any(
	target_os = "linux",
	target_os = "android",
	target_os = "freebsd"
))]
mod hwcap {
	// AT_HWCAP
	pub const HAS_ALTIVEC: u64 = 0x1000_0000;
	pub const HAS_VSX: u64 = 0x0000_0080;
	// AT_HWCAP2
	pub const ARCH_2_07: u64 = 0x8000_0000; // POWER8-class
	pub const ARCH_3_00: u64 = 0x0080_0000; // POWER9-class
	pub const VEC_CRYPTO: u64 = 0x0200_0000;
}

/// Enum list for Power ISA vector / generation markers (detect subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
	/// AltiVec / VMX.
	Altivec,
	/// VSX (implies AltiVec on real silicon).
	Vsx,
	/// POWER8 / v2.07 user arch (and related vector crypto group).
	Power8,
	/// POWER8 AltiVec extensions (tied to ARCH_2_07 in probe).
	Power8Altivec,
	/// POWER8 vector (tied to ARCH_2_07).
	Power8Vector,
	/// POWER8 crypto (VEC_CRYPTO when present, else ARCH_2_07).
	Power8Crypto,
	/// POWER9 / v3.0 user arch.
	Power9,
	/// POWER9 AltiVec (tied to ARCH_3_00).
	Power9Altivec,
	/// POWER9 vector (tied to ARCH_3_00).
	Power9Vector,
}

impl Feature {
	/// Index = bit in [`FeatureSet`].
	pub const ALL: &'static [Feature] = &[
		Feature::Altivec,
		Feature::Vsx,
		Feature::Power8,
		Feature::Power8Altivec,
		Feature::Power8Vector,
		Feature::Power8Crypto,
		Feature::Power9,
		Feature::Power9Altivec,
		Feature::Power9Vector,
	];

	/// Bit index in [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
	hwcap: u64,
	hwcap2: u64,
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
	fn read() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android"))]
		// SAFETY: read-only auxv; tags are libc consts.
		unsafe {
			Self {
				hwcap: libc::getauxval(libc::AT_HWCAP) as u64,
				hwcap2: libc::getauxval(libc::AT_HWCAP2) as u64,
			}
		}
		#[cfg(target_os = "freebsd")]
		{
			fn one(key: libc::c_int) -> u64 {
				let mut out: libc::c_ulong = 0;
				// SAFETY: elf_aux_info writes size_of out; FreeBSD 12+.
				let _ = unsafe {
					libc::elf_aux_info(
						key,
						(&mut out) as *mut libc::c_ulong as *mut libc::c_void,
						core::mem::size_of::<libc::c_ulong>() as libc::c_int,
					)
				};
				out as u64
			}
			Self {
				hwcap: one(libc::AT_HWCAP),
				hwcap2: one(libc::AT_HWCAP2),
			}
		}
	}

	fn supports(&self, feature: Feature) -> bool {
		let p8 = self.hwcap2 & hwcap::ARCH_2_07 != 0;
		let p9 = self.hwcap2 & hwcap::ARCH_3_00 != 0;
		let crypto = self.hwcap2 & hwcap::VEC_CRYPTO != 0 || p8;
		match feature {
			Feature::Altivec => self.hwcap & hwcap::HAS_ALTIVEC != 0,
			Feature::Vsx => self.hwcap & hwcap::HAS_VSX != 0,
			Feature::Power8 => p8,
			Feature::Power8Altivec => p8,
			Feature::Power8Vector => p8,
			Feature::Power8Crypto => crypto,
			Feature::Power9 => p9,
			Feature::Power9Altivec => p9,
			Feature::Power9Vector => p9,
		}
	}
}

/// Bitset: one bit per [`Feature::ALL`] entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Host via auxv (Linux/Android/FreeBSD) or compile-time cfg (else).
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
			Self(set)
		}

		#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
		{
			#[allow(unused_mut, reason = "mutated under target_feature cfgs")]
			let mut set = 0u128;
			#[cfg(target_feature = "altivec")]
			{
				set |= 1 << Feature::Altivec.bit();
			}
			#[cfg(target_feature = "vsx")]
			{
				set |= 1 << Feature::Vsx.bit();
				set |= 1 << Feature::Altivec.bit();
			}
			// power8 / power9 target features are unstable; leave to shortpath later.
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
