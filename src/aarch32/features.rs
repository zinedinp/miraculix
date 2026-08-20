//! AArch32 VFP/NEON + crypto Feature enum. Auxv (Linux/Android/FreeBSD);
//! compile-time cfg elsewhere. FreeBSD arm HWCAP differs for some bits;
//! NEON/AES path matches. UAPI `arch/arm/include/uapi/asm/hwcap.h`. No SVE.

/// `HWCAP_*`. UAPI `asm/hwcap.h` (not in libc for AArch32).
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap {
	pub const SWP: u32 = 1 << 0;
	pub const HALF: u32 = 1 << 1;
	pub const THUMB: u32 = 1 << 2;
	pub const BIT26: u32 = 1 << 3;
	pub const FAST_MULT: u32 = 1 << 4;
	pub const FPA: u32 = 1 << 5;
	pub const VFP: u32 = 1 << 6;
	pub const EDSP: u32 = 1 << 7;
	pub const JAVA: u32 = 1 << 8;
	pub const IWMMXT: u32 = 1 << 9;
	pub const CRUNCH: u32 = 1 << 10;
	pub const THUMBEE: u32 = 1 << 11;
	pub const NEON: u32 = 1 << 12;
	pub const VFPV3: u32 = 1 << 13;
	pub const VFPV3D16: u32 = 1 << 14;
	pub const TLS: u32 = 1 << 15;
	pub const VFPV4: u32 = 1 << 16;
	pub const IDIVA: u32 = 1 << 17;
	pub const IDIVT: u32 = 1 << 18;
	pub const VFPD32: u32 = 1 << 19;
	pub const LPAE: u32 = 1 << 20;
	pub const EVTSTRM: u32 = 1 << 21;
	pub const FPHP: u32 = 1 << 22;
	pub const ASIMDHP: u32 = 1 << 23;
	pub const ASIMDDP: u32 = 1 << 24;
	pub const ASIMDFHM: u32 = 1 << 25;
	pub const ASIMDBF16: u32 = 1 << 26;
	pub const I8MM: u32 = 1 << 27;
}

/// `HWCAP2_*`. UAPI `asm/hwcap.h` (not in libc for AArch32).
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap2 {
	pub const AES: u32 = 1 << 0;
	pub const PMULL: u32 = 1 << 1;
	pub const SHA1: u32 = 1 << 2;
	pub const SHA2: u32 = 1 << 3;
	pub const CRC32: u32 = 1 << 4;
	pub const SB: u32 = 1 << 5;
	pub const SSBS: u32 = 1 << 6;
}

/// Enum list for AArch32 HWCAP/HWCAP2 extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[rustfmt::skip]
pub enum Feature {
	// AT_HWCAP
	/// Atomic `SWP`/`SWPB`.
	Swp,
	/// Halfword loads/stores.
	Half,
	Thumb,
	/// 26-bit addressing (pre-ARMv3).
	Bit26,
	FastMult,
	/// FPA10 (pre-VFP, obsolete).
	Fpa,
	/// VFPv2.
	Vfp,
	/// Thumb-EDSP.
	Edsp,
	/// Jazelle (obsolete).
	Java,
	/// Intel Wireless MMX (XScale, obsolete).
	Iwmmxt,
	/// MaverickCrunch (EP93xx, obsolete).
	Crunch,
	/// ThumbEE (obsolete).
	Thumbee,
	/// Advanced SIMD (NEON).
	Neon,
	Vfpv3,
	/// VFPv3 with 16 D-regs only (no `D32`).
	Vfpv3D16,
	/// TLS via `CP15` `c13`.
	Tls,
	Vfpv4,
	/// HW `SDIV`/`UDIV` (ARM).
	Idiva,
	/// HW `SDIV`/`UDIV` (Thumb).
	Idivt,
	/// 32 D-regs (`D0`-`D31`).
	Vfpd32,
	/// Large Physical Address Extension.
	Lpae,
	/// Generic event stream.
	Evtstrm,
	/// Scalar half-precision arith.
	Fp16,
	/// NEON half-precision arith.
	AsimdHp,
	/// Dot product.
	Dotprod,
	/// NEON half-precision FMA.
	Fhm,
	/// NEON BF16.
	AsimdBf16,
	/// NEON i8 matrix mul.
	I8mm,

	// AT_HWCAP2
	Aes,
	/// `PMULL`/`PMULL2`, paired with AES.
	Pmull,
	Sha1,
	Sha2,
	Crc32,
	/// Speculation Barrier.
	Sb,
	/// Speculative Store Bypass Safe.
	Ssbs,
}

impl Feature {
	/// All features, HWCAP then HWCAP2 bit-index order. Index = bit in [`FeatureSet`].
	#[rustfmt::skip]
	pub const ALL: &'static [Feature] = &[
		Feature::Swp, Feature::Half, Feature::Thumb, Feature::Bit26, Feature::FastMult,
		Feature::Fpa, Feature::Vfp, Feature::Edsp, Feature::Java, Feature::Iwmmxt,
		Feature::Crunch, Feature::Thumbee, Feature::Neon, Feature::Vfpv3, Feature::Vfpv3D16,
		Feature::Tls, Feature::Vfpv4, Feature::Idiva, Feature::Idivt, Feature::Vfpd32,
		Feature::Lpae, Feature::Evtstrm, Feature::Fp16, Feature::AsimdHp, Feature::Dotprod,
		Feature::Fhm, Feature::AsimdBf16, Feature::I8mm,
		Feature::Aes, Feature::Pmull, Feature::Sha1, Feature::Sha2, Feature::Crc32,
		Feature::Sb, Feature::Ssbs,
	];

	/// Bit index inside a [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

/// Cached `AT_HWCAP`/`AT_HWCAP2` from one auxv pass.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
	hwcap: u32,
	hwcap2: u32,
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
	fn read() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android"))]
		// SAFETY: `getauxval` only reads this process's auxv; tags are libc consts.
		unsafe {
			Self {
				hwcap: libc::getauxval(libc::AT_HWCAP) as u32,
				hwcap2: libc::getauxval(libc::AT_HWCAP2) as u32,
			}
		}
		#[cfg(target_os = "freebsd")]
		{
			fn one(key: libc::c_int) -> u32 {
				let mut out: libc::c_ulong = 0;
				// SAFETY: elf_aux_info writes size_of out; FreeBSD 12+.
				let _ = unsafe {
					libc::elf_aux_info(
						key,
						(&mut out) as *mut libc::c_ulong as *mut libc::c_void,
						core::mem::size_of::<libc::c_ulong>() as libc::c_int,
					)
				};
				out as u32
			}
			Self {
				hwcap: one(libc::AT_HWCAP),
				hwcap2: one(libc::AT_HWCAP2),
			}
		}
	}

	/// Per-feature bit check. `match` (not if-chain) so missing arms fail compile.
	fn supports(&self, feature: Feature) -> bool {
		match feature {
			Feature::Swp => self.hwcap & hwcap::SWP != 0,
			Feature::Half => self.hwcap & hwcap::HALF != 0,
			Feature::Thumb => self.hwcap & hwcap::THUMB != 0,
			Feature::Bit26 => self.hwcap & hwcap::BIT26 != 0,
			Feature::FastMult => self.hwcap & hwcap::FAST_MULT != 0,
			Feature::Fpa => self.hwcap & hwcap::FPA != 0,
			Feature::Vfp => self.hwcap & hwcap::VFP != 0,
			Feature::Edsp => self.hwcap & hwcap::EDSP != 0,
			Feature::Java => self.hwcap & hwcap::JAVA != 0,
			Feature::Iwmmxt => self.hwcap & hwcap::IWMMXT != 0,
			Feature::Crunch => self.hwcap & hwcap::CRUNCH != 0,
			Feature::Thumbee => self.hwcap & hwcap::THUMBEE != 0,
			Feature::Neon => self.hwcap & hwcap::NEON != 0,
			Feature::Vfpv3 => self.hwcap & hwcap::VFPV3 != 0,
			Feature::Vfpv3D16 => self.hwcap & hwcap::VFPV3D16 != 0,
			Feature::Tls => self.hwcap & hwcap::TLS != 0,
			Feature::Vfpv4 => self.hwcap & hwcap::VFPV4 != 0,
			Feature::Idiva => self.hwcap & hwcap::IDIVA != 0,
			Feature::Idivt => self.hwcap & hwcap::IDIVT != 0,
			Feature::Vfpd32 => self.hwcap & hwcap::VFPD32 != 0,
			Feature::Lpae => self.hwcap & hwcap::LPAE != 0,
			Feature::Evtstrm => self.hwcap & hwcap::EVTSTRM != 0,
			Feature::Fp16 => self.hwcap & hwcap::FPHP != 0,
			Feature::AsimdHp => self.hwcap & hwcap::ASIMDHP != 0,
			Feature::Dotprod => self.hwcap & hwcap::ASIMDDP != 0,
			Feature::Fhm => self.hwcap & hwcap::ASIMDFHM != 0,
			Feature::AsimdBf16 => self.hwcap & hwcap::ASIMDBF16 != 0,
			Feature::I8mm => self.hwcap & hwcap::I8MM != 0,

			Feature::Aes => self.hwcap2 & hwcap2::AES != 0,
			Feature::Pmull => self.hwcap2 & hwcap2::PMULL != 0,
			Feature::Sha1 => self.hwcap2 & hwcap2::SHA1 != 0,
			Feature::Sha2 => self.hwcap2 & hwcap2::SHA2 != 0,
			Feature::Crc32 => self.hwcap2 & hwcap2::CRC32 != 0,
			Feature::Sb => self.hwcap2 & hwcap2::SB != 0,
			Feature::Ssbs => self.hwcap2 & hwcap2::SSBS != 0,
		}
	}
}

/// Bitset over [`Feature`] (one bit per [`Feature::ALL`] entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Probe host via auxv, or compile-time floor (bare-metal/other).
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
			// Fail-closed: only bits we can prove from cfg (no full HWCAP).
			#[allow(unused_mut, reason = "mutated under target_feature cfgs")]
			let mut set = 0u128;
			#[cfg(target_feature = "vfp2")]
			{
				set |= 1 << Feature::Vfp.bit();
			}
			#[cfg(target_feature = "vfp3")]
			{
				set |= 1 << Feature::Vfp.bit();
				set |= 1 << Feature::Vfpv3.bit();
			}
			#[cfg(target_feature = "vfp4")]
			{
				set |= 1 << Feature::Vfp.bit();
				set |= 1 << Feature::Vfpv3.bit();
				set |= 1 << Feature::Vfpv4.bit();
			}
			#[cfg(target_feature = "neon")]
			{
				set |= 1 << Feature::Neon.bit();
			}
			#[cfg(target_feature = "d32")]
			{
				set |= 1 << Feature::Vfpd32.bit();
			}
			Self(set)
		}
	}

	pub fn contains(self, feature: Feature) -> bool {
		self.0 & (1 << feature.bit()) != 0
	}

	/// All of `required` present.
	pub fn contains_all(self, required: &[Feature]) -> bool {
		required.iter().all(|&f| self.contains(f))
	}
}

/// Process-wide cache for one full [`FeatureSet`] probe (128 bits). Same
/// split-`AtomicU64` halves behind an `AtomicBool` as x86 (no stable
/// `AtomicU128`); keeps `auto_up` from re-reading `auxv` every call.
pub(crate) struct CachedFeatureSet {
	filled: core::sync::atomic::AtomicBool,
	lo: core::sync::atomic::AtomicU64,
	hi: core::sync::atomic::AtomicU64,
}

impl CachedFeatureSet {
	pub(crate) const fn new() -> Self {
		Self {
			filled: core::sync::atomic::AtomicBool::new(false),
			lo: core::sync::atomic::AtomicU64::new(0),
			hi: core::sync::atomic::AtomicU64::new(0),
		}
	}

	/// Cached value, or run `init` once (races may double-init; same CPU => same set).
	pub(crate) fn get_or_init(&self, init: impl FnOnce() -> FeatureSet) -> FeatureSet {
		use core::sync::atomic::Ordering;
		if self.filled.load(Ordering::Acquire) {
			let bits = (self.hi.load(Ordering::Relaxed) as u128) << 64 | self.lo.load(Ordering::Relaxed) as u128;
			return FeatureSet(bits);
		}
		let computed = init();
		self.lo.store(computed.0 as u64, Ordering::Relaxed);
		self.hi.store((computed.0 >> 64) as u64, Ordering::Relaxed);
		self.filled.store(true, Ordering::Release);
		computed
	}
}
