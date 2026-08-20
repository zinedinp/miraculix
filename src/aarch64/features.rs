//! AArch64 NEON/SVE/SME Feature enum. Probe: auxv (Linux/Android/FreeBSD),
//! `sysctlbyname` (macOS), `IsProcessorFeaturePresent` (Windows), NEON floor
//! (bare/other). [`hwcap`]/[`hwcap2`] restate UAPI bits.

/// `HWCAP_*` AT_HWCAP bits. Linux UAPI `asm/hwcap.h` (FreeBSD arm64 matches).
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap {
	pub const FP: u64 = 1 << 0;
	pub const ASIMD: u64 = 1 << 1;
	pub const EVTSTRM: u64 = 1 << 2;
	pub const AES: u64 = 1 << 3;
	pub const PMULL: u64 = 1 << 4;
	pub const SHA1: u64 = 1 << 5;
	pub const SHA2: u64 = 1 << 6;
	pub const CRC32: u64 = 1 << 7;
	pub const ATOMICS: u64 = 1 << 8;
	pub const FPHP: u64 = 1 << 9;
	pub const ASIMDHP: u64 = 1 << 10;
	pub const CPUID: u64 = 1 << 11;
	pub const ASIMDRDM: u64 = 1 << 12;
	pub const JSCVT: u64 = 1 << 13;
	pub const FCMA: u64 = 1 << 14;
	pub const LRCPC: u64 = 1 << 15;
	pub const DCPOP: u64 = 1 << 16;
	pub const SHA3: u64 = 1 << 17;
	pub const SM3: u64 = 1 << 18;
	pub const SM4: u64 = 1 << 19;
	pub const ASIMDDP: u64 = 1 << 20;
	pub const SHA512: u64 = 1 << 21;
	pub const SVE: u64 = 1 << 22;
	pub const ASIMDFHM: u64 = 1 << 23;
	pub const DIT: u64 = 1 << 24;
	pub const USCAT: u64 = 1 << 25;
	pub const ILRCPC: u64 = 1 << 26;
	pub const FLAGM: u64 = 1 << 27;
	pub const SSBS: u64 = 1 << 28;
	pub const SB: u64 = 1 << 29;
	pub const PACA: u64 = 1 << 30;
	pub const PACG: u64 = 1 << 31;
}

/// `HWCAP2_*` missing from some libcs. UAPI `asm/hwcap.h`.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
mod hwcap2 {
	pub const DCPODP: u64 = 1 << 0;
	pub const SVE2: u64 = 1 << 1;
	pub const SVE_AES: u64 = 1 << 2;
	pub const SVE_PMULL: u64 = 1 << 3;
	pub const SVE_BITPERM: u64 = 1 << 4;
	pub const SVE_SHA3: u64 = 1 << 5;
	pub const SVE_SM4: u64 = 1 << 6;
	pub const FLAGM2: u64 = 1 << 7;
	pub const FRINT: u64 = 1 << 8;
	pub const SVE_I8MM: u64 = 1 << 9;
	pub const SVE_F32MM: u64 = 1 << 10;
	pub const SVE_F64MM: u64 = 1 << 11;
	pub const SVE_BF16: u64 = 1 << 12;
	pub const I8MM: u64 = 1 << 13;
	pub const BF16: u64 = 1 << 14;
	pub const DGH: u64 = 1 << 15;
	pub const RNG: u64 = 1 << 16;
	pub const BTI: u64 = 1 << 17;
	pub const MTE: u64 = 1 << 18;
	pub const ECV: u64 = 1 << 19;
	pub const AFP: u64 = 1 << 20;
	pub const RPRES: u64 = 1 << 21;
	pub const MTE3: u64 = 1 << 22;
	pub const SME: u64 = 1 << 23;
	pub const SME_I16I64: u64 = 1 << 24;
	pub const SME_F64F64: u64 = 1 << 25;
	pub const SME_I8I32: u64 = 1 << 26;
	pub const SME_F16F32: u64 = 1 << 27;
	pub const SME_B16F32: u64 = 1 << 28;
	pub const SME_F32F32: u64 = 1 << 29;
	pub const SME_FA64: u64 = 1 << 30;
	pub const WFXT: u64 = 1 << 31;
	pub const EBF16: u64 = 1 << 32;
	pub const SVE_EBF16: u64 = 1 << 33;
}

/// Enum list for AArch64 NEON/SVE/SME-family extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[rustfmt::skip]
pub enum Feature {
	// AT_HWCAP
	/// Scalar FP.
	Fp,
	/// Advanced SIMD (NEON). Mandatory on every AArch64 CPU (ARMv8-A).
	Neon,
	/// Generic event stream.
	Evtstrm,
	Aes,
	/// `PMULL`/`PMULL2`, paired with AES.
	Pmull,
	Sha1,
	Sha2,
	Crc32,
	/// LSE atomics (`ARMv8.1-A`).
	Lse,
	/// Scalar half-precision arith.
	Fp16,
	/// NEON half-precision arith.
	AsimdHp,
	/// EL0 access to `ID_*` regs.
	CpuidReg,
	/// Rounding double multiply add/sub (`ARMv8.1-A`).
	Rdm,
	/// JS FP to int (`ARMv8.3-A`).
	Jscvt,
	/// Complex numbers (`ARMv8.3-A`).
	Fcma,
	/// RCpc atomics (`LDAPR`/`STLR`, `ARMv8.3-A`).
	Rcpc,
	/// DC clean to PoP (`ARMv8.2-A`).
	Dcpop,
	Sha3,
	Sm3,
	Sm4,
	/// Dot product (opt `ARMv8.2-A`, mand. `ARMv8.4-A`).
	Dotprod,
	Sha512,
	/// Scalable Vector Extension.
	Sve,
	/// NEON half-precision FMA.
	Fhm,
	/// Data Independent Timing (`ARMv8.4-A`).
	Dit,
	/// Unaligned atomics (`ARMv8.4-A`).
	Uscat,
	/// RCPC + immediate offsets (`ARMv8.4-A`).
	Rcpc2,
	/// Flag manip (`ARMv8.4-A`).
	Flagm,
	/// Speculative Store Bypass Safe (`ARMv8.5-A`).
	Ssbs,
	/// Speculation Barrier (`ARMv8.5-A`).
	Sb,
	/// Pointer Auth, address keys (`ARMv8.3-A`).
	Paca,
	/// Pointer Auth, generic keys (`ARMv8.3-A`).
	Pacg,

	// AT_HWCAP2
	/// DC clean to deep PoP (`ARMv8.5-A`).
	Dcpodp,
	/// SVE2.
	Sve2,
	/// SVE2 AES.
	Sve2Aes,
	/// SVE2 PMULL.
	Sve2Pmull,
	/// SVE2 bit permute.
	Sve2Bitperm,
	Sve2Sha3,
	Sve2Sm4,
	/// Alt NZCV for FP cmp (`ARMv8.5-A`).
	Flagm2,
	/// `FRINT32`/`FRINT64` (`ARMv8.5-A`).
	Frint,
	/// SVE i8 matrix mul.
	Sve2I8mm,
	/// SVE f32 matrix mul.
	Sve2F32mm,
	/// SVE f64 matrix mul.
	Sve2F64mm,
	/// SVE BF16.
	Sve2Bf16,
	/// NEON i8 matrix mul (`ARMv8.6-A`).
	I8mm,
	/// NEON BF16 (`ARMv8.6-A`).
	Bf16,
	/// Data Gathering Hint.
	Dgh,
	/// `RNDR`/`RNDRRS` (`ARMv8.5-A`).
	Rng,
	/// Branch Target Identification (`ARMv8.5-A`).
	Bti,
	/// Memory Tagging (`ARMv8.5-A` opt).
	Mte,
	/// Enhanced Counter Virtualization (`ARMv8.6-A`).
	Ecv,
	/// Alternate FP behavior (`ARMv8.7-A`).
	Afp,
	/// Higher-precision reciprocal est/step (`ARMv8.7-A`).
	Rpres,
	/// MTE v3.
	Mte3,
	/// Scalable Matrix Extension.
	Sme,
	SmeI16i64,
	SmeF64f64,
	SmeI8i32,
	SmeF16f32,
	SmeB16f32,
	SmeF32f32,
	/// Full A64 usable in streaming SVE mode.
	SmeFa64,
	/// `WFET`/`WFIT` (`ARMv8.7-A`).
	Wfxt,
	Ebf16,
	SveEbf16,
}

impl Feature {
	/// All features, HWCAP then HWCAP2 bit-index order. Index = bit in [`FeatureSet`].
	#[rustfmt::skip]
	pub const ALL: &'static [Feature] = &[
		Feature::Fp, Feature::Neon, Feature::Evtstrm, Feature::Aes, Feature::Pmull,
		Feature::Sha1, Feature::Sha2, Feature::Crc32, Feature::Lse, Feature::Fp16,
		Feature::AsimdHp, Feature::CpuidReg, Feature::Rdm, Feature::Jscvt, Feature::Fcma,
		Feature::Rcpc, Feature::Dcpop, Feature::Sha3, Feature::Sm3, Feature::Sm4,
		Feature::Dotprod, Feature::Sha512, Feature::Sve, Feature::Fhm, Feature::Dit,
		Feature::Uscat, Feature::Rcpc2, Feature::Flagm, Feature::Ssbs, Feature::Sb,
		Feature::Paca, Feature::Pacg,
		Feature::Dcpodp, Feature::Sve2, Feature::Sve2Aes, Feature::Sve2Pmull,
		Feature::Sve2Bitperm, Feature::Sve2Sha3, Feature::Sve2Sm4, Feature::Flagm2,
		Feature::Frint, Feature::Sve2I8mm, Feature::Sve2F32mm, Feature::Sve2F64mm,
		Feature::Sve2Bf16, Feature::I8mm, Feature::Bf16, Feature::Dgh, Feature::Rng,
		Feature::Bti, Feature::Mte, Feature::Ecv, Feature::Afp, Feature::Rpres,
		Feature::Mte3, Feature::Sme, Feature::SmeI16i64, Feature::SmeF64f64,
		Feature::SmeI8i32, Feature::SmeF16f32, Feature::SmeB16f32, Feature::SmeF32f32,
		Feature::SmeFa64, Feature::Wfxt, Feature::Ebf16, Feature::SveEbf16,
	];

	/// Bit index inside a [`FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

/// Cached `AT_HWCAP`/`AT_HWCAP2` from one auxv pass (getauxval or elf_aux_info).
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
struct AuxVal {
	hwcap: u64,
	hwcap2: u64,
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
impl AuxVal {
	fn read() -> Self {
		#[cfg(any(target_os = "linux", target_os = "android"))]
		// SAFETY: `getauxval` only reads this process's auxv; tags are libc consts.
		unsafe {
			return Self {
				hwcap: libc::getauxval(libc::AT_HWCAP) as u64,
				hwcap2: libc::getauxval(libc::AT_HWCAP2) as u64,
			};
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

	/// Per-feature bit check. `match` (not if-chain) so missing arms fail compile.
	fn supports(&self, feature: Feature) -> bool {
		match feature {
			Feature::Fp => self.hwcap & hwcap::FP != 0,
			Feature::Neon => self.hwcap & hwcap::ASIMD != 0,
			Feature::Evtstrm => self.hwcap & hwcap::EVTSTRM != 0,
			Feature::Aes => self.hwcap & hwcap::AES != 0,
			Feature::Pmull => self.hwcap & hwcap::PMULL != 0,
			Feature::Sha1 => self.hwcap & hwcap::SHA1 != 0,
			Feature::Sha2 => self.hwcap & hwcap::SHA2 != 0,
			Feature::Crc32 => self.hwcap & hwcap::CRC32 != 0,
			Feature::Lse => self.hwcap & hwcap::ATOMICS != 0,
			Feature::Fp16 => self.hwcap & hwcap::FPHP != 0,
			Feature::AsimdHp => self.hwcap & hwcap::ASIMDHP != 0,
			Feature::CpuidReg => self.hwcap & hwcap::CPUID != 0,
			Feature::Rdm => self.hwcap & hwcap::ASIMDRDM != 0,
			Feature::Jscvt => self.hwcap & hwcap::JSCVT != 0,
			Feature::Fcma => self.hwcap & hwcap::FCMA != 0,
			Feature::Rcpc => self.hwcap & hwcap::LRCPC != 0,
			Feature::Dcpop => self.hwcap & hwcap::DCPOP != 0,
			Feature::Sha3 => self.hwcap & hwcap::SHA3 != 0,
			Feature::Sm3 => self.hwcap & hwcap::SM3 != 0,
			Feature::Sm4 => self.hwcap & hwcap::SM4 != 0,
			Feature::Dotprod => self.hwcap & hwcap::ASIMDDP != 0,
			Feature::Sha512 => self.hwcap & hwcap::SHA512 != 0,
			Feature::Sve => self.hwcap & hwcap::SVE != 0,
			Feature::Fhm => self.hwcap & hwcap::ASIMDFHM != 0,
			Feature::Dit => self.hwcap & hwcap::DIT != 0,
			Feature::Uscat => self.hwcap & hwcap::USCAT != 0,
			Feature::Rcpc2 => self.hwcap & hwcap::ILRCPC != 0,
			Feature::Flagm => self.hwcap & hwcap::FLAGM != 0,
			Feature::Ssbs => self.hwcap & hwcap::SSBS != 0,
			Feature::Sb => self.hwcap & hwcap::SB != 0,
			Feature::Paca => self.hwcap & hwcap::PACA != 0,
			Feature::Pacg => self.hwcap & hwcap::PACG != 0,

			Feature::Dcpodp => self.hwcap2 & hwcap2::DCPODP != 0,
			Feature::Sve2 => self.hwcap2 & hwcap2::SVE2 != 0,
			Feature::Sve2Aes => self.hwcap2 & hwcap2::SVE_AES != 0,
			Feature::Sve2Pmull => self.hwcap2 & hwcap2::SVE_PMULL != 0,
			Feature::Sve2Bitperm => self.hwcap2 & hwcap2::SVE_BITPERM != 0,
			Feature::Sve2Sha3 => self.hwcap2 & hwcap2::SVE_SHA3 != 0,
			Feature::Sve2Sm4 => self.hwcap2 & hwcap2::SVE_SM4 != 0,
			Feature::Flagm2 => self.hwcap2 & hwcap2::FLAGM2 != 0,
			Feature::Frint => self.hwcap2 & hwcap2::FRINT != 0,
			Feature::Sve2I8mm => self.hwcap2 & hwcap2::SVE_I8MM != 0,
			Feature::Sve2F32mm => self.hwcap2 & hwcap2::SVE_F32MM != 0,
			Feature::Sve2F64mm => self.hwcap2 & hwcap2::SVE_F64MM != 0,
			Feature::Sve2Bf16 => self.hwcap2 & hwcap2::SVE_BF16 != 0,
			Feature::I8mm => self.hwcap2 & hwcap2::I8MM != 0,
			Feature::Bf16 => self.hwcap2 & hwcap2::BF16 != 0,
			Feature::Dgh => self.hwcap2 & hwcap2::DGH != 0,
			Feature::Rng => self.hwcap2 & hwcap2::RNG != 0,
			Feature::Bti => self.hwcap2 & hwcap2::BTI != 0,
			Feature::Mte => self.hwcap2 & hwcap2::MTE != 0,
			Feature::Ecv => self.hwcap2 & hwcap2::ECV != 0,
			Feature::Afp => self.hwcap2 & hwcap2::AFP != 0,
			Feature::Rpres => self.hwcap2 & hwcap2::RPRES != 0,
			Feature::Mte3 => self.hwcap2 & hwcap2::MTE3 != 0,
			Feature::Sme => self.hwcap2 & hwcap2::SME != 0,
			Feature::SmeI16i64 => self.hwcap2 & hwcap2::SME_I16I64 != 0,
			Feature::SmeF64f64 => self.hwcap2 & hwcap2::SME_F64F64 != 0,
			Feature::SmeI8i32 => self.hwcap2 & hwcap2::SME_I8I32 != 0,
			Feature::SmeF16f32 => self.hwcap2 & hwcap2::SME_F16F32 != 0,
			Feature::SmeB16f32 => self.hwcap2 & hwcap2::SME_B16F32 != 0,
			Feature::SmeF32f32 => self.hwcap2 & hwcap2::SME_F32F32 != 0,
			Feature::SmeFa64 => self.hwcap2 & hwcap2::SME_FA64 != 0,
			Feature::Wfxt => self.hwcap2 & hwcap2::WFXT != 0,
			Feature::Ebf16 => self.hwcap2 & hwcap2::EBF16 != 0,
			Feature::SveEbf16 => self.hwcap2 & hwcap2::SVE_EBF16 != 0,
		}
	}
}

/// macOS detect: one `sysctlbyname` per feature (no getauxval/bulk HWCAP).
/// Names from M1 Pro `sysctl -a` + Go `cpu_arm64_darwin.go` (XNU table closed).
#[cfg(target_os = "macos")]
mod apple {
	use super::Feature;
	use core::ffi::{c_void, CStr};

	/// Boolean sysctl; missing name or 0 => unsupported.
	fn sysctl_bool(name: &CStr) -> bool {
		let mut value: i32 = 0;
		let mut len: libc::size_t = core::mem::size_of::<i32>();
		// SAFETY: NUL-terminated name; value/len size match i32 out buffer.
		let ret = unsafe {
			libc::sysctlbyname(
				name.as_ptr(),
				(&mut value) as *mut i32 as *mut c_void,
				&mut len,
				core::ptr::null_mut(),
				0,
			)
		};
		ret == 0 && value != 0
	}

	/// `match` so a new Feature arm is a compile error.
	pub(super) fn supports(feature: Feature) -> bool {
		match feature {
			// aarch64-apple-darwin floor is M1-class.
			Feature::Fp | Feature::Neon => true,

			Feature::Aes => sysctl_bool(c"hw.optional.arm.FEAT_AES"),
			Feature::Pmull => sysctl_bool(c"hw.optional.arm.FEAT_PMULL"),
			Feature::Sha1 => sysctl_bool(c"hw.optional.arm.FEAT_SHA1"),
			Feature::Sha2 => sysctl_bool(c"hw.optional.arm.FEAT_SHA256"),
			Feature::Sha3 => sysctl_bool(c"hw.optional.arm.FEAT_SHA3"),
			Feature::Sha512 => sysctl_bool(c"hw.optional.arm.FEAT_SHA512"),
			// Legacy name only (not under FEAT_*); present on M1 Pro.
			Feature::Crc32 => sysctl_bool(c"hw.optional.armv8_crc32"),
			Feature::Lse => sysctl_bool(c"hw.optional.arm.FEAT_LSE"),
			Feature::Fp16 => sysctl_bool(c"hw.optional.arm.FEAT_FP16"),
			Feature::Rdm => sysctl_bool(c"hw.optional.arm.FEAT_RDM"),
			Feature::Jscvt => sysctl_bool(c"hw.optional.arm.FEAT_JSCVT"),
			Feature::Fcma => sysctl_bool(c"hw.optional.arm.FEAT_FCMA"),
			Feature::Rcpc => sysctl_bool(c"hw.optional.arm.FEAT_LRCPC"),
			Feature::Rcpc2 => sysctl_bool(c"hw.optional.arm.FEAT_LRCPC2"),
			Feature::Dcpop => sysctl_bool(c"hw.optional.arm.FEAT_DPB"),
			Feature::Dcpodp => sysctl_bool(c"hw.optional.arm.FEAT_DPB2"),
			Feature::Dotprod => sysctl_bool(c"hw.optional.arm.FEAT_DotProd"),
			Feature::Fhm => sysctl_bool(c"hw.optional.arm.FEAT_FHM"),
			Feature::Dit => sysctl_bool(c"hw.optional.arm.FEAT_DIT"),
			Feature::Flagm => sysctl_bool(c"hw.optional.arm.FEAT_FlagM"),
			Feature::Flagm2 => sysctl_bool(c"hw.optional.arm.FEAT_FlagM2"),
			Feature::Ssbs => sysctl_bool(c"hw.optional.arm.FEAT_SSBS"),
			Feature::Sb => sysctl_bool(c"hw.optional.arm.FEAT_SB"),
			Feature::Frint => sysctl_bool(c"hw.optional.arm.FEAT_FRINTTS"),
			Feature::I8mm => sysctl_bool(c"hw.optional.arm.FEAT_I8MM"),
			Feature::Bf16 => sysctl_bool(c"hw.optional.arm.FEAT_BF16"),
			Feature::Bti => sysctl_bool(c"hw.optional.arm.FEAT_BTI"),
			Feature::Ecv => sysctl_bool(c"hw.optional.arm.FEAT_ECV"),
			// One FEAT_PAuth for both PACA/PACG (Linux has two HWCAP bits).
			Feature::Paca | Feature::Pacg => sysctl_bool(c"hw.optional.arm.FEAT_PAuth"),

			// No known Apple sysctl (EL0-irrelevant, or no SVE/SME/MTE; AMX instead).
			Feature::Evtstrm
			| Feature::AsimdHp
			| Feature::CpuidReg
			| Feature::Sm3
			| Feature::Sm4
			| Feature::Sve
			| Feature::Uscat
			| Feature::Dgh
			| Feature::Rng
			| Feature::Mte
			| Feature::Afp
			| Feature::Rpres
			| Feature::Mte3
			| Feature::Wfxt
			| Feature::Ebf16
			| Feature::Sve2
			| Feature::Sve2Aes
			| Feature::Sve2Pmull
			| Feature::Sve2Bitperm
			| Feature::Sve2Sha3
			| Feature::Sve2Sm4
			| Feature::Sve2I8mm
			| Feature::Sve2F32mm
			| Feature::Sve2F64mm
			| Feature::Sve2Bf16
			| Feature::SveEbf16
			| Feature::Sme
			| Feature::SmeI16i64
			| Feature::SmeF64f64
			| Feature::SmeI8i32
			| Feature::SmeF16f32
			| Feature::SmeB16f32
			| Feature::SmeF32f32
			| Feature::SmeFa64 => false,
		}
	}
}

/// Windows-on-Arm detection uses `IsProcessorFeaturePresent` (via
/// `windows-sys`). This mapping is coarser than Linux HWCAP/sysctl; some
/// newer `PF_ARM_*` IDs may be absent in the current `windows-sys` and
/// therefore map to `false` until the crate is upgraded.
#[cfg(target_os = "windows")]
mod windows {
	use super::Feature;
	use windows_sys::Win32::System::Threading::{
		IsProcessorFeaturePresent, PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE,
		PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE,
		PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE, PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE,
		PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE,
	};

	fn pf(id: u32) -> bool {
		// SAFETY: `id` is a `PROCESSOR_FEATURE_ID`; the call touches no memory of ours.
		unsafe { IsProcessorFeaturePresent(id) != 0 }
	}

	/// `match` so a new Feature arm is a compile error.
	pub(super) fn supports(feature: Feature) -> bool {
		match feature {
			// aarch64-pc-windows-msvc floor: NEON always in the default cfg.
			Feature::Fp | Feature::Neon => true,

			// One flag for the whole bundle (Linux/Apple split these four).
			Feature::Aes | Feature::Pmull | Feature::Sha1 | Feature::Sha2 => {
				pf(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE)
			}
			Feature::Crc32 => pf(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE),
			Feature::Lse => pf(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE),
			Feature::Dotprod => pf(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE),
			Feature::Jscvt => pf(PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE),
			Feature::Rcpc => pf(PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE),

			// No PF_ARM_* ID in windows-sys 0.61.2: absent from the API
			// entirely (PAC, RDM, FlagM, ...) or added in WinSDK 10.0.26100
			// and not yet in the crate (LSE2, SHA3, SHA512, I8MM, BF16,
			// SVE(2), SME family).
			Feature::Evtstrm
			| Feature::AsimdHp
			| Feature::CpuidReg
			| Feature::Fp16
			| Feature::Rdm
			| Feature::Fcma
			| Feature::Dcpop
			| Feature::Sha3
			| Feature::Sm3
			| Feature::Sm4
			| Feature::Sha512
			| Feature::Sve
			| Feature::Fhm
			| Feature::Dit
			| Feature::Uscat
			| Feature::Rcpc2
			| Feature::Flagm
			| Feature::Ssbs
			| Feature::Sb
			| Feature::Paca
			| Feature::Pacg
			| Feature::Dcpodp
			| Feature::Sve2
			| Feature::Sve2Aes
			| Feature::Sve2Pmull
			| Feature::Sve2Bitperm
			| Feature::Sve2Sha3
			| Feature::Sve2Sm4
			| Feature::Flagm2
			| Feature::Frint
			| Feature::Sve2I8mm
			| Feature::Sve2F32mm
			| Feature::Sve2F64mm
			| Feature::Sve2Bf16
			| Feature::I8mm
			| Feature::Bf16
			| Feature::Dgh
			| Feature::Rng
			| Feature::Bti
			| Feature::Mte
			| Feature::Ecv
			| Feature::Afp
			| Feature::Rpres
			| Feature::Mte3
			| Feature::Sme
			| Feature::SmeI16i64
			| Feature::SmeF64f64
			| Feature::SmeI8i32
			| Feature::SmeF16f32
			| Feature::SmeB16f32
			| Feature::SmeF32f32
			| Feature::SmeFa64
			| Feature::Wfxt
			| Feature::Ebf16
			| Feature::SveEbf16 => false,
		}
	}
}

/// Bitset over [`Feature`] (one bit per [`Feature::ALL`] entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Probe host: auxv (Linux/Android/FreeBSD), `sysctlbyname` (macOS),
	/// `IsProcessorFeaturePresent` (Windows), or NEON floor (bare-metal/other).
	pub fn detect() -> Self {
		let mut set = 0u128;

		#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
		{
			let aux = AuxVal::read();
			for &feature in Feature::ALL {
				if aux.supports(feature) {
					set |= 1 << feature.bit();
				}
			}
		}
		#[cfg(target_os = "macos")]
		{
			for &feature in Feature::ALL {
				if apple::supports(feature) {
					set |= 1 << feature.bit();
				}
			}
		}
		#[cfg(target_os = "windows")]
		{
			for &feature in Feature::ALL {
				if windows::supports(feature) {
					set |= 1 << feature.bit();
				}
			}
		}
		// Bare-metal / other OS: fail-closed NEON floor (mandatory AArch64).
		// Higher tiers only via shortpath Assumed (`v8.1a`..`v9a` cfgs).
		#[cfg(not(any(
			target_os = "linux",
			target_os = "android",
			target_os = "freebsd",
			target_os = "macos",
			target_os = "windows"
		)))]
		{
			set |= 1 << Feature::Fp.bit();
			set |= 1 << Feature::Neon.bit();
		}

		Self(set)
	}

	pub fn contains(self, feature: Feature) -> bool {
		self.0 & (1 << feature.bit()) != 0
	}

	/// All of `required` present.
	pub fn contains_all(self, required: &[Feature]) -> bool {
		required.iter().all(|&f| self.contains(f))
	}
}
