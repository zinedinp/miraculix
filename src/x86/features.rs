//! # CPU features you can query
//!
//! [`Feature`] is one ISA flag (e.g. AVX2). [`FeatureSet`] is the bitset of
//! everything this host (or probe) has.
//!
//! End users normally call [`crate::x86::detect_features`] (cached) rather
//! than [`FeatureSet::detect`] directly. Use the set to gate tokens:
//! `Avx2::from_features(set)`.

use raw_cpuid::{CpuId, ExtendedFeatures, ExtendedProcessorFeatureIdentifiers, FeatureInfo};

/// One x86 / x86_64 ISA extension flag.
///
/// Use with [`FeatureSet::contains`]. Dates in variant docs are history only;
/// detection always reads `CPUID` (or compile-time shortpath bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[rustfmt::skip]
pub enum Feature {
	/// Pentium MMX, 1997.
	Mmx,
	/// AMD K6-2, 1998. AMD-only; hardware support dropped since Bulldozer (2011).
	ThreeDNow,
	/// Pentium III, 1999.
	Sse,
	/// Pentium 4, 2001.
	Sse2,
	/// Prescott, 2004.
	Sse3,
	/// AMD K8 rev E ("Venice"), 2005. AMD-only.
	Sse4a,
	/// Core 2 ("Merom"), 2006.
	Ssse3,
	/// Penryn, 2007.
	Sse41,
	/// Nehalem, 2008.
	Sse42,
	/// Nehalem, 2008.
	Popcnt,
	/// Westmere, 2010.
	Aes,
	/// Westmere, 2010.
	Pclmulqdq,
	/// Sandy Bridge, 2011.
	Avx,
	/// Sandy Bridge, 2011.
	Xsave,
	/// AMD Bulldozer, 2011. AMD-only; dropped after Excavator (~2015, pre-Zen).
	Fma4,
	/// AMD Bulldozer, 2011. AMD-only; dropped after Excavator (~2015, pre-Zen).
	Xop,
	/// Ivy Bridge, 2012.
	Rdrand,
	/// Ivy Bridge, 2012.
	F16c,
	/// Ivy Bridge, 2012.
	Fsgsbase,
	/// Haswell, 2013.
	Bmi1,
	/// Haswell, 2013.
	Bmi2,
	/// Haswell, 2013 (also AMD Piledriver "ABM", 2012).
	Lzcnt,
	/// Haswell, 2013.
	Fma,
	/// Haswell, 2013.
	Movbe,
	/// Haswell, 2013. TSX; mostly fused-off since 2021 erratum.
	Hle,
	/// Haswell, 2013. TSX; same as [`Feature::Hle`].
	Rtm,
	/// Haswell, 2013.
	Avx2,
	/// Broadwell, 2014.
	Adx,
	/// Broadwell, 2014.
	Rdseed,
	/// Goldmont, 2016 (big-core: Ice Lake, 2019).
	Sha,
	/// Knights Landing, 2016 (mainstream: Skylake-X, 2017).
	Avx512f,
	/// Knights Landing, 2016 (mainstream: Skylake-X, 2017).
	Avx512cd,
	/// Skylake-X, 2017.
	Avx512dq,
	/// Skylake-X, 2017.
	Avx512bw,
	/// Skylake-X, 2017.
	Avx512vl,
	/// Knights Landing, 2016. Xeon Phi only.
	Avx512er,
	/// Knights Landing, 2016. Xeon Phi only.
	Avx512pf,
	/// Cannon Lake, 2018.
	Avx512ifma,
	/// Cannon Lake, 2018 (wide: Ice Lake, 2019).
	Avx512vbmi,
	/// Knights Mill, 2018. Xeon Phi only.
	Avx5124vnniw,
	/// Knights Mill, 2018. Xeon Phi only.
	Avx5124fmaps,
	/// Ice Lake, 2019.
	Avx512vbmi2,
	/// Ice Lake, 2019.
	Gfni,
	/// Ice Lake, 2019.
	Vaes,
	/// Ice Lake, 2019.
	Vpclmulqdq,
	/// Ice Lake / Cascade Lake, 2019.
	Avx512vnni,
	/// Ice Lake, 2019.
	Avx512bitalg,
	/// Ice Lake, 2019.
	Avx512vpopcntdq,
	/// Cooper Lake, 2020.
	Avx512bf16,
	/// Tiger Lake, 2021.
	Avx512vp2intersect,
	/// Alder Lake, 2021. AVX2-width VNNI (no AVX-512 needed).
	AvxVnni,
	/// Sapphire Rapids, 2023.
	Avx512fp16,
	/// Sapphire Rapids, 2023.
	AmxTile,
	/// Sapphire Rapids, 2023.
	AmxInt8,
	/// Sapphire Rapids, 2023.
	AmxBf16,
	/// Sierra Forest, 2024.
	AvxIfma,
	/// Sierra Forest, 2024.
	AvxNeConvert,
	/// Sierra Forest, 2024.
	AvxVnniInt8,
	/// Granite Rapids / Arrow Lake, 2024.
	AvxVnniInt16,
	/// Spec 2023; silicon (Diamond Rapids) 2025+. Presence only; version/width
	/// via [`crate::x86::tiers::Avx10`] (leaf `0x24`; `raw-cpuid` doesn't parse yet).
	Avx10,
	/// ~2024, Sierra Forest/Granite Rapids era. Leaf `7` subleaf `1` `EAX`
	/// bit 0; `raw-cpuid` doesn't parse it, read directly (see `supports`).
	Sha512,
	/// ~2024, Sierra Forest/Granite Rapids era. Leaf `7` subleaf `1` `EAX`
	/// bit 1; `raw-cpuid` doesn't parse it, read directly (see `supports`).
	Sm3,
	/// ~2024, Sierra Forest/Granite Rapids era. Leaf `7` subleaf `1` `EAX`
	/// bit 2; `raw-cpuid` doesn't parse it, read directly (see `supports`).
	Sm4,
}

impl Feature {
	/// All features, oldest-first. Index = bit in [`super::FeatureSet`].
	#[rustfmt::skip]
	pub const ALL: &'static [Feature] = &[
		Feature::Mmx, Feature::ThreeDNow, Feature::Sse, Feature::Sse2, Feature::Sse3, Feature::Sse4a,
		Feature::Ssse3, Feature::Sse41, Feature::Sse42, Feature::Popcnt,
		Feature::Aes, Feature::Pclmulqdq, Feature::Avx, Feature::Xsave,
		Feature::Fma4, Feature::Xop,
		Feature::Rdrand, Feature::F16c, Feature::Fsgsbase, Feature::Bmi1,
		Feature::Bmi2, Feature::Lzcnt, Feature::Fma, Feature::Movbe,
		Feature::Hle, Feature::Rtm, Feature::Avx2, Feature::Adx, Feature::Rdseed,
		Feature::Sha, Feature::Avx512f, Feature::Avx512cd, Feature::Avx512dq,
		Feature::Avx512bw, Feature::Avx512vl, Feature::Avx512er, Feature::Avx512pf,
		Feature::Avx512ifma, Feature::Avx512vbmi, Feature::Avx5124vnniw,
		Feature::Avx5124fmaps, Feature::Avx512vbmi2, Feature::Gfni, Feature::Vaes,
		Feature::Vpclmulqdq, Feature::Avx512vnni, Feature::Avx512bitalg,
		Feature::Avx512vpopcntdq, Feature::Avx512bf16, Feature::Avx512vp2intersect,
		Feature::AvxVnni, Feature::Avx512fp16, Feature::AmxTile, Feature::AmxInt8,
		Feature::AmxBf16, Feature::AvxIfma, Feature::AvxNeConvert,
		Feature::AvxVnniInt8, Feature::AvxVnniInt16, Feature::Avx10,
		Feature::Sha512, Feature::Sm3, Feature::Sm4,
	];

	/// Bit index inside a [`super::FeatureSet`].
	pub fn bit(self) -> u32 {
		Self::ALL
			.iter()
			.position(|&f| f == self)
			.expect("Feature::ALL must list every variant") as u32
	}
}

/// Cached CPUID leaves this crate reads.
struct CpuidLeaves {
	feature_info: Option<FeatureInfo>,
	extended_features: Option<ExtendedFeatures>,
	extended_processor: Option<ExtendedProcessorFeatureIdentifiers>,
	/// Leaf 7 subleaf 1 EAX: SHA512/SM3/SM4 bits. `raw-cpuid` does not fold
	/// these into `ExtendedFeatures`, so we read this subleaf directly (see
	/// `super::tiers::Avx10::version` for precedent). Querying a non-implemented
	/// subleaf is well-defined and safe.
	sha_sm_leaf_eax: u32,
}

impl CpuidLeaves {
	fn read() -> Self {
		let cpuid = CpuId::new();
		Self {
			feature_info: cpuid.get_feature_info(),
			extended_features: cpuid.get_extended_feature_info(),
			extended_processor: cpuid.get_extended_processor_and_feature_identifiers(),
			sha_sm_leaf_eax: Self::read_sha_sm_leaf(),
		}
	}

	fn read_sha_sm_leaf() -> u32 {
		#[cfg(target_arch = "x86_64")]
		use core::arch::x86_64::__cpuid_count;
		#[cfg(target_arch = "x86")]
		use core::arch::x86::__cpuid_count;

		__cpuid_count(7, 1).eax
	}

	/// Per-feature leaf check. `match` (not if-chain) so missing arms fail compile.
	fn supports(&self, feature: Feature) -> bool {
		match feature {
			Feature::Mmx => self.feature_info.as_ref().is_some_and(FeatureInfo::has_mmx),
			Feature::ThreeDNow => self.extended_processor.as_ref().is_some_and(ExtendedProcessorFeatureIdentifiers::has_3dnow),
			Feature::Sse => self.feature_info.as_ref().is_some_and(FeatureInfo::has_sse),
			Feature::Sse2 => self.feature_info.as_ref().is_some_and(FeatureInfo::has_sse2),
			Feature::Sse3 => self.feature_info.as_ref().is_some_and(FeatureInfo::has_sse3),
			Feature::Sse4a => self.extended_processor.as_ref().is_some_and(ExtendedProcessorFeatureIdentifiers::has_sse4a),
			Feature::Ssse3 => self.feature_info.as_ref().is_some_and(FeatureInfo::has_ssse3),
			Feature::Sse41 => self.feature_info.as_ref().is_some_and(FeatureInfo::has_sse41),
			Feature::Sse42 => self.feature_info.as_ref().is_some_and(FeatureInfo::has_sse42),
			Feature::Popcnt => self.feature_info.as_ref().is_some_and(FeatureInfo::has_popcnt),
			Feature::Aes => self.feature_info.as_ref().is_some_and(FeatureInfo::has_aesni),
			Feature::Pclmulqdq => self.feature_info.as_ref().is_some_and(FeatureInfo::has_pclmulqdq),
			Feature::Avx => self.feature_info.as_ref().is_some_and(FeatureInfo::has_avx),
			Feature::Xsave => self.feature_info.as_ref().is_some_and(FeatureInfo::has_xsave),
			Feature::Fma4 => self.extended_processor.as_ref().is_some_and(ExtendedProcessorFeatureIdentifiers::has_fma4),
			Feature::Xop => self.extended_processor.as_ref().is_some_and(ExtendedProcessorFeatureIdentifiers::has_xop),
			Feature::Rdrand => self.feature_info.as_ref().is_some_and(FeatureInfo::has_rdrand),
			Feature::F16c => self.feature_info.as_ref().is_some_and(FeatureInfo::has_f16c),
			Feature::Fsgsbase => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_fsgsbase),
			Feature::Bmi1 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_bmi1),
			Feature::Bmi2 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_bmi2),
			Feature::Lzcnt => self.extended_processor.as_ref().is_some_and(ExtendedProcessorFeatureIdentifiers::has_lzcnt),
			Feature::Fma => self.feature_info.as_ref().is_some_and(FeatureInfo::has_fma),
			Feature::Movbe => self.feature_info.as_ref().is_some_and(FeatureInfo::has_movbe),
			Feature::Hle => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_hle),
			Feature::Rtm => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_rtm),
			Feature::Avx2 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx2),
			Feature::Adx => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_adx),
			Feature::Rdseed => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_rdseed),
			Feature::Sha => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_sha),
			Feature::Avx512f => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512f),
			Feature::Avx512cd => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512cd),
			Feature::Avx512dq => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512dq),
			Feature::Avx512bw => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512bw),
			Feature::Avx512vl => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512vl),
			Feature::Avx512er => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512er),
			Feature::Avx512pf => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512pf),
			Feature::Avx512ifma => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_ifma),
			Feature::Avx512vbmi => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512vbmi),
			Feature::Avx5124vnniw => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_4vnniw),
			Feature::Avx5124fmaps => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_4fmaps),
			Feature::Avx512vbmi2 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512vbmi2),
			Feature::Gfni => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_gfni),
			Feature::Vaes => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_vaes),
			Feature::Vpclmulqdq => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_vpclmulqdq),
			Feature::Avx512vnni => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512vnni),
			Feature::Avx512bitalg => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512bitalg),
			Feature::Avx512vpopcntdq => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512vpopcntdq),
			Feature::Avx512bf16 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_bf16),
			Feature::Avx512vp2intersect => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_vp2intersect),
			Feature::AvxVnni => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx_vnni),
			Feature::Avx512fp16 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx512_fp16),
			Feature::AmxTile => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_amx_tile),
			Feature::AmxInt8 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_amx_int8),
			Feature::AmxBf16 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_amx_bf16),
			Feature::AvxIfma => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx_ifma),
			Feature::AvxNeConvert => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx_ne_convert),
			Feature::AvxVnniInt8 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx_vnni_int8),
			Feature::AvxVnniInt16 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx_vnni_int16),
			Feature::Avx10 => self.extended_features.as_ref().is_some_and(ExtendedFeatures::has_avx10),
			Feature::Sha512 => self.sha_sm_leaf_eax & (1 << 0) != 0,
			Feature::Sm3 => self.sha_sm_leaf_eax & (1 << 1) != 0,
			Feature::Sm4 => self.sha_sm_leaf_eax & (1 << 2) != 0,
		}
	}
}

/// Bitset of which [`Feature`]s are present (one bit per [`Feature::ALL`] entry).
///
/// Prefer [`crate::x86::detect_features`] in application code: it is cached
/// and unions compile-time lower bounds. Call [`FeatureSet::detect`] only when
/// you need a one-off raw probe.
///
/// # Example
///
/// ```
/// # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
/// use miraculix::x86::{detect_features, Feature};
/// let set = detect_features();
/// if set.contains(Feature::Avx2) && set.contains_all(&[Feature::Fma, Feature::F16c]) {
///     // Ready for a V3-style float kernel.
/// }
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureSet(u128);

impl FeatureSet {
	/// Probe the host once via `CPUID` (no process cache).
	///
	/// Prefer [`crate::x86::detect_features`] for normal code.
	pub fn detect() -> Self {
		let leaves = CpuidLeaves::read();
		let mut set = 0u128;
		for &feature in Feature::ALL {
			if leaves.supports(feature) {
				set |= 1 << feature.bit();
			}
		}
		Self(set)
	}

	/// `true` if this single extension is present.
	pub fn contains(self, feature: Feature) -> bool {
		self.0 & (1 << feature.bit()) != 0
	}

	/// `true` if every feature in `required` is present.
	pub fn contains_all(self, required: &[Feature]) -> bool {
		required.iter().all(|&f| self.contains(f))
	}

	/// Features present in **either** set (bitwise OR).
	pub fn union(self, other: Self) -> Self {
		Self(self.0 | other.0)
	}

	pub(crate) fn with(self, feature: Feature) -> Self {
		Self(self.0 | (1 << feature.bit()))
	}
}

/// Process-wide cache for one full [`FeatureSet`] probe (128 bits). No stable
/// `AtomicU128` on `core`, so the bitset is split into two `AtomicU64` halves
/// behind an `AtomicBool` filled-flag (`Acquire`/`Release` pair) rather than
/// [`super::super::level_cache::CachedU8`]'s single-sentinel-byte trick.
/// `no_std`-safe: `core::sync::atomic` only. Same "parallel first callers may
/// both probe; same CPU => same result, harmless" contract as `CachedU8`.
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
