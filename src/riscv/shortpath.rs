//! Compile-time collapse of runtime dispatch. Gc always present (like NEON).
//! Key: `target_feature = "v"` only (rva20/rva22 are LLVM codegen names, not
//! usable in `cfg` except `v` / `rva23u64`; Feature is only the 7 HWCAP bits).

use super::tiers::RiscvLevel;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use super::FeatureSet;

#[cfg(target_feature = "v")]
const COMPILE_TIME_LEVEL: RiscvLevel = RiscvLevel::Vector;

#[cfg(not(target_feature = "v"))]
const COMPILE_TIME_LEVEL: RiscvLevel = RiscvLevel::Gc;

/// Assumed above Gc vs still need runtime `getauxval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// Assumed via `+v` / matching target-cpu; no runtime check for that level.
	Assumed(RiscvLevel),
	/// Only Gc known; use [`RiscvLevel::detect`].
	Baseline,
}

/// Skip runtime dispatch above Gc if compile-time assumed.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		RiscvLevel::Gc => ResolvedPath::Baseline,
		level => ResolvedPath::Assumed(level),
	}
}

/// Process-start re-check via auxv. No-op on Baseline and on compile-time-only
/// platforms.
///
/// # Panics
/// Assumed Vector missing on this CPU (when a runtime probe exists).
pub fn verify_or_panic() {
	#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
	{
		return;
	}
	#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
	if let ResolvedPath::Assumed(level) = resolve() {
		let set = FeatureSet::detect();
		if !set.contains_all(level.required_features()) {
			panic!(
				"miraculix: this binary was compiled assuming RISC-V {level:?} (via `-C \
				 target-feature=+v`, `-march=...v`, or a matching `-C target-cpu=`), but the CPU \
				 it is running on does not support that level. Recompile without `+v` (or plain \
				 riscv{{32,64}}gc baseline), or run this binary on hardware with the Vector \
				 extension."
			);
		}
	}
}
