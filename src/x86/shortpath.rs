//! # Compile-time shortpath (advanced / startup)
//!
//! Most callers never need this module. Prefer [`crate::x86::detect_level`] and
//! [`crate::x86::detect_features`].
//!
//! | Call | Who uses it |
//! |---|---|
//! | [`verify_or_panic`] | Apps that ship a high `-C target-cpu` binary and want a clear panic on weaker CPUs |
//! | [`resolve`] | Internals / tests: was a whole tier assumed at compile time? |
//! | [`compile_time_features`] | Internals: per-feature lower bound for auto dispatch |

use super::tiers::GenericLevel;
use super::{Feature, FeatureSet};

#[cfg(all(
	target_feature = "avx512f",
	target_feature = "avx512bw",
	target_feature = "avx512cd",
	target_feature = "avx512dq",
	target_feature = "avx512vl",
	target_feature = "avx2",
	target_feature = "bmi1",
	target_feature = "bmi2",
	target_feature = "f16c",
	target_feature = "fma",
	target_feature = "lzcnt",
	target_feature = "movbe",
	target_feature = "xsave",
	target_feature = "popcnt",
	target_feature = "sse3",
	target_feature = "ssse3",
	target_feature = "sse4.1",
	target_feature = "sse4.2",
))]
const COMPILE_TIME_LEVEL: Option<GenericLevel> = Some(GenericLevel::V4);

#[cfg(all(
	not(all(
		target_feature = "avx512f",
		target_feature = "avx512bw",
		target_feature = "avx512cd",
		target_feature = "avx512dq",
		target_feature = "avx512vl",
	)),
	target_feature = "avx2",
	target_feature = "bmi1",
	target_feature = "bmi2",
	target_feature = "f16c",
	target_feature = "fma",
	target_feature = "lzcnt",
	target_feature = "movbe",
	target_feature = "xsave",
	target_feature = "popcnt",
	target_feature = "sse3",
	target_feature = "ssse3",
	target_feature = "sse4.1",
	target_feature = "sse4.2",
))]
const COMPILE_TIME_LEVEL: Option<GenericLevel> = Some(GenericLevel::V3);

#[cfg(all(
	not(all(
		target_feature = "avx2",
		target_feature = "bmi1",
		target_feature = "bmi2",
		target_feature = "f16c",
		target_feature = "fma",
		target_feature = "lzcnt",
		target_feature = "movbe",
		target_feature = "xsave",
	)),
	target_feature = "popcnt",
	target_feature = "sse3",
	target_feature = "ssse3",
	target_feature = "sse4.1",
	target_feature = "sse4.2",
))]
const COMPILE_TIME_LEVEL: Option<GenericLevel> = Some(GenericLevel::V2);

#[cfg(not(all(
	target_feature = "popcnt",
	target_feature = "sse3",
	target_feature = "ssse3",
	target_feature = "sse4.1",
	target_feature = "sse4.2",
)))]
const COMPILE_TIME_LEVEL: Option<GenericLevel> = None;

/// Result of [`resolve`]: was a whole tier assumed when this binary was built?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPath {
	/// Compiler flags already guarantee this [`GenericLevel`] (no runtime probe needed for the tier).
	Assumed(GenericLevel),
	/// Must probe the host at runtime.
	RuntimeDispatch,
}

/// Did compile flags assume a full [`GenericLevel`] for this binary?
///
/// Used by [`crate::x86::detect_level_fresh`]. End users rarely call this.
pub const fn resolve() -> ResolvedPath {
	match COMPILE_TIME_LEVEL {
		Some(level) => ResolvedPath::Assumed(level),
		None => ResolvedPath::RuntimeDispatch,
	}
}

/// Raw per-feature compile-time **lower bound**: bits this compilation's
/// `-C target-feature`/`target-cpu` flags guarantee present, not a ceiling.
/// Unlike [`resolve`]'s [`GenericLevel`] bucketing (whole-bundle-or-nothing:
/// a real CPU with `avx512f`+`avx512vl` but not `avx512cd` gets classified
/// below V4 and none of its AVX-512 credit shows up there), this checks each
/// feature independently: a pinned build "only provides a shortpath for
/// its own *expected* CPU capabilities" (per feature, not per tier); the
/// real host may always have more than what's checked here, so callers
/// (`x86::detect_features`) union this with one real, cached `CPUID` probe
/// rather than treating it as authoritative on its own. Plain `#[inline]
/// fn`, not `const fn`: the fold into a `cfg!`-literal-only constant happens
/// via ordinary LLVM inlining/constant-propagation, the same mechanism
/// [`super::detect_level_fresh`] already relies on today: `Feature::bit()`
/// not being callable from a `const fn` is a non-issue here.
///
/// Covers every [`Feature`] any `x86::auto_up`/`x86::auto_down` dispatch
/// token checks (superset of the 18 features `COMPILE_TIME_LEVEL` above
/// tracks for [`GenericLevel`] bucketing, e.g. also `avx` standalone, and
/// the AVX-512/AVX extension family with no [`GenericLevel`] tier of their
/// own: BF16/BITALG/FP16/VNNI/IFMA/VPOPCNTDQ/AVX-VNNI/AVX-IFMA).
#[inline]
pub fn compile_time_features() -> FeatureSet {
	let mut set = FeatureSet::default();
	if cfg!(target_feature = "popcnt") {
		set = set.with(Feature::Popcnt);
	}
	if cfg!(target_feature = "sse3") {
		set = set.with(Feature::Sse3);
	}
	if cfg!(target_feature = "ssse3") {
		set = set.with(Feature::Ssse3);
	}
	if cfg!(target_feature = "sse4.1") {
		set = set.with(Feature::Sse41);
	}
	if cfg!(target_feature = "sse4.2") {
		set = set.with(Feature::Sse42);
	}
	if cfg!(target_feature = "avx") {
		set = set.with(Feature::Avx);
	}
	if cfg!(target_feature = "avx2") {
		set = set.with(Feature::Avx2);
	}
	if cfg!(target_feature = "bmi1") {
		set = set.with(Feature::Bmi1);
	}
	if cfg!(target_feature = "bmi2") {
		set = set.with(Feature::Bmi2);
	}
	if cfg!(target_feature = "f16c") {
		set = set.with(Feature::F16c);
	}
	if cfg!(target_feature = "fma") {
		set = set.with(Feature::Fma);
	}
	if cfg!(target_feature = "lzcnt") {
		set = set.with(Feature::Lzcnt);
	}
	if cfg!(target_feature = "movbe") {
		set = set.with(Feature::Movbe);
	}
	if cfg!(target_feature = "xsave") {
		set = set.with(Feature::Xsave);
	}
	if cfg!(target_feature = "avx512f") {
		set = set.with(Feature::Avx512f);
	}
	if cfg!(target_feature = "avx512bw") {
		set = set.with(Feature::Avx512bw);
	}
	if cfg!(target_feature = "avx512cd") {
		set = set.with(Feature::Avx512cd);
	}
	if cfg!(target_feature = "avx512dq") {
		set = set.with(Feature::Avx512dq);
	}
	if cfg!(target_feature = "avx512vl") {
		set = set.with(Feature::Avx512vl);
	}
	if cfg!(target_feature = "avx512bf16") {
		set = set.with(Feature::Avx512bf16);
	}
	if cfg!(target_feature = "avx512bitalg") {
		set = set.with(Feature::Avx512bitalg);
	}
	if cfg!(target_feature = "avx512fp16") {
		set = set.with(Feature::Avx512fp16);
	}
	if cfg!(target_feature = "avx512vnni") {
		set = set.with(Feature::Avx512vnni);
	}
	if cfg!(target_feature = "avx512ifma") {
		set = set.with(Feature::Avx512ifma);
	}
	if cfg!(target_feature = "avx512vpopcntdq") {
		set = set.with(Feature::Avx512vpopcntdq);
	}
	if cfg!(target_feature = "avxifma") {
		set = set.with(Feature::AvxIfma);
	}
	if cfg!(target_feature = "avxvnni") {
		set = set.with(Feature::AvxVnni);
	}
	set
}

/// Optional startup check: if this binary was compiled for a high tier, panic
/// when the CPU is weaker.
///
/// No-op when the build did not assume a tier ([`ResolvedPath::RuntimeDispatch`]).
/// Call once near `main` if you ship `-C target-cpu=native` (or similar) and
/// want a clear message instead of a later illegal instruction elsewhere.
///
/// # Panics
/// The binary assumes a [`GenericLevel`] the host does not fully support.
pub fn verify_or_panic() {
	if let ResolvedPath::Assumed(level) = resolve() {
		let set = FeatureSet::detect();
		if !set.contains_all(level.required_features()) {
			panic!(
				"miraculix: this binary was compiled assuming x86-64 {level:?} (via `-C \
				 target-cpu=...`, `-C target-cpu=native`, or explicit `-C target-feature=...`), \
				 but the CPU it is running on does not support that level. Recompile for a lower \
				 level (or plain `-C target-cpu=x86-64`), or run this binary on compatible \
				 hardware."
			);
		}
	}
}

#[cfg(test)]
#[path = "test/shortpath.rs"]
mod tests;
