use super::*;
use crate::x86::Feature;

/// `Feature` -> its `-C target-feature` cfg name. Only covers features that
/// actually appear in some [`GenericLevel::required_features`]; the catch-all
/// is the forcing function that keeps this list honest as tiers grow.
fn cfg_enabled(feature: Feature) -> bool {
	match feature {
		Feature::Popcnt => cfg!(target_feature = "popcnt"),
		Feature::Sse3 => cfg!(target_feature = "sse3"),
		Feature::Ssse3 => cfg!(target_feature = "ssse3"),
		Feature::Sse41 => cfg!(target_feature = "sse4.1"),
		Feature::Sse42 => cfg!(target_feature = "sse4.2"),
		Feature::Avx => cfg!(target_feature = "avx"),
		Feature::Avx2 => cfg!(target_feature = "avx2"),
		Feature::Bmi1 => cfg!(target_feature = "bmi1"),
		Feature::Bmi2 => cfg!(target_feature = "bmi2"),
		Feature::F16c => cfg!(target_feature = "f16c"),
		Feature::Fma => cfg!(target_feature = "fma"),
		Feature::Lzcnt => cfg!(target_feature = "lzcnt"),
		Feature::Movbe => cfg!(target_feature = "movbe"),
		Feature::Xsave => cfg!(target_feature = "xsave"),
		Feature::Avx512f => cfg!(target_feature = "avx512f"),
		Feature::Avx512bw => cfg!(target_feature = "avx512bw"),
		Feature::Avx512cd => cfg!(target_feature = "avx512cd"),
		Feature::Avx512dq => cfg!(target_feature = "avx512dq"),
		Feature::Avx512vl => cfg!(target_feature = "avx512vl"),
		Feature::Avx512bf16 => cfg!(target_feature = "avx512bf16"),
		Feature::Avx512bitalg => cfg!(target_feature = "avx512bitalg"),
		Feature::Avx512fp16 => cfg!(target_feature = "avx512fp16"),
		Feature::Avx512vnni => cfg!(target_feature = "avx512vnni"),
		Feature::Avx512ifma => cfg!(target_feature = "avx512ifma"),
		Feature::Avx512vpopcntdq => cfg!(target_feature = "avx512vpopcntdq"),
		Feature::AvxIfma => cfg!(target_feature = "avxifma"),
		Feature::AvxVnni => cfg!(target_feature = "avxvnni"),
		other => unimplemented!(
			"cfg_enabled: {other:?} appears in a GenericLevel::required_features() list, or in \
			 compile_time_features()'s coverage, but has no target_feature cfg arm here - add \
			 one, this test is what keeps shortpath.rs's cfg blocks honest against tiers.rs / \
			 x86::auto_up's dispatch tokens"
		),
	}
}

/// [`compile_time_features`]'s per-feature analog of
/// `compile_time_level_matches_required_features_via_cfg` below: every bit it
/// sets (or doesn't) must agree with the same `cfg!(target_feature = "...")` this
/// compilation actually has, independent of host hardware.
#[test]
fn compile_time_features_matches_cfg() {
	const COVERED: &[Feature] = &[
		Feature::Popcnt, Feature::Sse3, Feature::Ssse3, Feature::Sse41, Feature::Sse42, Feature::Avx,
		Feature::Avx2, Feature::Bmi1, Feature::Bmi2, Feature::F16c, Feature::Fma, Feature::Lzcnt,
		Feature::Movbe, Feature::Xsave, Feature::Avx512f, Feature::Avx512bw, Feature::Avx512cd,
		Feature::Avx512dq, Feature::Avx512vl, Feature::Avx512bf16, Feature::Avx512bitalg,
		Feature::Avx512fp16, Feature::Avx512vnni, Feature::Avx512ifma, Feature::Avx512vpopcntdq,
		Feature::AvxIfma, Feature::AvxVnni,
	];
	let set = compile_time_features();
	for &feature in COVERED {
		assert_eq!(
			set.contains(feature),
			cfg_enabled(feature),
			"compile_time_features() disagrees with cfg!(target_feature) for {feature:?}"
		);
	}
}

/// The real regression guard for the "shortpath.rs cfg lists must stay hand-synced
/// with `GenericLevel::required_features()`" risk (see module doc above). Pure
/// `cfg!`/`const` comparison, independent of which CPU runs the test, catches a
/// drift for *this compilation's* `-C target-feature` set regardless of host
/// hardware. Zero runtime cost: `#[cfg(test)]` only, never compiled into a shipped
/// build.
#[test]
fn compile_time_level_matches_required_features_via_cfg() {
	let expected = GenericLevel::ALL
		.iter()
		.rev()
		.copied()
		.find(|level| level.required_features().iter().all(|&f| cfg_enabled(f)))
		.unwrap_or(GenericLevel::V1);

	match resolve() {
		ResolvedPath::Assumed(level) => assert_eq!(
			level, expected,
			"shortpath.rs cfg blocks disagree with GenericLevel::required_features() for this \
			 compilation's target-feature set"
		),
		// shortpath.rs never emits Assumed(V1) (V1 needs nothing extra, so there is no
		// dedicated compile-time branch for it; RuntimeDispatch covers it via CPUID).
		ResolvedPath::RuntimeDispatch => assert_eq!(
			expected,
			GenericLevel::V1,
			"cfg flags satisfy {expected:?} per required_features(), but shortpath.rs fell \
			 back to RuntimeDispatch instead of assuming it"
		),
	}
}
