//! Token declaration: the zero-sized proof type plus its `detect` /
//! `from_features` pair. Every Arm token whose check is "all of these feature
//! bits" is exactly this shape, so it is written once here. Hand-written
//! exceptions: aarch64 `Neon` (also has `assume_baseline`) and aarch32 `Dsp`
//! (compiled without the `v7` baseline the Neon files need).
//!
//! A token may require **several** feature bits (`requires = [Sha1, Sha2]`):
//! LLVM sometimes gates two Arm `FEAT_*`s behind one target feature (`sha2`
//! covers SHA1 and SHA256), and the token must prove every bit the OS
//! reports separately. `Feature` and `FeatureSet` come from
//! [`super::super::host`], so the names are the host arch's own.

macro_rules! arm_token {
	(
		$(#[$doc:meta])*
		token = $Token:ident, requires = [$($F:ident),+ $(,)?],
	) => {
		$(#[$doc])*
		///
		/// Zero-sized, `Copy`. Obtain via `detect()` or `from_features()`, then
		/// call methods on the token.
		#[derive(Debug, Clone, Copy)]
		pub struct $Token(());

		impl $Token {
			/// Probe once: `Some(token)` if the extension is available, else `None`.
			pub fn detect() -> Option<Self> {
				Self::from_features($crate::arm_shared::host::FeatureSet::detect())
			}

			/// Build a token from a set you already have (e.g. from the arch
			/// module's `detect_features()`).
			#[doc = concat!("Returns `None` unless every one of ", $("`Feature::", stringify!($F), "` "),+, "is present.")]
			pub fn from_features(set: $crate::arm_shared::host::FeatureSet) -> Option<Self> {
				set.contains_all(&[$($crate::arm_shared::host::Feature::$F),+]).then_some($Token(()))
			}
		}
	};
}

pub(crate) use arm_token;
