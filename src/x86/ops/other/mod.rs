//! Guide "Other" + non-Intel extensions:
//! AMD [`amd3dnow`]/[`sse4a`]/[`fma4`]/[`xop`]; RTM [`rtm`];
//! [`popcnt`]/[`aes`]/[`pclmulqdq`]/[`gfni`]/[`vaes`]/[`vpclmulqdq`]/[`sha`].
//! See `resources/Intel Intrinsics Guide/index.html`.

pub mod aes;
pub mod amd3dnow;
pub mod fma4;
pub mod gfni;
pub mod pclmulqdq;
pub mod popcnt;
pub mod sha;
pub mod sse4a;
pub mod vaes;
pub mod vpclmulqdq;
pub mod xop;

#[cfg(feature = "nightly-rtm")]
pub mod rtm;
