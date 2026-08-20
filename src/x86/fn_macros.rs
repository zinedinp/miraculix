//! # Multi-op kernel macros (end-user)
//!
//! Use these when one kernel body calls **several** token methods and must
//! compile as a single `#[target_feature]` function (otherwise the compiler
//! may leave `callq` edges between ops and emit scalar loads/stores).
//!
//! ## How to use
//!
//! 1. Obtain the matching tokens (`Avx::detect()`, ...).
//! 2. Define the kernel with the macro (outer function stays safe).
//! 3. Call the outer function only when those tokens are in hand.
//!
//! ```
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::ops::avx::avx::Avx;
//! miraculix::avx_fn! {
//!     fn add_mul(avx: Avx, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
//!         let t = avx.add_f32x8(a, b);
//!         avx.mul_f32x8(t, c)
//!     }
//! }
//! if let Some(avx) = Avx::detect() {
//!     let _ = add_mul(avx, [1.0; 8], [2.0; 8], [3.0; 8]);
//! }
//! # }
//! ```
//!
//! ## Catalog
//!
//! Pick the **narrowest** macro whose enabled set covers every token method
//! the body calls. Wider bundles (e.g. [`avx512_v4_fn!`]) are fine when you
//! already hold a full tier token set; prefer a tight match for smaller
//! compile units and clearer proofs.
//!
//! ### SSE family
//!
//! | Macro | Features enabled | Hold these tokens |
//! |---|---|---|
//! | [`ssse3_fn!`] | `ssse3` | `Ssse3` |
//! | [`sse41_fn!`] | `sse2,ssse3,sse4.1` | `Sse2`, `Ssse3`, `Sse41` |
//! | [`sse41_f16c_fn!`] | `sse2,ssse3,sse4.1,f16c` | + `F16c` |
//! | [`sse42_fn!`] | `sse4.2` | `Sse42` |
//!
//! Baseline x86-64 already has `sse`/`sse2`, so pure SSE2 multi-op bodies
//! usually do **not** need a trampoline; the optional-feature rows above do.
//!
//! ### AVX / FMA / VNNI (V3-class)
//!
//! | Macro | Features enabled | Hold these tokens |
//! |---|---|---|
//! | [`avx_fn!`] | `avx` | `Avx` |
//! | [`f16c_fn!`] | `f16c` | `F16c` |
//! | [`fma_fn!`] | `fma` | `Fma` |
//! | [`avx_fma_fn!`] | `avx,fma` | `Avx`, `Fma` |
//! | [`avx2_fn!`] | `avx2,sse2,ssse3` | `Avx2`, `Sse2`, `Ssse3` |
//! | [`avx2_fma_fn!`] | `avx2,fma,sse2,ssse3` | + `Fma` |
//! | [`avx_v3_fn!`] | V3 SIMD bundle (see below) | V3-class tokens you use |
//! | [`avx_vnni_fn!`] | `avxvnni` | `AvxVnni` |
//!
//! `avx_v3_fn!` enables:
//! `sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,f16c`
//! (the SIMD/bitmanip core of `-march=x86-64-v3`).
//!
//! ### AVX-512
//!
//! | Macro | Features enabled | Hold these tokens |
//! |---|---|---|
//! | [`avx512f_fn!`] | `avx512f` | `Avx512f` |
//! | [`avx512_fn!`] | `avx512f,avx,avx512dq` | `Avx512f`, `Avx`, `Avx512Dq` |
//! | [`avx512bw_fn!`] | `avx512f,avx512bw,avx2,sse2,ssse3` | all five |
//! | [`avx512vl_fn!`] | `avx512f,avx512vl` | `Avx512f` + VL token(s) |
//! | [`avx512vnni_fn!`] | `avx512vnni` | `Avx512Vnni` |
//! | [`avx512vbmi2_fn!`] | `avx512vbmi2` | `Avx512Vbmi2` |
//! | [`avx512fp16_fn!`] | `avx512fp16` | `Avx512Fp16` |
//! | [`avx512bf16_fn!`] | `avx512bf16,avx512f` | `Avx512Bf16` |
//! | [`avx512_v4_fn!`] | full V4 SIMD bundle (see below) | V4-class tokens you use |
//!
//! `avx512_v4_fn!` enables:
//! `sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl`
//! (same shape as pulp's `v4_fn!` / `-march=x86-64-v4` SIMD core).
//!
//! ### Crypto / specialty
//!
//! | Macro | Features enabled | Hold these tokens |
//! |---|---|---|
//! | [`aes_fn!`] | `aes` | `Aes` |
//! | [`pclmulqdq_fn!`] | `pclmulqdq` | `Pclmulqdq` |
//! | [`aes_pclmul_fn!`] | `aes,pclmulqdq` | `Aes`, `Pclmulqdq` |
//! | [`sha_fn!`] | `sha` | `Sha` |
//! | [`gfni_fn!`] | `gfni` | `Gfni` |
//! | [`gfni_avx_fn!`] | `gfni,avx` | `Gfni` (+ AVX for 256-bit) |
//! | [`gfni512_fn!`] | `gfni,avx512f` | `Gfni512` |
//! | [`vaes_fn!`] | `vaes` | `Vaes` |
//! | [`vaes512_fn!`] | `vaes,avx512f` | `Vaes512` |
//!
//! ### Not every token needs its own macro
//!
//! Single-feature tokens whose methods already carry their own
//! `#[target_feature]` still need a trampoline **when you compose several
//! of them in one free function**. If a combination is missing here, either:
//!
//! - use the next wider bundle that already includes those features, or
//! - open a PR with a new one-line wrapper around [`__miraculix_tf_fn!`].
//!
//! Exotic / rarely composed tokens (RTM, AMD-only XOP/FMA4/3DNow, Phi-only
//! 4FMAPS/4VNNIW, AMX tile config, etc) stay out of this list until a real
//! multi-op consumer needs them.

/// Shared trampoline body used by every `*_fn!` macro.
///
/// Prefer the named macros in the catalog (`avx_fn!`, etc). Call this only
/// if you need a feature string that is not listed yet.
///
/// Syntax: `__miraculix_tf_fn!("feat1,feat2" ; fn name(...) { ... })`
#[doc(hidden)]
#[macro_export]
macro_rules! __miraculix_tf_fn {
	($features:literal ; $(#[$attr:meta])* $vis:vis fn $name:ident $(<$($gen:tt),* $(,)?>)? ($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
		$(#[$attr])*
		$vis fn $name $(<$($gen),*>)? ($($arg : $ty),*) $(-> $ret)? {
			#[target_feature(enable = $features)]
			#[inline]
			unsafe fn __impl $(<$($gen),*>)? ($($arg : $ty),*) $(-> $ret)? {
				$body
			}
			// Safety: callers only reach a named `*_fn!` wrapper while holding
			// tokens that prove every feature in `$features`.
			#[allow(unused_unsafe)]
			unsafe { __impl($($arg),*) }
		}
	};
}

// SSE family

/// Wrap a body under `ssse3` (byte shuffle / `palignr` kernels).
///
/// Hold an [`crate::x86::ops::sse::ssse3::Ssse3`] token.
#[macro_export]
macro_rules! ssse3_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "ssse3" ; $($tt)* } };
}

/// Wrap a body under `sse2`, `ssse3`, and `sse4.1`.
///
/// Hold `Sse2`, `Ssse3`, and `Sse41`. Shuffle/blend networks without
/// half-float convert; for F16C add-on use [`sse41_f16c_fn!`].
#[macro_export]
macro_rules! sse41_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse2,ssse3,sse4.1" ; $($tt)* } };
}

/// Wrap a body under `sse2`, `ssse3`, `sse4.1`, and `f16c`.
///
/// Hold `Sse2`, `Ssse3`, `Sse41`, and `F16c`. Useful for SSE shuffle
/// networks that finish with a half-float convert.
#[macro_export]
macro_rules! sse41_f16c_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse2,ssse3,sse4.1,f16c" ; $($tt)* } };
}

/// Wrap a body under `sse4.2` (CRC32 / 64-bit compares).
///
/// Hold an [`crate::x86::ops::sse::sse42::Sse42`] token.
#[macro_export]
macro_rules! sse42_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse4.2" ; $($tt)* } };
}

// AVX / FMA / VNNI (V3-class)

/// Wrap a multi-op body under `avx`.
///
/// Hold an [`crate::x86::ops::avx::avx::Avx`] token. Prefer this for multi-op
/// AVX kernels; single slice ops can use [`crate::x86::auto_up`].
#[macro_export]
macro_rules! avx_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx" ; $($tt)* } };
}

/// Wrap a body under `f16c` (half-float convert). Hold an `F16c` token.
#[macro_export]
macro_rules! f16c_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "f16c" ; $($tt)* } };
}

/// Wrap a body under `fma` (FMA3 fused multiply-add).
///
/// Hold an [`crate::x86::ops::avx::fma::Fma`] token. For mixed AVX math
/// + FMA prefer [`avx_fma_fn!`].
#[macro_export]
macro_rules! fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "fma" ; $($tt)* } };
}

/// Wrap a body under `avx` and `fma` together.
///
/// Hold `Avx` and `Fma`. Typical float kernels that mix `add`/`mul` with
/// fused `fmadd`.
#[macro_export]
macro_rules! avx_fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx,fma" ; $($tt)* } };
}

/// Wrap a body under `avx2`, `sse2`, and `ssse3`.
///
/// Hold `Avx2`, `Sse2`, and `Ssse3`. Common for AVX2 kernels that still
/// use 128-bit extract/shuffle helpers.
#[macro_export]
macro_rules! avx2_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx2,sse2,ssse3" ; $($tt)* } };
}

/// Wrap a body under `avx2`, `fma`, `sse2`, and `ssse3`.
///
/// Hold `Avx2`, `Fma`, `Sse2`, and `Ssse3`. V3-class int+float kernels
/// that also touch narrow SSE helpers.
#[macro_export]
macro_rules! avx2_fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx2,fma,sse2,ssse3" ; $($tt)* } };
}

/// Wrap a body under the x86-64-v3 SIMD/bitmanip core.
///
/// Enables `sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,f16c`.
/// Hold tokens for every extension the body actually calls. Prefer a
/// narrower macro when the kernel only needs one or two of those.
#[macro_export]
macro_rules! avx_v3_fn {
	($($tt:tt)*) => {
		$crate::__miraculix_tf_fn! {
			"sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,f16c" ; $($tt)*
		}
	};
}

/// Wrap a body under `avxvnni` (AVX2-width VNNI, no AVX-512 needed).
///
/// Hold an [`crate::x86::ops::avx::avx_vnni::AvxVnni`] token.
#[macro_export]
macro_rules! avx_vnni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avxvnni" ; $($tt)* } };
}

// AVX-512

/// Wrap a body under `avx512f` only.
///
/// Hold an `Avx512f` token. Does **not** enable DQ/BW/VL: use
/// [`avx512_fn!`], [`avx512bw_fn!`], or [`avx512vl_fn!`] if the body also
/// needs those.
#[macro_export]
macro_rules! avx512f_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f" ; $($tt)* } };
}

/// Wrap a body under `avx512f`, `avx`, and `avx512dq` together.
///
/// Hold `Avx512f`, `Avx`, and `Avx512Dq`. Typical for wide float kernels
/// that mix ZMM math with 256-bit extract/insert.
#[macro_export]
macro_rules! avx512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f,avx,avx512dq" ; $($tt)* } };
}

/// Wrap a body under `avx512f`, `avx512bw`, `avx2`, `sse2`, and `ssse3`.
///
/// Hold all five tokens. For wide byte-lane kernels with narrower remainders.
#[macro_export]
macro_rules! avx512bw_fn {
	($($tt:tt)*) => {
		$crate::__miraculix_tf_fn! { "avx512f,avx512bw,avx2,sse2,ssse3" ; $($tt)* }
	};
}

/// Wrap a body under `avx512f` and `avx512vl` (128/256-bit EVEX forms).
///
/// Hold `Avx512f` plus the VL token(s) you call (`Avx512FVl`,
/// `Avx512BwVl`, etc).
#[macro_export]
macro_rules! avx512vl_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f,avx512vl" ; $($tt)* } };
}

/// Wrap a body under `avx512vnni` (512-bit VNNI).
///
/// Hold an [`crate::x86::ops::avx512::avx512vnni::Avx512Vnni`] token.
#[macro_export]
macro_rules! avx512vnni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512vnni" ; $($tt)* } };
}

/// Wrap a body under `avx512vbmi2` (compress/expand, variable shifts).
///
/// Hold an [`crate::x86::ops::avx512::avx512vbmi2::Avx512Vbmi2`] token.
#[macro_export]
macro_rules! avx512vbmi2_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512vbmi2" ; $($tt)* } };
}

/// Wrap a body under `avx512fp16` (native half-precision arithmetic).
///
/// Hold an [`crate::x86::ops::avx512::avx512fp16::Avx512Fp16`] token.
/// VL half-width forms also need `avx512vl` on the callee; if you compose
/// both and hit a gap, open a dedicated `avx512fp16_vl_fn!`.
#[macro_export]
macro_rules! avx512fp16_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512fp16" ; $($tt)* } };
}

/// Wrap a body under `avx512bf16` and `avx512f`.
///
/// Hold an [`crate::x86::ops::avx512::avx512bf16::Avx512Bf16`] token
/// (methods already require both features).
#[macro_export]
macro_rules! avx512bf16_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512bf16,avx512f" ; $($tt)* } };
}

/// Wrap a body under the full x86-64-v4 SIMD core (pulp `v4_fn!` shape).
///
/// Enables
/// `sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl`.
/// Hold tokens for every extension the body actually calls. Prefer a
/// narrower macro when possible.
#[macro_export]
macro_rules! avx512_v4_fn {
	($($tt:tt)*) => {
		$crate::__miraculix_tf_fn! {
			"sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl" ; $($tt)*
		}
	};
}

// Crypto / specialty

/// Wrap a body under `aes` (AES-NI round chain).
///
/// Hold an [`crate::x86::ops::other::aes::Aes`] token.
#[macro_export]
macro_rules! aes_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "aes" ; $($tt)* } };
}

/// Wrap a body under `pclmulqdq` (carry-less multiply).
///
/// Hold a [`crate::x86::ops::other::pclmulqdq::Pclmulqdq`] token.
#[macro_export]
macro_rules! pclmulqdq_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "pclmulqdq" ; $($tt)* } };
}

/// Wrap a body under `aes` and `pclmulqdq` (AES-GCM style kernels).
///
/// Hold `Aes` and `Pclmulqdq`.
#[macro_export]
macro_rules! aes_pclmul_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "aes,pclmulqdq" ; $($tt)* } };
}

/// Wrap a body under `sha` (SHA-NI multi-round).
///
/// Hold a [`crate::x86::ops::other::sha::Sha`] token.
#[macro_export]
macro_rules! sha_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sha" ; $($tt)* } };
}

/// Wrap a body under `gfni` (128-bit GFNI forms).
///
/// Hold a [`crate::x86::ops::other::gfni::Gfni`] token. For 256-bit GFNI
/// methods that also need AVX, use [`gfni_avx_fn!`].
#[macro_export]
macro_rules! gfni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni" ; $($tt)* } };
}

/// Wrap a body under `gfni` and `avx` (256-bit GFNI).
///
/// Hold a `Gfni` token; body may call 256-bit `gf2p8mul` / affine forms.
#[macro_export]
macro_rules! gfni_avx_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni,avx" ; $($tt)* } };
}

/// Wrap a body under `gfni` and `avx512f` (512-bit GFNI).
///
/// Hold a [`crate::x86::ops::other::gfni::Gfni512`] token.
#[macro_export]
macro_rules! gfni512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni,avx512f" ; $($tt)* } };
}

/// Wrap a body under `vaes` (256-bit vector AES).
///
/// Hold a [`crate::x86::ops::other::vaes::Vaes`] token.
#[macro_export]
macro_rules! vaes_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "vaes" ; $($tt)* } };
}

/// Wrap a body under `vaes` and `avx512f` (512-bit vector AES).
///
/// Hold a [`crate::x86::ops::other::vaes::Vaes512`] token.
#[macro_export]
macro_rules! vaes512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "vaes,avx512f" ; $($tt)* } };
}

#[cfg(test)]
#[path = "test/fn_macros.rs"]
mod tests;
