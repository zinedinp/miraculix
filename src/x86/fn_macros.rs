//! # Multi-op kernel macros (end-user)
//!
//! Use these when one kernel body calls **several** token methods and must
//! compile as a single `#[target_feature]` function (otherwise the compiler
//! may leave `callq` edges between ops and emit scalar loads/stores).
//!
//! ## How to use
//!
//! 1. Obtain a token (`Avx::detect()`, `V3::detect()`, ...).
//! 2. Define the kernel with the macro. **The first parameter must be a
//!    token that proves every feature the macro enables** (column "First
//!    parameter" below). Anything else is a compile error, so the generated
//!    function is safe to call: you cannot reach it without detection.
//! 3. Pass any further tokens you want to call methods on as later
//!    parameters.
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
//! A first parameter that does not prove the features is rejected:
//!
//! ```compile_fail
//! # #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
//! use miraculix::x86::ops::avx::avx::Avx;
//! // `avx512f_fn!` enables AVX-512F, but an `Avx` token does not prove it.
//! miraculix::avx512f_fn! {
//!     fn bad(avx: Avx, a: [f32; 8]) -> [f32; 8] { avx.add_f32x8(a, a) }
//! }
//! # }
//! # #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
//! # compile_error!("x86 only");
//! ```
//!
//! "Proves" follows LLVM's feature implications, the same rule
//! `#[target_feature]` itself uses: `avx2` implies `avx` and the SSE family,
//! `fma` and `f16c` imply `avx`, `avx512f` implies `avx2`/`fma`/`f16c`,
//! `avx512vbmi` implies `avx512bw`, and so on. Two features with no
//! implication between them take a tuple: `(Avx2, Fma)`, `(Aes, Pclmulqdq)`.
//!
//! ## Generics
//!
//! Kernels may be generic over const parameters, lifetimes and type
//! parameters with a single bound: `fn f<const K: usize>(t: Avx2, a: [u32; K])`.
//! Multi-bound lists (`T: A + B`) and `where` clauses are not supported.
//!
//! ## Catalog
//!
//! Pick the **narrowest** macro whose enabled set covers every token method
//! the body calls. Bundles ([`crate::x86::bundles`]) prove every macro of
//! their level and below.
//!
//! ### SSE family
//!
//! | Macro | Features enabled | First parameter |
//! |---|---|---|
//! | [`ssse3_fn!`] | `ssse3` | `Ssse3`, `Sse41`, `Sse42`, `Avx`, `Avx2`, a bundle |
//! | [`sse41_fn!`] | `sse2,ssse3,sse4.1` | `Sse41`, `Sse42`, `Avx`, `Avx2`, a bundle |
//! | [`sse41_f16c_fn!`] | `sse2,ssse3,sse4.1,f16c` | `F16c`, a bundle |
//! | [`sse42_fn!`] | `sse4.2` | `Sse42`, `Avx`, `Avx2`, a bundle |
//!
//! Baseline x86-64 already has `sse`/`sse2`, so pure SSE2 multi-op bodies
//! usually do **not** need a trampoline; the optional-feature rows above do.
//!
//! ### AVX / FMA / VNNI (V3-class)
//!
//! | Macro | Features enabled | First parameter |
//! |---|---|---|
//! | [`avx_fn!`] | `avx` | `Avx`, `Avx2`, `Fma`, `F16c`, a bundle |
//! | [`f16c_fn!`] | `f16c` | `F16c`, a bundle |
//! | [`fma_fn!`] | `fma` | `Fma`, a bundle |
//! | [`avx_fma_fn!`] | `avx,fma` | `Fma`, a bundle |
//! | [`avx2_fn!`] | `avx2,sse2,ssse3` | `Avx2`, a bundle |
//! | [`avx2_fma_fn!`] | `avx2,fma,sse2,ssse3` | `(Avx2, Fma)`, a bundle |
//! | [`avx_v3_fn!`] | V3 SIMD bundle (see below) | `V3`, `V4`, `V4Vbmi` |
//! | [`avx_vnni_fn!`] | `avxvnni` | `AvxVnni` |
//!
//! `avx_v3_fn!` enables:
//! `sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,f16c`
//! (the SIMD/bitmanip core of `-march=x86-64-v3`).
//!
//! ### AVX-512
//!
//! | Macro | Features enabled | First parameter |
//! |---|---|---|
//! | [`avx512f_fn!`] | `avx512f` | `Avx512f`, `V4`, `V4Vbmi` |
//! | [`avx512_fn!`] | `avx512f,avx,avx512dq` | `Avx512Dq`, `V4`, `V4Vbmi` |
//! | [`avx512bw_fn!`] | `avx512f,avx512bw,avx2,sse2,ssse3` | `Avx512Bw`, `V4`, `V4Vbmi` |
//! | [`avx512vl_fn!`] | `avx512f,avx512vl` | `Avx512FVl`, `V4`, `V4Vbmi` |
//! | [`avx512vnni_fn!`] | `avx512vnni` | `Avx512Vnni` |
//! | [`avx512vbmi_fn!`] | `avx512f,avx512bw,avx512vbmi` | `Avx512Vbmi`, `V4Vbmi` |
//! | [`avx512_v4_vbmi_fn!`] | V4 bundle + `avx512vbmi` | `V4Vbmi` |
//! | [`avx512vbmi2_fn!`] | `avx512vbmi2` | `Avx512Vbmi2` |
//! | [`avx512fp16_fn!`] | `avx512fp16` | `Avx512Fp16` |
//! | [`avx512bf16_fn!`] | `avx512bf16,avx512f` | `Avx512Bf16` |
//! | [`avx512_v4_fn!`] | full V4 SIMD bundle (see below) | `V4`, `V4Vbmi` |
//!
//! `avx512_v4_fn!` enables:
//! `sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl`
//! (same shape as pulp's `v4_fn!` / `-march=x86-64-v4` SIMD core).
//!
//! ### Crypto / specialty
//!
//! | Macro | Features enabled | First parameter |
//! |---|---|---|
//! | [`aes_fn!`] | `aes` | `Aes` |
//! | [`pclmulqdq_fn!`] | `pclmulqdq` | `Pclmulqdq` |
//! | [`aes_pclmul_fn!`] | `aes,pclmulqdq` | `(Aes, Pclmulqdq)` |
//! | [`sha_fn!`] | `sha` | `Sha` |
//! | [`gfni_fn!`] | `gfni` | `Gfni` |
//! | [`gfni_avx_fn!`] | `gfni,avx` | `(Gfni, Avx)` (GFNI exists without AVX, e.g. Tremont) |
//! | [`gfni512_fn!`] | `gfni,avx512f` | `Gfni512` |
//! | [`vaes_fn!`] | `vaes` | `Vaes` |
//! | [`vaes512_fn!`] | `vaes,avx512f` | `Vaes512` |
//!
//! ### Not every token needs its own macro
//!
//! Single-feature tokens whose methods already carry their own
//! `#[target_feature]` still need a trampoline **when you compose several
//! of them in one free function**. If a combination is missing here, use
//! the next wider macro your first parameter proves, or open a PR adding a
//! named macro, its marker and the types that prove it (`__private` below).
//!
//! Exotic / rarely composed tokens (RTM, AMD-only XOP/FMA4/3DNow, Phi-only
//! 4FMAPS/4VNNIW, AMX tile config, etc) stay out of this list until a real
//! multi-op consumer needs them.

/// Compile-time proof that a kernel's first parameter covers the features
/// its macro enables. **Not public API**: only the `*_fn!` expansions use it.
///
/// `Proves<M>` is sealed (its supertrait lives in a private module), so only
/// the token types listed here can satisfy it. Each marker in `m` is one
/// macro; a type proves it when holding that type guarantees every feature
/// the macro enables. Implications follow LLVM's (and so `#[target_feature]`'s)
/// own: `avx2` implies `avx` and the SSE family, `avx512f` implies `avx2`,
/// `fma` and `f16c`, `avx512vbmi` implies `avx512bw`, and so on. That is the
/// same rule `std::is_x86_feature_detected!` + `#[target_feature]` rely on.
#[doc(hidden)]
pub mod __private {
    use crate::x86::bundles::{V3, V4, V4Vbmi};
    use crate::x86::ops::avx::{avx::Avx, avx_vnni::AvxVnni, avx2::Avx2, f16c::F16c, fma::Fma};
    use crate::x86::ops::avx512::{
        avx512bf16::Avx512Bf16, avx512bw::Avx512Bw, avx512dq::Avx512Dq, avx512f::Avx512f,
        avx512fp16::Avx512Fp16, avx512vbmi::Avx512Vbmi, avx512vbmi2::Avx512Vbmi2,
        avx512vl::Avx512FVl, avx512vnni::Avx512Vnni,
    };
    use crate::x86::ops::other::{
        aes::Aes,
        gfni::{Gfni, Gfni512},
        pclmulqdq::Pclmulqdq,
        sha::Sha,
        vaes::{Vaes, Vaes512},
    };
    use crate::x86::ops::sse::{sse41::Sse41, sse42::Sse42, ssse3::Ssse3};

    mod sealed {
        pub trait Sealed {}
    }

    /// `Self` proves every feature of the macro marked by `M`.
    pub trait Proves<M>: sealed::Sealed {}

    /// Type check only: compiles iff `P` proves `M`. Inlines to nothing.
    #[inline(always)]
    pub fn require_proof<M, P: Proves<M>>(_: &P) {}

    macro_rules! sealed {
        ($($t:ty),+ $(,)?) => { $(impl sealed::Sealed for $t {})+ };
    }
    sealed!(
        Ssse3,
        Sse41,
        Sse42,
        Avx,
        Avx2,
        Fma,
        F16c,
        AvxVnni,
        Avx512f,
        Avx512Dq,
        Avx512Bw,
        Avx512FVl,
        Avx512Vnni,
        Avx512Vbmi,
        Avx512Vbmi2,
        Avx512Fp16,
        Avx512Bf16,
        Aes,
        Pclmulqdq,
        Sha,
        Gfni,
        Gfni512,
        Vaes,
        Vaes512,
        V3,
        V4,
        V4Vbmi,
        (Avx2, Fma),
        (Aes, Pclmulqdq),
        (Gfni, Avx),
    );

    /// One zero-sized marker per `*_fn!` macro.
    pub mod m {
        macro_rules! markers {
            ($($m:ident),+ $(,)?) => { $(#[allow(missing_debug_implementations)] pub struct $m;)+ };
        }
        markers!(
            Ssse3Fn,
            Sse41Fn,
            Sse41F16cFn,
            Sse42Fn,
            AvxFn,
            F16cFn,
            FmaFn,
            AvxFmaFn,
            Avx2Fn,
            Avx2FmaFn,
            AvxV3Fn,
            AvxVnniFn,
            Avx512fFn,
            Avx512Fn,
            Avx512bwFn,
            Avx512vlFn,
            Avx512vnniFn,
            Avx512vbmiFn,
            Avx512vbmi2Fn,
            Avx512fp16Fn,
            Avx512bf16Fn,
            Avx512V4Fn,
            Avx512V4VbmiFn,
            AesFn,
            PclmulqdqFn,
            AesPclmulFn,
            ShaFn,
            GfniFn,
            GfniAvxFn,
            Gfni512Fn,
            VaesFn,
            Vaes512Fn,
        );
    }

    macro_rules! proves {
        ($($m:ident: $($t:ty),+;)+) => { $($(impl Proves<m::$m> for $t {})+)+ };
    }
    proves! {
        Ssse3Fn: Ssse3, Sse41, Sse42, Avx, Avx2, V3, V4, V4Vbmi;
        Sse41Fn: Sse41, Sse42, Avx, Avx2, V3, V4, V4Vbmi;
        Sse41F16cFn: F16c, V3, V4, V4Vbmi;
        Sse42Fn: Sse42, Avx, Avx2, V3, V4, V4Vbmi;
        AvxFn: Avx, Avx2, Fma, F16c, V3, V4, V4Vbmi;
        F16cFn: F16c, V3, V4, V4Vbmi;
        FmaFn: Fma, V3, V4, V4Vbmi;
        AvxFmaFn: Fma, V3, V4, V4Vbmi;
        Avx2Fn: Avx2, V3, V4, V4Vbmi;
        Avx2FmaFn: (Avx2, Fma), V3, V4, V4Vbmi;
        AvxV3Fn: V3, V4, V4Vbmi;
        AvxVnniFn: AvxVnni;
        Avx512fFn: Avx512f, V4, V4Vbmi;
        Avx512Fn: Avx512Dq, V4, V4Vbmi;
        Avx512bwFn: Avx512Bw, V4, V4Vbmi;
        Avx512vlFn: Avx512FVl, V4, V4Vbmi;
        Avx512vnniFn: Avx512Vnni;
        Avx512vbmiFn: Avx512Vbmi, V4Vbmi;
        Avx512vbmi2Fn: Avx512Vbmi2;
        Avx512fp16Fn: Avx512Fp16;
        Avx512bf16Fn: Avx512Bf16;
        Avx512V4Fn: V4, V4Vbmi;
        Avx512V4VbmiFn: V4Vbmi;
        AesFn: Aes;
        PclmulqdqFn: Pclmulqdq;
        AesPclmulFn: (Aes, Pclmulqdq);
        ShaFn: Sha;
        GfniFn: Gfni;
        GfniAvxFn: (Gfni, Avx);
        Gfni512Fn: Gfni512;
        VaesFn: Vaes;
        Vaes512Fn: Vaes512;
    }
}

/// Shared trampoline body used by every `*_fn!` macro. **Not public API.**
///
/// Use the named macros in the catalog (`avx_fn!`, etc). A new feature
/// combination needs a new named macro plus a marker and its proving types
/// in [`__private`], so the "first parameter proves the features" check
/// covers it.
///
/// Syntax: `__miraculix_tf_fn!("feat1,feat2", Marker ; fn name(t: Token, ...) { ... })`
///
/// Generics are supported: const parameters (`<const K: usize>`), lifetimes
/// (`<'a>`) and type parameters with at most one bound (`<T: Copy>`, a bound
/// may itself be generic, `T: Into<u32>`), in any mix. Multi-bound lists
/// (`T: A + B`) and `where` clauses are not.
#[doc(hidden)]
#[macro_export]
macro_rules! __miraculix_tf_fn {
	// No generics.
	($features:literal, $proof:path ; $(#[$attr:meta])* $vis:vis fn $name:ident ($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
		$crate::__miraculix_tf_emit! {
			$features, $proof ; [$(#[$attr])*] $vis $name [] [] ($($arg : $ty),*) [$(-> $ret)?] $body
		}
	};
	// Generics: walk the parameter list, accumulating it as plain tokens.
	($features:literal, $proof:path ; $(#[$attr:meta])* $vis:vis fn $name:ident < $($rest:tt)*) => {
		$crate::__miraculix_tf_gen! {
			$features, $proof ; [$(#[$attr])*] $vis $name [] [] $($rest)*
		}
	};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __miraculix_tf_gen {
	// const N: T ,   /   const N: T >
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] const $c:ident : $t:ty , $($r:tt)*) => {
		$crate::__miraculix_tf_gen! { $f, $pf ; $at $v $n [$($acc)* const $c : $t  ,] [$($tf)* $c ,] $($r)* }
	};
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] const $c:ident : $t:ty > $($r:tt)*) => {
		$crate::__miraculix_tf_args! { $f, $pf ; $at $v $n [$($acc)* const $c : $t  ,] [$($tf)* $c ,] $($r)* }
	};
	// 'a
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $l:lifetime , $($r:tt)*) => {
		$crate::__miraculix_tf_gen! { $f, $pf ; $at $v $n [$($acc)* $l  ,] [$($tf)*] $($r)* }
	};
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $l:lifetime > $($r:tt)*) => {
		$crate::__miraculix_tf_args! { $f, $pf ; $at $v $n [$($acc)* $l  ,] [$($tf)*] $($r)* }
	};
	// T: Bound
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $p:ident : $b:path , $($r:tt)*) => {
		$crate::__miraculix_tf_gen! { $f, $pf ; $at $v $n [$($acc)* $p : $b  ,] [$($tf)* $p ,] $($r)* }
	};
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $p:ident : $b:path > $($r:tt)*) => {
		$crate::__miraculix_tf_args! { $f, $pf ; $at $v $n [$($acc)* $p : $b  ,] [$($tf)* $p ,] $($r)* }
	};
	// T
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $p:ident , $($r:tt)*) => {
		$crate::__miraculix_tf_gen! { $f, $pf ; $at $v $n [$($acc)* $p  ,] [$($tf)* $p ,] $($r)* }
	};
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] $p:ident > $($r:tt)*) => {
		$crate::__miraculix_tf_args! { $f, $pf ; $at $v $n [$($acc)* $p  ,] [$($tf)* $p ,] $($r)* }
	};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __miraculix_tf_args {
	($f:literal, $pf:path ; $at:tt $v:vis $n:ident [$($acc:tt)*] [$($tf:tt)*] ($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
		$crate::__miraculix_tf_emit! { $f, $pf ; $at $v $n [$($acc)*] [$($tf)*] ($($arg : $ty),*) [$(-> $ret)?] $body }
	};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __miraculix_tf_emit {
	// No parameters means no token, so nothing proves the features.
	($features:literal, $proof:path ; [$($attr:tt)*] $vis:vis $name:ident [$($gen:tt)*] [$($tf:tt)*] () [$($ret:tt)*] $body:block) => {
		::core::compile_error!(
			"a `*_fn!` kernel takes its proof token as the first parameter, e.g. `fn k(t: Avx2, a: [i32; 8])`"
		);
	};
	($features:literal, $proof:path ; [$($attr:tt)*] $vis:vis $name:ident [$($gen:tt)*] [$($tf:tt)*] ($first:ident : $fty:ty $(, $arg:ident : $ty:ty)*) [$($ret:tt)*] $body:block) => {
		$($attr)*
		$vis fn $name <$($gen)*> ($first : $fty $(, $arg : $ty)*) $($ret)* {
			#[target_feature(enable = $features)]
			#[inline]
			unsafe fn __impl <$($gen)*> ($first : $fty $(, $arg : $ty)*) $($ret)* {
				$body
			}
			// Compile error unless the first parameter's type is a token
			// (sealed `Proves`) that covers every feature in `$features`.
			$crate::x86::fn_macros::__private::require_proof::<$proof, _>(&$first);
			// SAFETY: `$first` is a token (checked above), and a token only
			// comes from detection, so every feature in `$features` is live.
			#[allow(unused_unsafe)]
			unsafe { __impl::<$($tf)*>($first $(, $arg)*) }
		}
	};
}

// SSE family

/// Wrap a body under `ssse3` (byte shuffle / `palignr` kernels).
///
/// Hold an [`crate::x86::ops::sse::ssse3::Ssse3`] token.
#[macro_export]
macro_rules! ssse3_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "ssse3", $crate::x86::fn_macros::__private::m::Ssse3Fn ; $($tt)* } };
}

/// Wrap a body under `sse2`, `ssse3`, and `sse4.1`.
///
/// Hold `Sse2`, `Ssse3`, and `Sse41`. Shuffle/blend networks without
/// half-float convert; for F16C add-on use [`sse41_f16c_fn!`].
#[macro_export]
macro_rules! sse41_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse2,ssse3,sse4.1", $crate::x86::fn_macros::__private::m::Sse41Fn ; $($tt)* } };
}

/// Wrap a body under `sse2`, `ssse3`, `sse4.1`, and `f16c`.
///
/// Hold `Sse2`, `Ssse3`, `Sse41`, and `F16c`. Useful for SSE shuffle
/// networks that finish with a half-float convert.
#[macro_export]
macro_rules! sse41_f16c_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse2,ssse3,sse4.1,f16c", $crate::x86::fn_macros::__private::m::Sse41F16cFn ; $($tt)* } };
}

/// Wrap a body under `sse4.2` (CRC32 / 64-bit compares).
///
/// Hold an [`crate::x86::ops::sse::sse42::Sse42`] token.
#[macro_export]
macro_rules! sse42_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sse4.2", $crate::x86::fn_macros::__private::m::Sse42Fn ; $($tt)* } };
}

// AVX / FMA / VNNI (V3-class)

/// Wrap a multi-op body under `avx`.
///
/// Hold an [`crate::x86::ops::avx::avx::Avx`] token. Prefer this for multi-op
/// AVX kernels; single slice ops can use [`crate::x86::auto_up`].
#[macro_export]
macro_rules! avx_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx", $crate::x86::fn_macros::__private::m::AvxFn ; $($tt)* } };
}

/// Wrap a body under `f16c` (half-float convert). Hold an `F16c` token.
#[macro_export]
macro_rules! f16c_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "f16c", $crate::x86::fn_macros::__private::m::F16cFn ; $($tt)* } };
}

/// Wrap a body under `fma` (FMA3 fused multiply-add).
///
/// Hold an [`crate::x86::ops::avx::fma::Fma`] token. For mixed AVX math
/// + FMA prefer [`avx_fma_fn!`].
#[macro_export]
macro_rules! fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "fma", $crate::x86::fn_macros::__private::m::FmaFn ; $($tt)* } };
}

/// Wrap a body under `avx` and `fma` together.
///
/// Hold `Avx` and `Fma`. Typical float kernels that mix `add`/`mul` with
/// fused `fmadd`.
#[macro_export]
macro_rules! avx_fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx,fma", $crate::x86::fn_macros::__private::m::AvxFmaFn ; $($tt)* } };
}

/// Wrap a body under `avx2`, `sse2`, and `ssse3`.
///
/// Hold `Avx2`, `Sse2`, and `Ssse3`. Common for AVX2 kernels that still
/// use 128-bit extract/shuffle helpers.
#[macro_export]
macro_rules! avx2_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx2,sse2,ssse3", $crate::x86::fn_macros::__private::m::Avx2Fn ; $($tt)* } };
}

/// Wrap a body under `avx2`, `fma`, `sse2`, and `ssse3`.
///
/// First parameter `(Avx2, Fma)` or a bundle. V3-class int+float kernels
/// that also touch narrow SSE helpers.
#[macro_export]
macro_rules! avx2_fma_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx2,fma,sse2,ssse3", $crate::x86::fn_macros::__private::m::Avx2FmaFn ; $($tt)* } };
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
			"sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,f16c", $crate::x86::fn_macros::__private::m::AvxV3Fn ; $($tt)*
		}
	};
}

/// Wrap a body under `avxvnni` (AVX2-width VNNI, no AVX-512 needed).
///
/// Hold an [`crate::x86::ops::avx::avx_vnni::AvxVnni`] token.
#[macro_export]
macro_rules! avx_vnni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avxvnni", $crate::x86::fn_macros::__private::m::AvxVnniFn ; $($tt)* } };
}

// AVX-512

/// Wrap a body under `avx512f` only.
///
/// Hold an `Avx512f` token. Does **not** enable DQ/BW/VL: use
/// [`avx512_fn!`], [`avx512bw_fn!`], or [`avx512vl_fn!`] if the body also
/// needs those.
#[macro_export]
macro_rules! avx512f_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f", $crate::x86::fn_macros::__private::m::Avx512fFn ; $($tt)* } };
}

/// Wrap a body under `avx512f`, `avx`, and `avx512dq` together.
///
/// Hold `Avx512f`, `Avx`, and `Avx512Dq`. Typical for wide float kernels
/// that mix ZMM math with 256-bit extract/insert.
#[macro_export]
macro_rules! avx512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f,avx,avx512dq", $crate::x86::fn_macros::__private::m::Avx512Fn ; $($tt)* } };
}

/// Wrap a body under `avx512f`, `avx512bw`, `avx2`, `sse2`, and `ssse3`.
///
/// Hold all five tokens. For wide byte-lane kernels with narrower remainders.
#[macro_export]
macro_rules! avx512bw_fn {
	($($tt:tt)*) => {
		$crate::__miraculix_tf_fn! { "avx512f,avx512bw,avx2,sse2,ssse3", $crate::x86::fn_macros::__private::m::Avx512bwFn ; $($tt)* }
	};
}

/// Wrap a body under `avx512f` and `avx512vl` (128/256-bit EVEX forms).
///
/// Hold `Avx512f` plus the VL token(s) you call (`Avx512FVl`,
/// `Avx512BwVl`, etc).
#[macro_export]
macro_rules! avx512vl_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f,avx512vl", $crate::x86::fn_macros::__private::m::Avx512vlFn ; $($tt)* } };
}

/// Wrap a body under `avx512vnni` (512-bit VNNI).
///
/// Hold an [`crate::x86::ops::avx512::avx512vnni::Avx512Vnni`] token.
#[macro_export]
macro_rules! avx512vnni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512vnni", $crate::x86::fn_macros::__private::m::Avx512vnniFn ; $($tt)* } };
}

/// Wrap a body under `avx512f,avx512bw,avx512vbmi` (byte permutes such as
/// `vpermi2b`, together with the 512-bit loads and masks they need).
///
/// Hold [`crate::x86::ops::avx512::avx512vbmi::Avx512Vbmi`] plus the
/// `Avx512f` and `Avx512Bw` tokens.
#[macro_export]
macro_rules! avx512vbmi_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512f,avx512bw,avx512vbmi", $crate::x86::fn_macros::__private::m::Avx512vbmiFn ; $($tt)* } };
}

/// Wrap a body under `avx512vbmi2` (compress/expand, variable shifts).
///
/// Hold an [`crate::x86::ops::avx512::avx512vbmi2::Avx512Vbmi2`] token.
#[macro_export]
macro_rules! avx512vbmi2_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512vbmi2", $crate::x86::fn_macros::__private::m::Avx512vbmi2Fn ; $($tt)* } };
}

/// Wrap a body under `avx512fp16` (native half-precision arithmetic).
///
/// Hold an [`crate::x86::ops::avx512::avx512fp16::Avx512Fp16`] token.
/// VL half-width forms also need `avx512vl` on the callee; if you compose
/// both and hit a gap, open a dedicated `avx512fp16_vl_fn!`.
#[macro_export]
macro_rules! avx512fp16_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512fp16", $crate::x86::fn_macros::__private::m::Avx512fp16Fn ; $($tt)* } };
}

/// Wrap a body under `avx512bf16` and `avx512f`.
///
/// Hold an [`crate::x86::ops::avx512::avx512bf16::Avx512Bf16`] token
/// (methods already require both features).
#[macro_export]
macro_rules! avx512bf16_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "avx512bf16,avx512f", $crate::x86::fn_macros::__private::m::Avx512bf16Fn ; $($tt)* } };
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
			"sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl", $crate::x86::fn_macros::__private::m::Avx512V4Fn ; $($tt)*
		}
	};
}

/// Wrap a body under the x86-64-v4 core plus `avx512vbmi`.
///
/// Enables
/// `sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi`.
/// Hold a [`crate::x86::bundles::V4Vbmi`], or tokens for every extension the body actually calls. Prefer a
/// narrower macro when possible.
#[macro_export]
macro_rules! avx512_v4_vbmi_fn {
	($($tt:tt)*) => {
		$crate::__miraculix_tf_fn! {
			"sse,sse2,fxsr,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,bmi1,bmi2,fma,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi", $crate::x86::fn_macros::__private::m::Avx512V4VbmiFn ; $($tt)*
		}
	};
}

// Crypto / specialty

/// Wrap a body under `aes` (AES-NI round chain).
///
/// Hold an [`crate::x86::ops::other::aes::Aes`] token.
#[macro_export]
macro_rules! aes_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "aes", $crate::x86::fn_macros::__private::m::AesFn ; $($tt)* } };
}

/// Wrap a body under `pclmulqdq` (carry-less multiply).
///
/// Hold a [`crate::x86::ops::other::pclmulqdq::Pclmulqdq`] token.
#[macro_export]
macro_rules! pclmulqdq_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "pclmulqdq", $crate::x86::fn_macros::__private::m::PclmulqdqFn ; $($tt)* } };
}

/// Wrap a body under `aes` and `pclmulqdq` (AES-GCM style kernels).
///
/// First parameter `(Aes, Pclmulqdq)`.
#[macro_export]
macro_rules! aes_pclmul_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "aes,pclmulqdq", $crate::x86::fn_macros::__private::m::AesPclmulFn ; $($tt)* } };
}

/// Wrap a body under `sha` (SHA-NI multi-round).
///
/// Hold a [`crate::x86::ops::other::sha::Sha`] token.
#[macro_export]
macro_rules! sha_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "sha", $crate::x86::fn_macros::__private::m::ShaFn ; $($tt)* } };
}

/// Wrap a body under `gfni` (128-bit GFNI forms).
///
/// Hold a [`crate::x86::ops::other::gfni::Gfni`] token. For 256-bit GFNI
/// methods that also need AVX, use [`gfni_avx_fn!`].
#[macro_export]
macro_rules! gfni_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni", $crate::x86::fn_macros::__private::m::GfniFn ; $($tt)* } };
}

/// Wrap a body under `gfni` and `avx` (256-bit GFNI).
///
/// First parameter `(Gfni, Avx)`: GFNI ships without AVX on some cores
/// (Tremont, Jasper Lake), so a `Gfni` token alone does not prove AVX. The
/// body may call 256-bit `gf2p8mul` / affine forms.
#[macro_export]
macro_rules! gfni_avx_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni,avx", $crate::x86::fn_macros::__private::m::GfniAvxFn ; $($tt)* } };
}

/// Wrap a body under `gfni` and `avx512f` (512-bit GFNI).
///
/// Hold a [`crate::x86::ops::other::gfni::Gfni512`] token.
#[macro_export]
macro_rules! gfni512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "gfni,avx512f", $crate::x86::fn_macros::__private::m::Gfni512Fn ; $($tt)* } };
}

/// Wrap a body under `vaes` (256-bit vector AES).
///
/// Hold a [`crate::x86::ops::other::vaes::Vaes`] token.
#[macro_export]
macro_rules! vaes_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "vaes", $crate::x86::fn_macros::__private::m::VaesFn ; $($tt)* } };
}

/// Wrap a body under `vaes` and `avx512f` (512-bit vector AES).
///
/// Hold a [`crate::x86::ops::other::vaes::Vaes512`] token.
#[macro_export]
macro_rules! vaes512_fn {
	($($tt:tt)*) => { $crate::__miraculix_tf_fn! { "vaes,avx512f", $crate::x86::fn_macros::__private::m::Vaes512Fn ; $($tt)* } };
}

#[cfg(test)]
#[path = "test/fn_macros.rs"]
mod tests;
