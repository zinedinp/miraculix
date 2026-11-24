//! PowerPC SIMD tokens and slice kernels, one file per extension.
//!
//! | Module | Token | Cores | Needs |
//! |---|---|---|---|
//! | `vmx` | `Vmx` | G4, G5, Cell PPE (PS3), Xenon (Xbox 360) | AltiVec; big-endian targets |
//! | `paired` | `PairedSingle` | Gekko, Broadway, Espresso (GameCube, Wii, Wii U) | 32-bit `powerpc`, bare metal |
//! | `vmx128` | `Vmx128` | Xenon (Xbox 360) | big-endian `powerpc64`, bare metal |
//! | `altivec` | `Altivec` | any AltiVec core | `nightly-altivec` (intrinsics), `powerpc64` |
//!
//! Everything except `altivec` is stable `asm!` (stable on PowerPC since Rust
//! 1.95). Paired singles and VMX128 are unknown to every assembler, so they are
//! emitted as `.long` words from `encode`.

/// `asm!` with a mix of assembler lines and encoded words, in program order:
/// `s("mnemonic ...")` is a template line, `w(expr)` a `.long` of a `const`
/// word. After `;` come the usual named operands and options.
///
/// The words become positional `const` operands, which `asm!` wants before the
/// named ones, so the macro collects both lists in order and then emits them.
#[allow(unused_macros)]
macro_rules! asm_mixed {
    (@go [$($t:expr),*] [$($c:expr),*] w($e:expr), $($rest:tt)*) => {
        asm_mixed!(@go [$($t,)* ".long {}"] [$($c,)* $e] $($rest)*)
    };
    (@go [$($t:expr),*] [$($c:expr),*] s($l:expr), $($rest:tt)*) => {
        asm_mixed!(@go [$($t,)* $l] [$($c),*] $($rest)*)
    };
    (@go [$($t:expr),*] [$($c:expr),*] ; $($ops:tt)*) => {
        core::arch::asm!($($t,)* $(const $c,)* $($ops)*)
    };
    ($($body:tt)*) => {
        asm_mixed!(@go [] [] $($body)*)
    };
}

#[cfg(any(test, target_arch = "powerpc", target_arch = "powerpc64"))]
pub(crate) mod encode;

#[cfg(target_endian = "big")]
pub mod vmx;

#[cfg(target_arch = "powerpc")]
pub mod paired;

#[cfg(all(target_arch = "powerpc64", target_endian = "big"))]
pub mod vmx128;

#[cfg(all(feature = "nightly-altivec", target_arch = "powerpc64"))]
pub mod altivec;
