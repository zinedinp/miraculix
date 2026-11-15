//! # WebAssembly SIMD (simd128, relaxed-simd)
//!
//! Detection for `wasm32` and `wasm64`, plus on `wasm32` the same three call
//! levels as the other arches:
//!
//! | Level | Where | Example |
//! |---|---|---|
//! | Auto slices | `wasm::auto_up` (alias `wasm::auto`) | `auto::add_f32(&a, &b, &mut out)` |
//! | Tokens, `_slice` | `wasm::ops::simd128::Simd128` | `t.add_f32_slice(&a, &b, &mut out)` |
//! | Tokens, fixed width | same token | `t.add_f32x4([..; 4], [..; 4])` |
//! | Detection | this module | [`detect_level`], [`detect_features`] |
//!
//! `ops` and `auto_up` need the default cargo feature `wasm-simd` (or
//! `-C target-feature=+simd128`); the relaxed-simd token
//! (`ops::relaxed::RelaxedSimd`) needs `wasm-relaxed-simd` (or
//! `+relaxed-simd`). See [Cargo features](#cargo-features).
//!
//! ## Quick start
//!
//! wasm has no CPU probe. As in pulp, the host checks what the engine
//! supports and then switches the SIMD path on, once, at startup:
//!
//! ```
//! # #[cfg(all(target_arch = "wasm32", feature = "wasm-simd"))] {
//! use miraculix::wasm;
//!
//! wasm::enable_simd128(); // the engine runs simd128 (checked by the host)
//!
//! let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let b = [0.5f32; 5];
//! let mut out = [0f32; 5];
//! wasm::auto::add_f32(&a, &b, &mut out); // Simd128 rung; scalar if never enabled
//! assert_eq!(out, [1.5, 2.5, 3.5, 4.5, 5.5]);
//! # }
//! ```
//!
//! Without the call (and without `+simd128`), every `auto` fn takes the
//! scalar rung and every token's `detect()` returns `None`.
//!
//! ### Browser (`wasm32-unknown-unknown`)
//!
//! Every current browser engine runs simd128, so the usual setup is: keep
//! the default features and call [`enable_simd128`] from your start-up
//! export. To also serve an engine without simd128, build a second module
//! with `default-features = false` and let JS pick, for example with
//! `wasm-feature-detect`:
//!
//! ```js
//! import { simd } from "wasm-feature-detect";
//! const url = (await simd()) ? "app_simd.wasm" : "app_plain.wasm";
//! ```
//!
//! One module cannot do both: see
//! [What ends up in the module](#what-ends-up-in-the-module).
//!
//! relaxed-simd is not in every engine. Turn on `wasm-relaxed-simd` only
//! for a module that is served to engines known to have it, and call
//! [`enable_relaxed_simd`] there.
//!
//! ### wasmtime (plugins, `wasm32-wasip1` / `wasip2` / `wasip3`)
//!
//! wasmtime enables simd128 and relaxed-simd by default (`Config::wasm_simd`,
//! `Config::wasm_relaxed_simd`). When the host owns the engine config, the
//! simplest choice is compile time: build the guest with
//! `RUSTFLAGS="-C target-feature=+simd128"` (plus `+relaxed-simd` if wanted).
//! Then no flag call is needed, and every function in the guest can use
//! simd128 directly. The runtime flags work as well, e.g. called from the
//! guest's init export.
//!
//! ## How the rung is chosen
//!
//! [`detect_features`] is this binary's `cfg(target_feature)` **or** the
//! runtime flags. It is two atomic loads, not cached, so a later
//! [`enable_simd128`] takes effect at once. [`detect_level`] reads the same
//! set ([`WasmLevel`]: `Scalar < Simd128 < RelaxedSimd`).
//!
//! Each `auto` fn tries, in order:
//!
//! 1. `RelaxedSimd`: FMA family only, and only when compiled in.
//! 2. `Simd128`: every `auto` op.
//! 3. scalar, with the same lane semantics.
//!
//! | Call | Effect |
//! |---|---|
//! | [`enable_simd128`] | Simd128 rung from now on |
//! | [`enable_relaxed_simd`] | also the RelaxedSimd rung (implies simd128) |
//! | [`disable_simd128`] / [`disable_relaxed_simd`] | back down; no effect on what `+simd128` / `+relaxed-simd` compiled in |
//! | [`is_simd128_enabled`] / [`is_relaxed_simd_enabled`] | `cfg` or flag |
//!
//! The flags are process-global (`Relaxed` atomics). [`shortpath::resolve`]
//! is the compile-time floor; the flags can only raise the level above it.
//!
//! ## Cargo features
//!
//! | Feature | Default | Compiles |
//! |---|---|---|
//! | `wasm-simd` | on | `ops::simd128`, `auto_up` / `auto` |
//! | `wasm-relaxed-simd` | off | also `ops::relaxed` and the top `fmadd` rung (implies `wasm-simd`) |
//!
//! `-C target-feature=+simd128` brings `ops` and `auto` back even with
//! `wasm-simd` off (the module needs simd128 then anyway), and
//! `+relaxed-simd` does the same for the relaxed code. Detection and the
//! flags exist in every build. No effect on other arches.
//!
//! ## What ends up in the module
//!
//! **Validation is per module, not per function.** The engine checks every
//! opcode when the module is compiled or instantiated. An engine without
//! simd128 rejects a module with any simd128 instruction in it, even one
//! that never runs. The runtime flags therefore choose which code runs once
//! the module has loaded; they cannot make it load.
//!
//! **LLVM has no per-function target features on wasm.** Elsewhere,
//! `#[target_feature(enable = "...")]` affects only the function it is on.
//! LLVM's wasm backend instead merges the features of all functions in a
//! codegen unit. With `wasm-simd`, this crate's SIMD bodies enable simd128,
//! so the whole crate is compiled with simd128:
//!
//! - Code that never touches SIMD (`detect_*`, the scalar `auto` rung) may
//!   still be auto-vectorized and contain simd128 opcodes, and the object's
//!   `target_features` section declares simd128.
//! - "Flags off" therefore selects the scalar *algorithm*, not
//!   simd128-free machine code. Results are the same; only the module's
//!   engine requirement differs.
//! - A side benefit: rustc can inline the `#[target_feature]` bodies more
//!   freely than on other arches.
//!
//! This is why `wasm-simd` exists. With `default-features = false` nothing
//! in the crate enables simd128: the crate has no vector opcode, its object
//! does not declare simd128, and the module loads on an engine without
//! simd128. `scripts/check-asm-wasm.sh` checks the first two;
//! `scripts/wasm-test.sh` runs the test suite on wasmtime with
//! `-W simd=n` for the third. The default build is rejected by that engine.
//!
//! Choosing a build:
//!
//! | Module must load on | Build |
//! |---|---|
//! | engines with simd128 (all current browsers, wasmtime) | default features |
//! | also engines without simd128 | `default-features = false` for that module |
//! | engines with relaxed-simd | `features = ["wasm-relaxed-simd"]` |
//!
//! ## Semantics
//!
//! Lane rules (wrapping vs. saturating, mask types, per-bit select,
//! `a & !b` for `andnot`, x86-style constant shifts, float min/max with
//! NaN winning and `-0.0 < +0.0`, unfused `fmadd` on `Simd128`, lane-grouped
//! float sums) are listed in `auto_up`'s module doc. Every rung follows
//! them, so results match across rungs; NaN results match as NaN, since
//! wasm leaves the payload open.
//!
//! ## Testing
//!
//! ```sh
//! rustup target add wasm32-wasip1 wasm32-wasip2 wasm32-wasip3
//! cargo install wasmtime-cli --locked
//! scripts/wasm-test.sh            # node + wasmtime, wasip1/wasip2/wasip3, all feature/flag builds
//! scripts/wasm-test.sh --release
//! scripts/check-asm-wasm.sh       # release codegen: SIMD loops, and none without wasm-simd
//! ```
//!
//! node runs wasip1 (V8, the browser's engine); wasmtime runs every wasip
//! target. One test binary walks all rungs by switching the flags. wasm
//! builds use `panic = "abort"`: the first failing assert ends the run, so
//! re-run with `-- --nocapture` to see which test.
//!
//! ## Limits
//!
//! - `wasm64`: detection only (`core::arch::wasm64` is unstable).
//! - No FP16: stdarch has no wasm f16 intrinsics.
//! - Token-only, no `auto` fn: `sqrt` and the rounding ops (no `core`
//!   scalar fallback under `no_std`), `swizzle`, dot products, `q15mulr`,
//!   and the relaxed min/max/laneselect/swizzle/trunc ops.
//! - Ops wasm has no instruction for are left out rather than emulated:
//!   8-bit `mul`, 64-bit `min`/`max`, ordered `u64` compares,
//!   `hadd`/`hsub`, `popcnt` wider than `u8`.

pub mod features;
pub mod shortpath;
pub mod tiers;

/// Token-gated simd128 / relaxed-simd ops. `wasm32` with the `wasm-simd`
/// feature (default) or `+simd128`.
#[cfg(all(
    target_arch = "wasm32",
    any(feature = "wasm-simd", target_feature = "simd128")
))]
pub mod ops;

/// Elementwise slice work without picking a token by hand (see [`auto_up`]).
#[cfg(all(
    target_arch = "wasm32",
    any(feature = "wasm-simd", target_feature = "simd128")
))]
mod auto_down;
#[cfg(all(
    target_arch = "wasm32",
    any(feature = "wasm-simd", target_feature = "simd128")
))]
pub mod auto_up;
/// Short alias for [`auto_up`] (same module). Prefer `auto_up` in new code.
#[cfg(all(
    target_arch = "wasm32",
    any(feature = "wasm-simd", target_feature = "simd128")
))]
pub use auto_up as auto;

pub use features::{
    Feature, FeatureSet, disable_relaxed_simd, disable_simd128, enable_relaxed_simd,
    enable_simd128, is_relaxed_simd_enabled, is_simd128_enabled,
};
pub use tiers::WasmLevel;

/// Feature bits for this binary right now: `cfg` plus the runtime flags. Not
/// cached (two atomic loads), so a later [`enable_simd128`] takes effect at
/// once. Same name as on the other arches; used by `auto_up`.
pub fn detect_features() -> FeatureSet {
    FeatureSet::detect()
}

/// Same as [`detect_features`] (there is nothing to re-probe on wasm).
pub fn detect_features_fresh() -> FeatureSet {
    FeatureSet::detect()
}

/// WASM SIMD tier right now (`cfg` plus runtime flags).
///
/// Not cached, unlike the other arches: a cache would freeze the level from
/// before a later [`enable_simd128`]. Never below [`shortpath::resolve`].
pub fn detect_level() -> WasmLevel {
    detect_level_fresh()
}

/// Same as [`detect_level`].
pub fn detect_level_fresh() -> WasmLevel {
    WasmLevel::detect(FeatureSet::detect())
}

/// No-op, kept for API parity with the other arches (nothing to warm up).
pub fn warm_up() {}

#[cfg(all(
    test,
    target_arch = "wasm32",
    any(feature = "wasm-simd", target_feature = "simd128")
))]
#[path = "test/mod.rs"]
mod wasm_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct bit indices for the Feature bitset.
    #[test]
    fn every_feature_has_a_unique_bit() {
        let mut seen = std::collections::HashSet::new();
        for &feature in Feature::ALL {
            assert!(seen.insert(feature.bit()), "duplicate bit for {feature:?}");
        }
    }

    /// `detect` matches manual required-features on the same set.
    #[test]
    fn wasm_level_matches_manual_check() {
        let set = FeatureSet::detect();
        let detected = WasmLevel::detect(set);

        for &level in WasmLevel::ALL {
            let should_hold = set.contains_all(level.required_features());
            assert_eq!(
                level <= detected,
                should_hold || level == WasmLevel::Scalar,
                "level {level:?} required-features check disagrees with detect() result"
            );
        }
    }

    /// Higher tiers supersets of lower (copy-paste guard).
    #[test]
    fn wasm_levels_are_cumulative() {
        for pair in WasmLevel::ALL.windows(2) {
            let (lower, higher) = (pair[0].required_features(), pair[1].required_features());
            assert!(
                lower.iter().all(|f| higher.contains(f)),
                "{:?} must be a superset of {:?}",
                pair[1],
                pair[0]
            );
        }
    }

    /// The compile-time level is the floor; runtime flags can only raise it.
    #[test]
    fn detect_level_fresh_is_at_least_shortpath_resolve() {
        assert!(detect_level_fresh() >= shortpath::resolve());
    }

    /// `enable_*` raises the level, `disable_*` lowers it back to the
    /// compile-time floor (and never below).
    #[test]
    fn runtime_flags_move_the_level() {
        enable_relaxed_simd();
        assert_eq!(detect_level(), WasmLevel::RelaxedSimd);
        assert!(is_simd128_enabled(), "relaxed-simd implies simd128");
        disable_relaxed_simd();
        assert!(detect_level() >= WasmLevel::Simd128);
        disable_simd128();
        assert_eq!(detect_level(), shortpath::resolve());
        enable_simd128();
        assert!(detect_level() >= WasmLevel::Simd128);
    }

    #[test]
    fn detect_level_matches_fresh() {
        assert_eq!(detect_level(), detect_level_fresh());
    }

    #[test]
    fn detect_level_is_stable_across_calls() {
        let a = detect_level();
        let b = detect_level();
        let c = detect_level();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn warm_up_then_detect_agrees_with_fresh() {
        warm_up();
        assert_eq!(detect_level(), detect_level_fresh());
    }
}
