//! Coarse tiers over [`super::Feature`]. On `riscv*gc-*-linux-*`, M/A/F/D/C
//! are ABI floor (like NEON on AArch64). Only [`Feature::V`] is HW-dependent.
//! No RVA22/RVA23 table: Zba/Zbb/... not in Feature.
//!
//! Room to grow: [`RiscvLevel`] is `#[non_exhaustive]`. Discriminant `0` is
//! kept free for a level below Gc (bare-metal RV32 cores without F/D/C, e.g.
//! the Tensix "baby" cores on Tenstorrent Blackhole). Levels from
//! `riscv_hwprobe(2)` (Zba/Zbb, Zvfh, RVA23) go above `Vector`.

use super::features::{Feature, FeatureSet};

/// Enum list for RISC-V levels. Gc always (gc targets); no empty state.
///
/// Order is capability order. Discriminants are stable, `0` is reserved
/// (see module doc). Match with a `_` arm: more levels will come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
#[repr(u8)]
pub enum RiscvLevel {
    /// gc floor: M A F D C.
    Gc = 1,
    /// + V.
    Vector = 2,
}

impl RiscvLevel {
    pub const ALL: &'static [RiscvLevel] = &[RiscvLevel::Gc, RiscvLevel::Vector];

    /// Cumulative required features. Gc = ABI floor.
    pub fn required_features(self) -> &'static [Feature] {
        const GC: &[Feature] = &[Feature::M, Feature::A, Feature::F, Feature::D, Feature::C];
        const VECTOR: &[Feature] = &[
            Feature::M,
            Feature::A,
            Feature::F,
            Feature::D,
            Feature::C,
            Feature::V,
        ];

        match self {
            RiscvLevel::Gc => GC,
            RiscvLevel::Vector => VECTOR,
        }
    }

    /// Highest level fully covered by `set`. Falls back to Gc today, also when
    /// `set` lacks part of the gc floor.
    pub fn detect(set: FeatureSet) -> Self {
        RiscvLevel::ALL
            .iter()
            .rev()
            .copied()
            .find(|&level| set.contains_all(level.required_features()))
            .unwrap_or(RiscvLevel::Gc)
    }
}
