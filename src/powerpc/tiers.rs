//! Coarse vector tiers over [`super::Feature`], named after the open Power ISA
//! versions (OpenPOWER Foundation) rather than IBM chips, so open cores and
//! consoles fit the same ladder.
//!
//! | Level | ISA | Typical cores |
//! |---|---|---|
//! | `Scalar` | SFS / SFFS | Microwatt, A2O, 750 (GameCube/Wii/Wii U use paired singles) |
//! | `Altivec` | VMX | G4, G5, Cell PPE (PS3), Xenon (Xbox 360) |
//! | `Vsx` | 2.06 | POWER7 |
//! | `Isa207` | 2.07 | POWER8 |
//! | `Isa300` | 3.0 | POWER9 |
//! | `Isa31` | 3.1 | POWER10, POWER11 |
//!
//! Paired singles, Cell and VMX128 are features, not levels: they are not
//! cumulative (a Gekko has paired singles but no AltiVec).

use super::features::{Feature, FeatureSet};

/// Power vector levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum PowerLevel {
    /// No AltiVec/VSX proven.
    Scalar = 0,
    /// AltiVec / VMX.
    Altivec = 1,
    /// + VSX (Power ISA 2.06).
    Vsx = 2,
    /// + Power ISA 2.07 vector (POWER8).
    Isa207 = 3,
    /// + Power ISA 3.0 vector (POWER9).
    Isa300 = 4,
    /// + Power ISA 3.1 (POWER10, POWER11).
    Isa31 = 5,
}

#[allow(non_upper_case_globals, reason = "aliases of the former variant names")]
impl PowerLevel {
    /// Former name of [`PowerLevel::Isa207`].
    #[deprecated(note = "renamed to `PowerLevel::Isa207` (Power ISA 2.07)")]
    pub const Power8: PowerLevel = PowerLevel::Isa207;
    /// Former name of [`PowerLevel::Isa300`].
    #[deprecated(note = "renamed to `PowerLevel::Isa300` (Power ISA 3.0)")]
    pub const Power9: PowerLevel = PowerLevel::Isa300;
}

impl PowerLevel {
    pub const ALL: &'static [PowerLevel] = &[
        PowerLevel::Scalar,
        PowerLevel::Altivec,
        PowerLevel::Vsx,
        PowerLevel::Isa207,
        PowerLevel::Isa300,
        PowerLevel::Isa31,
    ];

    /// Cumulative required features. Scalar = empty.
    pub fn required_features(self) -> &'static [Feature] {
        const ALTIVEC: &[Feature] = &[Feature::Altivec];
        const VSX: &[Feature] = &[Feature::Altivec, Feature::Vsx];
        const ISA207: &[Feature] = &[
            Feature::Altivec,
            Feature::Vsx,
            Feature::Power8,
            Feature::Power8Altivec,
            Feature::Power8Vector,
        ];
        const ISA300: &[Feature] = &[
            Feature::Altivec,
            Feature::Vsx,
            Feature::Power8,
            Feature::Power8Altivec,
            Feature::Power8Vector,
            Feature::Power9,
            Feature::Power9Altivec,
            Feature::Power9Vector,
        ];
        const ISA31: &[Feature] = &[
            Feature::Altivec,
            Feature::Vsx,
            Feature::Power8,
            Feature::Power8Altivec,
            Feature::Power8Vector,
            Feature::Power9,
            Feature::Power9Altivec,
            Feature::Power9Vector,
            Feature::Isa31,
        ];

        match self {
            PowerLevel::Scalar => &[],
            PowerLevel::Altivec => ALTIVEC,
            PowerLevel::Vsx => VSX,
            PowerLevel::Isa207 => ISA207,
            PowerLevel::Isa300 => ISA300,
            PowerLevel::Isa31 => ISA31,
        }
    }

    /// Highest level fully covered by `set`.
    pub fn detect(set: FeatureSet) -> Self {
        PowerLevel::ALL
            .iter()
            .rev()
            .copied()
            .find(|&level| set.contains_all(level.required_features()))
            .unwrap_or(PowerLevel::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn old_names_still_resolve() {
        assert_eq!(PowerLevel::Power8, PowerLevel::Isa207);
        assert_eq!(PowerLevel::Power9 as u8, 4);
        // Usable as a pattern, as the former variants were.
        assert!(matches!(PowerLevel::Isa300, PowerLevel::Power9));
    }

    #[test]
    fn discriminants_are_stable() {
        // The level cache stores `as u8`; old values must not move.
        let want = [0u8, 1, 2, 3, 4, 5];
        for (&level, &v) in PowerLevel::ALL.iter().zip(&want) {
            assert_eq!(level as u8, v, "{level:?}");
        }
    }
}
