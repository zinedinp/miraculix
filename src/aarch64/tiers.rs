//! Coarse arch tiers over [`super::Feature`] (~ GCC `-march=armv8.x-a`/`v9a`).
//! Approx (optionals != clean versions); truth is FeatureSet. macOS caps without
//! Uscat: use [`super::AppleLevel`]. Windows lacks RDM `PF_ARM_*`: use
//! [`super::SnapdragonLevel`].

use super::features::{Feature, FeatureSet};

/// Enum list for ARMv8/v9. NEON always => >= V8_0; no empty state.
///
/// Linear and cumulative: every level requires everything below it. Which
/// features each `v8.x`/`v9.x` step adds was read from LLVM (nightly
/// `rustc --print cfg -C target-feature=+v9.5a` etc.), not from memory:
/// `v9.n` implies exactly `v8.(n+5)`, and past v8.9 nothing new is visible
/// to the kernel's HWCAP words. See [`ArmLevel::is_runtime_detectable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ArmLevel {
    /// ARMv8.0-A: NEON; opt AES/PMULL/SHA/CRC32.
    V8_0 = 0,
    /// ARMv8.1-A: +LSE, RDM, mand. CRC32.
    V8_1 = 1,
    /// ARMv8.2-A: +DC clean to PoP.
    V8_2 = 2,
    /// ARMv8.3-A: +JSCVT, FCMA, RCPC, PAC.
    V8_3 = 3,
    /// ARMv8.4-A: +DIT, Uscat, RCPC2, FLAGM, mand. DotProd.
    V8_4 = 4,
    /// ARMv8.5-A: +SSBS, SB, BTI, FRINT, FLAGM2, DCPODP (FEAT_RNG is optional, not required).
    V8_5 = 5,
    /// ARMv8.6-A: +I8MM, BF16, ECV.
    V8_6 = 6,
    /// ARMv8.7-A: +WFET/WFIT, AFP, RPRES.
    V8_7 = 7,
    /// ARMv8.8-A: +MOPS (memcpy/memset instrs), HBC (hinted cond. branch).
    V8_8 = 8,
    /// ARMv8.9-A: +CSSC (common short sequence compression: ABS/CNT/CTZ/SMAX...).
    V8_9 = 9,
    /// ARMv9.0-A: all of v8.9 above, plus SVE + SVE2. Stricter than the
    /// architecture (v9.0 ≈ v8.5 + SVE2) so the ladder stays linear.
    V9_0 = 10,
    /// ARMv9.1-A (≈ v8.6). **Compile-time only**: no HWCAP-visible
    /// mandatory addition over [`ArmLevel::V9_0`], so [`ArmLevel::detect`] never
    /// returns it; reached via the `v9.1a` target-feature cfg.
    V9_1 = 11,
    /// ARMv9.2-A (≈ v8.7). Compile-time only, like [`ArmLevel::V9_1`].
    V9_2 = 12,
    /// ARMv9.3-A (≈ v8.8). Compile-time only, like [`ArmLevel::V9_1`].
    V9_3 = 13,
    /// ARMv9.4-A (≈ v8.9). Compile-time only, like [`ArmLevel::V9_1`].
    V9_4 = 14,
    /// ARMv9.5-A. Compile-time only, like [`ArmLevel::V9_1`].
    V9_5 = 15,
}

impl ArmLevel {
    pub const ALL: &'static [ArmLevel] = &[
        ArmLevel::V8_0,
        ArmLevel::V8_1,
        ArmLevel::V8_2,
        ArmLevel::V8_3,
        ArmLevel::V8_4,
        ArmLevel::V8_5,
        ArmLevel::V8_6,
        ArmLevel::V8_7,
        ArmLevel::V8_8,
        ArmLevel::V8_9,
        ArmLevel::V9_0,
        ArmLevel::V9_1,
        ArmLevel::V9_2,
        ArmLevel::V9_3,
        ArmLevel::V9_4,
        ArmLevel::V9_5,
    ];

    /// Cumulative required features. V8_0 = empty (NEON always). V9_1..V9_5
    /// equal V9_0 (see [`ArmLevel::is_runtime_detectable`]).
    pub fn required_features(self) -> &'static [Feature] {
        const V8_1: &[Feature] = &[Feature::Lse, Feature::Rdm, Feature::Crc32];
        const V8_2: &[Feature] = &[Feature::Lse, Feature::Rdm, Feature::Crc32, Feature::Dcpop];
        const V8_3: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
        ];
        const V8_4: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
        ];
        const V8_5: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
        ];
        const V8_6: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
            Feature::I8mm,
            Feature::Bf16,
            Feature::Ecv,
        ];
        const V8_7: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
            Feature::I8mm,
            Feature::Bf16,
            Feature::Ecv,
            Feature::Wfxt,
            Feature::Afp,
            Feature::Rpres,
        ];
        const V8_8: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
            Feature::I8mm,
            Feature::Bf16,
            Feature::Ecv,
            Feature::Wfxt,
            Feature::Afp,
            Feature::Rpres,
            Feature::Mops,
            Feature::Hbc,
        ];
        const V8_9: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
            Feature::I8mm,
            Feature::Bf16,
            Feature::Ecv,
            Feature::Wfxt,
            Feature::Afp,
            Feature::Rpres,
            Feature::Mops,
            Feature::Hbc,
            Feature::Cssc,
        ];
        const V9_0: &[Feature] = &[
            Feature::Lse,
            Feature::Rdm,
            Feature::Crc32,
            Feature::Dcpop,
            Feature::Jscvt,
            Feature::Fcma,
            Feature::Rcpc,
            Feature::Paca,
            Feature::Pacg,
            Feature::Dit,
            Feature::Uscat,
            Feature::Rcpc2,
            Feature::Flagm,
            Feature::Dotprod,
            Feature::Ssbs,
            Feature::Sb,
            Feature::Bti,
            Feature::Frint,
            Feature::Flagm2,
            Feature::Dcpodp,
            Feature::I8mm,
            Feature::Bf16,
            Feature::Ecv,
            Feature::Wfxt,
            Feature::Afp,
            Feature::Rpres,
            Feature::Mops,
            Feature::Hbc,
            Feature::Cssc,
            Feature::Sve,
            Feature::Sve2,
        ];

        match self {
            ArmLevel::V8_0 => &[],
            ArmLevel::V8_1 => V8_1,
            ArmLevel::V8_2 => V8_2,
            ArmLevel::V8_3 => V8_3,
            ArmLevel::V8_4 => V8_4,
            ArmLevel::V8_5 => V8_5,
            ArmLevel::V8_6 => V8_6,
            ArmLevel::V8_7 => V8_7,
            ArmLevel::V8_8 => V8_8,
            ArmLevel::V8_9 => V8_9,
            ArmLevel::V9_0 => V9_0,
            ArmLevel::V9_1 | ArmLevel::V9_2 | ArmLevel::V9_3 | ArmLevel::V9_4 | ArmLevel::V9_5 => {
                V9_0
            }
        }
    }

    /// Whether [`ArmLevel::detect`] can return this level. `false` for
    /// V9_1..V9_5: their mandatory additions over v9.0 (the v8.6..v8.9 twins'
    /// features, all already in V9_0's list, plus system-level features such as
    /// FEAT_CPA/FEAT_E3DSE) leave no HWCAP trace, so a runtime probe cannot tell
    /// them apart. Without this guard every v9 CPU would detect as V9_5. They
    /// are still reachable at compile time ([`super::shortpath`]'s `v9.Xa` cfgs).
    pub const fn is_runtime_detectable(self) -> bool {
        (self as u8) <= (ArmLevel::V9_0 as u8)
    }

    /// Highest level fully covered by `set`.
    pub fn detect(set: FeatureSet) -> Self {
        ArmLevel::ALL
            .iter()
            .rev()
            .copied()
            .filter(|level| level.is_runtime_detectable())
            .find(|&level| set.contains_all(level.required_features()))
            .unwrap_or(ArmLevel::V8_0)
    }
}
