//! SVE2 `FEAT_SVE_BitPerm` (v9.0): per-lane bit gather / scatter / group.
//! Token: [`Sve2Bitperm`].
//!
//! - `bext` - **gather**: the bits of `a` where `mask` is set, packed into
//!   the low end (x86 BMI2 `pext`).
//! - `bdep` - **scatter**: the low bits of `a`, spread to the positions set in
//!   `mask` (x86 `pdep`).
//! - `bgrp` - **group**: `bext(a, mask)` in the low bits, followed by
//!   `bext(a, !mask)` above it - a stable partition of the lane's bits.
//!
//! Neon has no equivalent, so these are token-only (no `auto_up` fn), in all
//! four lane widths. x86 only has 32/64-bit scalar `pext`/`pdep`.

use super::super::macros::{arm_token, sve_slice, sve_stream};

arm_token! {
    /// Proof that SVE2 with `FEAT_SVE_BitPerm` (`BEXT`/`BDEP`/`BGRP`) is
    /// available.
    token = Sve2Bitperm, requires = [Sve, Sve2, Sve2Bitperm],
}

macro_rules! bitperm {
	($E:ty, $t:literal, $ld:literal, $st:literal, $sh:literal, $inc:literal, $(($name:ident, $inner:ident, $asm:literal, $doc:literal)),+) => {$(
		sve_stream! {
			target_feature = "sve2-bitperm", inner_fn = $inner,
			size = [$t, $ld, $st, $sh, $inc],
			ins = [a: $E => "0", mask: $E => "1"], out = $E, result = "0",
			op = [$asm],
		}
		sve_slice! {
			token = Sve2Bitperm, slice_fn = $name, inner_fn = $inner,
			ins = [a: $E, mask: $E], out = $E,
			slice_doc = $doc,
		}
	)+};
}

bitperm!(
    u8,
    "b",
    "ld1b",
    "st1b",
    "",
    "incb",
    (
        bext_u8_slice,
        sve_bext_u8,
        "bext z0.b, z0.b, z1.b",
        "Per-lane bit gather (`pext`): the bits of `a` under `mask`, packed low (`bext`)."
    ),
    (
        bdep_u8_slice,
        sve_bdep_u8,
        "bdep z0.b, z0.b, z1.b",
        "Per-lane bit scatter (`pdep`): the low bits of `a` spread to `mask`'s set positions (`bdep`)."
    ),
    (
        bgrp_u8_slice,
        sve_bgrp_u8,
        "bgrp z0.b, z0.b, z1.b",
        "Per-lane bit group: `a`'s bits under `mask` packed low, the rest packed above them, order kept (`bgrp`)."
    )
);
bitperm!(
    u16,
    "h",
    "ld1h",
    "st1h",
    ", lsl #1",
    "inch",
    (
        bext_u16_slice,
        sve_bext_u16,
        "bext z0.h, z0.h, z1.h",
        "Per-lane bit gather (`pext`): the bits of `a` under `mask`, packed low (`bext`)."
    ),
    (
        bdep_u16_slice,
        sve_bdep_u16,
        "bdep z0.h, z0.h, z1.h",
        "Per-lane bit scatter (`pdep`): the low bits of `a` spread to `mask`'s set positions (`bdep`)."
    ),
    (
        bgrp_u16_slice,
        sve_bgrp_u16,
        "bgrp z0.h, z0.h, z1.h",
        "Per-lane bit group: `a`'s bits under `mask` packed low, the rest packed above them, order kept (`bgrp`)."
    )
);
bitperm!(
    u32,
    "s",
    "ld1w",
    "st1w",
    ", lsl #2",
    "incw",
    (
        bext_u32_slice,
        sve_bext_u32,
        "bext z0.s, z0.s, z1.s",
        "Per-lane bit gather (`pext`): the bits of `a` under `mask`, packed low (`bext`)."
    ),
    (
        bdep_u32_slice,
        sve_bdep_u32,
        "bdep z0.s, z0.s, z1.s",
        "Per-lane bit scatter (`pdep`): the low bits of `a` spread to `mask`'s set positions (`bdep`)."
    ),
    (
        bgrp_u32_slice,
        sve_bgrp_u32,
        "bgrp z0.s, z0.s, z1.s",
        "Per-lane bit group: `a`'s bits under `mask` packed low, the rest packed above them, order kept (`bgrp`)."
    )
);
bitperm!(
    u64,
    "d",
    "ld1d",
    "st1d",
    ", lsl #3",
    "incd",
    (
        bext_u64_slice,
        sve_bext_u64,
        "bext z0.d, z0.d, z1.d",
        "Per-lane bit gather (`pext`): the bits of `a` under `mask`, packed low (`bext`)."
    ),
    (
        bdep_u64_slice,
        sve_bdep_u64,
        "bdep z0.d, z0.d, z1.d",
        "Per-lane bit scatter (`pdep`): the low bits of `a` spread to `mask`'s set positions (`bdep`)."
    ),
    (
        bgrp_u64_slice,
        sve_bgrp_u64,
        "bgrp z0.d, z0.d, z1.d",
        "Per-lane bit group: `a`'s bits under `mask` packed low, the rest packed above them, order kept (`bgrp`)."
    )
);

#[cfg(test)]
#[path = "../../test/ops/sve/sve_bitperm.rs"]
mod tests;
