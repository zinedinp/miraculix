//! SVE / SVE2 (scalable vectors). Every op is `asm!` under a stable target
//! feature. Rust's SVE intrinsics are nightly-only (`stdarch#1509`).
//!
//! | File | Arm version | Feature | Token |
//! |---|---|---|---|
//! | [`sve`] | v8.2 | `FEAT_SVE` | `Sve` |
//! | [`vl`] | - | `FEAT_SVE` + VL probe | `SveVl128` / `SveVl256` / `SveVl512` |
//! | [`sve2`] | v9.0 | `FEAT_SVE2` | `Sve2` |
//! | [`sve_bitperm`] | v9.0 | `FEAT_SVE_BitPerm` | `Sve2Bitperm` |
//! | [`sve_aes`] | v9.0 | `FEAT_SVE_AES` + `FEAT_SVE_PMULL128` | `Sve2Aes` |
//! | [`sve_i8mm`] | v8.6 | `FEAT_I8MM` + SVE | `SveI8mm` |
//! | [`sve_f32mm`] | v8.6 | `FEAT_F32MM` | `SveF32mm` |
//! | [`sve_f64mm`] | v8.6 | `FEAT_F64MM` | `SveF64mm` |
//! | [`sve_bf16`] | v8.6 | `FEAT_BF16` + SVE | `SveBf16` |
//!
//! The streaming ops take the `_slice` form (one `whilelo`-predicated loop,
//! no scalar tail) with the `Neon` token's names, and `auto_up` uses `Sve` /
//! `Sve2` as its upper rungs. See `ops/macros/sve.rs` for the loop.

// Family module and its baseline file share a name, as `neon::neon`.
#[allow(clippy::module_inception)]
pub mod sve;
pub mod sve2;
pub mod sve_aes;
pub mod sve_bf16;
pub mod sve_bitperm;
pub mod sve_f32mm;
pub mod sve_f64mm;
pub mod sve_i8mm;
pub mod vl;

/// Length contract shared by the accumulate-into-`out` ops: `acc.len() ==
/// out.len()`, a whole number of `group`-lane segments, and every narrow
/// input `ratio` times as long (4 bytes or 2 `bf16`s per 32-bit lane).
fn check_acc_shape(acc: usize, out: usize, ins: &[usize], ratio: usize, group: usize) {
    assert_eq!(acc, out, "`acc` and `out` lengths differ");
    assert!(
        out.is_multiple_of(group),
        "length {out} is not a multiple of the {group}-lane segment"
    );
    for &n in ins {
        assert_eq!(
            n,
            out * ratio,
            "input length {n} is not {ratio} x `out.len()` ({out})"
        );
    }
}
