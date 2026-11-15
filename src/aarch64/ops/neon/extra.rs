//! Neon ops that byte-table, bitset and mixed-precision kernels need:
//! f32 <-> f64, byte zip, bit test, `movemask`, table lookups (`tbl`/`tbx`),
//! 64-bit-half byte ops and a 256-entry byte lookup recipe.
//!
//! Names line up with the x86 side where the operation is the same, so a
//! kernel can be written once: `movemask_i8x16`, `unpacklo_u8x16` /
//! `unpackhi_u8x16`, `cvt_f32x4_to_f64x2`, `cvt_f64x2_to_f32x4`,
//! `cvt_f32x4_to_f64x4`, `cvt_f64x4_to_f32x4`. Where the semantics differ the
//! Neon name is used (`tbl_u8x16` is not `pshufb`: an index `>= 16` gives 0,
//! and the high bit is not special).
//!
//! **Big-endian.** Array lane `i` is vector lane `i` on both endiannesses:
//! loads and stores go through lane-normalising helpers (plain `vld1`/`vst1`
//! on little-endian, with a lane reversal on `aarch64_be`). Checked under
//! `qemu-aarch64_be` (see README). Code of your own that reinterprets `u64`
//! words as bytes is only valid on little-endian.

use core::arch::aarch64::*;

use super::neon::Neon;

// Lane-order normalising loads and stores.
//
// On little-endian these are plain `vld1`/`vst1`. On `aarch64_be` the stdarch
// load/store of a vector with a lane-position-dependent op (zip, half
// extraction, `tbl`, weights) put array lane `i` at vector lane `N-1-i`, found
// by running these ops under `qemu-aarch64_be`. Reversing the lanes after each
// load and before each store restores "array lane `i` is vector lane `i`".
// Reversal is its own inverse, so the same `rev_*` serves both directions.

#[cfg(target_endian = "little")]
macro_rules! lane_io {
    ($ld:ident, $st:ident, $vec:ty, $elem:ty, $vld:ident, $vst:ident, $rev:expr) => {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn $ld(p: *const $elem) -> $vec {
            unsafe { $vld(p) }
        }
        #[inline]
        #[allow(dead_code)]
        #[target_feature(enable = "neon")]
        unsafe fn $st(p: *mut $elem, v: $vec) {
            unsafe { $vst(p, v) }
        }
    };
}

#[cfg(target_endian = "big")]
macro_rules! lane_io {
    ($ld:ident, $st:ident, $vec:ty, $elem:ty, $vld:ident, $vst:ident, $rev:expr) => {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn $ld(p: *const $elem) -> $vec {
            let rev: unsafe fn($vec) -> $vec = $rev;
            unsafe { rev($vld(p)) }
        }
        #[inline]
        #[allow(dead_code)]
        #[target_feature(enable = "neon")]
        unsafe fn $st(p: *mut $elem, v: $vec) {
            let rev: unsafe fn($vec) -> $vec = $rev;
            unsafe { $vst(p, rev(v)) }
        }
    };
}

// Full-register lane reversal: reverse inside each 64-bit half, swap halves.
#[cfg(target_endian = "big")]
#[inline]
#[target_feature(enable = "neon")]
fn rev_u8x16(v: uint8x16_t) -> uint8x16_t {
    let r = vrev64q_u8(v);
    vextq_u8::<8>(r, r)
}
#[cfg(target_endian = "big")]
#[inline]
#[target_feature(enable = "neon")]
fn rev_s8x16(v: int8x16_t) -> int8x16_t {
    let r = vrev64q_s8(v);
    vextq_s8::<8>(r, r)
}
#[cfg(target_endian = "big")]
#[inline]
#[target_feature(enable = "neon")]
fn rev_f32x4(v: float32x4_t) -> float32x4_t {
    let r = vrev64q_f32(v);
    vextq_f32::<2>(r, r)
}
#[cfg(target_endian = "big")]
#[inline]
#[target_feature(enable = "neon")]
fn rev_f64x2(v: float64x2_t) -> float64x2_t {
    vextq_f64::<1>(v, v)
}

lane_io!(ld_u8x16, st_u8x16, uint8x16_t, u8, vld1q_u8, vst1q_u8, rev_u8x16);
lane_io!(ld_s8x16, st_s8x16, int8x16_t, i8, vld1q_s8, vst1q_s8, rev_s8x16);
lane_io!(ld_f32x4, st_f32x4, float32x4_t, f32, vld1q_f32, vst1q_f32, rev_f32x4);
lane_io!(ld_f64x2, st_f64x2, float64x2_t, f64, vld1q_f64, vst1q_f64, rev_f64x2);
lane_io!(ld_u8x8, st_u8x8, uint8x8_t, u8, vld1_u8, vst1_u8, vrev64_u8);
lane_io!(ld_f32x2, st_f32x2, float32x2_t, f32, vld1_f32, vst1_f32, vrev64_f32);

/// Four consecutive 16-byte table registers, each lane-normalised.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn ld_u8x16_x4(p: *const u8) -> uint8x16x4_t {
    unsafe { uint8x16x4_t(ld_u8x16(p), ld_u8x16(p.add(16)), ld_u8x16(p.add(32)), ld_u8x16(p.add(48))) }
}

// f32 <-> f64

impl Neon {
    /// Widens two f32 lanes to f64 (`fcvtl`).
    #[inline]
    pub fn cvt_f32x2_to_f64x2(self, a: [f32; 2]) -> [f64; 2] {
        unsafe { cvt_f32x2_to_f64x2_impl(&a) }
    }

    /// Widens the **low two** f32 lanes to f64 (`fcvtl`); lanes 2 and 3 are
    /// ignored. Same name and meaning as `Sse2::cvt_f32x4_to_f64x2`.
    #[inline]
    pub fn cvt_f32x4_to_f64x2(self, a: [f32; 4]) -> [f64; 2] {
        unsafe { cvt_f32x4_to_f64x2_impl(&a) }
    }

    /// Widens the **high two** f32 lanes to f64 (`fcvtl2`).
    #[inline]
    pub fn cvt_high_f32x4_to_f64x2(self, a: [f32; 4]) -> [f64; 2] {
        unsafe { cvt_high_f32x4_to_f64x2_impl(&a) }
    }

    /// Widens four f32 lanes to f64 (`fcvtl` + `fcvtl2`), in order.
    #[inline]
    pub fn cvt_f32x4_to_f64x4(self, a: [f32; 4]) -> [f64; 4] {
        unsafe { cvt_f32x4_to_f64x4_impl(&a) }
    }

    /// Narrows two f64 lanes to f32, round-to-nearest (`fcvtn`).
    #[inline]
    pub fn cvt_f64x2_to_f32x2(self, a: [f64; 2]) -> [f32; 2] {
        unsafe { cvt_f64x2_to_f32x2_impl(&a) }
    }

    /// Narrows two f64 lanes to f32; lanes 2 and 3 are zero. Same name and
    /// meaning as `Sse2::cvt_f64x2_to_f32x4`.
    #[inline]
    pub fn cvt_f64x2_to_f32x4(self, a: [f64; 2]) -> [f32; 4] {
        unsafe { cvt_f64x2_to_f32x4_impl(&a) }
    }

    /// Narrows four f64 lanes to f32, round-to-nearest (`fcvtn` + `fcvtn2`).
    #[inline]
    pub fn cvt_f64x4_to_f32x4(self, a: [f64; 4]) -> [f32; 4] {
        unsafe { cvt_f64x4_to_f32x4_impl(&a) }
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f32x2_to_f64x2_impl(a: &[f32; 2]) -> [f64; 2] {
    unsafe {
        let mut out = [0f64; 2];
        st_f64x2(out.as_mut_ptr(), vcvt_f64_f32(ld_f32x2(a.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f32x4_to_f64x2_impl(a: &[f32; 4]) -> [f64; 2] {
    unsafe {
        let mut out = [0f64; 2];
        st_f64x2(out.as_mut_ptr(), vcvt_f64_f32(vget_low_f32(ld_f32x4(a.as_ptr()))));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_high_f32x4_to_f64x2_impl(a: &[f32; 4]) -> [f64; 2] {
    unsafe {
        let mut out = [0f64; 2];
        st_f64x2(out.as_mut_ptr(), vcvt_high_f64_f32(ld_f32x4(a.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f32x4_to_f64x4_impl(a: &[f32; 4]) -> [f64; 4] {
    unsafe {
        let v = ld_f32x4(a.as_ptr());
        let mut out = [0f64; 4];
        st_f64x2(out.as_mut_ptr(), vcvt_f64_f32(vget_low_f32(v)));
        st_f64x2(out.as_mut_ptr().add(2), vcvt_high_f64_f32(v));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f64x2_to_f32x2_impl(a: &[f64; 2]) -> [f32; 2] {
    unsafe {
        let mut out = [0f32; 2];
        st_f32x2(out.as_mut_ptr(), vcvt_f32_f64(ld_f64x2(a.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f64x2_to_f32x4_impl(a: &[f64; 2]) -> [f32; 4] {
    unsafe {
        let lo = vcvt_f32_f64(ld_f64x2(a.as_ptr()));
        let mut out = [0f32; 4];
        st_f32x4(out.as_mut_ptr(), vcombine_f32(lo, vdup_n_f32(0.0)));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cvt_f64x4_to_f32x4_impl(a: &[f64; 4]) -> [f32; 4] {
    unsafe {
        let lo = vcvt_f32_f64(ld_f64x2(a.as_ptr()));
        let v = vcvt_high_f32_f64(lo, ld_f64x2(a.as_ptr().add(2)));
        let mut out = [0f32; 4];
        st_f32x4(out.as_mut_ptr(), v);
        out
    }
}

// Bytes: test, zip, shift, horizontal add

impl Neon {
    /// Per-byte bit test: `0xFF` where `a[i] & b[i] != 0`, else `0` (`cmtst`).
    #[inline]
    pub fn test_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        unsafe { test_u8x16_impl(&a, &b) }
    }

    /// Interleaves the low 8 bytes: `[a0, b0, a1, b1, ...]` (`zip1 v.16b`).
    #[inline]
    pub fn zip1_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        unsafe { zip_u8x16_impl::<false>(&a, &b) }
    }

    /// Interleaves the high 8 bytes: `[a8, b8, a9, b9, ...]` (`zip2 v.16b`).
    #[inline]
    pub fn zip2_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        unsafe { zip_u8x16_impl::<true>(&a, &b) }
    }

    /// Same as [`Neon::zip1_u8x16`], under the x86 name (`punpcklbw`).
    #[inline]
    pub fn unpacklo_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        self.zip1_u8x16(a, b)
    }

    /// Same as [`Neon::zip2_u8x16`], under the x86 name (`punpckhbw`).
    #[inline]
    pub fn unpackhi_u8x16(self, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        self.zip2_u8x16(a, b)
    }

    /// Interleaves the low 4 bytes of each 8-byte input (`zip1 v.8b`).
    #[inline]
    pub fn zip1_u8x8(self, a: [u8; 8], b: [u8; 8]) -> [u8; 8] {
        unsafe { zip_u8x8_impl::<false>(&a, &b) }
    }

    /// Interleaves the high 4 bytes of each 8-byte input (`zip2 v.8b`).
    #[inline]
    pub fn zip2_u8x8(self, a: [u8; 8], b: [u8; 8]) -> [u8; 8] {
        unsafe { zip_u8x8_impl::<true>(&a, &b) }
    }

    /// Per-lane logical shift right of 8 bytes by `N` (`ushr v.8b`). `N` must be `1..=8`.
    #[inline]
    pub fn srli_u8x8<const N: i32>(self, a: [u8; 8]) -> [u8; 8] {
        unsafe { shr_n_u8x8_impl::<N>(&a) }
    }

    /// Sum of 8 bytes, wrapping (`addv b, v.8b`).
    #[inline]
    pub fn sum_u8x8(self, a: [u8; 8]) -> u8 {
        unsafe { sum_u8x8_impl(&a) }
    }

    /// Sign-bit mask, one bit per byte, bit `i` = sign of `a[i]`. Same name,
    /// type and meaning as `Sse2::movemask_i8x16` (`pmovmskb`); Neon has no
    /// such instruction, so this is `cmlt` + AND with lane weights + two
    /// `addv`.
    #[inline]
    pub fn movemask_i8x16(self, a: [i8; 16]) -> u16 {
        unsafe { movemask_i8x16_impl(&a) }
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn test_u8x16_impl(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let mut out = [0u8; 16];
        st_u8x16(out.as_mut_ptr(), vtstq_u8(ld_u8x16(a.as_ptr()), ld_u8x16(b.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn zip_u8x16_impl<const HIGH: bool>(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let (va, vb) = (ld_u8x16(a.as_ptr()), ld_u8x16(b.as_ptr()));
        let vr = if HIGH { vzip2q_u8(va, vb) } else { vzip1q_u8(va, vb) };
        let mut out = [0u8; 16];
        st_u8x16(out.as_mut_ptr(), vr);
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn zip_u8x8_impl<const HIGH: bool>(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    unsafe {
        let (va, vb) = (ld_u8x8(a.as_ptr()), ld_u8x8(b.as_ptr()));
        let vr = if HIGH { vzip2_u8(va, vb) } else { vzip1_u8(va, vb) };
        let mut out = [0u8; 8];
        st_u8x8(out.as_mut_ptr(), vr);
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn shr_n_u8x8_impl<const N: i32>(a: &[u8; 8]) -> [u8; 8] {
    unsafe {
        let mut out = [0u8; 8];
        st_u8x8(out.as_mut_ptr(), vshr_n_u8::<N>(ld_u8x8(a.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sum_u8x8_impl(a: &[u8; 8]) -> u8 {
    unsafe { vaddv_u8(ld_u8x8(a.as_ptr())) }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn movemask_i8x16_impl(a: &[i8; 16]) -> u16 {
    unsafe {
        const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let neg = vcltzq_s8(ld_s8x16(a.as_ptr()));
        let bits = vandq_u8(neg, ld_u8x16(WEIGHTS.as_ptr()));
        let lo = vaddv_u8(vget_low_u8(bits)) as u16;
        let hi = vaddv_u8(vget_high_u8(bits)) as u16;
        lo | (hi << 8)
    }
}

// Table lookups (`tbl` / `tbx`): an index outside the table gives 0 (`tbl`)
// or leaves the `fallback` byte (`tbx`).

impl Neon {
    /// 16-entry byte lookup (`tbl v.16b`): `out[i] = table[idx[i]]`, or 0 if
    /// `idx[i] >= 16`. Unlike x86 `pshufb`, only the range matters, not bit 7.
    #[inline]
    pub fn tbl_u8x16(self, table: [u8; 16], idx: [u8; 16]) -> [u8; 16] {
        unsafe { tbl_u8x16_impl(&table, &idx) }
    }

    /// 16-entry byte lookup that keeps `fallback[i]` for `idx[i] >= 16` (`tbx`).
    #[inline]
    pub fn tbx_u8x16(self, fallback: [u8; 16], table: [u8; 16], idx: [u8; 16]) -> [u8; 16] {
        unsafe { tbx_u8x16_impl(&fallback, &table, &idx) }
    }

    /// 64-entry byte lookup over four registers (`tbl` with a 4-register
    /// list): `out[i] = table[idx[i]]`, or 0 if `idx[i] >= 64`.
    #[inline]
    pub fn tbl4_u8x64(self, table: [u8; 64], idx: [u8; 16]) -> [u8; 16] {
        unsafe { tbl4_u8x64_impl(&table, &idx) }
    }

    /// 64-entry byte lookup that keeps `fallback[i]` for `idx[i] >= 64` (`tbx`).
    #[inline]
    pub fn tbx4_u8x64(self, fallback: [u8; 16], table: [u8; 64], idx: [u8; 16]) -> [u8; 16] {
        unsafe { tbx4_u8x64_impl(&fallback, &table, &idx) }
    }

    /// 256-entry byte lookup: `out[i] = table[idx[i] as usize]` for every
    /// index. Four 64-entry `tbl`/`tbx` steps; the later steps use `idx - 64k`
    /// so that out-of-range indices keep the bytes found so far.
    #[inline]
    pub fn lookup_u8x256(self, table: &[u8; 256], idx: [u8; 16]) -> [u8; 16] {
        unsafe { lookup_u8x256_impl(table, &idx) }
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tbl_u8x16_impl(table: &[u8; 16], idx: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let mut out = [0u8; 16];
        st_u8x16(out.as_mut_ptr(), vqtbl1q_u8(ld_u8x16(table.as_ptr()), ld_u8x16(idx.as_ptr())));
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tbx_u8x16_impl(fallback: &[u8; 16], table: &[u8; 16], idx: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let mut out = [0u8; 16];
        let r = vqtbx1q_u8(ld_u8x16(fallback.as_ptr()), ld_u8x16(table.as_ptr()), ld_u8x16(idx.as_ptr()));
        st_u8x16(out.as_mut_ptr(), r);
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tbl4_u8x64_impl(table: &[u8; 64], idx: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let mut out = [0u8; 16];
        let r = vqtbl4q_u8(ld_u8x16_x4(table.as_ptr()), ld_u8x16(idx.as_ptr()));
        st_u8x16(out.as_mut_ptr(), r);
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tbx4_u8x64_impl(fallback: &[u8; 16], table: &[u8; 64], idx: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let mut out = [0u8; 16];
        let r = vqtbx4q_u8(ld_u8x16(fallback.as_ptr()), ld_u8x16_x4(table.as_ptr()), ld_u8x16(idx.as_ptr()));
        st_u8x16(out.as_mut_ptr(), r);
        out
    }
}

/// # Safety
/// Caller proved Neon via [`Neon`].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn lookup_u8x256_impl(table: &[u8; 256], idx: &[u8; 16]) -> [u8; 16] {
    unsafe {
        let vi = ld_u8x16(idx.as_ptr());
        let base = table.as_ptr();
        let mut r = vqtbl4q_u8(ld_u8x16_x4(base), vi);
        r = vqtbx4q_u8(r, ld_u8x16_x4(base.add(64)), vsubq_u8(vi, vdupq_n_u8(64)));
        r = vqtbx4q_u8(r, ld_u8x16_x4(base.add(128)), vsubq_u8(vi, vdupq_n_u8(128)));
        r = vqtbx4q_u8(r, ld_u8x16_x4(base.add(192)), vsubq_u8(vi, vdupq_n_u8(192)));
        let mut out = [0u8; 16];
        st_u8x16(out.as_mut_ptr(), r);
        out
    }
}

#[cfg(test)]
#[path = "../../test/ops/neon/extra.rs"]
mod tests;
