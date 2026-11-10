//! SVE2 crypto: `FEAT_SVE_AES` (AES rounds) and `FEAT_SVE_PMULL128`
//! (64×64→128 carry-less multiply). Tokens: [`Sve2Aes`], [`Sve2Pmull`].
//!
//! Same instructions as the Neon [`Aes`](super::super::crypto::aes::Aes) /
//! [`Pmull`](super::super::crypto::pmull::Pmull) tokens, applied to every
//! 128-bit block of a slice in one predicated loop - ECB-shaped, i.e. many
//! **independent** blocks (CTR keystream, several streams in parallel), not a
//! chained mode, which is serial by nature.
//!
//! Block ops panic unless the length is a multiple of 16 bytes (AES) or of 2
//! lanes (PMULL): the instructions work per 128-bit segment, and a partial
//! segment has no meaning.

use super::super::macros::{arm_token, sve_stream};

arm_token! {
    /// Proof that SVE2 with `FEAT_SVE_AES` (`AESE`/`AESD`/`AESMC`/`AESIMC` on
    /// `z` registers) is available.
    token = Sve2Aes, requires = [Sve, Sve2, Sve2Aes],
}

arm_token! {
    /// Proof that SVE2 with `FEAT_SVE_PMULL128` (`PMULLB`/`PMULLT` with
    /// 128-bit results) is available.
    token = Sve2Pmull, requires = [Sve, Sve2, Sve2Pmull],
}

sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_aese,
    size = ["b", "ld1b", "st1b", "", "incb"],
    ins = [data: u8 => "0", key: u8 => "1"], out = u8, result = "0",
    op = ["aese z0.b, z0.b, z1.b"],
}
sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_aesd,
    size = ["b", "ld1b", "st1b", "", "incb"],
    ins = [data: u8 => "0", key: u8 => "1"], out = u8, result = "0",
    op = ["aesd z0.b, z0.b, z1.b"],
}
sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_aesmc,
    size = ["b", "ld1b", "st1b", "", "incb"],
    ins = [data: u8 => "0"], out = u8, result = "0",
    op = ["aesmc z0.b, z0.b"],
}
sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_aesimc,
    size = ["b", "ld1b", "st1b", "", "incb"],
    ins = [data: u8 => "0"], out = u8, result = "0",
    op = ["aesimc z0.b, z0.b"],
}

impl Sve2Aes {
    /// One AES encryption round's first half on every 16-byte block:
    /// `SubBytes(ShiftRows(data ^ key))` (`aese z.b`). Block `k` of `data`
    /// pairs with block `k` of `key`.
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 16.
    #[inline]
    pub fn aese_slice(self, data: &[u8], key: &[u8], out: &mut [u8]) {
        blocks(&[data.len(), key.len()], out.len(), 16);
        unsafe { sve_aese(data.as_ptr(), key.as_ptr(), out.as_mut_ptr(), out.len()) }
    }

    /// One AES decryption round's first half on every block:
    /// `InvSubBytes(InvShiftRows(data ^ key))` (`aesd z.b`).
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 16.
    #[inline]
    pub fn aesd_slice(self, data: &[u8], key: &[u8], out: &mut [u8]) {
        blocks(&[data.len(), key.len()], out.len(), 16);
        unsafe { sve_aesd(data.as_ptr(), key.as_ptr(), out.as_mut_ptr(), out.len()) }
    }

    /// `MixColumns` on every block (`aesmc z.b`).
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 16.
    #[inline]
    pub fn aesmc_slice(self, data: &[u8], out: &mut [u8]) {
        blocks(&[data.len()], out.len(), 16);
        unsafe { sve_aesmc(data.as_ptr(), out.as_mut_ptr(), out.len()) }
    }

    /// `InvMixColumns` on every block (`aesimc z.b`).
    ///
    /// # Panics
    /// Lengths differ, or are not a multiple of 16.
    #[inline]
    pub fn aesimc_slice(self, data: &[u8], out: &mut [u8]) {
        blocks(&[data.len()], out.len(), 16);
        unsafe { sve_aesimc(data.as_ptr(), out.as_mut_ptr(), out.len()) }
    }
}

sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_pmullb,
    size = ["d", "ld1d", "st1d", ", lsl #3", "incd"],
    ins = [a: u64 => "0", b: u64 => "1"], out = u64, result = "2",
    op = ["pmullb z2.q, z0.d, z1.d"],
}
sve_stream! {
    target_feature = "sve2-aes", inner_fn = sve_pmullt,
    size = ["d", "ld1d", "st1d", ", lsl #3", "incd"],
    ins = [a: u64 => "0", b: u64 => "1"], out = u64, result = "2",
    op = ["pmullt z2.q, z0.d, z1.d"],
}

impl Sve2Pmull {
    /// Carry-less 64×64→128 multiply of the **even** lanes of every pair:
    /// `out[2k] | out[2k+1] << 64 = clmul(a[2k], b[2k])` (`pmullb z.q`). The
    /// slice form of Neon's `Pmull::pmull_u64`.
    ///
    /// # Panics
    /// Lengths differ, or are odd.
    #[inline]
    pub fn pmull_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
        blocks(&[a.len(), b.len()], out.len(), 2);
        unsafe { sve_pmullb(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), out.len()) }
    }

    /// Same for the **odd** lanes: `clmul(a[2k+1], b[2k+1])` into
    /// `out[2k..2k+2]` (`pmullt z.q`). Slice form of `Pmull::pmull_high_u64x2`.
    ///
    /// # Panics
    /// Lengths differ, or are odd.
    #[inline]
    pub fn pmull_high_u64_slice(self, a: &[u64], b: &[u64], out: &mut [u64]) {
        blocks(&[a.len(), b.len()], out.len(), 2);
        unsafe { sve_pmullt(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), out.len()) }
    }
}

/// Every input length equals `out_len`, which is a whole number of blocks.
fn blocks(ins: &[usize], out_len: usize, block: usize) {
    for &n in ins {
        assert_eq!(n, out_len, "input and output lengths differ");
    }
    assert!(
        out_len.is_multiple_of(block),
        "length {out_len} is not a multiple of {block}"
    );
}

#[cfg(test)]
#[path = "../../test/ops/sve/sve_aes.rs"]
mod tests;
