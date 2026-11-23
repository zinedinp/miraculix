use crate::x86::ops::avx::avx::Avx;
use crate::x86::ops::avx::avx2::Avx2;
use crate::x86::ops::avx512::avx512bw::Avx512Bw;
use crate::x86::ops::avx512::avx512f::Avx512f;
use crate::x86::ops::sse::sse2::Sse2;
use crate::x86::ops::sse::sse41::Sse41;

/// Scalar unpack: per `lane_bytes`-wide lane, interleave `elem`-sized halves.
fn unpack_ref<T: Copy + Default, const N: usize>(
    a: [T; N],
    b: [T; N],
    per_lane: usize,
    hi: bool,
) -> [T; N] {
    let mut out = [T::default(); N];
    for lane in 0..N / per_lane {
        let base = lane * per_lane;
        let off = if hi { per_lane / 2 } else { 0 };
        for i in 0..per_lane / 2 {
            out[base + 2 * i] = a[base + off + i];
            out[base + 2 * i + 1] = b[base + off + i];
        }
    }
    out
}

fn seq<const N: usize>(start: u8) -> [u8; N] {
    core::array::from_fn(|i| start.wrapping_add(i as u8).wrapping_mul(7))
}

#[test]
fn cvt_f32_f64_roundtrip_all_widths() {
    if let Some(t) = Sse2::detect() {
        let w = t.cvt_f32x4_to_f64x2([1.5, -2.25, 9.0, 9.0]);
        assert_eq!(w, [1.5, -2.25]);
        assert_eq!(
            t.cvt_f64x2_to_f32x4([0.1, -3.0]),
            [0.1f64 as f32, -3.0, 0.0, 0.0]
        );
    }
    if let Some(t) = Avx::detect() {
        let a = [1.0f32, -2.5, 3.25, 1e10];
        assert_eq!(t.cvt_f32x4_to_f64x4(a), a.map(f64::from));
        let d = [0.1f64, -2.0, 1e300, 5.5];
        assert_eq!(t.cvt_f64x4_to_f32x4(d), d.map(|x| x as f32));
    }
    if let Some(t) = Avx512f::detect() {
        let a: [f32; 8] = core::array::from_fn(|i| i as f32 * 1.1 - 3.0);
        assert_eq!(t.cvt_f32x8_to_f64x8(a), a.map(f64::from));
        let d: [f64; 8] = core::array::from_fn(|i| i as f64 * 0.1 + 1e-9);
        assert_eq!(t.cvt_f64x8_to_f32x8(d), d.map(|x| x as f32));
    }
}

#[test]
fn unpack_matches_scalar() {
    if let Some(t) = Sse2::detect() {
        let (a, b) = (seq::<16>(1), seq::<16>(100));
        assert_eq!(t.unpacklo_u8x16(a, b), unpack_ref(a, b, 16, false));
        assert_eq!(t.unpackhi_u8x16(a, b), unpack_ref(a, b, 16, true));
    }
    if let Some(t) = Avx2::detect() {
        let (a, b) = (seq::<32>(1), seq::<32>(100));
        assert_eq!(t.unpacklo_u8x32(a, b), unpack_ref(a, b, 16, false));
        assert_eq!(t.unpackhi_u8x32(a, b), unpack_ref(a, b, 16, true));
        let (a, b): ([i16; 16], [i16; 16]) = (
            core::array::from_fn(|i| i as i16),
            core::array::from_fn(|i| -(i as i16)),
        );
        assert_eq!(t.unpacklo_i16x16(a, b), unpack_ref(a, b, 8, false));
        assert_eq!(t.unpackhi_i16x16(a, b), unpack_ref(a, b, 8, true));
        let (a, b): ([i32; 8], [i32; 8]) = (
            core::array::from_fn(|i| i as i32),
            core::array::from_fn(|i| 100 + i as i32),
        );
        assert_eq!(t.unpacklo_i32x8(a, b), unpack_ref(a, b, 4, false));
        assert_eq!(t.unpackhi_i32x8(a, b), unpack_ref(a, b, 4, true));
        let (a, b): ([i64; 4], [i64; 4]) = ([1, 2, 3, 4], [10, 20, 30, 40]);
        assert_eq!(t.unpacklo_i64x4(a, b), unpack_ref(a, b, 2, false));
        assert_eq!(t.unpackhi_i64x4(a, b), unpack_ref(a, b, 2, true));
    }
    if let Some(t) = Avx512Bw::detect() {
        let (a, b) = (seq::<64>(1), seq::<64>(100));
        assert_eq!(t.unpacklo_u8x64(a, b), unpack_ref(a, b, 16, false));
        assert_eq!(t.unpackhi_u8x64(a, b), unpack_ref(a, b, 16, true));
        let (a, b): ([i16; 32], [i16; 32]) = (
            core::array::from_fn(|i| i as i16),
            core::array::from_fn(|i| 500 + i as i16),
        );
        assert_eq!(t.unpacklo_i16x32(a, b), unpack_ref(a, b, 8, false));
        assert_eq!(t.unpackhi_i16x32(a, b), unpack_ref(a, b, 8, true));
    }
    if let Some(t) = Avx512f::detect() {
        let (a, b): ([i32; 16], [i32; 16]) = (
            core::array::from_fn(|i| i as i32),
            core::array::from_fn(|i| 100 + i as i32),
        );
        assert_eq!(t.unpacklo_i32x16(a, b), unpack_ref(a, b, 4, false));
        assert_eq!(t.unpackhi_i32x16(a, b), unpack_ref(a, b, 4, true));
        let (a, b): ([i64; 8], [i64; 8]) = (
            core::array::from_fn(|i| i as i64),
            core::array::from_fn(|i| 100 + i as i64),
        );
        assert_eq!(t.unpacklo_i64x8(a, b), unpack_ref(a, b, 2, false));
        assert_eq!(t.unpackhi_i64x8(a, b), unpack_ref(a, b, 2, true));
    }
}

fn shuffle_ref<const N: usize>(table: [i8; N], idx: [i8; N]) -> [i8; N] {
    core::array::from_fn(|i| {
        if idx[i] < 0 {
            0
        } else {
            table[i / 16 * 16 + (idx[i] as usize & 15)]
        }
    })
}

#[test]
fn byte_shuffle_is_in_lane() {
    let idx32: [i8; 32] =
        core::array::from_fn(|i| if i % 5 == 0 { -1 } else { ((i * 3) % 32) as i8 });
    if let Some(t) = Avx2::detect() {
        let table: [i8; 32] = core::array::from_fn(|i| i as i8 + 1);
        assert_eq!(t.shuffle_i8x32(table, idx32), shuffle_ref(table, idx32));
    }
    if let Some(t) = Avx512Bw::detect() {
        let idx: [i8; 64] = core::array::from_fn(|i| {
            if i % 7 == 0 {
                -128
            } else {
                ((i * 5) % 64) as i8
            }
        });
        let table: [i8; 64] = core::array::from_fn(|i| i as i8 + 1);
        assert_eq!(t.shuffle_i8x64(table, idx), shuffle_ref(table, idx));
    }
}

#[test]
fn zero_extend_u8_u16() {
    let bytes: [u8; 32] = core::array::from_fn(|i| 255 - i as u8 * 3);
    if let Some(t) = Sse41::detect() {
        let a: [u8; 16] = core::array::from_fn(|i| bytes[i]);
        assert_eq!(
            t.cvtepu8_u16x8(a),
            core::array::from_fn(|i| u16::from(a[i]))
        );
    }
    if let Some(t) = Avx2::detect() {
        let a: [u8; 16] = core::array::from_fn(|i| bytes[i]);
        assert_eq!(
            t.cvtepu8_u16x16(a),
            core::array::from_fn(|i| u16::from(a[i]))
        );
    }
    if let Some(t) = Avx512Bw::detect() {
        assert_eq!(
            t.cvtepu8_u16x32(bytes),
            core::array::from_fn(|i| u16::from(bytes[i]))
        );
    }
}

#[test]
fn runtime_count_shifts() {
    let a8: [u16; 8] = core::array::from_fn(|i| 0x8001u16.wrapping_mul(i as u16 + 1));
    for count in [0u32, 1, 7, 15, 16, 40, u32::MAX] {
        let l = |x: u16| if count >= 16 { 0 } else { x << count };
        let r = |x: u16| if count >= 16 { 0 } else { x >> count };
        if let Some(t) = Sse2::detect() {
            assert_eq!(t.sll_u16x8(a8, count), a8.map(l));
            assert_eq!(t.srl_u16x8(a8, count), a8.map(r));
        }
        let a16: [u16; 16] = core::array::from_fn(|i| a8[i % 8] ^ i as u16);
        if let Some(t) = Avx2::detect() {
            assert_eq!(t.sll_u16x16(a16, count), a16.map(l));
            assert_eq!(t.srl_u16x16(a16, count), a16.map(r));
        }
        let a32: [u16; 32] = core::array::from_fn(|i| a8[i % 8] ^ i as u16);
        if let Some(t) = Avx512Bw::detect() {
            assert_eq!(t.sll_u16x32(a32, count), a32.map(l));
            assert_eq!(t.srl_u16x32(a32, count), a32.map(r));
        }
    }
}

#[test]
fn bit_k_of_every_byte_via_sll_movemask() {
    // The use case from the TODO: movemask(sll(v, 7 - bit)) == bit `bit` of every byte.
    let Some(t) = Sse2::detect() else { return };
    let bytes: [u8; 16] = seq(3);
    let words: [u16; 8] =
        core::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    for bit in 0..8u32 {
        let shifted = t.sll_u16x8(words, 7 - bit);
        let sh_bytes: [i8; 16] =
            core::array::from_fn(|i| shifted[i / 2].to_le_bytes()[i % 2] as i8);
        let got = t.movemask_i8x16(sh_bytes) as u32 & 0xFFFF;
        let want = (0..16).fold(0u32, |m, i| m | (u32::from(bytes[i] >> bit & 1) << i));
        assert_eq!(got, want, "bit {bit}");
    }
}

#[test]
fn partial_loads() {
    let Some(t) = Sse2::detect() else { return };
    assert_eq!(
        t.loadl_u8x8([1, 2, 3, 4, 5, 6, 7, 8]),
        [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(t.cvtsi32_u32x4(0xDEAD_BEEF), [0xDEAD_BEEF, 0, 0, 0]);
    assert_eq!(t.cvtsi64_u64x2(u64::MAX - 1), [u64::MAX - 1, 0]);
}

#[test]
fn cross_lane_fixups() {
    if let Some(t) = Avx2::detect() {
        let a = [1i64, 2, 3, 4];
        let b = [5i64, 6, 7, 8];
        assert_eq!(t.permute2x128_i64x4::<0x20>(a, b), [1, 2, 5, 6]);
        assert_eq!(t.permute2x128_i64x4::<0x31>(a, b), [3, 4, 7, 8]);
        assert_eq!(t.permute2x128_i64x4::<0x01>(a, b), [3, 4, 1, 2]);
        assert_eq!(t.permute2x128_i64x4::<0x80>(a, b), [1, 2, 0, 0]);
        let lane: [u8; 16] = seq(9);
        let w = t.broadcast_u8x16_to_x32(lane);
        assert_eq!(&w[..16], &lane);
        assert_eq!(&w[16..], &lane);
    }
    if let Some(t) = Avx512f::detect() {
        let lane: [u8; 16] = seq(9);
        let w = t.broadcast_u8x16_to_x64(lane);
        for chunk in w.chunks(16) {
            assert_eq!(chunk, &lane);
        }
        let a: [u64; 8] = core::array::from_fn(|i| i as u64);
        let b: [u64; 8] = core::array::from_fn(|i| 100 + i as u64);
        let idx: [u64; 8] = [8, 0, 15, 7, 9, 1, 14, 6];
        let want = idx.map(|i| {
            if i & 8 != 0 {
                b[(i & 7) as usize]
            } else {
                a[(i & 7) as usize]
            }
        });
        assert_eq!(t.permutex2var_u64x8(a, idx, b), want);
    }
}

#[test]
fn byte_masks_as_u64() {
    let Some(t) = Avx512Bw::detect() else { return };
    let a: [u8; 64] = seq(5);
    for bit in 0..8 {
        let got = t.test_u8x64_mask(a, [1 << bit; 64]);
        let want = (0..64).fold(0u64, |m, i| m | (u64::from(a[i] >> bit & 1) << i));
        assert_eq!(got, want, "bit {bit}");
    }
    let b: [u8; 64] = seq(77);
    let k = 0xF0F0_1234_ABCD_0F0Fu64;
    let got = t.mask_blend_u8x64(k, a, b);
    let want: [u8; 64] = core::array::from_fn(|i| if k >> i & 1 != 0 { b[i] } else { a[i] });
    assert_eq!(got, want);
}
