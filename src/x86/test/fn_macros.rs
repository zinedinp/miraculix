use crate::x86::detect_features;
use crate::x86::ops::avx::avx::Avx;
use crate::x86::ops::avx::avx2::Avx2;
use crate::x86::ops::avx::f16c::F16c;
use crate::x86::ops::avx::fma::Fma;
use crate::x86::ops::avx512::avx512bf16::Avx512Bf16;
use crate::x86::ops::avx512::avx512bw::Avx512Bw;
use crate::x86::ops::avx512::avx512dq::Avx512Dq;
use crate::x86::ops::avx512::avx512f::Avx512f;
use crate::x86::ops::avx512::avx512vl::Avx512FVl;
use crate::x86::ops::avx512::avx512vnni::Avx512Vnni;
use crate::x86::ops::avx512::avx512vbmi2::Avx512Vbmi2;
use crate::x86::ops::other::aes::Aes;
use crate::x86::ops::other::gfni::Gfni;
use crate::x86::ops::other::pclmulqdq::Pclmulqdq;
use crate::x86::ops::other::sha::Sha;
use crate::x86::ops::sse::sse2::Sse2;
use crate::x86::ops::sse::sse41::Sse41;
use crate::x86::ops::sse::sse42::Sse42;
use crate::x86::ops::sse::ssse3::Ssse3;
use crate::x86::GenericLevel;

crate::avx_fn! {
	fn add_via_avx_fn(avx: Avx, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
		avx.add_f32x8(a, b)
	}
}

#[test]
fn avx_fn_runs_body_and_returns_its_value() {
	let Some(avx) = Avx::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
	assert_eq!(add_via_avx_fn(avx, a, b), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
}

crate::avx512f_fn! {
	fn double_via_avx512f_fn(avx512f: Avx512f, a: [f32; 16]) -> [f32; 16] {
		avx512f.add_f32x16(a, a)
	}
}

#[test]
fn avx512f_fn_runs_body_and_returns_its_value() {
	let Some(avx512f) = Avx512f::detect() else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let expect: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.0);
	assert_eq!(double_via_avx512f_fn(avx512f, a), expect);
}

crate::avx512_fn! {
	fn extract_and_double(avx512f: Avx512f, avx: Avx, avx512dq: Avx512Dq, a: [f32; 16]) -> [f32; 8] {
		let half = avx512dq.extract_f32x8_from_x16::<1>(a);
		let doubled = avx.add_f32x8(half, half);
		// Touches `avx512f` too, proving all 3 tokens are simultaneously
		// live and callable inside one spliced body.
		avx512f.add_f32x16([0.0; 16], [0.0; 16]);
		doubled
	}
}

#[test]
fn avx512_fn_runs_body_and_returns_its_value() {
	let Some(avx512f) = Avx512f::detect() else { return };
	let Some(avx) = Avx::detect() else { return };
	let Some(avx512dq) = Avx512Dq::from_features(detect_features()) else { return };
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let expect: [f32; 8] = core::array::from_fn(|i| (8 + i) as f32 * 2.0);
	assert_eq!(extract_and_double(avx512f, avx, avx512dq, a), expect);
}

crate::f16c_fn! {
	fn roundtrip_via_f16c_fn(f16c: F16c, a: [f32; 8]) -> [f32; 8] {
		let halves = f16c.f32_to_f16x8::<{ core::arch::x86_64::_MM_FROUND_TO_NEAREST_INT }>(a);
		f16c.f16_to_f32x8(halves)
	}
}

#[test]
fn f16c_fn_runs_body_and_returns_its_value() {
	let Some(f16c) = F16c::detect() else { return };
	let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	assert_eq!(roundtrip_via_f16c_fn(f16c, a), a);
}

crate::sse41_f16c_fn! {
	fn touch_all_four(sse2: Sse2, ssse3: Ssse3, sse41: Sse41, f16c: F16c, a: [i16; 8]) -> [f32; 8] {
		let shuffled = sse2.shufflelo_i16x8::<0x1b>(a);
		let blended = sse41.blend_i16x8::<0xff>(shuffled, a);
		let bytes: [u8; 16] = core::array::from_fn(|i| {
			let lane = blended[i / 2];
			if i % 2 == 0 { lane as u8 } else { (lane >> 8) as u8 }
		});
		let aligned = ssse3.alignr_u8x16::<0>(bytes, bytes);
		let halves: [u16; 8] =
			core::array::from_fn(|i| u16::from_le_bytes([aligned[i * 2], aligned[i * 2 + 1]]));
		f16c.f16_to_f32x8(halves)
	}
}

#[test]
fn sse41_f16c_fn_runs_body_and_all_four_tokens_are_simultaneously_live() {
	let Some(sse2) = Sse2::detect() else { return };
	let Some(ssse3) = Ssse3::detect() else { return };
	let Some(sse41) = Sse41::detect() else { return };
	let Some(f16c) = F16c::detect() else { return };
	let a: [i16; 8] = core::array::from_fn(|i| i as i16);
	// Just proving it runs without SIGILL under all 4 features at once;
	// the exact shuffle result isn't the point (see `sse2.rs`/`ssse3.rs`/
	// `sse41.rs` for each op's own correctness tests).
	let _ = touch_all_four(sse2, ssse3, sse41, f16c, a);
}

crate::avx2_fn! {
	fn extract_shift_insert(avx2: Avx2, sse2: Sse2, ssse3: Ssse3, a: [u8; 32]) -> [u8; 32] {
		let lo = avx2.extract_u8x16_from_x32::<0>(a);
		let lo2 = sse2.slli_u8x16::<1>(lo);
		let hi = avx2.extract_u8x16_from_x32::<1>(a);
		let hi2 = ssse3.alignr_u8x16::<15>(hi, lo);
		let out = avx2.insert_u8x16_into_x32::<0>(a, lo2);
		avx2.insert_u8x16_into_x32::<1>(out, hi2)
	}
}

#[test]
fn avx2_fn_runs_body_and_all_three_tokens_are_simultaneously_live() {
	let Some(avx2) = Avx2::detect() else { return };
	let Some(sse2) = Sse2::detect() else { return };
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [u8; 32] = core::array::from_fn(|i| i as u8);
	// Just proving it runs without SIGILL under all 3 features at once;
	// the exact shuffle result isn't the point.
	let _ = extract_shift_insert(avx2, sse2, ssse3, a);
}

crate::avx512bw_fn! {
	fn extract_add_insert(
		f: Avx512f,
		bw: Avx512Bw,
		avx2: Avx2,
		sse2: Sse2,
		ssse3: Ssse3,
		a: [u8; 64],
	) -> [u8; 64] {
		let lo = f.extract_u8x16_from_x64::<0>(a);
		let shuffled = to_u8x16(ssse3.shuffle_i8x16(to_i8x16(lo), to_i8x16([15u8; 16])));
		let hi = f.extract_u8x16_from_x64::<1>(a);
		let hi2 = sse2.add_u8x16(hi, shuffled);
		// Touches `avx2` too, proving all 5 tokens are simultaneously live
		// and callable inside one spliced body.
		avx2.add_u8x32([0u8; 32], [0u8; 32]);
		let out = f.insert_u8x16_into_x64::<1>(a, hi2);
		bw.add_u8x64(out, [0u8; 64])
	}
}

fn to_i8x16(v: [u8; 16]) -> [i8; 16] {
	core::array::from_fn(|i| v[i] as i8)
}

fn to_u8x16(v: [i8; 16]) -> [u8; 16] {
	core::array::from_fn(|i| v[i] as u8)
}

#[test]
fn avx512bw_fn_runs_body_and_all_five_tokens_are_simultaneously_live() {
	let Some(f) = Avx512f::detect() else { return };
	let Some(bw) = Avx512Bw::from_features(detect_features()) else { return };
	let Some(avx2) = Avx2::detect() else { return };
	let Some(sse2) = Sse2::detect() else { return };
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [u8; 64] = core::array::from_fn(|i| i as u8);
	// Just proving it runs without SIGILL under all 5 features at once;
	// the exact shuffle result isn't the point.
	let _ = extract_add_insert(f, bw, avx2, sse2, ssse3, a);
}

// Newer catalog rows: smoke tests (early-return if host lacks feature).

crate::ssse3_fn! {
	fn alignr_via_ssse3_fn(ssse3: Ssse3, a: [u8; 16]) -> [u8; 16] {
		ssse3.alignr_u8x16::<0>(a, a)
	}
}

#[test]
fn ssse3_fn_runs() {
	let Some(ssse3) = Ssse3::detect() else { return };
	let a: [u8; 16] = core::array::from_fn(|i| i as u8);
	assert_eq!(alignr_via_ssse3_fn(ssse3, a), a);
}

crate::sse41_fn! {
	fn blend_via_sse41_fn(sse2: Sse2, ssse3: Ssse3, sse41: Sse41, a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
		let _ = ssse3.alignr_u8x16::<0>([0u8; 16], [0u8; 16]);
		let _ = sse2.shufflelo_i16x8::<0>(a);
		sse41.blend_i16x8::<0xff>(a, b)
	}
}

#[test]
fn sse41_fn_runs() {
	let Some(sse2) = Sse2::detect() else { return };
	let Some(ssse3) = Ssse3::detect() else { return };
	let Some(sse41) = Sse41::detect() else { return };
	let a: [i16; 8] = [0; 8];
	let b: [i16; 8] = [1; 8];
	assert_eq!(blend_via_sse41_fn(sse2, ssse3, sse41, a, b), b);
}

crate::sse42_fn! {
	fn crc_via_sse42_fn(sse42: Sse42, crc: u32, byte: u8) -> u32 {
		sse42.crc32_u8(crc, byte)
	}
}

#[test]
fn sse42_fn_runs() {
	let Some(sse42) = Sse42::detect() else { return };
	let _ = crc_via_sse42_fn(sse42, 0, 0xab);
}

crate::fma_fn! {
	fn fmadd_via_fma_fn(fma: Fma, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
		fma.fmadd_f32x8(a, b, c)
	}
}

#[test]
fn fma_fn_runs() {
	let Some(fma) = Fma::detect() else { return };
	let a = [1.0f32; 8];
	let b = [2.0f32; 8];
	let c = [3.0f32; 8];
	assert_eq!(fmadd_via_fma_fn(fma, a, b, c), [5.0; 8]);
}

crate::avx_fma_fn! {
	fn avx_then_fma(avx: Avx, fma: Fma, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
		let t = avx.add_f32x8(a, b);
		fma.fmadd_f32x8(t, b, c)
	}
}

#[test]
fn avx_fma_fn_runs() {
	let Some(avx) = Avx::detect() else { return };
	let Some(fma) = Fma::detect() else { return };
	let _ = avx_then_fma(avx, fma, [1.0; 8], [2.0; 8], [3.0; 8]);
}

crate::avx2_fma_fn! {
	fn avx2_fma_touch(avx2: Avx2, fma: Fma, sse2: Sse2, ssse3: Ssse3, a: [f32; 8]) -> [f32; 8] {
		let _ = avx2.add_u8x32([0u8; 32], [0u8; 32]);
		let _ = sse2.add_u8x16([0u8; 16], [0u8; 16]);
		let _ = ssse3.alignr_u8x16::<0>([0u8; 16], [0u8; 16]);
		fma.fmadd_f32x8(a, a, a)
	}
}

#[test]
fn avx2_fma_fn_runs() {
	let Some(avx2) = Avx2::detect() else { return };
	let Some(fma) = Fma::detect() else { return };
	let Some(sse2) = Sse2::detect() else { return };
	let Some(ssse3) = Ssse3::detect() else { return };
	let _ = avx2_fma_touch(avx2, fma, sse2, ssse3, [1.0; 8]);
}

crate::aes_fn! {
	fn aes_round(aes: Aes, state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
		aes.aesenc(state, key)
	}
}

#[test]
fn aes_fn_runs() {
	let Some(aes) = Aes::detect() else { return };
	let _ = aes_round(aes, [0u8; 16], [0u8; 16]);
}

crate::pclmulqdq_fn! {
	fn clmul_via_fn(pcl: Pclmulqdq, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
		pcl.clmul::<0>(a, b)
	}
}

#[test]
fn pclmulqdq_fn_runs() {
	let Some(pcl) = Pclmulqdq::detect() else { return };
	let _ = clmul_via_fn(pcl, [1, 0], [1, 0]);
}

crate::aes_pclmul_fn! {
	fn aes_then_clmul(aes: Aes, pcl: Pclmulqdq, state: [u8; 16], key: [u8; 16]) -> [u64; 2] {
		let s = aes.aesenc(state, key);
		let a = [
			u64::from_le_bytes(s[0..8].try_into().unwrap()),
			u64::from_le_bytes(s[8..16].try_into().unwrap()),
		];
		pcl.clmul::<0>(a, a)
	}
}

#[test]
fn aes_pclmul_fn_runs() {
	let Some(aes) = Aes::detect() else { return };
	let Some(pcl) = Pclmulqdq::detect() else { return };
	let _ = aes_then_clmul(aes, pcl, [0u8; 16], [1u8; 16]);
}

crate::sha_fn! {
	fn sha_msg1(sha: Sha, a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
		sha.sha1msg1(a, b)
	}
}

#[test]
fn sha_fn_runs() {
	let Some(sha) = Sha::detect() else { return };
	let _ = sha_msg1(sha, [0; 4], [0; 4]);
}

crate::gfni_fn! {
	fn gf_mul(gfni: Gfni, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
		gfni.gf2p8mul_epi8_u8x16(a, b)
	}
}

#[test]
fn gfni_fn_runs() {
	let Some(gfni) = Gfni::detect() else { return };
	let _ = gf_mul(gfni, [1u8; 16], [1u8; 16]);
}

crate::avx512vl_fn! {
	fn vl_zero_add(fvl: Avx512FVl, a: [f32; 8]) -> [f32; 8] {
		fvl.add_f32x8_zero_masked(0xff, a, a)
	}
}

#[test]
fn avx512vl_fn_runs() {
	let Some(fvl) = Avx512FVl::detect() else { return };
	let a = [1.0f32; 8];
	assert_eq!(vl_zero_add(fvl, a), [2.0; 8]);
}

crate::avx512vnni_fn! {
	fn vnni_dot(vnni: Avx512Vnni, src: [i32; 16], a: [u8; 64], b: [i8; 64]) -> [i32; 16] {
		vnni.dpbusd_i32x16(src, a, b)
	}
}

#[test]
fn avx512vnni_fn_runs() {
	let Some(vnni) = Avx512Vnni::detect() else { return };
	let _ = vnni_dot(vnni, [0i32; 16], [0u8; 64], [0i8; 64]);
}

crate::avx512vbmi2_fn! {
	fn shldv_via_fn(v: Avx512Vbmi2, a: [u16; 32], b: [u16; 32], c: [u16; 32]) -> [u16; 32] {
		v.shldv_u16x32(a, b, c)
	}
}

#[test]
fn avx512vbmi2_fn_runs() {
	let Some(v) = Avx512Vbmi2::detect() else { return };
	let _ = shldv_via_fn(v, [0u16; 32], [0u16; 32], [0u16; 32]);
}

crate::avx512bf16_fn! {
	fn bf16_cvt(t: Avx512Bf16, a: [f32; 16]) -> [u16; 16] {
		t.cvtneps_pbh_u16x16(a)
	}
}

#[test]
fn avx512bf16_fn_runs() {
	let Some(t) = Avx512Bf16::detect() else { return };
	let _ = bf16_cvt(t, [1.0f32; 16]);
}

crate::avx_v3_fn! {
	fn v3_add_fmadd(avx: Avx, fma: Fma, a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
		let t = avx.add_f32x8(a, b);
		fma.fmadd_f32x8(t, b, c)
	}
}

#[test]
fn avx_v3_fn_runs() {
	let Some(avx) = Avx::detect() else { return };
	let Some(fma) = Fma::detect() else { return };
	// Also need the rest of V3 on host for the enable string; skip if not.
	let set = detect_features();
	if !set.contains_all(GenericLevel::V3.required_features()) {
		return;
	}
	let _ = v3_add_fmadd(avx, fma, [1.0; 8], [2.0; 8], [3.0; 8]);
}

crate::avx512_v4_fn! {
	fn v4_double(f: Avx512f, a: [f32; 16]) -> [f32; 16] {
		f.add_f32x16(a, a)
	}
}

#[test]
fn avx512_v4_fn_runs() {
	let Some(f) = Avx512f::detect() else { return };
	let set = detect_features();
	if !set.contains_all(GenericLevel::V4.required_features()) {
		return;
	}
	let a: [f32; 16] = core::array::from_fn(|i| i as f32);
	let expect: [f32; 16] = core::array::from_fn(|i| i as f32 * 2.0);
	assert_eq!(v4_double(f, a), expect);
}
