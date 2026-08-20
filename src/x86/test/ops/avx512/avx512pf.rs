use super::*;

/// Knights Landing is EOL and outside this crate's reachable test
/// matrix; this asserts `detect` fails closed rather than
/// false-positive.
#[test]
fn detect_is_none_on_this_host() {
	assert!(Avx512Pf::detect().is_none(), "host has real AVX512PF, review this test");
}

/// No observable output exists for a prefetch hint, so this only checks
/// that every method compiles with the right arity/types and (were this
/// ever run on real Phi hardware) doesn't fault against a real backing
/// buffer. `detect()` gates it to dead code on every reachable host.
#[test]
fn every_prefetch_variant_runs_against_a_valid_buffer() {
	let Some(t) = Avx512Pf::detect() else { return };
	let buf_f32 = [0f32; 64];
	let buf_f64 = [0f64; 64];
	let idx16: [u32; 16] = core::array::from_fn(|i| i as u32);
	let idx8_32: [u32; 8] = core::array::from_fn(|i| i as u32);
	let idx8_64: [u64; 8] = core::array::from_fn(|i| i as u64);
	unsafe {
		t.gatherpf0_dps(buf_f32.as_ptr(), &idx16, 0xffff);
		t.gatherpf1_dps(buf_f32.as_ptr(), &idx16, 0xffff);
		t.scatterpf0_dps(buf_f32.as_ptr(), &idx16, 0xffff);
		t.scatterpf1_dps(buf_f32.as_ptr(), &idx16, 0xffff);
		t.gatherpf0_qps(buf_f32.as_ptr(), &idx8_64, 0xff);
		t.gatherpf1_qps(buf_f32.as_ptr(), &idx8_64, 0xff);
		t.scatterpf0_qps(buf_f32.as_ptr(), &idx8_64, 0xff);
		t.scatterpf1_qps(buf_f32.as_ptr(), &idx8_64, 0xff);
		t.gatherpf0_dpd(buf_f64.as_ptr(), &idx8_32, 0xff);
		t.gatherpf1_dpd(buf_f64.as_ptr(), &idx8_32, 0xff);
		t.scatterpf0_dpd(buf_f64.as_ptr(), &idx8_32, 0xff);
		t.scatterpf1_dpd(buf_f64.as_ptr(), &idx8_32, 0xff);
		t.gatherpf0_qpd(buf_f64.as_ptr(), &idx8_64, 0xff);
		t.gatherpf1_qpd(buf_f64.as_ptr(), &idx8_64, 0xff);
		t.scatterpf0_qpd(buf_f64.as_ptr(), &idx8_64, 0xff);
		t.scatterpf1_qpd(buf_f64.as_ptr(), &idx8_64, 0xff);
	}
}
