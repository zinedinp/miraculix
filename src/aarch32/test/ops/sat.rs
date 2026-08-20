use super::{ssat, usat};

#[test]
fn ssat_matches_scalar_signed_saturate() {
	for x in [0i32, 1, -1, 127, 128, -128, -129, 1000, -1000, i32::MAX, i32::MIN] {
		let expect = x.clamp(-128, 127);
		assert_eq!(ssat::<8>(x), expect, "ssat::<8>({x})");
	}
}

#[test]
fn usat_matches_scalar_unsigned_saturate() {
	for x in [0i32, 1, -1, 255, 256, -1000, 1000, i32::MAX, i32::MIN] {
		let expect = x.clamp(0, 255) as u32;
		assert_eq!(usat::<8>(x), expect, "usat::<8>({x})");
	}
}
