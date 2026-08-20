use super::Vfpv4Neon;

fn require() -> Option<Vfpv4Neon> {
	Vfpv4Neon::detect()
}

#[test]
fn fma_f32x4_matches_scalar_b_times_c_plus_a() {
	let Some(t) = require() else { return };
	let a = [1.0, -2.5, 0.0, 100.0];
	let b = [2.0, 3.0, 5.0, -1.0];
	let c = [3.0, -4.0, 7.0, 0.5];
	let expect: [f32; 4] = core::array::from_fn(|i| b[i] * c[i] + a[i]);
	assert_eq!(t.fma_f32x4(a, b, c), expect);
}
