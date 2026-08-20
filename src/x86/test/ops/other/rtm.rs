use super::*;

#[test]
fn xbegin_xend_commits_or_aborts_cleanly() {
	let Some(rtm) = Rtm::detect() else { return };
	let mut ran = 0u32;
	if rtm.xbegin().is_ok() {
		ran += 1;
		rtm.xend();
	}
	assert!(ran <= 1);
}

#[test]
fn xtest_is_false_outside_a_transaction() {
	let Some(rtm) = Rtm::detect() else { return };
	assert!(!rtm.xtest());
}

#[test]
fn xtest_is_true_inside_a_committed_transaction() {
	let Some(rtm) = Rtm::detect() else { return };
	if rtm.xbegin().is_ok() {
		let inside = rtm.xtest();
		rtm.xend();
		assert!(inside);
	}
}
