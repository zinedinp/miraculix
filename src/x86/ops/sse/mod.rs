//! SSE family: base [`sse`] (f32 XMM) plus
//! [`sse2`]..[`sse42`]..

// The family module and its primary/oldest extension share a name by design (zero path
// stutter: `ops::sse::Sse`).
#[allow(clippy::module_inception)]
pub mod sse;
pub mod sse2;
pub mod sse3;
pub mod sse41;
pub mod sse42;
pub mod ssse3;
