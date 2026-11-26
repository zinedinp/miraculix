//! Windows: `GetLogicalProcessorInformationEx(RelationCache)`, one record per
//! cache instance with the affinity mask of every CPU that shares it. The
//! buffer goes on the process heap (no `alloc` in this crate); decoding is in
//! [`super::parse::windows_matrix`].

use super::CacheMatrix;
use core::ffi::c_void;
use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapAlloc, HeapFree};
use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, RelationCache,
};

pub(super) fn probe() -> Option<CacheMatrix> {
    let mut len = 0u32;
    // SAFETY: a null buffer with length 0 only asks for the needed length.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, core::ptr::null_mut(), &mut len) };
    // The size can grow between calls (CPU hot-add); retry a few times.
    for _ in 0..4 {
        if len == 0 {
            return None;
        }
        let buf = HeapBuf::new(len as usize)?;
        let mut got = len;
        // SAFETY: `buf` holds `got` writable bytes, 16-byte aligned (heap).
        let ok =
            unsafe { GetLogicalProcessorInformationEx(RelationCache, buf.ptr.cast(), &mut got) };
        if ok != 0 {
            let n = (got as usize).min(buf.len);
            // SAFETY: the call wrote `got` bytes into the `buf.len`-byte block.
            let bytes = unsafe { core::slice::from_raw_parts(buf.ptr.cast::<u8>(), n) };
            return super::parse::windows_matrix(bytes, size_of::<usize>());
        }
        len = got.max(len.saturating_mul(2));
    }
    None
}

/// A `HeapAlloc` block, freed on drop.
struct HeapBuf {
    ptr: *mut c_void,
    len: usize,
}

impl HeapBuf {
    fn new(len: usize) -> Option<Self> {
        // SAFETY: the process heap always exists; flags 0 = no options.
        let ptr = unsafe { HeapAlloc(GetProcessHeap(), 0, len) };
        (!ptr.is_null()).then_some(HeapBuf { ptr, len })
    }
}

impl Drop for HeapBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `HeapAlloc` on the process heap, freed once.
        unsafe { HeapFree(GetProcessHeap(), 0, self.ptr) };
    }
}
