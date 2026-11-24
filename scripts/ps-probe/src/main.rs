//! Paired-single probe for Dolphin (GameCube / Wii). Bare metal, supervisor
//! mode: enables FP and paired singles itself, then checks every
//! `PairedSingle` kernel and `powerpc::auto` against scalar references on
//! inputs whose results are exact in `f32` (quarters), at several alignments.
#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use miraculix::powerpc::class::{PowerClass, PowerCore};
use miraculix::powerpc::ops::paired::PairedSingle;
use miraculix::powerpc::{Feature, auto, detect_features};

// Entry: zero .bss (the detection caches must start at 0), own stack with a
// terminated back chain, then Rust.
global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    "_start:",
    "lis 3, __bss_start@ha",
    "addi 3, 3, __bss_start@l",
    "lis 4, __bss_end@ha",
    "addi 4, 4, __bss_end@l",
    "li 5, 0",
    "1:",
    "cmplw 3, 4",
    "bge 2f",
    "stw 5, 0(3)",
    "addi 3, 3, 4",
    "b 1b",
    "2:",
    "lis 1, __stack_top@ha",
    "addi 1, 1, __stack_top@l",
    "li 0, 0",
    "stwu 0, -16(1)",
    "b probe_main",
);

/// 0 = all passed; bit `i` = check `i` failed; `0xFFFF_FFFF` = not finished.
#[unsafe(no_mangle)]
pub static mut PS_PROBE_RESULT: u32 = 0xFFFF_FFFF;

/// Dolphin intercepts a function with this symbol name and prints `r3` to its
/// log (OSReport HLE). On hardware it just returns.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn OSReport(msg: *const u8) {
    // Keep the call and the argument alive.
    unsafe { asm!("/* {0} */", in(reg) msg, options(nomem, nostack)) };
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { PS_PROBE_RESULT = 0xDEAD_0000 };
    OSReport(c"miraculix ps-probe: PANIC\n".as_ptr().cast());
    loop {}
}

static mut FAIL: u32 = 0;

fn check(id: u32, ok: bool) {
    if !ok {
        unsafe { FAIL |= 1 << id }
    }
}

/// Quarters in -8..=8: every product and sum below is exact in f32.
fn grid<const N: usize>(seed: u32) -> [f32; N] {
    let mut s = seed | 1;
    core::array::from_fn(|_| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s % 65) as f32 - 32.0) * 0.25
    })
}

/// Supervisor setup the SDK would normally do: MSR[FP], HID2[LSQE|PSE].
fn enable_paired_singles() {
    unsafe {
        asm!(
            "mfmsr {t}",
            "ori {t}, {t}, 0x2000",
            "mtmsr {t}",
            "isync",
            "mfspr {t}, 920",
            "oris {t}, {t}, 0xa000",
            "mtspr 920, {t}",
            "isync",
            t = out(reg) _,
            options(nostack)
        );
    }
}

const N: usize = 67; // not a multiple of 8 or 16: head, body and tail all run

#[unsafe(no_mangle)]
pub extern "C" fn probe_main() -> ! {
    enable_paired_singles();
    let f = detect_features();
    check(0, f.contains(Feature::PairedSingle));
    check(1, !f.contains(Feature::Altivec));
    check(
        2,
        matches!(
            PowerClass::detect(),
            PowerClass::Console(_)
        ) && PowerCore::detect().has_paired_single(),
    );
    let Some(t) = PairedSingle::detect() else {
        report();
    };

    let a: [f32; N + 2] = grid(1);
    let b: [f32; N + 2] = grid(2);
    let c: [f32; N + 2] = grid(3);
    for (k, (oa, ob, oo)) in [(0, 0, 0), (1, 0, 1), (0, 1, 0), (2, 1, 1)].into_iter().enumerate() {
        let (a, b, c) = (&a[oa..oa + N], &b[ob..ob + N], &c[oo..oo + N]);
        let mut buf = [f32::NAN; N + 2];
        let out = &mut buf[oo..oo + N];
        let base = 3 + 5 * k as u32;
        t.add_f32(a, b, out);
        check(base, (0..N).all(|i| out[i] == a[i] + b[i]));
        t.sub_f32(a, b, out);
        check(base + 1, (0..N).all(|i| out[i] == a[i] - b[i]));
        t.mul_f32(a, b, out);
        check(base + 2, (0..N).all(|i| out[i] == a[i] * b[i]));
        t.fmadd_f32(a, b, c, out);
        check(base + 3, (0..N).all(|i| out[i] == a[i] * b[i] + c[i]));
        let dot: f32 = (0..N).map(|i| a[i] * b[i]).sum();
        let sum: f32 = a.iter().sum();
        check(base + 4, t.dot_f32(a, b) == dot && t.reduce_sum_f32(a) == sum);
    }

    let q: [u16; N + 1] = core::array::from_fn(|i| (i as u16).wrapping_mul(40503));
    let mut out = [0f32; N + 1];
    for (k, (oq, oo)) in [(0, 0), (1, 1), (0, 1)].into_iter().enumerate() {
        let (q, out) = (&q[oq..oq + N], &mut out[oo..oo + N]);
        t.affine_u16_f32(q, 0.5, -3.0, out);
        check(23 + k as u32, (0..N).all(|i| out[i].to_bits() == (0.5 * q[i] as f32 - 3.0).to_bits()));
    }

    // auto picks the same kernel here.
    let mut o = [0f32; N];
    auto::add_f32(&a[..N], &b[..N], &mut o);
    check(26, (0..N).all(|i| o[i] == a[i] + b[i]));
    check(27, auto::dot_f32(&a[..N], &b[..N]) == t.dot_f32(&a[..N], &b[..N]));
    report();
}

fn report() -> ! {
    let fail = unsafe { FAIL };
    unsafe { PS_PROBE_RESULT = fail };
    if fail == 0 {
        OSReport(c"miraculix ps-probe: PASS\n".as_ptr().cast());
    } else {
        let mut msg = *b"miraculix ps-probe: FAIL 00000000\n\0";
        for i in 0..8 {
            let nib = (fail >> (28 - 4 * i)) & 0xf;
            msg[25 + i] = b"0123456789abcdef"[nib as usize];
        }
        OSReport(msg.as_ptr());
    }
    loop {}
}
