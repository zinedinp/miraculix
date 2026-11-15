#![no_std]
#![no_main]
use core::arch::asm;
use miraculix::formats::{MiniFloat, Overflow};
use miraculix::riscv::ops::zvfbf::{Zvfbfmin, Zvfbfwma};
use miraculix::riscv::ops::zvfh::Zvfh;

fn exit(code: u64) -> ! {
    unsafe { asm!("ecall", in("a7") 93, in("a0") code, options(noreturn)) }
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    exit(99)
}
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
// Stand-in for libc's getauxval: HWCAP = M A F D C V.
#[unsafe(no_mangle)]
pub extern "C" fn getauxval(t: u64) -> u64 {
    if t == 16 {
        (1 << (b'M' - b'A')) | 1 | (1 << (b'F' - b'A')) | (1 << (b'D' - b'A')) | (1 << (b'C' - b'A')) | (1 << (b'V' - b'A'))
    } else {
        0
    }
}

static mut FAIL: u32 = 0;
fn check(id: u32, ok: bool) {
    if !ok {
        unsafe { FAIL |= 1 << id }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let t = Zvfh::detect().unwrap_or_else(|| exit(98));
    // 37 elements: not a multiple of any vector length, exercises the strip-mine tail.
    let vals: [f32; 37] = core::array::from_fn(|i| ((i as f32 - 18.0) * 0.37) * ((i as f32) * 311.0 + 1.0));
    let mut hw = [0u16; 37];
    t.cvt_f32_to_f16_slice(&vals, &mut hw);
    check(0, (0..37).all(|i| hw[i] as u32 == MiniFloat::F16.encode(vals[i], Overflow::Special)));
    let mut back = [0f32; 37];
    t.cvt_f16_to_f32_slice(&hw, &mut back);
    check(1, (0..37).all(|i| back[i].to_bits() == MiniFloat::F16.decode(hw[i] as u32).to_bits()));
    let big = [1e9f32, -1e9, 65504.0, 65520.0, 1e-8, 6e-8, 0.0, -0.0];
    let mut o = [0u16; 8];
    t.cvt_f32_to_f16_slice(&big, &mut o);
    check(2, (0..8).all(|i| o[i] as u32 == MiniFloat::F16.encode(big[i], Overflow::Special)));
    // every code through the public formats API (dispatches to Zvfh)
    let codes: [u16; 64] = core::array::from_fn(|i| (i as u16).wrapping_mul(1021));
    let mut dec = [0f32; 64];
    MiniFloat::F16.decode_u16_slice(&codes, &mut dec);
    check(3, (0..64).all(|i| {
        let w = MiniFloat::F16.decode(codes[i] as u32);
        dec[i].to_bits() == w.to_bits() || (dec[i].is_nan() && w.is_nan())
    }));
    // BF16 converts (subnormals and overflow included; NaN payloads excluded)
    let bt = Zvfbfmin::detect().unwrap_or_else(|| exit(97));
    let bv: [f32; 37] = core::array::from_fn(|i| ((i as f32 - 18.0) * 0.37) * ((i as f32) * 3.1e35 + 1.0));
    let mut bh = [0u16; 37];
    bt.cvt_f32_to_bf16_slice(&bv, &mut bh);
    check(4, (0..37).all(|i| bh[i] as u32 == MiniFloat::Bf16.encode(bv[i], Overflow::Special)));
    let sp = [1e-40f32, -1e-40, 1e-45, 3.0e38, 3.4e38, f32::MAX, 0.0, -0.0];
    let mut so = [0u16; 8];
    bt.cvt_f32_to_bf16_slice(&sp, &mut so);
    check(5, (0..8).all(|i| so[i] as u32 == MiniFloat::Bf16.encode(sp[i], Overflow::Special)));
    let bc: [u16; 50] = core::array::from_fn(|i| (i as u16).wrapping_mul(1311));
    let mut bd = [0f32; 50];
    bt.cvt_bf16_to_f32_slice(&bc, &mut bd);
    check(6, (0..50).all(|i| {
        let w = MiniFloat::Bf16.decode(bc[i] as u32);
        bd[i].to_bits() == w.to_bits() || (bd[i].is_nan() && w.is_nan())
    }));
    // public API dispatches to the same hardware
    let mut api = [0u16; 37];
    MiniFloat::Bf16.encode_u16_slice(&bv, &mut api, Overflow::Special);
    check(7, api == bh);
    // widening FMA: acc + a*b, product exact in f32
    let fw = Zvfbfwma::detect().unwrap_or_else(|| exit(96));
    let acc: [f32; 37] = core::array::from_fn(|i| i as f32 * 0.5 - 3.0);
    let ea: [u16; 37] = core::array::from_fn(|i| bh[(i * 7) % 37]);
    let eb: [u16; 37] = core::array::from_fn(|i| bh[(i * 11 + 3) % 37]);
    let mut fo = [0f32; 37];
    fw.fmacc_bf16_f32_slice(&acc, &ea, &eb, &mut fo);
    check(8, (0..37).all(|i| {
        let w = acc[i] + MiniFloat::Bf16.decode(ea[i] as u32) * MiniFloat::Bf16.decode(eb[i] as u32);
        fo[i].to_bits() == w.to_bits() || (fo[i].is_nan() && w.is_nan())
    }));
    exit(unsafe { FAIL } as u64)
}

// No libc: byte loops with volatile accesses so they never lower to memcpy/memset themselves.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(d: *mut u8, s: *const u8, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe { d.add(i).write_volatile(s.add(i).read_volatile()) };
    }
    d
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(d: *mut u8, c: i32, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe { d.add(i).write_volatile(c as u8) };
    }
    d
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    for i in 0..n {
        let (x, y) = unsafe { (a.add(i).read_volatile(), b.add(i).read_volatile()) };
        if x != y {
            return x as i32 - y as i32;
        }
    }
    0
}
