#![no_std]
#![no_main]
use core::arch::asm;
use miraculix::aarch64::ops::neon::neon::Neon;

fn exit(code: i32) -> ! {
    unsafe { asm!("svc 0", in("x8") 93, in("x0") code, options(noreturn)) }
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    exit(99)
}

// Stand-in for libc's getauxval: HWCAP = FP|ASIMD so Neon::detect() succeeds.
#[unsafe(no_mangle)]
pub extern "C" fn getauxval(t: u64) -> u64 {
    if t == 16 { 3 } else { 0 }
}

static mut FAIL: u32 = 0;
fn check(id: u32, ok: bool) {
    if !ok {
        unsafe { FAIL |= 1 << id }
    }
}

fn bytes<const N: usize>(seed: u8) -> [u8; N] {
    core::array::from_fn(|i| seed.wrapping_add(i as u8).wrapping_mul(37) ^ 0x5A)
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let t = Neon::detect().unwrap_or_else(|| exit(98));
    check(0, 1u32.to_ne_bytes() == [0, 0, 0, 1]);

    let a: [i16; 8] = [100, -200, 300, -32768, 32767, 5, -7, 9];
    let b: [i16; 8] = [11, 22, -33, 44, 55, -66, 77, -88];
    let (mut o1, mut o2) = ([0i16; 6], [0i16; 6]);
    t.qasx_i16_slice(&a[..6], &b[..6], &mut o1);
    check(1, t.qasx_i16x8(a, b)[..6] == o1);
    t.qsax_i16_slice(&a[..6], &b[..6], &mut o2);
    check(2, t.qsax_i16x8(a, b)[..6] == o2);

    let x = [1u32, 0xFFFF_FFFF, 7, 0x8000_0000];
    check(3, t.add_u32x4(x, x) == x.map(|v| v.wrapping_add(v)));
    check(4, t.popcnt_u16x8([0xFFFF, 1, 0x8001, 0, 3, 7, 0xF0F0, 0x1234]) == [16, 1, 2, 0, 2, 3, 8, 5]);
    check(5, t.popcnt_u64x2([0x8000_0000_0000_0001, u64::MAX]) == [2, 64]);

    let f = [1.5f32, -2.25, 3.0e10, 0.1];
    check(6, t.cvt_f32x4_to_f64x2(f) == [1.5, -2.25]);
    check(7, t.cvt_f32x4_to_f64x4(f) == f.map(f64::from));
    let d = [0.1f64, -3.0, 1e300, 7.7];
    check(8, t.cvt_f64x4_to_f32x4(d) == d.map(|v| v as f32));
    let (p, q) = (bytes::<16>(1), bytes::<16>(99));
    check(9, t.zip1_u8x16(p, q) == core::array::from_fn(|i| if i % 2 == 0 { p[i / 2] } else { q[i / 2] }));
    check(10, t.zip2_u8x16(p, q) == core::array::from_fn(|i| if i % 2 == 0 { p[8 + i / 2] } else { q[8 + i / 2] }));
    let m: [i8; 16] = core::array::from_fn(|i| p[i] as i8);
    let mut want = 0u16;
    for (i, v) in m.iter().enumerate() {
        if *v < 0 {
            want |= 1 << i;
        }
    }
    check(11, t.movemask_i8x16(m) == want);
    let mut table = [0u8; 256];
    for (i, v) in table.iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(7).wrapping_add(3);
    }
    let idx = bytes::<16>(5);
    check(12, t.lookup_u8x256(&table, idx) == idx.map(|i| table[i as usize]));
    let small: [u8; 16] = bytes(3);
    check(13, t.tbl_u8x16(small, [0, 15, 16, 255, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])[..4] == [small[0], small[15], 0, 0]);
    check(14, t.sum_u8x8([1, 2, 3, 4, 5, 6, 7, 8]) == 36);

    exit(match option_env!("HI") { Some(_) => (unsafe { FAIL } >> 8) as i32, None => (unsafe { FAIL } & 0xff) as i32 })
}
