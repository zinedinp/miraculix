#![no_std]
#![no_main]
use core::arch::asm;
use miraculix::formats::{MiniFloat, Overflow};
use miraculix::riscv::ops::rvv::Rvv;
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

static mut FAIL: u64 = 0;
fn check(id: u32, ok: bool) {
    if !ok {
        unsafe { FAIL |= 1u64 << id }
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
    rvv_checks();
    extra_checks();
    // exit status is 8 bits: report the first failing check id + 1 (0 = all passed)
    let f = unsafe { FAIL };
    exit(if f == 0 { 0 } else { f.trailing_zeros() as u64 + 1 })
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

fn rvv_checks() {
    let t = Rvv::detect().unwrap_or_else(|| exit(95));
    const N: usize = 301; // not a multiple of any vector length
    let mut r = 0x1234_5678_9abc_def1u64;
    let mut nx = move || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    // small integers: sums are exact whatever the reduction order
    let a: [f32; N] = core::array::from_fn(|i| if i % 29 == 28 { f32::NAN } else { (nx() % 41) as f32 - 20.0 });
    let b: [f32; N] = core::array::from_fn(|_| (nx() % 17) as f32 - 8.0);
    let c: [f32; N] = core::array::from_fn(|_| (nx() % 9) as f32);
    let an: [f32; N] = core::array::from_fn(|i| if a[i].is_nan() { 1.0 } else { a[i] });
    let mut o = [0f32; N];
    t.add_f32(&an, &b, &mut o);
    check(9, (0..N).all(|i| o[i] == an[i] + b[i]));
    t.sub_f32(&an, &b, &mut o);
    check(10, (0..N).all(|i| o[i] == an[i] - b[i]));
    t.mul_f32(&an, &b, &mut o);
    check(11, (0..N).all(|i| o[i] == an[i] * b[i]));
    t.fmadd_f32(&an, &b, &c, &mut o);
    check(12, (0..N).all(|i| o[i] == an[i] * b[i] + c[i]));
    let (ia, ib): ([i32; N], [i32; N]) = (core::array::from_fn(|i| i as i32 * 1_000_003), core::array::from_fn(|i| (i as i32) << 20));
    let mut io = [0i32; N];
    t.add_i32(&ia, &ib, &mut io);
    check(13, (0..N).all(|i| io[i] == ia[i].wrapping_add(ib[i])));
    let sum: f32 = an.iter().sum();
    check(14, t.reduce_sum_f32(&an) == sum);
    let want_max = a.iter().copied().filter(|x| !x.is_nan()).fold(f32::NEG_INFINITY, f32::max);
    let want_min = a.iter().copied().filter(|x| !x.is_nan()).fold(f32::INFINITY, f32::min);
    check(15, t.reduce_max_f32(&a) == want_max && t.reduce_min_f32(&a) == want_min);
    check(16, t.reduce_max_f32(&[f32::NAN; 40]) == f32::NEG_INFINITY && t.reduce_max_f32(&[]) == f32::NEG_INFINITY);
    let first = a.iter().position(|&x| x == want_max).unwrap();
    check(17, t.argmax_f32(&a) == Some((first, want_max)) && t.argmax_f32(&[f32::NAN; 9]).is_none());
    let mut late = [0f32; N];
    late[N - 1] = 5.0;
    check(18, t.argmax_f32(&late) == Some((N - 1, 5.0)));
    let mut idx = [0u32; N];
    let n = t.compress_gt_f32(&a, 10.0, &mut idx);
    let mut want = [0u32; N];
    let mut wn = 0;
    for i in 0..N {
        if a[i] > 10.0 {
            want[wn] = i as u32;
            wn += 1;
        }
    }
    check(19, n == wn && idx[..n] == want[..wn]);
    check(20, t.compress_gt_f32(&a, 1e9, &mut idx) == 0);
    // top-k against a straightforward selection
    let (mut ti, mut tv) = ([0u32; 5], [0f32; 5]);
    let got = t.topk_f32(&a, &mut ti, &mut tv);
    let mut used = [false; N];
    let mut ok = got == 5;
    for j in 0..5 {
        let mut best: Option<usize> = None;
        for i in 0..N {
            if !used[i] && !a[i].is_nan() && best.map_or(true, |b| a[i] > a[b]) {
                best = Some(i);
            }
        }
        let b = best.unwrap();
        used[b] = true;
        ok &= ti[j] as usize == b && tv[j] == a[b];
    }
    check(21, ok);
    let (x, y): ([i8; N], [i8; N]) = (core::array::from_fn(|_| nx() as i8), core::array::from_fn(|_| nx() as i8));
    let want: i32 = (0..N).map(|i| x[i] as i32 * y[i] as i32).sum();
    check(22, t.dot_i8_i32(&x, &y) == want && t.dot_i8_i32(&[], &[]) == 0);
}

fn extra_checks() {
    use miraculix::formats::bfp::{BfpElem, bfp_encode_rows};
    use miraculix::riscv::{auto, class, class::{CoreId, RiscvClass}, Feature, FeatureSet};
    // auto dispatch agrees with the token on the same data
    let a: [f32; 77] = core::array::from_fn(|i| if i == 40 { f32::NAN } else { ((i * 37 % 23) as f32) - 11.0 });
    check(23, auto::reduce_max_f32(&a) == 11.0 && auto::reduce_min_f32(&a) == -11.0);
    check(24, auto::argmax_f32(&a) == Some((a.iter().position(|&x| x == 11.0).unwrap(), 11.0)));
    let (mut idx, mut val) = ([0u32; 3], [0f32; 3]);
    check(25, auto::topk_f32(&a, &mut idx, &mut val) == 3 && val == [11.0, 11.0, 11.0]);
    check(26, auto::dot_i8_i32(&[1, -2, 3], &[4, 5, -6]) == -24);
    // class: pure classification + live VLEN
    let v = FeatureSet::default().with(Feature::V);
    let x280 = CoreId { vendor: class::VENDOR_SIFIVE, arch: 0, imp: 0 };
    check(27, RiscvClass::from_parts(v, x280, Some(512)) == RiscvClass::VectorAccelerator);
    check(28, RiscvClass::from_parts(v, x280, Some(128)) == RiscvClass::Application);
    check(29, RiscvClass::from_parts(v, CoreId::default(), Some(512)) == RiscvClass::Application);
    check(30, RiscvClass::from_parts(FeatureSet::default(), x280, Some(512)) == RiscvClass::Application);
    check(31, matches!(class::vlen_bits(), Some(b) if b >= 128 && b.is_power_of_two()));
    // Tenstorrent BFP_B against the hash of tt-metal's own row_scalar (scripts/tt-oracle)
    for (elem, want) in [
        (BfpElem::M7, 0xb1fe9e3c9d5acc27u64),
        (BfpElem::M3, 0xae31e6784c4da60c),
        (BfpElem::M1, 0x71fe9f9c13841376),
    ] {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for row in 0..5000u32 {
            let src: [f32; 16] = core::array::from_fn(|_| f32::from_bits(bfp_word(&mut st, row)));
            let (mut e, mut d) = ([0u8; 1], [0u8; 16]);
            let rb = elem.row_bytes();
            bfp_encode_rows(elem, &src, &mut e, &mut d[..rb]);
            for &b in core::iter::once(&e[0]).chain(&d[..rb]) {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        check(32 + (7 - elem.mantissa_bits()) / 2, h == want);
    }
}

fn bfp_word(st: &mut u64, row: u32) -> u32 {
    *st ^= *st << 13;
    *st ^= *st >> 7;
    *st ^= *st << 17;
    let u = *st;
    let kind = u & 7;
    if kind == 0 {
        return (u >> 40) as u32 & 0x8000_0000;
    }
    if kind == 1 {
        return (u >> 8) as u32 & 0x807F_FFFF;
    }
    if kind == 2 && row % 97 == 0 {
        return (u >> 8) as u32 | 0x7F80_0000;
    }
    let base = 120 + row % 20;
    let e = base + ((u >> 40) % 10) as u32 - 5;
    (((u >> 63) as u32) << 31) | (e << 23) | ((u >> 8) as u32 & 0x7F_FFFF)
}
