//! Tier comparison: `auto_up` vs. the explicit tokens, per working-set size.
//!
//! Run with `cargo run --release --example bench_tiers`. Tiers the CPU lacks
//! are skipped, so on a CPU without AVX-512 only `auto_up` and `Avx2` appear.
//! The question this answers: does the 512-bit path that `auto_up` always
//! takes with AVX-512F beat the plain `Avx2` token (Skylake-SP / Cascade
//! Lake downclock, Zen 4 double-pumped)? Run it on such a CPU and compare
//! the `avx512f` and `avx2` columns.
#[cfg(target_arch = "x86_64")]
fn main() {
    use core::arch::x86_64::*;
    use miraculix::x86::auto_up;
    use miraculix::x86::ops::avx::avx2::Avx2;
    use miraculix::x86::ops::avx512::avx512f::Avx512f;
    use std::hint::black_box;
    use std::time::Instant;

    // Same loop shape on both widths, so only the vector width differs.
    miraculix::avx2_fn! {
        fn add_avx2(_t: Avx2, a: &[i32], b: &[i32], out: &mut [i32]) {
            let n = a.len() / 8 * 8;
            let mut i = 0;
            while i < n {
                unsafe {
                    let x = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                    let y = _mm256_loadu_si256(b.as_ptr().add(i).cast());
                    _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_add_epi32(x, y));
                }
                i += 8;
            }
            for j in n..a.len() {
                out[j] = a[j].wrapping_add(b[j]);
            }
        }
    }
    miraculix::avx512f_fn! {
        fn add_avx512(_t: Avx512f, a: &[i32], b: &[i32], out: &mut [i32]) {
            let n = a.len() / 16 * 16;
            let mut i = 0;
            while i < n {
                unsafe {
                    let x = _mm512_loadu_si512(a.as_ptr().add(i).cast());
                    let y = _mm512_loadu_si512(b.as_ptr().add(i).cast());
                    _mm512_storeu_si512(out.as_mut_ptr().add(i).cast(), _mm512_add_epi32(x, y));
                }
                i += 16;
            }
            for j in n..a.len() {
                out[j] = a[j].wrapping_add(b[j]);
            }
        }
    }
    let avx2 = Avx2::detect();
    let avx512 = Avx512f::detect();
    println!("avx2: {}  avx512f: {}", avx2.is_some(), avx512.is_some());

    // GiB/s of traffic (2 reads + 1 write), best of 7 runs of ~64 MiB each.
    fn gibs(n: usize, bytes_per_elem: usize, mut f: impl FnMut()) -> f64 {
        let reps = ((1usize << 26) / (n * bytes_per_elem)).max(1);
        let mut best = f64::MAX;
        for _ in 0..7 {
            let s = Instant::now();
            for _ in 0..reps {
                f();
            }
            best = best.min(s.elapsed().as_secs_f64());
        }
        (n * bytes_per_elem * 3 * reps) as f64 / best / (1u64 << 30) as f64
    }

    println!(
        "{:>10} {:>10} {:>10} {:>10}   (GiB/s, add_i32)",
        "elements", "auto_up", "avx2", "avx512f"
    );
    for shift in [10, 13, 16, 19, 22, 25] {
        let n = 1usize << shift;
        let a = vec![3i32; n];
        let b = vec![5i32; n];
        let mut o = vec![0i32; n];
        let auto = gibs(n, 4, || {
            auto_up::add_i32(black_box(&a), black_box(&b), &mut o);
            black_box(&o);
        });
        let t2 = avx2.map(|t| {
            gibs(n, 4, || {
                add_avx2(t, black_box(&a), black_box(&b), &mut o);
                black_box(&o);
            })
        });
        let t5 = avx512.map(|t| {
            gibs(n, 4, || {
                add_avx512(t, black_box(&a), black_box(&b), &mut o);
                black_box(&o);
            })
        });
        let col = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.1}"));
        println!("{n:>10} {auto:>10.1} {:>10} {:>10}", col(t2), col(t5));
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64 only");
}
