// Comprehensive feComposite Arithmetic benchmark
// Tests naive vs optimized (production) paths across all parameter combinations.
//
// We reimplement both paths here identically to composite.rs so that we can
// call them independently from outside the private module.
//
// Uses std::thread::scope for parallel execution across CPU cores.

use rgb::RGBA8;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Helpers copied from the filter module
// ---------------------------------------------------------------------------

#[inline]
fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if val < min {
        min
    } else {
        val
    }
}

/// Approximate zero check matching usvg::ApproxZeroUlps with 4 ULPs.
#[inline]
fn approx_zero(v: f32) -> bool {
    v.abs() <= 4.0 * f32::EPSILON || v == 0.0
}

// ---------------------------------------------------------------------------
// Naive implementation (verbatim from composite.rs)
// ---------------------------------------------------------------------------

fn arithmetic_naive(k1: f32, k2: f32, k3: f32, k4: f32, src1: &[RGBA8], src2: &[RGBA8], dest: &mut [RGBA8]) {
    let calc = |i1: u8, i2: u8, max: f32| -> f32 {
        let i1 = i1 as f32 / 255.0;
        let i2 = i2 as f32 / 255.0;
        let result = k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4;
        f32_bound(0.0, result, max)
    };

    let mut i = 0;
    for (c1, c2) in src1.iter().zip(src2.iter()) {
        let a = calc(c1.a, c2.a, 1.0);
        if approx_zero(a) {
            i += 1;
            continue;
        }

        let r = (calc(c1.r, c2.r, a) * 255.0) as u8;
        let g = (calc(c1.g, c2.g, a) * 255.0) as u8;
        let b = (calc(c1.b, c2.b, a) * 255.0) as u8;
        let a = (a * 255.0) as u8;

        dest[i] = RGBA8 { r, g, b, a };
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Optimized implementation (mirrors composite.rs with all fixes)
// ---------------------------------------------------------------------------

const BATCH: usize = 64;

#[inline(always)]
fn clamp_and_scale(val: f32, max: f32) -> u8 {
    (val.clamp(0.0, max) * 255.0) as u8
}

fn arithmetic_optimized(k1: f32, k2: f32, k3: f32, k4: f32, src1: &[RGBA8], src2: &[RGBA8], dest: &mut [RGBA8]) {
    let len = src1.len();
    let mut offset = 0;
    const INV_255: f32 = 1.0 / 255.0;

    let mut r1 = [0.0f32; BATCH];
    let mut g1 = [0.0f32; BATCH];
    let mut b1 = [0.0f32; BATCH];
    let mut a1 = [0.0f32; BATCH];
    let mut r2 = [0.0f32; BATCH];
    let mut g2 = [0.0f32; BATCH];
    let mut b2 = [0.0f32; BATCH];
    let mut a2 = [0.0f32; BATCH];

    let mut res_r = [0.0f32; BATCH];
    let mut res_g = [0.0f32; BATCH];
    let mut res_b = [0.0f32; BATCH];
    let mut res_a = [0.0f32; BATCH];

    let can_skip_transparent = k4 <= 0.0 || approx_zero(k4);

    while offset < len {
        let batch_len = (len - offset).min(BATCH);

        let s1 = &src1[offset..offset + batch_len];
        let s2 = &src2[offset..offset + batch_len];

        if can_skip_transparent {
            let all_transparent =
                s1.iter().all(|p| p.a == 0) && s2.iter().all(|p| p.a == 0);
            if all_transparent {
                offset += batch_len;
                continue;
            }
        }

        for j in 0..batch_len {
            r1[j] = s1[j].r as f32 * INV_255;
            g1[j] = s1[j].g as f32 * INV_255;
            b1[j] = s1[j].b as f32 * INV_255;
            a1[j] = s1[j].a as f32 * INV_255;

            r2[j] = s2[j].r as f32 * INV_255;
            g2[j] = s2[j].g as f32 * INV_255;
            b2[j] = s2[j].b as f32 * INV_255;
            a2[j] = s2[j].a as f32 * INV_255;
        }

        for j in 0..batch_len {
            res_r[j] = k1 * r1[j] * r2[j] + k2 * r1[j] + k3 * r2[j] + k4;
        }
        for j in 0..batch_len {
            res_g[j] = k1 * g1[j] * g2[j] + k2 * g1[j] + k3 * g2[j] + k4;
        }
        for j in 0..batch_len {
            res_b[j] = k1 * b1[j] * b2[j] + k2 * b1[j] + k3 * b2[j] + k4;
        }
        for j in 0..batch_len {
            res_a[j] = k1 * a1[j] * a2[j] + k2 * a1[j] + k3 * a2[j] + k4;
        }

        for j in 0..batch_len {
            res_a[j] = res_a[j].clamp(0.0, 1.0);
        }

        let all_zero = res_a[..batch_len].iter().all(|&a| approx_zero(a));
        if all_zero {
            offset += batch_len;
            continue;
        }

        let dest_slice = &mut dest[offset..offset + batch_len];
        for j in 0..batch_len {
            let a = res_a[j];
            if approx_zero(a) {
                continue;
            }

            let r = clamp_and_scale(res_r[j], a);
            let g = clamp_and_scale(res_g[j], a);
            let b = clamp_and_scale(res_b[j], a);
            let a = (a * 255.0) as u8;

            dest_slice[j] = RGBA8 { r, g, b, a };
        }

        offset += batch_len;
    }
}

// ---------------------------------------------------------------------------
// Production path (mirrors composite.rs switching logic)
// ---------------------------------------------------------------------------

const OPTIMIZED_CROSSOVER: usize = 64;

fn arithmetic_production(k1: f32, k2: f32, k3: f32, k4: f32, src1: &[RGBA8], src2: &[RGBA8], dest: &mut [RGBA8]) {
    if k1 == 0.0 && k2 == 0.0 && k3 == 0.0 && k4 == 0.0 {
        return;
    }

    let pixel_count = src1.len();
    if pixel_count < OPTIMIZED_CROSSOVER {
        arithmetic_naive(k1, k2, k3, k4, src1, src2, dest);
    } else {
        arithmetic_optimized(k1, k2, k3, k4, src1, src2, dest);
    }
}

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

fn generate_pixels(count: usize, pattern: &str, seed: u64) -> Vec<RGBA8> {
    let mut pixels = vec![RGBA8 { r: 0, g: 0, b: 0, a: 0 }; count];
    let mut rng = seed;
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    match pattern {
        "opaque" => {
            for p in pixels.iter_mut() {
                p.a = 255;
                p.r = (next() % 256) as u8;
                p.g = (next() % 256) as u8;
                p.b = (next() % 256) as u8;
            }
        }
        "transparent" => {}
        "sparse50" => {
            for p in pixels.iter_mut() {
                if next() % 2 == 0 {
                    p.a = 255;
                    p.r = (next() % 256) as u8;
                    p.g = (next() % 256) as u8;
                    p.b = (next() % 256) as u8;
                }
            }
        }
        "gradient" => {
            for (i, p) in pixels.iter_mut().enumerate() {
                let alpha = ((i * 255) / count.max(1)) as u8;
                p.a = alpha;
                let f = alpha as f32 / 255.0;
                let base = (next() % 256) as u8;
                p.r = (base as f32 * f) as u8;
                p.g = (base as f32 * f) as u8;
                p.b = (base as f32 * f) as u8;
            }
        }
        _ => panic!("Unknown pattern: {}", pattern),
    }
    pixels
}

fn bench_fn<F: FnMut()>(mut f: F, pixel_count: usize) -> Duration {
    let iters = if pixel_count <= 64 {
        50_000
    } else if pixel_count <= 1024 {
        10_000
    } else if pixel_count <= 16384 {
        2_000
    } else if pixel_count <= 262144 {
        500
    } else {
        100
    };

    for _ in 0..iters / 5 {
        f();
    }

    let num_samples = 5;
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        samples.push(elapsed / iters as u32);
    }

    samples.sort();
    samples[num_samples / 2]
}

// ---------------------------------------------------------------------------
// Configuration and result types for parallel execution
// ---------------------------------------------------------------------------

struct Config {
    size_label: &'static str,
    w: u32,
    h: u32,
    k_label: &'static str,
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    pattern: &'static str,
}

struct BenchResult {
    order: usize,
    size: &'static str,
    k_label: &'static str,
    pattern: &'static str,
    naive_ns: u128,
    prod_ns: u128,
    speedup: f64,
    regression: bool,
}

/// Run a single benchmark configuration, returning a BenchResult.
fn run_config(config: &Config, order: usize, progress: &AtomicUsize, total: usize) -> BenchResult {
    let pixel_count = (config.w as usize) * (config.h as usize);

    let src1 = generate_pixels(pixel_count, config.pattern, 12345);
    let src2 = generate_pixels(pixel_count, config.pattern, 67890);

    let src1_n = src1.clone();
    let src2_n = src2.clone();
    let mut dest_n = vec![RGBA8 { r: 0, g: 0, b: 0, a: 0 }; pixel_count];
    let naive_dur = bench_fn(|| {
        for d in dest_n.iter_mut() {
            *d = RGBA8 { r: 0, g: 0, b: 0, a: 0 };
        }
        arithmetic_naive(config.k1, config.k2, config.k3, config.k4, &src1_n, &src2_n, &mut dest_n);
    }, pixel_count);

    let src1_p = src1.clone();
    let src2_p = src2.clone();
    let mut dest_p = vec![RGBA8 { r: 0, g: 0, b: 0, a: 0 }; pixel_count];
    let prod_dur = bench_fn(|| {
        for d in dest_p.iter_mut() {
            *d = RGBA8 { r: 0, g: 0, b: 0, a: 0 };
        }
        arithmetic_production(config.k1, config.k2, config.k3, config.k4, &src1_p, &src2_p, &mut dest_p);
    }, pixel_count);

    let naive_ns = naive_dur.as_nanos();
    let prod_ns = prod_dur.as_nanos();
    let speedup = if prod_ns > 0 {
        naive_ns as f64 / prod_ns as f64
    } else {
        f64::INFINITY
    };
    let regression = speedup < 0.95;

    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
    eprint!(
        "\r[{}/{}] {}  {}  {} ...           ",
        done, total, config.size_label, config.k_label, config.pattern
    );

    BenchResult {
        order,
        size: config.size_label,
        k_label: config.k_label,
        pattern: config.pattern,
        naive_ns,
        prod_ns,
        speedup,
        regression,
    }
}

fn main() {
    let sizes: Vec<(&'static str, u32, u32)> = vec![
        ("4x4", 4, 4),
        ("16x16", 16, 16),
        ("32x32", 32, 32),
        ("64x64", 64, 64),
        ("128x128", 128, 128),
        ("256x256", 256, 256),
        ("512x512", 512, 512),
        ("1024x1024", 1024, 1024),
        ("2048x2048", 2048, 2048),
    ];

    let k_values: Vec<(&'static str, f32, f32, f32, f32)> = vec![
        ("k=(0,0,0,0)", 0.0, 0.0, 0.0, 0.0),
        ("k=(0,1,0,0)", 0.0, 1.0, 0.0, 0.0),
        ("k=(0,0,1,0)", 0.0, 0.0, 1.0, 0.0),
        ("k=(1,0,0,0)", 1.0, 0.0, 0.0, 0.0),
        ("k=(0.5,0.5,0.5,0)", 0.5, 0.5, 0.5, 0.0),
        ("k=(1,1,1,-0.5)", 1.0, 1.0, 1.0, -0.5),
        ("k=(0,0,0,1)", 0.0, 0.0, 0.0, 1.0),
        ("k=(1,0,0,0.5)", 1.0, 0.0, 0.0, 0.5),
        ("k=(0,1,0,0.5)", 0.0, 1.0, 0.0, 0.5),
    ];

    let patterns: [&'static str; 4] = ["opaque", "transparent", "sparse50", "gradient"];

    // Build all configurations upfront
    let mut configs: Vec<Config> = Vec::with_capacity(sizes.len() * k_values.len() * patterns.len());
    for &(size_label, w, h) in &sizes {
        for &(k_label, k1, k2, k3, k4) in &k_values {
            for &pattern in &patterns {
                configs.push(Config {
                    size_label,
                    w,
                    h,
                    k_label,
                    k1,
                    k2,
                    k3,
                    k4,
                    pattern,
                });
            }
        }
    }

    let total = configs.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    eprintln!("Running {} benchmark configurations on {} threads...", total, num_threads);
    eprintln!("(This may take a few minutes)\n");

    let progress = AtomicUsize::new(0);

    // Split configs into chunks and run in parallel using scoped threads
    let chunk_size = (total + num_threads - 1) / num_threads;
    let config_chunks: Vec<&[Config]> = configs.chunks(chunk_size).collect();

    let mut results: Vec<BenchResult> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(config_chunks.len());

        for (chunk_idx, chunk) in config_chunks.iter().enumerate() {
            let progress_ref = &progress;
            handles.push(s.spawn(move || {
                let mut chunk_results = Vec::with_capacity(chunk.len());
                for (i, config) in chunk.iter().enumerate() {
                    let order = chunk_idx * chunk_size + i;
                    chunk_results.push(run_config(config, order, progress_ref, total));
                }
                chunk_results
            }));
        }

        let mut all_results = Vec::with_capacity(total);
        for handle in handles {
            all_results.extend(handle.join().unwrap());
        }
        all_results
    });

    // Sort by original order to preserve deterministic output
    results.sort_by_key(|r| r.order);

    eprintln!("\r                                                                              ");
    eprintln!("Done.\n");

    println!("{:-<130}", "");
    println!(
        "{:<12} | {:<22} | {:<14} | {:>12} | {:>12} | {:>8} | {}",
        "Image Size", "K-values", "Input Pattern", "Naive (us)", "Prod (us)", "Speedup", "Status"
    );
    println!("{:-<130}", "");

    let mut regression_count = 0;
    let mut prev_size = "";

    for r in &results {
        if !prev_size.is_empty() && r.size != prev_size {
            println!("{:-<130}", "");
        }
        prev_size = r.size;

        let naive_us = r.naive_ns as f64 / 1000.0;
        let prod_us = r.prod_ns as f64 / 1000.0;
        let status = if r.regression {
            regression_count += 1;
            "REGRESSION"
        } else if r.speedup > 1.05 {
            "FASTER"
        } else {
            "OK"
        };

        println!(
            "{:<12} | {:<22} | {:<14} | {:>12.2} | {:>12.2} | {:>7.2}x | {}",
            r.size, r.k_label, r.pattern, naive_us, prod_us, r.speedup, status
        );
    }

    println!("{:-<130}", "");

    println!("\n=== SUMMARY ===");
    println!("Total configurations tested: {}", results.len());
    println!("Regressions (prod >5% slower than naive): {}", regression_count);

    if regression_count > 0 {
        println!("\n=== REGRESSIONS DETAIL ===");
        println!(
            "{:<12} | {:<22} | {:<14} | {:>12} | {:>12} | {:>8}",
            "Image Size", "K-values", "Input Pattern", "Naive (us)", "Prod (us)", "Slowdown"
        );
        for r in &results {
            if r.regression {
                let naive_us = r.naive_ns as f64 / 1000.0;
                let prod_us = r.prod_ns as f64 / 1000.0;
                let slowdown = prod_us / naive_us;
                println!(
                    "{:<12} | {:<22} | {:<14} | {:>12.2} | {:>12.2} | {:>7.2}x",
                    r.size, r.k_label, r.pattern, naive_us, prod_us, slowdown
                );
            }
        }
    }

    println!("\n=== AVERAGE SPEEDUP BY IMAGE SIZE ===");
    for &(size_label, _, _) in &sizes {
        let matching: Vec<&BenchResult> = results.iter().filter(|r| r.size == size_label).collect();
        if matching.is_empty() {
            continue;
        }
        let avg_speedup: f64 = matching.iter().map(|r| r.speedup).sum::<f64>() / matching.len() as f64;
        let min_speedup: f64 = matching.iter().map(|r| r.speedup).fold(f64::INFINITY, f64::min);
        let max_speedup: f64 = matching.iter().map(|r| r.speedup).fold(f64::NEG_INFINITY, f64::max);
        let reg_count = matching.iter().filter(|r| r.regression).count();
        println!(
            "  {:<12}  avg={:.2}x  min={:.2}x  max={:.2}x  regressions={}",
            size_label, avg_speedup, min_speedup, max_speedup, reg_count
        );
    }

    if regression_count > 0 {
        std::process::exit(1);
    }
}
