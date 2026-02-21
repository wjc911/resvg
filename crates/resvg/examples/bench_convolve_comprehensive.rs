// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive benchmark for `feConvolveMatrix` filter primitive.
//!
//! Tests ALL combinations of image sizes, kernel sizes, edge modes, preserve_alpha,
//! and special kernels. Compares `apply_naive` vs production `apply`, verifies
//! bit-exact output, and flags regressions where production is >5% slower.
//!
//! Uses multithreading (std::thread::scope) to run benchmark configurations in
//! parallel across all available CPU cores.
//!
//! Usage: cargo run --release --example bench_convolve_comprehensive

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use resvg::filter::convolve_matrix::{apply, apply_naive};
use resvg::filter::ImageRefMut;
use rgb::RGBA8;
use usvg::filter::{ConvolveMatrix, ConvolveMatrixData, EdgeMode};
use usvg::NonZeroF32;

/// Helper to create a ConvolveMatrix.
fn make_convolve_matrix(
    cols: u32,
    rows: u32,
    target_x: u32,
    target_y: u32,
    data: Vec<f32>,
    divisor: f32,
    bias: f32,
    edge_mode: EdgeMode,
    preserve_alpha: bool,
) -> ConvolveMatrix {
    ConvolveMatrix::new(
        usvg::filter::Input::SourceGraphic,
        ConvolveMatrixData::new(target_x, target_y, cols, rows, data).unwrap(),
        NonZeroF32::new(divisor).unwrap(),
        bias,
        edge_mode,
        preserve_alpha,
    )
}

/// Generate a test image with deterministic pixel values.
fn make_test_image(width: u32, height: u32, seed: u8) -> Vec<RGBA8> {
    let mut data = Vec::with_capacity((width * height) as usize);
    for i in 0..(width * height) {
        let v = ((i as u8).wrapping_mul(17).wrapping_add(seed)) as u8;
        data.push(RGBA8 {
            r: v,
            g: v.wrapping_add(50),
            b: v.wrapping_add(100),
            a: v.wrapping_add(150),
        });
    }
    data
}

/// Describes a kernel configuration for benchmarking.
#[derive(Clone)]
struct KernelConfig {
    name: &'static str,
    cols: u32,
    rows: u32,
    data: Vec<f32>,
    divisor: f32,
    bias: f32,
}

fn identity_kernel(cols: u32, rows: u32) -> KernelConfig {
    let mut data = vec![0.0f32; (cols * rows) as usize];
    let center = (rows / 2 * cols + cols / 2) as usize;
    data[center] = 1.0;
    KernelConfig {
        name: "identity",
        cols,
        rows,
        data,
        divisor: 1.0,
        bias: 0.0,
    }
}

fn uniform_kernel(cols: u32, rows: u32) -> KernelConfig {
    let n = (cols * rows) as usize;
    KernelConfig {
        name: "uniform",
        cols,
        rows,
        data: vec![1.0; n],
        divisor: n as f32,
        bias: 0.0,
    }
}

fn gaussian_3x3() -> KernelConfig {
    KernelConfig {
        name: "gaussian_3x3",
        cols: 3,
        rows: 3,
        data: vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        divisor: 16.0,
        bias: 0.0,
    }
}

fn gaussian_5x5() -> KernelConfig {
    KernelConfig {
        name: "gaussian_5x5",
        cols: 5,
        rows: 5,
        data: vec![
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, 36.0, 24.0, 6.0,
            4.0, 16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ],
        divisor: 256.0,
        bias: 0.0,
    }
}

fn sobel_x() -> KernelConfig {
    KernelConfig {
        name: "sobel_x",
        cols: 3,
        rows: 3,
        data: vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        divisor: 1.0,
        bias: 0.5,
    }
}

fn sharpen_3x3() -> KernelConfig {
    KernelConfig {
        name: "sharpen",
        cols: 3,
        rows: 3,
        data: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        divisor: 1.0,
        bias: 0.0,
    }
}

/// All parameters needed to execute a single benchmark configuration.
struct Config {
    order: usize,
    width: u32,
    height: u32,
    kernel: KernelConfig,
    edge_mode: EdgeMode,
    preserve_alpha: bool,
}

/// A single benchmark result row.
struct BenchResult {
    order: usize,
    image_size: String,
    kernel: String,
    edge_mode: String,
    preserve_alpha: bool,
    dispatch: &'static str,
    naive_us: f64,
    prod_us: f64,
    speedup: f64,
    bit_exact: bool,
    status: &'static str,
}

/// Determine which path `apply()` will dispatch to for given parameters.
fn dispatch_path(cols: u32, rows: u32, width: u32, height: u32) -> &'static str {
    if rows <= 1 && cols <= 1 {
        "naive(1x1)"
    } else if rows > height || cols > width {
        "naive(oversz)"
    } else {
        "general"
    }
}

/// Time a function by running it `iterations` times and returning microseconds per call.
fn bench_fn<F: FnMut()>(mut f: F, iterations: u32) -> f64 {
    // Warmup
    f();

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

/// Choose iterations to target roughly 50-200ms total time.
/// For very small images, use a high minimum to reduce noise.
fn pick_iterations(width: u32, height: u32, kernel_elements: u32) -> u32 {
    let pixels = width as u64 * height as u64;
    let ops = pixels * kernel_elements as u64;
    // Target ~100ms total
    let iters = (50_000_000u64 / ops.max(1)).max(1).min(5000);
    // Ensure a minimum of 200 iterations for tiny images to reduce noise
    let min_iters = if pixels <= 256 { 500 } else if pixels <= 4096 { 200 } else { 1 };
    iters.max(min_iters) as u32
}

fn edge_mode_str(em: EdgeMode) -> &'static str {
    match em {
        EdgeMode::None => "None",
        EdgeMode::Duplicate => "Duplicate",
        EdgeMode::Wrap => "Wrap",
    }
}

/// Execute a single benchmark configuration and return the result.
fn run_config(config: &Config) -> BenchResult {
    let kc = &config.kernel;
    let w = config.width;
    let h = config.height;
    let em = config.edge_mode;
    let pa = config.preserve_alpha;

    let target_x = kc.cols / 2;
    let target_y = kc.rows / 2;

    let matrix = make_convolve_matrix(
        kc.cols,
        kc.rows,
        target_x,
        target_y,
        kc.data.clone(),
        kc.divisor,
        kc.bias,
        em,
        pa,
    );

    let iterations = pick_iterations(w, h, kc.cols * kc.rows);

    // --- Naive timing ---
    let naive_us = {
        let img_template = make_test_image(w, h, 42);
        bench_fn(
            || {
                let mut img = img_template.clone();
                apply_naive(&matrix, ImageRefMut::new(w, h, &mut img));
            },
            iterations,
        )
    };

    // --- Production timing ---
    let prod_us = {
        let img_template = make_test_image(w, h, 42);
        bench_fn(
            || {
                let mut img = img_template.clone();
                apply(&matrix, ImageRefMut::new(w, h, &mut img));
            },
            iterations,
        )
    };

    // --- Correctness: bit-exact check ---
    let bit_exact = {
        let mut img_naive = make_test_image(w, h, 42);
        let mut img_prod = img_naive.clone();
        apply_naive(&matrix, ImageRefMut::new(w, h, &mut img_naive));
        apply(&matrix, ImageRefMut::new(w, h, &mut img_prod));
        img_naive == img_prod
    };

    let speedup = naive_us / prod_us;
    let status = if !bit_exact {
        "MISMATCH"
    } else if speedup < 0.95 {
        "REGRESSION"
    } else if speedup > 1.05 {
        "FASTER"
    } else {
        "SAME"
    };

    let kernel_str = if kc.name == "uniform" || kc.name == "identity" {
        format!("{}x{} {}", kc.cols, kc.rows, kc.name)
    } else {
        kc.name.to_string()
    };

    let dp = dispatch_path(kc.cols, kc.rows, w, h);

    BenchResult {
        order: config.order,
        image_size: format!("{}x{}", w, h),
        kernel: kernel_str,
        edge_mode: edge_mode_str(em).to_string(),
        preserve_alpha: pa,
        dispatch: dp,
        naive_us,
        prod_us,
        speedup,
        bit_exact,
        status,
    }
}

fn main() {
    let image_sizes: Vec<(u32, u32)> = vec![
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];

    let edge_modes = [EdgeMode::None, EdgeMode::Duplicate, EdgeMode::Wrap];
    let preserve_alphas = [false, true];

    // Build all kernel configs upfront (no closures needed).
    let base_kernels: Vec<KernelConfig> = vec![
        // Standard sizes
        uniform_kernel(1, 1),
        uniform_kernel(2, 2),
        uniform_kernel(3, 3),
        uniform_kernel(5, 5),
        uniform_kernel(7, 7),
        uniform_kernel(9, 9),
        // Asymmetric
        KernelConfig {
            name: "uniform_3x5",
            cols: 3,
            rows: 5,
            data: vec![1.0; 15],
            divisor: 15.0,
            bias: 0.0,
        },
        KernelConfig {
            name: "uniform_5x3",
            cols: 5,
            rows: 3,
            data: vec![1.0; 15],
            divisor: 15.0,
            bias: 0.0,
        },
        // Special kernels
        identity_kernel(3, 3),
        gaussian_3x3(),
        gaussian_5x5(),
        sobel_x(),
        sharpen_3x3(),
    ];

    // --- Build all configurations upfront ---
    let mut configs: Vec<Config> = Vec::new();
    let mut order = 0usize;

    for &(w, h) in &image_sizes {
        for kc in &base_kernels {
            for &em in &edge_modes {
                for &pa in &preserve_alphas {
                    configs.push(Config {
                        order,
                        width: w,
                        height: h,
                        kernel: kc.clone(),
                        edge_mode: em,
                        preserve_alpha: pa,
                    });
                    order += 1;
                }
            }
        }
    }

    let total_combos = configs.len();
    eprintln!("Running {} total combinations...\n", total_combos);

    // --- Determine parallelism ---
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!("Using {} threads\n", num_threads);

    // --- Split configs into chunks and run in parallel ---
    let progress = AtomicUsize::new(0);
    let chunk_size = (configs.len() + num_threads - 1) / num_threads;
    let chunks: Vec<&[Config]> = configs.chunks(chunk_size).collect();

    let mut all_results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let progress_ref = &progress;
                s.spawn(move || {
                    let mut thread_results = Vec::with_capacity(chunk.len());
                    for config in chunk {
                        let result = run_config(config);
                        thread_results.push(result);

                        let completed = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if completed % 50 == 0 {
                            eprintln!("  Progress: {}/{}", completed, total_combos);
                        }
                    }
                    thread_results
                })
            })
            .collect();

        // Collect results from all threads.
        let mut merged = Vec::with_capacity(total_combos);
        for handle in handles {
            merged.extend(handle.join().unwrap());
        }
        merged
    });

    // --- Sort by original order to get deterministic output ---
    all_results.sort_by_key(|r| r.order);

    let results = all_results;

    // Print results table
    println!();
    println!(
        "{:<12} {:<16} {:<12} {:<8} {:<14} {:>12} {:>12} {:>8} {:<10}",
        "Image Size", "Kernel", "Edge Mode", "PA", "Dispatch", "Naive (us)", "Prod (us)", "Speedup", "Status"
    );
    println!("{}", "-".repeat(112));

    for r in &results {
        println!(
            "{:<12} {:<16} {:<12} {:<8} {:<14} {:>12.1} {:>12.1} {:>7.2}x {:<10}",
            r.image_size,
            r.kernel,
            r.edge_mode,
            if r.preserve_alpha { "true" } else { "false" },
            r.dispatch,
            r.naive_us,
            r.prod_us,
            r.speedup,
            r.status,
        );
    }

    // Summary
    println!();
    println!("=== SUMMARY ===");
    println!("Total combinations: {}", results.len());

    let mismatches: Vec<&BenchResult> = results.iter().filter(|r| !r.bit_exact).collect();
    let regressions: Vec<&BenchResult> = results.iter().filter(|r| r.status == "REGRESSION").collect();
    let faster: Vec<&BenchResult> = results.iter().filter(|r| r.status == "FASTER").collect();
    let same: Vec<&BenchResult> = results.iter().filter(|r| r.status == "SAME").collect();

    println!("Bit-exact mismatches: {}", mismatches.len());
    println!("Regressions (prod >5% slower): {}", regressions.len());
    println!("Faster (prod >5% faster): {}", faster.len());
    println!("Same (within 5%): {}", same.len());

    if !mismatches.is_empty() {
        println!("\n!!! BIT-EXACT MISMATCHES !!!");
        for r in &mismatches {
            println!(
                "  {} | {} | {} | PA={}",
                r.image_size, r.kernel, r.edge_mode, r.preserve_alpha
            );
        }
    }

    if !regressions.is_empty() {
        println!("\n!!! PERFORMANCE REGRESSIONS (prod >5% slower than naive) !!!");
        for r in &regressions {
            println!(
                "  {} | {} | {} | PA={} | dispatch={} | naive={:.1}us prod={:.1}us speedup={:.2}x",
                r.image_size, r.kernel, r.edge_mode, r.preserve_alpha, r.dispatch, r.naive_us, r.prod_us, r.speedup
            );
        }
    }

    // Aggregate by image size
    println!("\n=== AGGREGATE BY IMAGE SIZE ===");
    for &(w, h) in &image_sizes {
        let size_str = format!("{}x{}", w, h);
        let size_results: Vec<&BenchResult> = results.iter().filter(|r| r.image_size == size_str).collect();
        if size_results.is_empty() {
            continue;
        }
        let avg_speedup: f64 =
            size_results.iter().map(|r| r.speedup).sum::<f64>() / size_results.len() as f64;
        let min_speedup = size_results
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        let max_speedup = size_results
            .iter()
            .map(|r| r.speedup)
            .fold(f64::NEG_INFINITY, f64::max);
        let regression_count = size_results.iter().filter(|r| r.status == "REGRESSION").count();
        println!(
            "  {:<12} avg={:.2}x  min={:.2}x  max={:.2}x  regressions={}",
            size_str, avg_speedup, min_speedup, max_speedup, regression_count
        );
    }

    // Aggregate by kernel
    println!("\n=== AGGREGATE BY KERNEL ===");
    let kernel_names: Vec<String> = {
        let mut names: Vec<String> = results.iter().map(|r| r.kernel.clone()).collect();
        names.sort();
        names.dedup();
        names
    };
    for kname in &kernel_names {
        let kern_results: Vec<&BenchResult> = results.iter().filter(|r| &r.kernel == kname).collect();
        if kern_results.is_empty() {
            continue;
        }
        let avg_speedup: f64 =
            kern_results.iter().map(|r| r.speedup).sum::<f64>() / kern_results.len() as f64;
        let min_speedup = kern_results
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        let regression_count = kern_results.iter().filter(|r| r.status == "REGRESSION").count();
        println!(
            "  {:<20} avg={:.2}x  min={:.2}x  regressions={}",
            kname, avg_speedup, min_speedup, regression_count
        );
    }

    let regression_count = results.iter().filter(|r| r.status == "REGRESSION").count();
    if regression_count > 0 {
        std::process::exit(1);
    }
}
