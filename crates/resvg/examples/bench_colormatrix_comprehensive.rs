// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive benchmark for `feColorMatrix` filter primitive.
//!
//! Tests 7 image sizes, 16 matrix configurations, 3 input patterns,
//! comparing the old (f32_bound) vs new (f32::clamp) from_normalized
//! implementations. The Matrix path uses the same row-major scalar
//! loop that LLVM auto-vectorizes across multiple pixels.
//!
//! Uses median-of-N-runs methodology with interleaved execution.
//! Benchmark configurations are executed in parallel across CPU cores.
//!
//! Run with:
//!   cargo run --release -p resvg --example bench_colormatrix_comprehensive

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rgb::RGBA8;

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------

/// Old from_normalized using manual bounds (matches main branch f32_bound).
#[inline]
fn from_normalized_old(c: f32) -> u8 {
    let v = if c > 1.0 {
        1.0
    } else if c < 0.0 {
        0.0
    } else {
        c
    };
    (v * 255.0) as u8
}

/// New from_normalized using f32::clamp (current branch).
#[inline]
fn from_normalized_new(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0) as u8
}

// ---------------------------------------------------------------------------
// Old implementations (main branch - uses f32_bound)
// ---------------------------------------------------------------------------

#[inline(never)]
fn old_matrix(m: &[f32], data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        let a = pixel.a as f32 / 255.0;

        let new_r = r * m[0] + g * m[1] + b * m[2] + a * m[3] + m[4];
        let new_g = r * m[5] + g * m[6] + b * m[7] + a * m[8] + m[9];
        let new_b = r * m[10] + g * m[11] + b * m[12] + a * m[13] + m[14];
        let new_a = r * m[15] + g * m[16] + b * m[17] + a * m[18] + m[19];

        pixel.r = from_normalized_old(new_r);
        pixel.g = from_normalized_old(new_g);
        pixel.b = from_normalized_old(new_b);
        pixel.a = from_normalized_old(new_a);
    }
}

#[inline(never)]
fn old_saturate(v: f32, data: &mut [RGBA8]) {
    let m = [
        0.213 + 0.787 * v,
        0.715 - 0.715 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 + 0.285 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 - 0.715 * v,
        0.072 + 0.928 * v,
    ];
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = from_normalized_old(r * m[0] + g * m[1] + b * m[2]);
        pixel.g = from_normalized_old(r * m[3] + g * m[4] + b * m[5]);
        pixel.b = from_normalized_old(r * m[6] + g * m[7] + b * m[8]);
    }
}

#[inline(never)]
fn old_hue_rotate(angle_deg: f32, data: &mut [RGBA8]) {
    let angle = angle_deg.to_radians();
    let (a1, a2) = (angle.cos(), angle.sin());
    let m = [
        0.213 + 0.787 * a1 - 0.213 * a2,
        0.715 - 0.715 * a1 - 0.715 * a2,
        0.072 - 0.072 * a1 + 0.928 * a2,
        0.213 - 0.213 * a1 + 0.143 * a2,
        0.715 + 0.285 * a1 + 0.140 * a2,
        0.072 - 0.072 * a1 - 0.283 * a2,
        0.213 - 0.213 * a1 - 0.787 * a2,
        0.715 - 0.715 * a1 + 0.715 * a2,
        0.072 + 0.928 * a1 + 0.072 * a2,
    ];
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = from_normalized_old(r * m[0] + g * m[1] + b * m[2]);
        pixel.g = from_normalized_old(r * m[3] + g * m[4] + b * m[5]);
        pixel.b = from_normalized_old(r * m[6] + g * m[7] + b * m[8]);
    }
}

#[inline(never)]
fn old_luminance_to_alpha(data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = 0;
        pixel.g = 0;
        pixel.b = 0;
        pixel.a = from_normalized_old(r * 0.2125 + g * 0.7154 + b * 0.0721);
    }
}

// ---------------------------------------------------------------------------
// New implementations (current branch - uses f32::clamp, row-major)
// ---------------------------------------------------------------------------

#[inline(never)]
fn new_matrix(m: &[f32], data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        let a = pixel.a as f32 / 255.0;

        let new_r = r * m[0] + g * m[1] + b * m[2] + a * m[3] + m[4];
        let new_g = r * m[5] + g * m[6] + b * m[7] + a * m[8] + m[9];
        let new_b = r * m[10] + g * m[11] + b * m[12] + a * m[13] + m[14];
        let new_a = r * m[15] + g * m[16] + b * m[17] + a * m[18] + m[19];

        pixel.r = from_normalized_new(new_r);
        pixel.g = from_normalized_new(new_g);
        pixel.b = from_normalized_new(new_b);
        pixel.a = from_normalized_new(new_a);
    }
}

#[inline(never)]
fn new_saturate(v: f32, data: &mut [RGBA8]) {
    let m = [
        0.213 + 0.787 * v,
        0.715 - 0.715 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 + 0.285 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 - 0.715 * v,
        0.072 + 0.928 * v,
    ];
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = from_normalized_new(r * m[0] + g * m[1] + b * m[2]);
        pixel.g = from_normalized_new(r * m[3] + g * m[4] + b * m[5]);
        pixel.b = from_normalized_new(r * m[6] + g * m[7] + b * m[8]);
    }
}

#[inline(never)]
fn new_hue_rotate(angle_deg: f32, data: &mut [RGBA8]) {
    let angle = angle_deg.to_radians();
    let (a1, a2) = (angle.cos(), angle.sin());
    let m = [
        0.213 + 0.787 * a1 - 0.213 * a2,
        0.715 - 0.715 * a1 - 0.715 * a2,
        0.072 - 0.072 * a1 + 0.928 * a2,
        0.213 - 0.213 * a1 + 0.143 * a2,
        0.715 + 0.285 * a1 + 0.140 * a2,
        0.072 - 0.072 * a1 - 0.283 * a2,
        0.213 - 0.213 * a1 - 0.787 * a2,
        0.715 - 0.715 * a1 + 0.715 * a2,
        0.072 + 0.928 * a1 + 0.072 * a2,
    ];
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = from_normalized_new(r * m[0] + g * m[1] + b * m[2]);
        pixel.g = from_normalized_new(r * m[3] + g * m[4] + b * m[5]);
        pixel.b = from_normalized_new(r * m[6] + g * m[7] + b * m[8]);
    }
}

#[inline(never)]
fn new_luminance_to_alpha(data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        pixel.r = 0;
        pixel.g = 0;
        pixel.b = 0;
        pixel.a = from_normalized_new(r * 0.2125 + g * 0.7154 + b * 0.0721);
    }
}

// ---------------------------------------------------------------------------
// Matrix coefficient factories
// ---------------------------------------------------------------------------

fn identity_4x5() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0,
    ]
}

fn sepia_4x5() -> Vec<f32> {
    vec![
        0.393, 0.769, 0.189, 0.0, 0.0, 0.349, 0.686, 0.168, 0.0, 0.0, 0.272, 0.534, 0.131, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ]
}

fn grayscale_4x5() -> Vec<f32> {
    vec![
        0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ]
}

fn channel_swap_4x5() -> Vec<f32> {
    vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0,
    ]
}

fn extreme_4x5() -> Vec<f32> {
    vec![
        2.0, -1.0, 0.5, 0.0, -0.5, 0.0, 3.0, -0.5, 0.0, 0.2, -1.0, 0.0, 2.5, 0.0, -0.3, 0.0, 0.0,
        0.0, 1.5, -0.1,
    ]
}

fn random_coeffs_4x5() -> Vec<f32> {
    vec![
        0.37, -0.12, 0.85, 0.02, -0.15, 0.91, 0.44, -0.33, 0.07, 0.10, -0.25, 0.68, 0.52, -0.11,
        0.03, 0.14, -0.06, 0.29, 0.78, -0.08,
    ]
}

// ---------------------------------------------------------------------------
// Input pattern generators
// ---------------------------------------------------------------------------

fn make_solid_opaque(n: usize) -> Vec<RGBA8> {
    vec![
        RGBA8 {
            r: 128,
            g: 64,
            b: 192,
            a: 255
        };
        n
    ]
}

fn make_gradient(n: usize) -> Vec<RGBA8> {
    (0..n)
        .map(|i| {
            let t = (i as f32) / (n.max(1) as f32);
            RGBA8 {
                r: (t * 255.0) as u8,
                g: ((1.0 - t) * 255.0) as u8,
                b: ((t * 0.5 + 0.25) * 255.0) as u8,
                a: ((1.0 - t * 0.3) * 255.0) as u8,
            }
        })
        .collect()
}

fn make_random_rgba(n: usize) -> Vec<RGBA8> {
    let mut seed: u32 = 0xDEADBEEF;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            RGBA8 {
                r: (seed >> 24) as u8,
                g: (seed >> 16) as u8,
                b: (seed >> 8) as u8,
                a: seed as u8,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

const NUM_ROUNDS: usize = 7;

fn measure_once<F: FnMut()>(f: &mut F, iters: u64) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn bench_pair<FN: FnMut(), FO: FnMut()>(
    old_fn: &mut FN,
    new_fn: &mut FO,
    pixel_count: usize,
) -> (f64, f64) {
    let iters = if pixel_count <= 16 {
        50000u64
    } else if pixel_count <= 256 {
        5000
    } else if pixel_count <= 4096 {
        500
    } else if pixel_count <= 65536 {
        50
    } else if pixel_count <= 262144 {
        10
    } else {
        3
    };

    // Warmup
    for _ in 0..3 {
        old_fn();
        new_fn();
    }

    let mut old_times = Vec::with_capacity(NUM_ROUNDS);
    let mut new_times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        old_times.push(measure_once(old_fn, iters));
        new_times.push(measure_once(new_fn, iters));
    }

    old_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    new_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    (old_times[NUM_ROUNDS / 2], new_times[NUM_ROUNDS / 2])
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct BenchResult {
    order: usize,
    image_size: String,
    matrix_type: String,
    input_pattern: String,
    old_us: f64,
    new_us: f64,
    speedup: f64,
    pixel_count: usize,
}

// ---------------------------------------------------------------------------
// Configuration for one benchmark run
// ---------------------------------------------------------------------------

/// The kind of color-matrix operation to benchmark.
#[derive(Clone)]
enum OpKind {
    Matrix(Vec<f32>),
    Saturate(f32),
    HueRotate(f32),
    LuminanceToAlpha,
}

/// Which input-pattern generator to use (must be representable as an enum
/// because function pointers are not `Send` through all code paths and an
/// enum is cleaner).
#[derive(Clone, Copy)]
enum PatternKind {
    SolidOpaque,
    Gradient,
    RandomRgba,
}

impl PatternKind {
    fn label(self) -> &'static str {
        match self {
            PatternKind::SolidOpaque => "solid_opaque",
            PatternKind::Gradient => "gradient",
            PatternKind::RandomRgba => "random_rgba",
        }
    }

    fn generate(self, n: usize) -> Vec<RGBA8> {
        match self {
            PatternKind::SolidOpaque => make_solid_opaque(n),
            PatternKind::Gradient => make_gradient(n),
            PatternKind::RandomRgba => make_random_rgba(n),
        }
    }
}

/// All parameters needed for a single benchmark measurement.
#[derive(Clone)]
struct Config {
    order: usize,
    width: u32,
    height: u32,
    matrix_label: String,
    pattern: PatternKind,
    op: OpKind,
}

/// Execute a single benchmark configuration and return the result.
fn run_config(cfg: &Config) -> BenchResult {
    let n = (cfg.width * cfg.height) as usize;
    let base = cfg.pattern.generate(n);
    let mut d1 = base.clone();
    let mut d2 = base.clone();

    let (old_ns, new_ns) = match &cfg.op {
        OpKind::Matrix(m) => bench_pair(
            &mut || {
                d1.copy_from_slice(&base);
                old_matrix(black_box(m), black_box(&mut d1));
            },
            &mut || {
                d2.copy_from_slice(&base);
                new_matrix(black_box(m), black_box(&mut d2));
            },
            n,
        ),
        OpKind::Saturate(v) => {
            let v = *v;
            bench_pair(
                &mut || {
                    d1.copy_from_slice(&base);
                    old_saturate(black_box(v), black_box(&mut d1));
                },
                &mut || {
                    d2.copy_from_slice(&base);
                    new_saturate(black_box(v), black_box(&mut d2));
                },
                n,
            )
        }
        OpKind::HueRotate(angle) => {
            let angle = *angle;
            bench_pair(
                &mut || {
                    d1.copy_from_slice(&base);
                    old_hue_rotate(black_box(angle), black_box(&mut d1));
                },
                &mut || {
                    d2.copy_from_slice(&base);
                    new_hue_rotate(black_box(angle), black_box(&mut d2));
                },
                n,
            )
        }
        OpKind::LuminanceToAlpha => bench_pair(
            &mut || {
                d1.copy_from_slice(&base);
                old_luminance_to_alpha(black_box(&mut d1));
            },
            &mut || {
                d2.copy_from_slice(&base);
                new_luminance_to_alpha(black_box(&mut d2));
            },
            n,
        ),
    };

    BenchResult {
        order: cfg.order,
        image_size: format!("{}x{}", cfg.width, cfg.height),
        matrix_type: cfg.matrix_label.clone(),
        input_pattern: cfg.pattern.label().to_string(),
        old_us: old_ns / 1000.0,
        new_us: new_ns / 1000.0,
        speedup: old_ns / new_ns,
        pixel_count: n,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let sizes: &[(u32, u32)] = &[
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    let patterns = [
        PatternKind::SolidOpaque,
        PatternKind::Gradient,
        PatternKind::RandomRgba,
    ];

    println!("feColorMatrix Comprehensive Benchmark (old f32_bound vs new f32::clamp)");
    println!("========================================================================");
    println!("Methodology: median of {} interleaved runs", NUM_ROUNDS);
    println!();

    // --- Build all benchmark configurations upfront ---
    let mut configs: Vec<Config> = Vec::new();

    // Matrix configs
    let matrix_configs: Vec<(&str, Vec<f32>)> = vec![
        ("Matrix/identity", identity_4x5()),
        ("Matrix/sepia", sepia_4x5()),
        ("Matrix/grayscale", grayscale_4x5()),
        ("Matrix/channel_swap", channel_swap_4x5()),
        ("Matrix/extreme", extreme_4x5()),
        ("Matrix/random_coeffs", random_coeffs_4x5()),
    ];

    for (name, m) in &matrix_configs {
        for &pat in &patterns {
            for &(w, h) in sizes {
                configs.push(Config {
                    order: configs.len(),
                    width: w,
                    height: h,
                    matrix_label: name.to_string(),
                    pattern: pat,
                    op: OpKind::Matrix(m.clone()),
                });
            }
        }
    }

    // Saturate configs
    for &sat_val in &[0.0001f32, 0.5, 1.0, 2.0] {
        let label = format!("Saturate/{:.1}", sat_val);
        for &pat in &patterns {
            for &(w, h) in sizes {
                configs.push(Config {
                    order: configs.len(),
                    width: w,
                    height: h,
                    matrix_label: label.clone(),
                    pattern: pat,
                    op: OpKind::Saturate(sat_val),
                });
            }
        }
    }

    // HueRotate configs
    for &angle in &[0.0f32, 45.0, 90.0, 180.0, 270.0] {
        let label = format!("HueRotate/{}deg", angle as i32);
        for &pat in &patterns {
            for &(w, h) in sizes {
                configs.push(Config {
                    order: configs.len(),
                    width: w,
                    height: h,
                    matrix_label: label.clone(),
                    pattern: pat,
                    op: OpKind::HueRotate(angle),
                });
            }
        }
    }

    // LuminanceToAlpha configs
    for &pat in &patterns {
        for &(w, h) in sizes {
            configs.push(Config {
                order: configs.len(),
                width: w,
                height: h,
                matrix_label: "LuminanceToAlpha".to_string(),
                pattern: pat,
                op: OpKind::LuminanceToAlpha,
            });
        }
    }

    let total = configs.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    eprintln!(
        "Running {} benchmark configurations across {} threads...",
        total, num_threads
    );

    // --- Parallel execution ---
    let progress = AtomicUsize::new(0);

    let mut results: Vec<BenchResult> = std::thread::scope(|s| {
        let chunk_size = (configs.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[Config]> = configs.chunks(chunk_size).collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let progress = &progress;
                s.spawn(move || {
                    let mut local_results = Vec::with_capacity(chunk.len());
                    for cfg in chunk {
                        let result = run_config(cfg);
                        local_results.push(result);
                        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                        eprint!("\r  Progress: {}/{}", done, total);
                    }
                    local_results
                })
            })
            .collect();

        let mut all_results = Vec::with_capacity(total);
        for handle in handles {
            all_results.extend(handle.join().unwrap());
        }
        all_results
    });

    eprintln!("\r  Progress: {}/{}  done", total, total);

    // Sort by original insertion order
    results.sort_by_key(|r| r.order);

    // --- Correctness ---
    println!();
    println!("Correctness Verification (old vs new produce identical output)");
    println!("--------------------------------------------------------------");
    let mut ok = true;
    for (name, m) in &matrix_configs {
        let base = make_random_rgba(65536);
        let (mut d1, mut d2) = (base.clone(), base.clone());
        old_matrix(m, &mut d1);
        new_matrix(m, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        let s = if mm == 0 {
            "PASS"
        } else {
            ok = false;
            "FAIL"
        };
        println!("  {:<25} {} ({} mismatches)", name, s, mm);
    }
    for &v in &[0.0001f32, 0.5, 1.0, 2.0] {
        let base = make_random_rgba(65536);
        let (mut d1, mut d2) = (base.clone(), base.clone());
        old_saturate(v, &mut d1);
        new_saturate(v, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        let s = if mm == 0 {
            "PASS"
        } else {
            ok = false;
            "FAIL"
        };
        println!("  Saturate({:.1})             {} ({} mismatches)", v, s, mm);
    }
    for &a in &[0.0f32, 45.0, 90.0, 180.0, 270.0] {
        let base = make_random_rgba(65536);
        let (mut d1, mut d2) = (base.clone(), base.clone());
        old_hue_rotate(a, &mut d1);
        new_hue_rotate(a, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        let s = if mm == 0 {
            "PASS"
        } else {
            ok = false;
            "FAIL"
        };
        println!(
            "  HueRotate({}deg)          {} ({} mismatches)",
            a as i32, s, mm
        );
    }
    {
        let base = make_random_rgba(65536);
        let (mut d1, mut d2) = (base.clone(), base.clone());
        old_luminance_to_alpha(&mut d1);
        new_luminance_to_alpha(&mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        let s = if mm == 0 {
            "PASS"
        } else {
            ok = false;
            "FAIL"
        };
        println!("  LuminanceToAlpha          {} ({} mismatches)", s, mm);
    }
    if !ok {
        println!("\n  WARNING: Some checks FAILED!");
    }

    // --- Results table ---
    println!();
    println!("Results Table (Old=main/f32_bound, New=branch/f32::clamp)");
    println!("==========================================================");
    println!();
    println!(
        "{:<12} | {:<25} | {:<12} | {:>12} | {:>12} | {:>8}",
        "Image Size", "Matrix Type", "Input", "Old (us)", "New (us)", "Speedup"
    );
    println!("{}", "-".repeat(95));

    for r in &results {
        let flag = if r.speedup < 0.95 {
            " *** REGRESSION"
        } else {
            ""
        };
        println!(
            "{:<12} | {:<25} | {:<12} | {:>12.2} | {:>12.2} | {:>7.2}x{}",
            r.image_size, r.matrix_type, r.input_pattern, r.old_us, r.new_us, r.speedup, flag
        );
    }

    // --- Regression summary ---
    println!();
    println!("Regression Analysis (>5% slower = speedup < 0.95, images >= 256x256)");
    println!("--------------------------------------------------------------------");
    let regs: Vec<&BenchResult> = results
        .iter()
        .filter(|r| r.speedup < 0.95 && r.pixel_count >= 256 * 256)
        .collect();
    if regs.is_empty() {
        println!("  No regressions detected.");
    } else {
        println!("  {} regressions found:", regs.len());
        for r in &regs {
            println!(
                "    {} | {} | {} | {:.2}x (old={:.2}us, new={:.2}us)",
                r.image_size, r.matrix_type, r.input_pattern, r.speedup, r.old_us, r.new_us
            );
        }
    }

    // --- Summary ---
    println!();
    println!("Summary by Matrix Type (averaged over 256x256+ images, all inputs)");
    println!("------------------------------------------------------------------");
    let categories = [
        "Matrix/identity",
        "Matrix/sepia",
        "Matrix/grayscale",
        "Matrix/channel_swap",
        "Matrix/extreme",
        "Matrix/random_coeffs",
        "Saturate/0.0",
        "Saturate/0.5",
        "Saturate/1.0",
        "Saturate/2.0",
        "HueRotate/0deg",
        "HueRotate/45deg",
        "HueRotate/90deg",
        "HueRotate/180deg",
        "HueRotate/270deg",
        "LuminanceToAlpha",
    ];
    println!(
        "{:<25} | {:>12} | {:>12} | {:>8} | {}",
        "Matrix Type", "Avg Old us", "Avg New us", "Speedup", "Status"
    );
    println!("{}", "-".repeat(78));
    for cat in &categories {
        let m: Vec<&BenchResult> = results
            .iter()
            .filter(|r| r.matrix_type == *cat && r.pixel_count >= 256 * 256)
            .collect();
        if m.is_empty() {
            continue;
        }
        let ao: f64 = m.iter().map(|r| r.old_us).sum::<f64>() / m.len() as f64;
        let an: f64 = m.iter().map(|r| r.new_us).sum::<f64>() / m.len() as f64;
        let s = ao / an;
        let st = if s < 0.95 {
            "REGRESSION"
        } else if s > 1.05 {
            "IMPROVED"
        } else {
            "UNCHANGED"
        };
        println!(
            "{:<25} | {:>12.2} | {:>12.2} | {:>7.2}x | {}",
            cat, ao, an, s, st
        );
    }

    println!();
    println!("Done.");
}
