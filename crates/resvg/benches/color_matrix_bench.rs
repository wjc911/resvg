// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feColorMatrix` filter primitive.
//!
//! Tests multiple resolutions (64x64, 256x256, 1024x1024, 4096x4096) and all
//! matrix types (Matrix, Saturate, HueRotate, LuminanceToAlpha).
//!
//! Run with: cargo bench -p resvg --bench color_matrix_bench

use std::hint::black_box;
use std::time::{Duration, Instant};

use rgb::RGBA8;

// ---------------------------------------------------------------------------
// Inline the filter helpers so the benchmark is self-contained and does NOT
// live inside the core algorithm file.
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

#[inline]
fn from_normalized(c: f32) -> u8 {
    (f32_bound(0.0, c, 1.0) * 255.0) as u8
}

// ---- Naive (original) implementation ----

fn apply_naive_matrix(m: &[f32], data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        let a = pixel.a as f32 / 255.0;

        let new_r = r * m[0] + g * m[1] + b * m[2] + a * m[3] + m[4];
        let new_g = r * m[5] + g * m[6] + b * m[7] + a * m[8] + m[9];
        let new_b = r * m[10] + g * m[11] + b * m[12] + a * m[13] + m[14];
        let new_a = r * m[15] + g * m[16] + b * m[17] + a * m[18] + m[19];

        pixel.r = from_normalized(new_r);
        pixel.g = from_normalized(new_g);
        pixel.b = from_normalized(new_b);
        pixel.a = from_normalized(new_a);
    }
}

fn apply_naive_3x3(m: &[f32; 9], data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;

        let new_r = r * m[0] + g * m[1] + b * m[2];
        let new_g = r * m[3] + g * m[4] + b * m[5];
        let new_b = r * m[6] + g * m[7] + b * m[8];

        pixel.r = from_normalized(new_r);
        pixel.g = from_normalized(new_g);
        pixel.b = from_normalized(new_b);
    }
}

fn apply_naive_luminance(data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;

        let new_a = r * 0.2125 + g * 0.7154 + b * 0.0721;

        pixel.r = 0;
        pixel.g = 0;
        pixel.b = 0;
        pixel.a = from_normalized(new_a);
    }
}

// ---- Optimized (column-major [f32;4]) implementation ----

#[inline(never)]
fn apply_opt_matrix_cols(
    data: &mut [RGBA8],
    col0: &[f32; 4],
    col1: &[f32; 4],
    col2: &[f32; 4],
    col3: &[f32; 4],
    col4: &[f32; 4],
    modifies_alpha: bool,
) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;
        let a = pixel.a as f32 / 255.0;

        let mut out = [0.0_f32; 4];
        for i in 0..4 {
            out[i] = col0[i] * r;
            out[i] += col1[i] * g;
            out[i] += col2[i] * b;
            out[i] += col3[i] * a;
            out[i] += col4[i];
        }

        pixel.r = from_normalized(out[0]);
        pixel.g = from_normalized(out[1]);
        pixel.b = from_normalized(out[2]);
        if modifies_alpha {
            pixel.a = from_normalized(out[3]);
        }
    }
}

#[inline(never)]
fn apply_opt_luminance(data: &mut [RGBA8]) {
    for pixel in data.iter_mut() {
        let r = pixel.r as f32 / 255.0;
        let g = pixel.g as f32 / 255.0;
        let b = pixel.b as f32 / 255.0;

        let new_a = r * 0.2125 + g * 0.7154 + b * 0.0721;

        pixel.r = 0;
        pixel.g = 0;
        pixel.b = 0;
        pixel.a = from_normalized(new_a);
    }
}

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

fn make_test_pixels(count: usize) -> Vec<RGBA8> {
    let mut pixels = Vec::with_capacity(count);
    for i in 0..count {
        pixels.push(RGBA8 {
            r: (i.wrapping_mul(17) % 256) as u8,
            g: (i.wrapping_mul(31) % 256) as u8,
            b: (i.wrapping_mul(59) % 256) as u8,
            a: (i.wrapping_mul(97) % 256) as u8,
        });
    }
    pixels
}

/// Run `f` for at least `min_time` and return (total_time, iterations).
fn bench_fn<F: FnMut()>(mut f: F, min_time: Duration) -> (Duration, u64) {
    // Warm up
    for _ in 0..5 {
        f();
    }

    let mut iters = 0u64;
    let start = Instant::now();
    while start.elapsed() < min_time {
        f();
        iters += 1;
    }
    (start.elapsed(), iters)
}

fn format_throughput(elapsed: Duration, iters: u64, pixels: usize) -> String {
    let total_pixels = iters as f64 * pixels as f64;
    let secs = elapsed.as_secs_f64();
    let mpix_per_sec = total_pixels / secs / 1_000_000.0;
    let ns_per_pixel = secs * 1e9 / total_pixels;
    format!("{mpix_per_sec:>8.1} Mpix/s  ({ns_per_pixel:.2} ns/px)")
}

// ---------------------------------------------------------------------------
// Matrix coefficients used for benchmarking
// ---------------------------------------------------------------------------

fn full_4x5_matrix() -> Vec<f32> {
    vec![
        0.393, 0.769, 0.189, 0.0, 0.0, // sepia-like R
        0.349, 0.686, 0.168, 0.0, 0.0, // sepia-like G
        0.272, 0.534, 0.131, 0.0, 0.0, // sepia-like B
        0.0, 0.0, 0.0, 1.0, 0.0, // A unchanged
    ]
}

fn saturate_3x3(v: f32) -> [f32; 9] {
    [
        0.213 + 0.787 * v,
        0.715 - 0.715 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 + 0.285 * v,
        0.072 - 0.072 * v,
        0.213 - 0.213 * v,
        0.715 - 0.715 * v,
        0.072 + 0.928 * v,
    ]
}

fn hue_rotate_3x3(angle: f32) -> [f32; 9] {
    let angle = angle.to_radians();
    let a1 = angle.cos();
    let a2 = angle.sin();
    [
        0.213 + 0.787 * a1 - 0.213 * a2,
        0.715 - 0.715 * a1 - 0.715 * a2,
        0.072 - 0.072 * a1 + 0.928 * a2,
        0.213 - 0.213 * a1 + 0.143 * a2,
        0.715 + 0.285 * a1 + 0.140 * a2,
        0.072 - 0.072 * a1 - 0.283 * a2,
        0.213 - 0.213 * a1 - 0.787 * a2,
        0.715 - 0.715 * a1 + 0.715 * a2,
        0.072 + 0.928 * a1 + 0.072 * a2,
    ]
}

fn transpose_4x5(m: &[f32]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    (
        [m[0], m[5], m[10], m[15]],
        [m[1], m[6], m[11], m[16]],
        [m[2], m[7], m[12], m[17]],
        [m[3], m[8], m[13], m[18]],
        [m[4], m[9], m[14], m[19]],
    )
}

fn expand_3x3(m: &[f32; 9]) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    (
        [m[0], m[3], m[6], 0.0],
        [m[1], m[4], m[7], 0.0],
        [m[2], m[5], m[8], 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    )
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];

    let min_time = Duration::from_millis(500);

    println!("feColorMatrix Benchmark");
    println!("=======================");
    println!();

    // --- Full 4x5 Matrix ---
    {
        let m = full_4x5_matrix();
        let (c0, c1, c2, c3, c4) = transpose_4x5(&m);

        println!("Type: Matrix (4x5, sepia-like)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Naive", "Optimized", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_naive, i_naive) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_naive_matrix(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_opt, i_opt) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_opt_matrix_cols(
                        black_box(&mut data),
                        black_box(&c0),
                        black_box(&c1),
                        black_box(&c2),
                        black_box(&c3),
                        black_box(&c4),
                        true,
                    );
                },
                min_time,
            );

            let naive_ns = t_naive.as_nanos() as f64 / i_naive as f64 / n as f64;
            let opt_ns = t_opt.as_nanos() as f64 / i_opt as f64 / n as f64;
            let speedup = naive_ns / opt_ns;

            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_naive, i_naive, n),
                format_throughput(t_opt, i_opt, n),
                speedup,
            );
        }
        println!();
    }

    // --- Saturate ---
    {
        let m = saturate_3x3(0.5);
        let (c0, c1, c2, c3, c4) = expand_3x3(&m);

        println!("Type: Saturate (v=0.5)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Naive", "Optimized", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_naive, i_naive) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_naive_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_opt, i_opt) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_opt_matrix_cols(
                        black_box(&mut data),
                        black_box(&c0),
                        black_box(&c1),
                        black_box(&c2),
                        black_box(&c3),
                        black_box(&c4),
                        false,
                    );
                },
                min_time,
            );

            let naive_ns = t_naive.as_nanos() as f64 / i_naive as f64 / n as f64;
            let opt_ns = t_opt.as_nanos() as f64 / i_opt as f64 / n as f64;
            let speedup = naive_ns / opt_ns;

            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_naive, i_naive, n),
                format_throughput(t_opt, i_opt, n),
                speedup,
            );
        }
        println!();
    }

    // --- HueRotate ---
    {
        let m = hue_rotate_3x3(45.0);
        let (c0, c1, c2, c3, c4) = expand_3x3(&m);

        println!("Type: HueRotate (45 deg)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Naive", "Optimized", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_naive, i_naive) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_naive_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_opt, i_opt) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_opt_matrix_cols(
                        black_box(&mut data),
                        black_box(&c0),
                        black_box(&c1),
                        black_box(&c2),
                        black_box(&c3),
                        black_box(&c4),
                        false,
                    );
                },
                min_time,
            );

            let naive_ns = t_naive.as_nanos() as f64 / i_naive as f64 / n as f64;
            let opt_ns = t_opt.as_nanos() as f64 / i_opt as f64 / n as f64;
            let speedup = naive_ns / opt_ns;

            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_naive, i_naive, n),
                format_throughput(t_opt, i_opt, n),
                speedup,
            );
        }
        println!();
    }

    // --- LuminanceToAlpha ---
    {
        println!("Type: LuminanceToAlpha");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Naive", "Optimized", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_naive, i_naive) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_naive_luminance(black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_opt, i_opt) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_opt_luminance(black_box(&mut data));
                },
                min_time,
            );

            let naive_ns = t_naive.as_nanos() as f64 / i_naive as f64 / n as f64;
            let opt_ns = t_opt.as_nanos() as f64 / i_opt as f64 / n as f64;
            let speedup = naive_ns / opt_ns;

            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_naive, i_naive, n),
                format_throughput(t_opt, i_opt, n),
                speedup,
            );
        }
        println!();
    }

    // --- Correctness verification ---
    println!("Correctness Verification");
    println!("------------------------");
    {
        let m = full_4x5_matrix();
        let (c0, c1, c2, c3, c4) = transpose_4x5(&m);
        let n = 256 * 256;
        let base = make_test_pixels(n);

        let mut naive = base.clone();
        let mut opt = base.clone();

        apply_naive_matrix(&m, &mut naive);
        apply_opt_matrix_cols(&mut opt, &c0, &c1, &c2, &c3, &c4, true);

        let mismatches: usize = naive.iter().zip(opt.iter()).filter(|(a, b)| a != b).count();
        println!(
            "  Full 4x5 matrix:   {} mismatches out of {} pixels",
            mismatches, n
        );
    }

    {
        let m = saturate_3x3(0.5);
        let (c0, c1, c2, c3, c4) = expand_3x3(&m);
        let n = 256 * 256;
        let base = make_test_pixels(n);

        let mut naive = base.clone();
        let mut opt = base.clone();

        apply_naive_3x3(&m, &mut naive);
        apply_opt_matrix_cols(&mut opt, &c0, &c1, &c2, &c3, &c4, false);

        let mismatches: usize = naive.iter().zip(opt.iter()).filter(|(a, b)| a != b).count();
        println!(
            "  Saturate (v=0.5):  {} mismatches out of {} pixels",
            mismatches, n
        );
    }

    {
        let m = hue_rotate_3x3(45.0);
        let (c0, c1, c2, c3, c4) = expand_3x3(&m);
        let n = 256 * 256;
        let base = make_test_pixels(n);

        let mut naive = base.clone();
        let mut opt = base.clone();

        apply_naive_3x3(&m, &mut naive);
        apply_opt_matrix_cols(&mut opt, &c0, &c1, &c2, &c3, &c4, false);

        let mismatches: usize = naive.iter().zip(opt.iter()).filter(|(a, b)| a != b).count();
        println!(
            "  HueRotate (45):    {} mismatches out of {} pixels",
            mismatches, n
        );
    }

    {
        let n = 256 * 256;
        let base = make_test_pixels(n);

        let mut naive = base.clone();
        let mut opt = base.clone();

        apply_naive_luminance(&mut naive);
        apply_opt_luminance(&mut opt);

        let mismatches: usize = naive.iter().zip(opt.iter()).filter(|(a, b)| a != b).count();
        println!(
            "  LuminanceToAlpha:  {} mismatches out of {} pixels",
            mismatches, n
        );
    }
}
