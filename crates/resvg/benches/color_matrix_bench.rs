// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark for `feColorMatrix` filter primitive.
//!
//! Compares old (f32_bound) vs new (f32::clamp) from_normalized across
//! multiple resolutions (64x64, 256x256, 1024x1024, 4096x4096) and all
//! matrix types (Matrix, Saturate, HueRotate, LuminanceToAlpha).
//!
//! Run with: cargo bench -p resvg --bench color_matrix_bench

use std::hint::black_box;
use std::time::{Duration, Instant};

use rgb::RGBA8;

// ---------------------------------------------------------------------------
// Inline the filter helpers so the benchmark is self-contained.
// ---------------------------------------------------------------------------

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

#[inline]
fn from_normalized_new(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0) as u8
}

// ---- Old (main branch) implementation ----

#[inline(never)]
fn apply_old_matrix(m: &[f32], data: &mut [RGBA8]) {
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
fn apply_old_3x3(m: &[f32; 9], data: &mut [RGBA8]) {
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
fn apply_old_luminance(data: &mut [RGBA8]) {
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

// ---- New (current branch) implementation ----

#[inline(never)]
fn apply_new_matrix(m: &[f32], data: &mut [RGBA8]) {
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
fn apply_new_3x3(m: &[f32; 9], data: &mut [RGBA8]) {
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
fn apply_new_luminance(data: &mut [RGBA8]) {
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
// Matrix coefficients
// ---------------------------------------------------------------------------

fn full_4x5_matrix() -> Vec<f32> {
    vec![
        0.393, 0.769, 0.189, 0.0, 0.0, 0.349, 0.686, 0.168, 0.0, 0.0, 0.272, 0.534, 0.131, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
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
    let (a1, a2) = (angle.cos(), angle.sin());
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let resolutions: &[(u32, u32)] = &[(64, 64), (256, 256), (1024, 1024), (4096, 4096)];
    let min_time = Duration::from_millis(500);

    println!("feColorMatrix Benchmark (old f32_bound vs new f32::clamp)");
    println!("=========================================================");
    println!();

    // --- Full 4x5 Matrix ---
    {
        let m = full_4x5_matrix();

        println!("Type: Matrix (4x5, sepia-like)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Old (f32_bound)", "New (clamp)", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_old, i_old) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_old_matrix(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_new, i_new) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_new_matrix(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let old_ns = t_old.as_nanos() as f64 / i_old as f64 / n as f64;
            let new_ns = t_new.as_nanos() as f64 / i_new as f64 / n as f64;
            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_old, i_old, n),
                format_throughput(t_new, i_new, n),
                old_ns / new_ns,
            );
        }
        println!();
    }

    // --- Saturate ---
    {
        let m = saturate_3x3(0.5);

        println!("Type: Saturate (v=0.5)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Old (f32_bound)", "New (clamp)", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_old, i_old) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_old_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_new, i_new) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_new_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let old_ns = t_old.as_nanos() as f64 / i_old as f64 / n as f64;
            let new_ns = t_new.as_nanos() as f64 / i_new as f64 / n as f64;
            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_old, i_old, n),
                format_throughput(t_new, i_new, n),
                old_ns / new_ns,
            );
        }
        println!();
    }

    // --- HueRotate ---
    {
        let m = hue_rotate_3x3(45.0);

        println!("Type: HueRotate (45 deg)");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Old (f32_bound)", "New (clamp)", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_old, i_old) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_old_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_new, i_new) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_new_3x3(black_box(&m), black_box(&mut data));
                },
                min_time,
            );

            let old_ns = t_old.as_nanos() as f64 / i_old as f64 / n as f64;
            let new_ns = t_new.as_nanos() as f64 / i_new as f64 / n as f64;
            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_old, i_old, n),
                format_throughput(t_new, i_new, n),
                old_ns / new_ns,
            );
        }
        println!();
    }

    // --- LuminanceToAlpha ---
    {
        println!("Type: LuminanceToAlpha");
        println!(
            "{:<12} {:>30}  {:>30}  {:>8}",
            "Resolution", "Old (f32_bound)", "New (clamp)", "Speedup"
        );
        println!("{:-<94}", "");

        for &(w, h) in resolutions {
            let n = (w * h) as usize;
            let base = make_test_pixels(n);

            let mut data = base.clone();
            let (t_old, i_old) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_old_luminance(black_box(&mut data));
                },
                min_time,
            );

            let mut data = base.clone();
            let (t_new, i_new) = bench_fn(
                || {
                    data.copy_from_slice(&base);
                    apply_new_luminance(black_box(&mut data));
                },
                min_time,
            );

            let old_ns = t_old.as_nanos() as f64 / i_old as f64 / n as f64;
            let new_ns = t_new.as_nanos() as f64 / i_new as f64 / n as f64;
            println!(
                "{:>4}x{:<4}    {}  {}  {:>7.2}x",
                w,
                h,
                format_throughput(t_old, i_old, n),
                format_throughput(t_new, i_new, n),
                old_ns / new_ns,
            );
        }
        println!();
    }

    // --- Correctness ---
    println!("Correctness Verification");
    println!("------------------------");
    {
        let m = full_4x5_matrix();
        let n = 256 * 256;
        let base = make_test_pixels(n);
        let mut d1 = base.clone();
        let mut d2 = base.clone();
        apply_old_matrix(&m, &mut d1);
        apply_new_matrix(&m, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        println!("  Full 4x5 matrix:   {} mismatches out of {} pixels", mm, n);
    }
    {
        let m = saturate_3x3(0.5);
        let n = 256 * 256;
        let base = make_test_pixels(n);
        let mut d1 = base.clone();
        let mut d2 = base.clone();
        apply_old_3x3(&m, &mut d1);
        apply_new_3x3(&m, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        println!("  Saturate (v=0.5):  {} mismatches out of {} pixels", mm, n);
    }
    {
        let m = hue_rotate_3x3(45.0);
        let n = 256 * 256;
        let base = make_test_pixels(n);
        let mut d1 = base.clone();
        let mut d2 = base.clone();
        apply_old_3x3(&m, &mut d1);
        apply_new_3x3(&m, &mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        println!("  HueRotate (45):    {} mismatches out of {} pixels", mm, n);
    }
    {
        let n = 256 * 256;
        let base = make_test_pixels(n);
        let mut d1 = base.clone();
        let mut d2 = base.clone();
        apply_old_luminance(&mut d1);
        apply_new_luminance(&mut d2);
        let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
        println!("  LuminanceToAlpha:  {} mismatches out of {} pixels", mm, n);
    }
}
