// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Comprehensive feTurbulence benchmark comparing naive vs optimized implementations.
//!
//! Run with: cargo run --example bench_turbulence_comprehensive -p resvg --release
//!
//! This benchmark exercises all parameter combinations:
//! - Image sizes: 4x4, 16x16, 64x64, 256x256, 512x512, 1024x1024
//! - Octaves: 1, 2, 4, 8
//! - Modes: fractalNoise, turbulence
//! - Stitching: on, off
//! - Base frequencies: (0.01,0.01), (0.05,0.05), (0.1,0.1), (0.05,0.01)
//! - Seeds: 0, 42, -7

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants (same as in turbulence.rs)
// ---------------------------------------------------------------------------

const RAND_M: i32 = 2147483647;
const RAND_A: i32 = 16807;
const RAND_Q: i32 = 127773;
const RAND_R: i32 = 2836;
const B_SIZE: usize = 0x100;
const B_SIZE_32: i32 = 0x100;
const B_LEN: usize = B_SIZE + B_SIZE + 2;
const BM: i32 = 0xff;
const PERLIN_N: i32 = 0x1000;

// ---------------------------------------------------------------------------
// Minimal pixel type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pixel {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Clone, Copy)]
struct StitchInfo {
    width: i32,
    height: i32,
    wrap_x: i32,
    wrap_y: i32,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn random(seed: i32) -> i32 {
    let mut result = RAND_A * (seed % RAND_Q) - RAND_R * (seed / RAND_Q);
    if result <= 0 {
        result += RAND_M;
    }
    result
}

#[inline]
fn s_curve(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if val < min {
        min
    } else {
        val
    }
}

fn approx_zero_ulps(v: f64, ulps: u32) -> bool {
    let bits = v.to_bits();
    let neg_bits = (-v).to_bits();
    bits <= ulps as u64 || neg_bits <= ulps as u64
}

// ---------------------------------------------------------------------------
// Naive implementation (verbatim copy from turbulence.rs #[cfg(test)] code)
// ---------------------------------------------------------------------------

fn init_naive(mut seed: i32) -> (Vec<usize>, Vec<Vec<Vec<f64>>>) {
    let mut lattice_selector = vec![0; B_LEN];
    let mut gradient = vec![vec![vec![0.0; 2]; B_LEN]; 4];

    if seed <= 0 {
        seed = -seed % (RAND_M - 1) + 1;
    }
    if seed > RAND_M - 1 {
        seed = RAND_M - 1;
    }

    for k in 0..4 {
        for i in 0..B_SIZE {
            lattice_selector[i] = i;
            for j in 0..2 {
                seed = random(seed);
                gradient[k][i][j] =
                    ((seed % (B_SIZE_32 + B_SIZE_32)) - B_SIZE_32) as f64 / B_SIZE_32 as f64;
            }
            let s = (gradient[k][i][0] * gradient[k][i][0]
                + gradient[k][i][1] * gradient[k][i][1])
                .sqrt();
            gradient[k][i][0] /= s;
            gradient[k][i][1] /= s;
        }
    }

    for i in (1..B_SIZE).rev() {
        let k = lattice_selector[i];
        seed = random(seed);
        let j = (seed % B_SIZE_32) as usize;
        lattice_selector[i] = lattice_selector[j];
        lattice_selector[j] = k;
    }

    for i in 0..B_SIZE + 2 {
        lattice_selector[B_SIZE + i] = lattice_selector[i];
        for g in gradient.iter_mut().take(4) {
            for j in 0..2 {
                g[B_SIZE + i][j] = g[i][j];
            }
        }
    }

    (lattice_selector, gradient)
}

fn noise2_naive(
    color_channel: usize,
    x: f64,
    y: f64,
    lattice_selector: &[usize],
    gradient: &[Vec<Vec<f64>>],
    stitch_info: Option<StitchInfo>,
) -> f64 {
    let t = x + PERLIN_N as f64;
    let mut bx0 = t as i32;
    let mut bx1 = bx0 + 1;
    let rx0 = t - t as i64 as f64;
    let rx1 = rx0 - 1.0;
    let t = y + PERLIN_N as f64;
    let mut by0 = t as i32;
    let mut by1 = by0 + 1;
    let ry0 = t - t as i64 as f64;
    let ry1 = ry0 - 1.0;

    if let Some(info) = stitch_info {
        if bx0 >= info.wrap_x {
            bx0 -= info.width;
        }
        if bx1 >= info.wrap_x {
            bx1 -= info.width;
        }
        if by0 >= info.wrap_y {
            by0 -= info.height;
        }
        if by1 >= info.wrap_y {
            by1 -= info.height;
        }
    }

    bx0 &= BM;
    bx1 &= BM;
    by0 &= BM;
    by1 &= BM;
    let i = lattice_selector[bx0 as usize];
    let j = lattice_selector[bx1 as usize];
    let b00 = lattice_selector[i + by0 as usize];
    let b10 = lattice_selector[j + by0 as usize];
    let b01 = lattice_selector[i + by1 as usize];
    let b11 = lattice_selector[j + by1 as usize];
    let sx = s_curve(rx0);
    let sy = s_curve(ry0);
    let q = &gradient[color_channel][b00];
    let u = rx0 * q[0] + ry0 * q[1];
    let q = &gradient[color_channel][b10];
    let v = rx1 * q[0] + ry0 * q[1];
    let a = lerp(sx, u, v);
    let q = &gradient[color_channel][b01];
    let u = rx0 * q[0] + ry1 * q[1];
    let q = &gradient[color_channel][b11];
    let v = rx1 * q[0] + ry1 * q[1];
    let b = lerp(sx, u, v);
    lerp(sy, a, b)
}

fn turbulence_naive(
    color_channel: usize,
    mut x: f64,
    mut y: f64,
    tile_x: f64,
    tile_y: f64,
    tile_width: f64,
    tile_height: f64,
    mut base_freq_x: f64,
    mut base_freq_y: f64,
    num_octaves: u32,
    fractal_sum: bool,
    do_stitching: bool,
    lattice_selector: &[usize],
    gradient: &[Vec<Vec<f64>>],
) -> f64 {
    let mut stitch = if do_stitching {
        if !approx_zero_ulps(base_freq_x, 4) {
            let lo_freq = (tile_width * base_freq_x).floor() / tile_width;
            let hi_freq = (tile_width * base_freq_x).ceil() / tile_width;
            if base_freq_x / lo_freq < hi_freq / base_freq_x {
                base_freq_x = lo_freq;
            } else {
                base_freq_x = hi_freq;
            }
        }
        if !approx_zero_ulps(base_freq_y, 4) {
            let lo_freq = (tile_height * base_freq_y).floor() / tile_height;
            let hi_freq = (tile_height * base_freq_y).ceil() / tile_height;
            if base_freq_y / lo_freq < hi_freq / base_freq_y {
                base_freq_y = lo_freq;
            } else {
                base_freq_y = hi_freq;
            }
        }
        let width = (tile_width * base_freq_x + 0.5) as i32;
        let height = (tile_height * base_freq_y + 0.5) as i32;
        let wrap_x = (tile_x * base_freq_x + PERLIN_N as f64 + width as f64) as i32;
        let wrap_y = (tile_y * base_freq_y + PERLIN_N as f64 + height as f64) as i32;
        Some(StitchInfo {
            width,
            height,
            wrap_x,
            wrap_y,
        })
    } else {
        None
    };

    let mut sum = 0.0;
    x *= base_freq_x;
    y *= base_freq_y;
    let mut ratio = 1.0;
    for _ in 0..num_octaves {
        if fractal_sum {
            sum += noise2_naive(color_channel, x, y, lattice_selector, gradient, stitch) / ratio;
        } else {
            sum += noise2_naive(color_channel, x, y, lattice_selector, gradient, stitch).abs()
                / ratio;
        }
        x *= 2.0;
        y *= 2.0;
        ratio *= 2.0;

        if let Some(ref mut stitch) = stitch {
            stitch.width *= 2;
            stitch.wrap_x = 2 * stitch.wrap_x - PERLIN_N;
            stitch.height *= 2;
            stitch.wrap_y = 2 * stitch.wrap_y - PERLIN_N;
        }
    }
    sum
}

fn apply_naive(
    offset_x: f64,
    offset_y: f64,
    sx: f64,
    sy: f64,
    base_frequency_x: f64,
    base_frequency_y: f64,
    num_octaves: u32,
    seed: i32,
    stitch_tiles: bool,
    fractal_noise: bool,
    width: u32,
    height: u32,
    data: &mut [Pixel],
) {
    let (lattice_selector, gradient) = init_naive(seed);
    let mut x = 0u32;
    let mut y = 0u32;
    for pixel in data.iter_mut() {
        let turb = |channel| {
            let (tx, ty) = ((x as f64 + offset_x) / sx, (y as f64 + offset_y) / sy);
            let n = turbulence_naive(
                channel,
                tx,
                ty,
                x as f64,
                y as f64,
                width as f64,
                height as f64,
                base_frequency_x,
                base_frequency_y,
                num_octaves,
                fractal_noise,
                stitch_tiles,
                &lattice_selector,
                &gradient,
            );
            let n = if fractal_noise {
                (n * 255.0 + 255.0) / 2.0
            } else {
                n * 255.0
            };
            (f32_bound(0.0, n as f32, 255.0) + 0.5) as u8
        };

        pixel.r = turb(0);
        pixel.g = turb(1);
        pixel.b = turb(2);
        pixel.a = turb(3);

        x += 1;
        if x == width {
            x = 0;
            y += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized implementation (verbatim copy from turbulence.rs)
// ---------------------------------------------------------------------------

struct GradientTable {
    data: [[[f64; 2]; B_LEN]; 4],
}

fn init_optimized(mut seed: i32) -> ([usize; B_LEN], GradientTable) {
    let mut lattice_selector = [0usize; B_LEN];
    let mut gradient = GradientTable {
        data: [[[0.0; 2]; B_LEN]; 4],
    };

    if seed <= 0 {
        seed = -seed % (RAND_M - 1) + 1;
    }
    if seed > RAND_M - 1 {
        seed = RAND_M - 1;
    }

    for k in 0..4 {
        for i in 0..B_SIZE {
            lattice_selector[i] = i;
            for j in 0..2 {
                seed = random(seed);
                gradient.data[k][i][j] =
                    ((seed % (B_SIZE_32 + B_SIZE_32)) - B_SIZE_32) as f64 / B_SIZE_32 as f64;
            }
            let s = (gradient.data[k][i][0] * gradient.data[k][i][0]
                + gradient.data[k][i][1] * gradient.data[k][i][1])
                .sqrt();
            gradient.data[k][i][0] /= s;
            gradient.data[k][i][1] /= s;
        }
    }

    for i in (1..B_SIZE).rev() {
        let k = lattice_selector[i];
        seed = random(seed);
        let j = (seed % B_SIZE_32) as usize;
        lattice_selector[i] = lattice_selector[j];
        lattice_selector[j] = k;
    }

    for i in 0..B_SIZE + 2 {
        lattice_selector[B_SIZE + i] = lattice_selector[i];
        for k in 0..4 {
            for j in 0..2 {
                gradient.data[k][B_SIZE + i][j] = gradient.data[k][i][j];
            }
        }
    }

    (lattice_selector, gradient)
}

#[inline]
fn noise2_4ch(
    x: f64,
    y: f64,
    lattice_selector: &[usize; B_LEN],
    gradient: &GradientTable,
    stitch_info: Option<StitchInfo>,
) -> [f64; 4] {
    let t = x + PERLIN_N as f64;
    let mut bx0 = t as i32;
    let mut bx1 = bx0 + 1;
    let rx0 = t - t as i64 as f64;
    let rx1 = rx0 - 1.0;
    let t = y + PERLIN_N as f64;
    let mut by0 = t as i32;
    let mut by1 = by0 + 1;
    let ry0 = t - t as i64 as f64;
    let ry1 = ry0 - 1.0;

    if let Some(info) = stitch_info {
        if bx0 >= info.wrap_x {
            bx0 -= info.width;
        }
        if bx1 >= info.wrap_x {
            bx1 -= info.width;
        }
        if by0 >= info.wrap_y {
            by0 -= info.height;
        }
        if by1 >= info.wrap_y {
            by1 -= info.height;
        }
    }

    bx0 &= BM;
    bx1 &= BM;
    by0 &= BM;
    by1 &= BM;

    let i = lattice_selector[bx0 as usize];
    let j = lattice_selector[bx1 as usize];
    let b00 = lattice_selector[i + by0 as usize];
    let b10 = lattice_selector[j + by0 as usize];
    let b01 = lattice_selector[i + by1 as usize];
    let b11 = lattice_selector[j + by1 as usize];
    let sx = s_curve(rx0);
    let sy = s_curve(ry0);

    let mut result = [0.0f64; 4];
    for ch in 0..4 {
        let g = &gradient.data[ch];
        let q00 = &g[b00];
        let u = rx0 * q00[0] + ry0 * q00[1];
        let q10 = &g[b10];
        let v = rx1 * q10[0] + ry0 * q10[1];
        let a = lerp(sx, u, v);
        let q01 = &g[b01];
        let u = rx0 * q01[0] + ry1 * q01[1];
        let q11 = &g[b11];
        let v = rx1 * q11[0] + ry1 * q11[1];
        let b = lerp(sx, u, v);
        result[ch] = lerp(sy, a, b);
    }
    result
}

#[inline]
fn turbulence_4ch(
    mut x: f64,
    mut y: f64,
    base_freq_x: f64,
    base_freq_y: f64,
    num_octaves: u32,
    fractal_sum: bool,
    mut stitch: Option<StitchInfo>,
    lattice_selector: &[usize; B_LEN],
    gradient: &GradientTable,
) -> [f64; 4] {
    let mut sums = [0.0f64; 4];
    x *= base_freq_x;
    y *= base_freq_y;
    let mut inv_ratio = 1.0_f64;
    for _ in 0..num_octaves {
        let n = noise2_4ch(x, y, lattice_selector, gradient, stitch);
        if fractal_sum {
            for ch in 0..4 {
                sums[ch] += n[ch] * inv_ratio;
            }
        } else {
            for ch in 0..4 {
                sums[ch] += n[ch].abs() * inv_ratio;
            }
        }
        x *= 2.0;
        y *= 2.0;
        inv_ratio *= 0.5;

        if let Some(ref mut s) = stitch {
            s.width *= 2;
            s.wrap_x = 2 * s.wrap_x - PERLIN_N;
            s.height *= 2;
            s.wrap_y = 2 * s.wrap_y - PERLIN_N;
        }
    }
    sums
}

fn apply_optimized(
    offset_x: f64,
    offset_y: f64,
    sx: f64,
    sy: f64,
    base_frequency_x: f64,
    base_frequency_y: f64,
    num_octaves: u32,
    seed: i32,
    stitch_tiles: bool,
    fractal_noise: bool,
    width: u32,
    height: u32,
    data: &mut [Pixel],
) {
    let (lattice_selector, gradient) = init_optimized(seed);

    let mut base_freq_x = base_frequency_x;
    let mut base_freq_y = base_frequency_y;
    let tile_width = width as f64;
    let tile_height = height as f64;

    let initial_stitch = if stitch_tiles {
        if !approx_zero_ulps(base_freq_x, 4) {
            let lo_freq = (tile_width * base_freq_x).floor() / tile_width;
            let hi_freq = (tile_width * base_freq_x).ceil() / tile_width;
            if base_freq_x / lo_freq < hi_freq / base_freq_x {
                base_freq_x = lo_freq;
            } else {
                base_freq_x = hi_freq;
            }
        }
        if !approx_zero_ulps(base_freq_y, 4) {
            let lo_freq = (tile_height * base_freq_y).floor() / tile_height;
            let hi_freq = (tile_height * base_freq_y).ceil() / tile_height;
            if base_freq_y / lo_freq < hi_freq / base_freq_y {
                base_freq_y = lo_freq;
            } else {
                base_freq_y = hi_freq;
            }
        }
        let sw = (tile_width * base_freq_x + 0.5) as i32;
        let sh = (tile_height * base_freq_y + 0.5) as i32;
        Some((base_freq_x, base_freq_y, sw, sh))
    } else {
        None
    };

    let mut px = 0u32;
    let mut py = 0u32;
    for pixel in data.iter_mut() {
        let tx = (px as f64 + offset_x) / sx;
        let ty = (py as f64 + offset_y) / sy;

        let stitch = initial_stitch.map(|(bfx, bfy, sw, sh)| {
            let wx = (px as f64 * bfx + PERLIN_N as f64 + sw as f64) as i32;
            let wy = (py as f64 * bfy + PERLIN_N as f64 + sh as f64) as i32;
            StitchInfo {
                width: sw,
                height: sh,
                wrap_x: wx,
                wrap_y: wy,
            }
        });

        let n = turbulence_4ch(
            tx,
            ty,
            base_freq_x,
            base_freq_y,
            num_octaves,
            fractal_noise,
            stitch,
            &lattice_selector,
            &gradient,
        );

        fn to_u8(val: f64, fractal_noise: bool) -> u8 {
            let v = if fractal_noise {
                (val * 255.0 + 255.0) / 2.0
            } else {
                val * 255.0
            };
            (f32_bound(0.0, v as f32, 255.0) + 0.5) as u8
        }

        pixel.r = to_u8(n[0], fractal_noise);
        pixel.g = to_u8(n[1], fractal_noise);
        pixel.b = to_u8(n[2], fractal_noise);
        pixel.a = to_u8(n[3], fractal_noise);

        px += 1;
        if px == width {
            px = 0;
            py += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchConfig {
    width: u32,
    height: u32,
    octaves: u32,
    fractal_noise: bool,
    stitch_tiles: bool,
    base_freq_x: f64,
    base_freq_y: f64,
    seed: i32,
}

#[allow(dead_code)]
struct BenchResult {
    order: usize,
    config: String,
    size_label: String,
    octaves: u32,
    mode: &'static str,
    stitch: &'static str,
    freq_label: String,
    naive_us: f64,
    opt_us: f64,
    speedup: f64,
    regression: bool,
}

/// A group of BenchConfig entries (one per seed) that share the same
/// size/octaves/mode/stitch/freq. Results from each seed are averaged
/// into a single BenchResult.
struct AggregatedConfig {
    order: usize,
    width: u32,
    height: u32,
    octaves: u32,
    fractal_noise: bool,
    stitch_tiles: bool,
    base_freq_x: f64,
    base_freq_y: f64,
    seeds: Vec<i32>,
}

fn make_image(w: u32, h: u32) -> Vec<Pixel> {
    vec![
        Pixel {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        (w * h) as usize
    ]
}

/// Returns (naive_us, opt_us) timing for a single seed configuration.
fn bench_one_timing(cfg: &BenchConfig) -> (f64, f64) {
    let pixels = cfg.width as u64 * cfg.height as u64;

    // Choose iteration count to get meaningful timing
    let iters: u32 = if pixels >= 1_048_576 {
        2
    } else if pixels >= 65_536 {
        5
    } else if pixels >= 4_096 {
        20
    } else if pixels >= 256 {
        100
    } else {
        500
    };

    // -- Naive warmup --
    {
        let mut buf = make_image(cfg.width, cfg.height);
        apply_naive(
            0.0,
            0.0,
            1.0,
            1.0,
            cfg.base_freq_x,
            cfg.base_freq_y,
            cfg.octaves,
            cfg.seed,
            cfg.stitch_tiles,
            cfg.fractal_noise,
            cfg.width,
            cfg.height,
            &mut buf,
        );
        black_box(&buf);
    }

    // -- Naive timed --
    let start = Instant::now();
    for _ in 0..iters {
        let mut buf = make_image(cfg.width, cfg.height);
        apply_naive(
            0.0,
            0.0,
            1.0,
            1.0,
            cfg.base_freq_x,
            cfg.base_freq_y,
            cfg.octaves,
            cfg.seed,
            cfg.stitch_tiles,
            cfg.fractal_noise,
            cfg.width,
            cfg.height,
            &mut buf,
        );
        black_box(&buf);
    }
    let naive_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    // -- Optimized warmup --
    {
        let mut buf = make_image(cfg.width, cfg.height);
        apply_optimized(
            0.0,
            0.0,
            1.0,
            1.0,
            cfg.base_freq_x,
            cfg.base_freq_y,
            cfg.octaves,
            cfg.seed,
            cfg.stitch_tiles,
            cfg.fractal_noise,
            cfg.width,
            cfg.height,
            &mut buf,
        );
        black_box(&buf);
    }

    // -- Optimized timed --
    let start = Instant::now();
    for _ in 0..iters {
        let mut buf = make_image(cfg.width, cfg.height);
        apply_optimized(
            0.0,
            0.0,
            1.0,
            1.0,
            cfg.base_freq_x,
            cfg.base_freq_y,
            cfg.octaves,
            cfg.seed,
            cfg.stitch_tiles,
            cfg.fractal_noise,
            cfg.width,
            cfg.height,
            &mut buf,
        );
        black_box(&buf);
    }
    let opt_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

    (naive_us, opt_us)
}

/// Run all seeds for an AggregatedConfig, average them, and produce a BenchResult.
fn bench_aggregated(acfg: &AggregatedConfig) -> BenchResult {
    let mut naive_sum = 0.0;
    let mut opt_sum = 0.0;

    for &seed in &acfg.seeds {
        let cfg = BenchConfig {
            width: acfg.width,
            height: acfg.height,
            octaves: acfg.octaves,
            fractal_noise: acfg.fractal_noise,
            stitch_tiles: acfg.stitch_tiles,
            base_freq_x: acfg.base_freq_x,
            base_freq_y: acfg.base_freq_y,
            seed,
        };
        let (n, o) = bench_one_timing(&cfg);
        naive_sum += n;
        opt_sum += o;
    }

    let num_seeds = acfg.seeds.len() as f64;
    let naive_avg = naive_sum / num_seeds;
    let opt_avg = opt_sum / num_seeds;
    let speedup = naive_avg / opt_avg;
    let regression = speedup < 0.95;

    let size_label = format!("{}x{}", acfg.width, acfg.height);
    let mode = if acfg.fractal_noise {
        "fractalNoise"
    } else {
        "turbulence"
    };
    let stitch = if acfg.stitch_tiles { "on" } else { "off" };
    let freq_label = if (acfg.base_freq_x - acfg.base_freq_y).abs() < 1e-9 {
        format!("{:.2}", acfg.base_freq_x)
    } else {
        format!("({:.2},{:.2})", acfg.base_freq_x, acfg.base_freq_y)
    };
    let config = format!(
        "{}x{} oct={} {} stitch={} freq={}",
        acfg.width, acfg.height, acfg.octaves, mode, stitch, freq_label
    );

    BenchResult {
        order: acfg.order,
        config,
        size_label,
        octaves: acfg.octaves,
        mode,
        stitch,
        freq_label,
        naive_us: naive_avg,
        opt_us: opt_avg,
        speedup,
        regression,
    }
}

fn main() {
    let sizes: &[(u32, u32)] = &[
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];
    let octaves_list: &[u32] = &[1, 2, 4, 8];
    let modes: &[bool] = &[false, true]; // false=turbulence, true=fractalNoise
    let stitching: &[bool] = &[false, true];
    let freq_pairs: &[(f64, f64)] = &[
        (0.01, 0.01),
        (0.05, 0.05),
        (0.1, 0.1),
        (0.05, 0.01), // asymmetric
    ];
    let seeds: Vec<i32> = vec![0, 42, -7];

    // First, verify correctness for a representative subset
    println!("=== Correctness verification ===\n");
    let mut correct = true;
    for &(w, h) in &[(16u32, 16u32), (64, 64), (256, 256)] {
        for &octaves in &[1u32, 4, 8] {
            for &fractal in modes {
                for &stitch in stitching {
                    for &(bfx, bfy) in freq_pairs {
                        for &seed in &seeds {
                            let mut naive_buf = make_image(w, h);
                            let mut opt_buf = make_image(w, h);
                            apply_naive(
                                0.0, 0.0, 1.0, 1.0, bfx, bfy, octaves, seed, stitch, fractal,
                                w, h, &mut naive_buf,
                            );
                            apply_optimized(
                                0.0, 0.0, 1.0, 1.0, bfx, bfy, octaves, seed, stitch, fractal,
                                w, h, &mut opt_buf,
                            );
                            if naive_buf != opt_buf {
                                let mode_str = if fractal { "fractal" } else { "turb" };
                                let st = if stitch { "stitch" } else { "noStitch" };
                                eprintln!(
                                    "MISMATCH: {}x{} oct={} {} {} freq=({},{}) seed={}",
                                    w, h, octaves, mode_str, st, bfx, bfy, seed
                                );
                                correct = false;
                            }
                        }
                    }
                }
            }
        }
    }
    if correct {
        println!("All correctness checks PASSED.\n");
    } else {
        println!("WARNING: Some correctness checks FAILED!\n");
    }

    // Now run the comprehensive benchmark
    println!("=== Comprehensive feTurbulence Benchmark (naive vs optimized) ===\n");

    // Build all aggregated configurations upfront
    let mut configs: Vec<AggregatedConfig> = Vec::new();
    let mut order = 0usize;
    for &(w, h) in sizes {
        for &octaves in octaves_list {
            for &fractal_noise in modes {
                for &stitch_tiles in stitching {
                    for &(bfx, bfy) in freq_pairs {
                        configs.push(AggregatedConfig {
                            order,
                            width: w,
                            height: h,
                            octaves,
                            fractal_noise,
                            stitch_tiles,
                            base_freq_x: bfx,
                            base_freq_y: bfy,
                            seeds: seeds.clone(),
                        });
                        order += 1;
                    }
                }
            }
        }
    }

    // Count total individual seed tests
    let total = configs.len() * seeds.len();
    println!(
        "Total configurations: {} ({} sizes x {} octaves x {} modes x {} stitch x {} freqs x {} seeds)\n",
        total,
        sizes.len(),
        octaves_list.len(),
        modes.len(),
        stitching.len(),
        freq_pairs.len(),
        seeds.len()
    );

    // Determine thread count
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "Running with {} threads across {} aggregated configs\n",
        num_threads,
        configs.len()
    );

    // Progress counter shared across threads
    let progress = AtomicUsize::new(0);
    let total_configs = configs.len();

    // Split configs into chunks, one per thread
    let chunk_size = (configs.len() + num_threads - 1) / num_threads;
    let chunks: Vec<&[AggregatedConfig]> = configs.chunks(chunk_size).collect();

    // Execute in parallel using scoped threads
    let mut all_results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let progress_ref = &progress;
                s.spawn(move || {
                    let mut thread_results = Vec::with_capacity(chunk.len());
                    for acfg in chunk {
                        let result = bench_aggregated(acfg);
                        thread_results.push(result);
                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        eprint!(
                            "\r  Progress: {}/{} aggregated configs complete...",
                            done, total_configs
                        );
                    }
                    thread_results
                })
            })
            .collect();

        // Collect results from all threads
        let mut merged: Vec<BenchResult> = Vec::with_capacity(total_configs);
        for handle in handles {
            merged.extend(handle.join().expect("worker thread panicked"));
        }
        merged
    });

    eprintln!(); // newline after progress

    // Sort by original order to restore deterministic output
    all_results.sort_by_key(|r| r.order);

    // Print header
    println!(
        "{:<12} | {:>7} | {:<12} | {:>6} | {:>12} | {:>12} | {:>12} | {:>8}",
        "Image Size", "Octaves", "Mode", "Stitch", "Freq", "Naive (us)", "Opt (us)", "Speedup"
    );
    println!("{}", "-".repeat(103));

    let mut regressions: Vec<String> = Vec::new();

    // Print results with size-group separators
    let mut last_size_group: Option<usize> = None;
    for r in &all_results {
        // Determine size_group from the size_label
        let current_group = sizes
            .iter()
            .position(|&(w, h)| format!("{}x{}", w, h) == r.size_label)
            .unwrap_or(0);

        // Print separator when size group changes (but not before the first group)
        if let Some(prev) = last_size_group {
            if current_group != prev {
                println!("{}", "-".repeat(103));
            }
        }
        last_size_group = Some(current_group);

        let flag = if r.regression { " <<<REGRESSION" } else { "" };
        println!(
            "{:<12} | {:>7} | {:<12} | {:>6} | {:>12} | {:>12.1} | {:>12.1} | {:>7.2}x{}",
            r.size_label, r.octaves, r.mode, r.stitch, r.freq_label, r.naive_us, r.opt_us,
            r.speedup, flag
        );

        if r.regression {
            let msg = format!(
                "{} oct={} {} stitch={} freq={}: speedup={:.2}x (REGRESSION)",
                r.size_label, r.octaves, r.mode, r.stitch, r.freq_label, r.speedup
            );
            regressions.push(msg);
        }
    }
    println!("{}", "-".repeat(103));

    // Summary statistics
    println!("\n=== Summary ===\n");
    let result_count = all_results.len();
    let num_regressions = all_results.iter().filter(|r| r.regression).count();
    let min_speedup = all_results
        .iter()
        .map(|r| r.speedup)
        .fold(f64::INFINITY, f64::min);
    let max_speedup = all_results
        .iter()
        .map(|r| r.speedup)
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_speedup: f64 =
        all_results.iter().map(|r| r.speedup).sum::<f64>() / result_count as f64;

    println!("Total configurations tested: {}", result_count);
    println!("Min speedup: {:.2}x", min_speedup);
    println!("Max speedup: {:.2}x", max_speedup);
    println!("Avg speedup: {:.2}x", avg_speedup);
    println!(
        "Regressions (>5% slower): {}/{}",
        num_regressions, result_count
    );

    if !regressions.is_empty() {
        println!("\n=== REGRESSIONS DETECTED ===\n");
        for r in &regressions {
            println!("  {}", r);
        }
    } else {
        println!("\nNo regressions detected. All configurations show improvement or are within tolerance.");
    }

    // Per-size summary
    println!("\n=== Per-size average speedup ===\n");
    for &(w, h) in sizes {
        let label = format!("{}x{}", w, h);
        let size_results: Vec<_> = all_results.iter().filter(|r| r.size_label == label).collect();
        let avg: f64 = size_results.iter().map(|r| r.speedup).sum::<f64>()
            / size_results.len() as f64;
        let min = size_results
            .iter()
            .map(|r| r.speedup)
            .fold(f64::INFINITY, f64::min);
        let max = size_results
            .iter()
            .map(|r| r.speedup)
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:<12}: avg={:.2}x  min={:.2}x  max={:.2}x",
            label, avg, min, max
        );
    }

    // Per-octave summary
    println!("\n=== Per-octave average speedup ===\n");
    for &oct in octaves_list {
        let oct_results: Vec<_> = all_results.iter().filter(|r| r.octaves == oct).collect();
        let avg: f64 =
            oct_results.iter().map(|r| r.speedup).sum::<f64>() / oct_results.len() as f64;
        println!("  octaves={}: avg={:.2}x", oct, avg);
    }

    // Exit with error if regressions found
    if !regressions.is_empty() {
        std::process::exit(1);
    }
}
